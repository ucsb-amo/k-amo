"""Choosing the laser model and the basis for a light-shift calculation.

``kamo.hamiltonian`` has three laser models, and each fails in a different regime:

``"rwa"``
    Explicit dipole couplings in a single-frequency rotating frame.  It resolves
    the hyperfine and Zeeman substructure of every manifold and is
    non-perturbative, but (a) it only sees channels whose manifold is in the
    basis, and (b) it drops the counter-rotating terms, a relative error of
    about ``|Δ| / (f_c + f_L)`` per channel (``Δ = f_c - f_L``).  It is the
    right model near a line.

``"stark"``
    Diagonal fine-structure polarizabilities: every channel, both rotating
    terms.  But it lumps each coupled manifold at one energy, so it carries a
    relative error of about ``W_c / |Δ|`` per channel, where ``W_c`` is the
    manifold's hyperfine plus Zeeman spread (about 1.5 GHz for 4P3/2 at
    520 G), it is linear in intensity, and it is *identically zero* for the
    differential shift of two states with the same ``(n, l, j, m_j)``.

``"perturbative"``
    The second-order sum over the exact eigenstates of ``h0 + B * Zeeman``
    with both rotating terms (:mod:`kamo.hamiltonian.perturbative`): the
    substructure error of "stark" and the counter-rotating error of "rwa" are
    both absent.  What remains is the next order of perturbation theory,
    about ``eta^2`` with ``eta = Rabi / (2 detuning)``, so it is the model of
    choice whenever the light is perturbative.  It needs the channel
    manifolds in the basis, like the RWA.

``choose_laser_model`` therefore picks between "rwa" and "perturbative":
"rwa" when ``eta`` exceeds ``eta_max`` or when ``eta^2`` exceeds the RWA's
counter-rotating error estimate, "perturbative" otherwise.  "stark" is kept
as the fast explicit option.  Its error estimate is still reported.

For K39 at 520 G the RWA and Stark error estimates cross about 1 THz from the
D lines; between 300 GHz and 3 THz those two models agree to better than
1e-3 (checked 2026-09-13 with both sweeps on the imaging line).  At 1064 nm
the RWA with a basis lacking 3D and 5S was 5.5x low on the imaging-line
shift; the contribution-based basis below removes that failure mode and
leaves the counter-rotating error (1.3 % on the imaging line).

Public helpers
--------------
``channel_weights(state, frequency_Hz, ...)``
    The scalar polarizability of a state, channel by channel, with each
    channel's share of the total, its detuning, substructure spread, and the
    per-channel RWA and Stark error estimates.
``light_shift_basis(states, frequency_Hz, ...)``
    Manifolds to put in an RWA basis so that every channel carrying more than
    ``tol`` of either state's polarizability is present, with a consistency
    check on the rotating-frame photon indices.
``choose_laser_model(states, frequency_Hz, ...)``
    ``"rwa"`` or ``"perturbative"`` for a given transition, laser and
    intensity, with the numbers behind the decision.

The polarizability breakdown comes from :func:`kamo.trap.polarizability.scalar_breakdown`
(UDel portal matrix elements, ARC fill-in), imported lazily so that this module
never touches the portal snapshot or ARC's database at import time.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import kamo.constants as kc

NLJ = Tuple[int, int, float]

_LABEL_RE = re.compile(r"^(\d+)([spdfghik])(\d+)/2$")
_L_LETTERS = "spdfghik"
_GAUSS_TO_TESLA = 1e-4

__all__ = [
    "Channel",
    "StateChannels",
    "BasisSelection",
    "LaserModelChoice",
    "channel_weights",
    "state_channels",
    "light_shift_basis",
    "choose_laser_model",
    "photon_indices",
    "substructure_spread_Hz",
]


# ------------------------------------------------------------------ helpers

def _nlj(state) -> NLJ:
    return int(state[0]), int(state[1]), float(state[2])


def _fmt(nlj: NLJ) -> str:
    n, l, j = nlj
    return f"{n}{_L_LETTERS[l]}{j:g}"


def _label_to_nlj(label: str) -> NLJ:
    m = _LABEL_RE.match(label)
    if m is None:
        raise ValueError(f"Cannot parse state label {label!r}.")
    return int(m.group(1)), _L_LETTERS.index(m.group(2)), int(m.group(3)) / 2.0


def _default_atom():
    from kamo.atom_properties.alkali import default_atom
    return default_atom()


def _source_of(atom) -> str:
    return "arc" if getattr(atom, "use_portal", True) is False else "portal"


def substructure_spread_Hz(nlj: NLJ, B_gauss: float, I: float = None, atom=None) -> float:
    """Hyperfine plus Zeeman spread (Hz) of manifold ``nlj`` at ``B_gauss``.

    Zeeman part: ``|g_J| J mu_B B / h`` (the half-spread, i.e. how far the
    outermost m_J sits from the manifold centroid).  Hyperfine part: the span
    from ``F = |I - J|`` to ``F = I + J`` of ``E_F = A K / 2`` using the
    atom's hyperfine constants (the electric-quadrupole ``B`` term is
    neglected).  This is the scale of the detuning variation that the Stark
    model lumps into one pole.  ``atom`` defaults to kamo's default atom
    (39K); ``I`` to the atom's nuclear spin.
    """
    from kamo.atom_properties.alkali import electronic_g, hyperfine, nuclear_spin
    if atom is None:
        atom = _default_atom()
    if I is None:
        I = nuclear_spin(atom)
    n, l, j = nlj
    g_j = electronic_g(atom, l, j, n=n)
    zeeman = abs(g_j) * j * kc.mu_b * abs(B_gauss) * _GAUSS_TO_TESLA / kc.h
    A = hyperfine(atom, n, l, j).A_Hz
    F_hi, F_lo = I + j, abs(I - j)
    hfs = abs(A) / 2.0 * (F_hi * (F_hi + 1) - F_lo * (F_lo + 1))
    return float(zeeman + hfs)


def photon_indices(nljs: Sequence[NLJ], energies: Dict[NLJ, float]
                   ) -> Tuple[Dict[NLJ, int], List[Tuple[NLJ, NLJ]]]:
    """Rotating-frame photon index per manifold, by breadth-first search over
    dipole-allowed (``|Δl| = 1``) connections from the lowest manifold.

    Same algorithm as ``HamiltonianBuilder._photon_index``.  Returns the index
    map and the list of connections whose implied index disagrees with the one
    already assigned, i.e. the loops a single-frequency RWA cannot represent.
    """
    order = sorted(nljs, key=lambda m: energies[m])
    idx: Dict[NLJ, int] = {order[0]: 0}
    bad: List[Tuple[NLJ, NLJ]] = []
    changed = True
    while changed:
        changed = False
        for ma in nljs:
            if ma not in idx:
                continue
            for mb in nljs:
                if abs(ma[1] - mb[1]) != 1:
                    continue
                step = 1 if energies[mb] > energies[ma] else -1
                want = idx[ma] + step
                if mb not in idx:
                    idx[mb] = want
                    changed = True
                elif idx[mb] != want and (mb, ma) not in bad and (ma, mb) not in bad:
                    bad.append((ma, mb))
    for m in nljs:
        idx.setdefault(m, 0)
    return idx, bad


# ------------------------------------------------------------------ channels

@dataclass(frozen=True)
class Channel:
    """One dipole channel of a state's scalar polarizability at the laser frequency."""

    state: NLJ                  #: the state whose polarizability this is a term of
    manifold: NLJ               #: the coupled manifold (n, l, j)
    final: str                  #: its label, e.g. "3d5/2"
    origin: str                 #: "portal" or "arc" matrix element
    d_au: float                 #: |reduced matrix element| (e a0)
    alpha_au: float             #: signed contribution to alpha_s (a.u.)
    share: float                #: |alpha_au| / |alpha_total| for this state
    detuning_Hz: float          #: f_c - f_L (positive: laser red of the line)
    substructure_Hz: float      #: hyperfine + Zeeman spread of the manifold at B
    energy_Hz: float = 0.0      #: signed (E_final - E_initial)/h
    f_laser_Hz: float = field(default=0.0, repr=False, compare=False)

    @property
    def eps_rwa(self) -> float:
        """Relative size of the dropped counter-rotating term for this channel."""
        f_c = abs(self.detuning_Hz + self.f_laser_Hz)
        return abs(self.detuning_Hz) / (f_c + self.f_laser_Hz)

    @property
    def eps_stark(self) -> float:
        """Relative error from lumping the manifold at one energy."""
        return self.substructure_Hz / max(abs(self.detuning_Hz), 1e-300)

    def rabi_Hz(self, intensity_W_m2: float) -> float:
        """``d E0 / h`` for the reduced matrix element (an upper bound on any
        sublevel coupling) at the given intensity."""
        E0 = math.sqrt(2.0 * float(intensity_W_m2) / (kc.c * kc.epsilon0))
        return self.d_au * kc.a0 * kc.e * E0 / kc.h


@dataclass(frozen=True)
class StateChannels:
    """A state's scalar polarizability at one laser frequency, split into channels."""

    state: NLJ
    f_laser_Hz: float
    alpha_total_au: float       #: signed scalar polarizability including the core
    core_au: float              #: ionic-core part (no dipole channel; invisible to the RWA)
    channels: Tuple[Channel, ...]   #: sorted by decreasing share

    @property
    def core_share(self) -> float:
        return abs(self.core_au) / abs(self.alpha_total_au)

    def eps_rwa(self) -> float:
        """Share-weighted counter-rotating error estimate for this state."""
        return sum(c.share * c.eps_rwa for c in self.channels)

    def eps_stark(self) -> float:
        """Share-weighted substructure error estimate for this state."""
        return sum(c.share * c.eps_stark for c in self.channels)


def state_channels(state, frequency_Hz: float, B_gauss: float = 0.0,
                   source: str = None, spread_floor: float = 1e-4,
                   atom=None) -> StateChannels:
    """Scalar polarizability of ``state`` at the laser frequency, channel by channel.

    ``state`` is anything whose first three entries are ``(n, l, j)``.  The
    substructure spread is evaluated only for channels with
    ``share >= spread_floor`` (it needs a hyperfine-constant lookup); smaller
    channels get zero, which only affects the Stark error estimate at the
    1e-4 level.  ``atom`` selects the species (default kamo's default atom,
    39K); ``source`` defaults to ``"arc"`` for an atom built with
    ``use_portal=False`` and ``"portal"`` otherwise.
    """
    from kamo.atom_properties.alkali import species_of
    from kamo.trap.polarizability import scalar_breakdown

    if atom is None:
        atom = _default_atom()
    if source is None:
        source = _source_of(atom)
    n, l, j = _nlj(state)
    f_L = float(frequency_Hz)
    bd = scalar_breakdown(n, l, j, kc.c / f_L, source, species=species_of(atom))
    total = abs(bd.total_au)
    if total == 0.0:
        raise ValueError(f"Zero polarizability for {(n, l, j)} at {f_L:.4e} Hz.")
    out = []
    for t in sorted(bd.transitions, key=lambda t: -abs(t.scalar_au)):
        share = abs(t.scalar_au) / total
        man = _label_to_nlj(t.final)
        f_c = abs(t.energy_J) / kc.h          # |transition| frequency of the channel
        W = (substructure_spread_Hz(man, B_gauss, atom=atom)
             if share >= spread_floor else 0.0)
        out.append(Channel(
            state=(n, l, j), manifold=man, final=t.final, origin=t.origin,
            d_au=float(t.d_au), alpha_au=float(t.scalar_au), share=float(share),
            detuning_Hz=float(f_c - f_L), substructure_Hz=float(W),
            energy_Hz=float(t.energy_J / kc.h), f_laser_Hz=f_L))
    return StateChannels((n, l, j), f_L, float(bd.total_au), float(bd.core_au), tuple(out))


def channel_weights(state, frequency_Hz: float, B_gauss: float = 0.0,
                    source: str = None, spread_floor: float = 1e-4,
                    atom=None) -> List[Channel]:
    """The channel list of :func:`state_channels`, sorted by decreasing share."""
    return list(state_channels(state, frequency_Hz, B_gauss, source, spread_floor,
                               atom=atom).channels)


def _unique_states(states: Iterable) -> List[NLJ]:
    out: List[NLJ] = []
    for s in states:
        st = _nlj(s)
        if st not in out:
            out.append(st)
    return out


# ------------------------------------------------------------------ basis

@dataclass(frozen=True)
class BasisSelection:
    """Manifolds chosen for an RWA light-shift basis and what they cover."""

    manifolds: Tuple[NLJ, ...]
    coverage: Dict[NLJ, float]          #: per state: 1 - (excluded channels + core)/|alpha|
    core_share: Dict[NLJ, float]        #: per state: the core part, never in an RWA basis
    dropped: Tuple[Tuple[NLJ, float, str], ...]   #: (manifold, share, reason)
    per_state: Tuple[StateChannels, ...]
    tol: float

    @property
    def min_coverage(self) -> float:
        return min(self.coverage.values())

    def describe(self) -> str:
        mans = ", ".join(_fmt(m) for m in self.manifolds)
        cov = ", ".join(f"{_fmt(st)}: {c:.4f} (core {self.core_share[st]:.4f})"
                        for st, c in self.coverage.items())
        s = f"basis [{mans}]; polarizability coverage {cov}"
        if self.dropped:
            s += "; dropped " + ", ".join(
                f"{_fmt(m)} ({sh:.1e}, {why})" for m, sh, why in self.dropped)
        return s


def _fs_partners(nlj: NLJ) -> List[NLJ]:
    n, l, j = nlj
    if l == 0:
        return [nlj]
    return [(n, l, l - 0.5), (n, l, l + 0.5)]


def light_shift_basis(states: Iterable, frequency_Hz: float, tol: float = 1e-3,
                      atom=None, B_gauss: float = 0.0,
                      per_state: Optional[Sequence[StateChannels]] = None,
                      coverage_warn: float = 0.97,
                      check_loops: bool = True) -> BasisSelection:
    """Manifolds for a basis that captures the light shift of ``states``.

    Always included: each state's own manifold and its fine-structure partner
    (Zeeman mixing between them matters for the bare transition frequency).
    Added in decreasing order of share: every channel carrying at least ``tol``
    of a state's scalar polarizability at the laser frequency.  With
    ``check_loops`` (needed for the RWA model, not for the perturbative one) a
    channel whose manifold would create a rotating-frame photon-index loop (a
    manifold graph a single-frequency RWA cannot represent consistently) is
    left out, reported in ``dropped`` and warned about.  A second warning
    fires if a state's coverage (everything the basis cannot represent:
    excluded channels and the ionic core) falls below ``coverage_warn``.

    The manifolds come back sorted by energy.  ``laser_rwa_operator`` assumes
    that the earlier-listed manifold of a coupled pair is the lower one when it
    assigns which sublevels a sigma photon connects, so an energy-ordered
    basis keeps circular polarizations right.

    The selection uses the perturbative breakdown at the *laser* frequency, so
    a channel the laser is nearly resonant with dominates the shares and is
    always kept.
    """
    uniq = _unique_states(states)
    if atom is None:
        atom = _default_atom()
    if per_state is None:
        per_state = [state_channels(st, frequency_Hz, B_gauss=B_gauss, atom=atom)
                     for st in uniq]
    own: List[NLJ] = []
    for st in uniq:
        for m in _fs_partners(st):
            if m not in own:
                own.append(m)

    # candidate channel manifolds, best first, with the max share over states
    cand: Dict[NLJ, float] = {}
    for sc in per_state:
        for ch in sc.channels:
            if ch.share >= tol and ch.manifold not in own:
                cand[ch.manifold] = max(cand.get(ch.manifold, 0.0), ch.share)
    ordered = sorted(cand, key=lambda m: -cand[m])

    energies: Dict[NLJ, float] = {m: atom.getEnergy(*m) for m in own}
    chosen = list(own)
    dropped = []
    for m in ordered:
        energies[m] = atom.getEnergy(*m)
        trial = chosen + [m]
        if check_loops:
            _, bad = photon_indices(trial, energies)
            if bad:
                other = sorted({b for pair in bad for b in pair if b != m})
                dropped.append((m, cand[m], "photon-index loop with "
                                + ", ".join(_fmt(b) for b in other)))
                continue
        chosen = trial
    chosen.sort(key=lambda m: energies[m])

    coverage: Dict[NLJ, float] = {}
    core_share: Dict[NLJ, float] = {}
    for sc in per_state:
        missing = sum(ch.share for ch in sc.channels if ch.manifold not in chosen)
        coverage[sc.state] = float(1.0 - missing - sc.core_share)
        core_share[sc.state] = float(sc.core_share)
    sel = BasisSelection(tuple(chosen), coverage, core_share, tuple(dropped),
                         tuple(per_state), float(tol))
    if dropped:
        warnings.warn("RWA light-shift basis had to leave out channel manifolds: "
                      + sel.describe(), RuntimeWarning, stacklevel=2)
    if sel.min_coverage < coverage_warn:
        warnings.warn(
            f"RWA light-shift basis represents only {sel.min_coverage:.3f} of a state's "
            "polarizability at this wavelength: " + sel.describe(),
            RuntimeWarning, stacklevel=2)
    return sel


# ------------------------------------------------------------------ choice

@dataclass(frozen=True)
class LaserModelChoice:
    """Outcome of :func:`choose_laser_model`."""

    model: str                          #: "rwa" or "perturbative"
    reason: str
    eps_rwa: float                      #: estimated relative error of the RWA result
    eps_stark: float                    #: estimated relative error of the explicit Stark model
    eta: float                          #: max Rabi / (2 detuning) at the intensity given
    eps_perturbative: float             #: eta^2, the next order of the perturbative sum
    err_rwa_au: float                   #: absolute error estimate on the differential (a.u.)
    err_stark_au: float
    differential_au: float              #: alpha(state2) - alpha(state1), scalar parts
    basis: BasisSelection               #: the RWA basis (used when model == "rwa")

    def describe(self) -> str:
        return (f"laser_model={self.model!r}: {self.reason} "
                f"(eps_rwa~{self.eps_rwa:.1e}, eps_perturbative~{self.eps_perturbative:.1e}, "
                f"eps_stark~{self.eps_stark:.1e}, eta={self.eta:.1e}); {self.basis.describe()}")


def choose_laser_model(states: Iterable, frequency_Hz: float,
                       intensity_W_m2: Optional[float] = None,
                       B_gauss: float = 0.0, atom=None, tol: float = 1e-3,
                       eta_max: float = 0.1, poor_warn: float = 0.05) -> LaserModelChoice:
    """Pick ``"rwa"`` or ``"perturbative"`` for the light shift of a transition.

    Decision, in order:

    1. ``eta = max_c Rabi_c / (2 |Δ_c|) > eta_max`` at the given intensity ->
       ``"rwa"``: second-order perturbation theory is not valid.
    2. ``eta^2 > eps_rwa`` -> ``"rwa"``: the next order of the perturbative
       sum would be larger than the RWA's counter-rotating error.
    3. Otherwise ``"perturbative"``.

    ``eps_rwa`` and ``eps_stark`` are the estimated absolute errors of those
    two models on the differential shift, ``Σ_states |α_s| Σ_channels share_c
    eps_c`` with ``eps_c = |Δ| / (f_c + f_L)`` for the RWA and ``W_c / |Δ|``
    for the Stark model, relative to the differential scalar shift (relative
    to the larger state shift when the two states share a manifold, where
    the Stark model gives zero).  Weighting by ``|α_s|`` matters: 40 GHz from
    D1 the ground state's shift is 40x the excited state's.  Both are upper
    bounds: against the two sweeps at 520 G the RWA estimate was 10-25x
    conservative, because the counter-rotating terms of the two states partly
    cancel in a differential, and the Stark estimate about 6x.  The Stark
    estimate is reported for the explicit ``laser_model="stark"`` user; it
    never drives the choice.  A warning is raised when the RWA is forced by
    ``eta`` and its counter-rotating error estimate exceeds ``poor_warn``.

    ``intensity_W_m2=None`` skips the perturbativity check (eta = 0), which
    selects the perturbative model.
    """
    states = [tuple(s) for s in states]
    if len(states) != 2:
        raise ValueError("choose_laser_model expects exactly two states.")
    f_L = float(frequency_Hz)
    uniq = _unique_states(states)
    if atom is None:
        atom = _default_atom()
    per_state = [state_channels(st, f_L, B_gauss=B_gauss, atom=atom) for st in uniq]
    by_state = {sc.state: sc for sc in per_state}
    basis = light_shift_basis(states, f_L, tol=tol, atom=atom, B_gauss=B_gauss,
                              per_state=per_state)

    same_manifold = len(uniq) == 1
    sc1, sc2 = by_state[_nlj(states[0])], by_state[_nlj(states[1])]
    err_rwa = err_stark = 0.0
    for sc in per_state:
        err_rwa += abs(sc.alpha_total_au) * sc.eps_rwa()
        err_stark += abs(sc.alpha_total_au) * sc.eps_stark()
    differential = sc2.alpha_total_au - sc1.alpha_total_au
    scale = (max(abs(sc1.alpha_total_au), abs(sc2.alpha_total_au)) if same_manifold
             else abs(differential))
    eps_rwa, eps_stark = err_rwa / scale, err_stark / scale

    eta = 0.0
    if intensity_W_m2 is not None and intensity_W_m2 > 0:
        eta = max(c.rabi_Hz(intensity_W_m2) / (2.0 * abs(c.detuning_Hz))
                  for sc in per_state for c in sc.channels if c.share >= tol)

    eps_pert = eta ** 2
    if eta > eta_max:
        model, reason = "rwa", f"non-perturbative: Rabi/(2 detuning) = {eta:.2f} > {eta_max}"
    elif eps_pert > eps_rwa:
        model, reason = "rwa", ("counter-rotating error below the next order of the "
                                "perturbative sum")
    else:
        model, reason = "perturbative", ("perturbative light; exact eigenstates with both "
                                         "rotating terms")

    choice = LaserModelChoice(model, reason, float(eps_rwa), float(eps_stark), float(eta),
                              float(eps_pert), float(err_rwa), float(err_stark),
                              float(differential), basis)
    if model == "rwa" and eta > eta_max and eps_rwa > poor_warn:
        warnings.warn("The light is non-perturbative so the RWA is required, but its "
                      f"counter-rotating error estimate is {eps_rwa:.1e}: " + choice.describe(),
                      RuntimeWarning, stacklevel=2)
    return choice
