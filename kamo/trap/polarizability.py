"""Ground-state polarizabilities for :mod:`kamo.trap`.

What is reused, unchanged
-------------------------
:class:`kamo.light_shift.ComputePolarizabilities` does the physics.  Its
``compute_polarizability(n, l, j, F, lambda)`` returns the frame-free hyperfine
scalar, vector and tensor components: a sum over every dipole-allowed state
(``n' in [3, 16]``, 6j algebra) using UDel-portal reduced matrix elements and
wavelengths where the portal lists the transition, ARC where it does not, plus
the 5.457 a.u. K+ ionic core in the scalar part.  Its
``compute_complete_polarizability`` is *not* called here: it takes a
polarization vector whose index 0 is the quantization axis (see
:mod:`kamo.trap.frames`), and it is used only as a test oracle.

What this module adds
---------------------
* **Lab-frame geometry.**  ``beta`` and ``gamma`` come from
  :func:`kamo.trap.frames.polarization_geometry`, then
  ``alpha_F = alpha_s - beta mF/(2F) alpha_v
  + gamma (3 mF^2 - F(F+1))/(F(2F-1)) alpha_t``, with the F = 0 and F = 1/2
  divisions guarded.
* **One source switch.**  ``ComputePolarizabilities(atom, force_arc)`` has two
  switches -- the atom's ``use_portal`` and ``force_arc`` -- that can disagree
  and silently mix portal and ARC data.  Here ``source="portal"`` or ``"arc"``
  sets both, as :mod:`kamo.hamiltonian.builder` already does.
* **A module-wide cache.**  The components depend only on
  ``(n, l, j, F, lambda, source)``; changing mF, the field direction or the
  polarization costs nothing after the first call.
* **Lazy construction.**  Building a ``Potassium39`` opens ARC's database and,
  on the portal path, fetches UDel-portal data over the network.  Nothing here
  does so until a polarizability is actually requested.
* **Provenance** (:func:`provenance`, :class:`ScalarBreakdown`): which copy of
  the portal data was read (local cache or the snapshot bundled with kamo),
  when it was fetched, and how much of ``alpha_s`` came from ARC fill-in.
  Required because the same code reads different data on different machines.
* **Uncertainty** from the portal's per-element uncertainties (dropped by
  ``to_legacy_table``), propagated to ``alpha_s``.
* **A near-resonance guard.**  The sum is the undamped far-detuned expression;
  within a relative 1e-3 of a summed transition it is wrong, and it warns.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

import kamo.constants as kc

from . import frames as fr

SOURCES = ("portal", "arc")
NEAR_RESONANCE = 1e-3          #: relative detuning below which the sum is not trusted
_CP: dict = {}                 #: {source: ComputePolarizabilities}, built on first use


def _check_source(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}; got {source!r}")
    return source


def compute_polarizabilities(source: str = "portal"):
    """The shared :class:`ComputePolarizabilities` for ``source``, built on first use."""
    _check_source(source)
    if source not in _CP:
        from kamo.atom_properties.k39 import Potassium39
        from kamo.light_shift.compute_polarizabilities import ComputePolarizabilities
        atom = Potassium39(use_portal=(source == "portal"))
        _CP[source] = ComputePolarizabilities(atom=atom, force_arc=(source == "arc"))
    return _CP[source]


def clear_caches():
    """Forget the built calculators and every cached polarizability."""
    _CP.clear()
    hyperfine_components_au.cache_clear()
    scalar_breakdown.cache_clear()


# --------------------------------------------------------------- components

@lru_cache(maxsize=None)
def hyperfine_components_au(n: int, l: int, j: float, F: float, wavelength_m: float,
                            source: str = "portal", I: float = 1.5):
    """``(alpha_s, alpha_v, alpha_t)`` in atomic units for ``|n l j F>``.

    Independent of mF, the field direction and the polarization -- which is why
    it is cached here and the geometry is applied afterwards.
    """
    cp = compute_polarizabilities(source)
    a_s, a_v, a_t = cp.compute_polarizability(n, l, j, F, float(wavelength_m), I)
    return tuple(float(np.atleast_1d(x)[0]) for x in (a_s, a_v, a_t))


def combine_au(components, F: float, mF: float, beta: float, gamma: float) -> float:
    """``alpha_F`` from the components and the geometry factors (a.u.)."""
    a_s, a_v, a_t = components
    alpha = a_s
    if F > 0:
        alpha -= beta * mF / (2.0 * F) * a_v
    if F > 0.5:
        alpha += gamma * (3.0 * mF ** 2 - F * (F + 1.0)) / (F * (2.0 * F - 1.0)) * a_t
    return float(alpha)


# ---------------------------------------------------------------- breakdown

@dataclass(frozen=True)
class Transition:
    """One term of the scalar sum."""

    final: str                  #: final-state label, e.g. "4p3/2"
    origin: str                 #: "portal" or "arc"
    d_au: float                 #: |reduced matrix element| (e a0)
    d_unc_au: float             #: its portal uncertainty (NaN if none)
    energy_J: float             #: signed E_final - E_initial (J)
    scalar_au: float            #: contribution to alpha_s (a.u.)


@dataclass(frozen=True)
class ScalarBreakdown:
    """The scalar polarizability term by term, with where each term came from."""

    wavelength_m: float
    transitions: tuple
    core_au: float

    @property
    def total_au(self) -> float:
        return self.core_au + math.fsum(t.scalar_au for t in self.transitions)

    @property
    def n_portal(self) -> int:
        return sum(t.origin == "portal" for t in self.transitions)

    @property
    def n_arc(self) -> int:
        return sum(t.origin == "arc" for t in self.transitions)

    @property
    def arc_share(self) -> float:
        """Fraction of ``alpha_s`` contributed by ARC-filled transitions."""
        return math.fsum(t.scalar_au for t in self.transitions if t.origin == "arc") / self.total_au

    @property
    def uncertainty_au(self) -> float:
        """``sqrt(sum (2 alpha_i dd_i / d_i)^2)`` over terms with a portal uncertainty."""
        terms = [2.0 * t.scalar_au * t.d_unc_au / t.d_au for t in self.transitions
                 if t.origin == "portal" and np.isfinite(t.d_unc_au) and t.d_au > 0]
        return float(np.sqrt(math.fsum(x * x for x in terms)))

    @property
    def nearest_relative_detuning(self) -> float:
        E_L = kc.h * kc.c / self.wavelength_m
        return min(abs(E_L - abs(t.energy_J)) / abs(t.energy_J) for t in self.transitions)


def _portal_uncertainties(species: str, initial: str) -> dict:
    from kamo.light_shift import udel_portal
    me = udel_portal.matrix_elements(species)
    out = {}
    for r in me[me.state1 == initial].itertuples():
        try:
            out[r.state2] = float(r.d_unc_au)
        except (TypeError, ValueError):
            out[r.state2] = float("nan")
    return out


@lru_cache(maxsize=None)
def scalar_breakdown(n: int, l: int, j: float, wavelength_m: float,
                     source: str = "portal") -> ScalarBreakdown:
    """The scalar sum of ``compute_fine_structure_polarizability``, term by term.

    Uses the same parser calls in the same order, so :attr:`ScalarBreakdown.total_au`
    reproduces the calculator's ``alpha_s`` (a test pins it to 1e-12).
    """
    cp = compute_polarizabilities(source)
    pdp = cp.pdp
    E_L = kc.h * kc.c / float(wavelength_m)
    initial = pdp.quantum_numbers_to_state_label(n, l, j)
    table = None if pdp.arc else pdp.reduced_dipole_matrix_element_table(n, l, j)
    unc = {} if pdp.arc else _portal_uncertainties(getattr(cp.atom, "portal_species", "K1"),
                                                   initial)
    pref = (2.0 / 3.0) / (2.0 * j + 1.0) / kc.convert_polarizability_au_to_SI
    terms = []
    for label in pdp.determine_allowed_final_states(l, j):
        nf, lf, jf = pdp.state_label_to_quantum_numbers(label)
        if pdp.arc:
            d, E = pdp.matrix_element_arc(n, l, j, nf, lf, jf)
            origin = "arc"
        else:
            d, E = pdp.matrix_element_from_transition_table(nf, lf, jf, table)
            origin = "portal" if bool((table["Final"] == label).any()) else "arc"
        d, E = float(d), float(E)
        d_SI = d * kc.a0 * kc.e
        terms.append(Transition(
            final=label, origin=origin, d_au=abs(d),
            d_unc_au=unc.get(label, float("nan")) if origin == "portal" else float("nan"),
            energy_J=E, scalar_au=pref * d_SI ** 2 * E / (E ** 2 - E_L ** 2)))
    core = cp.return_ionic_core_contribution() if getattr(cp, "include_core", False) else 0.0
    return ScalarBreakdown(float(wavelength_m), tuple(terms), float(core))


# --------------------------------------------------------------- provenance

def provenance(source: str = "portal", species: str = "K1") -> dict:
    """Where the matrix elements come from.

    ``origin`` is ``"cache"`` (the user cache, written by a live fetch now or
    earlier), ``"snapshot"`` (the copy bundled with kamo, used offline), or
    ``"arc"``.  ``fetched_at`` is the portal fetch time recorded in that file.
    Inferred from the files the fetcher reads, so call it after the data were
    first used.
    """
    _check_source(source)
    if source == "arc":
        return dict(source="arc", origin="arc", path=None, fetched_at=None)
    from kamo.light_shift import udel_portal
    name = f"{species}_matrix_elements.json"
    for origin, path in (("cache", udel_portal.cache_dir() / name),
                         ("snapshot", udel_portal._SNAPSHOT_DIR / name)):
        if path.exists():
            try:
                fetched = json.loads(path.read_text()).get("fetched_at")
            except (OSError, ValueError):
                fetched = None
            return dict(source="portal", origin=origin, path=str(path), fetched_at=fetched)
    return dict(source="portal", origin="unknown", path=None, fetched_at=None)


# ------------------------------------------------------------------- state

class StatePolarizability:
    """``alpha(state, lambda, polarization, B_hat)`` for one hyperfine state.

    Parameters
    ----------
    state : (n, l, j, F, mF)
        Low-field hyperfine labels, e.g. ``(4, 0, 0.5, 1, -1)``.
    source : {"portal", "arc"}
        UDel-portal matrix elements with ARC fill-in (default), or ARC only.
    nuclear_spin : float
        I = 3/2 for K-39.
    """

    def __init__(self, state, source: str = "portal", nuclear_spin: float = 1.5):
        try:
            n, l, j, F, mF = state
        except (TypeError, ValueError):
            raise ValueError(f"state must be a 5-tuple (n, l, j, F, mF); got {state!r}") from None
        n, l, j, F, mF = int(n), int(l), float(j), float(F), float(mF)
        if n < 1 or not 0 <= l < n:
            raise ValueError(f"invalid (n, l) = ({n}, {l})")
        if abs(j - (l + 0.5)) > 1e-9 and abs(j - (l - 0.5)) > 1e-9:
            raise ValueError(f"j = {j} is not l +- 1/2 for l = {l}")
        if F < 0 or abs(mF) > F + 1e-9 or abs((F - mF) - round(F - mF)) > 1e-9:
            raise ValueError(f"invalid (F, mF) = ({F}, {mF})")
        self.state = (n, l, j, F, mF)
        self.source = _check_source(source)
        self.nuclear_spin = float(nuclear_spin)
        self._warned: set = set()

    def components_au(self, wavelength_m: float):
        n, l, j, F, _ = self.state
        return hyperfine_components_au(n, l, j, F, float(wavelength_m), self.source,
                                       self.nuclear_spin)

    def breakdown(self, wavelength_m: float) -> ScalarBreakdown:
        n, l, j, _, _ = self.state
        return scalar_breakdown(n, l, j, float(wavelength_m), self.source)

    def alpha_au(self, wavelength_m: float, polarization, quantization_axis) -> float:
        """Total polarizability (a.u.) of this state in a beam of this
        polarization, quantized along ``quantization_axis`` (lab frame)."""
        self._resonance_guard(float(wavelength_m))
        beta, gamma = fr.polarization_geometry(polarization, quantization_axis)
        _, _, _, F, mF = self.state
        return combine_au(self.components_au(wavelength_m), F, mF, beta, gamma)

    def alpha_SI(self, wavelength_m: float, polarization, quantization_axis) -> float:
        """As :meth:`alpha_au`, in C m^2 / V."""
        return self.alpha_au(wavelength_m, polarization, quantization_axis) \
            * kc.convert_polarizability_au_to_SI

    def uncertainty_au(self, wavelength_m: float) -> float:
        """Portal uncertainty of the scalar part (a.u.); NaN on the ARC path."""
        if self.source != "portal":
            return float("nan")
        return self.breakdown(wavelength_m).uncertainty_au

    def provenance(self) -> dict:
        species = "K1"
        if self.source in _CP:
            species = getattr(_CP[self.source].atom, "portal_species", "K1")
        return provenance(self.source, species)

    def describe(self, wavelength_m: float) -> str:
        """One line for a summary: the data source and its share of alpha_s."""
        b, p = self.breakdown(wavelength_m), self.provenance()
        if p["source"] == "arc":
            return f"polarizability: ARC only, {len(b.transitions)} transitions"
        when = (p["fetched_at"] or "unknown date")[:10]
        return (f"polarizability: UDel portal ({p['origin']}, fetched {when}) "
                f"{b.n_portal} transitions / {100 * (1 - b.arc_share):.2f}% of alpha_s; "
                f"ARC {b.n_arc} / {100 * b.arc_share:.2f}%")

    def for_state(self, state) -> "StatePolarizability":
        """Another state, sharing every cache."""
        return StatePolarizability(state, self.source, self.nuclear_spin)

    def _resonance_guard(self, wavelength_m: float):
        if wavelength_m in self._warned:
            return
        detuning = self.breakdown(wavelength_m).nearest_relative_detuning
        if detuning < NEAR_RESONANCE:
            self._warned.add(wavelength_m)
            warnings.warn(
                f"{wavelength_m * 1e9:.4f} nm is within a relative {detuning:.1e} of a "
                f"transition of {self.state}: the far-detuned polarizability sum is not "
                "valid this close to resonance.  Use kamo.hamiltonian's laser_sweep or "
                "kamo.imaging.response instead.", UserWarning, stacklevel=3)

    def __repr__(self) -> str:
        return f"StatePolarizability({self.state}, source={self.source!r})"
