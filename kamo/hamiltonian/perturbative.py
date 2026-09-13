"""Second-order light shift over the exact eigenstates of ``h0 + B * Zeeman``:
the ``"perturbative"`` laser model.

The ``"stark"`` model applies fine-structure polarizabilities, which lump every
coupled manifold at one energy (error ~ hyperfine + Zeeman spread over the
detuning, and exactly zero for a differential between states that differ only
in m_I).  The ``"rwa"`` model resolves everything but drops the
counter-rotating terms (error ~ detuning over twice the optical frequency) and
sees only channels present in the basis.  This model keeps the best of both in
the perturbative regime.  With ``E(t) = (E0/2)(eps e^{-i w t} + eps* e^{+i w t})``
the second-order shift of eigenstate ``|k>`` is

    dE_k = (E0^2 / 4) sum_m [ |<m| d.eps |k>|^2 / (E_k - E_m + h f_L)
                            + |<k| d.eps |m>|^2 / (E_k - E_m - h f_L) ]

with ``|k>``, ``|m>`` the exact eigenstates of ``h0 + B * Zeeman`` (hyperfine
and Zeeman resolved, admixtures included) and the same portal-backed dipole
matrix elements that :meth:`HamiltonianBuilder.laser_rwa_operator` uses.  The
first term is absorption of a photon (``d.eps`` raises m by q for sigma_q
light), the second is emission (``d.eps*``); for circular light the two act on
different sublevels, which is what gives the vector polarizability its
``(-1)^K`` counter-rotating sign.  Both terms are kept, so far from resonance
the model reproduces the fine-structure polarizabilities, and near resonance
it reproduces the RWA until second-order perturbation theory itself fails
(Rabi / 2 detuning no longer small).

Channels whose manifold is not in the basis, and the ionic core, are added
back as a residual fine-structure polarizability per manifold (scalar, vector
and tensor parts, from the same per-channel formula as
:class:`kamo.light_shift.ComputePolarizabilities`) applied through
:meth:`HamiltonianBuilder.laser_stark_operator`.  Those channels are lumped
at one energy, which is exactly right for them: a channel is outside a
:func:`kamo.hamiltonian.light_shift_basis` basis only when it carries less
than 1e-3 of the polarizability, and its substructure error is 1e-3 of that.

The second-order effective operator is built within each ``(n, l)`` group of
eigenstates (the near-degenerate subspaces a Schrieffer-Wolff transformation
acts in), so Raman-type two-photon couplings between sublevels of one manifold
are included; couplings between different ``(n, l)`` are dropped, as they only
act against an optical energy denominator.

Entry points: :func:`sweep_intensity_perturbative` (what
``AtomicStructure.laser_sweep(model="perturbative")`` calls),
:func:`perturbative_stark_operator` for the operator itself,
:func:`dipole_operator` for ``d.eps`` in the bare basis, and
:func:`channel_polarizability_components` for one channel's scalar, vector and
tensor contributions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import kamo.constants as kc

from .diagonalize import LaserSweepResult, eigenshuffle

__all__ = ["perturbative_stark_operator", "sweep_intensity_perturbative",
           "choose_sweep_model", "SweepModelChoice",
           "dipole_operator", "channel_polarizability_components",
           "residual_polarizability_au"]

NLJ = Tuple[int, int, float]


def _beam_frequency(beam) -> float:
    f = getattr(beam, "frequency", None)
    if callable(f):
        return float(f())
    return float(beam.frequency_Hz)


# ------------------------------------------------------------ dipole operator

def dipole_operator(builder, polarization="pi") -> np.ndarray:
    """``A[m, k] = <m| d.eps |k> / (2 h)`` in the bare basis, Hz per (V/m).

    Unlike the Hermitian-filled ``coupling`` of
    :meth:`HamiltonianBuilder.laser_rwa_operator`, this is the operator
    ``d.eps`` itself, which is not Hermitian for circular light: ``A[m, k]`` is
    non-zero only for ``m_j(m) = m_j(k) + q`` with ``eps_q != 0`` (absorption
    of a sigma_q photon raises m), whichever of ``m`` and ``k`` lies higher in
    energy.  ``A^H`` is ``d.eps*``.  Nuclear spin is a spectator.

    Uses ``atom.getDipoleMatrixElement(k, m, q)``, whose ARC convention is
    ``<m| d_q |k>`` (non-zero for ``m_j(m) = m_j(k) + q``; checked numerically
    2026-09-13 against the stretched 4S-4P3/2 element).
    """
    pol = builder._resolve_polarization(polarization)
    dim = builder.basis.dim
    A = np.zeros((dim, dim), dtype=complex)
    scale = kc.a0 * kc.e / (2.0 * kc.h)
    states = list(builder.basis)
    for k in states:
        for m in states:
            if abs(k.l - m.l) != 1 or abs(k.m_i - m.m_i) > 1e-9:
                continue
            tot = 0.0
            for q, amp in pol.items():
                if abs((k.m_j + q) - m.m_j) > 1e-9:
                    continue
                tot += amp * builder.atom.getDipoleMatrixElement(
                    k.n, k.l, k.j, k.m_j, m.n, m.l, m.j, m.m_j, q)
            if tot != 0.0:
                A[m.index, k.index] = tot * scale
    return A


# ------------------------------------------------------------ residual channels

def channel_polarizability_components(j: float, jf: float, d_au: float,
                                      energy_Hz: float, f_laser_Hz: float
                                      ) -> Tuple[float, float, float]:
    """Scalar, vector and tensor polarizability (a.u.) of one dipole channel.

    ``j`` and ``jf`` are the initial and final J, ``d_au`` the reduced matrix
    element (e a0) and ``energy_Hz`` the signed transition frequency
    ``(E_f - E_i)/h``.  Term-by-term identical to
    :meth:`kamo.light_shift.ComputePolarizabilities.compute_fine_structure_polarizability`,
    so summing over every channel of a state reproduces that method (a test
    pins it); the vector part carries ``(-1)^K`` on the counter-rotating term
    through its ``E_L`` numerator.
    """
    from sympy.physics import wigner

    E = energy_Hz * kc.h
    E_L = f_laser_Hz * kc.h
    d_SI = d_au * kc.a0 * kc.e
    common = d_SI ** 2 / (E ** 2 - E_L ** 2)
    a_s = common * E * (2.0 / 3.0) / (2 * j + 1)
    a_v = (common * (-1) ** (j + jf + 1) * E_L * float(wigner.wigner_6j(j, 1, j, 1, jf, 1))
           * np.sqrt(24 * j / (j + 1) / (2 * j + 1)))
    a_t = (common * (-1) ** (j + jf) * E * float(wigner.wigner_6j(j, 2, j, 1, jf, 1))
           * np.sqrt(40 * j * (2 * j - 1) / (3 * (j + 1) * (2 * j + 3) * (2 * j + 1))))
    conv = kc.convert_polarizability_au_to_SI
    return float(a_s / conv), float(a_v / conv), float(a_t / conv)


def residual_polarizability_au(nlj: NLJ, f_laser_Hz: float, present: Iterable[NLJ],
                               B_gauss: float = 0.0) -> Tuple[float, float, float]:
    """``(alpha_s, alpha_v, alpha_t)`` in a.u. of manifold ``nlj`` from the channels
    whose final manifold is *not* in ``present``, plus the ionic core (scalar)."""
    from .laser_model import state_channels
    present = set(present)
    sc = state_channels(nlj, f_laser_Hz, B_gauss=B_gauss)
    a_s, a_v, a_t = sc.core_au, 0.0, 0.0
    for ch in sc.channels:
        if ch.manifold in present:
            continue
        s, v, t = channel_polarizability_components(nlj[2], ch.manifold[2], ch.d_au,
                                                    ch.energy_Hz, f_laser_Hz)
        a_s, a_v, a_t = a_s + s, a_v + v, a_t + t
    return float(a_s), float(a_v), float(a_t)


@dataclass
class _ResidualPolarizabilities:
    """Adapter with the ``compute_fine_structure_polarizability`` interface that
    :meth:`HamiltonianBuilder.laser_stark_operator` expects, serving fixed
    residual triples per manifold."""

    table: Dict[NLJ, Tuple[float, float, float]]

    def compute_fine_structure_polarizability(self, n, l, j, wavelength_m):
        return self.table[(int(n), int(l), float(j))]


# ----------------------------------------------------------------- operator

def perturbative_stark_operator(builder, beam, polarization="pi", B_gauss: float = 0.0,
                                include_quadrupole: bool = True, residual: bool = True,
                                eta_warn: Optional[float] = 0.1, I_ref: float = None) -> Dict:
    """Second-order light-shift operator *per W/m^2* over the exact eigenstates.

    Returns a dict with

    ``operator``
        ndarray (dim, dim), Hz per W/m^2, in the bare basis.
        ``H(I) = H0 + operator * I`` with ``H0 = h0 + B * zeeman``.
    ``H0``, ``energies``, ``vectors``
        the field Hamiltonian and its eigen-decomposition (Hz; columns are
        eigenvectors).
    ``shift_per_I``
        ndarray (dim,), the diagonal second-order shift of each eigenstate in
        Hz per W/m^2 from the channels in the basis (before the residual).
    ``residual_au``
        {(n, l, j): (alpha_s, alpha_v, alpha_t) in a.u.} of the channels
        outside the basis plus the core, or {} when ``residual=False``.
    ``eta``
        max over coupled eigenstate pairs of ``|V| / |E_k - E_m + h f_L|`` at
        ``I_ref`` (the near-resonant denominators): the small parameter of
        the expansion.  A warning is raised above ``eta_warn`` (``None``
        disables it).  ``I_ref`` defaults to the beam's ``I0``; pass 0 or
        None to skip.
    ``eps_rwa_dominant``
        counter-rotating error estimate ``|f_c - f_L| / (f_c + f_L)`` of the
        most strongly driven pair (the one setting ``eta``); what an RWA
        sweep of this basis would be off by.  0 when ``eta`` was not evaluated.
    """
    f_L = _beam_frequency(beam)
    H0 = builder.h0(include_quadrupole=include_quadrupole)
    if B_gauss:
        H0 = H0 + builder.zeeman_operator() * float(B_gauss)
    H0 = np.asarray(H0, dtype=complex)
    f, U = np.linalg.eigh(H0)                      # Hz, columns
    dim = f.size

    A = U.conj().T @ dipole_operator(builder, polarization) @ U   # <m|d.eps|k>/2h, eigenbasis
    e2 = 2.0 / (kc.c * kc.epsilon0)                          # E0^2 per W/m^2

    dk = f[:, None] - f[None, :]                              # f_k - f_m
    with np.errstate(divide="ignore", invalid="ignore"):
        Dp = 1.0 / (dk + f_L)                                 # absorption denominators
        Dm = 1.0 / (dk - f_L)                                 # emission denominators
    for D in (Dp, Dm):
        np.fill_diagonal(D, 0.0)
    Dp = np.nan_to_num(Dp, nan=0.0, posinf=0.0, neginf=0.0)
    Dm = np.nan_to_num(Dm, nan=0.0, posinf=0.0, neginf=0.0)

    # group eigenstates by dominant (n, l): the subspaces the effective
    # operator acts within
    slices = list(builder.basis.manifold_slices())
    nl_of_slice = [(man.n, man.l) for man, _ in slices]
    nl_ids = {nl: i for i, nl in enumerate(dict.fromkeys(nl_of_slice))}
    weights = np.abs(U) ** 2
    w_man = np.stack([weights[sl, :].sum(axis=0) for _, sl in slices])    # (n_man, dim)
    group = np.array([nl_ids[nl_of_slice[int(k)]] for k in np.argmax(w_man, axis=0)])
    G = (group[:, None] == group[None, :])

    # W[k,k'] = e2/2 sum_m [ A*_mk A_mk' (Dp_km + Dp_k'm) + A_km A*_k'm (Dm_km + Dm_k'm) ]
    Ac = np.conj(A)
    DpT = Dp.T                                                # DpT[m, k] = Dp[k, m]
    plus = (Ac * DpT).T @ A + A.conj().T @ (A * DpT)
    minus = (A * Dm) @ A.conj().T + A @ (Ac * Dm).T
    W = 0.5 * e2 * (plus + minus) * G
    W = 0.5 * (W + W.conj().T)                                # exact Hermitian
    shift_per_I = np.real(np.diag(W)).copy()

    residual_au: Dict[NLJ, Tuple[float, float, float]] = {}
    if residual:
        present = [man.nlj for man, _ in slices]
        for man, _ in slices:
            residual_au[man.nlj] = residual_polarizability_au(man.nlj, f_L, present,
                                                              B_gauss=B_gauss)
        S_res = builder.laser_stark_operator(
            beam, polarizabilities=_ResidualPolarizabilities(residual_au),
            polarization=polarization)
        # within-manifold operator, so restricting it to the (n, l) blocks is exact
        W = W + (U.conj().T @ np.asarray(S_res, dtype=complex) @ U) * G
    Sop = U @ W @ U.conj().T

    # small parameter of the expansion at the reference intensity
    eta = 0.0
    eps_rwa_dominant = 0.0
    if I_ref is None:
        I_ref = float(getattr(beam, "I0", 0.0) or 0.0)
    if I_ref and I_ref > 0:
        E0 = np.sqrt(e2 * float(I_ref))
        ratio = np.maximum(np.abs(A) * np.abs(DpT), np.abs(A).T * np.abs(Dm)) * E0
        m_i, k_i = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        eta = float(ratio[m_i, k_i])
        f_c = abs(f[m_i] - f[k_i])
        eps_rwa_dominant = float(abs(f_c - f_L) / (f_c + f_L)) if eta > 0 else 0.0
        if eta_warn is not None and eta > eta_warn:
            warnings.warn(
                f"perturbative laser model: Rabi/(2 detuning) reaches {eta:.2f} at "
                f"I = {I_ref:.3g} W/m^2; second-order perturbation theory is not "
                "reliable here, use model='rwa'.", RuntimeWarning, stacklevel=2)

    return {"operator": Sop, "H0": H0, "energies": f, "vectors": U,
            "W_eig": W, "group": group,
            "shift_per_I": shift_per_I, "residual_au": residual_au,
            "eta": eta, "eps_rwa_dominant": eps_rwa_dominant, "f_laser": f_L}


@dataclass(frozen=True)
class SweepModelChoice:
    """Outcome of :func:`choose_sweep_model` (basis-level, no transition given)."""

    model: str                  #: "rwa" or "perturbative"
    reason: str
    eta: float                  #: max Rabi / (2 detuning) over the basis at I_max
    eps_rwa: float              #: counter-rotating error of the most strongly driven pair
    eps_perturbative: float     #: eta^2

    def describe(self) -> str:
        return (f"model={self.model!r}: {self.reason} (eta={self.eta:.1e}, "
                f"eps_rwa~{self.eps_rwa:.1e}, eps_perturbative~{self.eps_perturbative:.1e})")


def choose_sweep_model(builder, beam, I_max: float, polarization="pi", B_gauss: float = 0.0,
                       include_quadrupole: bool = True, residual: bool = True,
                       eta_max: float = 0.1) -> Tuple[SweepModelChoice, Dict]:
    """Pick ``"rwa"`` or ``"perturbative"`` for a whole-basis intensity sweep.

    The transition-aware choice lives in :func:`kamo.hamiltonian.choose_laser_model`;
    this is its basis-level counterpart for :func:`kamo.hamiltonian.sweep_intensity`,
    where no pair of states is singled out.  It builds the perturbative
    operator once (returned as the second element so the caller can reuse it)
    and decides from the most strongly driven pair in the basis at ``I_max``:
    ``"rwa"`` when ``eta = Rabi/(2 detuning)`` exceeds ``eta_max`` or when
    ``eta^2`` exceeds that pair's counter-rotating error estimate,
    ``"perturbative"`` otherwise.
    """
    op = perturbative_stark_operator(builder, beam, polarization=polarization,
                                     B_gauss=B_gauss, include_quadrupole=include_quadrupole,
                                     residual=residual, eta_warn=None, I_ref=float(I_max))
    eta, eps_rwa = op["eta"], op["eps_rwa_dominant"]
    if eta > eta_max:
        model, reason = "rwa", f"non-perturbative: Rabi/(2 detuning) = {eta:.2f} > {eta_max}"
    elif eta ** 2 > eps_rwa:
        model, reason = "rwa", ("counter-rotating error below the next order of the "
                                "perturbative sum")
    else:
        model, reason = "perturbative", ("perturbative light; exact eigenstates with both "
                                         "rotating terms")
    return SweepModelChoice(model, reason, float(eta), float(eps_rwa), float(eta ** 2)), op


def sweep_intensity_perturbative(builder, beam, I_max: float, n_points: int = 200,
                                 polarization="pi", B_gauss: float = 0.0,
                                 include_quadrupole: bool = True,
                                 residual: bool = True, op: Optional[Dict] = None
                                 ) -> LaserSweepResult:
    """Intensity sweep with the second-order operator of
    :func:`perturbative_stark_operator`; same return type as
    :func:`kamo.hamiltonian.sweep_intensity`.

    The shift is linear in I by construction (apart from second-order effects
    of the within-manifold couplings against the Zeeman splitting), so a
    handful of points is enough; the sweep exists so that the result plugs
    into the same eigenshuffle-tracked :class:`LaserSweepResult` API.

    The effective operator is block diagonal in the ``(n, l)`` groups, so each
    block is diagonalized on its own about its mean energy.  Round-off then
    scales with the block's own spread (GHz for 4S, the fine-structure
    splitting for 4P) instead of the optical energies in the full matrix,
    which is what lets a within-manifold differential of a few mHz per
    kW/cm^2 come out clean; a full-matrix diagonalization with 4e14 Hz on the
    diagonal is only good to ~0.05 Hz.
    """
    if op is None:
        op = perturbative_stark_operator(builder, beam, polarization=polarization,
                                         B_gauss=B_gauss, include_quadrupole=include_quadrupole,
                                         residual=residual, I_ref=float(I_max))
    I = np.linspace(0.0, float(I_max), int(n_points))
    f, U, W, group = op["energies"], op["vectors"], op["W_eig"], op["group"]
    dim = f.size
    energies = np.empty((I.size, dim), dtype=float)
    vectors = np.empty((I.size, dim, dim), dtype=complex)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        f_bar = float(np.mean(f[idx]))
        Wg = W[np.ix_(idx, idx)]
        base = np.diag(f[idx] - f_bar).astype(complex)
        e_g, v_g = eigenshuffle([base + Wg * inten for inten in I])
        energies[:, idx] = e_g + f_bar
        Ug = U[:, idx]
        for s in range(I.size):
            vectors[s][:, idx] = Ug @ v_g[s]
    order = np.argsort(energies[0])                   # ascending at I = 0, as eigenshuffle does
    energies = energies[:, order]
    vectors = vectors[:, :, order]
    res = LaserSweepResult(
        I, "Intensity (W/m^2)", energies, vectors, builder.basis,
        beam=beam, polarization=polarization, B_gauss=B_gauss,
        _builder=builder,
    )
    res.perturbative = {k: op[k] for k in ("shift_per_I", "residual_au", "eta",
                                           "eps_rwa_dominant", "f_laser")}
    return res
