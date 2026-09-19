"""The N x N coupled-dipole linear system, its solvers and the sanity checks.

    (delta_j + i/2) beta_j - sum_{l != j} (J_jl - (i/2) Gamma_jl) beta_l = -Omega_j / 2

with ``Omega_j = conj(e_hat) . E_inc(r_j) / E0``.  The matrix is

    M = diag(delta + i/2) - (J - i G / 2),        J, G zero on the diagonal,

i.e. ``M = H + (i/2) Gamma`` with ``H = diag(delta) - J`` real symmetric and
``Gamma = 1 + G`` the real symmetric decay matrix.

.. warning::
   The diagonal of the COUPLING term must be excluded.  Forming
   ``M = diag(delta + i/2) - (J - i Gamma/2)`` with ``Gamma`` carrying its unit
   diagonal puts ``i Gamma`` instead of ``i Gamma/2`` on the diagonal: every atom
   gets twice its linewidth.  Off resonance that is a 1% error; on resonance it
   halves the peak response and doubles the resonant shell width, under-
   estimating the near-field excess by ~2x.  Sanity check S1 catches it at once.

The scalar solve is EXACT for two-level sigma- atoms.  A rank-1 polarizability
``alpha e_hat conj(e_hat)`` forces every induced dipole along ``e_hat``, so the
``3N`` vector problem reduces identically to this one (test T8).  Do not
"improve" it to a vector solve: that is only needed to add the OTHER
transitions from the same ground state (:mod:`kamo.dd_solver.vector`).

Sanity checks run on every solve (:class:`SanityReport`):

S1  optical theorem, ``beta^dag Gamma beta = -Im(beta^dag Omega)``;
S2  positivity of ``Gamma`` (optional, O(N^3)); ``sum_mu gamma_mu = N``;
S4  reciprocity, ``M`` complex symmetric.

.. note::
   What S1 and S4 can and cannot see (established 2026-09-17).  S1 is the
   imaginary part of ``beta^dag M beta = -beta^dag Omega / 2``, so ANY real
   symmetric ``J`` drops out of it: flipping the sign of ``J`` alone leaves
   S1 = 0 and S4 = 0 while changing ``sum |beta|^2`` by a factor 2.3.  S4 is a
   symmetry statement about ``M`` and is equally blind to it.  What pins the
   dispersive convention is the near-field limit (T4), the analytic two-atom
   solution (T6), and the forward-amplitude form of the optical theorem,
   ``extinction = (4 pi / k) Im[pol* . (3/2k) F(khat)] / sigma0`` (T22), which
   ties the 3/2 prefactor, the ``-Omega/2`` right-hand side and the ``+i/2``
   together.  S3 is an ``N = 1`` solve and therefore involves no Green tensor at
   all: it checks the unit conventions, not the kernel.

Solvers: dense LU (``scipy.linalg.lu_factor`` with ``overwrite_a``),
``precision='mixed'`` (factor in complex64, ``refine`` steps of iterative
refinement in complex128), Jacobi-preconditioned GMRES, and a torch/CUDA path
(``backend='gpu'``) with the same options.  The CPU LU is the reference; every
path must pass the same tests.

``M`` is NOT diagonally dominant at the operating density: close pairs give
``|J_ij|`` far above ``|delta + i/2| = 9.15`` and near-resonant subradiant pair
modes push the condition number to ~10^3 already at N = 500 (measured 1653 at
N = 2000).  One refinement step then gives a median relative error of 3e-13 with
a worst case of 8e-10 over 30 configurations, not the 1e-13 an earlier version
of this docstring claimed; ``refine=2`` brings every case below 1e-13.  Mixed
precision is a ~4 % saving on a solve at N = 500 and is worth it only on the GPU
or at N >~ 2000.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from .cloud import Configuration
from .kernel import (gamma_matrix, kernel_for_variant, scalar_couplings,
                     scalar_couplings_multi)
from .rg import DEFAULT_CUTOFF, RGResult, renormalize_configuration
from .system import IncidentField, OperatingPoint, default_incident

OPTICAL_THEOREM_TOL = 1e-12
RECIPROCITY_TOL = 1e-13
POSITIVITY_TOL = -1e-10


class SanityWarning(UserWarning):
    """A physical invariant failed on a production solve."""


@dataclass
class SanityReport:
    optical_theorem: float = float("nan")   #: |b^dag G b + Im(b^dag Omega)| / |b^dag G b|
    reciprocity: float = float("nan")       #: max|M - M^T| / max|M|
    positivity_min_eig: Optional[float] = None   #: min eigenvalue of Gamma (S2), if computed
    decay_sum: Optional[float] = None       #: trace(Gamma) / N (must be 1)
    passed: bool = True
    messages: list = field(default_factory=list)

    def __str__(self):
        s = (f"S1 optical theorem {self.optical_theorem:.2e}   "
             f"S4 reciprocity {self.reciprocity:.2e}")
        if self.positivity_min_eig is not None:
            s += f"   S2 min eig(Gamma) {self.positivity_min_eig:+.2e}"
        return s + ("   OK" if self.passed else "   FAILED: " + "; ".join(self.messages))


@dataclass
class SolveResult:
    """Dipole amplitudes for one configuration, with everything needed to rebuild
    any observable without re-solving."""

    beta: np.ndarray
    config: Configuration
    op: OperatingPoint
    variant: str
    detunings: np.ndarray            #: per-atom detuning actually used (bare or RG)
    Omega: np.ndarray
    incident: IncidentField
    checks: SanityReport
    method: str = "lu"
    J: Optional[np.ndarray] = None
    G: Optional[np.ndarray] = None
    rg: Optional[RGResult] = None
    timings: dict = field(default_factory=dict)
    #: Diagonal decay rates actually carried by ``M`` (``None`` means all ones).
    #: Only the brightness-tracked RG puts anything else there.
    decay_diagonal: Optional[np.ndarray] = None

    # ------------------------------------------------------------ scalars
    @property
    def N(self) -> int:
        return int(self.beta.size)

    @property
    def excitation(self) -> float:
        """``sum_j |beta_j|^2`` -- the total excited-state POPULATION, in units of
        ``s0 / 2``; the per-atom measure the near-field excess law refers to.
        Independent atoms give ``sum_j |Omega_j/2|^2 / (delta_j^2 + 1/4)``.

        This is NOT the photon scattering rate (corrected 2026-09-17).  The rate
        is :attr:`radiated_power` ``= beta^dag Gamma beta``, and the two differ
        exactly where the excess lives: a subradiant close pair
        (``beta_i ~ -beta_j``, ``Gamma_ij ~ 1``) holds large population and
        radiates almost nothing.  Measured on 30 configurations at the equator,
        the excitation excess over independent atoms is 1.175 +- 0.141 while the
        scattered-photon excess is 1.072 +- 0.032; the configuration with
        excitation excess 5.2 scatters 1.02x the photons.  Quote this quantity
        for excitation, saturation and light shifts, and
        :attr:`radiated_power` for scattering, heating and loss rates."""
        return float(np.sum(np.abs(self.beta) ** 2))

    @property
    def radiated_power(self) -> float:
        """``beta^dag Gamma beta`` -- the power radiated by these dipoles into free
        space, in units of ``sigma0 I``, interference included.  Equal to the
        extinction by the optical theorem.

        Since 2026-09-17 every kernel variant carries the EXACT radiative
        ``Gamma`` (the decay matrix has no near-field part -- see
        :mod:`kamo.dd_solver.kernel`), so this is the physical radiated power for
        every variant, not only for ``'full'``.  It is positive semidefinite by
        construction.  The brightness-tracked RG is the one exception: it
        replaces the diagonal by phenomenological ``gamma_j``, and then this
        contracts with those rates, which is the identity that solve obeys but
        not a free-space radiated power."""
        if self.G is None:
            return self.extinction
        diag = np.ones(self.N) if self.decay_diagonal is None else self.decay_diagonal
        Gb = self.G @ self.beta + diag * self.beta
        return float(np.real(np.vdot(self.beta, Gb)))

    scattered_power = radiated_power

    @property
    def extinction(self) -> float:
        """``-Im(beta^dag Omega)`` -- power removed from the incident beam, same units.

        With an oscillator strength below 1 this EXCEEDS :attr:`radiated_power`,
        because a fraction ``1 - f`` of the scattering is Raman into a state that
        is dark to the probe (see :attr:`raman_leak`)."""
        return float(-np.imag(np.vdot(self.beta, self.Omega)))

    @property
    def raman_leak(self) -> float:
        """``extinction - radiated_power``: power scattered OUT of the two-level
        line, ``sum_j |beta_j|^2 (1/f_j - 1)``.  Zero for an ideal closed line."""
        return self.extinction - self.radiated_power

    @property
    def independent_beta(self) -> np.ndarray:
        """``-(f Omega/2) / (delta + i/2)`` with the BARE detunings."""
        return (-0.5 * self.op.strengths(self.config.spins) * self.Omega
                / (self.op.detunings(self.config.spins) + 0.5j))

    @property
    def independent_excitation(self) -> float:
        return float(np.sum(np.abs(self.independent_beta) ** 2))

    independent_power = independent_excitation

    @property
    def excess(self) -> float:
        """``excitation / independent_excitation``."""
        return self.excitation / self.independent_excitation

    def forward_amplitude(self, khat=None) -> complex:
        """On-axis scattered amplitude ``sum_j beta_j exp(-i k khat . r_j)``.

        The proxy every result used before the detected-mode projection
        (:mod:`kamo.dd_solver.detect`) existed.
        """
        kh = self.incident.khat if khat is None else np.asarray(khat, dtype=float)
        return complex(np.sum(self.beta * np.exp(-1j * self.op.k * (self.config.positions @ kh))))

    def per_atom(self) -> dict:
        """Per-atom diagnostics: who scatters hardest, and why."""
        d, ang, idx = self.config.nearest_neighbours()
        out = dict(beta=self.beta, power=np.abs(self.beta) ** 2,
                   independent_power=np.abs(self.independent_beta) ** 2,
                   spin=self.config.spins, detuning_bare=self.op.detunings(self.config.spins),
                   detuning_effective=self.detunings, nn_distance=d, nn_angle_z=ang,
                   nn_index=idx)
        if self.rg is not None:
            out["rg_shift"] = self.rg.shifts
            out["rg_times_renormalized"] = self.rg.n_renormalized
        return out

    def lightweight(self) -> "SolveResult":
        """Copy without the O(N^2) matrices (for ensembles)."""
        return SolveResult(self.beta, self.config, self.op, self.variant, self.detunings,
                           self.Omega, self.incident, self.checks, self.method, None, None,
                           self.rg, self.timings, self.decay_diagonal)

    def __repr__(self):
        return (f"SolveResult(N={self.N}, variant={self.variant!r}, theta={self.config.theta:.2f}, "
                f"excitation={self.excitation:.4g}, excess={self.excess:.3f}, "
                f"checks: {self.checks})")


# --------------------------------------------------------------- matrices


def build_matrix(positions, detunings, op: OperatingPoint, variant: str = "full",
                 block: int = 512, strengths=None, shared=None):
    """``(M, J, G)`` for the given positions and per-atom detunings.

    With an oscillator strength ``f_j < 1`` the diagonal is ``(delta_j + i/2)/f_j``
    and the off-diagonal kernel is unchanged: ``beta`` already carries the dipole
    magnitude, so ``f`` enters only through ``alpha_j = -(f_j/2)/(delta_j + i/2)``.
    """
    if shared is None:
        J, G = scalar_couplings(positions, op.k, op.e_hat, kernel_for_variant(variant),
                                block=block)
    else:
        J, G = shared
    M = -(J - 0.5j * G)
    idx = np.arange(len(detunings))
    diag = np.asarray(detunings, dtype=float) + 0.5j
    if strengths is not None:
        diag = diag / np.asarray(strengths, dtype=float)
    M[idx, idx] = diag
    return M, J, G


def effective_detunings(config: Configuration, op: OperatingPoint, variant: str,
                        rg_cutoff: float = DEFAULT_CUTOFF, rg_tracked: bool = False,
                        incident=None):
    """Bare detunings, or the RG-renormalized ones for ``variant='rg'``."""
    if variant == "rg":
        rg = renormalize_configuration(config, op, rg_cutoff, rg_tracked, incident)
        return rg.detunings, rg
    return op.detunings(config.spins), None


# ----------------------------------------------------------------- solvers


def _torch_device(backend):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("backend='gpu' but torch.cuda.is_available() is False")
    return torch, torch.device("cuda")


def solve_linear(M: np.ndarray, b: np.ndarray, method: str = "lu", precision: str = "double",
                 backend: str = "cpu", rtol: float = 1e-10, refine: int = 1) -> tuple:
    """Solve ``M x = b`` and return ``(x, info)``.

    ``method``: ``'lu'`` (dense, reference), ``'gmres'`` (Jacobi-preconditioned).
    ``precision``: ``'double'`` or ``'mixed'`` (complex64 factorisation, ``refine``
    steps of iterative refinement in complex128).  ``backend``: ``'cpu'`` or
    ``'gpu'`` (torch/CUDA).  The matrix is NOT overwritten.
    """
    info = dict(method=method, precision=precision, backend=backend)
    t0 = time.perf_counter()
    if backend == "gpu":
        torch, dev = _torch_device(backend)
        if method != "lu":
            raise ValueError("the GPU path implements method='lu' only")
        Md = torch.as_tensor(M, device=dev)
        bd = torch.as_tensor(b, device=dev)
        if precision == "mixed":
            LU, piv = torch.linalg.lu_factor(Md.to(torch.complex64))
            x = torch.linalg.lu_solve(LU, piv, bd.to(torch.complex64)[:, None])[:, 0].to(torch.complex128)
            for _ in range(refine):
                r = bd - Md @ x
                x = x + torch.linalg.lu_solve(LU, piv, r.to(torch.complex64)[:, None])[:, 0].to(torch.complex128)
        else:
            x = torch.linalg.solve(Md, bd)
        torch.cuda.synchronize()
        out = x.cpu().numpy()
    elif method == "lu":
        if precision == "mixed":
            lu, piv = sla.lu_factor(M.astype(np.complex64), overwrite_a=True, check_finite=False)
            x = sla.lu_solve((lu, piv), b.astype(np.complex64), check_finite=False).astype(np.complex128)
            for _ in range(refine):
                r = b - M @ x
                x = x + sla.lu_solve((lu, piv), r.astype(np.complex64),
                                     check_finite=False).astype(np.complex128)
        elif precision == "double":
            lu, piv = sla.lu_factor(M.copy(), overwrite_a=True, check_finite=False)
            x = sla.lu_solve((lu, piv), b, check_finite=False)
        else:
            raise ValueError("precision must be 'double' or 'mixed'")
        out = x
    elif method == "gmres":
        dinv = 1.0 / np.diag(M)
        pre = spla.LinearOperator(M.shape, matvec=lambda v: dinv * v, dtype=M.dtype)
        it = [0]

        def cb(_):
            it[0] += 1

        x, flag = spla.gmres(M, b, M=pre, rtol=rtol, restart=200, maxiter=2000, callback=cb,
                             callback_type="pr_norm")
        if flag != 0:
            warnings.warn(f"GMRES did not converge (flag {flag}, {it[0]} iterations)", SanityWarning)
        info["iterations"] = it[0]
        out = x
    else:
        raise ValueError(f"unknown method {method!r}")
    info["seconds"] = time.perf_counter() - t0
    return out, info


def sanity_checks(M: np.ndarray, G: np.ndarray, beta: np.ndarray, Omega: np.ndarray,
                  positivity: bool = False, warn: bool = True, decay_diagonal=None) -> SanityReport:
    """S1, S4 and optionally S2 for a solved system.

    S1 contracts ``beta`` with ``diag(decay_diagonal) + G``, where
    ``decay_diagonal`` is what the PHYSICS says the per-atom extinction rate must
    be: 1 for an ideal closed line, ``1/f_j`` with an oscillator strength below
    1, and ``gamma_j/f_j`` for the brightness-tracked RG.  ``None`` means all
    ones.

    Two things depend on this being the expected value rather than one read back
    off ``M`` (2026-09-17).  Reading ``2 Im diag(M)`` would make S1 an algebraic
    identity of the linear system and therefore incapable of failing -- it would
    no longer catch the double-linewidth bug this check exists for.  Using a
    hard-wired 1 instead flagged every exact tracked-RG solve as a 29 %
    "optical theorem violated".
    """
    rep = SanityReport()
    diag_decay = 1.0 if decay_diagonal is None else np.asarray(decay_diagonal, dtype=float)
    Gb = G @ beta + diag_decay * beta
    lhs = float(np.real(np.vdot(beta, Gb)))
    rhs = float(-np.imag(np.vdot(beta, Omega)))
    rep.optical_theorem = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    rep.reciprocity = float(np.max(np.abs(M - M.T)) / np.max(np.abs(M)))
    rep.decay_sum = float(np.trace(G) / G.shape[0] + 1.0)
    if rep.optical_theorem > OPTICAL_THEOREM_TOL * max(1.0, np.sqrt(beta.size)):
        rep.passed = False
        rep.messages.append(f"optical theorem violated ({rep.optical_theorem:.2e})")
    if rep.reciprocity > RECIPROCITY_TOL:
        rep.passed = False
        rep.messages.append(f"matrix not symmetric ({rep.reciprocity:.2e})")
    if positivity:
        ev = np.linalg.eigvalsh(gamma_matrix(G))
        rep.positivity_min_eig = float(ev.min())
        if rep.positivity_min_eig < POSITIVITY_TOL:
            rep.passed = False
            rep.messages.append(f"Gamma not PSD (min eig {rep.positivity_min_eig:+.2e})")
    if warn and not rep.passed:
        warnings.warn("coupled-dipole sanity check FAILED: " + "; ".join(rep.messages),
                      SanityWarning, stacklevel=3)
    return rep


# ------------------------------------------------------------------- solve


def solve(config: Configuration, op: OperatingPoint, variant: str = "full",
          incident: Optional[IncidentField] = None, method: str = "lu",
          precision: str = "double", backend: str = "cpu", rg_cutoff: float = DEFAULT_CUTOFF,
          rg_tracked: bool = False, checks: bool = True, positivity: bool = False,
          keep_matrices: bool = True, warn: bool = True, block: int = 512,
          _shared=None) -> SolveResult:
    """Solve the coupled-dipole system for one configuration.

    Parameters
    ----------
    config : Configuration
    op : OperatingPoint
    variant : {'full', 'far', 'nonear', 'rg', 'independent'}
        See :mod:`kamo.dd_solver.kernel`.  ``'nonear'`` is the literature
        near-field ablation (exact ``Gamma``, static ``1/r^3`` removed);
        ``'far'`` keeps only the ``1/x`` coherent coupling.
    incident : IncidentField, optional
        Default: unit plane wave along +x polarized along y.
    method, precision, backend
        See :func:`solve_linear`.
    rg_cutoff, rg_tracked
        Strong-disorder RG options for ``variant='rg'``.
    checks, positivity
        Run the sanity checks (S1, S4 always; S2 when ``positivity`` -- O(N^3)).
    keep_matrices : bool
        Keep ``J`` and ``G`` on the result (needed for :attr:`scattered_power`
        by quadratic form; otherwise it falls back to the extinction, which the
        optical theorem makes equal).
    """
    inc = default_incident(op) if incident is None else incident
    t0 = time.perf_counter()
    det, rg = effective_detunings(config, op, variant, rg_cutoff, rg_tracked, inc)
    t_rg = time.perf_counter() - t0
    Omega = inc.drive(config.positions, op.e_hat)
    if rg is not None and rg.tracked:
        Omega = rg.drive
    t0 = time.perf_counter()
    f = op.strengths(config.spins)
    M, J, G = build_matrix(config.positions, det, op, variant, block=block,
                           strengths=f, shared=_shared)
    # Elastic decay diagonal (what the dipoles radiate on the driven line) and
    # the extinction diagonal (what they remove from the beam, larger by 1/f).
    decay_diag = None
    extinction_diag = 1.0 / f
    if rg is not None and rg.tracked:
        idx = np.arange(config.N)
        M[idx, idx] = (det + 0.5j * rg.gamma) / f
        decay_diag = np.asarray(rg.gamma, dtype=float)
        extinction_diag = decay_diag / f
    t_build = time.perf_counter() - t0
    beta, info = solve_linear(M, -0.5 * Omega, method, precision, backend)
    if checks:
        rep = sanity_checks(M, G, beta, Omega, positivity=positivity, warn=warn,
                            decay_diagonal=extinction_diag)
    else:
        rep = SanityReport()
    timings = dict(rg=t_rg, build=t_build, solve=info.get("seconds", float("nan")))
    if "iterations" in info:
        timings["iterations"] = info["iterations"]
    return SolveResult(beta, config, op, variant, det, Omega, inc, rep,
                       f"{method}/{precision}/{backend}",
                       J if keep_matrices else None, G if keep_matrices else None, rg, timings,
                       decay_diag)


def solve_variants(config: Configuration, op: OperatingPoint, variants=("full", "far"),
                   incident: Optional[IncidentField] = None, block: int = 512, **kw):
    """``{variant: SolveResult}`` for one configuration, sharing the geometry pass.

    Equivalent to calling :func:`solve` once per variant, and about 3x faster at
    N = 500 because the separations, unit vectors, angular factors and ``Gamma``
    are built once (see :func:`kamo.dd_solver.kernel.scalar_couplings_multi`).
    ``'independent'`` is returned from the closed form with no matrix at all.
    """
    inc = default_incident(op) if incident is None else incident
    kernels = [v for v in variants if v != "independent"]
    shared = (scalar_couplings_multi(config.positions, op.k, op.e_hat, kernels, block=block)
              if kernels else {})
    out = {}
    for v in variants:
        out[v] = solve(config, op, v, incident=inc, block=block,
                       _shared=shared.get(v), **kw)
    return out


def independent_solution(config: Configuration, op: OperatingPoint,
                         incident: Optional[IncidentField] = None) -> np.ndarray:
    """``beta`` for non-interacting atoms (J = 0, Gamma_ij = delta_ij)."""
    inc = default_incident(op) if incident is None else incident
    Omega = inc.drive(config.positions, op.e_hat)
    return -0.5 * op.strengths(config.spins) * Omega / (op.detunings(config.spins) + 0.5j)


def single_atom_cross_section(op: OperatingPoint, delta: float = 0.0,
                              polarization=None) -> float:
    """S3: the scattering cross section (m^2) of one atom at detuning ``delta``,
    computed by running the solver on N = 1 and converting with the code's own
    unit conventions -- ``P = sigma0 I |beta|^2`` -- so a mismatch between the
    dimensionless units and the lab shows up as a wrong number here.

    Driven by a sigma- plane wave (polarization ``e_hat``, propagating along z)
    on resonance this must return ``sigma0 = 3 lambda^2 / 2 pi``; for the lab's
    y-polarized probe along x it returns ``sigma0 / 2``, because only the sigma-
    projection of the field drives the transition.
    """
    from .system import PlaneWave
    if polarization is None:
        inc = PlaneWave(op.k, khat=(0, 0, 1), polarization=op.e_hat)
    else:
        inc = PlaneWave(op.k, khat=(1, 0, 0), polarization=polarization)
    cfg = Configuration(np.zeros((1, 3)), np.array([1], dtype=np.int8))
    op1 = OperatingPoint(op.linewidth_Hz, op.wavelength, delta_up=float(delta),
                         delta_dn=float(delta))
    res = solve(cfg, op1, "full", incident=inc)
    return op.sigma0 * float(np.abs(res.beta[0]) ** 2)


def scattering_rate_from_intensity(op: OperatingPoint, beta_sq: float, intensity: float) -> float:
    """``gamma_sc = sigma0 I |beta|^2 / (hbar omega)`` in photons/s, for S3."""
    import kamo.constants as kc
    omega = 2 * np.pi * kc.c / op.wavelength
    return op.sigma0 * intensity * beta_sq / (kc.hbar * omega)
