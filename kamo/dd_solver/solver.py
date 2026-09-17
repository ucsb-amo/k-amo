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

Solvers: dense LU (``scipy.linalg.lu_factor`` with ``overwrite_a``),
``precision='mixed'`` (factor in complex64, one step of iterative refinement in
complex128 -- the matrix is strongly diagonally dominant, cond ~ 10^1-10^2, so
one step recovers 1e-13), Jacobi-preconditioned GMRES, and a torch/CUDA path
(``backend='gpu'``) with the same options.  The CPU LU is the reference; every
path must pass the same tests.
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
from .kernel import gamma_matrix, scalar_couplings
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

    # ------------------------------------------------------------ scalars
    @property
    def N(self) -> int:
        return int(self.beta.size)

    @property
    def excitation(self) -> float:
        """``sum_j |beta_j|^2`` -- the total excited-state population (photon
        scattering events per atom, summed), the per-atom measure the near-field
        excess law of the build specification refers to.  Independent atoms give
        ``sum_j |Omega_j/2|^2 / (delta_j^2 + 1/4)``."""
        return float(np.sum(np.abs(self.beta) ** 2))

    @property
    def radiated_power(self) -> float:
        """``beta^dag Gamma beta`` -- the coherently RADIATED power in units of
        ``sigma0 I``, interference included.  Equal to the extinction by the
        optical theorem.  Physical only for the ``'full'`` kernel: the far-field-
        only ``Gamma`` is not positive semidefinite, so for ``'far'``/``'rg'``
        this can even be negative."""
        if self.G is None:
            return self.extinction
        Gb = self.G @ self.beta + self.beta
        return float(np.real(np.vdot(self.beta, Gb)))

    scattered_power = radiated_power

    @property
    def extinction(self) -> float:
        """``-Im(beta^dag Omega)`` -- power removed from the incident beam, same units."""
        return float(-np.imag(np.vdot(self.beta, self.Omega)))

    @property
    def independent_beta(self) -> np.ndarray:
        """``-(Omega/2) / (delta + i/2)`` with the BARE detunings."""
        return -0.5 * self.Omega / (self.op.detunings(self.config.spins) + 0.5j)

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
                           self.rg, self.timings)

    def __repr__(self):
        return (f"SolveResult(N={self.N}, variant={self.variant!r}, theta={self.config.theta:.2f}, "
                f"excitation={self.excitation:.4g}, excess={self.excess:.3f}, "
                f"checks: {self.checks})")


# --------------------------------------------------------------- matrices


def build_matrix(positions, detunings, op: OperatingPoint, variant: str = "full",
                 block: int = 512):
    """``(M, J, G)`` for the given positions and per-atom detunings."""
    kern = "far" if variant == "rg" else variant
    J, G = scalar_couplings(positions, op.k, op.e_hat, kern, block=block)
    M = -(J - 0.5j * G)
    idx = np.arange(len(detunings))
    M[idx, idx] = np.asarray(detunings, dtype=float) + 0.5j
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
                  positivity: bool = False, warn: bool = True) -> SanityReport:
    """S1, S4 and optionally S2 for a solved system."""
    rep = SanityReport()
    Gb = G @ beta + beta
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
          keep_matrices: bool = True, warn: bool = True, block: int = 512) -> SolveResult:
    """Solve the coupled-dipole system for one configuration.

    Parameters
    ----------
    config : Configuration
    op : OperatingPoint
    variant : {'full', 'far', 'rg', 'independent'}
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
    M, J, G = build_matrix(config.positions, det, op, variant, block=block)
    if rg is not None and rg.tracked:
        idx = np.arange(config.N)
        M[idx, idx] = det + 0.5j * rg.gamma
    t_build = time.perf_counter() - t0
    beta, info = solve_linear(M, -0.5 * Omega, method, precision, backend)
    if checks:
        rep = sanity_checks(M, G, beta, Omega, positivity=positivity, warn=warn)
    else:
        rep = SanityReport()
    timings = dict(rg=t_rg, build=t_build, solve=info.get("seconds", float("nan")))
    if "iterations" in info:
        timings["iterations"] = info["iterations"]
    return SolveResult(beta, config, op, variant, det, Omega, inc, rep,
                       f"{method}/{precision}/{backend}",
                       J if keep_matrices else None, G if keep_matrices else None, rg, timings)


def independent_solution(config: Configuration, op: OperatingPoint,
                         incident: Optional[IncidentField] = None) -> np.ndarray:
    """``beta`` for non-interacting atoms (J = 0, Gamma_ij = delta_ij)."""
    inc = default_incident(op) if incident is None else incident
    Omega = inc.drive(config.positions, op.e_hat)
    return -0.5 * Omega / (op.detunings(config.spins) + 0.5j)


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
