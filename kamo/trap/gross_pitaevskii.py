"""Exact Gross-Pitaevskii ground states on the real trap potential.

Solves ``[-hbar^2 nabla^2 / 2m + V(r) + g N |psi|^2] psi = mu psi`` in 3D, with
``int |psi|^2 = 1`` and ``g = 4 pi hbar^2 a / m``, on a uniform
:class:`~kamo.trap.grid.TrapGrid` with periodic (FFT) kinetic energy.

Method
------
1. **Grid.**  Resolution ``dx_i = min(xi, sigma_i) / points_per_scale`` -- the
   healing length binds in the Thomas-Fermi regime, the cloud width in the
   ideal-gas one, per axis, so the weak axis is legitimately coarser.  Extent
   ``L_i = box_factor * max(4 sigma_i, R_TF,i)`` from the Gaussian-variational
   cloud of the harmonic expansion; afterwards the density on the box faces
   must be below 1e-8 of the peak (the FFT is periodic, so this is the
   wraparound control) or the box grows 1.3x and the solve reruns.  ``grid=``
   fixes the grid instead (for convergence studies and finite differences in N).
2. **Basin.**  Outside the connected region below the escape energy the
   potential is flattened to it, so the solver cannot find the box corner that
   gravity pulls downhill.
3. **Imaginary time.**  Normalized gradient flow, Strang-split potential-first
   (the nonlinear term and the normalization are both real-space, so they sit
   between FFTs), from the variational Gaussian centred on the *sagged* minimum,
   until ``mu`` settles to ``stage_tol``.  This only has to get close: its
   fixed point is ``O(dtau^2)`` away from the true ground state.
4. **Polish.**  Preconditioned conjugate gradient on the unit sphere, minimizing
   the GP energy functional with an *exact* line search (along
   ``cos t psi + sin t d`` the energy is a closed-form quartic in ``cos t``,
   ``sin t``).  Every step lowers the energy; the fixed point is the GP equation
   itself, with no splitting bias.  It stops when
   ``||H_GP psi - mu psi|| / (mu - V_min) < residual_tol`` (1e-8).

Observables (per atom): ``E_kin`` spectrally, ``E_pot``, ``E_int = g N int |psi|^4 / 2``;
``mu = E_kin + E_pot + 2 E_int``, ``E = E_kin + E_pot + E_int``.  The virial residual
``(2 E_kin - <(r - r0).grad V> + 3 E_int) / (2 E_kin + 3 E_int)`` uses the trap's
analytic gradient.  Note ``E_int`` here is the energy term ``g N int |psi|^4 / 2``,
not ``g N int |psi|^4`` -- with the wrong factor a correct solution shows a 10%
virial residual.

Attractive ``a < 0``: the variational cloud's collapse check runs first
(:class:`~kamo.BEC_properties.variational.CollapseError`; the Gaussian threshold
is ~15% above the exact one, so N above 0.85 of it warns), and during imaginary
time a peak density grown 30x *and* a width under two grid cells raise it.

Backend: :class:`kamo.imaging._backend.ArrayBackend` (numpy, or torch on a GPU).
**Double precision is the default even on the GPU** -- a deliberate divergence
from ArrayBackend's GPU default.  The imaging propagator is a unitary map whose
fp32 error grows like ``sqrt(n) eps``; this solver is a fixed-point iteration
whose residual floors near ``eps``, so fp32 caps it around 1e-6, above the 1e-8
gate.  ``allow_single=True`` accepts that.

``V_extra`` (J; callable or array on the grid) is added to the potential: the
finite-temperature solver in :mod:`kamo.trap.finite_temperature` passes
``2 g n_thermal`` through it (with ``mean_field_feedback``) on a pinned grid, and
warm-starts each pass with ``psi0=``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.fft as sfft

import kamo.constants as kc
from kamo.BEC_properties.variational import CollapseError, GaussianVariationalCloud
from kamo.imaging._backend import ArrayBackend

from . import interactions as ia
from .cloud import ConvergenceError, TrapCloud, TrapTooShallowError
from .grid import TrapGrid, basin_mask


@dataclass
class GPStage:
    """One imaginary-time stage."""

    dtau_s: float                #: imaginary time step (s)
    steps: int                   #: steps taken
    mu_offset_J: float           #: mu - V_min at its end (J)


@dataclass
class GPResult:
    """Diagnostics of a Gross-Pitaevskii solve."""

    stages: list = field(default_factory=list)
    polish_iterations: int = 0
    residual: float = float("nan")      #: ||H psi - mu psi|| / (mu - V_min)
    virial: float = float("nan")        #: relative virial residual; NaN with V_extra
    kinetic_J: float = float("nan")     #: per atom
    potential_J: float = float("nan")   #: per atom, above V_min
    interaction_J: float = float("nan") #: per atom, g N int |psi|^4 / 2
    grid_shape: tuple = ()
    box_growths: int = 0
    edge_fraction: float = float("nan") #: face density / peak density
    backend: str = "numpy"
    precision: str = "double"
    warm_start: bool = False            #: started from a given psi0 (imaginary time skipped)
    psi: Optional[np.ndarray] = field(default=None, repr=False)  #: the normalized order parameter on the grid


def _fast_even(n: int) -> int:
    n = max(int(n), 8)
    while True:
        k = sfft.next_fast_len(n)
        if k % 2 == 0:
            return k
        n = k + 1


def _re_vdot(a, b) -> float:
    """``Re sum(conj(a) b)`` for numpy arrays or torch tensors, in double."""
    if type(a).__module__.split(".")[0] == "torch":
        return float((a.conj() * b).real.double().sum().item())
    return float(np.real(np.vdot(a, b)))


def _make_backend(spec, precision: str, allow_single: bool) -> ArrayBackend:
    bk = spec if isinstance(spec, ArrayBackend) else ArrayBackend(
        "cpu" if spec is None else str(spec), precision=precision)
    if bk.precision == "single" and not allow_single:
        raise ValueError("single precision caps the GP residual near 1e-6, above the "
                         "1e-8 gate; pass allow_single=True to accept that.")
    return bk


class GrossPitaevskiiSolver:
    """3D Gross-Pitaevskii ground state of ``trap`` (a Trap or HarmonicTrap).

    Parameters
    ----------
    trap : Trap or HarmonicTrap
    a_scattering : float, optional
        Scattering length (m); looked up from the trap's state and field if omitted.
        May be zero or negative.
    points_per_scale : float
        Grid points per ``min(healing length, cloud width)`` (default 4).
    box_factor : float
        Box half-width in units of ``max(4 sigma, R_TF)``.
    n_max : int
        Cap on points per axis.
    grid : TrapGrid, optional
        Use this grid instead of sizing one.
    backend, precision, allow_single
        See the module docstring.
    n_anneal, anneal_factor, dtau
        Imaginary-time stages; ``dtau`` defaults to ``hbar / max(V - V_min)``.
    stage_tol, check_every, max_stage_steps
        A stage ends when ``mu`` changes by less than ``stage_tol`` (relative)
        between checks.
    residual_tol, max_polish
        The final gate and the polish iteration budget.
    strict : bool
        Raise :class:`~kamo.trap.cloud.ConvergenceError` if the gate is missed
        (default); otherwise warn.

    ``solve(N, V_extra, psi0=...)`` warm-starts from a previous order parameter on
    the pinned ``grid`` (an array, or a TrapCloud from this solver): imaginary time
    is skipped and the polish runs from ``psi0``, 2.5-3x faster when the change
    since is small (a new ``V_extra`` or ``N`` in a Hartree-Fock loop).  If the polish
    misses the gate from there, the solve restarts cold once and warns.
    """

    def __init__(self, trap, *, a_scattering: Optional[float] = None,
                 points_per_scale: float = 4.0, box_factor: float = 1.6, n_max: int = 320,
                 grid: Optional[TrapGrid] = None, backend=None, precision: str = "double",
                 allow_single: bool = False, n_anneal: int = 1, anneal_factor: float = 4.0,
                 dtau: Optional[float] = None, stage_tol: float = 1e-5, check_every: int = 20,
                 max_stage_steps: int = 20000, residual_tol: float = 1e-8,
                 max_polish: int = 3000, max_growth: int = 3, strict: bool = True):
        self.trap = trap
        self.a_scattering = a_scattering
        self.points_per_scale = float(points_per_scale)
        self.box_factor = float(box_factor)
        self.n_max = int(n_max)
        self.grid = grid
        self.backend = _make_backend(backend, precision, allow_single)
        self.n_anneal = int(n_anneal)
        self.anneal_factor = float(anneal_factor)
        self.dtau = dtau
        self.stage_tol = float(stage_tol)
        self.check_every = int(check_every)
        self.max_stage_steps = int(max_stage_steps)
        self.residual_tol = float(residual_tol)
        self.max_polish = int(max_polish)
        self.max_growth = int(max_growth)
        self.strict = bool(strict)

    # -------------------------------------------------------------- solve
    def solve(self, N: float, V_extra=None, *, psi0=None) -> TrapCloud:
        N = float(N)
        if not N > 0:
            raise ValueError(f"N must be positive; got {N}")
        if psi0 is not None:
            if self.grid is None:
                raise ValueError("psi0 needs a pinned grid: pass grid= to the solver")
            if hasattr(psi0, "density_grid"):                 # a TrapCloud
                info0 = getattr(psi0, "info", None)
                start = getattr(info0, "psi", None)
                if start is None:
                    start = np.sqrt(np.clip(psi0.density_grid, 0.0, None) / psi0.N)
                psi0 = start
            psi0 = np.asarray(psi0)
            if psi0.shape != self.grid.shape:
                raise ValueError(f"psi0 shape {psi0.shape} != grid {self.grid.shape}")
        trap = self.trap
        m = trap.mass
        a = ia.trap_scattering_length(trap, self.a_scattering)
        mn = trap.minimum()
        if not mn.converged:
            raise TrapTooShallowError("the trap has no minimum to hold a cloud")
        tf = trap.trap_frequencies()
        if not tf.is_bound:
            raise TrapTooShallowError("the trap has a negative curvature at its minimum")
        r0, V_min = mn.position, mn.potential_J
        V_esc = V_min + float(trap.trap_depth_J())

        var = GaussianVariationalCloud(N, tf.omega, a, mass=m)
        if var.collapsed:
            raise CollapseError(
                f"no Gaussian minimum: N = {N:.0f} at a = {a / kc.a0:+.2f} a0 collapses "
                f"(Gaussian critical N ~ {var.critical_atom_number():.0f}; the exact GP "
                "threshold is ~15% lower).")
        if a < 0:
            n_crit = var.critical_atom_number()
            if N > 0.85 * n_crit:
                warnings.warn(f"N = {N:.0f} is above 0.85 of the Gaussian critical number "
                              f"{n_crit:.0f}; the exact threshold is ~15% lower.",
                              UserWarning, stacklevel=2)
        w_k = var.widths                                   # 1/e density radii, principal axes

        if self.grid is not None:
            grid = self.grid
            if psi0 is not None:
                try:
                    psi, info = self._run(grid, N, a, m, r0, V_min, V_esc, tf, w_k, V_extra,
                                          psi0=psi0)
                    ok = info.residual < self.residual_tol
                except ConvergenceError:
                    ok = False
                if not ok:
                    warnings.warn("the warm start did not reach residual_tol; restarting from "
                                  "the variational Gaussian.", UserWarning, stacklevel=2)
                    psi0 = None
            if psi0 is None:
                psi, info = self._run(grid, N, a, m, r0, V_min, V_esc, tf, w_k, V_extra)
            if info.edge_fraction >= 1e-8:
                warnings.warn(f"the condensate reaches the face of the given grid "
                              f"({info.edge_fraction:.1e} of its peak).", UserWarning,
                              stacklevel=2)
        else:
            dx, half = self._spacing_and_half_widths(var, a, tf)
            for growth in range(self.max_growth + 1):
                grid = self._grid_from(r0, half, dx)
                psi, info = self._run(grid, N, a, m, r0, V_min, V_esc, tf, w_k, V_extra)
                info.box_growths = growth
                if info.edge_fraction < 1e-8:
                    break
                half = half * 1.3
            else:
                warnings.warn(f"the condensate still reaches the box face "
                              f"({info.edge_fraction:.1e} of its peak); raise box_factor.",
                              UserWarning, stacklevel=2)

        rho = np.abs(psi) ** 2
        mu = info.kinetic_J + info.potential_J + 2.0 * info.interaction_J
        E = info.kinetic_J + info.potential_J + info.interaction_J
        return TrapCloud(grid, N * rho, N, trap, mode="gp", chemical_potential_J=V_min + mu,
                         V_min_J=V_min, energy_per_atom_J=V_min + E, a_scattering=a,
                         info=info)

    # -------------------------------------------------------- grid sizing
    def _spacing_and_half_widths(self, var, a, tf):
        """``(dx, half)`` per lab axis (m): ``min(xi, sigma) / points_per_scale`` and
        ``box_factor * max(4 sigma, R_TF)`` from the variational cloud."""
        w_k = var.widths
        sig_lab = np.sqrt((tf.axes ** 2).T @ (w_k / np.sqrt(2.0)) ** 2)
        if a > 0:
            xi = 1.0 / np.sqrt(8.0 * np.pi * var.peak_density * a)
            R_lab = np.sqrt((tf.axes ** 2).T @ var.tf_radii ** 2)
        else:
            xi, R_lab = np.inf, np.zeros(3)
        dx = np.minimum(xi, sig_lab) / self.points_per_scale
        half = self.box_factor * np.maximum(4.0 * sig_lab, R_lab)
        return dx, half

    def _grid_from(self, r0, half, dx) -> TrapGrid:
        n = []
        for h, d in zip(half, dx):
            k = _fast_even(np.ceil(2.0 * h / d))
            if k > self.n_max:
                warnings.warn(f"grid capped at n_max = {self.n_max} points per axis "
                              f"(wanted {k}).", UserWarning, stacklevel=3)
                k = self.n_max - self.n_max % 2
            n.append(k)
        return TrapGrid.around(r0, half, n)

    def grid_for(self, N: float, *, half_widths=None) -> TrapGrid:
        """The grid ``solve(N)`` would start from (before any box growth): the GP
        spacing rule, on ``half_widths`` (m, lab axes) if given -- e.g. a box that
        must also hold a thermal cloud -- else the GP's own box."""
        trap = self.trap
        a = ia.trap_scattering_length(trap, self.a_scattering)
        mn, tf = trap.minimum(), trap.trap_frequencies()
        if not (mn.converged and tf.is_bound):
            raise TrapTooShallowError("the trap has no bound minimum to size a grid from")
        var = GaussianVariationalCloud(float(N), tf.omega, a, mass=trap.mass)
        if var.collapsed:
            raise CollapseError(f"N = {N:.0f} at a = {a / kc.a0:+.2f} a0 collapses "
                                "(no Gaussian minimum to size a grid from)")
        dx, half = self._spacing_and_half_widths(var, a, tf)
        if half_widths is not None:
            half = np.broadcast_to(np.asarray(half_widths, dtype=float), (3,))
        return self._grid_from(mn.position, half, dx)

    # ----------------------------------------------------------- the core
    def _run(self, grid: TrapGrid, N, a, m, r0, V_min, V_esc, tf, w_k, V_extra, psi0=None):
        bk, hbar = self.backend, kc.hbar
        V = np.broadcast_to(np.asarray(self.trap.potential_J(grid.X, grid.Y, grid.Z),
                                       dtype=float), grid.shape)
        if V_extra is not None:
            extra = V_extra(grid.X, grid.Y, grid.Z) if callable(V_extra) else V_extra
            V = V + np.broadcast_to(np.asarray(extra, dtype=float), grid.shape)
        try:
            basin = basin_mask(V, grid, r0, V_esc)
        except ValueError as err:
            raise ConvergenceError(str(err)) from None
        plateau = V_esc if np.isfinite(V_esc) else float(np.max(V))
        Vs = np.where(basin, V, plateau) - V_min                 # >= 0 up to round-off
        E_scale = max(float(np.max(Vs)), hbar * float(np.max(tf.omega)))
        gN = ia.coupling_g(a, m) * N
        K = hbar ** 2 * grid.k_squared() / (2.0 * m)
        dV = grid.dV

        D = (grid.X - r0[0], grid.Y - r0[1], grid.Z - r0[2])
        warm = psi0 is not None
        if not warm:
            p = [tf.axes[k, 0] * D[0] + tf.axes[k, 1] * D[1] + tf.axes[k, 2] * D[2]
                 for k in range(3)]
            psi0 = np.broadcast_to(np.exp(-sum(p[k] ** 2 / (2.0 * w_k[k] ** 2) for k in range(3))),
                                   grid.shape)
        psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * dV)

        psi, Vd, Kd = bk.complex(psi0), bk.real(Vs), bk.real(K)

        def normalize(f):
            return f / bk.sqrt(bk.dsum(bk.abs2(f)) * dV)

        def dot(f, h):
            return _re_vdot(f, h) * dV

        def H_apply(f, Veff):
            return bk.ifftn(Kd * bk.fftn(f)) + Veff * f

        info = GPResult(grid_shape=grid.shape, backend=bk.name, precision=bk.precision,
                        warm_start=warm)

        # ---- imaginary time: get close (skipped from a warm start: the polish
        # converges from anywhere nearby, and the stage's mu-change test does not
        # recognize an already-converged state, so running it is a 1.4x pessimization)
        dtau = hbar / E_scale if self.dtau is None else float(self.dtau)
        peak0 = bk.fmax(bk.abs2(psi))
        for _ in range(self.n_anneal if not warm else 0):
            kin = bk.exp(-Kd * (dtau / hbar))
            half_step = dtau / (2.0 * hbar)
            mu_prev, steps, mu = None, 0, float("nan")
            while steps < self.max_stage_steps:
                for _ in range(self.check_every):
                    psi = psi * bk.exp(-(Vd + gN * bk.abs2(psi)) * half_step)
                    psi = bk.ifftn(bk.fftn(psi) * kin)
                    psi = psi * bk.exp(-(Vd + gN * bk.abs2(psi)) * half_step)
                    psi = normalize(psi)
                steps += self.check_every
                mu = dot(psi, H_apply(psi, Vd + gN * bk.abs2(psi)))
                if a < 0 and bk.fmax(bk.abs2(psi)) > 30.0 * peak0:
                    _, _, var_now = grid.moments(bk.numpy(bk.abs2(psi)))
                    if np.min(np.sqrt(var_now) / grid.d) < 2.0:
                        raise CollapseError(f"imaginary-time collapse after {steps} steps: the "
                                            "cloud narrowed below two grid cells.")
                if mu_prev is not None and abs(mu - mu_prev) < self.stage_tol * abs(mu):
                    break
                mu_prev = mu
            info.stages.append(GPStage(dtau, steps, mu))
            dtau /= self.anneal_factor

        # ---- polish: preconditioned conjugate gradient on the unit sphere
        # Minimizes E[psi] = <psi|K + V|psi> + (gN/2) int |psi|^4 with an EXACT line
        # search: along cos(t) psi + sin(t) d (both unit, orthogonal) the energy is a
        # closed-form quartic in (cos t, sin t), so a step costs three FFT pairs and
        # can only lower the energy.  Its fixed point is the GP equation itself.
        from scipy.optimize import minimize_scalar

        def kin(f):
            return bk.ifftn(Kd * bk.fftn(f))

        residual, it = np.inf, 0
        d_prev = r_prev = z_prev = None
        Kpsi = kin(psi)
        for it in range(self.max_polish + 1):
            rho = bk.abs2(psi)
            Hpsi = Kpsi + (Vd + gN * rho) * psi
            mu = dot(psi, Hpsi)
            r = Hpsi - mu * psi
            residual = float(np.sqrt(dot(r, r))) / abs(mu)
            if residual < self.residual_tol or it == self.max_polish:
                break
            z = bk.ifftn(bk.fftn(r) / (Kd + mu))          # preconditioned gradient
            z = z - dot(psi, z) * psi                       # tangent to the sphere
            if d_prev is None or it % 25 == 0:
                d = -z
            else:
                beta = max(0.0, (dot(z, r) - dot(z, r_prev)) / dot(z_prev, r_prev))
                d = -z + beta * d_prev
                d = d - dot(psi, d) * psi
            slope = dot(d, r)
            if slope >= 0.0:                                # not a descent direction: restart
                d = -z
            nd = float(np.sqrt(dot(d, d)))
            dn = d / nd
            Kdn = kin(dn)
            ek_pp, ek_dd, ek_pd = dot(psi, Kpsi), dot(dn, Kdn), dot(psi, Kdn)
            ev_pp, ev_dd, ev_pd = dot(psi, Vd * psi), dot(dn, Vd * dn), dot(psi, Vd * dn)
            pp, dd = psi * rho, dn * bk.abs2(dn)
            q0, q1, q2 = dot(pp, psi), dot(pp, dn), dot(psi * bk.abs2(dn), psi)
            q3, q4 = dot(dd, psi), dot(dd, dn)

            def energy(t):
                c, sn = np.cos(t), np.sin(t)
                quad = (c * c * (ek_pp + ev_pp) + sn * sn * (ek_dd + ev_dd)
                        + 2.0 * c * sn * (ek_pd + ev_pd))
                quart = (c ** 4 * q0 + 4.0 * c ** 3 * sn * q1 + 6.0 * c * c * sn * sn * q2
                         + 4.0 * c * sn ** 3 * q3 + sn ** 4 * q4)
                return quad + 0.5 * gN * quart

            # E'(0), E''(0) from the same inner products: near convergence the step is
            # the Newton one (the energy itself is flat to round-off there, so a
            # comparison of energies would stall at a residual ~ sqrt(eps)).
            dE0 = 2.0 * (ek_pd + ev_pd) + 2.0 * gN * q1
            d2E0 = 2.0 * (ek_dd + ev_dd - ek_pp - ev_pp) + gN * (6.0 * q2 - 2.0 * q0)
            t_newton = -dE0 / d2E0 if d2E0 > 0 else np.inf
            if residual < 1e-3 and 0.0 < t_newton < 0.25 * np.pi:
                t = float(t_newton)
            else:
                res = minimize_scalar(energy, bounds=(0.0, 0.5 * np.pi), method="bounded",
                                      options=dict(xatol=1e-12))
                t = float(res.x) if res.fun < energy(0.0) else 0.0
            if t <= 0.0:
                break                                        # no downhill step left
            c, sn = np.cos(t), np.sin(t)
            psi = c * psi + sn * dn
            Kpsi = c * Kpsi + sn * Kdn                       # linear: no extra FFT
            d_prev, r_prev, z_prev = d, r, z
        info.polish_iterations, info.residual = it, residual
        if residual >= self.residual_tol:
            msg = (f"GP residual {residual:.2e} above residual_tol {self.residual_tol:.0e} "
                   f"after {it} polish iterations.")
            if self.strict:
                raise ConvergenceError(msg + "  Raise max_polish or points_per_scale.")
            warnings.warn(msg, UserWarning, stacklevel=3)

        psi_h = np.asarray(bk.numpy(psi), dtype=np.complex128)
        psi_h = psi_h / np.sqrt(np.sum(np.abs(psi_h) ** 2) * dV)
        rho = np.abs(psi_h) ** 2
        fk = sfft.fftn(psi_h, workers=-1)
        info.kinetic_J = float(np.sum(K * np.abs(fk) ** 2) / np.sum(np.abs(fk) ** 2))
        info.potential_J = float(np.sum(Vs * rho) * dV)
        info.interaction_J = float(0.5 * gN * np.sum(rho * rho) * dV)
        info.edge_fraction = grid.face_max(rho) / float(rho.max())
        info.psi = psi_h
        if V_extra is None and hasattr(self.trap, "gradient"):
            gx, gy, gz = self.trap.gradient(grid.X, grid.Y, grid.Z)
            r_grad = float(np.sum(rho * np.where(basin, D[0] * gx + D[1] * gy + D[2] * gz, 0.0))
                           * dV)
            two_T_three_I = 2.0 * info.kinetic_J + 3.0 * info.interaction_J
            info.virial = (two_T_three_I - r_grad) / two_T_three_I
        return psi_h, info
