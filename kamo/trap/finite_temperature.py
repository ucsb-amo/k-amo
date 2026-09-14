"""Finite temperature: a condensate in equilibrium with a thermal cloud on the real trap.

``solve(trap, N, "gp", T_K=30e-9)`` routes here.  The condensate is the Gross-Pitaevskii
ground state of ``N_0`` atoms (the existing solver, warm-started between passes); the
thermal cloud occupies the Hartree-Fock single-particle states of
``V_eff = V + 2 g n_0`` (``+ 2 g n_th`` with ``mean_field_feedback``) with Bose factors at
one chemical potential, and ``N_0 + N_th = N`` fixes everything.

Why not the textbook semiclassical cloud
----------------------------------------
At the K-team operating point (3 um / 1 kHz tweezer, N = 500) the semiclassical
local-density thermal cloud ``g_{3/2}(z_loc) / lambda^3`` is wrong by a factor **1.7 to
6.5** in ``N_th`` across the whole lab window (measured 2026-09-13 against the discrete
sum on the real potential, both truncated at the escape energy: 6.5x at 30 nK, 3.6x at
50, 2.2x at 100, 1.8x at 170).  The cause is not the anharmonicity, which the LDA handles
exactly, but the density of states at the bottom of the well: only 3 single-particle
states exist below the condensate's own ``mu - V_min = 1081 Hz``, which is 95% zero-point
energy, and ``k T / hbar w_radial`` is 1-3.  The LDA places the bottom of the band at
``min V_eff`` instead of at the true ``E_0``, 858 Hz too low.

The hybrid spectrum
-------------------
Discrete levels below ``E_cut = E_0 + e_cut_hbar_omega * hbar w_max`` (default 2, ~70
levels here) and a truncated semiclassical (Weyl) tail above it:

* **Discrete part -- the calibrated separable product basis.**  Along each principal
  axis ``k`` the 1D sinc-DVR ladder ``eps_k(n) = E^k_n - E^k_0`` and eigenfunctions
  ``chi^k_n(u)`` (:meth:`~kamo.trap.noninteracting.NonInteractingSolver.axis_spectra`),
  combined as ``E(n_1, n_2, n_3) = E_0 + sum_k c_k eps_k(n_k)`` and
  ``psi = prod_k chi^k_{n_k}(u_k)``, ``u_k = e_k . (r - r_0)``.  ``E_0`` is the exact 3D
  ground energy from the 4-state solve :class:`NonInteractingSolver` already performs, and
  ``c_k`` rescales each ladder by the ratio of the exact 3D first gap along that axis to
  the separable one, for every axis those 4 states reach.  That factor is the
  Born-Oppenheimer depth renormalization of the soft axis (the stiff axes' zero point
  falls as the beam expands): predicted ``sqrt(1 - hbar (w_y + w_z) / 2 V_0) = 0.9406``,
  fitted 0.9377 for the tweezer, and it takes the lowest 14 levels from +10..+72 Hz high
  to within 6 Hz.  The raw basis misses ~23% of the states near the lip (cross
  anharmonicity a product cannot see), which is what the tail is for.  Validated against
  64 exact 3D eigenstates: ``N_th`` agrees to 0.05% (30 nK), 0.3% (50 nK), 0.9% (100 nK).
  The states are never materialized: the outer loop touches only the energy list, and
  the density is accumulated once at the converged chemical potential.
* **Tail -- the momentum integral** in ``s = p / sqrt(2 m k T)``::

      n_tail(r) = 4 / (sqrt(pi) lambda^3) * int_{s_lo}^{s_hi} s^2 ds / (exp(s^2 + d) - 1)
      s_lo^2 = (E_cut - V(r)) / kT,   s_hi^2 = (V_esc - V(r)) / kT,   d = (V_eff(r) - mu) / kT

  32-node Gauss-Legendre, machine precision, one kernel (:func:`semiclassical_density`)
  for the hybrid tail, the opt-in ``"lda"`` model (``s_lo = 0``) and ``"boltzmann"``
  (``exp`` for ``expm1``).  The seam is drawn in the *bare* spectrum (``s_lo`` from
  ``V``) while the weight uses the HF energy (``d`` from ``V_eff``): that is what makes
  the discrete count and the Weyl count add up, and it costs 0.2-0.3% of ``N_th`` against
  the alternative -- deliberate, do not "fix".  ``E_cut`` must clear ``mu`` by at least
  half an ``hbar w_max``: the tail integral diverges logarithmically as ``E_cut -> mu``.
* **Truncation at the escape energy is part of the state, not a correction**: the
  ideal-gas ``T_c`` of this trap is ~0.9 of its depth in temperature, so ``eta = U / kT``
  is 2-7 wherever a condensate is worth talking about, and a distribution cut at ``V_esc``
  holds 4x fewer atoms than an untruncated one at 170 nK.  ``s_hi = 0`` outside the basin
  (the connected region below ``V_esc``, cut at ``(1 - basin_epsilon)`` of the depth --
  at exactly the saddle energy the mask leaks through the saddle and the answer is
  infinite).  ``V_esc`` is frozen at the bare-trap saddle: the mean field there is
  ``< 1e-4 Hz``.

The chemical potential and the outer loop
-----------------------------------------
``mu(N_0) = mu_GP(N_0) - kT ln(1 + 1/N_0)`` -- the condensate is one Bose level at the GP
eigenvalue, occupied by ``N_0`` atoms.  It reduces to the exact finite-N ideal relation
as ``g -> 0`` and to ``mu_GP`` for large ``N_0``, and it is continuous through the
crossover, so there is **no separate above-T_c branch**:
``R(N_0) = N_0 + sum_{n >= 1} f_n(mu(N_0)) + N_tail(mu(N_0)) - N`` is strictly monotone
with ``R(0+) = -N`` and ``R(N) > 0`` for every T, and one ``brentq`` on ``N_0`` finds the
root.  The thermal sum runs over ``n >= 1``: the lowest HF level is the condensate mode.
For ``a > 0`` every HF level lies at or above ``mu_GP`` by the min-max principle
(``2 g n_0 >= g n_0``), so the Bose factors are positive by construction; a level at or
below ``mu`` (``a < 0``, or the first-order shifts at the bracket end) raises.

Each pass solves the condensate at the current ``N_0`` (warm-started, imaginary time
skipped), shifts the levels by ``<n| 2 g n_0 |n>`` (a contraction, not a state), rebuilds
the tail on ``V_eff``, and re-roots ``R``.  ``mu_GP`` moves 31 Hz over 200 atoms here, so
the map contracts by 0.02-0.08 per pass and converges undamped in 2-3 passes; damping
latches on (and halves again) if a step grows.  Below ``n_condensate_min`` atoms the
condensate is the ideal ground state with its first-order mean-field energy (which keeps
``R`` continuous across the switch), and above the crossover the root simply lands at a
small ``N_0``.  One-way coupling is the default: ``2 g n_th / kT`` is 0.015 at the
operating point, and feeding it back costs a GP solve per pass for a < 2% change in ``mu``.

Models, in ``thermal=``
-----------------------
``"hybrid"`` (default) as above; ``"lda"``: the tail kernel over the whole spectrum with
the fugacity measured from the true band bottom ``E_0^eff = mu_GP + g <n_0>`` (a rigorous
upper bound on the HF ground energy) rather than from ``min V_eff`` -- pole-free, no clip,
numerically the textbook saturated LDA (0.994-0.997 of it) and hence still 1.6-6.5x low
here; ``"boltzmann"``: the classical gas, no condensate, ``mu`` from ``N``.  Both opt-in
models warn and need no spectrum.  ``mode="thomas-fermi"`` is refused at T > 0 (the TF
radius is already 2x wrong at this operating point); the bimodal fit model the lab uses
is :class:`~kamo.BEC_properties.thermal.IdealHarmonicBoseGas`.

What is reported
----------------
:class:`FiniteTemperatureResult` carries the loop history, ``eta``, the whole-cloud
truncation fraction, the discrete/Weyl seam ratio, the calibration factors, the minimum
level margin ``(E_n - mu) / kT``, the containment gates and ``trustworthy``.
:class:`ModelValidityWarning` fires (never raises) when ``eta < 5`` or the truncation
fraction exceeds 0.3, when the basis quality is poor, or for the opt-in models.

Every default in the knob table of :class:`FiniteTemperatureSolver` was measured on the
lab tweezer on 2026-09-13; the K-team ground-truth values are in
``tests/test_finite_temperature.py``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import kamo.constants as kc
from kamo.BEC_properties.thermal import (critical_temperature_K, finite_size_shift,
                                         ideal_harmonic_condensate, mean_field_shift)

from . import interactions as ia
from .cloud import ConvergenceError, TrapCloud, TrapTooShallowError
from .grid import TrapGrid, basin_mask, touched_axes
from .gross_pitaevskii import _fast_even
from .trap import HarmonicTrap

THERMAL_MODELS = ("hybrid", "lda", "boltzmann")
CONDENSATE_MODELS = ("gp", "noninteracting")
BASIN_EPSILON = 1e-3          #: basin cut at (1 - eps) x depth: at eps = 0 the mask leaks through the saddle
BOSE_CUTOFF_KT = 40.0         #: momentum window cut at s^2 = 40 (+ max(0, -d)): 1 / expm1(40) = 4e-18
N_CONDENSATE_MIN = 10.0       #: below this N_0 the condensate is the ideal ground state, not a GP solve
N_LEVELS_MAX = 200_000        #: cap on enumerated product levels (Weyl pre-check)
TAIL_CACHE_MAX_BYTES = 256e6  #: above this the tail cache keeps only s_lo / span and recomputes
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)   # 32 nodes: 1e-11 at d = 2 over s in [0, 12] (2026-09-13)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W


class ModelValidityWarning(UserWarning):
    """The solve converged but the model is being used outside its comfort zone."""


# ----------------------------------------------------------------- kernel

def _gl_sum(s_lo, span, d, statistics: str):
    """``sum_q w_q span s_q^2 f(s_q^2 + d)`` on the 32-node rule with ``s_q = s_lo + span x_q``,
    ``f = 1/expm1`` (Bose) or ``exp(-.)`` (Boltzmann); 0 where ``span <= 0``.  A negative
    value means the Bose pole ``s^2 + d <= 0`` sits inside the window: the caller must
    keep ``E_cut`` (or ``mu``) clear of it."""
    if statistics not in ("bose", "boltzmann"):
        raise ValueError("statistics must be 'bose' or 'boltzmann'")
    acc = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for x, w in zip(_GL_X, _GL_W):
            s = s_lo + span * x
            arg = s * s + d
            f = s * s / np.expm1(arg) if statistics == "bose" else s * s * np.exp(-arg)
            acc = acc + w * span * np.where((span > 0) & np.isfinite(f), f, 0.0)
    return acc


def _window(V, kT, E_cut, V_escape):
    """``(s_lo, span_to_escape)``: the bare momentum window per node; ``span`` is inf
    without an escape energy (the ``BOSE_CUTOFF_KT`` cap is applied against ``d``)."""
    V = np.asarray(V, dtype=float)
    s_lo = np.zeros_like(V) if E_cut is None else np.sqrt(np.clip((float(E_cut) - V) / kT, 0.0, None))
    if np.isfinite(V_escape):
        s_hi = np.sqrt(np.clip((float(V_escape) - V) / kT, 0.0, None))
    else:
        s_hi = np.full_like(V, np.inf)
    return s_lo, np.clip(s_hi - s_lo, 0.0, None)


def _capped_span(s_lo, span, d):
    """The span with ``s_hi <= sqrt(BOSE_CUTOFF_KT + max(0, -d))``, and whether it bound."""
    cap = np.sqrt(BOSE_CUTOFF_KT + np.clip(-d, 0.0, None))
    capped = np.minimum(span, np.clip(cap - s_lo, 0.0, None))
    return capped, capped < span


def semiclassical_density(V, V_eff, mu: float, T_K: float, mass: float, *, E_cut=None,
                          V_escape: float = np.inf, statistics: str = "bose"):
    """Semiclassical thermal density (1/m^3) at each node.

    ``V`` the bare potential (J) sets the momentum window: the tail starts at the
    bare energy ``E_cut`` (``None``: at zero momentum, the plain local-density
    model) and stops at ``V_escape`` (the truncation).  ``V_eff`` (J) sets the
    Bose weight ``d = (V_eff - mu) / kT``.  ``statistics`` ``"bose"`` or
    ``"boltzmann"``.  Broadcasts; zero wherever the window is empty.
    """
    kT = kc.kB * float(T_K)
    lam = kc.h / np.sqrt(2.0 * np.pi * float(mass) * kT)
    V, V_eff = np.broadcast_arrays(np.asarray(V, dtype=float), np.asarray(V_eff, dtype=float))
    d = (V_eff - float(mu)) / kT
    s_lo, span = _window(V, kT, E_cut, V_escape)
    span, _ = _capped_span(s_lo, span, d)
    return 4.0 / (np.sqrt(np.pi) * lam ** 3) * _gl_sum(s_lo, span, d, statistics)


class _TailCache:
    """``N_tail(mu)`` and ``n_tail(mu)`` on fixed nodes for fixed ``V``, ``V_eff``, ``E_cut``
    and ``V_escape``.  With a finite escape energy and under ``max_bytes`` the per-node,
    per-quadrature-node arrays are built once and a call is one ``expm1`` pass; otherwise
    (an unbounded window, or too many nodes) each call runs the 32-step quadrature."""

    def __init__(self, V, V_eff, T_K, mass, dV, *, E_cut, V_escape, statistics="bose",
                 max_bytes=TAIL_CACHE_MAX_BYTES):
        self.kT = kc.kB * float(T_K)
        self.lam = kc.h / np.sqrt(2.0 * np.pi * float(mass) * self.kT)
        self.dV, self.statistics = float(dV), statistics
        self.V, self.V_eff = np.asarray(V, dtype=float), np.asarray(V_eff, dtype=float)
        self.E_cut, self.V_escape = E_cut, float(V_escape)
        self.pref = 4.0 / (np.sqrt(np.pi) * self.lam ** 3)
        self._s_lo, self._span = _window(self.V, self.kT, E_cut, self.V_escape)
        # the cached branch integrates the uncapped window; that equals the capped one to
        # e^-40 whenever every span is under sqrt(BOSE_CUTOFF_KT), so cache only then
        self._cached = (bool(np.all(self._span < np.sqrt(BOSE_CUTOFF_KT)))
                        and 3 * 32 * self.V.size * 8 <= max_bytes)
        if self._cached:
            s = self._s_lo[None, :] + self._span[None, :] * _GL_X[:, None]
            self._A = s * s + self.V_eff[None, :] / self.kT
            self._W = _GL_W[:, None] * self._span[None, :] * s * s

    def _sum(self, mu, s_lo=None, span=None):
        """The quadrature sum per node at ``mu``."""
        s_lo = self._s_lo if s_lo is None else s_lo
        span = self._span if span is None else span
        d = (self.V_eff - mu) / self.kT
        if self._cached and span is self._span:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                arg = self._A - mu / self.kT
                f = self._W / np.expm1(arg) if self.statistics == "bose" else self._W * np.exp(-arg)
            return np.where(np.isfinite(f), f, 0.0).sum(axis=0)
        capped, _ = _capped_span(s_lo, span, d)
        return _gl_sum(s_lo, capped, d, self.statistics)

    def density(self, mu):
        """``n_tail`` on the cached nodes (1/m^3)."""
        return self.pref * self._sum(mu)

    def total(self, mu) -> float:
        """``N_tail(mu)``."""
        return float(np.sum(self._sum(mu), dtype=np.float64)) * self.pref * self.dV

    def total_untruncated(self, mu) -> float:
        """``N_tail`` on the same nodes with the escape truncation lifted (the
        ``BOSE_CUTOFF_KT`` cap only) -- a lower bound on the untruncated cloud, since
        nodes beyond the escape contour are not part of the basin."""
        s_lo, span = _window(self.V, self.kT, self.E_cut, np.inf)
        return float(np.sum(self._sum(mu, s_lo, span), dtype=np.float64)) * self.pref * self.dV

    def clipped_fraction(self, mu) -> float:
        """Fraction of nodes where the ``BOSE_CUTOFF_KT`` cap bound at this ``mu``
        (0 for a truncated trap; 1 for a HarmonicTrap)."""
        return float(_capped_span(self._s_lo, self._span, (self.V_eff - mu) / self.kT)[1].mean())


def weyl_count(E: float, V, dV: float, mass: float) -> float:
    """Semiclassical number of single-particle states below ``E`` on the nodes ``V``:
    ``(2m)^{3/2} / (6 pi^2 hbar^3) integral (E - V)^{3/2} dV``."""
    ex = np.clip(float(E) - np.asarray(V, dtype=float), 0.0, None)
    return float((2.0 * mass) ** 1.5 / (6.0 * np.pi ** 2 * kc.hbar ** 3)
                 * np.sum(ex ** 1.5, dtype=np.float64) * dV)


# -------------------------------------------------------- the product basis

def _ho_functions(xi, n_max: int) -> np.ndarray:
    """Normalized 1D oscillator eigenfunctions ``phi_n(xi)``, ``n = 0..n_max``, in the
    dimensionless coordinate ``xi = u sqrt(m w / hbar)`` (so ``int phi^2 dxi = 1``); the
    stable three-term recurrence."""
    xi = np.asarray(xi, dtype=float)
    out = np.empty((n_max + 1,) + xi.shape)
    out[0] = np.pi ** -0.25 * np.exp(-0.5 * xi * xi)
    if n_max >= 1:
        out[1] = np.sqrt(2.0) * xi * out[0]
    for n in range(1, n_max):
        out[n + 1] = np.sqrt(2.0 / (n + 1)) * xi * out[n] - np.sqrt(n / (n + 1)) * out[n - 1]
    return out


class _Ladder:
    """One principal axis: excitation energies and ``chi_n(u)`` for ``n <= n_max``."""

    def __init__(self, eps, evaluate, n_bound: int, omega: float):
        self.eps = np.asarray(eps, dtype=float)     #: (n_bound,), eps[0] = 0
        self._evaluate = evaluate                    #: (n_max, u) -> (n_max + 1, *u.shape)
        self.n_bound = int(n_bound)
        self.omega = float(omega)

    def functions(self, n_max: int, u) -> np.ndarray:
        return self._evaluate(int(n_max), np.asarray(u, dtype=float))


def _dvr_ladder(spec) -> _Ladder:
    E = spec.bound_energies
    u = spec.u - spec.u_min
    psi = spec.psi                                   # (n_grid, n_bound), int psi^2 du = 1

    def evaluate(n_max, uu):
        out = np.zeros((n_max + 1,) + uu.shape)
        inside = (uu >= u[0]) & (uu <= u[-1])
        for n in range(min(n_max, psi.shape[1] - 1) + 1):
            f = CubicSpline(u, psi[:, n], extrapolate=False)(uu)
            out[n] = np.where(inside, np.nan_to_num(f, nan=0.0), 0.0)
        return out

    return _Ladder(E - E[0], evaluate, E.size, spec.omega)


def _ho_ladder(omega: float, mass: float, n_max: int) -> _Ladder:
    scale = np.sqrt(mass * omega / kc.hbar)

    def evaluate(nm, uu):
        return _ho_functions(uu * scale, nm) * np.sqrt(scale)

    return _Ladder(kc.hbar * omega * np.arange(n_max + 1), evaluate, n_max + 1, omega)


@dataclass
class Calibration:
    """How the separable ladders were anchored and rescaled."""

    factors: np.ndarray            #: (3,) c_k per principal axis
    matched_states: np.ndarray     #: (3,) index of the 3D state that calibrated axis k, -1 if none
    overlaps: np.ndarray           #: (3,) |<trial|psi_m>|^2 of that match, NaN if none
    defaulted_axes: dict           #: axis -> why c_k = 1
    bo_prediction: float           #: sqrt(1 - hbar sum_stiff w / 2 V_0) for the softest axis; NaN if unknown
    E_0_separable_J: float         #: the raw separable ground energy (J), for the anchor shift
    n_states_3d: int


class ProductSpectrum:
    """The calibrated separable product basis of a trap (see the module docstring).

    Built with :meth:`from_trap` (1D sinc-DVR ladders + the 4-state 3D solve) or
    :meth:`harmonic` (the analytic oscillator ladder).  ``levels(E_max)`` enumerates
    ``E = E_0 + sum_k c_k eps_k(n_k) < E_max``; ``density`` and ``diagonal_expectation``
    work on a :class:`~kamo.trap.grid.TrapGrid` through the principal coordinates
    ``u_k = e_k . (r - r_0)`` and never store a 3D state.  Each 1D factor is
    renormalized on the grid (lab-aligned axes) or each product on it (rotated axes),
    since the condensate's spacing under-resolves the top levels by a few 1e-3.
    """

    def __init__(self, E_0: float, ladders, axes, r0, mass: float, *, harmonic: bool,
                 calibration: Optional[Calibration] = None, separability_index: float = np.nan):
        self.E_0 = float(E_0)
        self.ladders = tuple(ladders)
        self.axes = np.asarray(axes, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)
        self.mass = float(mass)
        self.harmonic = bool(harmonic)
        self.calibration = calibration
        self.separability_index = float(separability_index)
        self.c = np.ones(3) if calibration is None else np.asarray(calibration.factors, dtype=float)
        self._levels_cache = {}
        self._u_cache = {}

    # -------------------------------------------------------------- levels
    def levels(self, E_max: float):
        """``(energies, quantum_numbers)`` of every product level below ``E_max`` (J),
        ascending; the ground state is index 0.  ``E_max`` must be finite (for the
        ideal harmonic gas without a seam use
        :func:`kamo.BEC_properties.thermal.ideal_harmonic_condensate`)."""
        key = float(E_max)
        if key in self._levels_cache:
            return self._levels_cache[key]
        if not np.isfinite(E_max):
            raise ValueError("levels() needs a finite E_max: the product basis is enumerated, "
                             "not summed in closed form")
        ranges = []
        for k, lad in enumerate(self.ladders):
            n_k = int(np.count_nonzero(self.E_0 + self.c[k] * lad.eps < E_max))
            if n_k == 0:
                out = (np.empty(0), np.empty((0, 3), dtype=int))
                self._levels_cache[key] = out
                return out
            ranges.append(n_k)
        if int(np.prod(ranges)) > 20 * N_LEVELS_MAX:
            raise ValueError(f"enumerating up to E_max would touch {np.prod(ranges):.2e} product "
                             f"levels (cap {N_LEVELS_MAX}); lower e_cut_hbar_omega or T_K.")
        e = [self.c[k] * self.ladders[k].eps[:ranges[k]] for k in range(3)]
        E = self.E_0 + e[0][:, None, None] + e[1][None, :, None] + e[2][None, None, :]
        mask = E < E_max
        if int(np.count_nonzero(mask)) > N_LEVELS_MAX:
            raise ValueError(f"{np.count_nonzero(mask)} product levels below E_max exceed the "
                             f"cap {N_LEVELS_MAX}; lower e_cut_hbar_omega or T_K.")
        idx = np.argwhere(mask)
        En = E[mask]
        order = np.argsort(En, kind="stable")
        out = (En[order], idx[order].astype(int))
        self._levels_cache[key] = out
        return out

    def count_below(self, E_max: float) -> int:
        return int(self.levels(E_max)[0].size)

    # ----------------------------------------------------------- geometry
    def aligned(self) -> Optional[tuple]:
        """``(lab index, sign)`` per principal axis when the axes are (signed) lab axes."""
        out = []
        for e in self.axes:
            i = int(np.argmax(np.abs(e)))
            if abs(abs(e[i]) - 1.0) > 1e-9 or np.any(np.abs(np.delete(e, i)) > 1e-9):
                return None
            out.append((i, float(np.sign(e[i]))))
        return tuple(out)

    def coordinates(self, grid: TrapGrid):
        """``u_k`` on the grid: three 1D arrays (aligned axes) or three 3D arrays."""
        key = id(grid)
        if key in self._u_cache and self._u_cache[key][0] is grid:
            return self._u_cache[key][1]
        al = self.aligned()
        lab = (grid.x - self.r0[0], grid.y - self.r0[1], grid.z - self.r0[2])
        if al is not None:
            u = tuple(s * lab[i] for i, s in al)
        else:
            D = (grid.X - self.r0[0], grid.Y - self.r0[1], grid.Z - self.r0[2])
            u = tuple(np.broadcast_to(e[0] * D[0] + e[1] * D[1] + e[2] * D[2], grid.shape).copy()
                      for e in self.axes)
        self._u_cache = {key: (grid, u)}
        return u

    def _aligned_functions(self, grid, n_maxs):
        """Per axis, ``chi_n(u_k)`` for ``n <= n_max_k`` as 1D arrays broadcastable to
        the grid, each renormalized on the grid's own spacing."""
        u, al = self.coordinates(grid), self.aligned()
        out = []
        for k, lad in enumerate(self.ladders):
            f = lad.functions(n_maxs[k], u[k])
            norm = np.sqrt(np.sum(f * f, axis=1) * grid.d[al[k][0]])
            f = f / np.where(norm > 0, norm, 1.0)[:, None]
            shape = [1, 1, 1]
            shape[al[k][0]] = -1
            out.append(f.reshape((f.shape[0],) + tuple(shape)))
        return out

    def _rotated_products(self, grid, q):
        """Generator of ``|psi_j|^2`` on the grid, one level at a time, each normalized on
        the grid (rotated axes: three 3D temporaries live at once)."""
        u = self.coordinates(grid)
        n_maxs = [int(q[:, k].max()) for k in range(3)]
        chi = [self.ladders[k].functions(n_maxs[k], u[k]) for k in range(3)]   # (n_max+1, *shape)
        for j in range(q.shape[0]):
            rho = chi[0][q[j, 0]] ** 2 * chi[1][q[j, 1]] ** 2 * chi[2][q[j, 2]] ** 2
            norm = np.sum(rho) * grid.dV
            yield rho / norm if norm > 0 else rho

    def density(self, grid: TrapGrid, E_max: float, occupations) -> np.ndarray:
        """``sum_n f_n |psi_n(r)|^2`` over the levels below ``E_max`` (``occupations``
        indexed like :meth:`levels`; levels with a positive weight are summed, the
        ground state included if its weight is positive)."""
        E, q = self.levels(E_max)
        f = np.asarray(occupations, dtype=float)
        use = np.flatnonzero(f > 0)
        out = np.zeros(grid.shape)
        if use.size == 0:
            return out
        if self.aligned() is None:
            for j, rho in zip(use, self._rotated_products(grid, q[use])):
                out += f[j] * rho
            return out
        n_maxs = [int(q[use, k].max()) for k in range(3)]
        P = self._aligned_functions(grid, n_maxs)
        for n1 in np.unique(q[use, 0]):
            sel = use[q[use, 0] == n1]
            plane = 0.0
            for j in sel:
                plane = plane + f[j] * (P[1][q[j, 1]] ** 2) * (P[2][q[j, 2]] ** 2)
            out += P[0][n1] ** 2 * plane
        return out

    def diagonal_expectation(self, grid: TrapGrid, E_max: float, W) -> np.ndarray:
        """``<n| W |n>`` for every level below ``E_max`` (``W`` on the grid, J)."""
        E, q = self.levels(E_max)
        W = np.broadcast_to(np.asarray(W, dtype=float), grid.shape)
        out = np.empty(E.size)
        if E.size == 0:
            return out
        dV = grid.dV
        if self.aligned() is None:
            for j, rho in enumerate(self._rotated_products(grid, q)):
                out[j] = float(np.sum(W * rho)) * dV
            return out
        n_maxs = [int(q[:, k].max()) for k in range(3)]
        P = self._aligned_functions(grid, n_maxs)
        perm = [i for i, _ in self.aligned()]
        Wp = np.transpose(W, perm)                       # axes ordered as (u1, u2, u3)
        P1 = [P[0][n].reshape(-1) ** 2 for n in range(n_maxs[0] + 1)]
        P2 = [P[1][n].reshape(-1) ** 2 for n in range(n_maxs[1] + 1)]
        P3 = [P[2][n].reshape(-1) ** 2 for n in range(n_maxs[2] + 1)]
        T = {}
        for j in range(E.size):
            n1, n2, n3 = q[j]
            if n1 not in T:
                T[n1] = np.tensordot(P1[n1], Wp, axes=([0], [0]))
            out[j] = float(P2[n2] @ T[n1] @ P3[n3]) * dV
        return out

    # ------------------------------------------------------------- builders
    @classmethod
    def harmonic(cls, trap: HarmonicTrap, *, n_max: int = 400) -> "ProductSpectrum":
        """The analytic ladder of a ``HarmonicTrap``: exact, no DVR, no 3D solve."""
        tf = trap.trap_frequencies()
        if not tf.is_bound:
            raise TrapTooShallowError("the harmonic trap is not bound")
        mn = trap.minimum()
        ladders = [_ho_ladder(w, trap.mass, n_max) for w in tf.omega]
        E_0 = mn.potential_J + 0.5 * kc.hbar * float(np.sum(tf.omega))
        return cls(E_0, ladders, tf.axes, mn.position, trap.mass, harmonic=True,
                   separability_index=0.0)

    @classmethod
    def from_trap(cls, trap, *, n_states_3d: int = 4, axis_n_grid_max: int = 2401,
                  overlap_gate: float = 0.5) -> "ProductSpectrum":
        """1D sinc-DVR ladders along the principal axes, anchored at the exact 3D
        ground energy and rescaled per axis by the exact 3D first gap where the
        ``n_states_3d``-state solve reaches it (see the module docstring)."""
        from .noninteracting import NonInteractingSolver
        if isinstance(trap, HarmonicTrap):
            return cls.harmonic(trap)
        ni = NonInteractingSolver(trap, n_states=int(n_states_3d))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            specs = ni.axis_spectra(n_grid_max=int(axis_n_grid_max))
            cloud = ni.solve()
        tf = trap.trap_frequencies()
        mn = trap.minimum()
        r0 = mn.position
        E3, psi3, g3 = cloud.info.energies_3d, cloud.info.psi_3d, cloud.grid
        ladders = [_dvr_ladder(s) for s in specs]
        E_0 = float(E3[0])
        # the 1D box of a long axis auto-grows (to +-540 um for the lab tweezer) and is
        # capped at axis_n_grid_max points; the sinc-DVR ladder is exact while the
        # spacing stays under half a wavelength of the highest level used (at 801
        # points, 1.35 um, the tweezer's axial levels pair up from n = 6)
        E_ref = E_0 + 3.0 * kc.hbar * float(np.max(tf.omega)) - mn.potential_J
        lam_ref = kc.h / np.sqrt(2.0 * trap.mass * E_ref)
        for k, s in enumerate(specs):
            if s.du > 0.5 * lam_ref:
                warnings.warn(f"the 1D spectrum along principal axis {k} is sampled at "
                              f"{s.du * 1e6:.2f} um, over half the de Broglie wavelength "
                              f"{lam_ref * 1e6:.2f} um at E_0 + 3 hbar w_max; raise "
                              "axis_n_grid_max.", ModelValidityWarning, stacklevel=3)
        E_0_sep = mn.potential_J + float(sum(s.bound_energies[0] - s.V_min for s in specs))
        cal = cls._calibrate(ladders, tf, r0, E3, psi3, g3, overlap_gate)
        cal.E_0_separable_J, cal.n_states_3d = E_0_sep, int(n_states_3d)
        V0 = getattr(trap, "light_potential_J", None)
        if V0 is not None:                       # Born-Oppenheimer check for the softest axis
            depth = -float(np.asarray(V0(*r0)))
            soft = int(np.argmin([lad.eps[1] if lad.n_bound > 1 else np.inf for lad in ladders]))
            arg = 1.0 - kc.hbar * float(np.sum(np.delete(tf.omega, soft))) / (2.0 * depth)
            cal.bo_prediction = float(np.sqrt(arg)) if arg > 0 else np.nan
        return cls(E_0, ladders, tf.axes, r0, trap.mass, harmonic=False, calibration=cal,
                   separability_index=cloud.info.separability_index)

    @staticmethod
    def _calibrate(ladders, tf, r0, E3, psi3, g3, overlap_gate) -> Calibration:
        """Match the 3D excited states to the axes (overlap with the separable trial
        state, cross-checked by the second moment) and rescale the matched ladders."""
        D = (g3.X - r0[0], g3.Y - r0[1], g3.Z - r0[2])
        u = [np.broadcast_to(e[0] * D[0] + e[1] * D[1] + e[2] * D[2], g3.shape) for e in tf.axes]
        chi = [lad.functions(1, u[k]) for k, lad in enumerate(ladders)]
        factors, matched, overlaps = np.ones(3), np.full(3, -1), np.full(3, np.nan)
        defaulted = {}
        gaps = np.array([lad.eps[1] if lad.n_bound > 1 else np.inf for lad in ladders])
        reach = float(E3[-1] - E3[0]) * 1.05

        def second_moment(m, k):
            return float(np.sum(psi3[m] ** 2 * u[k] ** 2) * g3.dV)

        for k in range(3):
            if not np.isfinite(gaps[k]):
                defaulted[k] = "no bound excitation along this axis"
                continue
            if gaps[k] > reach:
                defaulted[k] = "beyond the reach of the 3D solve"
                continue
            trial = chi[k][1] * chi[(k + 1) % 3][0] * chi[(k + 2) % 3][0]
            trial = trial / np.sqrt(np.sum(trial ** 2) * g3.dV)
            ov = np.array([np.sum(trial * psi3[m]) * g3.dV for m in range(1, E3.size)]) ** 2
            m = int(np.argmax(ov)) + 1
            moment_axis = int(np.argmax([(second_moment(m, j) - second_moment(0, j)) * tf.omega[j]
                                         for j in range(3)]))
            if ov[m - 1] >= overlap_gate and moment_axis == k and m not in matched:
                factors[k] = float(E3[m] - E3[0]) / gaps[k]
                matched[k], overlaps[k] = m, ov[m - 1]
            else:
                defaulted[k] = f"overlap gate failed (best {ov[m - 1]:.2f} with state {m})"
        calibrated = [k for k in range(3) if matched[k] >= 0]
        if calibrated:
            k_cal = min(calibrated, key=lambda k: gaps[k])
            for k, why in list(defaulted.items()):
                near = abs(gaps[k] - gaps[k_cal]) < 0.1 * gaps[k_cal]
                if gaps[k] < gaps[k_cal] * 1.1 and not near and not why.startswith("no bound"):
                    warnings.warn(f"principal axis {k} (gap {gaps[k] / kc.h:.1f} Hz) is not "
                                  f"stiffer than the calibrated axis {k_cal} and got no "
                                  f"calibration ({why}); raise n_states_3d.", ModelValidityWarning,
                                  stacklevel=4)
                elif gaps[k] > gaps[k_cal]:
                    defaulted[k] = "stiffer than the calibrated axis: " + why
        else:
            warnings.warn("no principal axis could be calibrated against the 3D solve; "
                          "the separable ladders are used raw.", ModelValidityWarning, stacklevel=4)
        return Calibration(factors, matched, overlaps, defaulted, np.nan, np.nan, 0)


# ---------------------------------------------------------------- result

@dataclass
class FiniteTemperatureResult:
    """Diagnostics of a finite-temperature solve."""

    T_K: float                             #: temperature (K)
    thermal_model: str                     #: "hybrid" | "lda" | "boltzmann"
    condensate_model: str                  #: "gp", "noninteracting", "ideal-ground-state" or "none"
    mean_field_feedback: bool              #: was 2 g n_th fed back into the condensate
    mu_J: float = np.nan                   #: the equilibrium chemical potential (absolute, J)
    mu_GP_J: float = np.nan                #: the condensate eigenvalue it was built from (J)
    E_0_J: float = np.nan                  #: single-particle ground energy (J)
    E_cut_J: float = np.nan                #: seam between the discrete part and the tail (J); NaN for lda/boltzmann
    n_discrete: int = 0                    #: levels summed (excluding the ground state)
    n_levels_below_kT: float = np.nan      #: single-particle states below E_0 + kT (the ground state counts)
    n_levels_below_escape: float = np.nan  #: product levels below the escape energy (inf for a HarmonicTrap)
    weyl_count_escape: float = np.nan      #: semiclassical count below the escape energy
    seam_count_ratio: float = np.nan       #: discrete / Weyl count at E_cut (0.87 for the lab tweezer)
    quantum_fraction: float = np.nan       #: N_discrete / N_th
    min_level_margin: float = np.nan       #: min_n (E_n - mu) / kT over the discrete levels
    eta: float = np.nan                    #: (V_esc - V_min) / kT
    truncation_fraction: float = np.nan    #: 1 - N_th / N_th(untruncated tail), whole cloud; a lower bound
    tail_fraction: float = np.nan          #: N_tail / N_th
    passes: int = 0
    converged: bool = False
    N0_history: list = field(default_factory=list)       #: N_0 after each pass
    mu_history: list = field(default_factory=list)       #: mu - V_min (J) after each pass
    residual_history: list = field(default_factory=list) #: |advance| / max(1, N_0) per pass
    mixing_used: float = 1.0
    damped_at_pass: int = -1
    warm_start_fallbacks: int = 0          #: GP passes whose warm start was rejected
    grid_shape: tuple = ()
    basin_fraction: float = np.nan
    basin_volume_drift: float = np.nan     #: basin nodes of V + 2 g n_0 over those of V, minus 1
    basin_epsilon: float = BASIN_EPSILON
    basin_face_contact: bool = False
    gl_smax_clipped_fraction: float = 0.0  #: nodes where the BOSE_CUTOFF_KT cap bound (1 for a HarmonicTrap)
    edge_fraction_thermal: float = np.nan  #: face n_th / peak n_th
    normalization_error: float = np.nan    #: (N_0 + N_th) / N - 1
    calibration: Optional[Calibration] = None
    separability_index: float = np.nan
    wall_time_s: float = np.nan
    condensate_info: object = field(default=None, repr=False)

    @property
    def T_nK(self) -> float:
        return self.T_K * 1e9

    @property
    def trustworthy(self) -> bool:
        """Converged, ``eta >= 3``, the basin bounded, thermal density off the faces,
        normalization within 1e-6."""
        return bool(self.converged and (self.eta >= 3.0) and not self.basin_face_contact
                    and (self.edge_fraction_thermal < 1e-6) and abs(self.normalization_error) < 1e-6)

    def summary(self) -> str:
        h = kc.h
        lines = [f"finite temperature: {self.thermal_model} thermal cloud, {self.condensate_model} "
                 f"condensate, T = {self.T_nK:.2f} nK, feedback "
                 f"{'on' if self.mean_field_feedback else 'off'}",
                 f"  {self.passes} passes, converged {self.converged}, residuals "
                 + ", ".join(f"{r:.1e}" for r in self.residual_history)
                 + (f" (damped from pass {self.damped_at_pass})" if self.damped_at_pass >= 0 else ""),
                 f"  eta = {self.eta:.2f}, truncated fraction {self.truncation_fraction:.3f}, "
                 f"levels below E_0 + kT {self.n_levels_below_kT:.3g}, below escape "
                 f"{self.n_levels_below_escape:.4g} (Weyl {self.weyl_count_escape:.4g})"]
        if np.isfinite(self.E_cut_J):
            lines.append(f"  discrete {self.n_discrete} levels below E_cut = E_0 + "
                         f"{(self.E_cut_J - self.E_0_J) / h:.0f} Hz carry "
                         f"{self.quantum_fraction:.3f} of N_th; seam count ratio "
                         f"{self.seam_count_ratio:.3f}; min (E_n - mu)/kT = {self.min_level_margin:.3f}")
        if self.calibration is not None:
            c = self.calibration
            lines.append(f"  calibration factors {np.round(c.factors, 5).tolist()} (states "
                         f"{c.matched_states.tolist()}, BO prediction {c.bo_prediction:.4f}), "
                         f"separability index {self.separability_index:.2f}")
        lines.append(f"  grid {self.grid_shape}, basin {self.basin_fraction:.3f} of nodes, "
                     f"edge fraction {self.edge_fraction_thermal:.1e}, normalization "
                     f"{self.normalization_error:+.1e}, trustworthy {self.trustworthy}")
        return "\n".join(lines)


@dataclass
class _IdealGroundState:
    """The GP solver's ``a = 0`` ground state on the shared grid."""

    grid: TrapGrid
    density: np.ndarray        #: |psi_0|^2 normalized to 1 on the grid
    energy_J: float            #: E_0 on this grid
    inverse_volume: float      #: int |psi_0|^4 dV -- the first-order mean-field energy is g N_0 times it


@dataclass
class _SolveContext:
    """Everything a solve needs that does not change between passes."""

    N: float
    T_K: float
    kT: float
    a: float
    g: float
    mass: float
    V_min: float
    V_esc: float
    hw_max: float
    grid: TrapGrid
    V: np.ndarray
    basin: np.ndarray
    Vb: np.ndarray             #: V on the basin nodes
    spec: Optional[ProductSpectrum]
    E_0: float
    E_cut: float               #: NaN without a discrete part
    E_top: float               #: min(E_cut, V_esc); NaN without a discrete part
    E_lev: np.ndarray          #: product levels below E_top (ground state first)
    excited: np.ndarray        #: indices of the levels summed (n >= 1)
    ideal: _IdealGroundState

    @property
    def E_exc(self) -> np.ndarray:
        return self.E_lev[self.excited]

    def mu_of(self, N_0: float, mu_GP: float) -> float:
        """The closure: the condensate is one Bose level at ``mu_GP`` holding ``N_0``."""
        return mu_GP - self.kT * np.log1p(1.0 / N_0)


# ---------------------------------------------------------------- solver

class FiniteTemperatureSolver:
    """Condensate plus thermal cloud at ``T_K > 0`` (see the module docstring).

    Parameters
    ----------
    trap : Trap or HarmonicTrap
    a_scattering : float, optional
        Scattering length (m); looked up from the trap's state and field if omitted.
    condensate : {"gp", "noninteracting"}
        The condensate model (``"noninteracting"`` is the GP solver at ``a = 0``).
    thermal : {"hybrid", "lda", "boltzmann"}
    e_cut_hbar_omega : float
        Seam ``E_cut = E_0 + this x hbar w_max`` (2.0: hybrid within 1% of the exact
        harmonic sum; 0.5 is 4% off; must exceed ~0.5 or the tail diverges).
    n_states_3d : int
        States of the calibration solve (4 reaches the soft axis of a tweezer, the only
        one where the correction matters; ~15 reaches all three; 32 is the practical cap
        of the block eigensolver).
    axis_n_grid_max : int
        1D DVR resolution (2401: the axial box of the lab tweezer auto-grows to +-540 um;
        801 points there space the nodes at 1.35 um and corrupt the ladder above n = 5,
        1601 is exact for a 3 um waist and marginal for 3.4 um.  A warning fires when
        the spacing exceeds half the de Broglie wavelength at E_0 + 3 hbar w_max).
    basin_epsilon : float
        Basin cut at ``(1 - eps) x depth``; see :data:`BASIN_EPSILON`.
    mean_field_feedback : bool
        Feed ``2 g n_th`` back into the condensate and the levels (default False).
    max_passes, mixing, tol
        Outer loop: pass budget, initial mixing (1.0; damping latches automatically),
        gate on ``|advance| / max(1, N_0)``.
    n_thermal_widths : float
        HarmonicTrap box half-width in thermal (or ground-state) rms widths (6: +-4
        leaves 6e-5 per axis outside, at the containment gate).
    n_max_total : int
        Point budget for the shared grid; above it a ValueError names the geometry.
        The box grows like ``T^{3/2}`` for a HarmonicTrap (the lab-frequency trap
        exceeds 4e6 points above ~0.8 T_c with the GP spacing).
    grid : TrapGrid, optional
        Pin the shared grid.
    n_condensate_min : float
        Below this ``N_0`` the condensate is the ideal ground state.
    condensate_options : dict, optional
        Keyword arguments of :class:`~kamo.trap.gross_pitaevskii.GrossPitaevskiiSolver`
        (their names collide with this solver's, so they travel separately).
    strict : bool
        Raise :class:`~kamo.trap.cloud.ConvergenceError` when the loop misses ``tol``.

    One solver object serves a temperature sweep: the spectrum is built once, the
    shared grid is kept while the requested box fits in it, and each solve warm-starts
    the condensate from the previous one.
    """

    def __init__(self, trap, *, a_scattering: Optional[float] = None, condensate: str = "gp",
                 thermal: str = "hybrid", e_cut_hbar_omega: float = 2.0, n_states_3d: int = 4,
                 axis_n_grid_max: int = 2401, basin_epsilon: float = BASIN_EPSILON,
                 mean_field_feedback: bool = False, max_passes: int = 8, mixing: float = 1.0,
                 tol: float = 1e-4, n_thermal_widths: float = 6.0, n_max_total: int = 4_000_000,
                 grid: Optional[TrapGrid] = None, n_condensate_min: float = N_CONDENSATE_MIN,
                 condensate_options: Optional[dict] = None, strict: bool = True):
        if condensate not in CONDENSATE_MODELS:
            if condensate in ("thomas-fermi", "tf", "thomas_fermi"):
                raise ValueError("mode='thomas-fermi' is refused at T > 0: the Thomas-Fermi radius "
                                 "is ~2x wrong at the K-team operating point (mu - V_min is 95% "
                                 "zero point).  Use mode='gp'; the bimodal fit model is "
                                 "kamo.BEC_properties.thermal.IdealHarmonicBoseGas.")
            raise ValueError(f"condensate must be one of {CONDENSATE_MODELS}; got {condensate!r}")
        if thermal not in THERMAL_MODELS:
            raise ValueError(f"thermal must be one of {THERMAL_MODELS}; got {thermal!r}")
        if not e_cut_hbar_omega >= 0.5:
            raise ValueError("e_cut_hbar_omega must be >= 0.5: the tail integral diverges "
                             "logarithmically as E_cut -> mu")
        self.trap = trap
        self.a_scattering = a_scattering
        self.condensate = condensate
        self.thermal = thermal
        self.e_cut_hbar_omega = float(e_cut_hbar_omega)
        self.n_states_3d = int(n_states_3d)
        self.axis_n_grid_max = int(axis_n_grid_max)
        self.basin_epsilon = float(basin_epsilon)
        self.mean_field_feedback = bool(mean_field_feedback)
        self.max_passes = int(max_passes)
        self.mixing = float(mixing)
        self.tol = float(tol)
        self.n_thermal_widths = float(n_thermal_widths)
        self.n_max_total = int(n_max_total)
        self.grid = grid
        self.n_condensate_min = float(n_condensate_min)
        self.condensate_options = dict(condensate_options or {})
        self.strict = bool(strict)
        self._spectrum: Optional[ProductSpectrum] = None
        self._grid_cache = None                 # (grid, V, basin, V_esc)
        self._ideal: Optional[_IdealGroundState] = None
        self._psi_last = None                   # (grid, psi) of the last condensate solve
        self._fallbacks = 0

    # ------------------------------------------------------------- pieces
    def spectrum(self) -> ProductSpectrum:
        """The calibrated product spectrum; cached (it depends on neither N nor T)."""
        if self._spectrum is None:
            self._spectrum = ProductSpectrum.from_trap(
                self.trap, n_states_3d=self.n_states_3d, axis_n_grid_max=self.axis_n_grid_max)
        return self._spectrum

    def _gp(self, a, grid=None):
        from .gross_pitaevskii import GrossPitaevskiiSolver
        return GrossPitaevskiiSolver(self.trap, a_scattering=a, grid=grid, **self.condensate_options)

    def _scattering_length(self) -> float:
        if self.condensate != "gp":
            return 0.0
        return ia.trap_scattering_length(self.trap, self.a_scattering)

    def _thermal_half_widths(self, T_K: float, tf) -> np.ndarray:
        """Half-widths of the box the thermal cloud needs: the basin of a bounded trap,
        ``n_thermal_widths`` thermal (or ground-state) rms widths of a HarmonicTrap."""
        if np.isfinite(self.trap.trap_depth_J()):
            return self.trap.basin_half_widths(epsilon=self.basin_epsilon)
        kT = kc.kB * float(T_K)
        var = np.maximum(kT / (tf.mass * tf.omega ** 2), kc.hbar / (2.0 * tf.mass * tf.omega))
        return self.n_thermal_widths * np.sqrt((tf.axes ** 2).T @ var)

    def grid_for(self, N: float, T_K: float) -> TrapGrid:
        """The shared grid: the GP spacing on a box holding both the condensate and the
        thermal cloud.  A grid already built by this solver is reused while the box it
        would build fits inside it, so a temperature sweep keeps its warm start."""
        if self.grid is not None:
            return self.grid
        trap = self.trap
        base = self._gp(self._scattering_length()).grid_for(N)
        mn, tf = trap.minimum(), trap.trap_frequencies()
        half = np.maximum(base.half_widths, self._thermal_half_widths(T_K, tf))
        if (self._grid_cache is not None and np.all(half <= self._grid_cache[0].half_widths)
                and np.all(base.d >= 0.999 * self._grid_cache[0].d)):
            return self._grid_cache[0]
        dx = base.d
        if not np.isfinite(trap.trap_depth_J()):    # a HarmonicTrap: the basin is the whole box
            grid = self._grid_from(mn.position, half, dx)
            self._grid_cache = (grid,) + self._potential(grid, strict=False)
            return grid
        for _ in range(5):                       # grow any face the basin still touches
            grid = self._grid_from(mn.position, half, dx)
            V, basin, V_esc = self._potential(grid, strict=False)
            faces = touched_axes(basin)
            if not faces.any():
                self._grid_cache = (grid, V, basin, V_esc)
                return grid
            half = half * np.where(faces, 1.3, 1.0)
        raise ConvergenceError("the basin of the trap still touches a face of the shared grid "
                               "after five 1.3x growths: the mask leaks through the saddle (raise "
                               "basin_epsilon) or the trap is not bounded.")

    def _grid_from(self, r0, half, dx) -> TrapGrid:
        n = [_fast_even(np.ceil(2.0 * h / d)) for h, d in zip(half, dx)]
        total = int(np.prod(n))
        if total > self.n_max_total:
            raise ValueError(f"the shared grid would need {n[0]}x{n[1]}x{n[2]} = {total:.3g} points "
                             f"(spacing {np.round(np.asarray(dx) * 1e6, 3).tolist()} um for the "
                             f"condensate, half widths {np.round(np.asarray(half) * 1e6, 2).tolist()} um "
                             f"for the thermal cloud) above n_max_total = {self.n_max_total}; raise "
                             "it, coarsen condensate_options['points_per_scale'], lower "
                             "n_thermal_widths (HarmonicTrap), or pin grid=.")
        return TrapGrid.around(r0, half, n)

    def _potential(self, grid, strict: bool = True):
        trap = self.trap
        mn = trap.minimum()
        V = np.broadcast_to(np.asarray(trap.potential_J(grid.X, grid.Y, grid.Z), dtype=float),
                            grid.shape).copy()
        V_esc = mn.potential_J + float(trap.trap_depth_J())
        basin = basin_mask(V, grid, mn.position, V_esc, epsilon=self.basin_epsilon)
        if strict and np.isfinite(V_esc) and touched_axes(basin).any():
            raise ConvergenceError("the basin of the trap touches a face of the given grid: the "
                                   "box does not hold the whole region below the escape energy "
                                   "(or the mask leaked through the saddle; raise basin_epsilon).")
        return V, basin, V_esc

    def _ideal_ground_state(self, grid) -> _IdealGroundState:
        """The GP solver's ``a = 0`` ground state on this grid; cached per grid."""
        if self._ideal is None or self._ideal.grid is not grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                c = self._gp(0.0, grid).solve(1.0)
            rho = c.density_grid / grid.integrate(c.density_grid)
            self._ideal = _IdealGroundState(grid, rho, c.chemical_potential, grid.integrate(rho * rho))
        return self._ideal

    # ------------------------------------------------------------ prepare
    def _prepare(self, N: float, T_K: float) -> _SolveContext:
        trap = self.trap
        mn, tf = trap.minimum(), trap.trap_frequencies()
        if not (mn.converged and tf.is_bound):
            raise TrapTooShallowError("the trap has no bound minimum to hold a cloud")
        a = self._scattering_length()
        grid = self.grid_for(N, T_K)
        if self._grid_cache is None or self._grid_cache[0] is not grid:
            self._grid_cache = (grid,) + self._potential(grid)
        _, V, basin, V_esc = self._grid_cache
        if self._psi_last is not None and self._psi_last[0] is not grid:
            self._psi_last = None                   # a new box: the warm start does not transfer
        ideal = self._ideal_ground_state(grid)
        hw_max = kc.hbar * float(np.max(tf.omega))
        if self.thermal == "hybrid":
            spec = self.spectrum()
            E_0 = spec.E_0
            E_cut = E_0 + self.e_cut_hbar_omega * hw_max
            E_top = min(E_cut, V_esc)
            E_lev, _ = spec.levels(E_top)
            excited = np.arange(1, E_lev.size)
        else:                                       # no discrete part: no spectrum needed
            spec, E_0 = None, ideal.energy_J
            E_cut, E_top, E_lev, excited = np.nan, np.nan, np.empty(0), np.empty(0, dtype=int)
        return _SolveContext(N=N, T_K=T_K, kT=kc.kB * T_K, a=a, g=ia.coupling_g(a, trap.mass),
                             mass=trap.mass, V_min=mn.potential_J, V_esc=V_esc, hw_max=hw_max,
                             grid=grid, V=V, basin=basin, Vb=V[basin], spec=spec, E_0=E_0,
                             E_cut=E_cut, E_top=E_top, E_lev=E_lev, excited=excited, ideal=ideal)

    # -------------------------------------------------------- the pieces
    def _tail(self, ctx: _SolveContext, V_eff_b, E_0_eff=None) -> _TailCache:
        if self.thermal == "hybrid":
            return _TailCache(ctx.Vb, V_eff_b, ctx.T_K, ctx.mass, ctx.grid.dV, E_cut=ctx.E_cut,
                              V_escape=ctx.V_esc)
        if self.thermal == "lda":                   # fugacity from the true band bottom
            shifted = V_eff_b - float(np.min(V_eff_b)) + E_0_eff
            return _TailCache(ctx.Vb, shifted, ctx.T_K, ctx.mass, ctx.grid.dV, E_cut=None,
                              V_escape=ctx.V_esc)
        return _TailCache(ctx.Vb, ctx.Vb, ctx.T_K, ctx.mass, ctx.grid.dV, E_cut=None,
                          V_escape=ctx.V_esc, statistics="boltzmann")

    @staticmethod
    def _discrete_total(ctx: _SolveContext, mu: float, shifts) -> float:
        if ctx.excited.size == 0:
            return 0.0
        arg = (ctx.E_exc + shifts - mu) / ctx.kT
        if np.any(arg <= 0):
            j = int(np.argmin(arg))
            raise ConvergenceError(f"discrete level {ctx.excited[j]} at (E_n - mu)/kT = {arg[j]:.3g} "
                                   "is at or below the chemical potential, so its occupation would "
                                   "be negative: an attractive cloud, or a first-order level shift "
                                   "smaller than the mean-field rise of mu.")
        with np.errstate(over="ignore"):                      # 1 / expm1(huge) -> 0 as T -> 0
            return float(np.sum(1.0 / np.expm1(arg)))

    def _condensate(self, ctx: _SolveContext, N_0: float, n_th):
        """``(rho_shape, mu_GP, info, label)`` at ``N_0`` atoms."""
        if N_0 < self.n_condensate_min:
            ideal = ctx.ideal
            mu_GP = ideal.energy_J + ctx.g * N_0 * ideal.inverse_volume
            return ideal.density, mu_GP, None, "ideal-ground-state"
        V_extra = 2.0 * ctx.g * n_th if (self.mean_field_feedback and n_th is not None) else None
        psi0 = self._psi_last[1] if self._psi_last is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            c = self._gp(ctx.a, ctx.grid).solve(N_0, V_extra, psi0=psi0)
        if psi0 is not None and not c.info.warm_start:
            self._fallbacks += 1
        self._psi_last = (ctx.grid, c.info.psi)
        return c.density_grid / N_0, c.chemical_potential, c.info, self.condensate

    def _R(self, ctx: _SolveContext, N_0: float, mu_GP: float, shifts, tail: _TailCache) -> float:
        mu = ctx.mu_of(N_0, mu_GP)
        return N_0 + self._discrete_total(ctx, mu, shifts) + tail.total(mu) - ctx.N

    def _root(self, ctx: _SolveContext, mu_GP: float, shifts, tail: _TailCache) -> float:
        """``N_0`` with ``N_0 + N_th(mu(N_0)) = N``; bracketed on ``(0, N]`` for every T."""
        return brentq(lambda x: self._R(ctx, x, mu_GP, shifts, tail), 1e-9 * ctx.N, ctx.N,
                      xtol=1e-3, rtol=1e-8)

    def _thermal_density(self, ctx: _SolveContext, mu: float, shifts, tail: _TailCache) -> np.ndarray:
        n_th = np.zeros(ctx.grid.shape)
        n_th[ctx.basin] = tail.density(mu)
        if ctx.excited.size:
            occ = np.zeros(ctx.E_lev.size)
            with np.errstate(over="ignore"):
                occ[ctx.excited] = 1.0 / np.expm1((ctx.E_exc + shifts - mu) / ctx.kT)
            n_th += ctx.spec.density(ctx.grid, ctx.E_top, occ)
        return n_th

    # ---------------------------------------------------------- the loops
    def _solve_boltzmann(self, ctx: _SolveContext, result: FiniteTemperatureResult):
        """The classical gas: no condensate, ``mu`` from ``N``."""
        tail = self._tail(ctx, ctx.Vb)
        lo, hi = ctx.V_min - 80.0 * ctx.kT, ctx.V_min + BOSE_CUTOFF_KT * ctx.kT
        mu = brentq(lambda m: tail.total(m) - ctx.N, lo, hi, xtol=1e-12 * ctx.kT, rtol=1e-13)
        n_th = np.zeros(ctx.grid.shape)
        n_th[ctx.basin] = tail.density(mu)
        result.condensate_model, result.passes, result.converged = "none", 1, True
        result.mu_J, result.mu_GP_J = mu, np.nan
        return np.zeros(ctx.grid.shape), n_th, tail, mu, 0.0

    def _self_consistent(self, ctx: _SolveContext, result: FiniteTemperatureResult):
        """Passes of (condensate at N_0) -> (level shifts, tail on V_eff) -> (root of R)."""
        ideal = ctx.ideal
        n_0_shape, mu_GP, cinfo, label = ideal.density, ctx.E_0, None, "ideal-ground-state"
        shifts = np.zeros(ctx.excited.size)
        n_th, V_eff_b = None, ctx.Vb
        tail = self._tail(ctx, ctx.Vb, E_0_eff=ctx.E_0)
        N_0 = self._root(ctx, mu_GP, shifts, tail)     # pass 0: the bare spectrum
        result.N0_history.append(N_0)
        result.mu_history.append(ctx.mu_of(N_0, mu_GP) - ctx.V_min)
        if ctx.g == 0.0:
            # ideal gas: mu_GP = E_0 exactly and nothing shifts, so the pass-0 root is the
            # answer and the condensate is the ideal ground state at any N_0
            label = "ideal-ground-state" if N_0 < self.n_condensate_min else self.condensate
            result.passes, result.converged = 1, True
            result.residual_history.append(0.0)
            n_passes = 0
        else:
            n_passes = self.max_passes
        mixing, damped_at, prev_advance = self.mixing, -1, np.inf
        for p in range(1, n_passes + 1):
            n_0_shape, mu_GP, cinfo, label = self._condensate(ctx, N_0, n_th)
            n_0 = N_0 * n_0_shape
            W = 2.0 * ctx.g * n_0
            if self.mean_field_feedback and n_th is not None:
                W = W + 2.0 * ctx.g * n_th
            shifts = (ctx.spec.diagonal_expectation(ctx.grid, ctx.E_top, W)[ctx.excited]
                      if ctx.excited.size else np.empty(0))
            V_eff_b = ctx.Vb + W[ctx.basin]
            if N_0 >= self.n_condensate_min:
                E_0_eff = mu_GP + ctx.g * N_0 * ctx.grid.integrate(n_0_shape ** 2)
            else:
                E_0_eff = ctx.E_0 + ctx.g * N_0 * ideal.inverse_volume
            tail = self._tail(ctx, V_eff_b, E_0_eff=E_0_eff)
            N_0_new = self._root(ctx, mu_GP, shifts, tail)
            if self.mean_field_feedback:
                n_th = self._thermal_density(ctx, ctx.mu_of(N_0_new, mu_GP), shifts, tail)
            step = N_0_new - N_0
            if p > 1 and abs(mixing * step) > prev_advance:     # the map is not contracting:
                mixing = 0.5 if damped_at < 0 else 0.5 * mixing   # damp, and stay damped
                damped_at = p if damped_at < 0 else damped_at
            advance = mixing * step
            prev_advance = abs(advance)
            residual = abs(advance) / max(1.0, N_0)
            N_0 = N_0 + advance
            result.residual_history.append(residual)
            result.N0_history.append(N_0)
            result.mu_history.append(ctx.mu_of(N_0, mu_GP) - ctx.V_min)
            result.passes = p
            if residual < self.tol:
                result.converged = True
                break
        result.mixing_used, result.damped_at_pass = mixing, damped_at
        if not result.converged:
            msg = (f"the finite-temperature loop did not reach tol = {self.tol:g} in "
                   f"{self.max_passes} passes (residuals {result.residual_history}).")
            if self.strict:
                raise ConvergenceError(msg + "  Raise max_passes or lower mixing.")
            warnings.warn(msg, UserWarning, stacklevel=3)
        if ctx.a < 0:                                # monotonicity of R is not guaranteed
            dR = (self._R(ctx, N_0 * 1.01 + 1e-6, mu_GP, shifts, tail)
                  - self._R(ctx, N_0 * 0.99, mu_GP, shifts, tail))
            if dR <= 0:
                raise ConvergenceError("R(N_0) is not increasing at the root for this attractive "
                                       "cloud; the equilibrium is not unique here.")
        mu = ctx.mu_of(N_0, mu_GP)
        result.condensate_model = label
        result.mu_J, result.mu_GP_J = mu, mu_GP
        result.warm_start_fallbacks = self._fallbacks
        result.condensate_info = cinfo
        n_th = self._thermal_density(ctx, mu, shifts, tail)
        if ctx.excited.size:
            result.min_level_margin = float(np.min((ctx.E_exc + shifts - mu) / ctx.kT))
        return N_0 * n_0_shape, n_th, tail, mu, self._discrete_total(ctx, mu, shifts)

    # ------------------------------------------------------------- report
    def _report(self, ctx: _SolveContext, result: FiniteTemperatureResult, n_0, n_th, tail, mu,
                disc_total: float) -> None:
        grid, Vb, dV, m = ctx.grid, ctx.Vb, ctx.grid.dV, ctx.mass
        tail_total = tail.total(mu)
        N_th = disc_total + tail_total
        result.E_0_J, result.E_cut_J, result.n_discrete = ctx.E_0, ctx.E_cut, int(ctx.excited.size)
        result.eta = (ctx.V_esc - ctx.V_min) / ctx.kT
        result.quantum_fraction = disc_total / N_th if N_th > 0 else np.nan
        result.tail_fraction = tail_total / N_th if N_th > 0 else np.nan
        result.truncation_fraction = (0.0 if not np.isfinite(ctx.V_esc)
                                      else 1.0 - N_th / (disc_total + tail.total_untruncated(mu)))
        result.gl_smax_clipped_fraction = tail.clipped_fraction(mu)
        n_below_kT = weyl_count(ctx.E_0 + ctx.kT, Vb, dV, m)
        if ctx.spec is not None:
            n_disc_kT = int(np.count_nonzero(ctx.E_lev < ctx.E_0 + ctx.kT))
            w_top = weyl_count(ctx.E_top, Vb, dV, m)
            if ctx.E_0 + ctx.kT <= ctx.E_top:
                result.n_levels_below_kT = float(n_disc_kT)
            else:
                result.n_levels_below_kT = n_disc_kT + n_below_kT - w_top
            result.seam_count_ratio = ctx.E_lev.size / w_top if w_top > 0 else np.nan
            if not np.isfinite(ctx.V_esc):
                result.n_levels_below_escape = np.inf
            elif weyl_count(ctx.V_esc, Vb, dV, m) < N_LEVELS_MAX:
                result.n_levels_below_escape = float(ctx.spec.count_below(ctx.V_esc))
            result.calibration = ctx.spec.calibration
            result.separability_index = ctx.spec.separability_index
        else:
            result.n_levels_below_kT = n_below_kT
        result.weyl_count_escape = weyl_count(ctx.V_esc, Vb, dV, m) if np.isfinite(ctx.V_esc) else np.inf
        result.normalization_error = grid.integrate(n_0 + n_th) / ctx.N - 1.0
        peak = float(np.max(n_th))
        result.edge_fraction_thermal = grid.face_max(n_th) / peak if peak > 0 else 0.0
        result.basin_face_contact = bool(np.isfinite(ctx.V_esc) and touched_axes(ctx.basin).any())
        result.grid_shape, result.basin_fraction = grid.shape, float(np.mean(ctx.basin))
        result.basin_epsilon = self.basin_epsilon
        if ctx.g != 0.0 and np.isfinite(ctx.V_esc):
            basin_eff = basin_mask(ctx.V + 2.0 * ctx.g * n_0, grid, self.trap.minimum().position,
                                   ctx.V_esc, epsilon=self.basin_epsilon)
            result.basin_volume_drift = float(np.count_nonzero(basin_eff)
                                              / max(1, np.count_nonzero(ctx.basin)) - 1.0)
        else:
            result.basin_volume_drift = 0.0

    def _warn(self, result: FiniteTemperatureResult) -> None:
        if self.thermal == "lda":
            warnings.warn("thermal='lda' is the semiclassical model; at the K-team operating point "
                          "it is low in N_th by 6.5x (30 nK), 3.6x (50 nK), 2.2x (100 nK), 1.8x "
                          "(170 nK) against the hybrid.  Use thermal='hybrid'.",
                          ModelValidityWarning, stacklevel=3)
        elif self.thermal == "boltzmann":
            warnings.warn("thermal='boltzmann' is the classical limit: no Bose enhancement, no "
                          "condensate.", ModelValidityWarning, stacklevel=3)
        eta = result.eta
        if np.isfinite(eta) and (eta < 5.0 or result.truncation_fraction > 0.3):
            warnings.warn(f"eta = U/kT = {eta:.2f} and {100 * result.truncation_fraction:.1f}% of the "
                          f"thermal cloud is cut at the escape energy ({result.n_levels_below_kT:.3g} "
                          "single-particle levels below E_0 + kT): this is an evaporating "
                          "quasi-equilibrium, not an equilibrium state.", ModelValidityWarning,
                          stacklevel=3)
        if (self.thermal == "hybrid" and np.isfinite(result.seam_count_ratio)
                and result.seam_count_ratio < 0.5):
            warnings.warn(f"the product basis holds only {result.seam_count_ratio:.2f} of the Weyl "
                          "count at E_cut: the trap is far from separable and the discrete part is "
                          "incomplete.", ModelValidityWarning, stacklevel=3)

    # --------------------------------------------------------------- solve
    def solve(self, N: float, T_K: float) -> TrapCloud:
        """The condensate and thermal cloud of ``N`` atoms at ``T_K`` (K), as one
        :class:`~kamo.trap.cloud.TrapCloud` with both components."""
        t_start = time.perf_counter()
        N, T_K = float(N), float(T_K)
        if not N > 0:
            raise ValueError(f"N must be positive; got {N}")
        if not T_K > 0:
            raise ValueError("T_K must be > 0 here; solve(..., T_K=0) uses the ground-state solvers")
        ctx = self._prepare(N, T_K)
        self._fallbacks = 0
        result = FiniteTemperatureResult(T_K=T_K, thermal_model=self.thermal,
                                         condensate_model=self.condensate,
                                         mean_field_feedback=self.mean_field_feedback,
                                         mixing_used=self.mixing)
        if self.thermal == "boltzmann":
            n_0, n_th, tail, mu, disc_total = self._solve_boltzmann(ctx, result)
        else:
            n_0, n_th, tail, mu, disc_total = self._self_consistent(ctx, result)
        self._report(ctx, result, n_0, n_th, tail, mu, disc_total)
        self._warn(result)
        result.wall_time_s = time.perf_counter() - t_start
        return TrapCloud(ctx.grid, n_0 + n_th, N, self.trap, mode=self.condensate,
                         chemical_potential_J=mu, V_min_J=ctx.V_min, energy_per_atom_J=float("nan"),
                         a_scattering=ctx.a, info=result, T_K=T_K, density_condensate=n_0,
                         density_thermal=n_th)


# ------------------------------------------------------- the inverse questions

@dataclass
class CriticalTemperature:
    """Reference temperatures for ``N`` atoms in a trap -- not a sharp transition.

    ``eta_at_T_c`` is 1.37 for the K-team tweezer: at ``T_c`` the trap is barely deeper
    than ``kT``, so the ``N_0(T)`` curve, not a ``T_c``, is the answer.  Each
    temperature has an ``_nK`` view.
    """

    N: float
    omega: np.ndarray                  #: principal angular frequencies (rad/s)
    depth_J: float                     #: escape depth (J); inf for a HarmonicTrap
    a_scattering: Optional[float]      #: (m) for the mean-field shift; None if unknown
    fraction_target: float             #: the N_0/N the "numerical" and "exact series" values cross
    T_c_ideal_K: float                 #: harmonic closed form, k T_c = hbar wbar (N / zeta(3))^{1/3}
    T_c_finite_size_K: float           #: with the leading N^{-1/3} correction
    T_c_exact_series_K: float          #: where the exact finite-N ideal harmonic sum reaches fraction_target
    T_c_interaction_shift: float       #: relative mean-field shift, -1.33 (a / a_ho) N^{1/6}; NaN without a
    T_c_numerical_K: float = np.nan    #: the solver's own crossing; NaN unless asked for

    @property
    def T_c_ideal_nK(self) -> float:
        return self.T_c_ideal_K * 1e9

    @property
    def T_c_finite_size_nK(self) -> float:
        return self.T_c_finite_size_K * 1e9

    @property
    def T_c_exact_series_nK(self) -> float:
        return self.T_c_exact_series_K * 1e9

    @property
    def T_c_numerical_nK(self) -> float:
        return self.T_c_numerical_K * 1e9

    @property
    def eta_at_T_c(self) -> float:
        return self.depth_J / (kc.kB * self.T_c_ideal_K)

    def summary(self) -> str:
        f = self.omega / (2 * np.pi)
        lines = [f"critical temperature references, N = {self.N:.0f}, f = "
                 f"({f[0]:.1f}, {f[1]:.1f}, {f[2]:.1f}) Hz:",
                 f"  ideal harmonic       {self.T_c_ideal_nK:8.2f} nK   "
                 f"(eta = depth/kT_c = {self.eta_at_T_c:.2f})",
                 f"  finite-size formula  {self.T_c_finite_size_nK:8.2f} nK",
                 f"  exact finite-N sum   {self.T_c_exact_series_nK:8.2f} nK   "
                 f"(N0/N = {self.fraction_target:g})"]
        if np.isfinite(self.T_c_interaction_shift):
            lines.append(f"  mean-field shift     {100 * self.T_c_interaction_shift:+8.2f} %")
        if np.isfinite(self.T_c_numerical_K):
            lines.append(f"  solver (real trap)   {self.T_c_numerical_nK:8.2f} nK   "
                         f"(N0/N = {self.fraction_target:g})")
        lines.append("  there is no sharp transition at this N; the N_0(T) curve is the answer")
        return "\n".join(lines)


def _check_solver_arguments(solver, a_scattering, solver_kwargs):
    if solver is not None and (solver_kwargs or a_scattering is not None):
        raise ValueError("pass either solver= or the solver's arguments, not both")


def critical_temperature(trap, N: float, *, a_scattering: Optional[float] = None,
                         numerical: bool = False, fraction_target: float = 0.01,
                         solver: Optional[FiniteTemperatureSolver] = None,
                         T_bracket_K=None, **solver_kwargs) -> CriticalTemperature:
    """Reference ``T_c`` values for ``N`` atoms in ``trap`` (see :class:`CriticalTemperature`);
    with ``numerical`` also the temperature at which the finite-temperature solver's
    condensate fraction falls to ``fraction_target`` (a bisection over full solves).
    ``a_scattering`` is looked up from the trap when omitted; the solver repeats that
    lookup itself."""
    _check_solver_arguments(solver, a_scattering, solver_kwargs)
    tf = trap.trap_frequencies()
    N = float(N)
    a = a_scattering
    if a is None and not isinstance(trap, HarmonicTrap):
        try:
            a = ia.trap_scattering_length(trap, None)
        except ValueError:
            a = None
    T0 = critical_temperature_K(N, tf.omega)
    fs = T0 * (1.0 + finite_size_shift(N, tf.omega))
    f_exact = lambda T: ideal_harmonic_condensate(N, T, tf.omega)[0] / N - fraction_target
    T_exact = brentq(f_exact, 0.05 * T0, 3.0 * T0, xtol=1e-6 * T0)
    shift = np.nan if a is None else mean_field_shift(N, tf.omega, a, trap.mass)
    depth = float(trap.trap_depth_J())
    out = CriticalTemperature(N, tf.omega.copy(), depth, a, fraction_target, T0, fs, T_exact, shift)
    if numerical:
        s = solver or FiniteTemperatureSolver(trap, a_scattering=a_scattering, **solver_kwargs)
        lo, hi = (0.2 * T0, 1.6 * T0) if T_bracket_K is None else tuple(T_bracket_K)

        def f(T):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ModelValidityWarning)
                return s.solve(N, T).condensate_fraction - fraction_target

        out.T_c_numerical_K = brentq(f, lo, hi, xtol=2e-3 * T0)
    return out


def temperature_from_condensate_fraction(trap, N: float, fraction: float, *,
                                         a_scattering: Optional[float] = None,
                                         solver: Optional[FiniteTemperatureSolver] = None,
                                         bracket_K=None, rtol: float = 1e-3,
                                         full_output: bool = False, **solver_kwargs):
    """The temperature at which ``N`` atoms in ``trap`` have condensate fraction
    ``fraction``: ``brentq`` on the solver's fraction, bracketed by default on
    ``[0.05, 1.5] x T_c_ideal`` (checked, not assumed; the two endpoint solves are
    reused).  Returns ``T_K``, or ``(T_K, cloud)`` with ``full_output``.  This is
    the lab-facing direction: a bimodal fit measures the fraction."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    _check_solver_arguments(solver, a_scattering, solver_kwargs)
    tf = trap.trap_frequencies()
    T0 = critical_temperature_K(N, tf.omega)
    s = solver or FiniteTemperatureSolver(trap, a_scattering=a_scattering, **solver_kwargs)
    lo, hi = (0.05 * T0, 1.5 * T0) if bracket_K is None else tuple(bracket_K)
    clouds = {}

    def f(T):
        if T not in clouds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ModelValidityWarning)
                clouds[T] = s.solve(N, T)
        return clouds[T].condensate_fraction - fraction

    f_lo, f_hi = f(lo), f(hi)
    if not (f_lo > 0 > f_hi):
        raise ValueError(f"the bracket [{lo * 1e9:.1f}, {hi * 1e9:.1f}] nK does not enclose "
                         f"fraction {fraction}: N0/N = {f_lo + fraction:.3f} and {f_hi + fraction:.3f} "
                         "at its ends; pass bracket_K.")
    T = brentq(f, lo, hi, rtol=rtol, xtol=1e-3 * T0)
    if full_output:
        return T, clouds[T] if T in clouds else s.solve(N, T)
    return T
