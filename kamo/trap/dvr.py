"""One trap axis, solved exactly: the Colbert-Miller sinc-DVR eigensolver.

This productizes ``analysis/atomic_physics/gaussian_well_wavefunctions.ipynb``.
The notebook solved the dimensionless Gaussian well ``-eta exp(-2u^2)`` and its
gravity-tilted cousin ``-eta exp(-2u^2) + gamma u``.  Here the potential is any
callable ``V(u)`` in SI, so one solver covers a radial cut, an axial (Lorentzian)
cut, a principal axis of a crossed trap or a light-sheet axis -- and ``eta``,
``gamma`` become reported diagnostics rather than the interface.

Method
------
Uniform grid with spacing ``du``; kinetic energy in the sinc discrete variable
representation (Colbert & Miller, J. Chem. Phys. 96, 1982 (1992))::

    T_ii = (hbar^2 / 2 m du^2) pi^2 / 3
    T_ij = (hbar^2 / 2 m du^2) 2 (-1)^(i-j) / (i-j)^2

with ``V`` diagonal.  For a smooth potential the eigenvalues converge
exponentially in ``du``, and the eigenvectors are ``psi(u_i) sqrt(du)``.

What counts as "bound"
----------------------
A finite box turns the continuum into box states, and with gravity nothing is
truly bound: every state is a resonance that eventually tunnels out.  So the
solver finds the escape energy first and classifies states against it.

* ``E_escape`` is the 1D mountain pass out of the well: the lower of the two
  barriers (running maxima) between the minimum and the two ends of the probe
  range.  If that barrier is an interior local maximum it is an *escape lip*
  (the tilted case) and is refined to machine precision.  Otherwise the well
  runs out at the box edge (the untilted case, where ``E_escape -> 0``).
* A state is **bound** if ``E < E_escape``, more than ``well_weight_min`` of
  its probability lies in the well (the connected region around the minimum
  where ``V < E_escape``), and less than ``edge_tol`` of it lies in the outer
  ``edge_frac`` of the box on every side *without* a lip.  These are the
  notebook's two tests (``solve_gaussian_well`` and ``trapped_states``) unified.
* A state that passes the first two tests but touches the wall is a weakly
  bound state the box is too small for, not a box state.  With ``auto_box`` the
  box is grown (spacing fixed) until none are left; otherwise a warning says how
  many were dropped.  The notebook's own ``L = 6`` row loses the top state of
  our tweezer exactly this way.

Box defaults follow the notebook's rule, in units of the axis ``length_scale``
``L`` (the waist for a radial cut, the Rayleigh range for an axial one) and the
depth ``eta = (E_escape - V_min) / (hbar^2 / m L^2)``::

    half-width = (6 + 6/eta) L,     du = min(0.05, 0.35/sqrt(eta)) L,

with the downhill side of a tilted well cut to ``max(|u_lip - u_min| + 1.5 L,
3 L)`` so the box's downhill pocket stays small.

Quick start
-----------
>>> import numpy as np, kamo.constants as kc
>>> from kamo.trap.dvr import solve_axis
>>> V0, w0 = kc.h * 8.674e3, 3.0e-6
>>> s = solve_axis(lambda u: -V0 * np.exp(-2 * u**2 / w0**2), length_scale=w0)
>>> s.n_bound, s.frequency_Hz, s.anharmonicity
>>> print(s.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize_scalar

import kamo.constants as kc

G_EARTH = 9.80665          #: standard gravity, m/s^2 (scipy.constants.g)


# ------------------------------------------------------------------- scales

def waist_energy(length: float, mass: float) -> float:
    """``E_w = hbar^2 / (m L^2)``, the kinetic scale of an atom confined to ``L`` (J)."""
    return kc.hbar ** 2 / (mass * length ** 2)


def gaussian_eta(depth: float, length: float, mass: float) -> float:
    """Dimensionless depth ``eta = V0 / E_w`` of a well of ``depth`` (J) and width ``length``."""
    return depth / waist_energy(length, mass)


def gravity_gamma(length: float, mass: float, g: float = G_EARTH) -> float:
    """Dimensionless tilt ``gamma = m g L / E_w`` of gravity along an axis of width ``length``."""
    return mass * g * length / waist_energy(length, mass)


def critical_gamma(eta: float) -> float:
    """Tilt at which the Gaussian well ``-eta exp(-2u^2)`` loses its minimum, ``2 eta / sqrt(e)``."""
    return 2.0 * eta / np.sqrt(np.e)


# ------------------------------------------------------------------ kinetic

def sinc_dvr_kinetic(n_grid: int, du: float, mass: float) -> np.ndarray:
    """Colbert-Miller sinc-DVR kinetic-energy matrix (J) on ``n_grid`` points spaced ``du``."""
    i = np.arange(int(n_grid))
    dij = (i[:, None] - i[None, :]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = 2.0 * (-1.0) ** dij / dij ** 2
    np.fill_diagonal(T, np.pi ** 2 / 3.0)
    return (kc.hbar ** 2 / (2.0 * mass * du ** 2)) * T


# ------------------------------------------------------------------- result

@dataclass
class AxisSpectrum:
    """Eigenstates of one trap axis, classified against the escape energy."""

    u: np.ndarray             #: DVR nodes (m)
    V: np.ndarray             #: potential on the nodes (J)
    energies: np.ndarray      #: every eigenvalue of the box, ascending (J)
    psi: np.ndarray           #: (n_grid, n_bound) bound eigenfunctions, int |psi|^2 du = 1
    bound: np.ndarray         #: bool mask over ``energies``
    well_weight: np.ndarray   #: probability in the well, per eigenstate
    edge_weight: np.ndarray   #: probability in the checked box edges, per eigenstate
    u_min: float              #: position of the (sagged) minimum (m)
    V_min: float              #: potential at the minimum (J)
    E_escape: float           #: escape energy the states are classified against (J)
    u_lip: float              #: position of the escape lip (m); NaN if none
    omega: float              #: curvature frequency at the minimum (rad/s)
    mass: float               #: atomic mass (kg)
    n_unresolved: int = 0     #: states below E_escape dropped because they touch the wall
    box_growths: int = 0      #: times the box was grown by ``auto_box``

    # ------------------------------------------------------------- geometry
    @property
    def du(self) -> float:
        return float(self.u[1] - self.u[0])

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.u[0]), float(self.u[-1])

    @property
    def has_lip(self) -> bool:
        return bool(np.isfinite(self.u_lip))

    @property
    def usable_depth(self) -> float:
        """``E_escape - V_min`` (J): the depth measured from the sagged minimum."""
        return self.E_escape - self.V_min

    # --------------------------------------------------------------- levels
    @property
    def n_bound(self) -> int:
        return int(np.count_nonzero(self.bound))

    @property
    def bound_energies(self) -> np.ndarray:
        return self.energies[self.bound]

    @property
    def n_effective(self) -> float:
        """Total well-weight of every state below ``E_escape``: a count that does
        not flicker when a well state hybridizes with a box-pocket state."""
        below = self.energies < self.E_escape
        return float(np.sum(self.well_weight[below]))

    @property
    def ground_energy(self) -> float:
        if self.n_bound == 0:
            raise ValueError("this axis holds no bound states")
        return float(self.bound_energies[0])

    @property
    def ground_state(self) -> np.ndarray:
        return self.psi[:, 0]

    @property
    def hbar_omega(self) -> float:
        return kc.hbar * self.omega

    @property
    def frequency_Hz(self) -> float:
        """Harmonic frequency from the curvature at the (sagged) minimum."""
        return self.omega / (2.0 * np.pi)

    @property
    def spacings(self) -> np.ndarray:
        return np.diff(self.bound_energies)

    @property
    def spacings_over_hbar_omega(self) -> np.ndarray:
        return self.spacings / self.hbar_omega

    @property
    def anharmonicity(self) -> float:
        """``(E_0 - (V_min + hbar omega / 2)) / hbar omega``: the ground state's
        departure from the harmonic ladder, in units of the level spacing."""
        return (self.ground_energy - (self.V_min + 0.5 * self.hbar_omega)) / self.hbar_omega

    # ----------------------------------------------------------- moments
    def _ground_density(self) -> np.ndarray:
        return self.ground_state ** 2 * self.du

    @property
    def mean_position(self) -> float:
        """Ground-state ``<u>`` (m)."""
        return float(np.sum(self.u * self._ground_density()))

    @property
    def sigma(self) -> float:
        """Ground-state rms width about its mean (m)."""
        p = self._ground_density()
        mu = float(np.sum(self.u * p))
        return float(np.sqrt(np.sum((self.u - mu) ** 2 * p)))

    @property
    def sigma_harmonic(self) -> float:
        """``sqrt(hbar / (2 m omega))``: the harmonic ground-state width at this curvature."""
        return float(np.sqrt(kc.hbar / (2.0 * self.mass * self.omega)))

    def summary(self) -> str:
        """Human-readable report, energies as E/h."""
        h = kc.h
        lines = [f"AxisSpectrum: {self.n_bound} bound state(s)"
                 + (f", {self.n_unresolved} unresolved at the box edge" if self.n_unresolved else ""),
                 f"  minimum      u = {self.u_min * 1e6:+.4f} um,  V_min/h = {self.V_min / h / 1e3:.4f} kHz",
                 "  escape       " + (f"lip at u = {self.u_lip * 1e6:+.3f} um" if self.has_lip
                                      else "over the box edge")
                 + f",  E_escape/h = {self.E_escape / h / 1e3:.4f} kHz,"
                   f"  usable depth/h = {self.usable_depth / h / 1e3:.4f} kHz",
                 f"  curvature    f = {self.frequency_Hz:.3f} Hz at the minimum"]
        if self.n_bound:
            sp = ", ".join(f"{x:.4f}" for x in self.spacings_over_hbar_omega[:5])
            lines += [f"  ground state (E_0 - V_min)/h = {(self.ground_energy - self.V_min) / h:.3f} Hz,"
                      f"  anharmonicity {100 * self.anharmonicity:+.3f}% of hbar omega",
                      f"  spacings / hbar omega: {sp}{', ...' if self.n_bound > 6 else ''}",
                      f"  sigma = {self.sigma * 1e6:.4f} um  (harmonic {self.sigma_harmonic * 1e6:.4f} um)"]
        return "\n".join(lines)


# -------------------------------------------------------------- internals

def _descend(Vs: np.ndarray, i: int) -> int:
    """Walk downhill on a sampled curve from index ``i`` to a local minimum."""
    n = Vs.size
    while True:
        best = i
        if i > 0 and Vs[i - 1] < Vs[best]:
            best = i - 1
        if i < n - 1 and Vs[i + 1] < Vs[best]:
            best = i + 1
        if best == i:
            return i
        i = best


def _barrier(Vs: np.ndarray, i_min: int, direction: int):
    """Highest point between the minimum and the grid end in ``direction``.

    Returns ``(index, value, interior)``; ``interior`` means the barrier is a
    local maximum inside the range (a lip) rather than the grid end itself.
    """
    seg = Vs[i_min:] if direction > 0 else Vs[i_min::-1]
    k = int(np.argmax(seg))
    # an interior maximum counts as a lip only if it stands clear of the end of
    # the range: on a flat asymptote round-off puts the argmax anywhere.
    interior = 0 < k < seg.size - 1 and (seg[k] - seg[-1]) > 1e-6 * (seg[k] - Vs[i_min])
    return i_min + direction * k, float(seg[k]), bool(interior)


def _refine(V_on, a: float, b: float, sign: float):
    """Refine a minimum (``sign=+1``) or maximum (``sign=-1``) of V in ``[a, b]``."""
    f = lambda x: sign * float(V_on(np.array([x], dtype=float))[0])
    res = minimize_scalar(f, bounds=(a, b), method="bounded",
                          options=dict(xatol=abs(b - a) * 1e-10))
    return float(res.x), sign * float(res.fun)


# ------------------------------------------------------------------ solver

def solve_axis(V: Callable[[np.ndarray], np.ndarray], *,
               length_scale: Optional[float] = None, center: float = 0.0,
               bounds=None, half_width: Optional[float] = None,
               n_grid: Optional[int] = None, mass: Optional[float] = None,
               E_escape: Optional[float] = None, edge_frac: float = 0.15,
               edge_tol: float = 1e-6, well_weight_min: float = 0.5,
               auto_box: Optional[bool] = None, max_box_growth: int = 3,
               n_grid_min: int = 801, n_grid_max: int = 4001,
               probe_points: int = 4001) -> AxisSpectrum:
    """Eigenstates of ``-hbar^2/2m d^2/du^2 + V(u)`` on one trap axis.

    Parameters
    ----------
    V : callable
        Potential in J of a position array in m (numpy in, numpy out).  Include
        gravity yourself if the axis has a vertical component.
    length_scale : float, optional
        The axis's natural width -- the waist for a radial cut, the Rayleigh
        range for an axial one.  Sets the default box and spacing.  Required
        unless an explicit box *and* ``n_grid`` are given.
    center : float
        Where to start looking for the minimum (m); the nearest local minimum
        downhill of it is used.
    bounds, half_width : optional
        An explicit box, ``(lo, hi)`` or ``center +- half_width`` (m).  An
        explicit box is not grown unless ``auto_box=True``.
    n_grid : int, optional
        Grid points; default from the spacing rule.
    mass : float, optional
        Atomic mass (kg); defaults to K-39 (``kamo.constants.m_K``, via ARC).
    E_escape : float, optional
        Override the escape energy (J) -- e.g. ``0.0`` for a potential whose
        long-range tail a finite box cannot reach.
    edge_frac, edge_tol, well_weight_min : float
        The bound-state tests (see the module docstring).
    auto_box : bool, optional
        Grow the box until no state below ``E_escape`` touches the wall.
        Default: on for an automatic box, off for an explicit one.
    max_box_growth : int
        Maximum number of 1.5x box growths.
    n_grid_min, n_grid_max : int
        Floor and cap on the automatic grid size (dense ``eigh`` is O(n^3)).
    probe_points : int
        Samples used to locate the minimum and the escape lip.
    """
    m = float(kc.m_K if mass is None else mass)
    if bounds is not None and half_width is not None:
        raise ValueError("give `bounds` or `half_width`, not both")
    if half_width is not None:
        bounds = (center - float(half_width), center + float(half_width))
    explicit = bounds is not None
    L = None if length_scale is None else float(length_scale)
    if L is not None and not L > 0:
        raise ValueError(f"length_scale must be positive; got {length_scale}")
    if explicit:
        lo, hi = (float(b) for b in bounds)
        if not hi > lo:
            raise ValueError(f"bounds must be increasing; got {bounds}")
        if n_grid is None and L is None:
            raise ValueError("an explicit box needs `n_grid` (or `length_scale`, "
                             "from which the spacing is chosen)")
    elif L is None:
        raise ValueError("give `length_scale` (the waist or Rayleigh range along "
                         "this axis), or an explicit box with `n_grid`")
    if auto_box is None:
        auto_box = not explicit

    def V_on(x):
        out = np.asarray(V(x), dtype=float)
        return out if out.shape == x.shape else np.broadcast_to(out, x.shape).astype(float)

    # ---- locate the well and its escape route on a probe grid
    plo, phi = (lo, hi) if explicit else (center - 12.0 * L, center + 12.0 * L)
    up = np.linspace(plo, phi, int(probe_points))
    Vp = V_on(up)
    i_min = _descend(Vp, int(np.argmin(np.abs(up - center))))
    if i_min in (0, up.size - 1):
        raise ValueError(
            f"no local minimum near `center` within [{plo:.4g}, {phi:.4g}] m: the "
            "potential runs downhill to the edge of the range (e.g. gravity "
            "stronger than the trap can hold).")
    u_min, V_min = _refine(V_on, up[i_min - 1], up[i_min + 1], +1.0)
    iL, vL, lipL = _barrier(Vp, i_min, -1)
    iR, vR, lipR = _barrier(Vp, i_min, +1)
    side = -1 if vL <= vR else +1
    lip_side, u_lip, E_lip = 0, float("nan"), float("nan")
    if (lipL if side == -1 else lipR):
        i_lip = iL if side == -1 else iR
        u_lip, E_lip = _refine(V_on, up[i_lip - 1], up[i_lip + 1], -1.0)
        lip_side = side

    # ---- default box and spacing: the notebook's rule, in units of L
    if L is not None:
        depth = (E_lip if lip_side else min(vL, vR)) - V_min
        if not depth > 0:
            raise ValueError("the well has no depth: its minimum is not below "
                             "the escape energy.")
        eta = depth / waist_energy(L, m)
        du_target = min(0.05, 0.35 / np.sqrt(eta)) * L
    if not explicit:
        hw = (6.0 + 6.0 / eta) * L
        ext = {-1: hw, +1: hw}
        if lip_side:
            ext[lip_side] = min(hw, max(abs(u_lip - u_min) + 1.5 * L, 3.0 * L))
        lo, hi = u_min - ext[-1], u_min + ext[+1]
    if n_grid is not None:
        n = int(n_grid)
    else:
        n = max(int(n_grid_min), int(np.ceil((hi - lo) / du_target)) + 1)

    # ---- diagonalize, classify, and grow the box while bound states touch it
    growths, capped = 0, False
    while True:
        if n > n_grid_max and n_grid is None:
            if not capped:
                warnings.warn(f"grid capped at n_grid_max = {n_grid_max} points "
                              f"(wanted {n}); the spacing is coarser than the "
                              "default rule.", UserWarning, stacklevel=2)
                capped = True
            n = int(n_grid_max)
        u = np.linspace(lo, hi, n)
        du = float(u[1] - u[0])
        Vu = V_on(u)
        scale = kc.hbar ** 2 / (2.0 * m * du ** 2)       # keep eigh's numbers O(1)
        H = sinc_dvr_kinetic(n, du, m)
        H[np.diag_indices(n)] += Vu
        evals, vecs = np.linalg.eigh(H / scale)
        evals = evals * scale
        psi_all = vecs / np.sqrt(du)
        peak = np.argmax(np.abs(psi_all), axis=0)
        psi_all *= np.sign(psi_all[peak, np.arange(n)])   # fix the arbitrary sign

        if E_escape is not None:
            E_esc = float(E_escape)
        elif lip_side:
            E_esc = float(E_lip)
        else:
            E_esc = float(min(Vu[0], Vu[-1]))

        j = int(np.argmin(np.abs(u - u_min)))
        a = j
        while a > 0 and Vu[a - 1] < E_esc:
            a -= 1
        b = j
        while b < n - 1 and Vu[b + 1] < E_esc:
            b += 1
        # Never walk across the escape lip: its sampled height sits just below
        # the true maximum, so without this the well would leak into the
        # downhill pocket beyond it and count pocket states as trapped.
        if lip_side == -1:
            a = max(a, int(np.searchsorted(u, u_lip, side="right")))
        elif lip_side == +1:
            b = min(b, int(np.searchsorted(u, u_lip, side="left")) - 1)
        well_w = np.sum(psi_all[a:b + 1] ** 2, axis=0) * du
        edge = np.zeros(n, dtype=bool)
        if lip_side != -1:
            edge |= u < lo + edge_frac * (u_min - lo)
        if lip_side != +1:
            edge |= u > hi - edge_frac * (hi - u_min)
        edge_w = np.sum(psi_all[edge] ** 2, axis=0) * du

        candidate = (evals < E_esc) & (well_w > well_weight_min)
        bound = candidate & (edge_w < edge_tol)
        unresolved = candidate & ~bound
        if unresolved.any() and auto_box and growths < max_box_growth:
            growths += 1
            if lip_side != -1:
                lo = u_min - 1.5 * (u_min - lo)
            if lip_side != +1:
                hi = u_min + 1.5 * (hi - u_min)
            n = int(np.ceil((hi - lo) / du)) + 1
            continue
        break

    n_unresolved = int(np.count_nonzero(unresolved))
    if n_unresolved:
        hint = ("Pass auto_box=True or a larger box." if not auto_box else
                f"The box was grown {growths} time(s) already; the potential may "
                "approach its threshold too slowly (a long-range tail) for a finite "
                "box -- pass E_escape and a larger box explicitly.")
        warnings.warn(f"{n_unresolved} state(s) below the escape energy reach the edge "
                      f"of the box [{lo:.4g}, {hi:.4g}] m and were not counted as "
                      f"bound.  {hint}", UserWarning, stacklevel=2)

    h_fd = 1e-4 * (L if L is not None else (hi - lo) / 20.0)
    v3 = V_on(np.array([u_min - h_fd, u_min, u_min + h_fd]))
    curvature = (v3[0] - 2.0 * v3[1] + v3[2]) / h_fd ** 2
    omega = float(np.sqrt(curvature / m)) if curvature > 0 else float("nan")

    return AxisSpectrum(u=u, V=Vu, energies=evals, psi=psi_all[:, bound].copy(),
                        bound=bound, well_weight=well_w, edge_weight=edge_w,
                        u_min=u_min, V_min=V_min, E_escape=E_esc, u_lip=u_lip,
                        omega=omega, mass=m, n_unresolved=n_unresolved,
                        box_growths=growths)


def check_convergence(V, spectrum: AxisSpectrum, grow: float = 1.3,
                      refine: float = 1.5, **solve_kwargs) -> dict:
    """Re-solve on a box ``grow`` x larger and a grid ``refine`` x finer.

    Returns ``max_dE`` (J, over the bound states both solutions share), the two
    bound counts, and the reference ``AxisSpectrum``.  The notebook's own check
    finds ~1e-7 E_w for our tweezer; the count may legitimately differ for a
    weakly bound top state.
    """
    lo, hi = spectrum.bounds
    c = spectrum.u_min
    new = (c - grow * (c - lo), c + grow * (hi - c))
    n = int(np.ceil((new[1] - new[0]) / (spectrum.du / refine))) + 1
    ref = solve_axis(V, bounds=new, n_grid=n, mass=spectrum.mass, center=c,
                     auto_box=False, **solve_kwargs)
    k = min(spectrum.n_bound, ref.n_bound)
    dE = (float(np.max(np.abs(ref.bound_energies[:k] - spectrum.bound_energies[:k])))
          if k else float("nan"))
    return dict(max_dE=dE, n_bound=(spectrum.n_bound, ref.n_bound), reference=ref)
