"""Optical dipole traps: :class:`Trap`, the potential a set of beams makes for one state.

``Trap(beams, state=..., B_gauss=..., gravity=...)`` turns a
:class:`~kamo.trap.beams.Beam`, a :class:`~kamo.trap.beams.Crossed` or a list of
beams into the potential energy::

    U(r) = sum_b  -alpha_b I_b(r) / (2 c eps0 n_b)  -  m g (r . g_hat)

Each beam carries its **own** polarizability ``alpha_b`` -- its own wavelength
and its own polarization, resolved about the field direction ``B_hat`` (see
:mod:`kamo.trap.polarizability`).  Summing intensities and multiplying by one
alpha is wrong for any multi-colour or multi-polarization trap.

Everything else is derived from ``U``:

* :meth:`Trap.minimum` -- the gravity-sagged minimum.  A coarse grid on the
  *light* potential (gravity-free, so a weak trap's box corner cannot win) gives
  the start; BFGS on the full potential with the analytic gradient, then Newton
  steps, give the minimum to machine precision.
* :meth:`Trap.trap_frequencies` -- the Hessian there, from central differences
  of the analytic gradient with Richardson extrapolation, diagonalized:
  principal frequencies and axes (for a crossed trap they are not lab axes).
  A negative curvature gives NaN and a warning, never an exception: parameter
  scans walk through unbound configurations routinely.
* :attr:`Trap.sag` -- the minimum with gravity minus the minimum without.
* :meth:`Trap.trap_depth_J` -- the escape barrier.  Straight rays from the
  minimum (sampled geometrically out to 1000x the largest beam scale, so a
  Lorentzian axial tail is not cut short) locate the escape route; Newton on
  ``grad U = 0`` from the best ray's highest point then finds the first-order
  saddle, whose energy is the depth.  For a horizontal tweezer against gravity
  that saddle is *not* below the focus: it sits about a Rayleigh range along
  the beam axis, where the beam is weaker and wider, and the point straight
  below the focus is a second-order stationary point.  A 1D vertical cut (the
  gaussian_well notebook) therefore overestimates the 3D depth -- by 11% for
  the 3 um, 1 kHz tweezer.  :meth:`Trap.escape_barrier_J` gives the barrier
  along any one straight ray.
* :meth:`Trap.harmonic` -- the quadratic expansion at the minimum, a
  :class:`HarmonicTrap` with the same interface, so a solver can run the
  harmonic limit and the full potential through one code path.

The single-beam formulas of ``GaussianBeam.trap_frequency`` are the test oracle
for the Hessian, not the implementation: they do not generalize to crossed
geometries, sagged minima or elliptical beams.

Conventions and approximations
------------------------------
SI throughout.  ``potential_J`` is signed and attractive (negative at a
red-detuned focus), unlike ``GaussianBeam.trap_depth``.  The field is uniform:
the Zeeman energy is a constant, so B enters only through its direction, the
quantization axis -- and ``|F, mF>`` is the right basis only when the Zeeman
splitting dominates the vector light shift (``B_gradient_G_per_cm`` is the named
extension point for magnetic levitation).  Second-order AC Stark shift only: no
saturation, no scattering, no hyperfine mixing.  Construction never touches ARC
or the network; the atom and its polarizability are built on first use, and
``polarizability_SI=`` bypasses them entirely.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

import kamo.constants as kc

from . import frames as fr
from .beams import Beam, Crossed, IntensityField, InterferenceWarning
from .polarizability import StatePolarizability

G_EARTH = 9.80665               #: standard gravity, m/s^2


# ------------------------------------------------------------------ results

@dataclass
class TrapMinimum:
    """The (sagged) minimum of a trap."""

    position: np.ndarray        #: (3,) lab position (m)
    potential_J: float          #: potential there (J)
    converged: bool             #: optimizer settled inside the trap, Hessian positive definite


@dataclass
class TrapFrequencies:
    """Principal harmonic frequencies at the minimum."""

    frequencies_Hz: np.ndarray  #: (3,) descending; NaN where the curvature is not positive
    axes: np.ndarray            #: (3, 3); row i is the principal axis of frequencies_Hz[i]
    hessian: np.ndarray         #: (3, 3) second derivatives at ``position`` (J/m^2)
    position: np.ndarray        #: (3,) where the expansion was taken (m)
    mass: float                 #: atomic mass (kg)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * self.frequencies_Hz

    @property
    def is_bound(self) -> bool:
        return bool(np.all(np.isfinite(self.frequencies_Hz)))

    @property
    def omega_bar(self) -> float:
        """Geometric-mean angular frequency, what the BEC formulas want (rad/s)."""
        return float(np.prod(self.omega) ** (1.0 / 3.0))

    @property
    def oscillator_lengths(self) -> np.ndarray:
        """``sqrt(hbar / (m omega_i))`` along each principal axis (m)."""
        return np.sqrt(kc.hbar / (self.mass * self.omega))


class TrapPotential(Protocol):
    """What the density solvers consume: :class:`Trap` and :class:`HarmonicTrap`."""

    mass: float

    def potential_J(self, X, Y, Z): ...
    def gradient(self, X, Y, Z): ...
    def minimum(self) -> TrapMinimum: ...
    def trap_frequencies(self) -> TrapFrequencies: ...
    def trap_depth_J(self) -> float: ...
    def suggested_axes(self, n=64, n_widths=4.0, T_K=None): ...


# ----------------------------------------------------------------- helpers

def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(int(n)) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azim = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar),
                     np.cos(polar)], axis=1)


def _frequencies_from_hessian(H: np.ndarray, position, mass: float) -> TrapFrequencies:
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    with np.errstate(invalid="ignore"):
        f = np.where(evals > 0, np.sqrt(np.abs(evals) / mass) / (2.0 * np.pi), np.nan)
    return TrapFrequencies(frequencies_Hz=f, axes=evecs.T.copy(), hessian=H.copy(),
                           position=np.array(position, dtype=float), mass=float(mass))


def _axes_from_frequencies(tf: TrapFrequencies, n, n_widths: float, T_K):
    if not tf.is_bound:
        raise ValueError("the trap is not bound: no harmonic widths to size a grid from")
    n = (int(n),) * 3 if np.ndim(n) == 0 else tuple(int(x) for x in n)
    if T_K is None:
        var = kc.hbar / (2.0 * tf.mass * tf.omega)                      # ground state
    else:
        var = np.maximum(kc.kB * float(T_K) / (tf.mass * tf.omega ** 2),
                         kc.hbar / (2.0 * tf.mass * tf.omega))           # thermal
    var_lab = (tf.axes ** 2).T @ var            # sigma_i^2 = sum_k R_ki^2 var_k
    half = float(n_widths) * np.sqrt(var_lab)
    return tuple(np.linspace(c - hw, c + hw, m) for c, hw, m in zip(tf.position, half, n))


# --------------------------------------------------------------------- Trap

class Trap:
    """The optical potential of one or more beams for one atomic state.

    Parameters
    ----------
    beams : Beam, Crossed, or sequence of beams
    atom : Potassium39, optional
        Supplies the mass and, via ``use_portal``, the data source.  Never
        required: omit it and nothing is built until a polarizability is needed.
    state : (n, l, j, F, mF)
        Hyperfine state, default ``|4S1/2, F=1, mF=-1>``.
    B_gauss : float or (3,)
        Field magnitude in Gauss along ``B_direction`` (default ``z``), or the
        field vector itself.  Only its direction matters (the quantization axis).
    source : {"portal", "arc"}, optional
        Matrix-element source; default from ``atom.use_portal``, else "portal".
    mass : float, optional
        Atomic mass (kg); default ``atom.mass`` or ``kamo.constants.m_K``.
    gravity : bool
        Include ``-m g (r . g_hat)`` (default True).
    gravity_direction : (3,)
        The direction gravity *pulls*; default ``-z``.
    polarizability_SI : float or sequence, optional
        Bypass the atom: one polarizability (C m^2/V) for every beam, or one per
        beam.
    """

    def __init__(self, beams, *, atom=None, state=(4, 0, 0.5, 1, -1), B_gauss=0.0,
                 B_direction=None, source: Optional[str] = None,
                 mass: Optional[float] = None, nuclear_spin: Optional[float] = None,
                 gravity: bool = True, gravity_direction=(0.0, 0.0, -1.0),
                 g: float = G_EARTH, polarizability_SI=None,
                 label: Optional[str] = None):
        self._init = dict(atom=atom, state=state, B_gauss=B_gauss, B_direction=B_direction,
                          source=source, mass=mass, nuclear_spin=nuclear_spin,
                          gravity=gravity, gravity_direction=gravity_direction, g=g,
                          polarizability_SI=polarizability_SI, label=label)
        self.field = beams if isinstance(beams, IntensityField) else Crossed(list(beams))
        self._beams = self.field.beams

        use_portal = getattr(atom, "use_portal", None)
        if source is None:
            source = "arc" if use_portal is False else "portal"
        elif use_portal is not None and bool(use_portal) != (source == "portal"):
            raise ValueError(f"source={source!r} disagrees with atom.use_portal="
                             f"{use_portal}; give one or the other.")
        self.atom = atom
        self.source = source
        self._sp = StatePolarizability(state, source, nuclear_spin, atom=atom)
        self.state = self._sp.state
        if mass is not None and not (np.isfinite(float(mass)) and float(mass) > 0):
            raise ValueError(f"mass must be positive; got {mass}")
        self._mass = None if mass is None else float(mass)

        B = np.asarray(B_gauss, dtype=float)
        if B.ndim == 0:
            bhat = fr.as_unit_real_vector((0.0, 0.0, 1.0) if B_direction is None
                                          else B_direction, "B_direction")
            mag = float(B)
            if mag < 0:
                mag, bhat = -mag, -bhat
        elif B.shape == (3,):
            if B_direction is not None:
                raise ValueError("give B_gauss as a 3-vector, or a magnitude with "
                                 "B_direction -- not both.")
            mag = float(np.linalg.norm(B))
            bhat = B / mag if mag > 0 else np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"B_gauss must be a scalar or a 3-vector; got shape {B.shape}")
        if not np.isfinite(mag):
            raise ValueError("B_gauss must be finite")
        self.B_gauss = mag
        self._bhat = np.array(bhat, dtype=float)
        self._bhat.flags.writeable = False

        self.gravity = bool(gravity)
        self.g = float(g)
        if not self.g >= 0:
            raise ValueError(f"g must be >= 0; got {g}")
        self._ghat = fr.as_unit_real_vector(gravity_direction, "gravity_direction")
        self._ghat.flags.writeable = False
        self._ghat_f = tuple(float(x) for x in self._ghat)

        if polarizability_SI is None:
            self._alpha, self._alpha_given = None, False
        else:
            a = np.atleast_1d(np.asarray(polarizability_SI, dtype=float)).ravel()
            if a.size == 1:
                a = np.repeat(a, len(self._beams))
            if a.size != len(self._beams):
                raise ValueError(f"polarizability_SI: give one value, or one per beam "
                                 f"({len(self._beams)}); got {a.size}.")
            if not np.all(np.isfinite(a)):
                raise ValueError("polarizability_SI must be finite")
            self._alpha, self._alpha_given = tuple(float(x) for x in a), True
        self.label = label
        self._minimum: Optional[TrapMinimum] = None
        self._frequencies: Optional[TrapFrequencies] = None
        self._depth: Optional[float] = None
        self._escape_direction: Optional[np.ndarray] = None
        self._saddle: Optional[np.ndarray] = None

    # --------------------------------------------------------- attributes
    @property
    def mass(self) -> float:
        if self._mass is None:
            m = getattr(self.atom, "mass", None)
            self._mass = float(m if m is not None else kc.m_K)
        return self._mass

    @property
    def B_hat(self) -> np.ndarray:
        return self._bhat.copy()

    @property
    def g_hat(self) -> np.ndarray:
        return self._ghat.copy()

    @property
    def beams(self) -> tuple:
        return self._beams

    def alphas(self) -> tuple:
        """Polarizability (C m^2/V) of this state in each beam."""
        if self._alpha is None:
            circular = any(np.linalg.norm(np.imag(np.cross(b.polarization,
                                                           np.conj(b.polarization)))) > 1e-12
                           for b in self._beams)
            if self.B_gauss == 0.0 and (circular or self.state[2] > 0.5):
                warnings.warn(
                    "B = 0: the quantization axis is undefined, so the vector/tensor "
                    f"light shifts are taken about B_direction {self._bhat.tolist()}.  "
                    "They mean something only when the Zeeman splitting dominates the "
                    "vector light shift; pass the real field.", UserWarning, stacklevel=2)
            self._alpha = tuple(self._sp.alpha_SI(b.wavelength_m, b.polarization, self._bhat)
                                for b in self._beams)
        return self._alpha

    # ---------------------------------------------------------- potential
    def light_potential_J(self, X, Y, Z):
        """The AC Stark potential alone (J); broadcasts, numpy or torch."""
        total = None
        for b, a in zip(self._beams, self.alphas()):
            term = kc.ac_stark_shift_J(a, b.intensity(X, Y, Z), b.n_medium)
            total = term if total is None else total + term
        return total

    def gravity_potential_J(self, X, Y, Z):
        """``-m g (r . g_hat)`` (J); 0 with gravity off."""
        if not self.gravity:
            return 0.0
        gx, gy, gz = self._ghat_f
        return -self.mass * self.g * (gx * X + gy * Y + gz * Z)

    def potential_J(self, X, Y, Z):
        """Total potential (J), signed: negative at a red-detuned focus."""
        return self.light_potential_J(X, Y, Z) + self.gravity_potential_J(X, Y, Z)

    def potential_Hz(self, X, Y, Z):
        return self.potential_J(X, Y, Z) / kc.h

    def potential_K(self, X, Y, Z):
        return self.potential_J(X, Y, Z) / kc.kB

    def potential_uK(self, X, Y, Z):
        return self.potential_J(X, Y, Z) / kc.kB * 1e6

    def gradient(self, X, Y, Z):
        """Analytic ``(dU/dx, dU/dy, dU/dz)`` (J/m); broadcasts, numpy or torch."""
        gx = gy = gz = 0.0
        for b, a in zip(self._beams, self.alphas()):
            s = kc.ac_stark_shift_J(a, 1.0, b.n_medium)          # J per W/m^2
            dx, dy, dz = b.intensity_gradient(X, Y, Z)
            gx, gy, gz = gx + s * dx, gy + s * dy, gz + s * dz
        if self.gravity:
            mg = self.mass * self.g
            hx, hy, hz = self._ghat_f
            gx, gy, gz = gx - mg * hx, gy - mg * hy, gz - mg * hz
        return gx, gy, gz

    def differential_potential_J(self, other_state, X, Y, Z):
        """``U_other - U_this`` (J): the light-shift difference; gravity cancels."""
        return (self.for_state(other_state).light_potential_J(X, Y, Z)
                - self.light_potential_J(X, Y, Z))

    def hessian(self, position, step: Optional[float] = None) -> np.ndarray:
        """Second derivatives (J/m^2) at ``position``: central differences of the
        analytic gradient at ``h`` and ``h/2``, Richardson-extrapolated, symmetrized."""
        x = np.asarray(position, dtype=float)
        h = 1e-3 * self._scales()[0] if step is None else float(step)

        def central(hh):
            H = np.empty((3, 3))
            for i in range(3):
                e = np.zeros(3)
                e[i] = hh
                gp = np.array(self.gradient(*(x + e)), dtype=float)
                gm = np.array(self.gradient(*(x - e)), dtype=float)
                H[:, i] = (gp - gm) / (2.0 * hh)
            return 0.5 * (H + H.T)

        return (4.0 * central(0.5 * h) - central(h)) / 3.0

    # --------------------------------------------------------- the minimum
    def _scales(self):
        """(smallest waist, largest waist, largest of waists and Rayleigh ranges)."""
        w_lo = min(min(b.waist_u, b.waist_v) for b in self._beams)
        w_hi = max(max(b.waist_u, b.waist_v) for b in self._beams)
        z_hi = max(max(b.rayleigh_range_u, b.rayleigh_range_v) for b in self._beams)
        return w_lo, w_hi, max(w_hi, z_hi)

    def minimum(self) -> TrapMinimum:
        """The (gravity-sagged) minimum; cached."""
        if self._minimum is None:
            self._minimum = self._find_minimum()
        return self._minimum

    def _find_minimum(self) -> TrapMinimum:
        w_lo, w_hi, _ = self._scales()
        origins = np.array([b.origin for b in self._beams])
        lo, hi = origins.min(axis=0) - 2.0 * w_hi, origins.max(axis=0) + 2.0 * w_hi
        ax = [np.linspace(lo[i], hi[i], 41) for i in range(3)]
        UL = self.light_potential_J(ax[0][:, None, None], ax[1][None, :, None],
                                    ax[2][None, None, :])
        idx = np.unravel_index(int(np.argmin(UL)), UL.shape)
        x0 = np.array([ax[i][idx[i]] for i in range(3)])
        scale = w_lo
        U_scale = abs(float(self.light_potential_J(*x0))) or kc.kB * 1e-9

        def objective(q):
            r = x0 + q * scale
            return (float(self.potential_J(*r)) / U_scale,
                    np.array(self.gradient(*r), dtype=float) * scale / U_scale)

        res = minimize(objective, np.zeros(3), jac=True, method="BFGS",
                       options=dict(gtol=1e-11, maxiter=2000))
        x = x0 + res.x * scale
        for _ in range(4):                                   # Newton polish
            H = self.hessian(x)
            grad = np.array(self.gradient(*x), dtype=float)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)) or np.linalg.norm(step) > w_lo:
                break
            x = x - step
            if np.linalg.norm(step) < 1e-13 * scale:
                break
        H = self.hessian(x)
        converged = (bool(np.all(np.linalg.eigvalsh(H) > 0))
                     and bool(np.linalg.norm(x - x0) < 10.0 * w_hi))
        return TrapMinimum(position=x, potential_J=float(self.potential_J(*x)),
                           converged=converged)

    @property
    def sag(self) -> np.ndarray:
        """Minimum with gravity minus minimum without (m)."""
        if not self.gravity:
            return np.zeros(3)
        return self.minimum().position - self.without_gravity().minimum().position

    @property
    def sag_along_gravity(self) -> float:
        """How far the minimum moved in the direction gravity pulls (m)."""
        return float(self.sag @ self._ghat)

    # ------------------------------------------------- convenience views
    @property
    def minimum_position(self) -> np.ndarray:
        """Lab position of the (sagged) minimum (m)."""
        return self.minimum().position.copy()

    @property
    def frequencies_Hz(self) -> np.ndarray:
        """Principal trap frequencies, descending (Hz); NaN where unbound."""
        return self.trap_frequencies().frequencies_Hz.copy()

    @property
    def principal_axes(self) -> np.ndarray:
        """(3, 3): row i is the unit vector of ``frequencies_Hz[i]``."""
        return self.trap_frequencies().axes.copy()

    @property
    def depth_J(self) -> float:
        """Escape depth above the minimum (J); see :meth:`trap_depth_J`."""
        return self.trap_depth_J()

    @property
    def depth_Hz(self) -> float:
        return self.trap_depth_J() / kc.h

    @property
    def depth_K(self) -> float:
        return self.trap_depth_J() / kc.kB

    @property
    def depth_uK(self) -> float:
        return self.trap_depth_J() / kc.kB * 1e6

    # ---------------------------------------------------- frequencies, depth
    def trap_frequencies(self) -> TrapFrequencies:
        """Principal frequencies and axes at the minimum; cached."""
        if self._frequencies is None:
            m = self.minimum()
            if not m.converged:
                warnings.warn("the trap has no minimum here (gravity wins, or nothing "
                              "confines): frequencies are NaN.", UserWarning, stacklevel=2)
                self._frequencies = TrapFrequencies(np.full(3, np.nan), np.eye(3),
                                                    np.full((3, 3), np.nan), m.position,
                                                    self.mass)
            else:
                self._frequencies = _frequencies_from_hessian(self.hessian(m.position),
                                                              m.position, self.mass)
                if not self._frequencies.is_bound:
                    warnings.warn("negative curvature at the minimum: some frequencies "
                                  "are NaN.", UserWarning, stacklevel=2)
        return self._frequencies

    def _ray_directions(self, n: int) -> np.ndarray:
        special = [np.eye(3), -np.eye(3), self._ghat[None], -self._ghat[None]]
        for b in self._beams:
            f = np.array(b.frame)
            special += [f, -f]
        return np.concatenate([_fibonacci_sphere(n)] + special, axis=0)

    def _ray_grid(self, n_radial, r_max) -> np.ndarray:
        w_lo, _, L_hi = self._scales()
        return np.geomspace(1e-3 * w_lo, 1e3 * L_hi if r_max is None else float(r_max),
                            int(n_radial))

    def _ray_barrier(self, r0, direction, t):
        """Highest point on a straight ray from ``r0``: ``(U_max, t_max, interior)``."""
        n = direction / np.linalg.norm(direction)
        U = self.potential_J(r0[0] + n[0] * t, r0[1] + n[1] * t, r0[2] + n[2] * t)
        k = int(np.argmax(U))
        if k == 0 or k == t.size - 1:
            return float(U[k]), float(t[k]), False
        res = minimize_scalar(
            lambda s: -float(self.potential_J(*(r0 + n * s))),
            bounds=(t[k - 1], t[k + 1]), method="bounded",
            options=dict(xatol=(t[k + 1] - t[k - 1]) * 1e-10))
        if -float(res.fun) >= float(U[k]):
            return -float(res.fun), float(res.x), True
        return float(U[k]), float(t[k]), True

    def _saddle_near(self, r_start):
        """Newton on ``grad U = 0`` from ``r_start``: ``(position, U)`` of a
        first-order saddle, or None if it lands on anything else."""
        w_lo = self._scales()[0]
        x = np.array(r_start, dtype=float)
        for _ in range(100):
            g = np.array(self.gradient(*x), dtype=float)
            try:
                step = np.linalg.solve(self.hessian(x), g)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(step)):
                return None
            size = float(np.linalg.norm(step))
            if size > 0.25 * w_lo:
                step *= 0.25 * w_lo / size
            x = x - step
            if size < 1e-13 * w_lo:
                break
        force = abs(self.minimum().potential_J) / w_lo
        if (np.linalg.norm(np.array(self.gradient(*x), dtype=float)) > 1e-8 * force
                or int(np.sum(np.linalg.eigvalsh(self.hessian(x)) < 0)) != 1):
            return None
        return x, float(self.potential_J(*x))

    def trap_depth_J(self, n_directions: int = 256, n_radial: int = 400,
                     r_max: Optional[float] = None, refine: bool = True) -> float:
        """Escape barrier above the minimum (J); 0 if there is no minimum.

        Rays locate the escape route; with ``refine`` the best ray's direction is
        polished and Newton on ``grad U = 0`` from its highest point finds the
        first-order saddle itself, whose energy is the depth.  Without one (no
        gravity: the well flattens toward 0 at infinity) the ray bound is used.
        Cached for the default arguments.
        """
        default = (n_directions, n_radial, r_max, refine) == (256, 400, None, True)
        if default and self._depth is not None:
            return self._depth
        m = self.minimum()
        if not m.converged:
            return 0.0
        r0, U0 = m.position, m.potential_J
        t = self._ray_grid(n_radial, r_max)
        dirs = self._ray_directions(n_directions)
        U = self.potential_J(r0[0] + dirs[:, 0:1] * t, r0[1] + dirs[:, 1:2] * t,
                             r0[2] + dirs[:, 2:3] * t)
        j = int(np.argmin(np.max(U, axis=1)))
        best_dir = dirs[j] / np.linalg.norm(dirs[j])
        barrier, t_max, interior = self._ray_barrier(r0, best_dir, t)
        saddle = None
        if refine:
            e1, e2, _ = fr.orthonormal_frame(best_dir)

            def along(ab):
                return self._ray_barrier(r0, best_dir + ab[0] * e1 + ab[1] * e2, t)[0]

            res = minimize(along, np.zeros(2), method="Nelder-Mead",
                           options=dict(xatol=1e-7, fatol=abs(U0) * 1e-13,
                                        initial_simplex=[[0, 0], [0.05, 0], [0, 0.05]]))
            if res.fun < barrier:
                d = best_dir + res.x[0] * e1 + res.x[1] * e2
                best_dir = d / np.linalg.norm(d)
                barrier, t_max, interior = self._ray_barrier(r0, best_dir, t)
            if interior:
                found = self._saddle_near(r0 + best_dir * t_max)
                if found is not None and found[1] <= barrier + 1e-12 * abs(U0):
                    saddle, barrier = found
                    best_dir = (saddle - r0) / np.linalg.norm(saddle - r0)
        depth = barrier - U0
        if default:
            self._depth, self._escape_direction, self._saddle = depth, best_dir, saddle
        return depth

    def escape_barrier_J(self, direction, n_radial: int = 400,
                         r_max: Optional[float] = None) -> float:
        """Barrier above the minimum along one straight ray from it (J) -- straight
        down, for instance, is the lip of a 1D vertical cut."""
        m = self.minimum()
        if not m.converged:
            return 0.0
        d = fr.as_unit_real_vector(direction, "direction")
        return (self._ray_barrier(m.position, d, self._ray_grid(n_radial, r_max))[0]
                - m.potential_J)

    @property
    def escape_direction(self) -> Optional[np.ndarray]:
        """Unit vector from the minimum toward the escape route (after trap_depth_J)."""
        if self._depth is None:
            self.trap_depth_J()
        return None if self._escape_direction is None else self._escape_direction.copy()

    @property
    def escape_saddle(self) -> Optional[np.ndarray]:
        """The first-order saddle atoms escape through (computed by trap_depth_J),
        or None when the escape is over the asymptote."""
        if self._depth is None:
            self.trap_depth_J()
        return None if self._saddle is None else self._saddle.copy()

    def is_bound(self) -> bool:
        """A minimum exists, every curvature is positive, and the depth is positive."""
        m = self.minimum()
        if not m.converged:
            return False
        return bool(self.trap_frequencies().is_bound and self.trap_depth_J() > 0)

    # --------------------------------------------------- derived potentials
    def harmonic(self) -> "HarmonicTrap":
        """The quadratic expansion of this trap at its minimum."""
        m = self.minimum()
        return HarmonicTrap(m.position, m.potential_J, self.hessian(m.position), self.mass)

    def suggested_axes(self, n=64, n_widths: float = 4.0, T_K: Optional[float] = None):
        """Three 1D lab axes centred on the minimum, spanning ``+- n_widths``
        harmonic ground-state (or, with ``T_K``, thermal) widths.  Broadcast
        them (``x[:, None, None]``) rather than meshgrid them."""
        return _axes_from_frequencies(self.trap_frequencies(), n, n_widths, T_K)

    # ------------------------------------------------------------ variants
    def _rebuild(self, field=None, carry_alpha: bool = False, **changes) -> "Trap":
        kw = dict(self._init)
        kw.update(changes)
        new = Trap(self.field if field is None else field, **kw)
        if carry_alpha and self._alpha is not None and not new._alpha_given:
            new._alpha = self._alpha
        return new

    def without_gravity(self) -> "Trap":
        return self._rebuild(carry_alpha=True, gravity=False)

    def with_field(self, B_gauss, B_direction=None) -> "Trap":
        return self._rebuild(B_gauss=B_gauss, B_direction=B_direction)

    def for_state(self, state) -> "Trap":
        return self._rebuild(state=state)

    def rescaled(self, factor: float) -> "Trap":
        """Every beam's power multiplied by ``factor``."""
        beams = [b.scaled(factor) for b in self._beams]
        if isinstance(self.field, Beam):
            field = beams[0]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterferenceWarning)
                field = Crossed(beams, label=self.field.label,
                                min_frequency_offset_Hz=self.field.min_frequency_offset_Hz)
        return self._rebuild(field=field, carry_alpha=True)

    def _reference(self) -> "Trap":
        """This trap, or a 1 mW-per-beam copy if every beam is at 0 W -- so a
        trap can be specified by geometry alone and then scaled to a measured
        frequency or depth."""
        if any(b.power > 0 for b in self._beams):
            return self
        beams = [b.with_power(1e-3) for b in self._beams]
        field = beams[0] if isinstance(self.field, Beam) else Crossed(
            beams, label=self.field.label,
            min_frequency_offset_Hz=self.field.min_frequency_offset_Hz)
        return self._rebuild(field=field, carry_alpha=True)

    def _solve_factor(self, quantity, target: float, guess: float) -> "Trap":
        if not self.gravity:
            return self.rescaled(guess)

        def h(logk):
            with warnings.catch_warnings():       # the bracket probes powers gravity wins at
                warnings.simplefilter("ignore", UserWarning)
                v = quantity(self.rescaled(np.exp(logk)))
            return (v if np.isfinite(v) else -np.inf) - target

        a, b = np.log(guess) - 1.0, np.log(guess) + 1.0
        for _ in range(20):
            if h(a) < 0 < h(b):
                break
            a, b = a - 1.0, b + 1.0
        return self.rescaled(np.exp(brentq(h, a, b, xtol=1e-12)))

    def rescaled_to_frequency(self, f_Hz: float, index: int = 0) -> "Trap":
        """Same geometry, powers scaled so principal frequency ``index`` is ``f_Hz``
        (index 0 is the highest).  Works from a 0 W geometry."""
        ref = self._reference()
        f0 = ref.without_gravity().trap_frequencies().frequencies_Hz[index]
        return ref._solve_factor(lambda tr: tr.trap_frequencies().frequencies_Hz[index],
                                 float(f_Hz), (float(f_Hz) / f0) ** 2)

    def rescaled_to_depth(self, depth_K: float) -> "Trap":
        """Same geometry, powers scaled so the escape depth is ``depth_K`` (K).
        Works from a 0 W geometry."""
        target = float(depth_K) * kc.kB
        ref = self._reference()
        return ref._solve_factor(lambda tr: tr.trap_depth_J(), target,
                                 target / ref.without_gravity().trap_depth_J())

    # ------------------------------------------------------------- report
    def summary(self) -> str:
        """Human-readable report; energies as E/h and in uK."""
        h = kc.h
        lab = f" {self.label!r}" if self.label else ""
        lines = [f"Trap{lab}: {len(self._beams)} beam(s), state {self.state}, "
                 f"B = {self.B_gauss:.4g} G along {np.round(self._bhat, 4).tolist()}, "
                 f"gravity {'on' if self.gravity else 'off'}"]
        for b, a in zip(self._beams, self.alphas()):
            lines.append(f"  {b!r}: alpha = {a / kc.convert_polarizability_au_to_SI:.3f} a.u.")
        if not self._alpha_given:
            lines.append("  " + self._sp.describe(self._beams[0].wavelength_m))
        m = self.minimum()
        if not m.converged:
            lines.append("  NO MINIMUM: the trap does not hold (gravity wins, or "
                         "nothing confines).")
            return "\n".join(lines)
        p = m.position * 1e6
        lines.append(f"  minimum at ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) um, "
                     f"U/h = {m.potential_J / h / 1e3:.4f} kHz")
        if self.gravity:
            lines.append(f"  gravitational sag {self.sag_along_gravity * 1e9:.2f} nm")
        f = self.trap_frequencies()
        lines.append("  principal frequencies "
                     + ", ".join(f"{x:.3f}" for x in f.frequencies_Hz) + " Hz")
        d = self.trap_depth_J()
        lines.append(f"  escape depth {d / h / 1e3:.4f} kHz = {d / kc.kB * 1e6:.4f} uK")
        if self._saddle is not None:
            q = self._saddle * 1e6
            lines.append(f"  escape saddle at ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}) um")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Trap({len(self._beams)} beam(s), state={self.state}, "
                f"B={self.B_gauss:.4g} G, gravity={self.gravity})")


# ----------------------------------------------------------- HarmonicTrap

class HarmonicTrap:
    """``U(r) = U_min + (r - r0)^T H (r - r0) / 2``: a trap's quadratic expansion."""

    def __init__(self, position, potential_J: float, hessian, mass: float):
        self._r0 = tuple(float(x) for x in position)
        self.U_min = float(potential_J)
        H = np.asarray(hessian, dtype=float)
        H = 0.5 * (H + H.T)
        self.hessian_matrix = H
        self._xx, self._xy, self._xz = float(H[0, 0]), float(H[0, 1]), float(H[0, 2])
        self._yy, self._yz, self._zz = float(H[1, 1]), float(H[1, 2]), float(H[2, 2])
        self.mass = float(mass)

    def potential_J(self, X, Y, Z):
        x0, y0, z0 = self._r0
        dx, dy, dz = X - x0, Y - y0, Z - z0
        return self.U_min + 0.5 * (self._xx * dx * dx + self._yy * dy * dy
                                   + self._zz * dz * dz
                                   + 2.0 * (self._xy * dx * dy + self._xz * dx * dz
                                            + self._yz * dy * dz))

    def gradient(self, X, Y, Z):
        x0, y0, z0 = self._r0
        dx, dy, dz = X - x0, Y - y0, Z - z0
        return (self._xx * dx + self._xy * dy + self._xz * dz,
                self._xy * dx + self._yy * dy + self._yz * dz,
                self._xz * dx + self._yz * dy + self._zz * dz)

    def minimum(self) -> TrapMinimum:
        return TrapMinimum(np.array(self._r0), self.U_min,
                           bool(np.all(np.linalg.eigvalsh(self.hessian_matrix) > 0)))

    def trap_frequencies(self) -> TrapFrequencies:
        return _frequencies_from_hessian(self.hessian_matrix, self._r0, self.mass)

    def trap_depth_J(self) -> float:
        return float("inf")

    def suggested_axes(self, n=64, n_widths: float = 4.0, T_K: Optional[float] = None):
        return _axes_from_frequencies(self.trap_frequencies(), n, n_widths, T_K)

    def __repr__(self) -> str:
        f = self.trap_frequencies().frequencies_Hz
        return f"HarmonicTrap(f={np.round(f, 3).tolist()} Hz)"
