"""Gaussian beams in the lab frame: :class:`Tweezer`, :class:`LightSheet`, :class:`Crossed`.

Every beam carries a lab-frame polarization and propagation direction (complex
and real 3-vectors), an origin (the centre of its waist) and a transverse frame
``(u_hat, v_hat, k_hat)``.  Its intensity is evaluated at lab coordinates
``intensity(X, Y, Z)`` -- three broadcastable arrays, matching
:meth:`kamo.imaging.bpm.SusceptibilitySource.density` and the broadcast grids of
:mod:`kamo.spin` -- so a beam drops straight into a 3D trap potential.

Specifying a beam
-----------------
Three independent groups; give **exactly one** keyword from each (the error
names the group and what was given):

========  ==========================================================  ==========
group     keywords                                                    default
========  ==========================================================  ==========
colour    ``wavelength_m`` | ``frequency_Hz``                          required
size      ``waist`` | ``NA`` | ``rayleigh_range`` | ``divergence_angle``  required
strength  ``power`` | ``peak_intensity``                               0 W
========  ==========================================================  ==========

``NA`` is the paraxial Gaussian divergence ``n theta = lambda / (pi w0)``, not a
hard aperture; a truncated aperture of the same NA gives a waist 10-20% larger.
For an objective with a real aperture use :class:`kamo.trap.thin_lens.Objective`.
Trap depth is deliberately *not* a beam specifier -- it needs an atom, a state
and a field direction -- and lives on :class:`kamo.trap.Trap` instead.
:meth:`Tweezer.from_trap_frequency` is the inverse: it solves for the strength
given a radial frequency, with the polarizability from :class:`kamo.Potassium39`
by default or passed in explicitly.

Beams are immutable: ``with_power``, ``with_peak_intensity``, ``scaled``,
``moved_to`` and ``rotated`` return new objects.

Approximations
--------------
Fundamental Gaussian modes, paraxial.  No Gouy phase, no wavefront curvature,
and no longitudinal field: a tight tweezer has ``|E_s| / |E_t| ~ NA``, so at
NA 0.5 the polarization is ~10% non-transverse and a vector light shift built
from the transverse polarization is off at that level.  :class:`Crossed` adds
intensities incoherently (see :class:`InterferenceWarning`).
"""

from __future__ import annotations

import functools
import types
import warnings
from typing import Optional

import numpy as np

import kamo.constants as kc
from kamo.imaging._backend import array_exp, array_sqrt

from . import frames as fr

DEFAULT_POLARIZATION = (0.0, 1.0, 1.0j)
DEFAULT_PROPAGATION = (1.0, 0.0, 0.0)
WAVELENGTH_RANGE_M = (100e-9, 20e-6)


class InterferenceWarning(UserWarning):
    """Two beams of a Crossed trap could interfere, but are summed incoherently."""


# ----------------------------------------------------------------- helpers

def _exactly_one(group: str, **kw):
    given = {k: v for k, v in kw.items() if v is not None}
    if len(given) != 1:
        raise ValueError(f"{group}: give exactly one of {sorted(kw)}; "
                         f"got {sorted(given) or 'none'}.")
    return next(iter(given.items()))


def _positive(name: str, x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number; got {x!r}") from None
    if not (np.isfinite(x) and x > 0):
        raise ValueError(f"{name} must be positive and finite; got {x}")
    return x


def _resolve_wavelength(wavelength_m, frequency_Hz) -> float:
    key, val = _exactly_one("colour", wavelength_m=wavelength_m, frequency_Hz=frequency_Hz)
    lam = _positive(key, val)
    if key == "frequency_Hz":
        lam = kc.c / lam
    lo, hi = WAVELENGTH_RANGE_M
    if not lo <= lam <= hi:
        raise ValueError(f"wavelength {lam:.4g} m is outside [{lo:g}, {hi:g}] m, the range "
                         "kamo.light_shift's polarizabilities cover.")
    return lam


def _waist_from(key: str, value, wavelength: float, n_medium: float) -> float:
    """Convert one size specifier to a 1/e^2 intensity waist (m)."""
    v = _positive(key, value)
    if key == "waist":
        return v
    if key == "NA":
        if v >= n_medium:
            raise ValueError(f"NA = {v} must be below n_medium = {n_medium}")
        return wavelength / (np.pi * v)
    if key == "rayleigh_range":
        return float(np.sqrt(wavelength * v / (np.pi * n_medium)))
    if key == "divergence_angle":
        return wavelength / (np.pi * n_medium * v)
    raise ValueError(f"unknown size specifier {key!r}")


def _size_values(value, allow_pair: bool):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return float(a), float(a)
    if allow_pair and a.shape == (2,):
        return float(a[0]), float(a[1])
    raise ValueError("a Tweezer is radially symmetric: give one scalar size (use "
                     "LightSheet for two waists)" if not allow_pair else
                     f"a LightSheet size is a scalar or a (u, v) pair; got shape {a.shape}")


# ------------------------------------------------------------ the interface

class IntensityField:
    """Anything with a lab-frame intensity: a :class:`Beam` or a :class:`Crossed`."""

    @property
    def beams(self) -> tuple:
        raise NotImplementedError

    def intensity(self, X, Y, Z):
        raise NotImplementedError

    def intensity_gradient(self, X, Y, Z):
        raise NotImplementedError

    @property
    def characteristic_size(self) -> float:
        raise NotImplementedError

    def intensity_at(self, points):
        """Intensity at an ``(..., 3)`` array of lab points."""
        p = np.asarray(points, dtype=float)
        return self.intensity(p[..., 0], p[..., 1], p[..., 2])

    def __add__(self, other):
        if not isinstance(other, IntensityField):
            return NotImplemented
        return Crossed([self, other])

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:    # sum() starts from 0
            return self
        if not isinstance(other, IntensityField):
            return NotImplemented
        return Crossed([other, self])


# --------------------------------------------------------------------- Beam


class _hybridmethod:
    """A method that binds to the instance when called on one and to the class
    otherwise (``Tweezer.from_trap_frequency(...)`` builds the geometry from
    keywords; ``tweezer.from_trap_frequency(...)`` reuses the beam's)."""

    def __init__(self, func):
        self.__func__ = func
        functools.update_wrapper(self, func)

    def __get__(self, obj, objtype=None):
        return types.MethodType(self.__func__, obj if obj is not None else objtype)

class Beam(IntensityField):
    """A fundamental Gaussian mode with an elliptical waist, in the lab frame.

    Use :class:`Tweezer` or :class:`LightSheet`; this base takes the two 1/e^2
    waists directly.  ``(u_hat, v_hat, k_hat)`` is right-handed, with ``u_hat``
    along ``transverse_axis`` (projected perpendicular to ``k_hat``) when given.
    """

    def __init__(self, waist_u, waist_v, *, wavelength_m=None, frequency_Hz=None,
                 power=None, peak_intensity=None, n_medium=1.0,
                 polarization=DEFAULT_POLARIZATION,
                 propagation_direction=DEFAULT_PROPAGATION,
                 origin=(0.0, 0.0, 0.0), transverse_axis=None,
                 transversality: str = "raise", label: Optional[str] = None):
        self._lam = _resolve_wavelength(wavelength_m, frequency_Hz)
        self._n = _positive("n_medium", n_medium)
        self._w_u = _positive("waist_u", waist_u)
        self._w_v = _positive("waist_v", waist_v)
        if power is not None and peak_intensity is not None:
            raise ValueError("strength: give exactly one of ['peak_intensity', 'power'] "
                             "(or neither, for 0 W); got both.")
        if peak_intensity is not None:
            P = _positive("peak_intensity", peak_intensity) * np.pi * self._w_u * self._w_v / 2.0
        else:
            P = 0.0 if power is None else float(power)
            if not (np.isfinite(P) and P >= 0):
                raise ValueError(f"power must be finite and >= 0; got {power}")
        self._P = float(P)

        k_hat = fr.as_unit_real_vector(propagation_direction, "propagation_direction")
        u_hat, v_hat, k_hat = fr.orthonormal_frame(k_hat, first=transverse_axis)
        try:
            eps = fr.check_transverse(polarization, k_hat, mode=transversality)
        except ValueError as err:
            if polarization is DEFAULT_POLARIZATION and "not transverse" in str(err):
                raise ValueError(
                    f"{err}  The default polarization (0, 1, i)/sqrt(2) is transverse "
                    "only to the default propagation direction x: pass a "
                    "polarization for this beam.") from None
            raise
        origin = np.array(origin, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError(f"origin must be a finite 3-vector (m); got {origin!r}")

        self._u, self._v, self._k, self._eps, self._origin = u_hat, v_hat, k_hat, eps, origin
        for a in (self._u, self._v, self._k, self._eps, self._origin):
            a.flags.writeable = False
        self._transversality = transversality
        self.label = label

        # Plain Python floats for the hot path: no numpy array enters intensity(),
        # so it runs unchanged on numpy arrays and torch tensors.
        self._uf = tuple(float(x) for x in u_hat)
        self._vf = tuple(float(x) for x in v_hat)
        self._kf = tuple(float(x) for x in k_hat)
        self._of = tuple(float(x) for x in origin)
        self._w_u2, self._w_v2 = self._w_u ** 2, self._w_v ** 2
        self._zr_u = np.pi * self._w_u2 * self._n / self._lam
        self._zr_v = np.pi * self._w_v2 * self._n / self._lam
        self._two_P_over_pi = 2.0 * self._P / np.pi

    # ------------------------------------------------------------ colour
    @property
    def wavelength_m(self) -> float:
        """Vacuum wavelength (m)."""
        return self._lam

    @property
    def wavelength(self) -> float:
        """Alias of :attr:`wavelength_m` (the attribute kamo.hamiltonian reads)."""
        return self._lam

    @property
    def frequency_Hz(self) -> float:
        return kc.c / self._lam

    @property
    def n_medium(self) -> float:
        return self._n

    @property
    def k(self) -> float:
        """Wavenumber in the medium, ``2 pi n / lambda`` (1/m)."""
        return 2.0 * np.pi * self._n / self._lam

    # -------------------------------------------------------------- size
    @property
    def waist_u(self) -> float:
        return self._w_u

    @property
    def waist_v(self) -> float:
        return self._w_v

    @property
    def rayleigh_range_u(self) -> float:
        return self._zr_u

    @property
    def rayleigh_range_v(self) -> float:
        return self._zr_v

    @property
    def divergence_angle_u(self) -> float:
        return self._lam / (np.pi * self._n * self._w_u)

    @property
    def divergence_angle_v(self) -> float:
        return self._lam / (np.pi * self._n * self._w_v)

    @property
    def NA_u(self) -> float:
        """Paraxial Gaussian NA, ``n theta = lambda / (pi w)``."""
        return self._lam / (np.pi * self._w_u)

    @property
    def NA_v(self) -> float:
        return self._lam / (np.pi * self._w_v)

    @property
    def characteristic_size(self) -> float:
        return max(self._w_u, self._w_v)

    # ---------------------------------------------------------- strength
    @property
    def power(self) -> float:
        return self._P

    @property
    def peak_intensity(self) -> float:
        """On-axis intensity at the waist, ``2 P / (pi w_u w_v)`` (W/m^2)."""
        return self._two_P_over_pi / (self._w_u * self._w_v)

    @property
    def I0(self) -> float:
        """Alias of :attr:`peak_intensity` (the attribute kamo.hamiltonian reads)."""
        return self.peak_intensity

    # ---------------------------------------------------------- geometry
    @property
    def polarization(self) -> np.ndarray:
        return self._eps.copy()

    @property
    def propagation_direction(self) -> np.ndarray:
        return self._k.copy()

    @property
    def origin(self) -> np.ndarray:
        return self._origin.copy()

    @property
    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(u_hat, v_hat, k_hat)``, right-handed."""
        return self._u.copy(), self._v.copy(), self._k.copy()

    @property
    def beams(self) -> tuple:
        return (self,)

    # --------------------------------------------------------- the field
    def _beam_coords(self, X, Y, Z):
        """Lab -> beam frame ``(u, v, s)``: nine Python floats, arithmetic only."""
        ox, oy, oz = self._of
        dx, dy, dz = X - ox, Y - oy, Z - oz
        (ux, uy, uz), (vx, vy, vz), (kx, ky, kz) = self._uf, self._vf, self._kf
        return (ux * dx + uy * dy + uz * dz,
                vx * dx + vy * dy + vz * dz,
                kx * dx + ky * dy + kz * dz)

    def intensity(self, X, Y, Z):
        """Intensity (W/m^2) at lab coordinates; broadcasts, numpy or torch."""
        u, v, s = self._beam_coords(X, Y, Z)
        wu2 = self._w_u2 * (1.0 + (s / self._zr_u) ** 2)
        wv2 = self._w_v2 * (1.0 + (s / self._zr_v) ** 2)
        return (self._two_P_over_pi / array_sqrt(wu2 * wv2)
                * array_exp(-2.0 * (u * u / wu2 + v * v / wv2)))

    def intensity_gradient(self, X, Y, Z):
        """Analytic lab-frame gradient ``(dI/dx, dI/dy, dI/dz)`` (W/m^3).

        With ``w_i(s)^2 = w_i^2 f_i``, ``f_i = 1 + (s / zR_i)^2``::

            dI/du = -4 u / w_u(s)^2 I,      dI/dv = -4 v / w_v(s)^2 I,
            dI/ds = I sum_i 2 s / (zR_i^2 f_i) (2 u_i^2 / w_i(s)^2 - 1/2),

        rotated back to the lab frame.  Same array genericity as :meth:`intensity`.
        """
        u, v, s = self._beam_coords(X, Y, Z)
        fu = 1.0 + (s / self._zr_u) ** 2
        fv = 1.0 + (s / self._zr_v) ** 2
        wu2 = self._w_u2 * fu
        wv2 = self._w_v2 * fv
        I = (self._two_P_over_pi / array_sqrt(wu2 * wv2)
             * array_exp(-2.0 * (u * u / wu2 + v * v / wv2)))
        dIu = -4.0 * u / wu2 * I
        dIv = -4.0 * v / wv2 * I
        dIs = I * (2.0 * s / (self._zr_u ** 2 * fu) * (2.0 * u * u / wu2 - 0.5)
                   + 2.0 * s / (self._zr_v ** 2 * fv) * (2.0 * v * v / wv2 - 0.5))
        (ux, uy, uz), (vx, vy, vz), (kx, ky, kz) = self._uf, self._vf, self._kf
        return (ux * dIu + vx * dIv + kx * dIs,
                uy * dIu + vy * dIv + ky * dIs,
                uz * dIu + vz * dIv + kz * dIs)

    def beam_radius_u(self, s):
        """1/e^2 radius along u at distance ``s`` from the waist (m)."""
        return self._w_u * array_sqrt(1.0 + (s / self._zr_u) ** 2)

    def beam_radius_v(self, s):
        return self._w_v * array_sqrt(1.0 + (s / self._zr_v) ** 2)

    def electric_field_amplitude(self, X, Y, Z):
        """``|E| = sqrt(2 I / (c eps0 n))`` (V/m)."""
        return array_sqrt(2.0 * self.intensity(X, Y, Z) / (kc.c * kc.epsilon0 * self._n))

    # --------------------------------------------------------- mutators
    def _params(self) -> dict:
        return dict(waist_u=self._w_u, waist_v=self._w_v, wavelength_m=self._lam,
                    power=self._P, n_medium=self._n, polarization=self._eps,
                    propagation_direction=self._k, origin=self._origin,
                    transverse_axis=self._u, transversality=self._transversality,
                    label=self.label)

    def _replace(self, **changes) -> "Beam":
        p = self._params()
        p.update(changes)
        new = object.__new__(type(self))
        Beam.__init__(new, p.pop("waist_u"), p.pop("waist_v"), **p)
        return new

    def with_power(self, power: float) -> "Beam":
        return self._replace(power=power)

    def with_peak_intensity(self, peak_intensity: float) -> "Beam":
        P = _positive("peak_intensity", peak_intensity) * np.pi * self._w_u * self._w_v / 2.0
        return self._replace(power=P)

    def scaled(self, factor: float) -> "Beam":
        """Same beam with its power multiplied by ``factor``."""
        return self._replace(power=self._P * float(factor))

    def moved_to(self, origin) -> "Beam":
        return self._replace(origin=origin)

    def rotated(self, R) -> "Beam":
        """Rotate the beam (direction, polarization, frame and origin) by a proper
        rotation ``R`` about the lab origin."""
        M = fr.as_rotation_matrix(R)
        return self._replace(propagation_direction=M @ self._k,
                             polarization=M @ self._eps, origin=M @ self._origin,
                             transverse_axis=M @ self._u)

    def __repr__(self) -> str:
        name = type(self).__name__
        size = (f"w={self._w_u * 1e6:.3g} um" if self._w_u == self._w_v else
                f"w=({self._w_u * 1e6:.3g}, {self._w_v * 1e6:.3g}) um")
        lab = f", label={self.label!r}" if self.label else ""
        return (f"{name}({size}, lambda={self._lam * 1e9:.1f} nm, P={self._P:.4g} W, "
                f"k={np.round(self._k, 4).tolist()}{lab})")


class Tweezer(Beam):
    """Radially symmetric fundamental Gaussian beam (a focused tweezer).

    Give exactly one colour (``wavelength_m`` / ``frequency_Hz``), one size
    (``waist`` / ``NA`` / ``rayleigh_range`` / ``divergence_angle``) and at most
    one strength (``power`` / ``peak_intensity``).  Defaults: polarization
    ``(0, 1, i)/sqrt(2)``, propagation along ``+x``, waist at the origin.
    """

    def __init__(self, *, waist=None, NA=None, rayleigh_range=None,
                 divergence_angle=None, wavelength_m=None, frequency_Hz=None,
                 power=None, peak_intensity=None, n_medium=1.0,
                 polarization=DEFAULT_POLARIZATION,
                 propagation_direction=DEFAULT_PROPAGATION,
                 origin=(0.0, 0.0, 0.0), transversality: str = "raise",
                 label: Optional[str] = None):
        lam = _resolve_wavelength(wavelength_m, frequency_Hz)
        n = _positive("n_medium", n_medium)
        key, val = _exactly_one("size", waist=waist, NA=NA, rayleigh_range=rayleigh_range,
                                divergence_angle=divergence_angle)
        w = _waist_from(key, _size_values(val, allow_pair=False)[0], lam, n)
        super().__init__(w, w, wavelength_m=lam, power=power,
                         peak_intensity=peak_intensity, n_medium=n,
                         polarization=polarization,
                         propagation_direction=propagation_direction, origin=origin,
                         transversality=transversality, label=label)

    @property
    def waist(self) -> float:
        return self._w_u

    w0 = waist

    @property
    def rayleigh_range(self) -> float:
        return self._zr_u

    zR = rayleigh_range

    @property
    def NA(self) -> float:
        return self.NA_u

    @classmethod
    def from_gaussian_beam(cls, gb, **kw) -> "Tweezer":
        """Lift a legacy :class:`~kamo.trap.gaussian.GaussianBeam` into the lab frame."""
        return cls(waist=gb.waist, wavelength_m=gb.wavelength, power=gb.power,
                   n_medium=gb.n_medium, **kw)

    @_hybridmethod
    def from_trap_frequency(cls_or_self, f_radial_Hz: float, *,
                            polarizability_SI: Optional[float] = None,
                            atom=None, state=(4, 0, 0.5, 1, -1), B_direction=None,
                            source: Optional[str] = None, nuclear_spin: float = 1.5,
                            mass: Optional[float] = None, **kw) -> "Tweezer":
        """The tweezer whose harmonic radial frequency is ``f_radial_Hz``.

        Inverts ``omega_r = sqrt(4 U0 / (m w0^2))`` with
        ``U0 = alpha I0 / (2 c eps0 n)`` -- the relation (curvature factor 4)
        documented in :meth:`GaussianBeam.trap_frequency`.  The strength is
        what is solved for; the geometry comes either from ``kw`` (colour,
        size, direction, polarization) when called on the class, or from the
        existing beam when called on an instance::

            Tweezer.from_trap_frequency(1e3, waist=3e-6, wavelength_m=1064e-9)
            Tweezer(waist=3e-6, wavelength_m=1064e-9).from_trap_frequency(1e3)

        Parameters
        ----------
        polarizability_SI : float, optional
            Bypass the atom with a known polarizability (C m^2/V).  By default
            it is computed for ``state`` from :class:`kamo.Potassium39` via
            :class:`kamo.trap.polarizability.StatePolarizability` -- the same
            path :class:`kamo.trap.Trap` takes -- at this beam's wavelength and
            polarization, quantized along ``B_direction`` (default ``z``).
        atom : Potassium39, optional
            Supplies the mass and, via ``use_portal``, the data source.
        state : (n, l, j, F, mF)
            Hyperfine state, default ``|4S1/2, F=1, mF=-1>``.
        source : {"portal", "arc"}, optional
            Matrix-element source; default from ``atom.use_portal``, else "portal".
        mass : float, optional
            Atomic mass (kg); default ``atom.mass`` or ``kamo.constants.m_K``.
        """
        if "power" in kw or "peak_intensity" in kw:
            raise ValueError("from_trap_frequency solves for the strength; do not "
                             "pass power or peak_intensity.")
        if isinstance(cls_or_self, Beam):
            if kw:
                raise ValueError("from_trap_frequency on an existing beam reuses its "
                                 f"geometry; unexpected keywords {sorted(kw)}.")
            t = cls_or_self
        else:
            t = cls_or_self(**kw)

        if polarizability_SI is None:
            from .polarizability import StatePolarizability
            use_portal = getattr(atom, "use_portal", None)
            if source is None:
                source = "arc" if use_portal is False else "portal"
            elif use_portal is not None and bool(use_portal) != (source == "portal"):
                raise ValueError(f"source={source!r} disagrees with atom.use_portal="
                                 f"{use_portal}; give one or the other.")
            bhat = fr.as_unit_real_vector((0.0, 0.0, 1.0) if B_direction is None
                                          else B_direction, "B_direction")
            alpha = StatePolarizability(state, source, nuclear_spin).alpha_SI(
                t.wavelength_m, t.polarization, bhat)
        else:
            alpha = float(polarizability_SI)
        if not alpha > 0:
            raise ValueError("a red-detuned (alpha > 0) polarizability is needed for "
                             "the intensity maximum to trap.")
        if mass is None:
            mass = getattr(atom, "mass", None)
        m = float(kc.m_K if mass is None else mass)
        if not (np.isfinite(m) and m > 0):
            raise ValueError(f"mass must be positive; got {mass}")
        omega = 2.0 * np.pi * _positive("f_radial_Hz", f_radial_Hz)
        U0 = m * omega ** 2 * t.waist ** 2 / 4.0
        I0 = U0 * 2.0 * kc.c * kc.epsilon0 * t.n_medium / alpha
        return t.with_peak_intensity(I0)


class LightSheet(Beam):
    """Elliptical fundamental Gaussian beam: two waists, coincident waist planes.

    Each size keyword takes a scalar (both axes) or a ``(u, v)`` pair; still
    exactly one keyword from the size group.  ``u`` lies along
    ``transverse_axis`` (default ``z``: the thin axis vertical). The two axes
    have *different* Rayleigh ranges, ``zR_i = pi w_i^2 n / lambda``: the thin
    axis diverges fastest, and a single zR gets the axial confinement wrong by
    ``(w_u / w_v)^2``.
    """

    def __init__(self, *, waist=None, NA=None, rayleigh_range=None,
                 divergence_angle=None, 
                 transverse_axis=(1., 0., 0.),
                 waist_offset_v: float = 0.0, wavelength_m=None, frequency_Hz=None,
                 power=None, peak_intensity=None, n_medium=1.0,
                 polarization=(0.,1.,0.),
                 propagation_direction=(0.,0.,1.),
                 origin=(0.0, 0.0, 0.0), transversality: str = "raise",
                 label: Optional[str] = None):
        if waist_offset_v != 0.0:
            raise NotImplementedError(
                "an astigmatic sheet (waist planes offset along k) is not modelled; "
                "waist_offset_v is the reserved extension point.")
        lam = _resolve_wavelength(wavelength_m, frequency_Hz)
        n = _positive("n_medium", n_medium)
        key, val = _exactly_one("size", waist=waist, NA=NA, rayleigh_range=rayleigh_range,
                                divergence_angle=divergence_angle)
        vu, vv = _size_values(val, allow_pair=True)
        super().__init__(_waist_from(key, vu, lam, n), _waist_from(key, vv, lam, n),
                         wavelength_m=lam, power=power, peak_intensity=peak_intensity,
                         n_medium=n, polarization=polarization,
                         propagation_direction=propagation_direction, origin=origin,
                         transverse_axis=transverse_axis, transversality=transversality,
                         label=label)


# ------------------------------------------------------------------ Crossed

class Crossed(IntensityField):
    """Several beams summed incoherently.

    Nested ``Crossed`` objects are flattened, so ``a + b + c`` is one
    three-beam trap.  The incoherent sum is only right if the cross terms
    time-average away, so every pair closer than ``min_frequency_offset_Hz``
    in frequency *and* with overlapping polarizations raises an
    :class:`InterferenceWarning` (a ``ValueError`` with ``strict=True``): offset
    the frequencies (an AOM does it) or cross the polarizations.  Coherent
    addition -- a lattice -- is the named extension point.
    """

    def __init__(self, beams, *, label: Optional[str] = None, strict: bool = False,
                 min_frequency_offset_Hz: float = 1.0e6):
        items = [beams] if isinstance(beams, IntensityField) else list(beams)
        flat = []
        for b in items:
            if isinstance(b, Crossed):
                flat.extend(b.beams)
            elif isinstance(b, Beam):
                flat.append(b)
            else:
                raise ValueError(f"Crossed takes Beam or Crossed objects; got "
                                 f"{type(b).__name__}")
        if not flat:
            raise ValueError("Crossed needs at least one beam")
        self._beams = tuple(flat)
        self.label = label
        self.strict = bool(strict)
        self.min_frequency_offset_Hz = float(min_frequency_offset_Hz)
        self._check_interference()

    def _check_interference(self):
        for i, a in enumerate(self._beams):
            for b in self._beams[i + 1:]:
                close = abs(a.frequency_Hz - b.frequency_Hz) < self.min_frequency_offset_Hz
                overlap = abs(np.vdot(a.polarization, b.polarization))
                if close and overlap > 1e-6:
                    msg = (f"{a!r} and {b!r} have frequencies within "
                           f"{self.min_frequency_offset_Hz:g} Hz and overlapping "
                           f"polarizations (|eps_a* . eps_b| = {overlap:.3g}); "
                           "they would interfere, but Crossed sums intensities "
                           "incoherently.  Offset their frequencies or cross their "
                           "polarizations.")
                    if self.strict:
                        raise ValueError(msg)
                    warnings.warn(msg, InterferenceWarning, stacklevel=3)

    @property
    def beams(self) -> tuple:
        return self._beams

    @property
    def characteristic_size(self) -> float:
        return max(b.characteristic_size for b in self._beams)

    def intensity(self, X, Y, Z):
        total = self._beams[0].intensity(X, Y, Z)
        for b in self._beams[1:]:
            total = total + b.intensity(X, Y, Z)      # not +=: may be a read-only tensor
        return total

    def intensity_gradient(self, X, Y, Z):
        gx, gy, gz = self._beams[0].intensity_gradient(X, Y, Z)
        for b in self._beams[1:]:
            dx, dy, dz = b.intensity_gradient(X, Y, Z)
            gx, gy, gz = gx + dx, gy + dy, gz + dz
        return gx, gy, gz

    def __len__(self) -> int:
        return len(self._beams)

    def __repr__(self) -> str:
        lab = f", label={self.label!r}" if self.label else ""
        return f"Crossed({len(self._beams)} beams{lab})"
