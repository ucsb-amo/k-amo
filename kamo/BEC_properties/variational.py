"""Gaussian-variational ground state of a harmonically trapped BEC.

Minimizes the Gross-Pitaevskii energy over an anisotropic Gaussian ansatz.  This
fills the gap between the two limits the rest of :mod:`kamo` already covers: the
non-interacting harmonic ground state (exact as ``a -> 0``) and Thomas-Fermi
(:class:`kamo.BECCloud`, valid only for ``mu >> hbar omega``).  At the few-hundred
atom, tens-of-a0 scattering lengths of a tweezer BEC neither limit applies -- the
Thomas-Fermi parameter ``chi = N a / a_ho`` is of order unity, where TF
overestimates the cloud and the ideal gas underestimates it.

Quick start
-----------
>>> from kamo.BEC_properties.variational import GaussianVariationalCloud
>>> import kamo.constants as kc
>>> cloud = GaussianVariationalCloud.from_tweezer(
...     N=500., a_scattering=11.33 * kc.a0, f_radial_Hz=1.0e3, waist=3.0e-6)
>>> cloud.sigma * 1e6            # rms widths (x axial, y, z radial), microns
>>> cloud.peak_density * 1e-6    # cm^-3

Ansatz
------
``n(r) = N / (pi^{3/2} wx wy wz) exp(-sum_i x_i^2 / w_i^2)``, whose energy is

    E/N = sum_i [ hbar^2 / (4 m w_i^2) + m omega_i^2 w_i^2 / 4 ]
          + g N / (2 (2 pi)^{3/2} wx wy wz),      g = 4 pi hbar^2 a / m

Scaling to ``u_i = w_i / a_ho`` and ``hbar omega_bar`` gives the dimensionless form
minimized below,

    E/N = sum_i [ 1/(4 u_i^2) + lambda_i^2 u_i^2 / 4 ] + chi / (sqrt(2 pi) ux uy uz)

with ``lambda_i = omega_i / omega_bar`` and ``chi = N a / a_ho``.  For ``a < 0``
the interaction term is unbounded below and a minimum exists only under a critical
|chi|; above it the cloud collapses (see :attr:`collapsed`,
:meth:`critical_atom_number`).  The Gaussian estimate of that threshold sits about
15% above the exact GPE value.

Note that ``w`` are the 1/e radii of the *density*, so the rms widths are
``sigma = w / sqrt(2)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

import kamo.constants as kc


class CollapseError(RuntimeError):
    """Raised when the attractive Gaussian ansatz has no local minimum."""


class GaussianVariationalCloud:
    """Gaussian-ansatz GPE ground state in an anisotropic harmonic trap.

    Parameters
    ----------
    N : float
        Atom number.
    omega : (3,) sequence
        Trap frequencies in rad/s, ordered (x, y, z).
    a_scattering : float
        s-wave scattering length in METRES (multiply Bohr values by
        ``kamo.constants.a0``).  May be negative.
    mass : float, optional
        Atomic mass (kg).  Defaults to K-39.
    """

    def __init__(self, N: float, omega: Sequence[float], a_scattering: float,
                 mass: Optional[float] = None):
        self.N = float(N)
        self.omega = np.asarray(omega, dtype=float)
        if self.omega.shape != (3,):
            raise ValueError("omega must be a 3-vector of trap frequencies in rad/s")
        self.a_scattering = float(a_scattering)
        self.mass = float(kc.m_K if mass is None else mass)

        self.omega_bar = float(np.prod(self.omega) ** (1. / 3.))
        self.lambda_i = self.omega / self.omega_bar
        self.a_ho = np.sqrt(kc.hbar / (self.mass * self.omega_bar))
        self.chi = self.N * self.a_scattering / self.a_ho

        self._u, self._stable = self._minimize(self.chi)

    # ---------------------------------------------------------------- ansatz

    def _energy_and_grad(self, u, chi):
        """E/N and dE/du in units of hbar*omega_bar, u = w / a_ho."""
        inter = chi / (np.sqrt(2 * np.pi) * np.prod(u))
        e = np.sum(0.25 / u**2 + 0.25 * self.lambda_i**2 * u**2) + inter
        grad = -0.5 / u**3 + 0.5 * self.lambda_i**2 * u - inter / u
        return e, grad

    def _minimize(self, chi, u_min=1e-3, u_max=1e3):
        """Local minimum of the ansatz; (None, False) if the cloud collapses.

        Minimizing in log u keeps the widths positive and turns collapse into a
        run to the lower bound, which is what the bound check below detects.
        """
        def obj(x):
            e, grad = self._energy_and_grad(np.exp(x), chi)
            return e, grad * np.exp(x)

        res = minimize(obj, np.zeros(3), jac=True, method="L-BFGS-B",
                       bounds=[(np.log(u_min), np.log(u_max))] * 3,
                       options=dict(ftol=1e-15, gtol=1e-12, maxiter=2000))
        u = np.exp(res.x)
        if np.any(u < 3 * u_min) or np.any(u > u_max / 3):
            return None, False
        inter = chi / (np.sqrt(2 * np.pi) * np.prod(u))
        H = np.outer(1 / u, 1 / u) * inter
        H[np.diag_indices(3)] = 1.5 / u**4 + 0.5 * self.lambda_i**2 + 2 * inter / u**2
        return u, bool(np.all(np.linalg.eigvalsh(H) > 0))

    # ------------------------------------------------------------ properties

    @property
    def collapsed(self) -> bool:
        """True if the attractive ansatz has no local minimum at this N."""
        return self._u is None

    @property
    def stable(self) -> bool:
        """True if the located extremum is a genuine minimum (positive Hessian)."""
        return self._stable

    def _require(self):
        if self._u is None:
            raise CollapseError(
                f"no Gaussian minimum: N = {self.N:.0f} atoms at a = "
                f"{self.a_scattering / kc.a0:+.2f} a0 collapses "
                f"(chi = {self.chi:+.3f}; critical N ~ "
                f"{self.critical_atom_number():.0f})")
        return self._u

    @property
    def u(self) -> np.ndarray:
        """Widths in oscillator units, w / a_ho."""
        return self._require()

    @property
    def widths(self) -> np.ndarray:
        """1/e radii of the DENSITY, ``n ~ exp(-sum x_i^2 / w_i^2)``, in metres."""
        return self._require() * self.a_ho

    @property
    def sigma(self) -> np.ndarray:
        """rms density widths (m); ``sigma = w / sqrt(2)``."""
        return self.widths / np.sqrt(2)

    @property
    def sigma_noninteracting(self) -> np.ndarray:
        """rms widths of the ideal-gas harmonic ground state (m)."""
        return np.sqrt(kc.hbar / (2 * self.mass * self.omega))

    @property
    def peak_density(self) -> float:
        """n0 = N / (pi^{3/2} wx wy wz), m^-3."""
        return self.N / (np.pi**1.5 * np.prod(self.widths))

    @property
    def peak_column_density(self) -> float:
        """Peak column density along x, N / (pi wy wz), m^-2."""
        w = self.widths
        return self.N / (np.pi * w[1] * w[2])

    @property
    def energy_per_atom(self) -> float:
        """E/N in Joules."""
        e, _ = self._energy_and_grad(self.u, self.chi)
        return kc.hbar * self.omega_bar * e

    @property
    def chemical_potential(self) -> float:
        """mu = d(E_tot)/dN in Joules."""
        u = self.u
        e, _ = self._energy_and_grad(u, self.chi)
        return kc.hbar * self.omega_bar * (
            e + self.chi / (np.sqrt(2 * np.pi) * np.prod(u)))

    @property
    def healing_length(self) -> float:
        """xi = 1 / sqrt(8 pi n0 a), metres.  NaN for a <= 0."""
        if self.a_scattering <= 0:
            return float("nan")
        return 1. / np.sqrt(8 * np.pi * self.peak_density * self.a_scattering)

    @property
    def coupling_g(self) -> float:
        """g = 4 pi hbar^2 a / m, J m^3."""
        return 4 * np.pi * kc.hbar**2 * self.a_scattering / self.mass

    # ----------------------------------------------------- Thomas-Fermi refs

    @property
    def chemical_potential_tf(self) -> float:
        """Thomas-Fermi mu = (1/2) hbar omega_bar (15 chi)^{2/5}, J.  NaN for a<=0."""
        if self.chi <= 0:
            return float("nan")
        return 0.5 * kc.hbar * self.omega_bar * (15 * self.chi) ** 0.4

    @property
    def tf_radii(self) -> np.ndarray:
        """Thomas-Fermi radii R_i = sqrt(2 mu_TF / (m omega_i^2)), m."""
        return np.sqrt(2 * self.chemical_potential_tf / (self.mass * self.omega**2))

    @property
    def thomas_fermi_valid(self) -> bool:
        """mu_TF / (hbar * max omega) > 10 -- the usual TF admissibility test."""
        return bool(self.chemical_potential_tf / (kc.hbar * self.omega.max()) > 10)

    # ---------------------------------------------------------------- limits

    def critical_atom_number(self, tol=1.0002) -> float:
        """Largest N with a metastable Gaussian minimum (attractive clouds).

        Bisects in log N.  Returns inf for a >= 0.  The exact GPE threshold sits
        roughly 15% below this Gaussian estimate.
        """
        if self.a_scattering >= 0:
            return float("inf")
        lo, hi = 1.0, 1e9
        while hi / lo > tol:
            mid = np.sqrt(lo * hi)
            if self._minimize(mid * self.a_scattering / self.a_ho)[0] is None:
                hi = mid
            else:
                lo = mid
        return lo

    def with_atom_number(self, N: float) -> "GaussianVariationalCloud":
        """Same trap and scattering length, different N."""
        return GaussianVariationalCloud(N, self.omega, self.a_scattering, self.mass)

    # --------------------------------------------------------------- density

    def density(self, x, y, z):
        """n(x, y, z) in m^-3.  Broadcasts over array arguments."""
        w = self.widths
        return self.peak_density * np.exp(
            -(np.asarray(x)**2 / w[0]**2
              + np.asarray(y)**2 / w[1]**2
              + np.asarray(z)**2 / w[2]**2))

    def column_density(self, y, z):
        """Column density along x, integral n dx, in m^-2."""
        w = self.widths
        return self.peak_column_density * np.exp(
            -(np.asarray(y)**2 / w[1]**2 + np.asarray(z)**2 / w[2]**2))

    # ----------------------------------------------------------- convenience

    @classmethod
    def from_tweezer(cls, N, a_scattering, f_radial_Hz, waist=None,
                     wavelength=1064e-9, f_axial_Hz=None, mass=None):
        """Build from tweezer geometry.

        The beam propagates along x, so x is the weak (axial) axis and y, z are
        the tight radial ones.  Give EITHER ``waist`` -- in which case the axial
        frequency follows from the beam geometry,
        ``omega_x / omega_r = lambda / (sqrt(2) pi w0)`` -- OR ``f_axial_Hz``
        directly.

        Parameters
        ----------
        N : float
            Atom number.
        a_scattering : float
            Scattering length in METRES.
        f_radial_Hz : float
            Radial trap frequency in ordinary Hz.
        waist : float, optional
            1/e^2 beam waist (m).
        wavelength : float
            Trap wavelength (m), default 1064 nm.
        f_axial_Hz : float, optional
            Axial trap frequency in Hz; alternative to ``waist``.
        """
        if (waist is None) == (f_axial_Hz is None):
            raise ValueError("give exactly one of `waist` or `f_axial_Hz`")
        omega_r = 2 * np.pi * float(f_radial_Hz)
        if f_axial_Hz is not None:
            omega_x = 2 * np.pi * float(f_axial_Hz)
        else:
            omega_x = omega_r * wavelength / (np.sqrt(2) * np.pi * float(waist))
        return cls(N, [omega_x, omega_r, omega_r], a_scattering, mass=mass)

    def summary(self) -> str:
        """Human-readable multi-line report."""
        f = self.omega / (2 * np.pi)
        lines = [
            f"GaussianVariationalCloud  N = {self.N:.0f}, "
            f"a = {self.a_scattering / kc.a0:+.2f} a0",
            f"  trap (fx, fy, fz) = ({f[0]:.1f}, {f[1]:.1f}, {f[2]:.1f}) Hz, "
            f"fbar = {self.omega_bar / 2 / np.pi:.1f} Hz",
            f"  a_ho = {self.a_ho * 1e6:.3f} um,  chi = N a / a_ho = {self.chi:+.3f}",
        ]
        if self.collapsed:
            lines.append(f"  COLLAPSED (critical N ~ {self.critical_atom_number():.0f})")
            return "\n".join(lines)
        s, s0 = self.sigma, self.sigma_noninteracting
        for nm, si, s0i in zip(("x (axial) ", "y (radial)", "z (radial)"), s, s0):
            lines.append(f"  {nm} sigma = {si * 1e6:.3f} um  "
                         f"(x{si / s0i:.2f} vs non-interacting)")
        lines.append(f"  n0 = {self.peak_density * 1e-6:.3e} cm^-3, "
                     f"ncol = {self.peak_column_density * 1e-4:.3e} cm^-2")
        lines.append(f"  mu = h x {self.chemical_potential / kc.h:.0f} Hz, "
                     f"E/N = h x {self.energy_per_atom / kc.h:.0f} Hz")
        if self.a_scattering > 0:
            lines.append(f"  healing length xi = {self.healing_length * 1e6:.3f} um, "
                         f"TF {'valid' if self.thomas_fermi_valid else 'NOT valid'}")
        if not self.stable:
            lines.append("  WARNING: extremum is not a positive-definite minimum")
        return "\n".join(lines)

    def __repr__(self):
        f = self.omega / (2 * np.pi)
        return (f"GaussianVariationalCloud(N={self.N:.0f}, "
                f"f=({f[0]:.0f}, {f[1]:.0f}, {f[2]:.0f}) Hz, "
                f"a={self.a_scattering / kc.a0:+.2f} a0"
                + (", COLLAPSED)" if self.collapsed else ")"))
