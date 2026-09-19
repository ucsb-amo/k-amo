"""Density profiles and the sampling of atomic configurations.

The profile is the Gaussian-variational Gross-Pitaevskii ground state of
:class:`kamo.BEC_properties.variational.GaussianVariationalCloud`, wrapped so
that it can be sampled, diluted, and handed unchanged to :mod:`kamo.imaging`
(it exposes ``N``, ``widths`` and ``density`` -- the propagator's cloud contract).

Why the sampling is exact, not an approximation
-----------------------------------------------
Positions are drawn **iid** from ``|phi(r)|^2`` and spins **iid** with
``p_up = cos^2(theta/2)`` for a coherent spin state at polar angle ``theta``.

1. For a pure single-mode condensate ``g2(0) = 1 - 1/N``, so positions are
   uncorrelated to O(1/N).  A THERMAL Bose gas has ``g2(0) = 2`` and would need
   correlated sampling; do not carry this sampler into a non-condensate problem.
2. The probe is spin-conserving (Rayleigh scattering on closed sigma- lines) and
   the exchange kernel conserves ``m_I`` atom by atom, so the linear-response
   problem is block-diagonal in the ground-spin configuration basis.  Spin
   coherences never enter and the CSS azimuth drops out.

This is therefore NOT mean field: a mean-field treatment would give every atom
the average polarizability ``p_up alpha_up + p_dn alpha_dn`` and miss entirely
that a like-spin pair can be resonant while an unlike-spin pair cannot.

Regression note (2026-09-16)
----------------------------
The build specification quoted a Gaussian-ansatz energy with kinetic term
``hbar^2/(4m) (2/sigma_r^2 + 1/sigma_x^2)``, twice the correct
``hbar^2/(8 m sigma^2)`` per axis (a Gaussian density of rms ``sigma`` has
``<p^2> = hbar^2 / 4 sigma^2``), and its regression table (401 nm, 1.64 um,
eta_eff 19.2 at N = 500) is reproduced ONLY with that doubled term; the correct
functional -- kamo's -- gives 341 nm, 1.59 um, eta_eff 27.3.  The doubled term
also gives a non-interacting ground state ``2^{1/4}`` too wide.  kamo's profile
is the operating point here; :meth:`GaussianProfile.spec_reference` rebuilds the
specification's widths so its validated numbers can still be reproduced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import kamo.constants as kc

from .system import SPIN_DN, SPIN_UP

# The build specification's trap; the lab's measured radial frequency is 1.0 kHz
# (CLAUDE.md) -- pass f_radial_Hz explicitly when that matters.
F_RADIAL_HZ = 1170.0
F_AXIAL_HZ = 93.0
A_SCATTERING_BOHR = 11.3     # UNVERIFIED at 520.6 G; kamo's table gives 10.96 (2026-09-16)




class SpecReferenceWarning(UserWarning):
    """A profile was built from the build specification's doubled-kinetic ansatz,
    which is not the ground state (added 2026-09-19)."""


# ------------------------------------------------------------------ profile


class GaussianProfile:
    """A Gaussian density ``n(r) = n_peak exp(-sum x_i^2 / 2 sigma_i^2)``.

    Parameters
    ----------
    N : float
        Atom number.
    sigma : (3,) sequence
        rms widths (m), ordered (x, y, z); x is the probe axis.
    origin : (3,) sequence
        Centre (m).
    """

    def __init__(self, N: float, sigma: Sequence[float], origin=(0.0, 0.0, 0.0),
                 provenance: str = ""):
        self.N = float(N)
        self.sigma = np.asarray(sigma, dtype=float) * np.ones(3)
        self.origin = np.asarray(origin, dtype=float)
        self.provenance = provenance

    # ----------------------------------------------------------- builders
    @classmethod
    def from_variational(cls, cloud, axes=None) -> "GaussianProfile":
        """From a :class:`GaussianVariationalCloud` (rms widths ``cloud.sigma``).

        .. warning::
           ``sigma`` must come out ordered ``(x, y, z)`` with x the probe axis.
           A ``GaussianVariationalCloud`` built the :mod:`kamo.trap` way -- from
           ``TrapCloud.harmonic_reference()`` or from
           ``trap.trap_frequencies().omega`` -- carries its axes in PRINCIPAL
           order, highest frequency first, which for the lab tweezer is
           ``(y, z, x)``.  Copying that verbatim puts the 1.7 um long axis along
           B and a 0.37 um axis along the probe, cutting the column density seen
           by the probe to 0.22x.  Pass ``axes=trap.principal_axes`` and the
           widths are mapped to the lab frame; ``operating_point`` below is safe
           because it orders ``omega`` by hand.  (Guard added 2026-09-17.)
        """
        sigma = np.asarray(cloud.sigma, dtype=float)
        note = ""
        if axes is not None:
            A = np.asarray(axes, dtype=float)
            if A.shape != (3, 3):
                raise ValueError("axes must be a (3, 3) matrix of principal axes (rows)")
            if np.max(np.abs(np.abs(A) - np.eye(3)[np.argmax(np.abs(A), axis=1)])) > 1e-6:
                raise ValueError("from_variational needs an axis-aligned principal frame; "
                                 "this trap's axes are tilted, so a Gaussian with per-axis "
                                 "sigma cannot represent it -- sample the gridded cloud "
                                 "with GridProfile instead")
            sigma = np.sqrt((A ** 2).T @ sigma ** 2)
            note = ", mapped to lab axes"
        return cls(cloud.N, sigma,
                   provenance=f"kamo GaussianVariationalCloud, a = "
                              f"{cloud.a_scattering / kc.a0:+.2f} a0, "
                              f"f = ({', '.join(f'{w / 2 / np.pi:.0f}' for w in cloud.omega)}) Hz"
                              + note)

    @classmethod
    def lab_operating_point(cls, N: float = 500, waist: float = 3e-6,
                            wavelength_m: float = 1064e-9, f_radial_Hz: float = 1.0e3,
                            B_gauss: float = 520.583, state=(4, 0, 0.5, 1, -1),
                            a_bohr: Optional[float] = None) -> "GaussianProfile":
        """The Gaussian moment-match of the LAB trap's true GP cloud (opens kamo.trap).

        Builds ``Trap(Tweezer(waist, wavelength_m)).rescaled_to_frequency(f_radial_Hz)``
        with gravity and the real anharmonic potential, solves the 3D GP ground
        state, and returns a Gaussian with its rms widths, in lab ``(x, y, z)``
        order.  Slower than :meth:`operating_point` (a full 3D solve) and worth
        it: the package's harmonic default (1170 / 93 Hz, 11.3 a0, Gaussian
        ansatz) gives ``eta_eff = 27.3`` at N = 500 while this gives 19.3.

        Use :class:`GridProfile` instead when the shape matters: this cloud is
        flatter than Gaussian (axial excess kurtosis -0.35), so a moment-matched
        Gaussian still reports ``eta_eff`` 2.7 % high.
        """
        from kamo.trap import Trap, Tweezer, solve as trap_solve
        tw = Tweezer(waist=waist, wavelength_m=wavelength_m)
        trap = Trap(tw, state=state, B_gauss=B_gauss,
                    B_direction=(0, 0, 1)).rescaled_to_frequency(f_radial_Hz)
        kw = {} if a_bohr is None else dict(a_scattering=a_bohr * kc.a0)
        tc = trap_solve(trap, N=N, mode="gp", **kw)
        return cls(tc.N, tc.sigma, origin=getattr(tc, "center", (0.0, 0.0, 0.0)),
                   provenance=f"kamo.trap 3D GP, {waist * 1e6:.1f} um tweezer at "
                              f"{wavelength_m * 1e9:.0f} nm rescaled to "
                              f"{f_radial_Hz / 1e3:.2f} kHz radial, moment-matched Gaussian")

    @classmethod
    def operating_point(cls, N: float, f_radial_Hz: float = F_RADIAL_HZ,
                        f_axial_Hz: float = F_AXIAL_HZ,
                        a_bohr: float = A_SCATTERING_BOHR, mass=None) -> "GaussianProfile":
        """kamo's Gaussian-variational GP ground state (the correct functional)."""
        from kamo.BEC_properties.variational import GaussianVariationalCloud
        omega = 2 * np.pi * np.array([f_axial_Hz, f_radial_Hz, f_radial_Hz], dtype=float)
        cloud = GaussianVariationalCloud(N, omega, a_bohr * kc.a0, mass=mass)
        return cls.from_variational(cloud)

    @classmethod
    def spec_reference(cls, N: float, f_radial_Hz: float = F_RADIAL_HZ,
                       f_axial_Hz: float = F_AXIAL_HZ,
                       a_bohr: float = A_SCATTERING_BOHR, mass=None) -> "GaussianProfile":
        """The build specification's profile: doubled kinetic term, ``N - 1`` in the
        interaction.  Reproduces its regression table; NOT the physical ground
        state (see the module docstring).

        Its kinetic term ``hbar^2/(4 m sigma^2)`` per axis is twice the correct
        ``hbar^2/(8 m sigma^2)``, so even the non-interacting cloud comes out
        ``2^{1/4}`` too wide (396 nm / 1.404 um at 1170 / 93 Hz instead of
        ``sqrt(hbar / 2 m omega)`` = 333 nm / 1.181 um).  Emits
        :class:`SpecReferenceWarning`; use :meth:`operating_point` or
        :meth:`lab_operating_point` for physics.
        """
        warnings.warn("GaussianProfile.spec_reference uses the build specification's "
                      "doubled kinetic term (cloud 2^(1/4) too wide in the "
                      "non-interacting limit); it reproduces the spec's regression "
                      "table and is not the ground state. Use operating_point or "
                      "lab_operating_point for physics.",
                      SpecReferenceWarning, stacklevel=2)
        m = float(kc.m_K if mass is None else mass)
        g = 4 * np.pi * kc.hbar ** 2 * a_bohr * kc.a0 / m
        wr, wx = 2 * np.pi * f_radial_Hz, 2 * np.pi * f_axial_Hz

        def energy(p):
            sr, sx = np.exp(p)
            return (kc.hbar ** 2 / (4 * m) * (2 / sr ** 2 + 1 / sx ** 2)
                    + 0.5 * m * (2 * wr ** 2 * sr ** 2 + wx ** 2 * sx ** 2)
                    + g * (N - 1) / (16 * np.pi ** 1.5 * sr ** 2 * sx))

        res = minimize(energy, np.log([4e-7, 1.6e-6]), method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-40, maxiter=5000))
        sr, sx = np.exp(res.x)
        return cls(N, (sx, sr, sr), provenance="build-spec Gaussian ansatz "
                                               "(doubled kinetic term; reference only)")

    # --------------------------------------------------------- properties
    @property
    def widths(self) -> np.ndarray:
        """1/e radii of the density, ``sqrt(2) sigma`` -- kamo.imaging's convention."""
        return np.sqrt(2.0) * self.sigma

    @property
    def peak_density(self) -> float:
        """``N / ((2 pi)^{3/2} sigma_x sigma_y sigma_z)`` (m^-3)."""
        return self.N / ((2 * np.pi) ** 1.5 * np.prod(self.sigma))

    @property
    def mean_spacing(self) -> float:
        """``n_peak^{-1/3}`` (m)."""
        return self.peak_density ** (-1 / 3)

    def eta_eff(self, wavelength: float) -> float:
        """``lambda^3 int n^2 / int n = n_peak lambda^3 / 2^{3/2}``.

        The closed form is GAUSSIAN-ONLY.  For any other shape use
        ``wavelength**3 * int n^2 / N`` directly (a kamo ``TrapCloud`` exposes
        ``density_squared_integral``); :class:`GridProfile` does that.  Applying
        this form to a GP cloud's peak density understates ``eta_eff`` by 7 %,
        and a Gaussian moment-matched to a GP cloud overstates it by 2.7 %.


        The atom-number-WEIGHTED density: near-field shifts are set by nearest
        neighbours and sample the density weighted by how many atoms experience
        it.  Using the peak density instead overestimates every near-field
        quantity by ``2^{3/2} = 2.8``.
        """
        return self.peak_density * wavelength ** 3 / 2 ** 1.5

    def density(self, x, y, z):
        """``n(x, y, z)`` in m^-3 (broadcasts)."""
        s, o = self.sigma, self.origin
        return self.peak_density * np.exp(
            -((np.asarray(x) - o[0]) ** 2 / (2 * s[0] ** 2)
              + (np.asarray(y) - o[1]) ** 2 / (2 * s[1] ** 2)
              + (np.asarray(z) - o[2]) ** 2 / (2 * s[2] ** 2)))

    def column_density(self, y, z, axis: int = 0):
        """Column density integrated along ``axis`` (default x), m^-2."""
        s = self.sigma
        others = [i for i in range(3) if i != axis]
        u = (np.asarray(y) - self.origin[others[0]]) / s[others[0]]
        v = (np.asarray(z) - self.origin[others[1]]) / s[others[1]]
        return self.N / (2 * np.pi * s[others[0]] * s[others[1]]) * np.exp(-0.5 * (u ** 2 + v ** 2))

    def scaled(self, factor, N: Optional[float] = None) -> "GaussianProfile":
        """Widths multiplied by ``factor`` (scalar or per axis) at fixed N: dilution."""
        return GaussianProfile(self.N if N is None else N, self.sigma * np.asarray(factor),
                               self.origin, self.provenance + f" x{factor}")

    def with_atom_number(self, N: float) -> "GaussianProfile":
        """Same widths, different N (a density rescale, NOT a new GP solution)."""
        return GaussianProfile(N, self.sigma, self.origin, self.provenance)

    def sample_positions(self, n: int, rng) -> np.ndarray:
        return rng.normal(size=(int(n), 3)) * self.sigma + self.origin

    def summary(self, wavelength: Optional[float] = None) -> str:
        s = self.sigma * 1e9
        lines = [f"GaussianProfile  N = {self.N:.0f}   sigma = ({s[0]:.0f}, {s[1]:.0f}, "
                 f"{s[2]:.0f}) nm   n_peak = {self.peak_density * 1e-6:.3e} cm^-3   "
                 f"spacing {self.mean_spacing * 1e9:.0f} nm"]
        if wavelength is not None:
            lines.append(f"  eta_eff = {self.eta_eff(wavelength):.2f}   "
                         f"n_peak lambda^3 = {self.peak_density * wavelength ** 3:.2f}   "
                         f"n_peak / k^3 = {self.peak_density * (wavelength / 2 / np.pi) ** 3:.4f}")
        if self.provenance:
            lines.append(f"  ({self.provenance})")
        return "\n".join(lines)

    def __repr__(self):
        s = self.sigma * 1e9
        return f"GaussianProfile(N={self.N:.0f}, sigma=({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) nm)"


# ------------------------------------------------------- gridded kamo clouds


class GridProfile:
    """Any gridded kamo density as a dd_solver profile: sample it, propagate it.

    Wraps an object carrying a density on a regular grid -- a
    :class:`kamo.trap.cloud.TrapCloud` from ``solve(trap, N, 'gp')``, a
    Thomas-Fermi cloud, a finite-temperature one -- so the SAME object feeds the
    microscopic solver and :mod:`kamo.imaging`.  Without this the two sides of
    the A/B could only ever be Gaussian: ``kamo.imaging.bpm.UniformMixture``
    rebuilds a centred Gaussian from ``cloud.N`` and ``cloud.widths`` and never
    calls ``density()``, and ``sample_configuration`` accepted only a
    :class:`GaussianProfile` (added 2026-09-17).

    Sampling is hierarchical inverse-CDF with linear interpolation: ``x`` from
    the x marginal, ``y`` from the y marginal of the chosen x plane, ``z`` from
    the z profile of the chosen ``(x, y)`` line, each inverted on the
    piecewise-linear cumulative distribution rather than by picking a voxel and
    filling it uniformly.  The difference is not cosmetic: voxel-uniform
    sampling fails a chi-square against the true marginals by a factor of
    hundreds, while this passes at chi2/dof ~ 1.

    Residual bias, worth quoting with any near-field number: the sampler is
    exact for the INTERPOLANT, and the interpolant on a healing-length grid
    (about 88 nm radially at the lab point) inflates each variance by ``h^2/6``,
    which lowers the sampled ``int n^2`` by about 1 % and raises the rms widths
    by about 0.5 %.  ``upsample=2`` (zero-padded FFT of ``sqrt(n)``, so the
    result cannot go negative) removes most of it.

    Parameters
    ----------
    cloud : object
        Must expose ``N``, a 3D ``density_grid`` (m^-3) and either a ``grid``
        carrying per-axis coordinates or explicit ``axes``.
    axes : sequence of 3 arrays, optional
        Coordinates (m) along x, y, z; taken from ``cloud.grid`` when omitted.
    upsample : int
        Refine each axis by this factor before building the CDFs.
    """

    def __init__(self, cloud, axes=None, upsample: int = 1, recenter: bool = True):
        self.cloud = cloud
        self.N = float(cloud.N)
        n = np.asarray(getattr(cloud, "density_grid"), dtype=float)
        if axes is None:
            axes = self._axes_from(cloud)
        ax = [np.asarray(a, dtype=float) for a in axes]
        self.recenter = bool(recenter)
        if upsample > 1:
            n, ax = self._refine(n, ax, int(upsample))
        if n.shape != tuple(a.size for a in ax):
            raise ValueError(f"density_grid {n.shape} does not match axes "
                             f"{tuple(a.size for a in ax)}")
        self.axes = ax
        self.density_grid = np.maximum(n, 0.0)
        self.upsample = int(upsample)
        if self.recenter:
            # kamo.trap's GriddedMixture recentres on the centroid by default, so
            # the sampler must too or the two codes see clouds displaced by the
            # gravitational sag (286 nm along z for the lab tweezer: enough to
            # double the A/B residual, and a 1 rad tilt across an NA 0.42 mode).
            # Added 2026-09-17 after exactly that showed up in the sweep.
            self._build()
            c = self.origin
            self.axes = [a - ci for a, ci in zip(self.axes, c)]
            self._shift = c
        else:
            self._shift = np.zeros(3)
        self.provenance = (f"{type(cloud).__name__} on a "
                           f"{'x'.join(str(m) for m in n.shape)} grid"
                           + (f", upsampled x{upsample}" if upsample > 1 else "")
                           + (f", recentred by {np.round(self._shift * 1e9, 1)} nm"
                              if self.recenter and np.any(np.abs(self._shift) > 1e-12) else ""))
        self._interp = None
        self._build()

    # ------------------------------------------------------------- setup
    @staticmethod
    def _axes_from(cloud):
        g = getattr(cloud, "grid", None)
        if g is None:
            raise ValueError("pass axes=: the cloud has no .grid")
        for names in (("x", "y", "z"), ("axis_x", "axis_y", "axis_z")):
            if all(hasattr(g, nm) for nm in names):
                return [np.asarray(getattr(g, nm), dtype=float) for nm in names]
        if hasattr(g, "axes"):
            return [np.asarray(a, dtype=float) for a in g.axes]
        raise ValueError("cannot read the grid coordinates; pass axes=")

    @staticmethod
    def _refine(n, ax, factor):
        """Zero-pad the FFT of sqrt(n) and square: non-negative by construction."""
        F = np.fft.fftn(np.sqrt(n))
        new = tuple(factor * m for m in n.shape)
        G = np.zeros(new, dtype=complex)
        lo = [slice(0, m // 2 + 1) for m in n.shape]
        hi = [slice(-(m - m // 2 - 1), None) for m in n.shape]
        for sx in (lo[0], hi[0]):
            for sy in (lo[1], hi[1]):
                for sz in (lo[2], hi[2]):
                    G[sx, sy, sz] = F[sx, sy, sz]
        out = np.abs(np.fft.ifftn(G)) ** 2 * factor ** 3
        # n is a DENSITY: the refined grid has factor^3 more cells, each of
        # volume dV/factor^3, so the sum must grow by factor^3 to keep int n dV.
        out = out * (n.sum() * factor ** 3 / max(out.sum(), 1e-300))
        newax = []
        for a_, m in zip(ax, new):
            d = (a_[-1] - a_[0]) / (a_.size - 1) if a_.size > 1 else 0.0
            newax.append(a_[0] + d * np.arange(m) / factor)
        return out, newax

    @staticmethod
    def _cdf(w, a):
        """Piecewise-linear CDF of a density sampled at ``a`` with weights ``w``."""
        w = np.maximum(np.asarray(w, dtype=float), 0.0)
        if a.size == 1:
            return None
        seg = 0.5 * (w[:-1] + w[1:]) * np.diff(a)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        tot = c[-1]
        return c / tot if tot > 0 else None

    def _build(self):
        n = self.density_grid
        x, y, z = self.axes
        self._cx = self._cdf(n.sum(axis=(1, 2)), x)
        self._wy = n.sum(axis=2)
        self._cy = [self._cdf(self._wy[i], y) for i in range(x.size)]
        self._n = n

    # -------------------------------------------------------- properties
    def _marginal(self, axis):
        n = self.density_grid
        m = n.sum(axis=tuple(j for j in range(3) if j != axis))
        return m / m.sum()

    @property
    def sigma(self) -> np.ndarray:
        """rms widths (m) of the gridded density."""
        out = []
        for i, a in enumerate(self.axes):
            m = self._marginal(i)
            mu = float(m @ a)
            out.append(float(np.sqrt(max(m @ (a - mu) ** 2, 0.0))))
        return np.array(out)

    @property
    def widths(self) -> np.ndarray:
        """1/e radii, ``sqrt(2) sigma`` -- kamo.imaging's convention."""
        return np.sqrt(2.0) * self.sigma

    @property
    def origin(self) -> np.ndarray:
        return np.array([float(self._marginal(i) @ a) for i, a in enumerate(self.axes)])

    @property
    def peak_density(self) -> float:
        return float(self.density_grid.max())

    @property
    def mean_spacing(self) -> float:
        return self.peak_density ** (-1 / 3)

    @property
    def cell_volume(self) -> float:
        return float(np.prod([a[1] - a[0] for a in self.axes if a.size > 1]))

    @property
    def density_squared_integral(self) -> float:
        """``int n^2 dV`` (m^-3), by the grid's own quadrature."""
        return float(np.sum(self.density_grid ** 2) * self.cell_volume)

    def eta_eff(self, wavelength: float) -> float:
        """``lambda^3 int n^2 / N``, from the grid.

        NOT the Gaussian closed form ``n_peak lambda^3 / 2^{3/2}``, which does
        not apply to a GP or Thomas-Fermi shape: on the lab GP cloud it would
        read 17.9 against the true 19.3.
        """
        return float(wavelength ** 3 * self.density_squared_integral / self.N)

    def density(self, x, y, z):
        """Trilinear interpolation of the gridded density (m^-3)."""
        from scipy.interpolate import RegularGridInterpolator
        if getattr(self, "_interp", None) is None:
            self._interp = RegularGridInterpolator(
                tuple(self.axes), self.density_grid, bounds_error=False, fill_value=0.0)
        xb, yb, zb = np.broadcast_arrays(np.asarray(x, dtype=float),
                                         np.asarray(y, dtype=float),
                                         np.asarray(z, dtype=float))
        pts = np.stack([xb.ravel(), yb.ravel(), zb.ravel()], axis=-1)
        return self._interp(pts).reshape(xb.shape)

    def column_density(self, y, z, axis: int = 0):
        """Column density integrated along ``axis`` (default x), m^-2."""
        from scipy.interpolate import RegularGridInterpolator
        others = [i for i in range(3) if i != axis]
        d = self.axes[axis]
        dx = (d[1] - d[0]) if d.size > 1 else 0.0
        col = self.density_grid.sum(axis=axis) * dx
        it = RegularGridInterpolator((self.axes[others[0]], self.axes[others[1]]), col,
                                     bounds_error=False, fill_value=0.0)
        yb, zb = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        return it(np.stack([yb.ravel(), zb.ravel()], -1)).reshape(yb.shape)

    def _clone(self, axes, grid, N, note):
        out = object.__new__(GridProfile)
        out.cloud = self.cloud
        out.recenter = self.recenter
        out._shift = self._shift
        out.N = float(N)
        out.axes = axes
        out.density_grid = grid
        out.upsample = self.upsample
        out.provenance = self.provenance + note
        out._interp = None
        out._build()
        return out

    def scaled(self, factor, N: Optional[float] = None) -> "GridProfile":
        """Dilute by stretching the grid coordinates at fixed N."""
        f = np.asarray(factor, dtype=float) * np.ones(3)
        return self._clone([a * fi for a, fi in zip(self.axes, f)],
                           self.density_grid / float(np.prod(f)),
                           self.N if N is None else N, f" x{factor}")

    def with_atom_number(self, N: float) -> "GridProfile":
        """Same shape, different N (a density rescale, NOT a new GP solution)."""
        return self._clone(list(self.axes), self.density_grid * (float(N) / self.N), N,
                           f" N={N:.0f}")

    # --------------------------------------------------------- sampling
    @staticmethod
    def _invert(c, a, u):
        """Invert a piecewise-linear CDF ``c`` on nodes ``a`` at quantiles ``u``."""
        if c is None:
            return np.full(np.shape(u), a[0], dtype=float)
        i = np.clip(np.searchsorted(c, u, side="right") - 1, 0, c.size - 2)
        w = c[i + 1] - c[i]
        t = np.where(w > 0, (u - c[i]) / np.where(w > 0, w, 1.0), 0.0)
        return a[i] + t * (a[i + 1] - a[i])

    def sample_positions(self, n: int, rng) -> np.ndarray:
        n = int(n)
        x_ax, y_ax, z_ax = self.axes
        xs = self._invert(self._cx, x_ax, rng.uniform(size=n))
        ix = np.clip(np.searchsorted(x_ax, xs) - 1, 0, x_ax.size - 2)
        ys = np.empty(n)
        uy = rng.uniform(size=n)
        for i in np.unique(ix):
            m = ix == i
            ys[m] = self._invert(self._cy[i], y_ax, uy[m])
        iy = np.clip(np.searchsorted(y_ax, ys) - 1, 0, y_ax.size - 2)
        zs = np.empty(n)
        uz = rng.uniform(size=n)
        flat = ix * y_ax.size + iy
        for key in np.unique(flat):
            m = flat == key
            i, j = divmod(int(key), y_ax.size)
            zs[m] = self._invert(self._cdf(self._n[i, j], z_ax), z_ax, uz[m])
        return np.stack([xs, ys, zs], axis=1)

    # ----------------------------------------------------- kamo.imaging
    def bpm_source(self, propagator, response, species, **kw):
        """A :class:`kamo.trap.imaging_bridge.GriddedMixture` on the SAME density.

        Use this instead of ``UniformMixture`` so the propagation sees the real
        shape rather than an rms-matched Gaussian centred on the origin.
        """
        from kamo.trap.imaging_bridge import GriddedMixture
        # match the sampler's framing exactly (see __init__)
        kw.setdefault("recenter", self.recenter)
        mix = GriddedMixture.for_propagator(propagator, self.cloud, response, species, **kw)
        if not np.allclose(np.asarray(mix.center), self._shift, atol=1e-12):
            raise ValueError(f"the propagator's centre {np.asarray(mix.center)} does not match "
                             f"the sampler's {self._shift}; the two codes would see clouds "
                             "displaced relative to each other")
        return mix

    def summary(self, wavelength: Optional[float] = None) -> str:
        s = self.sigma * 1e9
        lines = [f"GridProfile  N = {self.N:.0f}   sigma = ({s[0]:.0f}, {s[1]:.0f}, "
                 f"{s[2]:.0f}) nm   n_peak = {self.peak_density * 1e-6:.3e} cm^-3"]
        if wavelength is not None:
            lines.append(f"  eta_eff = {self.eta_eff(wavelength):.2f} (from int n^2)")
        lines.append(f"  ({self.provenance})")
        return chr(10).join(lines)

    def __repr__(self):
        s = self.sigma * 1e9
        return f"GridProfile(N={self.N:.0f}, sigma=({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) nm)"


def profile_from_kamo(cloud, upsample: int = 1, axes=None):
    """A dd_solver profile from any kamo cloud.

    Gridded (``density_grid``) -> :class:`GridProfile`; a
    ``GaussianVariationalCloud`` -> :class:`GaussianProfile`.  This is the entry
    point for handing a :func:`kamo.trap.solve` result to the microscopic solver
    and to :mod:`kamo.imaging` as the same object.
    """
    if hasattr(cloud, "density_grid"):
        return GridProfile(cloud, axes=axes, upsample=upsample)
    return GaussianProfile.from_variational(cloud)


# ----------------------------------------------------------------- spins


def spin_probability_up(theta: float) -> float:
    """``cos^2(theta/2)`` for a coherent spin state at polar angle theta."""
    return float(np.cos(0.5 * theta) ** 2)


def like_pair_fraction(theta) -> np.ndarray:
    """``cos^4(theta/2) + sin^4(theta/2) = 1 - sin^2(theta) / 2``."""
    return 1.0 - 0.5 * np.sin(np.asarray(theta, dtype=float)) ** 2


# ------------------------------------------------------------ configuration


@dataclass
class Configuration:
    """Frozen positions and spins of one shot.

    Attributes
    ----------
    positions : (N, 3) float
        Metres.
    spins : (N,) int
        ``+1`` for ``|up>``, ``-1`` for ``|dn>``.
    theta : float
        CSS polar angle the spins were drawn at.
    seed, spin_seed : int or None
        The seeds, so the shot can be regenerated.
    """

    positions: np.ndarray
    spins: np.ndarray
    theta: float = 0.0
    seed: Optional[int] = None
    spin_seed: Optional[int] = None

    @property
    def N(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_up(self) -> int:
        return int(np.sum(self.spins > 0))

    @property
    def n_dn(self) -> int:
        return int(np.sum(self.spins < 0))

    @property
    def S_z(self) -> float:
        """``(N_up - N_dn) / 2``."""
        return 0.5 * (self.n_up - self.n_dn)

    def resample_spins(self, theta: Optional[float] = None,
                       spin_seed: Optional[int] = None) -> "Configuration":
        """New spins at fixed positions (separates spin from position disorder)."""
        th = self.theta if theta is None else float(theta)
        rng = np.random.default_rng(None if spin_seed is None else [int(spin_seed), 1])
        spins = draw_spins(self.N, th, rng)
        return replace(self, spins=spins, theta=th, spin_seed=spin_seed)

    def resample_positions(self, profile: GaussianProfile,
                           seed: Optional[int] = None) -> "Configuration":
        """New positions at fixed spins."""
        rng = np.random.default_rng(None if seed is None else [int(seed), 0])
        return replace(self, positions=profile.sample_positions(self.N, rng), seed=seed)

    def nearest_neighbours(self):
        """``(distance, polar_angle_from_z, index)`` of every atom's nearest neighbour."""
        tree = cKDTree(self.positions)
        d, idx = tree.query(self.positions, k=2)
        d, idx = d[:, 1], idx[:, 1]
        dvec = self.positions[idx] - self.positions
        cos_t = dvec[:, 2] / np.maximum(d, 1e-300)
        return d, np.arccos(np.clip(cos_t, -1, 1)), idx

    def __repr__(self):
        return (f"Configuration(N={self.N}, n_up={self.n_up}, n_dn={self.n_dn}, "
                f"theta={self.theta:.3f}, seed={self.seed})")


def draw_spins(N: int, theta: float, rng) -> np.ndarray:
    p_up = spin_probability_up(theta)
    return np.where(rng.uniform(size=int(N)) < p_up, SPIN_UP, SPIN_DN).astype(np.int8)


def sample_configuration(profile, theta: float = 0.0,
                         seed: Optional[int] = None, N: Optional[int] = None,
                         spin_seed: Optional[int] = None) -> Configuration:
    """Draw one configuration: iid positions from the profile, iid spins at ``theta``.

    ``profile`` is anything with ``N`` and ``sample_positions(n, rng)``:
    :class:`GaussianProfile` or :class:`GridProfile` (any gridded kamo cloud).

    Positions use the stream ``default_rng([seed, 0])`` and spins
    ``default_rng([spin_seed or seed, 1])``, so the two kinds of disorder can be
    re-drawn independently with :meth:`Configuration.resample_spins` and
    :meth:`Configuration.resample_positions`.
    """
    n = int(round(profile.N)) if N is None else int(N)
    ss = seed if spin_seed is None else spin_seed
    rng_pos = np.random.default_rng(None if seed is None else [int(seed), 0])
    rng_spin = np.random.default_rng(None if ss is None else [int(ss), 1])
    return Configuration(profile.sample_positions(n, rng_pos), draw_spins(n, theta, rng_spin),
                         float(theta), seed, ss)


def uniform_sphere_configuration(N: int, density: float, seed: Optional[int] = None,
                                 theta: float = 0.0) -> Configuration:
    """N atoms uniformly in a sphere at the given number density (m^-3).

    For tail tests against the uniform-density RG literature (T13).
    """
    rng = np.random.default_rng(None if seed is None else [int(seed), 0])
    radius = (3 * N / (4 * np.pi * density)) ** (1 / 3)
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = radius * rng.uniform(size=(N, 1)) ** (1 / 3)
    spins = draw_spins(N, theta, np.random.default_rng(None if seed is None else [int(seed), 1]))
    return Configuration(v * r, spins, theta, seed, seed)
