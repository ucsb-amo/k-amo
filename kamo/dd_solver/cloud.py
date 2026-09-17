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
    def from_variational(cls, cloud) -> "GaussianProfile":
        """From a :class:`GaussianVariationalCloud` (rms widths ``cloud.sigma``)."""
        return cls(cloud.N, cloud.sigma,
                   provenance=f"kamo GaussianVariationalCloud, a = "
                              f"{cloud.a_scattering / kc.a0:+.2f} a0, "
                              f"f = ({', '.join(f'{w / 2 / np.pi:.0f}' for w in cloud.omega)}) Hz")

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
        state (see the module docstring)."""
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


def sample_configuration(profile: GaussianProfile, theta: float = 0.0,
                         seed: Optional[int] = None, N: Optional[int] = None,
                         spin_seed: Optional[int] = None) -> Configuration:
    """Draw one configuration: iid positions from the profile, iid spins at ``theta``.

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
