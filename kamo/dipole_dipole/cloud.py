"""Thomas-Fermi BEC profile, pair statistics and the relative-velocity scale.

The loss model needs three things from the cloud:

1. the density-squared integral, since two-body loss goes as n^2;
2. the pair-separation distribution at the Condon radius, i.e. the pair correlation
   g2(r) (1 for a condensate, 2 for a thermal cloud);
3. a relative-velocity scale, which decides whether the collision is in the
   quasi-static or the flux (Landau-Zener) regime.

kamo.BEC_properties.bec.BEC computes the chemical potential and Thomas-Fermi radii
too, but it derives the axial trap frequency from a hard-coded 1064 nm / 3.8 um
beam.  BECCloud takes all three trap frequencies explicitly instead.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from kamo import constants as c

_A0 = 5.29177210544e-11


class BECCloud:
    """Zero-temperature Thomas-Fermi condensate in a harmonic trap.

    Parameters
    ----------
    N : float
        Atom number.
    trap_frequencies_Hz : 3-sequence
        Trap frequencies (ordinary Hz, not rad/s).
    a_s_bohr : float
        s-wave scattering length in Bohr radii.  Left as a free parameter:
        Potassium39.get_scattering_length reads a network drive and is not portable.
    mass : float, optional
        Defaults to the K-39 mass.
    temperature_K : float, optional
        Residual thermal component.  If given, the relative-velocity scale uses it;
        otherwise the mean-field (chemical-potential) scale is used.
    """

    def __init__(self, N: float, trap_frequencies_Hz: Sequence[float],
                 a_s_bohr: float = 100.0, mass: Optional[float] = None,
                 temperature_K: Optional[float] = None):
        self.N = float(N)
        self.trap_frequencies_Hz = np.asarray(trap_frequencies_Hz, dtype=float)
        if self.trap_frequencies_Hz.shape != (3,):
            raise ValueError("trap_frequencies_Hz must have three entries.")
        self.a_s_bohr = float(a_s_bohr)
        self.mass = float(c.m_K if mass is None else mass)
        self.temperature_K = temperature_K

    # ---------------------------------------------------------------- basics
    @property
    def omega(self) -> np.ndarray:
        """Trap angular frequencies (rad/s)."""
        return 2.0 * np.pi * self.trap_frequencies_Hz

    @property
    def omega_bar(self) -> float:
        """Geometric-mean trap angular frequency (rad/s)."""
        return float(np.prod(self.omega) ** (1.0 / 3.0))

    @property
    def a_s(self) -> float:
        """s-wave scattering length in metres."""
        return self.a_s_bohr * _A0

    @property
    def a_ho(self) -> float:
        """Mean harmonic-oscillator length (m)."""
        return float(np.sqrt(c.hbar / (self.mass * self.omega_bar)))

    @property
    def chemical_potential_J(self) -> float:
        """Thomas-Fermi chemical potential mu = (hbar wbar / 2) (15 N a / a_ho)^{2/5}."""
        return float(0.5 * c.hbar * self.omega_bar
                     * (15.0 * self.N * self.a_s / self.a_ho) ** 0.4)

    @property
    def coupling_g(self) -> float:
        """Mean-field coupling g = 4 pi hbar^2 a / m."""
        return 4.0 * np.pi * c.hbar ** 2 * self.a_s / self.mass

    @property
    def peak_density(self) -> float:
        """Peak density n0 = mu / g, in m^-3."""
        return self.chemical_potential_J / self.coupling_g

    @property
    def tf_radii(self) -> np.ndarray:
        """Thomas-Fermi radii (m) along each trap axis."""
        return np.sqrt(2.0 * self.chemical_potential_J
                       / (self.mass * self.omega ** 2))

    @property
    def tf_volume(self) -> float:
        """(4 pi / 3) Rx Ry Rz, in m^3."""
        return float(4.0 * np.pi / 3.0 * np.prod(self.tf_radii))

    @property
    def density_squared_integral(self) -> float:
        """Integral of n^2 dV, in m^-3.

        For a Thomas-Fermi profile this is (32 pi / 105) n0^2 Rx Ry Rz, so the
        density-weighted mean density is n2_int / N = 4 n0 / 7.
        """
        return float(32.0 * np.pi / 105.0 * self.peak_density ** 2
                     * np.prod(self.tf_radii))

    @property
    def mean_density(self) -> float:
        """Density-weighted mean density, 4 n0 / 7 (m^-3)."""
        return self.density_squared_integral / self.N

    @property
    def healing_length(self) -> float:
        """xi = hbar / sqrt(2 m mu), in m."""
        return float(c.hbar / np.sqrt(2.0 * self.mass * self.chemical_potential_J))

    # ------------------------------------------------------ pair statistics
    def g2(self, r):
        """Pair correlation function at separation r.

        A condensate has g2 = 1 (no bunching, unlike the factor 2 of a thermal
        cloud).  The optional short-range Jastrow factor (1 - a/r)^2 accounts for
        the two-body scattering hole; with a of order a few nm and Condon radii of
        50-160 nm this is a few-percent correction, but it is cheap to keep.
        """
        r = np.asarray(r, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            jastrow = (1.0 - self.a_s / r) ** 2
        return np.where(r > 0, np.clip(jastrow, 0.0, None), 0.0)

    def relative_velocity(self) -> float:
        """Characteristic relative velocity of a close pair (m/s).

        Uses sqrt(2 kB T / m) if a temperature was given, otherwise the mean-field
        scale sqrt(2 mu / m).  This is the quantity that decides quasi-static
        versus flux (Landau-Zener) behaviour: in a condensate it is of order
        1 cm/s, far too slow for the flux picture.
        """
        if self.temperature_K is not None:
            return float(np.sqrt(2.0 * c.kB * self.temperature_K / self.mass))
        return float(np.sqrt(2.0 * self.chemical_potential_J / self.mass))

    # ------------------------------------------------------------- sampling
    def sample_positions(self, n_samples: int, rng=None) -> np.ndarray:
        """Draw positions from the Thomas-Fermi density by rejection sampling.

        Returns
        -------
        (n_samples, 3) ndarray of positions in metres.
        """
        rng = np.random.default_rng() if rng is None else rng
        R = self.tf_radii
        out = np.empty((n_samples, 3))
        filled = 0
        while filled < n_samples:
            batch = max(n_samples - filled, 1024)
            pts = rng.uniform(-1.0, 1.0, size=(batch, 3))
            rho2 = np.sum(pts ** 2, axis=1)
            shape = 1.0 - rho2
            keep = (rho2 < 1.0) & (rng.uniform(size=batch) < shape)
            good = pts[keep]
            take = min(len(good), n_samples - filled)
            out[filled:filled + take] = good[:take] * R
            filled += take
        return out

    def summary(self) -> str:
        """Human-readable dump of the derived cloud quantities."""
        R = self.tf_radii * 1e6
        n0 = self.peak_density
        return "\n".join([
            "BECCloud  N = %.3e   a_s = %.1f a0" % (self.N, self.a_s_bohr),
            "  trap (Hz)      = (%.1f, %.1f, %.1f)" % tuple(self.trap_frequencies_Hz),
            "  mu/h           = %.3f kHz" % (self.chemical_potential_J / c.h / 1e3),
            "  peak density   = %.3e cm^-3" % (n0 / 1e6),
            "  mean density   = %.3e cm^-3" % (self.mean_density / 1e6),
            "  TF radii       = (%.2f, %.2f, %.2f) um" % tuple(R),
            "  healing length = %.1f nm" % (self.healing_length * 1e9),
            "  mean spacing   = %.1f nm" % (n0 ** (-1.0 / 3.0) * 1e9),
            "  v_rel scale    = %.2f cm/s" % (self.relative_velocity() * 100.0),
        ])

    def __repr__(self) -> str:
        return ("BECCloud(N=%.3e, peak_density=%.3e cm^-3)"
                % (self.N, self.peak_density / 1e6))
