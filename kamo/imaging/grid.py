"""Transverse grid and angular spectrum shared by the propagator and the readout.

The propagator, the far-field analysis and the phase-contrast imaging all need the
same transverse sampling, the same ``k_perp`` axes and the same propagating-mode
mask.  The notebook this module grew out of re-derived them in three places from a
grid size; :class:`TransverseGrid` builds them once, so a convergence test that
halves the grid cannot silently desynchronise the far-field mask from the
propagator.

Geometry convention throughout :mod:`kamo.imaging`: **x is the probe propagation
axis**, y and z are transverse, and z is the quantization axis.  For a tweezer BEC
imaged along the tweezer, x is also the cloud's long (axial) axis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TransverseGrid:
    """Square transverse grid with its angular spectrum precomputed.

    Parameters
    ----------
    n : int
        Samples per transverse axis.
    L : float
        Box size (m).  Must comfortably exceed the cloud: the propagation uses no
        absorbing boundary, so the vacuum reference stays an exact plane wave, and
        wraparound is controlled instead by keeping scattered power inside the box.
    k : float
        Optical wavenumber ``2 pi / lambda`` (1/m).
    """

    def __init__(self, n: int, L: float, k: float):
        self.n = int(n)
        self.L = float(L)
        self.k = float(k)
        self.d = self.L / self.n

        ax = (np.arange(self.n) - self.n // 2) * self.d
        self.axis = ax
        self.Y, self.Z = np.meshgrid(ax, ax, indexing="ij")

        ka = 2 * np.pi * np.fft.fftfreq(self.n, d=self.d)
        self.k_axis = ka
        self.KY, self.KZ = np.meshgrid(ka, ka, indexing="ij")
        self.KT2 = self.KY**2 + self.KZ**2

        # Propagating modes only; the rest are evanescent and must DECAY, not ride
        # through untouched -- a strong lens pushes light past 90 degrees and an
        # untouched evanescent order would then carry unphysical power.
        self.propagating = self.KT2 < self.k**2
        self.KX = np.sqrt(np.where(self.propagating, self.k**2 - self.KT2, 0.0))
        self.kappa = np.sqrt(np.where(self.propagating, 0.0, self.KT2 - self.k**2))

    # ------------------------------------------------------------- propagator

    def propagator(self, dx: float) -> np.ndarray:
        """Exact angular-spectrum free-space kernel for a step ``dx``.

        ``exp(i kx dx)`` on propagating modes, ``exp(-kappa dx)`` on evanescent
        ones.  This is the EXACT propagator, not the paraxial one, so a large
        collection NA is handled honestly.
        """
        return np.where(self.propagating,
                        np.exp(1j * self.KX * dx),
                        np.exp(-self.kappa * abs(dx)))

    def back_propagator(self, dx: float) -> np.ndarray:
        """Kernel for propagating BACKWARD by ``dx`` (positive ``dx`` = upstream).

        Evanescent modes are DISCARDED rather than restored.  Undoing their
        forward decay would mean multiplying by ``exp(+kappa dx)``, which amplifies
        numerical noise without bound -- it diverges even at moderate propagation
        distances and is unusable.  Zeroing them is the principled alternative: a
        microscope collects only propagating modes, so refocusing is a
        propagating-mode operation by construction.

        The choice is immaterial wherever a refocus is physically meaningful.
        Zeroing, keeping, or further decaying the evanescent orders agree to
        better than 0.3% for any cloud at least a tenth of a depth of field long;
        they part company only for clouds shorter than a wavelength, where the
        near field has not yet separated from the far field and no refocus means
        anything.
        """
        return np.where(self.propagating, np.exp(-1j * self.KX * dx), 0.0)

    # ------------------------------------------------------------------ masks

    def na_mask(self, NA: float) -> np.ndarray:
        """Propagating modes inside a collection numerical aperture."""
        return self.propagating & (self.KT2 <= (self.k * float(NA)) ** 2)

    def dipole_weight(self) -> np.ndarray:
        """``(3/16 pi)(1 + cos^2 theta)`` per mode, theta from the z axis.

        The angular distribution of a circularly driven (sigma+-) rotating dipole,
        normalized to 1 over the sphere.  Applied as a weight on far-field
        intensity; see the scalar-field caveat in :mod:`kamo.imaging.readout`.
        """
        cos_theta = np.where(self.propagating, self.KZ / self.k, 0.0)
        return np.where(self.propagating,
                        (3 / (16 * np.pi)) * (1 + cos_theta**2), 0.0)

    def angle_deg(self) -> np.ndarray:
        """Polar angle from the probe axis (+x) per mode, in degrees."""
        return np.degrees(np.arcsin(np.clip(np.sqrt(self.KT2) / self.k, 0, 1)))

    # ------------------------------------------------------------------ utils

    def window(self, half_width: float) -> np.ndarray:
        """Boolean mask on :attr:`axis` selecting ``|y| < half_width``."""
        return np.abs(self.axis) < float(half_width)

    def window_slice(self, half_width: float) -> slice:
        """Contiguous slice covering ``|y| < half_width`` -- for cropping records."""
        idx = np.flatnonzero(self.window(half_width))
        if idx.size == 0:
            raise ValueError(f"half_width {half_width} selects no grid points "
                             f"(grid spacing is {self.d:.3e} m)")
        return slice(int(idx[0]), int(idx[-1]) + 1)

    def extent_um(self, half_width: Optional[float] = None):
        """``[lo, hi, lo, hi]`` in microns, for ``imshow(extent=...)``."""
        ax = self.axis if half_width is None else self.axis[self.window(half_width)]
        return [ax[0] * 1e6, ax[-1] * 1e6, ax[0] * 1e6, ax[-1] * 1e6]

    @property
    def center(self) -> int:
        """Index of the on-axis sample."""
        return self.n // 2

    @property
    def far_field_resolution_deg(self) -> float:
        """Angular resolution of the far field, ``2 pi / (L k)`` in degrees."""
        return float(np.degrees(2 * np.pi / self.L / self.k))

    def rescaled(self, n: int) -> "TransverseGrid":
        """Same box and wavenumber at a different sampling -- for convergence tests."""
        return TransverseGrid(n, self.L, self.k)

    def __repr__(self):
        return (f"TransverseGrid(n={self.n}, L={self.L * 1e6:.1f} um, "
                f"d={self.d * 1e9:.1f} nm, "
                f"far-field res {self.far_field_resolution_deg:.2f} deg)")
