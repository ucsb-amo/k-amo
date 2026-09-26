"""The tensor-product box every kamo.trap density lives on.

:class:`TrapGrid` holds three uniform 1D axes and hands out broadcast views
(``X = x[:, None, None]`` and so on, the :class:`kamo.spin.SpinGeometry` idiom --
never ``meshgrid``, which triples the memory for nothing), the quadrature weight,
the FFT wavenumbers a split-step solver needs, moments and interpolation.

``TrapGrid.around(center, half_widths, n)`` uses **midpoint** nodes by default,
``x_i = c - L + (i + 1/2) dx``: no node sits exactly on the centre, where a
Thomas-Fermi profile in a harmonic trap has its cusp-free but singular-derivative
reference point, and the box is symmetric about ``c``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class TrapGrid:
    """Uniform tensor-product grid; ``x`` is the imaging (probe) axis."""

    def __init__(self, x, y, z):
        axes = []
        for name, a in (("x", x), ("y", y), ("z", z)):
            a = np.asarray(a, dtype=float)
            if a.ndim != 1 or a.size < 2:
                raise ValueError(f"{name} must be a 1D axis with >= 2 points")
            d = np.diff(a)
            if not (np.all(d > 0) and np.allclose(d, d[0], rtol=1e-9, atol=0)):
                raise ValueError(f"{name} must be uniform and increasing")
            axes.append(a)
        self.x, self.y, self.z = axes
        self.d = np.array([a[1] - a[0] for a in axes])

    @classmethod
    def around(cls, center, half_widths, n, midpoint: bool = True) -> "TrapGrid":
        """A box ``center +- half_widths`` with ``n`` points per axis (scalars broadcast)."""
        c = np.broadcast_to(np.asarray(center, dtype=float), (3,))
        h = np.broadcast_to(np.asarray(half_widths, dtype=float), (3,))
        n = np.broadcast_to(np.asarray(n, dtype=int), (3,))
        if np.any(h <= 0) or np.any(n < 2):
            raise ValueError("half_widths must be positive and n >= 2")
        if midpoint:
            d = 2.0 * h / n
            axes = [c[i] - h[i] + (np.arange(n[i]) + 0.5) * d[i] for i in range(3)]
        else:
            axes = [np.linspace(c[i] - h[i], c[i] + h[i], n[i]) for i in range(3)]
        return cls(*axes)

    # ---------------------------------------------------------- geometry
    @property
    def shape(self) -> tuple:
        return (self.x.size, self.y.size, self.z.size)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def dV(self) -> float:
        return float(np.prod(self.d))

    @property
    def X(self):
        return self.x[:, None, None]

    @property
    def Y(self):
        return self.y[None, :, None]

    @property
    def Z(self):
        return self.z[None, None, :]

    @property
    def center(self) -> np.ndarray:
        return np.array([0.5 * (a[0] + a[-1]) for a in (self.x, self.y, self.z)])

    @property
    def half_widths(self) -> np.ndarray:
        """Distance from the centre to the outer cell faces."""
        return np.array([0.5 * (a[-1] - a[0]) + 0.5 * d
                         for a, d in zip((self.x, self.y, self.z), self.d)])

    def k_squared(self) -> np.ndarray:
        """``kx^2 + ky^2 + kz^2`` on the FFT grid, broadcast to the full shape (1/m^2)."""
        k = [2.0 * np.pi * np.fft.fftfreq(n, d) for n, d in zip(self.shape, self.d)]
        return k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2

    # -------------------------------------------------------- quadrature
    def integrate(self, f) -> float:
        """``sum(f) dV``, accumulated in double."""
        return float(np.sum(f, dtype=np.float64) * self.dV)

    def moments(self, density):
        """``(N, centroid (3,), variance (3,))`` of a density on the grid."""
        N = self.integrate(density)
        if N <= 0:
            return N, np.full(3, np.nan), np.full(3, np.nan)
        marg = [np.sum(density, axis=(1, 2)), np.sum(density, axis=(0, 2)),
                np.sum(density, axis=(0, 1))]
        total = [float(np.sum(m)) for m in marg]
        axes = (self.x, self.y, self.z)
        c = np.array([float(np.sum(m * a)) / t for m, a, t in zip(marg, axes, total)])
        var = np.array([float(np.sum(m * (a - ci) ** 2)) / t
                        for m, a, ci, t in zip(marg, axes, c, total)])
        return N, c, var

    def face_max(self, f) -> float:
        """Largest value on the six faces of the box (the wraparound / clipping check)."""
        return float(max(np.max(np.abs(f[0])), np.max(np.abs(f[-1])),
                         np.max(np.abs(f[:, 0])), np.max(np.abs(f[:, -1])),
                         np.max(np.abs(f[:, :, 0])), np.max(np.abs(f[:, :, -1]))))

    def nearest_index(self, point) -> tuple:
        p = np.asarray(point, dtype=float)
        return tuple(int(np.argmin(np.abs(a - pi))) for a, pi in zip((self.x, self.y, self.z), p))

    def interpolator(self, values, fill_value: float = 0.0) -> RegularGridInterpolator:
        """Trilinear interpolation of gridded values; ``fill_value`` outside the box."""
        return RegularGridInterpolator((self.x, self.y, self.z), np.asarray(values),
                                       method="linear", bounds_error=False,
                                       fill_value=fill_value)

    def __repr__(self) -> str:
        h = self.half_widths * 1e6
        return (f"TrapGrid({self.shape[0]}x{self.shape[1]}x{self.shape[2]}, "
                f"+-({h[0]:.3g}, {h[1]:.3g}, {h[2]:.3g}) um)")


def basin_mask(V, grid: TrapGrid, minimum, V_escape: float, epsilon: float = 0.0) -> np.ndarray:
    """The connected component of ``{V < V_escape}`` holding ``minimum``
    (6-connectivity).  Everything else -- above the escape energy, or a downhill
    pocket of the box that gravity pulls below it -- is not part of the trap, and
    a solver that fills it finds the box corner instead.  All True when
    ``V_escape`` is infinite (a HarmonicTrap).

    ``epsilon > 0`` cuts at ``V_min + (1 - epsilon) (V_escape - V_min)`` instead
    (``V_min`` the potential at the node nearest ``minimum``): at exactly the
    saddle energy the grid nodes straddling the saddle fall marginally below it,
    the component leaks through and follows gravity to the box corner.  A
    thermal density integrated over such a basin is unbounded; the ground-state
    solvers do not notice.  ``epsilon = 1e-3`` bounds it and moves the basin
    volume by 0.2% (measured 2026-09-13 on the 3 um / 1 kHz tweezer)."""
    from scipy import ndimage
    if not np.isfinite(V_escape):
        return np.ones(np.shape(V), dtype=bool)
    V = np.asarray(V)
    if epsilon > 0:
        V_min = float(V[grid.nearest_index(minimum)])
        V_escape = V_min + (1.0 - float(epsilon)) * (V_escape - V_min)
    labels, _ = ndimage.label(V < V_escape)
    lab = labels[grid.nearest_index(minimum)]
    if lab == 0:
        raise ValueError("the trap minimum is not inside {V < V_escape} on this grid")
    return labels == lab


def touched_axes(mask) -> np.ndarray:
    """Per lab axis, True if any True node of ``mask`` sits on either face of that axis."""
    m = np.asarray(mask, dtype=bool)
    return np.array([m[0].any() or m[-1].any(), m[:, 0].any() or m[:, -1].any(),
                     m[:, :, 0].any() or m[:, :, -1].any()])


def touches_face(mask) -> bool:
    """True if any True node of ``mask`` sits on one of the six faces of the box."""
    return bool(touched_axes(mask).any())
