"""From kamo.trap clouds to :mod:`kamo.imaging` (and :mod:`kamo.spin`).

kamo.imaging assumed a Gaussian cloud: :class:`~kamo.imaging.bpm.UniformMixture`
builds a separable Gaussian from ``cloud.widths``, and
:meth:`~kamo.imaging.bpm.Propagator.for_cloud`, the ``record_window`` default and
:meth:`kamo.spin.SpinGeometry.from_propagator` size their boxes from
``cloud.widths``.  Those last three uses are *box sizing*, so the criterion for
a non-Gaussian cloud's widths is containment, and :func:`effective_widths` uses
the rms widths ``w = sqrt(2 <x^2>)``: exact for a Gaussian (so nothing that
works today changes) and generous for a Thomas-Fermi profile (``w = 0.535 R``, so
the propagator's ``+-3 w`` clears the hard edge by 60%).  Peak-matched widths
would give a Thomas-Fermi cloud a box *smaller* than the cloud.

The density itself comes from :class:`GriddedMixture`, a sibling of
UniformMixture: the solver grid is resampled once onto the propagator's exact
slice midpoints (PCHIP along x -- monotone, so no negative density) and each
slice is mapped to the optical grid by two precomputed linear-interpolation
matrices, ``n = W_y P_i W_z^T``.  That is a pair of matmuls per slice, fast on
both backends, and well under the cost of the propagator's own FFTs.  It keeps
``species``, so the propagator stays on its array-generic (GPU) fast path.

By default the cloud is recentred on its centroid (``recenter=True``): the
gravitational sag moves a tweezer cloud ~0.3 um off the focus, cosmetic for the
physics but it keeps ``refocus`` and the far field symmetric and lets a GP cloud
and a Gaussian one be compared directly.  :class:`kamo.spin.SpinGeometry`
follows the same convention.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from kamo.imaging._backend import as_backend
from kamo.imaging.bpm import Propagator, SusceptibilitySource


# --------------------------------------------------------------- widths

def contained_fraction(cloud, half_widths, center=None) -> float:
    """Fraction of a gridded cloud's atoms inside ``center +- half_widths``."""
    g = cloud.grid
    c = np.asarray(cloud.centroid if center is None else center, dtype=float)
    h = np.broadcast_to(np.asarray(half_widths, dtype=float), (3,))
    inside = ((np.abs(g.x - c[0]) <= h[0])[:, None, None]
              & (np.abs(g.y - c[1]) <= h[1])[None, :, None]
              & (np.abs(g.z - c[2]) <= h[2])[None, None, :])
    return float(np.sum(cloud.density_grid * inside) / np.sum(cloud.density_grid))


def containment_half_widths(cloud, frac: float = 1.0 - 1e-6) -> np.ndarray:
    """Per lab axis, the half-width about the centroid holding ``frac`` of the
    atoms in that axis's marginal (m)."""
    g, rho = cloud.grid, cloud.density_grid
    out = np.empty(3)
    for i, (ax, marg) in enumerate(((g.x, rho.sum(axis=(1, 2))), (g.y, rho.sum(axis=(0, 2))),
                                    (g.z, rho.sum(axis=(0, 1))))):
        d = np.abs(ax - cloud.centroid[i])
        order = np.argsort(d)
        cum = np.cumsum(marg[order]) / marg.sum()
        k = int(np.searchsorted(cum, frac))
        out[i] = d[order][min(k, d.size - 1)] + 0.5 * g.d[i]
    return out


def effective_widths(cloud, kind: str = "rms") -> np.ndarray:
    """Widths standing in for a Gaussian's 1/e radii (m).

    ``"rms"`` (default): ``sqrt(2 <x_i^2>)``, what ``cloud.widths`` returns.
    ``"peak"``: the rms aspect ratio, scaled so ``N / (pi^1.5 prod w)`` is the
    peak density -- *not* for box sizing (see the module docstring).
    ``"containment"``: :func:`containment_half_widths` (1 - 1e-6 of the atoms).
    """
    w = np.asarray(cloud.widths, dtype=float)
    if kind == "rms":
        return w
    if kind == "peak":
        return w * (cloud.N / (np.pi ** 1.5 * cloud.peak_density * np.prod(w))) ** (1.0 / 3.0)
    if kind == "containment":
        return containment_half_widths(cloud)
    raise ValueError(f"kind must be 'rms', 'peak' or 'containment'; got {kind!r}")


# ------------------------------------------------------------ resampling

def _linear_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rows of linear-interpolation weights from ``src`` nodes to ``dst`` points;
    all-zero rows outside ``src``."""
    W = np.zeros((dst.size, src.size))
    inside = (dst >= src[0]) & (dst <= src[-1])
    k = np.clip(np.searchsorted(src, dst[inside]) - 1, 0, src.size - 2)
    t = (dst[inside] - src[k]) / (src[k + 1] - src[k])
    rows = np.flatnonzero(inside)
    W[rows, k] = 1.0 - t
    W[rows, k + 1] = t
    return W


class GriddedDensity:
    """A gridded cloud resampled onto an optical grid's slices.

    ``density(x)`` returns the ``(n, n)`` slice at the propagator slice ``x``
    (an exact match is required: build it for the propagator that uses it).
    Arrays live on ``backend``.
    """

    def __init__(self, cloud, x_slices, transverse_axis, backend=None,
                 density_scale: float = 1.0, recenter: bool = True):
        self.cloud = cloud
        self._bk = as_backend(backend)
        self.center = np.asarray(cloud.centroid if recenter else np.zeros(3), dtype=float)
        g = cloud.grid
        self._x = np.asarray(x_slices, dtype=float)
        rho = cloud.density_grid * float(density_scale)
        planes = PchipInterpolator(g.x, rho, axis=0, extrapolate=False)(self._x + self.center[0])
        planes = np.clip(np.nan_to_num(planes, nan=0.0), 0.0, None)
        ax = np.asarray(transverse_axis, dtype=float)
        Wy = _linear_matrix(g.y, ax + self.center[1])
        Wz = _linear_matrix(g.z, ax + self.center[2])
        d_opt = float(ax[1] - ax[0])
        dx = float(self._x[1] - self._x[0]) if self._x.size > 1 else 1.0
        total = sum(float(np.sum(Wy @ p @ Wz.T)) for p in planes) * dx * d_opt ** 2
        self.atom_number_error = total / (cloud.N * float(density_scale)) - 1.0
        self._planes = [self._bk.real(p) for p in planes]
        self._Wy, self._WzT = self._bk.real(Wy), self._bk.real(Wz.T)
        self._dx = dx

    def slice_index(self, x: float) -> int:
        i = int(np.argmin(np.abs(self._x - x)))
        if abs(self._x[i] - x) > 1e-9 * self._dx:
            raise ValueError(f"x = {x:.6g} m is not one of this density's slices: it was "
                             "built for a different propagator.")
        return i

    def density(self, x: float):
        return self._Wy @ self._planes[self.slice_index(x)] @ self._WzT


class GriddedMixture(SusceptibilitySource):
    """A spin mixture whose shared spatial profile is a gridded kamo.trap cloud.

    The gridded counterpart of :class:`kamo.imaging.bpm.UniformMixture`; build it
    with :meth:`for_propagator` so the slices line up exactly.
    """

    def __init__(self, cloud, response, species: Sequence[Tuple[float, float]], *,
                 x_slices, transverse_axis, backend=None, density_scale: float = 1.0,
                 recenter: bool = True):
        self.cloud = cloud
        self.response = response
        self.species = tuple((float(f), float(d)) for f, d in species)
        if not self.species:
            raise ValueError("species must list (fraction, delta) pairs: without them the "
                             "propagator falls back to an opaque chi() and the GPU refuses it.")
        self.widths = np.asarray(cloud.widths, dtype=float)
        self._density = GriddedDensity(cloud, x_slices, transverse_axis, backend=backend,
                                       density_scale=density_scale, recenter=recenter)
        self.center = self._density.center
        self.atom_number_error = self._density.atom_number_error

    @classmethod
    def for_propagator(cls, propagator: Propagator, cloud, response, species,
                       **kw) -> "GriddedMixture":
        """Slices and transverse grid taken from ``propagator`` (and its backend)."""
        ns = propagator.n_slices
        dx = 2.0 * propagator.x_edge / ns
        xs = -propagator.x_edge + (np.arange(ns) + 0.5) * dx
        kw.setdefault("backend", propagator.backend)
        return cls(cloud, response, species, x_slices=xs,
                   transverse_axis=propagator.grid.axis, **kw)

    def density(self, x, Y, Z):
        return self._density.density(float(x))

    def chi(self, x, Y, Z, s_local):
        n = self.density(x, Y, Z)
        total = 0.0
        for frac, delta in self.species:
            total = total + frac * self.response.susceptibility(n, delta, s_local)
        return total


def propagator_for(cloud, response, *, n_grid: int = 768, L_box: float = 36.0e-6,
                   x_span_w: float = 3.0, n_slices: int = 180, backend=None) -> Propagator:
    """:meth:`Propagator.for_cloud` for a gridded cloud, with the containment checks
    its Gaussian sizing does not make."""
    prop = Propagator.for_cloud(response, cloud, n_grid=n_grid, L_box=L_box,
                                x_span_w=x_span_w, n_slices=n_slices, backend=backend)
    w = np.asarray(cloud.widths, dtype=float)
    if hasattr(cloud, "density_grid"):
        frac = contained_fraction(cloud, [prop.x_edge, 0.5 * L_box, 0.5 * L_box])
        if frac < 1.0 - 1e-4:
            warnings.warn(f"the propagation box holds only {frac:.6f} of the atoms; raise "
                          "x_span_w or L_box.", UserWarning, stacklevel=2)
    if 0.5 * L_box < 4.0 * float(np.max(w[1:])):
        warnings.warn(f"L_box/2 = {0.5 * L_box * 1e6:.2f} um is under four transverse "
                      "widths; wraparound may matter.", UserWarning, stacklevel=2)
    return prop
