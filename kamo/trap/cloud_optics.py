"""Optical properties of a gridded cloud, with no Gaussian assumed.

The numerical counterparts of the closed forms a Gaussian cloud allows: the
thin-screen phase and optical depth, the first-Born form factor and far field,
chord optical depths, and the emission-averaged escape probability that sets
reabsorption.  Everything takes a :class:`~kamo.trap.cloud.TrapCloud` (any mode),
so a Gross-Pitaevskii density goes through the same analysis a Gaussian one did.

>>> from kamo.trap import cloud_optics as co
>>> D, phi = co.thin_screen(cloud, response, [(1.0, +18.3)])
>>> ff = co.form_factor_sq(cloud, q)                   # q: (..., 3) in 1/m
>>> W = co.born_far_field(transverse_grid, cloud)      # on a kamo.imaging grid
>>> T = co.EscapeTable(cloud, sigma_sc)(nhat)          # nhat: (..., 3)

Geometry as in :mod:`kamo.imaging`: x is the probe axis.  The cloud is taken
about its centroid throughout, as :class:`~kamo.trap.imaging_bridge.GriddedMixture`
does by default.
"""

from __future__ import annotations

import copy
from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage

import kamo.constants as kc

from .grid import TrapGrid


def stretched(cloud, scale_x: float):
    """The same cloud squeezed or stretched along x about its centroid, atom
    number -- and so the column density along x -- held fixed."""
    s = float(scale_x)
    if s <= 0:
        raise ValueError("scale_x must be positive")
    g = cloud.grid
    x0 = float(cloud.centroid[0])
    out = copy.copy(cloud)
    out.grid = TrapGrid(x0 + s * (g.x - x0), g.y, g.z)
    out.density_grid = cloud.density_grid / s
    out._peak = None if cloud._peak is None else cloud._peak / s
    out._N_grid, out.centroid, var = out.grid.moments(out.density_grid)
    out.sigma = np.sqrt(var)
    out._interp = out._column = out._density_fn = None
    return out


def thin_screen(cloud, response, species: Sequence[Tuple[float, float]]):
    """``(D, phi)`` on the peak column along x for ``(fraction, delta)`` species.

    The slice exponent the propagator applies, taken over the whole column at
    once: ``D = -2 Re``, ``phi = Im``.
    """
    ncol = cloud.peak_column_density
    expo = sum(1j * response.k * 0.5 * f * ncol * response.polarizability(d) / kc.epsilon_0
               for f, d in species)
    return float(-2 * expo.real), float(expo.imag)


def peak_column_gradient(cloud) -> float:
    """Steepest transverse slope of the column density along x (1/m^3): with the
    thin-screen phase it bounds the ray deflection, ``|grad phi| / k``."""
    col = cloud.column_density_grid()
    gy, gz = np.gradient(col, cloud.grid.d[1], cloud.grid.d[2])
    return float(np.max(np.hypot(gy, gz)))


def chord_optical_depth(cloud, sigma_sc: float, axis: int = 0) -> float:
    """Optical depth of the densest full chord along lab axis ``axis``."""
    col = np.sum(cloud.density_grid, axis=axis) * cloud.grid.d[axis]
    return float(sigma_sc * np.max(col))


def _significant_voxels(cloud, keep: float = 1.0 - 1e-7):
    """Positions about the centroid and atom-number weights of the voxels that
    hold ``keep`` of the atoms (the rest of the box is empty padding)."""
    g = cloud.grid
    w = cloud.density_grid.ravel() * float(np.prod(g.d))
    order = np.argsort(w)[::-1]
    n = int(np.searchsorted(np.cumsum(w[order]), keep * w.sum())) + 1
    idx = np.unravel_index(order[:n], cloud.density_grid.shape)
    r = np.stack([g.x[idx[0]], g.y[idx[1]], g.z[idx[2]]], axis=-1) - cloud.centroid
    return r, w[order[:n]] / w[order[:n]].sum()


def form_factor_sq(cloud, q, chunk: int = 256) -> np.ndarray:
    """``|f(q)|^2`` with ``f = integral n(r) exp(i q.r) dV / N``, by direct sum.

    Exact on the solver grid (no FFT interpolation), so it is good out to
    ``|q| = 2k`` backscatter.  Cost is (number of q) x (voxels holding atoms);
    runs on the GPU when torch has one.
    """
    q = np.asarray(q, dtype=float)
    flat = q.reshape(-1, 3)
    r, w = _significant_voxels(cloud)
    out = np.empty(flat.shape[0])
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else None
    except ImportError:
        dev = None
    if dev:
        rt = torch.as_tensor(r, device=dev)
        wt = torch.as_tensor(w, device=dev).to(torch.complex128)
        qt = torch.as_tensor(flat, device=dev)
        step = max(chunk, int(2e8 // max(r.shape[0], 1)))
        for i in range(0, flat.shape[0], step):
            f = torch.exp(1j * (qt[i:i + step] @ rt.T)) @ wt
            out[i:i + step] = (f.real ** 2 + f.imag ** 2).cpu().numpy()
    else:
        for i in range(0, flat.shape[0], chunk):
            f = np.exp(1j * (flat[i:i + chunk] @ r.T)) @ w
            out[i:i + chunk] = f.real ** 2 + f.imag ** 2
    return out.reshape(q.shape[:-1])


def born_far_field(grid, cloud, n_slices: int = 240, x_span_w: float = 4.0) -> np.ndarray:
    """First-Born far field of the cloud on a :class:`kamo.imaging` TransverseGrid.

    The gridded counterpart of :func:`kamo.imaging.readout.born_far_field`:
    ``|f(q)|^2`` at ``q = (k_x - k, k_y, k_z)`` for every propagating mode, times
    the dipole weight and the obliquity factor.  Built slice by slice from 2D FFTs,
    so it shares the propagator's transverse sampling but none of its split-step,
    which is what makes it a reference for the weak-phase limit.
    """
    from .imaging_bridge import GriddedDensity

    x_edge = x_span_w * float(cloud.widths[0])
    dx = 2.0 * x_edge / n_slices
    xs = -x_edge + (np.arange(n_slices) + 0.5) * dx
    dens = GriddedDensity(cloud, xs, grid.axis)
    qx = np.where(grid.propagating, grid.KX - grid.k, 0.0)
    f = np.zeros(grid.KX.shape, dtype=complex)
    for x in xs:
        f += np.fft.fft2(np.asarray(dens.density(x))) * np.exp(-1j * qx * x)
    ff = np.abs(f * dx * grid.d ** 2 / cloud.N) ** 2
    return np.where(grid.propagating, ff * grid.dipole_weight() * (grid.KX / grid.k), 0.0)


def _escape_one(rho, d, center_idx, nhat, sigma_sc, h):
    """<exp(-sigma * column from r to infinity along nhat)> over emission points."""
    e3 = np.asarray(nhat, dtype=float) / np.linalg.norm(nhat)
    seed = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(seed, e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    # rotated sampling box: ray axis first, isotropic step h, big enough for any tilt
    n = int(np.ceil(np.linalg.norm(np.asarray(rho.shape) * d) / h)) | 1
    R = np.stack([e3, e1, e2], axis=1)                       # lab = R @ (s0, s1, s2)
    matrix = R * h / d[:, None]                              # output index -> input index
    rot = ndimage.affine_transform(rho, matrix, offset=center_idx - matrix @ np.full(3, n // 2),
                                   output_shape=(n, n, n), order=1, mode="constant", cval=0.0)
    ahead = (np.cumsum(rot[::-1], axis=0)[::-1] - 0.5 * rot) * h   # column from each voxel outward
    total = rot.sum()
    return float((rot * np.exp(-sigma_sc * ahead)).sum() / total) if total > 0 else 1.0


class EscapeTable:
    """Emission-point-averaged transmission out of a cloud, ``T(nhat)``.

    ``T(nhat) = (1/N) integral n(r) exp(-sigma_sc * int_0^inf n(r + s nhat) ds) dV``:
    what reabsorption leaves of light emitted along ``nhat``.  For a Gaussian this
    depends on direction only through the chord optical depth; for a real cloud it
    does not, so it is ray-marched once on a coarse ``n_u x n_phi`` grid of
    directions (``u = cos(theta)`` from z, ``phi`` from x) over one octant -- the
    cloud is taken as mirror symmetric in x, y and z, which ignores the
    gravitational skew along z -- and interpolated on every call.  ``step`` is the
    ray-march voxel (default: the finest grid spacing).
    """

    def __init__(self, cloud, sigma_sc: float, n_u: int = 9, n_phi: int = 7,
                 step: float | None = None):
        from scipy.interpolate import RegularGridInterpolator

        g = cloud.grid
        d = np.asarray(g.d, dtype=float)
        h = float(np.min(d)) if step is None else float(step)
        # ray-march only the box that holds atoms; the solver grid is mostly padding
        rho = cloud.density_grid
        occupied = np.nonzero(rho > 1e-6 * rho.max())
        lo = [int(i.min()) for i in occupied]
        rho = rho[tuple(slice(l, int(i.max()) + 1) for l, i in zip(lo, occupied))]
        origin = np.array([g.x[lo[0]], g.y[lo[1]], g.z[lo[2]]])
        center_idx = (np.asarray(cloud.centroid) - origin) / d
        self.sigma_sc = float(sigma_sc)
        self.u_nodes = np.linspace(0.0, 1.0, n_u)
        self.phi_nodes = np.linspace(0.0, 0.5 * np.pi, n_phi)
        self.table = np.empty((n_u, n_phi))
        for i, u in enumerate(self.u_nodes):
            s = np.sqrt(max(1.0 - u * u, 0.0))
            for j, phi in enumerate(self.phi_nodes):
                self.table[i, j] = _escape_one(rho, d, center_idx,
                                               (s * np.cos(phi), s * np.sin(phi), u),
                                               self.sigma_sc, h)
        self._interp = RegularGridInterpolator((self.u_nodes, self.phi_nodes), self.table,
                                               method="linear")

    def __call__(self, nhat) -> np.ndarray:
        nhat = np.asarray(nhat, dtype=float)
        n = nhat / np.linalg.norm(nhat, axis=-1, keepdims=True)
        u = np.clip(np.abs(n[..., 2]), 0.0, 1.0)
        phi = np.clip(np.arctan2(np.abs(n[..., 1]), np.abs(n[..., 0])), 0.0, 0.5 * np.pi)
        return self._interp(np.stack([u.ravel(), phi.ravel()], axis=-1)).reshape(u.shape)


def escape_probability(cloud, sigma_sc: float, nhat, **table_kwargs) -> np.ndarray:
    """One-off :class:`EscapeTable` lookup; build the table yourself to reuse it."""
    return EscapeTable(cloud, sigma_sc, **table_kwargs)(nhat)
