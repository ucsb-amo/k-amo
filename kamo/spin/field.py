"""A spatially resolved spin state for a two-component cloud.

The state is a **three-dimensional field of mean spin vectors**, one per voxel of
the propagation grid.  Why 3D and not a 2D map of columns, which is all an image
can resolve:

A pure z-rotation commutes with ``S_z``, so the imaging pulse's differential light
shift leaves populations -- and hence the refractive index -- untouched everywhere.
It writes its structure into the *phase*.  But the local intensity varies strongly
ALONG the probe (a converging cloud concentrates the beam by several-fold over its
own length), so that phase is a genuinely three-dimensional grating.  A closing
pi/2 pulse then converts it into a three-dimensional POPULATION grating,
``S_z(r) = (1/2) C(r) cos(phi_z(r) + phi_0)``, which the next probe pulse
propagates through.  Averaging the phase over columns before that conversion would
destroy exactly the structure the propagation exists to resolve -- the cloud is
about one depth of field long, which is why a thin screen fails on it in the first
place.

So the field is 3D, and column quantities are DERIVED from it
(:meth:`SpinField.collapse_to_columns`) rather than being the state.

Representation
--------------
Mean-field Bloch vectors: ``s = (sx, sy, sz)`` per voxel with ``|s| <= 1/2``.
``sz = xi/2`` sets the populations, ``|s_perp|`` the surviving coherence.  This
captures rotations, inhomogeneous dephasing and spontaneous-scattering
decoherence.  It does NOT carry quantum correlations; for shot-to-shot projection
noise use :meth:`SpinField.sample_shot`, and see :mod:`kamo.spin` for the
Gaussian-moments extension point.

Quick start
-----------
>>> from kamo.spin import SpinField, SpinGeometry
>>> geom = SpinGeometry.from_propagator(prop, cloud)
>>> field = SpinField.spin_coherent(geom, Sz_total=0.0)   # equal superposition
>>> field.rotate_y(np.pi / 2)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from kamo.imaging.bpm import SusceptibilitySource


class SpinGeometry:
    """The voxel grid a :class:`SpinField` lives on.

    Axially these are exactly the propagator's atom slices; transversely a
    contiguous window of the propagation grid, cropped to where the cloud actually
    is.  Registration with the optical grid is therefore exact -- no interpolation
    anywhere -- while the memory stays bounded: the full grid at imaging resolution
    would be mostly empty box.
    """

    def __init__(self, grid, x_slices, window: slice, cloud,
                 widths: Optional[np.ndarray] = None):
        self.grid = grid
        self.x = np.asarray(x_slices, dtype=float)
        self.window = window
        self.cloud = cloud
        self.widths = np.asarray(cloud.widths if widths is None else widths,
                                 dtype=float)

        self.y = grid.axis[window]
        self.z = grid.axis[window]
        self.nw = len(self.y)
        self.n_slices = len(self.x)
        self.shape = (self.n_slices, self.nw, self.nw)

        X = self.x[:, None, None]
        Y = self.y[None, :, None]
        Z = self.z[None, None, :]
        w = self.widths
        peak = cloud.N / (np.pi**1.5 * np.prod(w))
        self.density = peak * np.exp(-(X**2 / w[0]**2 + Y**2 / w[1]**2
                                       + Z**2 / w[2]**2))
        self.dx = float(self.x[1] - self.x[0]) if self.n_slices > 1 else 0.0
        self.dy = float(grid.d)

    @classmethod
    def from_propagator(cls, propagator, cloud, window_half_width=None,
                        widths=None) -> "SpinGeometry":
        """Build the geometry that matches a propagator's slicing exactly."""
        w = np.asarray(cloud.widths if widths is None else widths, dtype=float)
        if window_half_width is None:
            window_half_width = 4.0 * float(np.max(w[1:]))
        ns = propagator.n_slices
        dx = 2 * propagator.x_edge / ns
        xs = -propagator.x_edge + (np.arange(ns) + 0.5) * dx
        return cls(propagator.grid, xs, propagator.grid.window_slice(window_half_width),
                   cloud, widths=w)

    @property
    def window_half_width(self) -> float:
        """Half-width (m) of the transverse crop."""
        return float(max(abs(self.y[0]), abs(self.y[-1])))

    @property
    def column_atom_number(self) -> np.ndarray:
        """Atoms per transverse pixel column, ``(nw, nw)``.

        Sets the local projection-noise scale in :meth:`SpinField.sample_shot`.
        """
        return self.density.sum(axis=0) * self.dx * self.dy**2

    def __repr__(self):
        return (f"SpinGeometry({self.n_slices} slices x {self.nw}^2, "
                f"window +-{self.window_half_width * 1e6:.2f} um)")


class SpinField:
    """Mean-field Bloch-vector field over a :class:`SpinGeometry`.

    Attributes
    ----------
    sx, sy, sz : ndarray
        Mean spin components per voxel, each of shape ``geometry.shape``.
        ``|s| <= 1/2``; a pure state saturates it.
    """

    def __init__(self, geometry: SpinGeometry, sx, sy, sz):
        self.geometry = geometry
        self.sx = np.asarray(sx, dtype=float)
        self.sy = np.asarray(sy, dtype=float)
        self.sz = np.asarray(sz, dtype=float)
        for name, a in (("sx", self.sx), ("sy", self.sy), ("sz", self.sz)):
            if a.shape != geometry.shape:
                raise ValueError(f"{name} has shape {a.shape}, "
                                 f"expected {geometry.shape}")

    # ------------------------------------------------------------ construction

    @classmethod
    def spin_coherent(cls, geometry: SpinGeometry, Sz_total: float,
                      phase: float = 0.0) -> "SpinField":
        """Uniform spin-coherent state with normalized imbalance ``Sz_total``.

        Parameters
        ----------
        Sz_total : float
            ``2 <S_z> / N = p_up - p_dn`` in [-1, 1].  ``+1`` is all ``|up>``,
            ``-1`` all ``|dn>``, ``0`` an equal superposition.
        phase : float
            Azimuth of the transverse spin (rad).

        The coherence follows from the imbalance for a pure state:
        ``|s_perp| = (1/2) sqrt(1 - Sz_total^2)``.  A fully polarized cloud has
        none, which is why the balanced state is the sensitive one.
        """
        xi = float(Sz_total)
        if not -1.0 <= xi <= 1.0:
            raise ValueError(f"Sz_total must be in [-1, 1], got {xi}")
        perp = 0.5 * np.sqrt(max(0.0, 1.0 - xi**2))
        ones = np.ones(geometry.shape)
        return cls(geometry,
                   perp * np.cos(phase) * ones,
                   perp * np.sin(phase) * ones,
                   0.5 * xi * ones)

    def copy(self) -> "SpinField":
        return SpinField(self.geometry, self.sx.copy(), self.sy.copy(),
                         self.sz.copy())

    # -------------------------------------------------------------- observables

    @property
    def xi(self) -> np.ndarray:
        """Per-voxel population imbalance ``p_up - p_dn = 2 sz``."""
        return 2.0 * self.sz

    @property
    def coherence(self) -> np.ndarray:
        """Per-voxel transverse coherence ``2 |s_perp|``, 1 for a pure equatorial state."""
        return 2.0 * np.hypot(self.sx, self.sy)

    @property
    def azimuth(self) -> np.ndarray:
        """Per-voxel transverse phase ``atan2(sy, sx)`` (rad)."""
        return np.arctan2(self.sy, self.sx)

    @property
    def purity(self) -> np.ndarray:
        """``2 |s|``; 1 for a pure spin, less once decohered."""
        return 2.0 * np.sqrt(self.sx**2 + self.sy**2 + self.sz**2)

    def mean_spin(self) -> np.ndarray:
        """Density-weighted mean ``(sx, sy, sz)`` over the whole cloud."""
        n = self.geometry.density
        den = n.sum()
        return np.array([float((n * c).sum() / den)
                         for c in (self.sx, self.sy, self.sz)])

    @property
    def Sz_total(self) -> float:
        """Cloud-averaged normalized imbalance -- the scalar the user specifies."""
        return float(2.0 * self.mean_spin()[2])

    @property
    def contrast(self) -> float:
        """Cloud-integrated Ramsey contrast, ``2 |<s_perp>|``.

        Distinct from the mean of :attr:`coherence`: this averages the transverse
        spin as a VECTOR, so voxels that have precessed to different azimuths
        cancel.  That difference is exactly inhomogeneous dephasing -- a cloud can
        be everywhere locally coherent and still show no fringe.
        """
        m = self.mean_spin()
        return float(2.0 * np.hypot(m[0], m[1]))

    def collapse_to_columns(self):
        """Density-weighted column averages ``(xi_col, Z_col)``.

        ``xi_col(y, z)`` is the column population imbalance -- the quantity that
        actually sets each column's refractive phase.  ``Z_col(y, z)`` is the
        complex transverse order parameter

            Z = int n exp(i phi) dx / int n dx,

        whose modulus is the within-column coherence surviving axial dephasing and
        whose argument is that column's mean precession angle.
        """
        n = self.geometry.density
        den = n.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            xi_col = (n * self.xi).sum(axis=0) / den
            Z_col = ((n * (self.sx + 1j * self.sy)).sum(axis=0) / den) * 2.0
        return np.nan_to_num(xi_col), np.nan_to_num(Z_col)

    # --------------------------------------------------------------- operations
    #
    # Extension point: a Gaussian-moments state (carrying <S> AND its covariance,
    # for squeezing/QND backaction) subclasses SpinField and overrides these two
    # primitives.  Every operator in kamo.spin.operations is written in terms of
    # them, so nothing above this line needs to change.

    def _apply_rotation(self, axis, angle) -> "SpinField":
        """Rodrigues rotation of every voxel about ``axis`` by ``angle``.

        ``angle`` may be a scalar or an array broadcasting to the field shape.
        """
        k = np.asarray(axis, dtype=float)
        k = k / np.linalg.norm(k)
        th = np.asarray(angle, dtype=float)
        c, s = np.cos(th), np.sin(th)

        # Cardinal axes get an exact path.  The general Rodrigues form computes the
        # invariant component as v*cos + v*(1 - cos), which is NOT exactly v in
        # floating point -- it drifts by ~1e-15 per rotation.  For z that would
        # quietly break the exact invariance of S_z that the whole design rests on
        # (see kamo.spin.operations), so keep it exact rather than nearly so.
        for i, (kx, ky, kz) in enumerate(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))):
            if k[0] == kx and k[1] == ky and k[2] == kz:
                j, l = (i + 1) % 3, (i + 2) % 3
                v = [self.sx, self.sy, self.sz]
                vj, vl = v[j], v[l]
                v[j] = vj * c - vl * s
                v[l] = vj * s + vl * c
                self.sx, self.sy, self.sz = v
                return self

        v = (self.sx, self.sy, self.sz)
        kdotv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
        cross = (k[1] * v[2] - k[2] * v[1],
                 k[2] * v[0] - k[0] * v[2],
                 k[0] * v[1] - k[1] * v[0])
        out = [v[i] * c + cross[i] * s + k[i] * kdotv * (1 - c) for i in range(3)]
        self.sx, self.sy, self.sz = out
        return self

    def _apply_decoherence(self, factor) -> "SpinField":
        """Shrink the TRANSVERSE spin by ``factor``, leaving populations alone.

        This is the map a which-path measurement makes: it destroys coherence
        without moving ``S_z``.  ``factor`` broadcasts to the field shape.
        """
        f = np.asarray(factor, dtype=float)
        self.sx = self.sx * f
        self.sy = self.sy * f
        return self

    def rotate(self, axis, angle) -> "SpinField":
        """Rotate every voxel about ``axis`` by ``angle`` (rad).  In place."""
        return self._apply_rotation(axis, angle)

    def rotate_x(self, angle) -> "SpinField":
        return self._apply_rotation((1.0, 0.0, 0.0), angle)

    def rotate_y(self, angle) -> "SpinField":
        return self._apply_rotation((0.0, 1.0, 0.0), angle)

    def rotate_z(self, angle) -> "SpinField":
        """Precess about z.  ``angle`` may be a full 3D map -- the usual case, since
        the light shift follows the local intensity."""
        return self._apply_rotation((0.0, 0.0, 1.0), angle)

    def decohere(self, factor) -> "SpinField":
        """Multiply the transverse spin by ``factor`` (in place)."""
        return self._apply_decoherence(factor)

    # ------------------------------------------------------------------- noise

    def noise_block_size(self, min_atoms: float = 30.0) -> int:
        """Side length (in pixels) of the cell projection noise is drawn on.

        The noise granularity must NOT be the pixel.  A spin-coherent state is a
        product state over ATOMS, so a region holding ``N`` of them has imbalance
        variance ``(1 - xi^2)/N``; on a fine imaging grid a single pixel column
        holds of order one atom, the implied standard deviation exceeds the
        physical range of ``xi``, and clipping to [-1, 1] then silently suppresses
        the total variance -- by 36% at 768^2 in the case this module was built for.
        Worse, the suppression depends on the grid, so the projection noise would
        change when the optics were refined.

        Blocks are therefore sized so a typical one holds ``min_atoms``, which
        keeps every draw well inside range.  The total variance is unaffected by
        the choice: summing ``(N_b/N)^2 (1 - xi^2)/N_b`` over blocks gives
        ``(1 - xi^2)/N`` for any partition.

        The default of 30 is set by the WINGS, not the centre.  Blocks out where
        the density has fallen still hold few atoms and still clip; at 10 the
        residual bias is about 4%, and at 30 it is within the sampling error of
        the target.  Larger values cost spatial resolution in the noise for no
        further gain.
        """
        peak = float(np.max(self.geometry.column_atom_number))
        if peak <= 0:
            return 1
        return max(1, int(np.ceil(np.sqrt(min_atoms / peak))))

    def sample_shot(self, rng=None, inplace: bool = False,
                    min_atoms: float = 30.0) -> "SpinField":
        """Draw one shot's projection noise on ``S_z``.

        A spin-coherent state is a PRODUCT state over atoms, so ``S_z`` fluctuates
        with variance ``(1 - xi^2)/N`` -- independently in each region, since
        different atoms are independent.  With the balanced state that scatter is
        the entire signal a dispersive spin measurement sees: at the midpoint there
        is no mean phase to read, only its shot-to-shot variance.

        The imbalance is redrawn per spatial cell (see :meth:`noise_block_size`)
        and applied uniformly down each column -- every atom in a column was
        prepared by the same pulse -- with the transverse spin rescaled to keep
        each voxel pure.
        """
        rng = np.random.default_rng() if rng is None else rng
        g = self.geometry
        out = self if inplace else self.copy()

        xi_col, _ = out.collapse_to_columns()
        N_col = np.maximum(g.column_atom_number, 0.0)

        b = out.noise_block_size(min_atoms)
        nw = g.nw
        idx = np.arange(nw) // b
        nb = int(idx.max()) + 1
        block = (idx[:, None] * nb + idx[None, :]).ravel()

        N_blk = np.bincount(block, weights=N_col.ravel(), minlength=nb * nb)
        xi_blk = np.bincount(block, weights=(N_col * xi_col).ravel(),
                             minlength=nb * nb)
        safe = np.maximum(N_blk, 1e-30)
        xi_blk = np.where(N_blk > 0, xi_blk / safe, 0.0)
        sd = np.sqrt(np.clip(1.0 - xi_blk**2, 0.0, None) / safe)
        drawn = np.clip(xi_blk + rng.normal(0.0, 1.0, N_blk.shape) * sd, -1.0, 1.0)
        delta = (drawn - xi_blk)[block].reshape(nw, nw)[None, :, :]
        out.sz = np.clip(out.sz + 0.5 * delta, -0.5, 0.5)
        # keep each voxel's purity: rescale the transverse part to the new |s_perp|
        new_perp = 0.5 * np.sqrt(np.clip(1.0 - (2 * out.sz) ** 2, 0.0, None))
        old_perp = np.hypot(out.sx, out.sy)
        scale = np.where(old_perp > 1e-15, new_perp / np.where(old_perp > 1e-15,
                                                               old_perp, 1.0), 0.0)
        out.sx = out.sx * scale
        out.sy = out.sy * scale
        return out

    # ------------------------------------------------------------ optics bridge

    def susceptibility_source(self, response, probe) -> "SpinFieldSource":
        """Wrap this field as a medium the propagator can step through."""
        return SpinFieldSource(self, response, probe)

    def summary(self) -> str:
        m = self.mean_spin()
        xi_col, Z_col = self.collapse_to_columns()
        return "\n".join([
            f"SpinField over {self.geometry!r}",
            f"  <s> = ({m[0]:+.4f}, {m[1]:+.4f}, {m[2]:+.4f})   "
            f"Sz_total = {self.Sz_total:+.4f}",
            f"  cloud-integrated contrast = {self.contrast:.4f}",
            f"  local coherence: mean {self.coherence.mean():.4f}, "
            f"min {self.coherence.min():.4f}, max {self.coherence.max():.4f}",
            f"  column xi in [{xi_col.min():+.4f}, {xi_col.max():+.4f}], "
            f"|Z_col| in [{np.abs(Z_col).min():.4f}, {np.abs(Z_col).max():.4f}]",
        ])

    def __repr__(self):
        return (f"SpinField(Sz_total={self.Sz_total:+.4f}, "
                f"contrast={self.contrast:.4f}, shape={self.geometry.shape})")


class SpinFieldSource(SusceptibilitySource):
    """Adapts a :class:`SpinField` into a medium for the propagator.

    Each voxel contributes ``n(r) [p_up(r) alpha(delta_up) + p_dn(r) alpha(delta_dn)]``:
    susceptibilities add, weighted by the LOCAL populations, so a spatially
    structured ``S_z`` produces a spatially structured index.  This is the one
    thing a uniform mixture cannot express, and it is what makes a second imaging
    pulse see anything at all.

    Outside the spin field's transverse window the density has fallen by ``e^-16``.
    Those voxels take the field's density-weighted mean imbalance rather than being
    treated as balanced, so a UNIFORM spin field reproduces
    :class:`~kamo.imaging.bpm.UniformMixture` exactly rather than to within the
    out-of-window density.
    """

    def __init__(self, field: SpinField, response, probe):
        self.field = field
        self.response = response
        self.probe = probe
        self.geometry = field.geometry
        self.widths = self.geometry.widths
        self.cloud = self.geometry.cloud
        self._win = self.geometry.window
        self._x = self.geometry.x

    def _background_xi(self) -> float:
        """Imbalance to assume outside the window: the field's own mean.

        The density there is down by ``e^-16``, so the choice is numerically
        irrelevant -- but taking the mean rather than zero makes a uniform field
        agree with UniformMixture identically, which is what lets the two layers
        be cross-checked exactly.
        """
        return float(self.field.Sz_total)

    def _index(self, x: float) -> int:
        i = int(np.searchsorted(self._x, x))
        if i >= len(self._x):
            return len(self._x) - 1
        if i > 0 and abs(self._x[i - 1] - x) < abs(self._x[i] - x):
            return i - 1
        return i

    def density(self, x, Y, Z):
        w = self.widths
        peak = self.cloud.N / (np.pi**1.5 * np.prod(w))
        return peak * np.exp(-(x**2 / w[0]**2 + Y**2 / w[1]**2 + Z**2 / w[2]**2))

    def chi(self, x, Y, Z, s_local):
        n = self.density(x, Y, Z)
        xi = np.full(n.shape, self._background_xi())
        xi[self._win, self._win] = self.field.xi[self._index(x)]
        s = np.asarray(s_local) if np.ndim(s_local) else s_local
        a_up = self.response.polarizability(self.probe.delta_up, s)
        a_dn = self.response.polarizability(self.probe.delta_dn, s)
        import kamo.constants as kc
        return n * (0.5 * (1 + xi) * a_up + 0.5 * (1 - xi) * a_dn) / kc.epsilon_0
