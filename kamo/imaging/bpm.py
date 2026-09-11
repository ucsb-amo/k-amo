"""Split-step angular-spectrum propagation of a probe through an atomic cloud.

When a cloud is a strong phase object AND as long as the imaging depth of field,
neither of the two usual shortcuts survives: first Born needs ``|phi| << 1``, and a
single thin phase screen needs the cloud short compared with ``k w_perp^2``.  A
tweezer BEC probed a few linewidths off resonance violates both -- the peak phase
runs to several radians and light diffracts *inside* the cloud.  This module
propagates the field through the three-dimensional density instead, alternating

    free step   E(k_perp) *= exp(i sqrt(k^2 - k_perp^2) dx)
    atom step   E(y, z)   *= exp(i k (n_ref - 1) dx)

with the EXACT angular-spectrum propagator (not the paraxial one) and the full
complex polarizability (not just its absorptive part).  Because light scattered by
each slice accumulates a phase ``exp(i(k_x - k)x)`` relative to the probe, this
reproduces the full three-dimensional form factor, including the ``q_x`` dependence
a projected phase screen discards.

Quick start
-----------
>>> from kamo.imaging import Propagator, UniformMixture
>>> source = UniformMixture(cloud, response, probe.species(xi=1.0))
>>> prop = Propagator.for_cloud(response, cloud)
>>> result = prop.propagate(source, s0_incident=probe.s0_incident)
>>> result.psi_exit              # E / E_vac at the exit plane

Standing assumptions
--------------------
- **Independent atoms.**  ``n_ref - 1 = chi/2`` unless the response was built with
  ``local_field=True``; at ``n/k^3`` of order 0.25 the Lorentz-Lorenz correction is
  several percent and recurrent scattering is unbounded.
- **Scalar field.**  The vector dipole pattern is applied as a weight on far-field
  intensity (see :mod:`kamo.imaging.readout`), which is acceptable while the
  coherent light stays in a forward cone.
- **Elastic scattering only.**  The propagation carries the coherent Rayleigh
  component; incoherent fluorescence is a separate channel
  (:mod:`kamo.imaging.farfield`).
- **No absorbing boundary**, so the vacuum reference is exactly ``exp(i k L)`` and
  both ``psi = E/E_vac`` and ``E - E_vac`` are free of edge artefacts.  Wraparound
  is controlled by making the box large enough that negligible scattered power
  reaches its edge -- check with :meth:`PropagationResult.edge_power_fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.fft as _sfft

from ._backend import ArrayBackend, array_exp, as_backend
from .grid import TransverseGrid
from .response import TwoLevelResponse

_FFT_WORKERS = -1


def _fft2(a):
    return _sfft.fft2(a, workers=_FFT_WORKERS)


def _ifft2(a):
    return _sfft.ifft2(a, workers=_FFT_WORKERS)


# --------------------------------------------------------------------- sources


class SusceptibilitySource:
    """What the propagator needs from a medium, slice by slice.

    Implement :meth:`chi` to be propagated through.  :meth:`density` is optional
    but lets the propagator report density-weighted diagnostics -- notably the mean
    intensity an average ATOM sees, which is not the intensity the probe delivers
    to the front face once the cloud lenses.
    """

    def chi(self, x: float, Y: np.ndarray, Z: np.ndarray, s_local) -> np.ndarray:
        """Complex susceptibility on the transverse grid at axial position ``x``."""
        raise NotImplementedError

    def density(self, x: float, Y: np.ndarray, Z: np.ndarray) -> Optional[np.ndarray]:
        """Atomic density (m^-3) at ``x``, or None if unavailable."""
        return None


class UniformMixture(SusceptibilitySource):
    """A cloud whose spin composition is the same everywhere.

    Parameters
    ----------
    cloud : object
        Anything with a ``density(x, y, z)`` method, e.g.
        :class:`~kamo.BEC_properties.variational.GaussianVariationalCloud`.
    response : TwoLevelResponse
        The atomic response.
    species : sequence of (fraction, delta)
        Population fractions and reduced detunings sharing one spatial profile.
        Susceptibilities add, which is what a spin mixture does to the linear
        response -- so a balanced mixture at the midpoint cancels ``Re alpha``
        exactly while leaving ``Im alpha`` untouched.
    density_scale : float
        Multiply the density by this.  Used to reach the weak-phase limit in
        validation without changing anything else.
    wx_scale : float
        Multiply the AXIAL width by this at fixed atom number.  Since the peak
        density carries a ``1/w_x``, this holds the COLUMN density -- and hence the
        thin-screen phase -- constant while making the cloud optically thin along
        the probe.  That separates a genuine depth-of-field effect from a bug.
    w_scale : (3,) sequence, optional
        Multiply all three widths.  Applied before ``wx_scale``.  Use it with a
        compensating ``density_scale`` to reach limits the real cloud is not in --
        e.g. scaling the two transverse widths by ``f`` with
        ``density_scale = f**2`` holds the column density fixed while making the
        cloud transversely EXTENDED, which is where Beer's law is recovered.
    """

    def __init__(self, cloud, response: TwoLevelResponse,
                 species: Sequence[Tuple[float, float]],
                 density_scale: float = 1.0, wx_scale: float = 1.0,
                 w_scale=None):
        self.cloud = cloud
        self.response = response
        self.species = tuple((float(f), float(d)) for f, d in species)
        self.density_scale = float(density_scale)
        self.wx_scale = float(wx_scale)
        self.w_scale = (np.ones(3) if w_scale is None
                        else np.asarray(w_scale, dtype=float) * np.ones(3))

        w = np.asarray(cloud.widths, dtype=float).copy() * self.w_scale
        w[0] *= self.wx_scale
        self.widths = w
        self._tY = self._tZ = self._transverse = None   # transverse-profile cache
        # n_peak carries 1/prod(w), so scaling w_x at fixed N preserves the column
        self.peak_density = (self.density_scale * cloud.N
                             / (np.pi**1.5 * np.prod(w)))

    def density(self, x, Y, Z):
        # A Gaussian cloud SEPARATES: the transverse factor is the same at every
        # slice, so it is built once per (Y, Z) pair and the axial dependence enters
        # as a scalar.  Rebuilding the whole 3D exponential every slice costs an
        # n_grid^2 exp per step, worth ~15% of the CPU runtime at 768^2 (21.0 s ->
        # 17.9 s measured); the larger share of the gap against a hand-written loop
        # was complex arithmetic in the slice operator, fixed in
        # TwoLevelResponse.mixture_slice_exponent.
        #
        # Cached against the ARRAYS, not their ids, and the references are held, so
        # a freed array's id cannot be reused underneath the check.  array_exp
        # rather than np.exp: on the GPU path Y and Z arrive as device arrays, and
        # the profile has to be built where the field already lives.
        w = self.widths
        if self._tY is not Y or self._tZ is not Z:
            self._transverse = array_exp(-(Y**2 / w[1]**2 + Z**2 / w[2]**2))
            self._tY, self._tZ = Y, Z
        return (self.peak_density * np.exp(-x**2 / w[0]**2)) * self._transverse

    def chi(self, x, Y, Z, s_local):
        n = self.density(x, Y, Z)
        total = 0.0
        for frac, delta in self.species:
            total = total + frac * self.response.susceptibility(n, delta, s_local)
        return total


# --------------------------------------------------------------------- result


@dataclass
class PropagationResult:
    """Everything one pass of the propagator produced."""

    grid: TransverseGrid
    psi_exit: np.ndarray            #: ``E / E_vac`` at the exit plane
    E_scat: np.ndarray              #: ``E - E_vac`` at the exit plane
    x_edge: float                   #: half the propagation length (m)
    x_slices: np.ndarray            #: axial midpoints of the atom slices (m)
    s_peak: float = 0.0             #: largest local saturation parameter reached
    mean_intensity: float = float("nan")  #: density-weighted ``<I>/I0``
    intensity_plane: Optional[np.ndarray] = None   #: ``(n_slices, n)`` at y = 0
    intensity_3d: Optional[np.ndarray] = None      #: ``(n_slices, nw, nw)``, float32
    window: Optional[slice] = None  #: transverse crop used by :attr:`intensity_3d`
    widths: Optional[np.ndarray] = None            #: cloud widths used (m)

    @property
    def transmission(self) -> np.ndarray:
        """``|psi|^2`` at the exit plane."""
        return np.abs(self.psi_exit) ** 2

    @property
    def z_axis(self) -> np.ndarray:
        """Transverse axis of :attr:`intensity_plane` (m)."""
        return self.grid.axis

    def edge_power_fraction(self, frac: float = 0.9) -> float:
        """Fraction of scattered power outside ``frac`` of the box half-width.

        A wraparound check: if this is not tiny, the box is too small and the
        far field is contaminated by aliasing.
        """
        I = np.abs(self.E_scat) ** 2
        ax = self.grid.axis
        edge = (np.abs(ax) > frac * np.abs(ax).max())
        mask = edge[:, None] | edge[None, :]
        tot = float(I.sum())
        return float(I[mask].sum()) / tot if tot > 0 else 0.0


# ----------------------------------------------------------------- propagator


class Propagator:
    """Split-step angular-spectrum beam propagation through a 3D medium.

    Parameters
    ----------
    response : TwoLevelResponse
        Supplies the wavenumber and the ``chi -> n_ref`` conversion (including the
        Lorentz-Lorenz option, if the response was built with it).
    grid : TransverseGrid
        Transverse sampling and angular spectrum.
    x_edge : float
        Propagate over ``[-x_edge, +x_edge]`` (m).
    n_slices : int
        Number of atom slices.  Keep the per-slice phase below ~0.1 rad; halving
        this is the convergence test.
    """

    def __init__(self, response: TwoLevelResponse, grid: TransverseGrid,
                 x_edge: float, n_slices: int, backend=None):
        self.response = response
        self.grid = grid
        self.x_edge = float(x_edge)
        self.n_slices = int(n_slices)
        self.backend = as_backend(backend)

    @classmethod
    def for_cloud(cls, response: TwoLevelResponse, cloud, n_grid: int = 768,
                  L_box: float = 36.0e-6, x_span_w: float = 3.0,
                  n_slices: int = 180, wx_scale: float = 1.0,
                  backend=None) -> "Propagator":
        """Size the propagation to a cloud: ``+- x_span_w`` axial 1/e radii.

        Three axial widths puts the exit plane about three Rayleigh ranges of the
        cloud's own transverse structure downstream, by which point the imprinted
        phase has diffracted away -- which is exactly why
        :func:`~kamo.imaging.readout.refocus` exists.
        """
        grid = TransverseGrid(n_grid, L_box, response.k)
        x_edge = x_span_w * float(cloud.widths[0]) * wx_scale
        return cls(response, grid, x_edge, n_slices, backend=backend)

    def rescaled(self, n_grid: Optional[int] = None,
                 n_slices: Optional[int] = None) -> "Propagator":
        """Copy at a different transverse sampling and/or slice count."""
        return Propagator(
            self.response,
            self.grid if n_grid is None else self.grid.rescaled(n_grid),
            self.x_edge,
            self.n_slices if n_slices is None else n_slices,
            backend=self.backend)

    def with_backend(self, backend) -> "Propagator":
        """Copy running on a different device.  ``'gpu'``, ``'cpu'`` or a backend."""
        return Propagator(self.response, self.grid, self.x_edge, self.n_slices,
                          backend=backend)

    # ------------------------------------------------------------------- run

    def propagate(self, source: SusceptibilitySource, saturate: bool = True,
                  s0_incident: float = 0.0, record: Optional[str] = None,
                  record_window: Optional[float] = None,
                  x_edge: Optional[float] = None) -> PropagationResult:
        """Propagate a unit-amplitude plane wave through ``source``.

        Parameters
        ----------
        source : SusceptibilitySource
            The medium.
        saturate : bool
            Form the local saturation parameter from the intensity that has
            actually propagated to each point, ``s = s0_incident |E|^2``.  This is
            not the incident intensity, because the cloud lenses.
        s0_incident : float
            Incident on-resonance saturation parameter.
        record : {None, 'plane', '3d'}
            ``'plane'`` stores ``|E|^2`` on the ``y = 0`` plane (cheap);
            ``'3d'`` stores the full ``|E(x, y, z)|^2``, cropped to
            ``record_window``, as float32.  The slices are already being computed,
            so recording is nearly free in time -- but ``'3d'`` costs
            ``n_slices * nw^2 * 4`` bytes.
        record_window : float or slice, optional
            Crop for the ``'3d'`` record: a transverse half-width in metres, or a
            ``slice`` into the grid axis to use verbatim.  Pass the slice when the
            record must line up with an existing voxel grid -- going via a float
            is not idempotent, since the realized window edge is smaller than the
            half-width that produced it.  Defaults to four transverse cloud
            widths, where the density has already fallen by ``e^-16``.

        Returns
        -------
        PropagationResult
        """
        if record not in (None, "plane", "3d"):
            raise ValueError("record must be None, 'plane' or '3d'")

        g = self.grid
        n = g.n
        x_edge = self.x_edge if x_edge is None else float(x_edge)
        ns = self.n_slices
        dx = 2 * x_edge / ns
        xs = -x_edge + (np.arange(ns) + 0.5) * dx   # midpoints: sum(n dx) is exact

        bk = self.backend

        # crop window for the 3D record
        win = None
        if record == "3d":
            if isinstance(record_window, slice):
                win = record_window
            else:
                if record_window is None:
                    w = getattr(source, "widths", None)
                    record_window = 4.0 * float(np.max(w[1:])) if w is not None \
                        else 0.25 * g.L
                win = g.window_slice(record_window)

        # Two ways to get the slice operator.  A source that exposes its `species`
        # alongside `density` goes through the single array-generic closed form in
        # TwoLevelResponse, which is what the GPU path needs and which also avoids
        # rebuilding the density a second time inside `chi`.  Anything else hands
        # back an opaque numpy `chi` and therefore stays on the host.
        species = getattr(source, "species", None)
        if species is not None and not species:
            species = None
        if species is None and bk.on_gpu:
            raise TypeError(
                f"{type(source).__name__} supplies only an opaque numpy chi(), so "
                "it cannot be propagated on the GPU. Give it a `species` sequence "
                "of (fraction, delta) pairs alongside `density`, or propagate it "
                "with backend='cpu'.")

        H_half = bk.complex(g.propagator(dx / 2))
        H_full = bk.complex(g.propagator(dx))
        Y, Z = bk.real(g.Y), bk.real(g.Z)

        E = bk.complex(np.ones((n, n)))
        E = bk.ifft2(bk.fft2(E) * H_half)

        # Accumulators stay ON the device and are read out once, at the end: a
        # float() per slice would sync the host to the GPU every step and throw
        # away most of the speedup.
        s_peak_acc = bk.zeros_like_scalar()
        plane_rec = [] if record == "plane" else None
        vol_rec = [] if record == "3d" else None
        w_num = bk.zeros_like_scalar()
        w_den = bk.zeros_like_scalar()
        weighted = False

        for i, x in enumerate(xs):
            I_here = bk.abs2(E)
            s_loc = s0_incident * I_here if saturate else 0.0
            if saturate:
                s_peak_acc = bk.maximum(s_peak_acc, bk.dmax(I_here) * s0_incident)

            n_slice = source.density(float(x), Y, Z)
            if n_slice is not None:
                weighted = True
                w_num = w_num + bk.dsum(n_slice * I_here)
                w_den = w_den + bk.dsum(n_slice)
            if plane_rec is not None:
                plane_rec.append(np.array(bk.numpy(I_here[g.center, :]), copy=True))
            if vol_rec is not None:
                vol_rec.append(bk.numpy(I_here[win, win]).astype(np.float32))

            if species is not None:
                expo = self.response.mixture_slice_exponent(
                    n_slice, species, dx, s_loc, backend=bk)
            else:
                chi = source.chi(float(x), g.Y, g.Z, s_loc)
                expo = 1j * g.k * self.response.index_minus_one(chi) * dx
            E = E * bk.exp(expo)
            E = bk.ifft2(bk.fft2(E) * (H_half if i == ns - 1 else H_full))

        # No absorber anywhere in the box, so the vacuum reference is a pure phase.
        E_vac = np.exp(1j * g.k * 2 * x_edge)
        E = bk.numpy(E)
        s_peak = bk.to_float(s_peak_acc)
        # `weighted` says the source reported a density at all; w_den can still be
        # zero when it did but the density is identically zero (the vacuum test).
        den = bk.to_float(w_den) if weighted else 0.0
        mean_I = (bk.to_float(w_num) / den) if den > 0 else float("nan")

        return PropagationResult(
            grid=g,
            psi_exit=E / E_vac,
            E_scat=E - E_vac,
            x_edge=x_edge,
            x_slices=xs,
            s_peak=s_peak,
            mean_intensity=mean_I,
            intensity_plane=(np.asarray(plane_rec) if plane_rec is not None else None),
            intensity_3d=(np.asarray(vol_rec) if vol_rec is not None else None),
            window=win,
            widths=getattr(source, "widths", None),
        )


# ------------------------------------------------------------ backend check


def check_backend(propagator: "Propagator", source: SusceptibilitySource,
                  s0_incident: float = 0.0, precisions=("double", "single"),
                  verbose: bool = True) -> dict:
    """Run one case on the CPU and on the GPU and report the difference.

    The GPU path is worth having only if it computes the same thing, so this runs
    the identical propagation on both devices and prints the disagreement in the
    exit field, the density-weighted mean intensity and the peak saturation, next
    to the wall-clock speedup.  Reach for it after changing anything in the loop,
    or once on a new machine, rather than taking the equivalence on trust.

    Returns a dict keyed by precision, each holding ``psi_rel`` (max exit-field
    difference relative to the peak amplitude), ``mean_I_rel``, ``s_peak_rel``,
    ``seconds`` and ``speedup``.
    """
    import time

    cpu = propagator.with_backend(ArrayBackend("cpu"))
    t0 = time.perf_counter()
    ref = cpu.propagate(source, s0_incident=s0_incident)
    t_cpu = time.perf_counter() - t0
    scale = float(np.max(np.abs(ref.psi_exit)))

    out = {}
    for prec in precisions:
        bk = ArrayBackend("gpu", prec)
        gpu = propagator.with_backend(bk)
        gpu.propagate(source, s0_incident=s0_incident)      # warm the cuFFT plan
        bk.sync()
        t0 = time.perf_counter()
        res = gpu.propagate(source, s0_incident=s0_incident)
        bk.sync()
        t = time.perf_counter() - t0
        d = dict(
            psi_rel=float(np.max(np.abs(res.psi_exit - ref.psi_exit)) / scale),
            mean_I_rel=abs(res.mean_intensity / ref.mean_intensity - 1.0),
            s_peak_rel=abs(res.s_peak / ref.s_peak - 1.0) if ref.s_peak else 0.0,
            seconds=t, speedup=t_cpu / t)
        out[prec] = d
        if verbose:
            print(f"  gpu/{prec:<6s} {t:7.3f} s ({d['speedup']:5.1f}x)   "
                  f"max|dpsi|/|psi|max {d['psi_rel']:.2e}   "
                  f"<I> {d['mean_I_rel']:.2e}   s_peak {d['s_peak_rel']:.2e}")
    if verbose:
        print(f"  cpu        {t_cpu:7.3f} s  (reference)")
    return out
