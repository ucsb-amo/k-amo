"""Plots of beams, traps and the clouds they hold.

* :func:`plot_plane_cut` -- colour map of the intensity, the potential or a
  cloud's density on an arbitrary plane, given its normal and (optionally) a
  point in it; for a trap it marks the minimum, the escape contour and the
  direction of gravity.
* :func:`plot_column_density` -- a cloud's density integrated along a lab axis:
  what an image along that axis sees.
* :func:`plot_line_cuts` -- the same quantities along straight lines; for a trap,
  through the minimum along the principal axes with the harmonic approximation
  dashed.  Where the two part company is the anharmonicity, and it is the
  cheapest visual check on the Hessian.
* :func:`plot_beam_profile` -- the two beam radii ``w_u(s)``, ``w_v(s)``.
* :func:`plot_trap_summary` -- three lab planes through the minimum plus the
  line cuts.

The data behind the plots are available without matplotlib from
:func:`plane_cut`, :func:`column_density_map` and :func:`line_cuts`.

House style of :mod:`kamo.imaging.plotting`: every function takes an optional
``ax`` and returns ``(fig, ax)``; matplotlib is imported inside the functions;
inputs are SI and lengths are scaled to microns only when drawn.

Live plots
----------
Intensity and potential plots (plane cuts, line cuts, the summary) are *live*:
they re-sample the field over whatever region is shown, every time the figure
is drawn -- after ``plt.xlim``/``ax.set_ylim``, an interactive zoom or pan, and at
``savefig``'s dpi -- so zooming never pixelates and panning never runs off the
computed region.  Resolution is one sample per screen pixel (two per pixel along
a line cut), capped at :data:`MAX_SAMPLES` per panel; ``n=`` fixes the count
instead.  A redraw with an unchanged view costs nothing.  The cost is ~50 ns per
sample per beam: 3-15 ms per panel on screen, 25-110 ms at 300 dpi, comparable
to matplotlib's own drawing.

* The colour scale is fixed at the first view (``autoscale="first"``) so a colour
  keeps its meaning while zooming; ``autoscale="view"`` rescales each view, and
  ``vmin``/``vmax`` override either.
* Potential contours are absolute energies (10/50/90% of the escape depth and
  the escape energy itself), so they are right in every view; intensity
  contours sit at 1/e^2 and 1/e of the overall peak.
* The image's ``cut`` attribute is the :class:`PlaneCut` on screen, and
  ``n_evaluations`` counts the samplings.
* Cloud densities are *not* live: they exist on the solver grid only, so there is
  nothing to compute outside it and nothing new to see by zooming in.  Those
  plots are sampled once.

Windows
-------
``xlim=(lo, hi)`` and ``ylim=(lo, hi)`` set the initial view (and, for the
static cloud plots, the sampled region).  On a map they are lab coordinates (m)
along the plot's horizontal and vertical axes -- the tick labels times 1e-6 --
and either end may be ``None`` to keep its default; ``half_width`` (scalar or
pair, m) is the symmetric shortcut about ``center``.  After the call, matplotlib's
own ``plt.xlim`` works in the plotted microns.  Defaults:

========================  ============================================================
beams, :class:`Trap`      ``center`` +- the scale of the TIGHTEST beam along each plot
                          axis, widened by how far the beam origins spread from
                          their mean.  A beam's scale along a direction is 2.5 waists
                          across it (the waist of the transverse axis that direction
                          lies along) plus, along it, the distance where the on-axis
                          intensity has fallen to 1/7.25 of the peak -- 2.5 zR for a
                          round beam, set by the smaller Rayleigh range of an
                          elliptical one.  ``center`` is the trap minimum, or the mean
                          beam origin.  (3 um / 1064 nm tweezer: +-66.5 um along the
                          beam, +-7.5 um across it.  That tweezer crossing a light
                          sheet: the crossing, not the sheet's millimetres.)
``HarmonicTrap``          ``center`` +- 6 ground-state rms widths.
a cloud                   the centroid +- 4 rms widths along each plot axis.
:func:`plot_line_cuts`    ``xlim`` is the displacement from ``center``, default the
                          longest of the rules above over the directions; ``ylim`` is
                          in ``units`` for a potential (default from 5% of the escape
                          depth below the minimum to 130% above), SI otherwise.
:func:`plot_beam_profile` ``xlim`` is the distance from the waist, default +-3 zR;
                          ``ylim`` the beam radius (m), default from 0.
========================  ============================================================

:func:`plot_trap_summary` takes one ``limits={"x": (lo, hi), "z": (lo, hi)}``
dict keyed by lab axis, applied to every panel that shows that axis.

``aspect`` (maps only) goes to matplotlib: ``"equal"`` (the default, so a
micron is a micron both ways), ``"auto"``, or a number.

Plot orientation is chosen for reading, not handedness: the vertical plot axis
is the in-plane direction closest to lab ``+z`` (``+y`` for a horizontal cut)
and the horizontal one points toward ``+x`` (``+y`` when the normal is along x),
so ``normal=(0, 1, 0)`` shows the familiar x-z picture with x to the right and
z up.  Pass ``in_plane_axis`` to fix the horizontal direction yourself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import kamo.constants as kc
from kamo.imaging.plotting import _axes, _tidy

from . import frames as fr

UNIT_SCALE = {"J": 1.0, "K": 1.0 / kc.kB, "uK": 1e6 / kc.kB, "Hz": 1.0 / kc.h,
              "kHz": 1e-3 / kc.h}
UNIT_LABEL = {"J": "J", "K": "K", "uK": r"$\mu$K", "Hz": "Hz", "kHz": "kHz"}
MAX_SAMPLES = 2_000_000          #: cap on samples per live map panel (one per pixel below it)
_LINE_SAMPLES = (201, 20_000)    #: live line cuts: two per pixel, clipped to this range
_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown")
_AXIS_NAMES = ("x", "y", "z")

#: how each stored quantity is drawn: (stored -> plotted scale, label, default cmap)
_MAP_STYLE = {
    "intensity": (1.0, r"I  (W/m$^2$)", "magma"),
    "density": (1e-6, r"n  (cm$^{-3}$)", "magma"),
    "column density": (1e-4, r"column density  (cm$^{-2}$)", "magma"),
}


# ----------------------------------------------------------------- helpers

def _is_trap(obj) -> bool:
    return hasattr(obj, "potential_J") and hasattr(obj, "minimum")


def _is_cloud(obj) -> bool:
    return hasattr(obj, "density") and hasattr(obj, "sigma") and not _is_trap(obj)


def _beams_of(obj) -> tuple:
    beams = getattr(obj, "beams", None)
    return tuple(beams) if beams else ()


def _field_of(obj):
    return obj.field if _is_trap(obj) and hasattr(obj, "field") else obj


def _check_units(units: str) -> float:
    if units not in UNIT_SCALE:
        raise ValueError(f"units must be one of {sorted(UNIT_SCALE)}; got {units!r}")
    return UNIT_SCALE[units]


def _quantity_for(obj, quantity: Optional[str]) -> str:
    trap, cloud = _is_trap(obj), _is_cloud(obj)
    default = "potential" if trap else "density" if cloud else "intensity"
    quantity = default if quantity is None else quantity
    if quantity not in ("potential", "intensity", "density"):
        raise ValueError(f"quantity must be 'potential', 'intensity' or 'density'; "
                         f"got {quantity!r}")
    if quantity == "potential" and not trap:
        raise ValueError("a potential needs a Trap: beams alone carry no atom or state.")
    if quantity == "density" and not cloud:
        raise ValueError("a density needs a cloud (a TrapCloud from kamo.trap.solve).")
    if quantity == "intensity" and cloud:
        raise ValueError("a cloud has no intensity; plot its trap, or quantity='density'.")
    return quantity


def _center_for(obj, center) -> np.ndarray:
    if center is None or (isinstance(center, str) and center == "minimum"):
        if _is_trap(obj):
            return np.array(obj.minimum().position, dtype=float)
        if _is_cloud(obj):
            return np.array(getattr(obj, "centroid", np.zeros(3)), dtype=float)
        return np.mean([b.origin for b in _beams_of(obj)], axis=0)
    c = np.array(center, dtype=float)
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        raise ValueError(f"center must be a finite 3-vector (m) or 'minimum'; got {center!r}")
    return c


def _plotted(quantity: str, units: str):
    """(stored -> plotted scale, label, default cmap).  A potential is stored in
    ``units`` already; everything else in SI."""
    if quantity == "potential":
        return 1.0, f"U  ({UNIT_LABEL[units]})", "viridis_r"
    return _MAP_STYLE[quantity]


def _plane_axes(normal, in_plane_axis=None):
    """Readable in-plane axes ``(e1, e2, n)``: e2 'up', e1 toward +x (or +y)."""
    n = fr.as_unit_real_vector(normal, "normal")
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    if in_plane_axis is None:
        e2 = up - n * (n @ up)
        e2 /= np.linalg.norm(e2)
        e1 = np.cross(e2, n)
        right = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        if e1 @ right < 0:
            e1 = -e1
    else:
        e1 = fr.orthonormal_frame(n, first=in_plane_axis)[0]
        e2 = np.cross(n, e1)
        if e2 @ up < 0:
            e2 = -e2
    return e1, e2, n


_N_SCALES = 2.5                                  #: beam scales in a default half-width
_AXIAL_DROP = (1.0 + _N_SCALES ** 2) ** 2         #: (I0 / I)^2 on axis at 2.5 zR, round beam


def _axial_extent(beam) -> float:
    """Distance from the waist where the on-axis intensity has fallen as far as a
    round beam's has at 2.5 zR: solves ``(1 + s^2/zu^2)(1 + s^2/zv^2) = 7.25^2``.
    An elliptical beam's is set by its SMALLER Rayleigh range -- a light sheet
    fades like 1/s well before its wide axis diverges."""
    a, b = 1.0 / beam.rayleigh_range_u ** 2, 1.0 / beam.rayleigh_range_v ** 2
    k1 = _AXIAL_DROP - 1.0                         # stable root of ab x^2 + (a+b) x - k1 = 0
    return float(np.sqrt(2.0 * k1 / ((a + b) + np.sqrt((a + b) ** 2 + 4.0 * a * b * k1))))


def _beam_extent(beam, e) -> float:
    """One beam's default half-width along unit vector ``e``."""
    u, v, k = beam.frame
    across = np.hypot(float(e @ u) * beam.waist_u, float(e @ v) * beam.waist_v)
    return abs(float(e @ k)) * _axial_extent(beam) + _N_SCALES * across


def _extent_along(obj, e) -> float:
    """Default half-width along ``e`` (see the module docstring)."""
    e = np.asarray(e, dtype=float)
    if _is_cloud(obj):
        sigma = np.asarray(obj.sigma, dtype=float)
        return 4.0 * float(np.sqrt(np.sum(e ** 2 * sigma ** 2)))
    beams = _beams_of(obj)
    if not beams:                                   # a HarmonicTrap: oscillator widths
        tf = obj.trap_frequencies()
        var = kc.hbar / (2.0 * tf.mass * tf.omega)
        return 6.0 * float(np.sqrt(np.sum((tf.axes @ e) ** 2 * var)))
    tightest = min(_beam_extent(b, e) for b in beams)
    mean = np.mean([b.origin for b in beams], axis=0)      # spread among the beams only:
    spread = max(abs(float((np.asarray(b.origin) - mean) @ e)) for b in beams)
    return tightest + spread                                # a trap's sag must not widen it


def _half_widths(obj, e1, e2, half_width):
    if half_width is None:
        return _extent_along(obj, e1), _extent_along(obj, e2)
    hw = np.atleast_1d(np.asarray(half_width, dtype=float))
    if hw.size not in (1, 2) or not np.all(np.isfinite(hw)) or np.any(hw <= 0):
        raise ValueError(f"half_width must be a positive scalar or (h1, h2) pair (m); "
                         f"got {half_width!r}")
    return (float(hw[0]), float(hw[0])) if hw.size == 1 else (float(hw[0]), float(hw[1]))


def _window(lim, center: float, half: float, name: str):
    """``(lo, hi)`` for one axis: ``lim`` where given, else ``center -+ half``."""
    lo, hi = center - half, center + half
    if lim is not None:
        if isinstance(lim, str) or np.ndim(lim) != 1 or len(lim) != 2:
            raise ValueError(f"{name} must be a (min, max) pair, either end None; got {lim!r}")
        lo = lo if lim[0] is None else float(lim[0])
        hi = hi if lim[1] is None else float(lim[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"{name} must have finite min < max; got ({lo!r}, {hi!r})")
    return lo, hi


def _n_pair(n):
    return (int(n), int(n)) if np.ndim(n) == 0 else (int(n[0]), int(n[1]))


def _directions(obj, directions) -> np.ndarray:
    if directions is None:
        return obj.trap_frequencies().axes if _is_trap(obj) else np.eye(3)
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"directions must be one 3-vector or a sequence of them; "
                         f"got shape {np.shape(directions)}")
    return np.array([fr.as_unit_real_vector(row, "direction") for row in d])


def _axis_index(axis) -> int:
    if isinstance(axis, str) and axis.lower() in _AXIS_NAMES:
        return _AXIS_NAMES.index(axis.lower())
    if not isinstance(axis, str) and np.ndim(axis) == 0 and int(axis) in (0, 1, 2):
        return int(axis)
    raise ValueError(f"axis must be 'x', 'y', 'z' or 0, 1, 2; got {axis!r}")


def _lab_axis_name(e) -> Optional[str]:
    """'x', 'y' or 'z' if ``e`` is that positive lab axis, else None."""
    k = int(np.argmax(np.abs(e)))
    return _AXIS_NAMES[k] if abs(e[k] - 1.0) < 1e-8 else None


def _direction_label(e) -> str:
    for name, v in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        d = float(np.dot(e, v))
        if abs(abs(d) - 1.0) < 1e-8:
            return f"$-{name}$" if d < 0 else f"${name}$"
    return "(" + ", ".join(f"{c:.2f}" for c in e) + ")"


def _axis_label(e) -> str:
    return _direction_label(e) + r"  ($\mu$m)"


# ------------------------------------------------------------------- data

@dataclass
class PlaneCut:
    """Samples of a quantity on a plane through ``center`` spanned by ``e1, e2``."""

    a1: np.ndarray              #: offsets from ``center`` along e1 (m)
    a2: np.ndarray              #: offsets from ``center`` along e2 (m)
    values: np.ndarray          #: (len(a1), len(a2)), SI; a potential in ``units``
    e1: np.ndarray
    e2: np.ndarray
    normal: np.ndarray
    center: np.ndarray          #: (3,) lab point at a1 = a2 = 0 (m)
    quantity: str               #: "intensity", "potential", "density" or "column density"
    units: str                  #: "W/m^2", "1/m^3", "1/m^2", or the potential units

    @property
    def horizontal(self) -> np.ndarray:
        """Lab coordinate ``r . e1`` of the samples (m): the horizontal tick values."""
        return self.a1 + float(self.center @ self.e1)

    @property
    def vertical(self) -> np.ndarray:
        """Lab coordinate ``r . e2`` of the samples (m): the vertical tick values."""
        return self.a2 + float(self.center @ self.e2)

    def lab_points(self):
        """The lab coordinates ``(X, Y, Z)`` of the samples, broadcast (n1, n2)."""
        A1, A2 = self.a1[:, None], self.a2[None, :]
        return tuple(self.center[i] + A1 * self.e1[i] + A2 * self.e2[i] for i in range(3))


def plane_cut(obj, normal=(0.0, 1.0, 0.0), center=None, *, quantity: Optional[str] = None,
              xlim=None, ylim=None, half_width=None, n=(241, 241), in_plane_axis=None,
              units: str = "uK") -> PlaneCut:
    """Sample a quantity on the plane ``{r : (r - center) . normal = 0}``.

    ``obj`` is a beam or crossed beams (``quantity="intensity"``), a trap
    (``"potential"``, the default, or ``"intensity"``) or a cloud with
    ``density(x, y, z)`` (``"density"``).  ``center`` defaults to the trap
    minimum (also ``"minimum"``), the cloud centroid, or the mean beam origin.
    ``xlim``/``ylim`` and ``half_width`` set the sampled window; see the module
    docstring for the syntax and the defaults.
    """
    quantity = _quantity_for(obj, quantity)
    scale = _check_units(units)
    e1, e2, nrm = _plane_axes(normal, in_plane_axis)
    c = _center_for(obj, center)
    h1, h2 = _half_widths(obj, e1, e2, half_width)
    c1, c2 = float(c @ e1), float(c @ e2)
    lo1, hi1 = _window(xlim, c1, h1, "xlim")
    lo2, hi2 = _window(ylim, c2, h2, "ylim")
    n1, n2 = _n_pair(n)
    units_out = {"intensity": "W/m^2", "density": "1/m^3"}.get(quantity, units)
    cut = PlaneCut(np.linspace(lo1, hi1, n1) - c1, np.linspace(lo2, hi2, n2) - c2,
                   np.empty(0), e1, e2, nrm, c, quantity, units_out)
    X, Y, Z = cut.lab_points()
    if quantity == "intensity":
        cut.values = np.asarray(_field_of(obj).intensity(X, Y, Z), dtype=float)
    elif quantity == "density":
        cut.values = np.broadcast_to(np.asarray(obj.density(X, Y, Z), dtype=float),
                                     (n1, n2)).copy()
    else:
        cut.values = np.asarray(obj.potential_J(X, Y, Z), dtype=float) * scale
    return cut


def column_density_map(cloud, axis="x", *, xlim=None, ylim=None, half_width=None,
                       n=(241, 241)) -> PlaneCut:
    """The density of a gridded cloud integrated along lab ``axis`` (1/m^2).

    The plot axes are the other two lab axes, oriented as in :func:`plane_cut`
    (along x: y right, z up; along y: x right, z up; along z: x right, y up).
    The integral is the solver grid's own quadrature (so the peak matches
    ``TrapCloud.peak_column_density`` along x), resampled bilinearly onto the
    window, which defaults to the centroid +- 4 rms widths.
    """
    if not (hasattr(cloud, "grid") and hasattr(cloud, "density_grid")):
        raise TypeError(f"column_density_map needs a gridded cloud (a TrapCloud); "
                        f"got {type(cloud).__name__}")
    from scipy.interpolate import RegularGridInterpolator

    k = _axis_index(axis)
    e1, e2, nrm = _plane_axes(np.eye(3)[k])
    i1, i2 = int(np.argmax(np.abs(e1))), int(np.argmax(np.abs(e2)))
    c = np.asarray(cloud.centroid, dtype=float)
    h1, h2 = _half_widths(cloud, e1, e2, half_width)
    lo1, hi1 = _window(xlim, float(c[i1]), h1, "xlim")
    lo2, hi2 = _window(ylim, float(c[i2]), h2, "ylim")
    n1, n2 = _n_pair(n)
    p1, p2 = np.linspace(lo1, hi1, n1), np.linspace(lo2, hi2, n2)

    g = cloud.grid
    column = np.sum(cloud.density_grid, axis=k) * g.d[k]      # remaining axes, index order
    if i1 > i2:
        column = column.T
    coords = (g.x, g.y, g.z)
    f = RegularGridInterpolator((coords[i1], coords[i2]), column, method="linear",
                                bounds_error=False, fill_value=0.0)
    P1, P2 = np.meshgrid(p1, p2, indexing="ij")
    values = f(np.stack([P1.ravel(), P2.ravel()], axis=-1)).reshape(P1.shape)
    return PlaneCut(p1 - c[i1], p2 - c[i2], values, e1, e2, nrm, c, "column density", "1/m^2")


def line_cuts(obj, directions=None, center=None, *, quantity: Optional[str] = None,
              xlim=None, half_width=None, n: int = 401, units: str = "uK"):
    """1D cuts along straight lines through ``center``.

    The potential for a trap (through the minimum by default), the intensity for
    beams (through the mean beam origin), the density for a cloud (through the
    centroid).  ``directions`` is one 3-vector or a sequence of them; by default
    a trap's principal axes, else the lab x, y and z.  Each cut is a dict with
    ``direction``, ``s`` (m, displacement from ``center``), ``values`` (the
    potential in ``units``; W/m^2; 1/m^3), ``harmonic`` (a potential's harmonic
    expansion about the minimum, else ``None``), ``quantity`` and ``units``.
    ``xlim=(lo, hi)`` (m) sets the displacement range for every direction;
    otherwise ``half_width``, or the default window along each.
    """
    quantity = _quantity_for(obj, quantity)
    scale = _check_units(units)
    c = _center_for(obj, center)
    hm = (obj.harmonic() if hasattr(obj, "harmonic") else obj) if quantity == "potential" else None
    units_out = {"intensity": "W/m^2", "density": "1/m^3"}.get(quantity, units)
    out = []
    for d in _directions(obj, directions):
        h = _extent_along(obj, d) if half_width is None else float(half_width)
        lo, hi = _window(xlim, 0.0, h, "xlim")
        s = np.linspace(lo, hi, int(n))
        X, Y, Z = (c[i] + s * d[i] for i in range(3))
        harmonic = None
        if quantity == "potential":
            values = np.asarray(obj.potential_J(X, Y, Z), dtype=float) * scale
            harmonic = np.asarray(hm.potential_J(X, Y, Z), dtype=float) * scale
        elif quantity == "intensity":
            values = np.asarray(_field_of(obj).intensity(X, Y, Z), dtype=float)
        else:
            values = np.asarray(obj.density(X, Y, Z), dtype=float)
        out.append(dict(direction=d, s=s, values=values, harmonic=harmonic,
                        quantity=quantity, units=units_out))
    return out


# ------------------------------------------------------- live matplotlib artists

_LIVE: dict = {}


def _live_classes():
    """The live artists, built on first use so importing this module does not
    import matplotlib."""
    if _LIVE:
        return _LIVE["map"], _LIVE["lines"]
    from contextlib import contextmanager

    import contourpy
    from matplotlib.artist import Artist
    from matplotlib.collections import LineCollection
    from matplotlib.image import AxesImage

    @contextmanager
    def quiet(*artists):
        """No stale propagation while refreshing inside a draw: the refreshed
        artists are drawn in this same pass, and a stale figure would ask an
        interactive canvas for another draw."""
        saved = [(a, a.stale_callback) for a in artists]
        for a, _ in saved:
            a.stale_callback = None
        try:
            yield
        finally:
            for a, cb in saved:
                a.stale_callback = cb

    class LivePlaneCut(AxesImage):
        """A plane-cut image that re-samples over the current view when drawn."""

        def __init__(self, ax, sample, *, n, scale, autoscale, vmin, vmax, levels, escape,
                     level_colour="0.15", **kw):
            super().__init__(ax, origin="lower", interpolation="nearest", **kw)
            self._sample, self._n, self._scale = sample, n, scale
            self._autoscale, self._vmin, self._vmax = autoscale, vmin, vmax
            self._levels, self._escape = levels, escape
            self._view = None
            self.cut = None
            self.n_evaluations = 0
            self.contour_lines = LineCollection([], colors=level_colour, linewidths=0.6,
                                                alpha=0.6)
            self.escape_lines = LineCollection([], colors="tab:red", linewidths=1.2)
            for lc in (self.contour_lines, self.escape_lines):
                ax.add_collection(lc, autolim=False)

        def _samples(self):
            if self._n is not None:
                return self._n
            bb = self.axes.get_window_extent()
            n1, n2 = max(16, int(round(bb.width))), max(16, int(round(bb.height)))
            if n1 * n2 > MAX_SAMPLES:
                f = np.sqrt(MAX_SAMPLES / (n1 * n2))
                n1, n2 = max(16, int(n1 * f)), max(16, int(n2 * f))
            return n1, n2

        def refresh(self) -> bool:
            """Re-sample if the view or the sample count changed; True if it did."""
            x0, x1 = sorted(self.axes.get_xlim())
            y0, y1 = sorted(self.axes.get_ylim())
            n = self._samples()
            view = (x0, x1, y0, y1, n)
            if view == self._view:
                return False
            cut = self._sample((x0 * 1e-6, x1 * 1e-6), (y0 * 1e-6, y1 * 1e-6), n)
            F = cut.values * self._scale
            first = self._view is None
            with quiet(self, self.contour_lines, self.escape_lines):
                self.set_data(F.T)
                self.set_extent((x0, x1, y0, y1))
                if first or self._autoscale == "view":
                    self.set_clim(float(np.nanmin(F)) if self._vmin is None else self._vmin,
                                  float(np.nanmax(F)) if self._vmax is None else self._vmax)
                if callable(self._levels):                  # frozen on the first view
                    self._levels = np.asarray(self._levels(F), dtype=float)
                segments, escape = [], []
                if self._levels is not None or self._escape is not None:
                    gen = contourpy.contour_generator(cut.horizontal * 1e6,
                                                      cut.vertical * 1e6, F.T)
                    for level in () if self._levels is None else self._levels:
                        segments.extend(gen.lines(float(level)))
                    if self._escape is not None:
                        escape = list(gen.lines(float(self._escape)))
                self.contour_lines.set_segments(segments)
                self.escape_lines.set_segments(escape)
            self._view, self.cut = view, cut
            self.n_evaluations += 1
            return True

        def draw(self, renderer):
            self.refresh()
            super().draw(renderer)

    class LiveLineCuts(Artist):
        """Invisible helper drawn just before the cut lines: re-samples every cut
        over the current horizontal view."""

        def __init__(self, ax, sample, lines, *, n, scale):
            super().__init__()
            self._sample, self._lines, self._n, self._scale = sample, lines, n, scale
            self._view = None
            self.cuts = None
            self.n_evaluations = 0
            drawn = [a for pair in lines for a in pair if a is not None]
            self.set_zorder(min(a.get_zorder() for a in drawn) - 0.01)
            self.set_in_layout(False)
            ax.add_artist(self)

        def _samples(self):
            if self._n is not None:
                return int(self._n)
            lo, hi = _LINE_SAMPLES
            return int(min(max(lo, 2 * round(self.axes.get_window_extent().width)), hi))

        def refresh(self) -> bool:
            x0, x1 = sorted(self.axes.get_xlim())
            n = self._samples()
            view = (x0, x1, n)
            if view == self._view:
                return False
            cuts = self._sample((x0 * 1e-6, x1 * 1e-6), n)
            with quiet(*[a for pair in self._lines for a in pair if a is not None]):
                for cut, (line, harmonic) in zip(cuts, self._lines):
                    line.set_data(cut["s"] * 1e6, cut["values"] * self._scale)
                    if harmonic is not None:
                        harmonic.set_data(cut["s"] * 1e6, cut["harmonic"] * self._scale)
            self._view, self.cuts = view, cuts
            self.n_evaluations += 1
            return True

        def draw(self, renderer):
            if self.get_visible():
                self.refresh()

    _LIVE.update(map=LivePlaneCut, lines=LiveLineCuts)
    return LivePlaneCut, LiveLineCuts


# ------------------------------------------------------------------ plots

def _draw_map(cut: PlaneCut, ax, *, scale: float, cmap, vmin, vmax, aspect, colorbar,
              cbar_label: str, title: str, levels=None, contour_colour="0.15"):
    """Static imshow + optional contours of a PlaneCut on lab-coordinate axes (um)."""
    fig = ax.figure
    F = cut.values * scale
    h, v = cut.horizontal * 1e6, cut.vertical * 1e6
    im = ax.imshow(F.T, extent=[h[0], h[-1], v[0], v[-1]], origin="lower", aspect=aspect,
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if levels is not None:
        lv = np.sort(np.asarray(levels, dtype=float))
        lv = lv[(lv > F.min()) & (lv < F.max())]
        if lv.size:
            H, V = np.meshgrid(h, v, indexing="ij")
            ax.contour(H.T, V.T, F.T, levels=lv, colors=contour_colour, linewidths=0.6,
                       alpha=0.6)
    ax.set_xlabel(_axis_label(cut.e1))
    ax.set_ylabel(_axis_label(cut.e2))
    ax.set_title(title, fontsize="small")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(cbar_label)
    _tidy(ax)
    return im, F


def plot_plane_cut(obj, normal=(0.0, 1.0, 0.0), center=None, *,
                   quantity: Optional[str] = None, xlim=None, ylim=None, half_width=None,
                   n=None, in_plane_axis=None, units: str = "uK", aspect="equal", ax=None,
                   cmap=None, vmin=None, vmax=None, autoscale: str = "first",
                   contours: bool = True, levels=None, colorbar: bool = True,
                   mark_minimum: bool = True, show_gravity: bool = True,
                   title: Optional[str] = None):
    """Colour map of the intensity, potential or density on an arbitrary plane.

    Axes are lab coordinates along the in-plane directions (um).  ``xlim``,
    ``ylim`` and ``half_width`` set the initial view (module docstring).  An
    intensity or potential map is live: it re-samples whatever is shown at each
    draw, one sample per pixel (``n`` fixes the count), with the colour scale
    fixed at the first view unless ``autoscale="view"`` or ``vmin``/``vmax``
    (plotted units) are given.  A cloud's density is sampled once (``n``
    default 241 x 241).

    For a potential the thin grey contours sit at 10, 50 and 90% of the escape
    depth above the minimum and the red one at the escape energy itself -- the
    single most informative line on the plot.  For an intensity or a density
    they are the 1/e^2 and 1/e contours of the peak; ``levels`` (plotted units)
    replaces them.  The minimum is marked with a white ``+`` when it lies in the
    plane (a hollow circle when projected into it), and an arrow shows the
    in-plane component of gravity.  Returns ``(fig, ax)``.
    """
    quantity = _quantity_for(obj, quantity)
    if autoscale not in ("first", "view"):
        raise ValueError(f"autoscale must be 'first' or 'view'; got {autoscale!r}")
    _check_units(units)
    e1, e2, nrm = _plane_axes(normal, in_plane_axis)
    c = _center_for(obj, center)
    h1, h2 = _half_widths(obj, e1, e2, half_width)
    lo1, hi1 = _window(xlim, float(c @ e1), h1, "xlim")
    lo2, hi2 = _window(ylim, float(c @ e2), h2, "ylim")
    fig, ax = _axes(ax, figsize=(4.8, 4.0))
    trap = _is_trap(obj)
    scale, label, default_cmap = _plotted(quantity, units)
    default_title = (f"density cut normal to {_direction_label(nrm)}" if quantity == "density"
                     else f"cut normal to {_direction_label(nrm)}")
    title = default_title if title is None else title

    def sample(xl, yl, m):
        return plane_cut(obj, nrm, c, quantity=quantity, xlim=xl, ylim=yl, n=m,
                         in_plane_axis=e1 if in_plane_axis is not None else None, units=units)

    if quantity == "density":                                   # gridded: sampled once
        cut = sample((lo1, hi1), (lo2, hi2), (241, 241) if n is None else _n_pair(n))
        peak = float(np.max(cut.values)) * scale
        lv = levels if levels is not None else (
            peak * np.array([np.exp(-2.0), np.exp(-1.0)]) if peak > 0 else None)
        _draw_map(cut, ax, scale=scale, cmap=cmap or default_cmap, vmin=vmin, vmax=vmax,
                  aspect=aspect, colorbar=colorbar, cbar_label=label, title=title,
                  levels=lv if contours else None)
        return fig, ax

    escape = None
    lv = None if levels is None else np.asarray(levels, dtype=float)
    if quantity == "potential":
        U_min = obj.minimum().potential_J * UNIT_SCALE[units]
        depth = obj.trap_depth_J() * UNIT_SCALE[units]
        if np.isfinite(depth) and depth > 0:
            escape = U_min + depth
            if lv is None:
                lv = U_min + depth * np.array([0.1, 0.5, 0.9])
    elif lv is None:                                            # intensity: the overall peak
        field = _field_of(obj)
        at_origins = max(float(field.intensity(*b.origin)) for b in _beams_of(obj))
        lv = lambda F: max(float(np.nanmax(F)), at_origins) * np.array([np.exp(-2.0),
                                                                        np.exp(-1.0)])
    LivePlaneCut, _ = _live_classes()
    ax.set_xlim(lo1 * 1e6, hi1 * 1e6)
    ax.set_ylim(lo2 * 1e6, hi2 * 1e6)
    ax.set_autoscale_on(False)
    im = LivePlaneCut(ax, sample, n=None if n is None else _n_pair(n), scale=scale,
                      autoscale=autoscale, vmin=vmin, vmax=vmax,
                      levels=lv if contours else None, escape=escape if contours else None,
                      cmap=cmap or default_cmap)
    im.set_clip_path(ax.patch)
    ax.add_image(im)
    ax.set_aspect(aspect)
    im.refresh()                        # so the colour bar and im.cut exist before a draw
    ax.set_xlabel(_axis_label(e1))
    ax.set_ylabel(_axis_label(e2))
    ax.set_title(title, fontsize="small")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(label)

    if trap and mark_minimum:
        r = obj.minimum().position
        p1, p2 = float(r @ e1), float(r @ e2)
        off = abs(float((r - c) @ nrm))
        pixel = min(np.diff(im.cut.a1[:2])[0], np.diff(im.cut.a2[:2])[0])
        if off <= pixel:
            ax.plot([p1 * 1e6], [p2 * 1e6], "+", color="w", ms=9, mew=1.6)
        else:
            ax.plot([p1 * 1e6], [p2 * 1e6], "o", mfc="none", mec="w", ms=7)
    if trap and show_gravity and getattr(obj, "gravity", False):
        g = obj.g_hat
        gp = np.array([g @ e1, g @ e2])
        if np.linalg.norm(gp) > 1e-3:
            gp = gp / np.linalg.norm(gp)
            x0, y0 = 0.87, 0.80
            ax.annotate("g", xy=(x0 + 0.08 * gp[0], y0 + 0.08 * gp[1]), xytext=(x0, y0),
                        xycoords="axes fraction", textcoords="axes fraction",
                        color="w", fontsize=9, ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", color="w", lw=1.2))
    _tidy(ax)
    return fig, ax


def plot_column_density(cloud, axis="x", *, xlim=None, ylim=None, half_width=None,
                        n=(241, 241), aspect="equal", ax=None, cmap="magma", vmin=0.0,
                        vmax=None, contours: bool = False, colorbar: bool = True,
                        title: Optional[str] = None):
    """Column density of a gridded cloud along lab ``axis`` (cm^-2 on the plot),
    on the other two lab axes; ``xlim``/``ylim``/``half_width`` as in
    :func:`column_density_map`.  ``vmin``/``vmax`` (cm^-2) fix the colour scale
    so panels can share it.  Sampled once (the density is gridded).
    Returns ``(fig, ax)``."""
    cut = column_density_map(cloud, axis, xlim=xlim, ylim=ylim, half_width=half_width, n=n)
    fig, ax = _axes(ax, figsize=(4.6, 4.0))
    scale, label, _ = _MAP_STYLE["column density"]
    peak = float(np.max(cut.values)) * scale
    levels = peak * np.array([np.exp(-2.0), np.exp(-1.0)]) if contours and peak > 0 else None
    mode = getattr(cloud, "mode", "")
    default = (f"column density along {_direction_label(cut.normal)}"
               + (f", {mode}" if mode else "") + f"  (peak {peak:.3g} cm$^{{-2}}$)")
    _draw_map(cut, ax, scale=scale, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
              colorbar=colorbar, cbar_label=label,
              title=title if title is not None else default, levels=levels,
              contour_colour="w")
    return fig, ax


def plot_line_cuts(obj, directions=None, center=None, *, quantity: Optional[str] = None,
                   xlim=None, ylim=None, half_width=None, n=None, units: str = "uK",
                   harmonic: bool = True, ax=None, title: Optional[str] = None):
    """The potential (a trap), intensity (beams) or density (a cloud) along lines
    through ``center``, as in :func:`line_cuts`.  For a trap the harmonic
    approximation is dashed and the escape energy dotted.

    ``xlim`` (m, displacement) sets the initial view; intensity and potential
    cuts are live and re-sample the shown range at each draw (two samples per
    pixel; ``n`` fixes the count).  ``ylim`` is in ``units`` for a potential and
    SI otherwise, either end ``None`` for its default.  Returns ``(fig, ax)``.
    """
    quantity = _quantity_for(obj, quantity)
    scale = _check_units(units)
    c = _center_for(obj, center)
    dirs = _directions(obj, directions)
    if xlim is None and half_width is None:
        h = max(_extent_along(obj, d) for d in dirs)
    else:
        h = _extent_along(obj, dirs[0]) if half_width is None else float(half_width)
    lo, hi = _window(xlim, 0.0, h, "xlim")
    cuts = line_cuts(obj, dirs, c, quantity=quantity, xlim=(lo, hi),
                     n=401 if n is None else int(n), units=units)
    fig, ax = _axes(ax, figsize=(5.0, 3.4))
    pscale, label, _ = _plotted(quantity, units)

    lines = []
    for cut, colour in zip(cuts, _COLOURS * 3):
        (main,) = ax.plot(cut["s"] * 1e6, cut["values"] * pscale, color=colour, lw=1.4,
                          label=f"along {_direction_label(cut['direction'])}")
        dashed = None
        if harmonic and cut["harmonic"] is not None:
            (dashed,) = ax.plot(cut["s"] * 1e6, cut["harmonic"] * pscale, color=colour,
                                lw=0.9, ls="--")
        lines.append((main, dashed))
    ax.set_xlim(lo * 1e6, hi * 1e6)
    ax.set_autoscalex_on(False)

    default_y = None
    if quantity == "potential":
        U_min = obj.minimum().potential_J * scale
        depth = obj.trap_depth_J() * scale
        if np.isfinite(depth) and depth > 0:
            ax.axhline(U_min + depth, color="tab:red", lw=0.9, ls=":", label="escape")
            default_y = (U_min - 0.05 * depth, U_min + 1.3 * depth)
    if default_y is not None or ylim is not None:
        y0, y1 = default_y if default_y is not None else tuple(v / pscale for v in ax.get_ylim())
        y0, y1 = _window(ylim, 0.5 * (y0 + y1), 0.5 * (y1 - y0), "ylim")
        ax.set_ylim(y0 * pscale, y1 * pscale)

    if quantity != "density":
        _, LiveLineCuts = _live_classes()

        def sample(xl, m):
            return line_cuts(obj, dirs, c, quantity=quantity, xlim=xl, n=m, units=units)

        LiveLineCuts(ax, sample, lines, n=n, scale=pscale)

    origin = ("the minimum" if _is_trap(obj) and center is None else
              "the centroid" if _is_cloud(obj) and center is None else "the centre")
    ax.set_xlabel(f"displacement from {origin}" + r"  ($\mu$m)")
    ax.set_ylabel(label)
    ax.legend(fontsize="xx-small", frameon=False)
    if title is not None:
        ax.set_title(title, fontsize="small")
    _tidy(ax)
    return fig, ax


def plot_beam_profile(beam, *, xlim=None, ylim=None, s_half: Optional[float] = None,
                      ax=None, title: Optional[str] = None):
    """The beam radii ``w_u(s)`` and ``w_v(s)`` against distance from the waist.
    ``xlim`` (m) is the sampled range of ``s`` (default +-3 zR; ``s_half`` is the
    symmetric shortcut), ``ylim`` (m) the radius shown (default from 0).
    Returns ``(fig, ax)``."""
    fig, ax = _axes(ax, figsize=(5.0, 3.2))
    zr = max(beam.rayleigh_range_u, beam.rayleigh_range_v)
    lo, hi = _window(xlim, 0.0, 3.0 * zr if s_half is None else float(s_half), "xlim")
    s = np.linspace(lo, hi, 401)
    ax.plot(s * 1e6, beam.beam_radius_u(s) * 1e6, color="tab:blue", lw=1.4, label=r"$w_u(s)$")
    ax.plot(s * 1e6, beam.beam_radius_v(s) * 1e6, color="tab:orange", lw=1.4,
            ls="--" if beam.waist_u == beam.waist_v else "-", label=r"$w_v(s)$")
    for z, colour in ((beam.rayleigh_range_u, "tab:blue"), (beam.rayleigh_range_v, "tab:orange")):
        for zz in (-z, z):
            if lo <= zz <= hi:
                ax.axvline(zz * 1e6, color=colour, lw=0.7, ls=":")
    ax.set_xlim(lo * 1e6, hi * 1e6)
    if ylim is None:
        ax.set_ylim(0, None)
    else:
        y_hi = ax.get_ylim()[1] * 1e-6
        y0, y1 = _window(ylim, 0.5 * y_hi, 0.5 * y_hi, "ylim")
        ax.set_ylim(y0 * 1e6, y1 * 1e6)
    ax.set_xlabel(r"$s$, distance from the waist  ($\mu$m)")
    ax.set_ylabel(r"beam radius  ($\mu$m)")
    ax.legend(fontsize="xx-small", frameon=False)
    ax.set_title(title if title is not None else repr(beam), fontsize="x-small")
    _tidy(ax)
    return fig, ax


def plot_trap_summary(trap, *, units: str = "uK", limits: Optional[dict] = None,
                      aspect="equal", axes=None, title: Optional[str] = None):
    """Three lab planes through the minimum and the principal-axis line cuts,
    2 x 2.  ``limits`` maps lab axes to windows, ``{"x": (lo, hi), "z": (lo, hi)}``
    (m, either end None), applied to every plane that shows that axis; ``aspect``
    applies to the three planes.  Returns ``(fig, axes)``."""
    import matplotlib.pyplot as plt
    limits = {} if limits is None else dict(limits)
    unknown = set(limits) - set(_AXIS_NAMES)
    if unknown:
        raise ValueError(f"limits keys must be lab axes 'x', 'y', 'z'; got {sorted(unknown)}")
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.8), layout="constrained")
    else:
        fig = np.asarray(axes).flat[0].figure
    axes = np.asarray(axes)
    for ax, normal in zip(axes.flat[:3], ((0, 0, 1), (0, 1, 0), (1, 0, 0))):
        e1, e2, _ = _plane_axes(normal)
        plot_plane_cut(trap, normal=normal, units=units, ax=ax, aspect=aspect,
                       xlim=limits.get(_lab_axis_name(e1)), ylim=limits.get(_lab_axis_name(e2)))
    plot_line_cuts(trap, units=units, ax=axes.flat[3])
    fig.suptitle(title if title is not None else repr(trap), fontsize="small")
    return fig, axes
