"""Plots of beams and traps.

* :func:`plot_plane_cut` -- contour cut of the intensity or the potential on an
  arbitrary plane, given its normal and (optionally) a point in it; for a trap
  it marks the minimum, the escape contour and the direction of gravity.
* :func:`plot_line_cuts` -- 1D cuts through the minimum along the principal
  axes, with the harmonic approximation dashed.  Where the two part company is
  the anharmonicity, and it is the cheapest visual check on the Hessian.
* :func:`plot_beam_profile` -- the two beam radii ``w_u(s)``, ``w_v(s)``.
* :func:`plot_trap_summary` -- three lab planes through the minimum plus the
  line cuts.

The data behind the plane and line plots are available without matplotlib from
:func:`plane_cut` and :func:`line_cuts`.

House style of :mod:`kamo.imaging.plotting`: every function takes an optional
``ax`` and returns ``(fig, ax)``; matplotlib is imported inside the functions;
lengths are scaled to microns only when drawn.

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
_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown")


# ----------------------------------------------------------------- helpers

def _is_trap(obj) -> bool:
    return hasattr(obj, "potential_J") and hasattr(obj, "minimum")


def _beams_of(obj) -> tuple:
    beams = getattr(obj, "beams", None)
    return tuple(beams) if beams else ()


def _check_units(units: str) -> float:
    if units not in UNIT_SCALE:
        raise ValueError(f"units must be one of {sorted(UNIT_SCALE)}; got {units!r}")
    return UNIT_SCALE[units]


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


def _extent_along(obj, e) -> float:
    """Default half-width along ``e``: ~2.5 beam scales, interpolating between the
    waist (transverse) and the Rayleigh range (along the beam)."""
    beams = _beams_of(obj)
    if not beams:                                   # a HarmonicTrap: oscillator widths
        tf = obj.trap_frequencies()
        var = kc.hbar / (2.0 * tf.mass * tf.omega)
        return 6.0 * float(np.sqrt(np.sum((tf.axes @ e) ** 2 * var)))
    h = 0.0
    for b in beams:
        c = min(1.0, abs(float(e @ b.propagation_direction)))
        zr = max(b.rayleigh_range_u, b.rayleigh_range_v)
        w = max(b.waist_u, b.waist_v)
        h = max(h, c * zr + np.sqrt(1.0 - c * c) * w)
    return 2.5 * h


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

    a1: np.ndarray              #: coordinates along e1 (m)
    a2: np.ndarray              #: coordinates along e2 (m)
    values: np.ndarray          #: (len(a1), len(a2)): W/m^2, or the potential in `units`
    e1: np.ndarray
    e2: np.ndarray
    normal: np.ndarray
    center: np.ndarray          #: (3,) lab point at a1 = a2 = 0 (m)
    quantity: str               #: "intensity" or "potential"
    units: str                  #: potential units ("W/m^2" for an intensity)

    def lab_points(self):
        """The lab coordinates ``(X, Y, Z)`` of the samples, broadcast (n1, n2)."""
        A1, A2 = self.a1[:, None], self.a2[None, :]
        return tuple(self.center[i] + A1 * self.e1[i] + A2 * self.e2[i] for i in range(3))


def plane_cut(obj, normal=(0.0, 1.0, 0.0), center=None, *, quantity: Optional[str] = None,
              half_width=None, n=(241, 241), in_plane_axis=None,
              units: str = "uK") -> PlaneCut:
    """Sample the intensity (a beam, crossed beams or a trap) or the potential (a
    trap) on the plane ``{r : (r - center) . normal = 0}``.

    ``center`` defaults to the trap minimum (``"minimum"``) or, for beams, the
    mean of their origins.  ``half_width`` is a scalar or an ``(h1, h2)`` pair
    (m); by default ~2.5 beam scales along each in-plane axis.
    """
    trap = _is_trap(obj)
    quantity = ("potential" if trap else "intensity") if quantity is None else quantity
    if quantity not in ("potential", "intensity"):
        raise ValueError(f"quantity must be 'potential' or 'intensity'; got {quantity!r}")
    if quantity == "potential" and not trap:
        raise ValueError("a potential needs a Trap: beams alone carry no atom or state.")
    scale = _check_units(units)
    e1, e2, nrm = _plane_axes(normal, in_plane_axis)

    if center is None or (isinstance(center, str) and center == "minimum"):
        if trap:
            c = np.array(obj.minimum().position, dtype=float)
        else:
            c = np.mean([b.origin for b in _beams_of(obj)], axis=0)
    else:
        c = np.array(center, dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            raise ValueError(f"center must be a finite 3-vector (m) or 'minimum'; got {center!r}")

    if half_width is None:
        h1, h2 = _extent_along(obj, e1), _extent_along(obj, e2)
    else:
        hw = np.atleast_1d(np.asarray(half_width, dtype=float))
        h1, h2 = (float(hw[0]), float(hw[0])) if hw.size == 1 else (float(hw[0]), float(hw[1]))
    n1, n2 = (int(n), int(n)) if np.ndim(n) == 0 else (int(n[0]), int(n[1]))
    cut = PlaneCut(np.linspace(-h1, h1, n1), np.linspace(-h2, h2, n2), np.empty(0),
                   e1, e2, nrm, c, quantity, "W/m^2" if quantity == "intensity" else units)
    X, Y, Z = cut.lab_points()
    if quantity == "intensity":
        field = obj.field if trap and hasattr(obj, "field") else obj
        cut.values = np.asarray(field.intensity(X, Y, Z), dtype=float)
    else:
        cut.values = np.asarray(obj.potential_J(X, Y, Z), dtype=float) * scale
    return cut


def line_cuts(trap, directions=None, *, half_width=None, n: int = 401, units: str = "uK"):
    """1D cuts through the minimum.  Returns a list of dicts with ``direction``,
    ``s`` (m), ``U`` and ``U_harmonic`` (in ``units``); directions default to the
    principal axes."""
    scale = _check_units(units)
    m = trap.minimum()
    hm = trap.harmonic()
    dirs = trap.trap_frequencies().axes if directions is None else directions
    out = []
    for d in dirs:
        d = fr.as_unit_real_vector(d, "direction")
        h = _extent_along(trap, d) if half_width is None else float(half_width)
        s = np.linspace(-h, h, int(n))
        X, Y, Z = (m.position[i] + s * d[i] for i in range(3))
        out.append(dict(direction=d, s=s,
                        U=np.asarray(trap.potential_J(X, Y, Z), dtype=float) * scale,
                        U_harmonic=np.asarray(hm.potential_J(X, Y, Z), dtype=float) * scale))
    return out


# ------------------------------------------------------------------ plots

def plot_plane_cut(obj, normal=(0.0, 1.0, 0.0), center=None, *,
                   quantity: Optional[str] = None, half_width=None, n=(241, 241),
                   in_plane_axis=None, units: str = "uK", ax=None, cmap=None,
                   contours: bool = True, levels=None, colorbar: bool = True,
                   mark_minimum: bool = True, show_gravity: bool = True,
                   title: Optional[str] = None):
    """Contour cut of the intensity or potential on an arbitrary plane.

    For a potential the thin grey contours sit at 10, 50 and 90% of the escape
    depth above the minimum and the red one at the escape energy itself -- the
    single most informative line on the plot.  For an intensity they are the
    1/e^2 and 1/e contours of the peak.  The minimum is marked with a white
    ``+`` when it lies in the plane (a hollow circle when projected into it),
    and an arrow shows the in-plane component of gravity.  Returns ``(fig, ax)``.
    """
    cut = plane_cut(obj, normal, center, quantity=quantity, half_width=half_width, n=n,
                    in_plane_axis=in_plane_axis, units=units)
    fig, ax = _axes(ax, figsize=(4.8, 4.0))
    potential = cut.quantity == "potential"
    F = cut.values
    ext = [cut.a1[0] * 1e6, cut.a1[-1] * 1e6, cut.a2[0] * 1e6, cut.a2[-1] * 1e6]
    im = ax.imshow(F.T, extent=ext, origin="lower", aspect="equal",
                   cmap=cmap or ("viridis_r" if potential else "magma"),
                   interpolation="bilinear")
    A1, A2 = np.meshgrid(cut.a1 * 1e6, cut.a2 * 1e6, indexing="ij")
    trap = _is_trap(obj)
    U_min = U_esc = None
    if potential:
        scale = UNIT_SCALE[units]
        U_min = obj.minimum().potential_J * scale
        depth = obj.trap_depth_J() * scale
        if np.isfinite(depth) and depth > 0:
            U_esc = U_min + depth
    if contours:
        if levels is None:
            if potential and U_esc is not None:
                levels = U_min + (U_esc - U_min) * np.array([0.1, 0.5, 0.9])
            elif not potential:
                levels = F.max() * np.array([np.exp(-2.0), np.exp(-1.0)])
        if levels is not None:
            lv = np.sort(np.asarray(levels, dtype=float))
            lv = lv[(lv > F.min()) & (lv < F.max())]
            if lv.size:
                ax.contour(A1.T, A2.T, F.T, levels=lv, colors="0.15", linewidths=0.6,
                           alpha=0.6)
        if U_esc is not None and F.min() < U_esc < F.max():
            ax.contour(A1.T, A2.T, F.T, levels=[U_esc], colors="tab:red", linewidths=1.2)

    if trap and mark_minimum:
        r = obj.minimum().position - cut.center
        p1, p2, off = float(r @ cut.e1), float(r @ cut.e2), abs(float(r @ cut.normal))
        pixel = min(np.diff(cut.a1[:2])[0], np.diff(cut.a2[:2])[0])
        if off <= pixel:
            ax.plot([p1 * 1e6], [p2 * 1e6], "+", color="w", ms=9, mew=1.6)
        else:
            ax.plot([p1 * 1e6], [p2 * 1e6], "o", mfc="none", mec="w", ms=7)
    if trap and show_gravity and getattr(obj, "gravity", False):
        g = obj.g_hat
        gp = np.array([g @ cut.e1, g @ cut.e2])
        if np.linalg.norm(gp) > 1e-3:
            gp = gp / np.linalg.norm(gp)
            x0, y0 = 0.87, 0.80
            ax.annotate("g", xy=(x0 + 0.08 * gp[0], y0 + 0.08 * gp[1]), xytext=(x0, y0),
                        xycoords="axes fraction", textcoords="axes fraction",
                        color="w", fontsize=9, ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", color="w", lw=1.2))

    ax.set_xlabel(_axis_label(cut.e1))
    ax.set_ylabel(_axis_label(cut.e2))
    ax.set_title(title if title is not None else f"cut normal to {_direction_label(cut.normal)}",
                 fontsize="small")
    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(f"U  ({UNIT_LABEL[units]})" if potential else r"I  (W/m$^2$)")
    _tidy(ax)
    return fig, ax


def plot_line_cuts(trap, directions=None, *, half_width=None, n: int = 401,
                   units: str = "uK", harmonic: bool = True, ax=None,
                   title: Optional[str] = None):
    """Potential along lines through the minimum (default: the principal axes),
    with the harmonic approximation dashed and the escape energy dotted.
    Returns ``(fig, ax)``."""
    cuts = line_cuts(trap, directions, half_width=half_width, n=n, units=units)
    fig, ax = _axes(ax, figsize=(5.0, 3.4))
    scale = UNIT_SCALE[units]
    U_min = trap.minimum().potential_J * scale
    depth = trap.trap_depth_J() * scale
    for c, colour in zip(cuts, _COLOURS * 3):
        ax.plot(c["s"] * 1e6, c["U"], color=colour, lw=1.4,
                label=f"along {_direction_label(c['direction'])}")
        if harmonic:
            ax.plot(c["s"] * 1e6, c["U_harmonic"], color=colour, lw=0.9, ls="--")
    if np.isfinite(depth) and depth > 0:
        ax.axhline(U_min + depth, color="tab:red", lw=0.9, ls=":", label="escape")
        ax.set_ylim(U_min - 0.05 * depth, U_min + 1.3 * depth)
    ax.set_xlabel(r"displacement from the minimum  ($\mu$m)")
    ax.set_ylabel(f"U  ({UNIT_LABEL[units]})")
    ax.legend(fontsize="xx-small", frameon=False)
    if title is not None:
        ax.set_title(title, fontsize="small")
    _tidy(ax)
    return fig, ax


def plot_beam_profile(beam, *, s_half: Optional[float] = None, ax=None,
                      title: Optional[str] = None):
    """The beam radii ``w_u(s)`` and ``w_v(s)`` against distance from the waist.
    Returns ``(fig, ax)``."""
    fig, ax = _axes(ax, figsize=(5.0, 3.2))
    zr = max(beam.rayleigh_range_u, beam.rayleigh_range_v)
    s = np.linspace(-1, 1, 401) * (3.0 * zr if s_half is None else float(s_half))
    ax.plot(s * 1e6, beam.beam_radius_u(s) * 1e6, color="tab:blue", lw=1.4, label=r"$w_u(s)$")
    ax.plot(s * 1e6, beam.beam_radius_v(s) * 1e6, color="tab:orange", lw=1.4,
            ls="--" if beam.waist_u == beam.waist_v else "-", label=r"$w_v(s)$")
    for z, colour in ((beam.rayleigh_range_u, "tab:blue"), (beam.rayleigh_range_v, "tab:orange")):
        ax.axvline(z * 1e6, color=colour, lw=0.7, ls=":")
        ax.axvline(-z * 1e6, color=colour, lw=0.7, ls=":")
    ax.set_xlabel(r"$s$, distance from the waist  ($\mu$m)")
    ax.set_ylabel(r"beam radius  ($\mu$m)")
    ax.set_ylim(0, None)
    ax.legend(fontsize="xx-small", frameon=False)
    ax.set_title(title if title is not None else repr(beam), fontsize="x-small")
    _tidy(ax)
    return fig, ax


def plot_trap_summary(trap, *, units: str = "uK", axes=None, title: Optional[str] = None):
    """Three lab planes through the minimum and the principal-axis line cuts,
    2 x 2.  Returns ``(fig, axes)``."""
    import matplotlib.pyplot as plt
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.8), layout="constrained")
    else:
        fig = np.asarray(axes).flat[0].figure
    axes = np.asarray(axes)
    for ax, normal in zip(axes.flat[:3], ((0, 0, 1), (0, 1, 0), (1, 0, 0))):
        plot_plane_cut(trap, normal=normal, units=units, ax=ax)
    plot_line_cuts(trap, units=units, ax=axes.flat[3])
    fig.suptitle(title if title is not None else repr(trap), fontsize="small")
    return fig, axes
