"""Plotting helpers for kamo.spin (matplotlib, waxa conventions)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from kamo.imaging.plotting import (_axes, _covering_half_width,
                                   _tidy, waist_limits)


def plot_spin_columns(field, axes=None, title=None, waist=None,
                      n_waists=3.0):
    """Column imbalance and column coherence, as an image would resolve them."""
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.ravel(axes)

    g = field.geometry
    xi_col, Z_col = field.collapse_to_columns()
    ext = [g.y[0] * 1e6, g.y[-1] * 1e6, g.z[0] * 1e6, g.z[-1] * 1e6]

    im0 = axes[0].imshow(xi_col.T, extent=ext, origin="lower", cmap="RdBu_r",
                         vmin=-1, vmax=1)
    axes[0].set_title(r"column imbalance $\xi$", fontsize="small")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.abs(Z_col).T, extent=ext, origin="lower",
                         vmin=0, vmax=1)
    axes[1].set_title(r"column coherence $|Z|$", fontsize="small")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xlabel(r"$y$ ($\mu$m)")
        waist_limits(ax, waist, n_waists)
        _tidy(ax)
    axes[0].set_ylabel(r"$z$ ($\mu$m)")
    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def plot_phase_grating(result, geometry, axes=None, title=None,
                       waist=None, n_waists=3.0):
    """The 3D imprint: ``phi_z`` on the ``y = 0`` plane, and its column statistics.

    The left panel is what makes the case for a three-dimensional spin state --
    if ``phi_z`` were flat along x, a column-averaged state would lose nothing.

    The third panel is drawn on a LOG colour scale, as the dephasing *deficit*
    ``1 - |<exp(i phi)>|`` rather than the coherence itself.  The coherence at
    the operating point runs 0.998 to 1, so on a linear 0-1 scale the panel is
    a blank white square -- and a log scale on the coherence is no better,
    since ``log(0.998)`` to ``log(1)`` renormalizes back to something
    indistinguishable from linear.  The deficit is the quantity that actually
    spans decades (order 1e-6 to 1e-3 here), so it is the one that carries the
    structure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.ravel(axes)

    g = geometry
    mid = g.nw // 2
    ext_xz = [g.x[0] * 1e6, g.x[-1] * 1e6, g.z[0] * 1e6, g.z[-1] * 1e6]
    ext_yz = [g.y[0] * 1e6, g.y[-1] * 1e6, g.z[0] * 1e6, g.z[-1] * 1e6]

    im0 = axes[0].imshow(result.phi_z[:, mid, :].T, extent=ext_xz, origin="lower",
                         aspect="auto", cmap="magma")
    axes[0].set_title(r"$\phi_z(x, 0, z)$ (rad)", fontsize="small")
    axes[0].set_xlabel(r"$x$ ($\mu$m)")
    axes[0].set_ylabel(r"$z$ ($\mu$m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(result.phi_z_column.T, extent=ext_yz, origin="lower",
                         cmap="magma")
    axes[1].set_title(r"column mean $\langle\phi_z\rangle$ (rad)", fontsize="small")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    deficit = 1.0 - result.dephasing_factor
    # A floor keeps LogNorm well posed where a column is empty and the deficit
    # underflows to exactly zero; it sits a decade below the smallest real value.
    pos = deficit[deficit > 0]
    floor = float(pos.min()) * 0.1 if pos.size else 1e-12
    im2 = axes[2].imshow(np.maximum(deficit, floor).T, extent=ext_yz,
                         origin="lower", cmap="magma",
                         norm=LogNorm(vmin=floor,
                                      vmax=float(max(deficit.max(),
                                                     floor * 10))))
    axes[2].set_title(r"within-column dephasing "
                      r"$1 - |\langle e^{i\phi}\rangle|$",
                      fontsize="small")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046)
    cb2.set_label(rf"min $|\langle e^{{i\phi}}\rangle| = "
                  rf"{result.dephasing_factor.min():.4f}$", fontsize="xx-small")

    for ax in axes[1:]:
        ax.set_xlabel(r"$y$ ($\mu$m)")
        ax.set_ylabel(r"$z$ ($\mu$m)")
    for ax in axes:
        waist_limits(ax, waist, n_waists)
    for ax in axes:
        _tidy(ax)
    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def plot_sequence_images(seq_result, grid, half_width=3.0e-6, labels=None,
                         axes=None, title=None, waist=None, n_waists=3.0):
    """Phase-contrast images from every pulse in a sequence, on one colour scale."""
    import matplotlib.pyplot as plt

    imgs = seq_result.images
    n = len(imgs)
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.8), layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.atleast_1d(axes)

    half_width = _covering_half_width(half_width, waist, n_waists)
    win = grid.window(half_width)
    sl = np.ix_(win, win)
    ext = grid.extent_um(half_width)
    lo = min(float(r.image[sl].min()) for r in imgs)
    hi = max(float(r.image[sl].max()) for r in imgs)

    for i, (ax, r) in enumerate(zip(axes, imgs)):
        im = ax.imshow(r.image[sl].T, extent=ext, origin="lower", vmin=lo, vmax=hi)
        lbl = labels[i] if labels else f"pulse {i + 1}"
        ax.set_title(f"{lbl}\n" + rf"$I/I_0-1 = {r.signal:+.4f}$", fontsize="small")
        ax.set_xlabel(r"$y$ ($\mu$m)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        waist_limits(ax, waist, n_waists)
        _tidy(ax)
    axes[0].set_ylabel(r"$z$ ($\mu$m)")
    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def plot_bloch_summary(fields, labels=None, ax=None, title=None):
    """Cloud-averaged Bloch vectors on a sphere, one arrow per state."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(4.2, 4.2), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="0.85", lw=0.4)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, f in enumerate(np.atleast_1d(fields)):
        s = 2 * f.mean_spin()
        ax.quiver(0, 0, 0, s[0], s[1], s[2], color=colors[i % len(colors)],
                  lw=2.0, arrow_length_ratio=0.12,
                  label=(labels[i] if labels else f"state {i}"))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel(r"$s_x$"); ax.set_ylabel(r"$s_y$"); ax.set_zlabel(r"$s_z$")
    ax.legend(fontsize="xx-small", frameon=False, loc="upper left")
    if title:
        ax.set_title(title, fontsize="small")
    return fig, ax


def signed_offset_ticks(ax, max_ticks: int = 7, unit: str = "kHz", fmt: str = "{:+.1f}"):
    """Thin an axis to ``max_ticks`` labels, each written as a signed offset.

    A fringe scan has many more sample points than an axis has room for labels;
    ticking every sample overlaps them into an unreadable band.  This picks a
    round subset and gives each an explicit ``+``/``-`` so the sign relative to the
    midpoint is unambiguous at a glance, with the unit stated once in the axis
    label rather than repeated on every tick.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None,
                                           steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else fmt.format(v)))
    return ax


def plot_ramsey_fringe(detunings_Hz, signals, contrast=None, ax=None, title=None,
                       max_ticks: int = 7):
    """Phase-contrast signal against detuning from the midpoint -- the fringe.

    The x axis is the SIGNED offset from the midpoint, ticked sparsely so the
    labels stay legible however finely the fringe is sampled.
    """
    fig, ax = _axes(ax, figsize=(5.4, 3.2))
    x = np.asarray(detunings_Hz) * 1e-3
    ax.plot(x, signals, "o-", color="tab:red", lw=1.4, ms=3.5,
            label="phase-contrast signal")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("detuning from midpoint (kHz)")
    ax.set_ylabel(r"on-axis $I/I_0 - 1$")

    handles, labels = ax.get_legend_handles_labels()
    if contrast is not None:
        # Contrast is a different quantity on a different scale; give it its own
        # axis rather than letting it compress the fringe it is meant to explain.
        axc = ax.twinx()
        axc.plot(x, contrast, "s--", color="0.45", lw=1.0, ms=3.0,
                 label="spin contrast")
        axc.set_ylabel("spin contrast", color="0.35")
        axc.set_ylim(0, 1.05)
        axc.tick_params(axis="y", colors="0.35", direction="out")
        axc.minorticks_off()
        h2, l2 = axc.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, fontsize="xx-small", frameon=False,
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    if title:
        ax.set_title(title, fontsize="small")
    _tidy(ax)
    signed_offset_ticks(ax, max_ticks=max_ticks)
    return fig, ax
