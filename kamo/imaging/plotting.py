"""Plotting helpers for kamo.imaging (matplotlib, waxa conventions).

Every function takes an optional ``ax`` (or builds its own figure) and returns
``(fig, ax)``, so they compose into larger layouts.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _axes(ax=None, figsize=(5, 3.4)):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    return fig, ax


def _tidy(ax):
    ax.tick_params(axis="both", which="both", direction="out",
                   top=False, right=False)
    ax.minorticks_off()
    return ax


def waist_limits(ax, waist, n_waists=3.0):
    """View ``+-n_waists`` trap waists about zero on both spatial axes.

    ``waist`` is the TWEEZER waist in metres -- the trap parameter the cloud is
    built in, not the cloud's own 1/e radii.  The two differ a lot here: a 3 um
    waist holds a cloud only 0.52 um wide transversely, so scaling the view to
    the cloud would crop every panel down to its own core and hide the wings,
    the diffraction and the box.  Scaling to the trap instead puts every spatial
    panel on ONE common scale, which is what makes them comparable across
    figures.

    The limits are symmetric because that is where the cloud centre sits, and
    identical on both axes for the same reason: a shared scale only works if it
    is actually shared.  Where the underlying data stops short of the window
    (the propagation domain along x ends at ``x_span_w`` cloud lengths) the
    margin is left blank rather than stretched -- that edge is real.

    A no-op when ``waist`` is None, so callers can pass it through unguarded.
    """
    if waist is None:
        return ax
    lim = float(n_waists) * float(waist) * 1e6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return ax


def _covering_half_width(half_width, waist, n_waists):
    """Crop half-width wide enough to contain the ``+-n_waists`` view.

    The spatial plots crop the data before drawing it, so a view window wider
    than the crop would show blank margin where there is really data.
    """
    if waist is None:
        return half_width
    return max(float(half_width), 1.02 * float(n_waists) * float(waist))


def plot_exit_wave(psi, grid, image=None, half_width=4.0e-6, title=None,
                   axes=None, waist=None, n_waists=3.0):
    """Transmission, wrapped phase and (optionally) the phase-contrast image.

    Pass ``waist`` (the trap waist, m) to view ``+-n_waists`` of it on both
    transverse axes rather than the whole crop.
    """
    import matplotlib.pyplot as plt

    n_panels = 2 if image is None else 3
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.6),
                                 layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.ravel(axes)

    half_width = _covering_half_width(half_width, waist, n_waists)
    win = grid.window(half_width)
    sl = np.ix_(win, win)
    ext = grid.extent_um(half_width)

    panels = [(np.abs(psi[sl]).T ** 2, r"transmission $|\psi|^2$", None),
              (np.angle(psi[sl]).T, "exit phase (rad, wrapped)", "twilight")]
    if image is not None:
        panels.append((image[sl].T, r"phase contrast $I/I_0$", None))

    for ax, (data, ttl, cmap) in zip(axes, panels):
        kw = dict(extent=ext, origin="lower")
        if cmap:
            kw.update(cmap=cmap, vmin=-np.pi, vmax=np.pi)
        im = ax.imshow(data, **kw)
        ax.set_title(ttl, fontsize="small")
        ax.set_xlabel(r"$y$ ($\mu$m)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        waist_limits(ax, waist, n_waists)
        _tidy(ax)
    axes[0].set_ylabel(r"$z$ ($\mu$m)")
    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def plot_far_field_cut(W, grid, reference=None, NA=None, ax=None,
                       label="BPM (complex polarizability)",
                       reference_label="Born form factor", title=None):
    """Cut through ``k_z = 0`` of the far field, in degrees from the probe axis."""
    fig, ax = _axes(ax, figsize=(5.0, 3.0))
    kl = np.fft.fftshift(grid.k_axis)
    ok = np.abs(kl) < grid.k
    angle = np.degrees(np.arcsin(np.clip(kl / grid.k, -1, 1)))
    c = grid.center

    cut = np.fft.fftshift(W)[:, c]
    ax.semilogy(angle[ok], (cut / cut.max())[ok], color="tab:purple", lw=1.6,
                label=label)
    if reference is not None:
        rc = np.fft.fftshift(reference)[:, c]
        ax.semilogy(angle[ok], (rc / rc.max())[ok], color="tab:red", lw=1.2,
                    ls="--", label=reference_label)
    if NA is not None:
        a = np.degrees(np.arcsin(NA))
        ax.axvspan(-a, a, color="0.85", zorder=0)
    ax.set_xlabel("angle from the probe axis (deg)")
    ax.set_ylabel("scattered power (peak = 1)")
    ax.set_xlim(-60, 60)
    ax.set_ylim(1e-5, 2)
    ax.legend(fontsize="xx-small", frameon=False, loc="upper right")
    if title:
        ax.set_title(title, fontsize="small")
    _tidy(ax)
    return fig, ax


def plot_intensity_xz(result, widths=None, z_half=2.5e-6, ax=None, vmax=None,
                      label=None, show_cloud=True, cbar=True, waist=None,
                      n_waists=3.0):
    """Probe intensity on the ``y = 0`` plane as it crosses the cloud.

    Diverging colour scale centred on 1, so focusing reads warm and shadowing
    cool at a glance.  Requires ``record='plane'`` or ``'3d'``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if result.intensity_plane is None:
        raise ValueError("propagate(..., record='plane') is needed for this plot")
    fig, ax = _axes(ax, figsize=(6.0, 2.6))

    w = result.widths if widths is None else np.asarray(widths)
    z = result.grid.axis
    zm = np.abs(z) <= _covering_half_width(z_half, waist, n_waists)
    I = result.intensity_plane[:, zm]
    x = result.x_slices
    ext = [x[0] * 1e6, x[-1] * 1e6, z[zm][0] * 1e6, z[zm][-1] * 1e6]

    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0,
                        vmax=float(I.max() if vmax is None else vmax))
    im = ax.imshow(I.T, extent=ext, origin="lower", aspect="auto", cmap="bwr",
                   norm=norm, interpolation="bilinear")
    if show_cloud and w is not None:
        X, Z = np.meshgrid(x, z[zm], indexing="ij")
        nrat = np.exp(-(X**2 / w[0]**2 + Z**2 / w[2]**2))
        ax.contour(X.T * 1e6, Z.T * 1e6, nrat.T,
                   levels=[np.exp(-2.0), np.exp(-1.0)],
                   colors="0.15", linewidths=(0.6, 0.9), alpha=0.6)
    if label:
        ax.text(0.015, 0.94, label, transform=ax.transAxes, ha="left", va="top",
                fontsize="small", color="0.1",
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="none", alpha=0.8))
    ax.set_xlabel(r"$x$, probe propagation ($\mu$m)")
    ax.set_ylabel(r"$z$ ($\mu$m)")
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(r"$I/I_0$", fontsize="small")
        cb.ax.axhline(1.0, color="0.25", lw=0.9)
    waist_limits(ax, waist, n_waists)
    _tidy(ax)
    return fig, ax


def plot_sky_maps(sky, sigma_sc, keys=("bare", "reabs", "coh", "coh+re"),
                  NA=None, axis=(1.0, 0.0, 0.0), axes=None, title=None):
    """Equal-solid-angle sky maps: every pixel subtends the same ``dOmega``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patheffects import withStroke
    from .farfield import PATTERN_LABELS, sphere_quadrature

    phi, u, maps = sky.sky_map(sigma_sc)
    n_sph, w_sph = sphere_quadrature()
    tot = {k: float((v * w_sph).sum())
           for k, v in sky.patterns(n_sph, sigma_sc).items()}
    rel = {k: 4 * np.pi * maps[k] / tot[k] for k in keys}
    vmin = min(float(np.nanmin(m)) for m in rel.values())
    vmax = max(float(np.nanmax(m)) for m in rel.values())

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True,
                                 sharey=True, layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.ravel(axes)

    if NA is not None:
        alpha = np.arcsin(NA)
        e3 = np.asarray(axis, dtype=float)
        e3 = e3 / np.linalg.norm(e3)
        beta = np.linspace(0, 2 * np.pi, 401)
        cone = (np.cos(alpha) * e3
                + np.sin(alpha) * (np.cos(beta)[:, None] * np.array([0., 1., 0.])
                                   + np.sin(beta)[:, None] * np.array([0., 0., 1.])))
        cone_phi = np.degrees(np.arctan2(cone[:, 1], cone[:, 0]))
        cone_u = cone[:, 2]

    for ax, k in zip(axes, keys):
        im = ax.pcolormesh(np.degrees(phi), u, rel[k],
                           norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto",
                           rasterized=True)
        if NA is not None:
            ax.plot(cone_phi, cone_u, color="w", lw=1.0,
                    path_effects=[withStroke(linewidth=2.2, foreground="k")])
        ax.set_title(PATTERN_LABELS.get(k, k), fontsize="small")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-1, 0, 1])
        _tidy(ax)
    for ax in axes[-2:]:
        ax.set_xlabel(r"$\varphi$ (deg), $0 = +x$")
    for ax in axes[::2]:
        ax.set_ylabel(r"$\cos\theta$")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.045, pad=0.02)
    cb.set_label(r"$4\pi\,(dP/d\Omega)/P$", fontsize="small")
    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def plot_cloud_size_vs_atom_number(cloud, N_range=None, ax=None, mark_N=True):
    """Variational widths against atom number, with both bracketing limits."""
    from matplotlib.lines import Line2D
    from .. BEC_properties.variational import GaussianVariationalCloud

    fig, ax = _axes(ax)
    N_range = np.logspace(1, 6, 121) if N_range is None else np.asarray(N_range)
    sig = np.full((len(N_range), 3), np.nan)
    for i, N in enumerate(N_range):
        c = cloud.with_atom_number(float(N))
        if not c.collapsed:
            sig[i] = c.sigma
    import kamo.constants as kc
    s_ni = cloud.sigma_noninteracting
    mu_tf = 0.5 * kc.hbar * cloud.omega_bar * (
        15 * N_range * cloud.a_scattering / cloud.a_ho) ** 0.4
    sig_tf = np.sqrt(2 * mu_tf[:, None] / (cloud.mass * cloud.omega[None, :] ** 2)) \
        / np.sqrt(7)

    for j, (color, label) in enumerate((("tab:blue", r"$\sigma_x$ (axial)"),
                                        ("tab:red", r"$\sigma_{y,z}$ (radial)"))):
        ax.loglog(N_range, sig[:, j] * 1e6, color=color, lw=1.6, label=label)
        ax.loglog(N_range, sig_tf[:, j] * 1e6, color=color, lw=1.0, ls=":")
        ax.axhline(s_ni[j] * 1e6, color=color, lw=1.0, ls="--")
    if mark_N and not cloud.collapsed:
        ax.plot([cloud.N] * 2, cloud.sigma[:2] * 1e6, "o", ms=5, mfc="w",
                color="k", zorder=5, label=f"$N = {cloud.N:.0f}$")

    ax.set_xlabel("Atom number $N$")
    ax.set_ylabel(r"rms cloud size ($\mu$m)")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color="0.5", lw=1.0, ls=":"),
          Line2D([], [], color="0.5", lw=1.0, ls="--")]
    l += ["Thomas-Fermi", "non-interacting"]
    ax.legend(h, l, fontsize="x-small", frameon=False, ncol=2, loc="upper left")
    ax.set_xlim(N_range[0], N_range[-1])
    _tidy(ax)
    return fig, ax


def plot_forward_scaling(sky, N_values, angles_deg=(0.0, 10.0, 40.0),
                         mark_N=None, axes=None, title=None, scan=None):
    """How coherent forward scattering pulls ahead of everything else with N.

    For positions drawn from ``n(r)/N`` the per-mode scattered power is

        <|sum_j exp(i q.r_j)|^2> = N + N(N-1) |f(q)|^2,

    so the amplitude adds in phase only where the form factor survives.  In the
    forward direction ``q = 0``, ``|f|^2 = 1`` and the power goes as ``N^2``; well
    outside the lobe ``|f|^2 -> 0`` and it goes as ``N``.  The forward mode
    therefore stands above the rest by a factor of order ``N`` -- which is the whole
    reason a dispersive measurement can work at NA far from the dipole maximum.

    Pass ``scan`` -- a :class:`~kamo.imaging.readout.ForwardModeScan` -- to
    overlay the same quantity as the FULL lensing model actually computes it,
    rather than only the scaling law.  The BPM points are normalized to the
    analytic forward curve at the smallest ``N`` in the scan, which is the point
    deepest in the weak-phase limit and therefore the one place the two are
    guaranteed to agree; everything after that is the model departing from
    ``N^2`` on its own.

    Left panel: power against ``N`` at each angle, with ``N^2`` and ``N`` guides.
    Middle panel (only with ``scan``): the BPM forward mode divided by ``N^2``,
    which is where the departure actually lives -- it is invisible on a log-log
    plot spanning ten decades.
    Right panel: the angular pattern normalized per atom, so a curve that is flat
    is incoherent and a curve that rises by ``N`` is coherent.
    """
    import matplotlib.pyplot as plt

    N = np.asarray(N_values, dtype=float)
    n_panels = 2 if scan is None else 3
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.3),
                                 layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.ravel(axes)

    def nhat(deg):
        t = np.radians(deg)
        return np.array([np.cos(t), np.sin(t), 0.0])

    ax = axes[0]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
    for i, deg in enumerate(angles_deg):
        f2 = float(sky.form_factor_sq(nhat(deg)))
        P = N + N * (N - 1) * f2
        c = colors[i % len(colors)]
        ax.loglog(N, P, "-", color=c, lw=1.8,
                  label=rf"${deg:.0f}^\circ$,  $|f|^2 = {f2:.1e}$")
    ax.loglog(N, N**2, ":", color="0.35", lw=1.1, label=r"$N^2$")
    ax.loglog(N, N, "--", color="0.35", lw=1.1, label=r"$N$")

    ratio = None
    if scan is not None:
        Ns = np.asarray(scan.N, dtype=float)
        P_bpm = np.asarray(scan.forward, dtype=float)
        # Anchor on the weakest-phase point, where the propagation and the Born
        # form factor must agree; the scale carries the physical units away.
        k0 = int(np.argmin(Ns))
        anchor = (Ns[k0] + Ns[k0] * (Ns[k0] - 1)
                  * float(sky.form_factor_sq(nhat(0.0))))
        P_scaled = P_bpm * (anchor / P_bpm[k0])
        ax.loglog(Ns, P_scaled, "o", color="k", ms=5, mfc="none", mew=1.4,
                  zorder=6, label="BPM forward mode")
        ratio = P_scaled / Ns**2 * (Ns[k0]**2 / anchor)

    if mark_N:
        ax.axvline(mark_N, color="0.7", lw=1.0)
    ax.set_xlabel("atom number $N$")
    ax.set_ylabel("scattered power into the mode")
    ax.legend(fontsize="xx-small", frameon=False, loc="upper left")
    _tidy(ax)

    if scan is not None:
        ax = axes[1]
        ax.semilogx(scan.N, ratio, "o-", color="k", lw=1.5, ms=5)
        ax.axhline(1.0, color="0.35", ls=":", lw=1.2, label=r"exactly $N^2$")
        if mark_N:
            ax.axvline(mark_N, color="0.7", lw=1.0)
        # Where the weak-phase description stops being the right one.
        one_rad = np.asarray(scan.phi_thin, dtype=float) > 1.0
        if one_rad.any():
            ax.axvspan(float(np.asarray(scan.N)[one_rad].min()),
                       float(np.asarray(scan.N).max()),
                       color="0.9", zorder=0,
                       label=r"$\varphi_{\rm thin} > 1$ rad")
        ax.set_xlabel("atom number $N$")
        ax.set_ylabel(r"BPM forward mode / $N^2$")
        ax.legend(fontsize="xx-small", frameon=False, loc="lower left")
        _tidy(ax)

    ax = axes[-1]
    theta = np.linspace(0, 60, 400)
    f2 = np.array([float(sky.form_factor_sq(nhat(d))) for d in theta])
    for i, n in enumerate([N[0], N[len(N) // 2], N[-1]]):
        ax.semilogy(theta, 1.0 + (n - 1) * f2, lw=1.6,
                    color=colors[i % len(colors)], label=f"$N = {n:.0f}$")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.1)
    ax.set_xlabel("angle from the probe axis (deg)")
    ax.set_ylabel("power per atom / incoherent")
    ax.legend(fontsize="xx-small", frameon=False, loc="upper right")
    _tidy(ax)

    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def plot_beer_limit(fresnel, ratio, profiles=None, mark=None, axes=None,
                    title=None, n_waists=3.0):
    """Convergence to Beer's law as a cloud is made transversely extended.

    Parameters
    ----------
    fresnel : array_like
        Fresnel number ``F = k w_perp^2 / L`` of each case.  ``F >> 1`` means the
        cloud's own transverse structure does not diffract appreciably over the
        cloud's length -- the definition of an extended (Beer-law) absorber.
    ratio : array_like
        Measured on-axis optical depth divided by the Beer prediction
        ``sigma n_col``.
    profiles : (y, T_meas, T_beer, label) tuples, optional
        Transmission cuts to overlay in a second panel.
    mark : (F, ratio, label), optional
        Highlight one case -- normally the real cloud.
    """
    import matplotlib.pyplot as plt

    n = 1 if profiles is None else 2
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.2), layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.semilogx(fresnel, ratio, "o-", color="tab:blue", lw=1.5, ms=4)
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0, label="Beer's law")
    ax.axvline(1.0, color="0.75", ls=":", lw=1.0)
    if mark is not None:
        ax.plot([mark[0]], [mark[1]], "o", ms=9, mfc="none", mew=1.8,
                color="tab:red", label=mark[2], zorder=5)
    ax.set_xlabel(r"Fresnel number $F = k w_\perp^2 / L$")
    ax.set_ylabel(r"measured $D$ / Beer $\sigma \tilde n$")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize="xx-small", frameon=False, loc="lower right")
    _tidy(ax)

    if profiles is not None:
        ax = axes[1]
        colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
        for i, (y, T_meas, T_beer, label) in enumerate(profiles):
            c = colors[i % len(colors)]
            ax.plot(np.asarray(y), T_meas, "-", color=c, lw=1.6, label=label)
            ax.plot(np.asarray(y), T_beer, "--", color=c, lw=1.0, alpha=0.8)
        ax.set_xlabel(r"$y / w_\perp$")
        ax.set_ylabel(r"transmission $|\psi|^2$")
        # already in waist units, so the +-n_waists view is literally +-n_waists
        ax.set_xlim(-n_waists, n_waists)
        ax.legend(fontsize="xx-small", frameon=False, loc="lower right",
                  title="solid: propagated,  dashed: Beer",
                  title_fontsize="xx-small")
        _tidy(ax)

    if title:
        fig.suptitle(title, fontsize="small")
    return fig, axes


def linear_response(xi, signal, half_width: float = 0.2, min_points: int = 3):
    """Slope ``dS/dxi`` near balance, from the points inside ``half_width``.

    Widens the window if a coarse scan puts fewer than ``min_points`` inside it --
    a fit over one point is not an error worth raising here, it just means the
    scan was sparse.
    """
    xi = np.asarray(xi, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if xi.size < 2:
        return float("nan")
    order = np.argsort(np.abs(xi))
    n = max(int((np.abs(xi) < half_width).sum()), min(min_points, xi.size))
    sel = order[:n]
    return float(np.polyfit(xi[sel], signal[sel], 1)[0])


def plot_imbalance_scan(xi, signal, phase=None, thin_screen_phase=None,
                        N_atoms=None, axes=None):
    """Recovered phase and phase-contrast signal against spin imbalance."""
    import matplotlib.pyplot as plt

    n_panels = 1 if phase is None else 2
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.0),
                                 layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    axes = np.atleast_1d(axes)

    shot = 1.0 / np.sqrt(N_atoms) if N_atoms else None
    k = 0
    if phase is not None:
        ax = axes[0]
        k = 1
        if thin_screen_phase is not None:
            ax.plot(xi, xi * thin_screen_phase, color="0.55", lw=1.2, ls="--",
                    label="thin screen")
        ax.plot(xi, phase, "o-", color="tab:purple", lw=1.5, ms=3.5,
                label="BPM, refocused")
        ax.set_ylabel("recovered phase (rad)")
        ax.legend(fontsize="xx-small", frameon=False, loc="upper left")

    ax = axes[k]
    slope = linear_response(xi, signal)
    ax.plot(xi, slope * np.asarray(xi), color="0.55", lw=1.2, ls="--",
            label=r"linear in $\xi$")
    ax.plot(xi, signal, "o-", color="tab:red", lw=1.5, ms=3.5,
            label="BPM phase contrast")
    ax.set_ylabel(r"on-axis $I/I_0 - 1$")
    ax.legend(fontsize="xx-small", frameon=False, loc="upper left")

    for ax in axes:
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        if shot:
            ax.axvspan(-shot, shot, color="0.88", zorder=0)
        ax.set_xlabel(r"spin imbalance $\xi = p_\uparrow - p_\downarrow$")
        _tidy(ax)
    return fig, axes
