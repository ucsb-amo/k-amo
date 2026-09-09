"""Geometric-optics (refraction-only) lensing of the probe by the all-up cloud.

Cloud density from the tweezer trap parameters, refractive index from the
two-level polarizability at the operating detuning, then EXACT ray tracing
through the graded index.  No absorption, no diffraction, no saturation --
this is the pure-refraction picture, for comparison with the full BPM result.

    n_index(r) = sqrt(1 + 4 pi n_density(r) alpha_vol),
    alpha_vol  = Re alpha_SI / (4 pi eps0)      [m^3, polarizability volume]

Run:  ..\.venv\Scripts\python.exe docs\refraction_ray_lensing.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import kamo.constants as kc
from kamo.BEC_properties.variational import GaussianVariationalCloud

OUT = r"G:\Shared drives\Tweezers\Projects - Active\jpagett_PCI_in_BEC_writeup"

# ---------------------------------------------------------------- parameters
import sys
N_ATOMS = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0   # all-up cloud
A_UPUP_A0 = 11.333713034704147   # a0, at B = 520.594 G
F_RADIAL = 1.0e3                 # Hz
WAIST = 3.0e-6                   # m, 1064 nm tweezer
F_D2_UP = 391.018093e12          # Hz, the |up> sigma- transition
DET_HALF = 55.1103e6             # Hz, probe sits this far BELOW the |up> line
GAMMA_HZ = 6.0050e6              # Gamma / 2 pi
SIGMA0 = 2.8316243e-13           # m^2, as used by the notebook (waxa's value)

# ------------------------------------------------------------------- cloud
cloud = GaussianVariationalCloud.from_tweezer(
    N=N_ATOMS, a_scattering=A_UPUP_A0 * kc.a0, f_radial_Hz=F_RADIAL, waist=WAIST)
wx, wy, wz = cloud.widths            # 1/e radii of the DENSITY
n_peak = cloud.peak_density

# --------------------------------------------------------- polarizability
# Probe is RED detuned from |up> (it sits at the midpoint, below that line), so
# delta < 0 -> Re alpha > 0 -> n > 1 -> converging.  Same convention as the
# notebook's polarizability():  alpha = -(eps0 sigma0 / k)(delta - i)/(1 + delta^2)
lam = kc.c / F_D2_UP
k_img = 2 * np.pi / lam
delta = -2.0 * DET_HALF / GAMMA_HZ                    # signed, = -18.35
alpha_re = -(kc.epsilon_0 * SIGMA0 / k_img) * delta / (1 + delta**2)
alpha_vol = alpha_re / (4 * np.pi * kc.epsilon_0)     # m^3

chi_peak = 4 * np.pi * n_peak * alpha_vol
n_peak_idx = np.sqrt(1 + chi_peak)
n_lin = 1 + chi_peak / 2                              # the notebook's linearization

print("=" * 72)
print("CLOUD  (all-up, N = %.0f)" % N_ATOMS)
print("  widths w (1/e density radii)  = (%.3f, %.3f, %.3f) um"
      % (wx * 1e6, wy * 1e6, wz * 1e6))
print("  peak density                  = %.4e m^-3  = %.4e cm^-3"
      % (n_peak, n_peak * 1e-6))
print("  n_peak / k^3                  = %.4f" % (n_peak / k_img**3))
print("\nPOLARIZABILITY  (two-level, delta = %.4f)" % delta)
print("  lambda                        = %.3f nm" % (lam * 1e9))
print("  Re alpha_SI                   = %.5e C m^2 / V" % alpha_re)
print("  alpha_vol = Re alpha/(4 pi e0)= %.5e m^3" % alpha_vol)
print("                                = %.4f  (in units of 1/k^3)"
      % (alpha_vol * k_img**3))
print("\nINDEX")
print("  chi_peak = 4 pi n alpha_vol   = %.5f" % chi_peak)
print("  n_peak  = sqrt(1 + chi)       = %.6f   (n-1 = %.5f)"
      % (n_peak_idx, n_peak_idx - 1))
print("  n_peak  = 1 + chi/2 (notebook)= %.6f   (n-1 = %.5f)" % (n_lin, n_lin - 1))
print("  sqrt vs linear, in (n-1)      = %.2f%% "
      % (100 * (n_lin - n_peak_idx) / (n_peak_idx - 1)))


# ------------------------------------------------------------ index & rays
def chi_of(x, y):
    return chi_peak * np.exp(-(x**2 / wx**2 + y**2 / wy**2))


def n_of(x, y):
    return np.sqrt(1.0 + chi_of(x, y))


def grad_n(x, y):
    """Analytic gradient of n = sqrt(1+chi)."""
    c = chi_of(x, y)
    n = np.sqrt(1.0 + c)
    return (c * (-2 * x / wx**2)) / (2 * n), (c * (-2 * y / wy**2)) / (2 * n)


def trace(y0, x_start, x_end, n_steps=6000):
    """Exact ray equation d/ds (n dr/ds) = grad n, RK4 in arclength.

    dT/ds = [grad n - T (T . grad n)] / n  keeps |T| = 1 without renormalizing.
    """
    ds = (x_end - x_start) / n_steps
    r = np.array([x_start, y0], float)
    T = np.array([1.0, 0.0], float)
    path = [r.copy()]

    def deriv(r, T):
        gx, gy = grad_n(r[0], r[1])
        g = np.array([gx, gy])
        n = n_of(r[0], r[1])
        return T, (g - T * float(g @ T)) / n

    for _ in range(n_steps):
        k1r, k1t = deriv(r, T)
        k2r, k2t = deriv(r + 0.5 * ds * k1r, T + 0.5 * ds * k1t)
        k3r, k3t = deriv(r + 0.5 * ds * k2r, T + 0.5 * ds * k2t)
        k4r, k4t = deriv(r + ds * k3r, T + ds * k3t)
        r = r + (ds / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        T = T + (ds / 6) * (k1t + 2 * k2t + 2 * k3t + k4t)
        T /= np.linalg.norm(T)
        path.append(r.copy())
    return np.array(path)


X0, X1 = -4.0 * wx, 12.0 * wx
Y_MAX = 2.5 * wy
y_in = np.linspace(-Y_MAX, Y_MAX, 41)
paths = [trace(y, X0, X1) for y in y_in]

# --- GRIN parameters -------------------------------------------------------
# Near the axis chi ~ chi0 (1 - r^2/wy^2), so n ~ n0 (1 - g^2 r^2 / 2) with
# g = sqrt(chi0) / (n0 wy).  That is a graded-index waveguide, not a thin lens:
# rays oscillate with pitch 2 pi / g.
n0 = n_peak_idx
g_grin = np.sqrt(chi_peak) / (n0 * wy)
pitch = 2 * np.pi / g_grin
quarter = 0.25 * pitch

_p = trace(0.15 * wy, X0, X1, n_steps=12000)
_cross = np.where(np.diff(np.sign(_p[:, 1])) != 0)[0]
x_f = (np.interp(0.0, [_p[_cross[0], 1], _p[_cross[0] + 1, 1]],
                 [_p[_cross[0], 0], _p[_cross[0] + 1, 0]])
       if len(_cross) else np.nan)
n_cross_inside = int(np.sum(np.abs(_p[_cross, 0]) < 2 * wx)) if len(_cross) else 0

# --- intensity by RAY BINNING (valid through caustics) ---------------------
# Past the first crossing the map r_in -> r_out is multivalued, so the Jacobian
# formula I = (r_in/r_out)|dr_in/dr_out| breaks down (it diverges at every fold).
# Binning instead SUMS every ray landing in an annulus, which is finite and
# correct wherever rays overlap.  Weight ∝ r_in dr_in is the annular area of the
# incident parallel bundle.
N_RAYS = 3000
R_LAUNCH = 3.0 * wy
r_in = (np.arange(N_RAYS) + 0.5) * R_LAUNCH / N_RAYS
r_paths = np.array([trace(r, X0, X1, n_steps=2500)[:, 1] for r in r_in])
x_axis = trace(0.0, X0, X1, n_steps=2500)[:, 0]
w_ray = r_in * (R_LAUNCH / N_RAYS)                 # ∝ annulus area

R_EDGES = np.linspace(0.0, R_LAUNCH, 90)
R_CENT = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
A_ANN = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)
I0_NORM = np.sum(w_ray) / (np.pi * R_LAUNCH**2)    # incident uniform intensity


def intensity_profile(ix):
    """I(r)/I0 at plane ix, by binning all rays into annuli."""
    h, _ = np.histogram(np.abs(r_paths[:, ix]), bins=R_EDGES, weights=w_ray)
    return h / A_ANN / I0_NORM


I_map = np.array([intensity_profile(ix) for ix in range(len(x_axis))])

# density-weighted <I>, the same average run_bpm's mean_I reports
WGT = np.exp(-(x_axis[:, None] ** 2 / wx**2 + R_CENT[None, :] ** 2 / wy**2)) \
    * R_CENT[None, :]
mean_I_ray = float(np.sum(WGT * I_map) / np.sum(WGT))
I_axis = I_map[:, 0]

print("\nGRIN WAVEGUIDE  (the cloud is NOT a thin lens)")
print("  g = sqrt(chi0)/(n0 wy)          = %.4e m^-1" % g_grin)
print("  full pitch 2 pi / g             = %.3f um" % (pitch * 1e6))
print("  quarter pitch (first focus)     = %.3f um" % (quarter * 1e6))
print("  cloud full length 2 w_x         = %.3f um  = %.2f pitches"
      % (2 * wx * 1e6, 2 * wx / pitch))
print("  first axis crossing at x        = %+.3f um (inside the cloud)" % (x_f * 1e6))
print("  axis crossings within +-2 w_x   = %d" % n_cross_inside)

print("\nRAY TRACING  (refraction only: no absorption, no diffraction, no saturation)")
print("  on-axis peak I/I0               = %.1f" % I_axis.max())
print("  density-weighted <I>/I0         = %.3f" % mean_I_ray)
print("  BPM <I>/I0 (N=500, with         = 1.818")
print("    absorption+diffraction+sat)")
print("=" * 72)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)

XZ, YZ = 4.0 * wx, 2.0 * wy          # zoom on the cloud for (a), (b)
xg = np.linspace(-XZ, XZ, 500)
yg = np.linspace(-YZ, YZ, 500)
XG, YG = np.meshgrid(xg, yg, indexing="ij")
NG = n_of(XG, YG)

# (a) index map
ax = fig.add_subplot(gs[0, 0])
im = ax.pcolormesh(XG * 1e6, YG * 1e6, NG - 1, cmap="magma", shading="auto")
ax.contour(XG * 1e6, YG * 1e6, np.exp(-(XG**2 / wx**2 + YG**2 / wy**2)),
           levels=[np.exp(-2.0), np.exp(-1.0)], colors="w", linewidths=0.7,
           alpha=0.7)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"$n_{\rm index}-1$", fontsize="small")
ax.set_xlabel(r"$x$ ($\mu$m)", fontsize="small")
ax.set_ylabel(r"$y$ ($\mu$m)", fontsize="small")
ax.tick_params(labelsize="x-small")
ax.set_title(r"(a) $n=\sqrt{1+4\pi n_{\rm at}\alpha_{\rm vol}}$,"
             rf"  peak $n-1={n_peak_idx-1:.4f}$", fontsize="small")

# (b) ray bundle, zoomed so the GRIN oscillation is visible
ax = fig.add_subplot(gs[0, 1])
ax.pcolormesh(XG * 1e6, YG * 1e6, NG - 1, cmap="magma", shading="auto", alpha=0.55)
for p in paths:
    ax.plot(p[:, 0] * 1e6, p[:, 1] * 1e6, lw=0.6, color="tab:cyan", alpha=0.9)
ax.axvline(x_f * 1e6, color="w", ls="--", lw=1.0)
ax.text(x_f * 1e6, YZ * 1e6 * 0.94, rf"  first focus  {x_f*1e6:+.2f}$\,\mu$m",
        color="w", fontsize="xx-small", va="top")
ax.set_xlim(-XZ * 1e6, XZ * 1e6)
ax.set_ylim(-YZ * 1e6, YZ * 1e6)
ax.set_xlabel(r"$x$ ($\mu$m)", fontsize="small")
ax.set_ylabel(r"$y$ ($\mu$m)", fontsize="small")
ax.tick_params(labelsize="x-small")
ax.set_title(rf"(b) rays: GRIN pitch ${pitch*1e6:.2f}\,\mu$m,"
             rf"  ${2*wx/pitch:.2f}$ pitches per cloud", fontsize="small")

# (c) intensity map from ray binning
ax = fig.add_subplot(gs[1, 0])
im = ax.pcolormesh(x_axis * 1e6, R_CENT * 1e6, np.clip(I_map.T, 0.03, 30),
                   cmap="RdBu_r", norm=LogNorm(vmin=0.03, vmax=30),
                   shading="auto")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"$I/I_0$", fontsize="small")
ax.axvline(0.0, color="0.3", lw=0.7, ls=":")
ax.set_xlim(-XZ * 1e6, 8 * wx * 1e6)
ax.set_ylim(0, 2.0 * wy * 1e6)
ax.set_xlabel(r"$x$ ($\mu$m)", fontsize="small")
ax.set_ylabel(r"$r$ ($\mu$m)", fontsize="small")
ax.tick_params(labelsize="x-small")
ax.set_title("(c) intensity from ray binning (caustic-safe)", fontsize="small")

# (d) on-axis intensity
ax = fig.add_subplot(gs[1, 1])
ax.plot(x_axis * 1e6, I_axis, lw=1.5, color="tab:red", label=r"on axis, $r\approx0$")
ax.axhline(1.0, color="0.55", lw=0.9, ls=":")
ax.axhline(mean_I_ray, color="tab:purple", lw=1.2, ls="--",
           label=rf"density-weighted $\langle I\rangle={mean_I_ray:.2f}$")
ax.axhline(1.818, color="tab:green", lw=1.2, ls="-.",
           label=r"BPM $\langle I\rangle=1.82$ ($N{=}500$)")
ax.axvspan(-wx * 1e6, wx * 1e6, color="0.85", alpha=0.5, zorder=0)
ax.text(0, ax.get_ylim()[1], " cloud", fontsize="xx-small", color="0.45",
        va="top", ha="center")
ax.set_yscale("log")
ax.set_xlim(-XZ * 1e6, 8 * wx * 1e6)
ax.set_xlabel(r"$x$ ($\mu$m)", fontsize="small")
ax.set_ylabel(r"$I/I_0$ on axis", fontsize="small")
ax.tick_params(labelsize="x-small")
ax.legend(fontsize="xx-small", frameon=False, loc="upper right")
ax.set_title("(d) on-axis intensity along the beam", fontsize="small")

fig.suptitle("Pure-refraction lensing of the probe by the all-up cloud "
             rf"($N={N_ATOMS:.0f}$, $\Delta=-{DET_HALF/1e6:.2f}$ MHz)",
             fontsize="medium", y=0.985)
import os
fig.savefig(os.path.join(OUT, "refraction_ray_lensing.png" if N_ATOMS==1000 else "_scratch_%d.png"%N_ATOMS), dpi=250,
            bbox_inches="tight")
print("saved:", os.path.join(OUT, "refraction_ray_lensing.png"))
