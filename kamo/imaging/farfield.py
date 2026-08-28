"""Where the light a cloud scatters actually goes.

The complement to :mod:`kamo.imaging.bpm`.  A propagation carries the coherent
(elastic) field; this module handles the angular bookkeeping that a propagation
does not: the incoherent fluorescence channel, reabsorption of scattered photons
on their way out, and the first-Born coherent form factor used as a reference.

Summed incoherently over N atoms, the single-atom pattern is just N copies of
itself -- the density profile cancels out.  It survives in exactly two ways, and
both are computed here:

**Reabsorption.**  A photon born at ``r`` leaving along ``nhat`` is attenuated by
``exp[-sigma col(r, nhat)]``.  For a Gaussian cloud the average over emission
points collapses to a single 1D quadrature depending on direction ONLY through the
peak line optical depth.  Rescaling ``x_i -> x_i / w_i`` makes the cloud spherical,
so with longitudinal coordinate ``zeta`` and transverse ``rho`` along the ray,

    sigma col = (tau/2) exp(-rho^2) erfc(zeta),
    tau(nhat) = sigma n0 sqrt(pi) / |m|,        m_i = nhat_i / w_i,

and the ``rho`` integral is elementary in ``v = exp(-rho^2)``, leaving

    T(tau) = <exp(-OD)> = pi^{-1/2} int dzeta e^{-zeta^2} (1 - e^{-K}) / K,
                                                 K = (tau/2) erfc(zeta).

**Coherence.**  Driven by one probe, the dipoles add with phases.  For positions
drawn from ``n(r)/N``,

    <|sum_j exp(i q.r_j)|^2> = N + N(N-1) |f(q)|^2,     q = k (nhat - khat),
    f(q) = exp(-sum_i q_i^2 w_i^2 / 4)

-- a forward lobe of 1/e half-width ``~sqrt(2)/(k w_perp)``.  This is the elastic
component only; at saturation the inelastic part is incoherent and follows the
reabsorption channel instead.

Quick start
-----------
>>> from kamo.imaging import farfield
>>> sky = farfield.Sky(widths=cloud.widths, N=cloud.N)
>>> sky.collected_fractions(sigma_sc, NA=0.42, axis=(1, 0, 0))
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erfc

# ------------------------------------------------------------ escape probability

_ZETA_N, _ZETA_W = np.polynomial.legendre.leggauss(400)
_ZETA_N = 7.0 * _ZETA_N
_ZETA_W = 7.0 * _ZETA_W * np.exp(-_ZETA_N**2)
_ERFC_ZETA = erfc(_ZETA_N)


def _escape_probability_exact(tau):
    """T(tau) by direct quadrature.  Gauss-Legendre carrying the Gaussian weight
    explicitly; Gauss-Hermite at the order needed here overflows."""
    tau = np.asarray(tau, dtype=float)
    K = 0.5 * tau[..., None] * _ERFC_ZETA
    small = K < 1e-8
    Ks = np.where(small, 1.0, K)
    g = np.where(small, 1.0 - 0.5 * K, -np.expm1(np.maximum(-Ks, -700.0)) / Ks)
    return (g * _ZETA_W).sum(-1) / np.sqrt(np.pi)


_LOGTAU_TAB = np.linspace(-8.0, 8.0, 2001)
_T_TAB = _escape_probability_exact(10.0 ** _LOGTAU_TAB)


def escape_probability(tau):
    """Emission-point-averaged transmission out of a Gaussian cloud of chord OD ``tau``.

    Interpolated from a table in ``log tau``: ``T`` depends on emission direction
    only through that one scalar, so tabulating it once keeps every direction on
    the sky from dragging 400 quadrature nodes along.  Below ``tau = 1e-8`` the
    answer is 1 to machine precision; above 1e8 the cloud is opaque by any measure.
    """
    tau = np.asarray(tau, dtype=float)
    return np.interp(np.log10(np.clip(tau, 1e-300, None)), _LOGTAU_TAB, _T_TAB,
                     left=1.0, right=_T_TAB[-1])


# ------------------------------------------------------------------ quadratures


def sphere_quadrature(n_u: int = 200, n_phi: int = 256):
    """Gauss-Legendre in ``cos theta`` x uniform in ``phi``.  Weights sum to 4 pi."""
    u, wu = np.polynomial.legendre.leggauss(n_u)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    U, P = np.meshgrid(u, phi, indexing="ij")
    s = np.sqrt(np.clip(1 - U**2, 0, None))
    nhat = np.stack([s * np.cos(P), s * np.sin(P), U], axis=-1)
    return nhat, np.outer(wu, np.full(n_phi, 2 * np.pi / n_phi))


def cone_quadrature(cos_half: float, axis, n_a: int = 160, n_b: int = 256):
    """Same, restricted to a cone of half-angle ``arccos(cos_half)`` about ``axis``."""
    e3 = np.asarray(axis, dtype=float)
    e3 = e3 / np.linalg.norm(e3)
    seed = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(seed, e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    t, wt = np.polynomial.legendre.leggauss(n_a)
    ca = 0.5 * (1 - cos_half) * t + 0.5 * (1 + cos_half)
    wa = 0.5 * (1 - cos_half) * wt
    beta = 2 * np.pi * np.arange(n_b) / n_b
    CA, Bt = np.meshgrid(ca, beta, indexing="ij")
    sa = np.sqrt(np.clip(1 - CA**2, 0, None))
    nhat = (CA[..., None] * e3
            + sa[..., None] * (np.cos(Bt)[..., None] * e1 + np.sin(Bt)[..., None] * e2))
    return nhat, np.outer(wa, np.full(n_b, 2 * np.pi / n_b))


def dipole_pattern(nhat):
    """``(3/16 pi)(1 + cos^2 theta)``, theta from z.  Integrates to 1 over the sphere.

    The pattern of a circularly driven (sigma+-) rotating dipole in the x-y plane.
    Note that a collection axis lying IN that plane stares at the DIM lobe.
    """
    return (3 / (16 * np.pi)) * (1 + np.asarray(nhat)[..., 2] ** 2)


# ------------------------------------------------------------------------ sky


PATTERN_KEYS = ("bare", "reabs", "coh", "coh+re", "coh_only")
PATTERN_LABELS = {
    "bare": "bare dipole (no cloud)",
    "reabs": "incoherent + reabsorption",
    "coh": "coherent + incoherent",
    "coh+re": "coherent + incoherent + reabs",
    "coh_only": "coherent only (Born)",
}


class Sky:
    """Angular distribution of light scattered by a Gaussian cloud.

    Parameters
    ----------
    widths : (3,) array
        1/e density radii ``w`` (m), ordered (x, y, z).
    N : float
        Atom number.
    k : float
        Optical wavenumber (1/m).
    khat_probe : (3,) sequence
        Probe propagation direction, for the coherent form factor.
    """

    def __init__(self, widths, N: float, k: float,
                 khat_probe: Sequence[float] = (1.0, 0.0, 0.0)):
        self.w = np.asarray(widths, dtype=float)
        self.N = float(N)
        self.k = float(k)
        self.khat_probe = np.asarray(khat_probe, dtype=float)
        self.peak_density = self.N / (np.pi**1.5 * np.prod(self.w))

    def chord_optical_depth(self, nhat, sigma_sc: float):
        """Optical depth of the full chord through the cloud centre along ``nhat``."""
        m = np.sqrt(((np.asarray(nhat) ** 2) / self.w**2).sum(-1))
        return sigma_sc * self.peak_density * np.sqrt(np.pi) / m

    def form_factor_sq(self, nhat):
        """``|f(q)|^2`` for ``q = k (nhat - khat_probe)``, Gaussian density."""
        q = self.k * (np.asarray(nhat) - self.khat_probe)
        return np.exp(-((q * self.w) ** 2).sum(-1) / 2.0)

    def coherent_lobe_halfwidth_deg(self):
        """1/e half-width of the forward coherent lobe about each transverse axis."""
        return np.degrees(np.sqrt(2) / (self.k * self.w))

    def patterns(self, nhat, sigma_sc: float) -> Dict[str, np.ndarray]:
        """Unnormalized ``dP/dOmega`` per atom, for each of the four models."""
        d = dipole_pattern(nhat)
        T = escape_probability(self.chord_optical_depth(nhat, sigma_sc))
        f2 = self.form_factor_sq(nhat)
        C = 1.0 + (self.N - 1.0) * f2      # N incoherent + N(N-1)|f|^2 coherent
        return {"bare": d, "reabs": d * T, "coh": d * C, "coh+re": d * C * T,
                # The ELASTIC term alone.  This -- not 'coh' -- is the like-for-like
                # reference for a beam propagation, which carries only the coherent
                # field.  'coh' additionally dilutes it with the N incoherent
                # photons, which go out over the full dipole pattern instead.
                "coh_only": d * f2}

    def collected_fractions(self, sigma_sc: float, NA: float,
                            axis: Sequence[float] = (1.0, 0.0, 0.0),
                            n_u: int = 200, n_phi: int = 256,
                            n_a: int = 160, n_b: int = 256):
        """``(total, in_NA, fraction)`` dicts keyed by pattern.

        ``total`` is over the whole sphere, ``in_NA`` over the collection cone.
        The ratio is what a lens of this numerical aperture actually collects.
        """
        cos_half = np.sqrt(1 - float(NA) ** 2)
        n_sph, w_sph = sphere_quadrature(n_u, n_phi)
        n_cone, w_cone = cone_quadrature(cos_half, axis, n_a, n_b)
        Ps = self.patterns(n_sph, sigma_sc)
        Pc = self.patterns(n_cone, sigma_sc)
        tot = {kk: float((v * w_sph).sum()) for kk, v in Ps.items()}
        inNA = {kk: float((v * w_cone).sum()) for kk, v in Pc.items()}
        eta = {kk: inNA[kk] / tot[kk] for kk in tot}
        return tot, inNA, eta

    def sky_map(self, sigma_sc: float, n_phi: int = 361, n_u: int = 241):
        """``(phi, cos_theta, {key: map})`` on an equal-solid-angle projection.

        ``phi`` runs over [-pi, pi] so a collection cone centred on +x stays in one
        piece rather than wrapping at the seam.  Every pixel subtends the same
        ``dOmega``, so the eye reads power rather than density.
        """
        phi = np.linspace(-np.pi, np.pi, n_phi)
        u = np.linspace(-1, 1, n_u)
        U, P = np.meshgrid(u, phi, indexing="ij")
        S = np.sqrt(np.clip(1 - U**2, 0, None))
        nhat = np.stack([S * np.cos(P), S * np.sin(P), U], axis=-1)
        return phi, u, self.patterns(nhat, sigma_sc)

    def summary(self, sigma_sc: float, NA: float,
                axis: Sequence[float] = (1.0, 0.0, 0.0)) -> str:
        """Human-readable report of optical depths and collected fractions."""
        tot, inNA, eta = self.collected_fractions(sigma_sc, NA, axis)
        cos_half = np.sqrt(1 - float(NA) ** 2)
        lines = [f"Sky  N = {self.N:.0f}, w = "
                 f"({', '.join(f'{x*1e6:.3f}' for x in self.w)}) um, "
                 f"sigma_sc = {sigma_sc * 1e4:.3e} cm^2",
                 "  chord optical depth through the centre:"]
        for nm, e in zip("xyz", np.eye(3)):
            t = self.chord_optical_depth(e, sigma_sc)
            lines.append(f"    tau_{nm} = {t:8.3f}  -> e^-tau = {np.exp(-t):.3e}")
        lines.append(f"  cloud-averaged escape fraction = {tot['reabs']:.4f}")
        lines.append(f"  coherent enhancement vs incoherent = {tot['coh']:.2f}x "
                     f"(ceiling is N = {self.N:.0f})")
        hw = self.coherent_lobe_halfwidth_deg()
        lines.append(f"  forward lobe 1/e half-width: "
                     f"{hw[1]:.2f} deg (y), {hw[2]:.2f} deg (z)")
        lines.append(f"  collected into NA {NA} along {np.asarray(axis).astype(int)}:")
        for kk in PATTERN_KEYS:
            lines.append(f"    {PATTERN_LABELS[kk]:28s} {eta[kk]:9.5f}  "
                         f"({eta[kk] / eta['bare']:.2f}x bare dipole)")
        lines.append(f"    (an isotropic emitter would give {(1 - cos_half) / 2:.5f})")
        return "\n".join(lines)

    def __repr__(self):
        return (f"Sky(N={self.N:.0f}, w="
                f"({', '.join(f'{x*1e6:.3f}' for x in self.w)}) um)")
