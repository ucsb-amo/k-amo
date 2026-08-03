"""Model singlet/triplet Born-Oppenheimer potentials for K39, tuned to a_S/a_T.

Full coupled-channels needs scalar singlet (X^1Sigma_g+) and triplet
(a^3Sigma_u+) potentials.  We do not vendor the Tiemann/Falke tabulated
potentials; instead we use Lennard-Jones (6,12) model potentials

    V(r) = -C6/r^6 + C12/r^12                         (atomic units)

with the **correct** van der Waals tail (literature C6) and the short-range
wall ``C12`` tuned per spin so the single-channel s-wave scattering length
equals the literature ``a_S`` / ``a_T``.  Near-threshold scattering depends only
on the C6 tail and the scattering length (van der Waals / QDT universality), so
this reproduces the correct near-threshold bound states -> correct Feshbach
resonance positions, independent of the (unphysical) deep-well details.

All quantities are in **atomic units** (Bohr, Hartree, hbar = m_e = 1).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from kamo import constants as c
from .data import k39_params as kp

# reduced mass of a K39 pair in atomic units (m_e = 1)
MU_AU = (c.m_K / 2.0) / c.m_e
C6_AU = kp.C6_AU


def v_lj(r, C6: float, C12: float):
    """Lennard-Jones (6,12) potential (Hartree) at radius ``r`` (a0)."""
    r6 = r ** 6
    return -C6 / r6 + C12 / (r6 * r6)


def vdw_length_au(C6: float = C6_AU, mu: float = MU_AU) -> float:
    """van der Waals length beta6 = (2 mu C6)^(1/4) in a0."""
    return (2.0 * mu * C6) ** 0.25


def _numerov_zero_energy(C6: float, C12: float, mu: float,
                         r_in: float, r_out: float, h: float):
    """Integrate u'' = 2 mu V(r) u outward at E=0; return (r, u) arrays.

    ``u = r * psi`` with u(r_in) = 0 (deep in the repulsive wall).  At large r
    (V -> 0) u becomes linear, u ~ (r - a).
    """
    n = int(round((r_out - r_in) / h)) + 1
    r = r_in + h * np.arange(n)
    f = 2.0 * mu * v_lj(r, C6, C12)          # u'' = f u
    w = 1.0 - (h * h / 12.0) * f             # Numerov auxiliary
    u = np.zeros(n)
    u[0] = 0.0
    u[1] = 1e-12
    h2 = h * h
    for i in range(1, n - 1):
        u[i + 1] = (2.0 * u[i] * (1.0 + 5.0 * h2 / 12.0 * f[i])
                    - u[i - 1] * w[i - 1]) / w[i + 1]
        # rescale to avoid overflow through the deep well
        if abs(u[i + 1]) > 1e100:
            u[:i + 2] /= 1e100
    return r, u


def scattering_length_1ch(C6: float, C12: float, mu: float = MU_AU,
                          r_in: float = 12.0, r_out: float = 4000.0,
                          h: float = 0.02, fit_from: float = 1500.0) -> float:
    """Zero-energy s-wave scattering length (a0) of ``V_LJ(C6, C12)``.

    Extracted from the asymptotic linear form ``u ~ c (r - a)`` fitted over
    ``[fit_from, r_out]`` (where V is negligible).
    """
    r, u = _numerov_zero_energy(C6, C12, mu, r_in, r_out, h)
    m = r >= fit_from
    rr, uu = r[m], u[m]
    # linear fit u = c*r + d  ->  a = -d/c
    A = np.vstack([rr, np.ones_like(rr)]).T
    cc, dd = np.linalg.lstsq(A, uu, rcond=None)[0]
    return float(-dd / cc)


@lru_cache(maxsize=None)
def tune_C12(a_target: float, C6: float = C6_AU, mu: float = MU_AU,
             depth_rank: int = 0, n_scan: int = 400,
             C12_lo: float = 2.5e10, C12_hi: float = 3.2e12) -> Tuple[float, float]:
    """Find ``C12`` so the single-channel scattering length equals ``a_target``.

    Collects every clean (pole-free) bracket of ``a_target`` across
    ``[C12_lo, C12_hi]`` and returns the one selected by ``depth_rank``:
    0 = shallowest well (largest C12, fewest bound states), increasing rank =
    deeper well (more bound states, closer to van der Waals universality).
    Returns ``(C12, a_achieved)``.
    """
    from scipy.optimize import brentq
    C12s = np.linspace(C12_lo, C12_hi, n_scan)
    avals = np.array([scattering_length_1ch(C6, x, mu) for x in C12s])
    brackets = []
    for i in range(len(C12s) - 1):
        if ((avals[i] - a_target) * (avals[i + 1] - a_target) < 0
                and abs(avals[i + 1] - avals[i]) < 400):
            brackets.append((C12s[i], C12s[i + 1]))
    if not brackets:
        raise RuntimeError(f"tune_C12 found no bracket for a={a_target} in "
                           f"[{C12_lo:.1e},{C12_hi:.1e}].")
    # rank 0 = shallowest = largest C12 = last bracket
    brackets = sorted(brackets, key=lambda b: b[0], reverse=True)
    if depth_rank >= len(brackets):
        raise RuntimeError(f"depth_rank={depth_rank} out of range "
                           f"({len(brackets)} brackets found for a={a_target}).")
    lo, hi = brackets[depth_rank]
    C12f = brentq(lambda x: scattering_length_1ch(C6, x, mu) - a_target,
                  lo, hi, xtol=1e6, rtol=1e-10, maxiter=60)
    return float(C12f), scattering_length_1ch(C6, C12f, mu)
