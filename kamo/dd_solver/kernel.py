"""The dipole-dipole kernel: exchange shifts ``J_ij`` and collective decay ``Gamma_ij``.

Everything is the free-space dyadic Green tensor of
:mod:`kamo.dipole_dipole.green_tensor`, projected on the driven (sigma-) dipole.
With ``x = k r``, ``c = |conj(e_hat) . r_hat|^2 = sin^2(theta_z) / 2``,
``A = 1 - c`` and ``B = 1 - 3c``, in units of ``Gamma``,

    Gamma_ij = (3/2) [ A sin(x)/x + B (cos(x)/x^2 - sin(x)/x^3) ]
    J_ij     = -(3/4) [ A cos(x)/x - B (sin(x)/x^2 + cos(x)/x^3) ]

Diagonal: ``Gamma_ii = 1`` (single-atom linewidth), ``J_ii = 0`` (the Lamb shift
is absorbed in ``omega_0``).  Both come from one contraction
``g = conj(e_hat) . Gt . e_hat = -2 J + i Gamma`` of the dimensionless dyadic.

The decay matrix has NO near-field part (corrected 2026-09-17)
-------------------------------------------------------------
``Gamma_ij = Im(conj(e) . Gt . e)`` is identically the overlap of the two atoms'
far-field radiation patterns,

    Gamma_ij = (3 / 8 pi) Int dOmega_n [1 - |n . e|^2] exp(i x n . rhat),

so it is a band-limited, purely radiative quantity: the individually divergent
``1/x^2`` and ``1/x^3`` pieces of ``Im g`` cancel exactly and
``Gamma_ij -> 1 - (2 - c) x^2 / 10`` as ``x -> 0``.  By Bochner's theorem the
kernel is positive semidefinite, which is what makes ``Gamma = 1 + G`` a valid
Lindblad dissipator.  **Only the REAL part ``J`` has a near field.**  Truncating
``Im g`` is therefore not an ablation of the near field; it breaks positivity,
and a solver built on it has linear gain modes (``Im eig(M) < 0``) whose
"steady state" is not the long-time limit of any dynamics.  Before 2026-09-17
the ``'far'`` variant did exactly that: at the operating point 141 of 500
eigenvalues of its ``Gamma`` were negative (min ``-1.51``) and 97 of 500 modes
had gain.  Every variant now carries the exact ``Gamma``.

Variants; the ablation between them is the point of the package:

``'full'``    the expressions above.
``'far'``     retarded propagation only in the coherent coupling,
              ``J = -(3/4) A cos(x)/x`` (the ``1/x`` term), with the EXACT
              ``Gamma``.  "What a wave propagating through the sample does",
              with the ``1/x^2`` and ``1/x^3`` exchange terms switched off.
``'nonear'``  the literature ablation (Andreoli, Gullans, High, Browaeys and
              Chang, Phys. Rev. X 11, 011026 (2021), Eqs. A.2-A.3): the exact
              kernel minus the STATIC near field ``H_near = (3/4) B / x^3``,
              which is purely real.  This is the correct "no near field"
              comparison: it keeps retardation, the ``1/x^2`` term and the exact
              ``Gamma``, and removes only the term that diverges.
``'rg'``      the ``'nonear'`` residual kernel with each atom's bare detuning
              replaced by its strong-disorder-RG renormalized resonance
              (:mod:`kamo.dd_solver.rg`), as in that paper's Eq. A.3.

At this density the two ablations are NOT interchangeable.  With
``k r_nn = 1.23`` there is no scale separation: for a polar pair at ``x = 1.23``
the three terms of ``J`` are ``-0.204`` (``1/x``), ``+0.467`` (``1/x^2``) and
``+0.135`` (``1/x^3``), so ``'far'`` removes a term three times larger than the
one it is meant to isolate.  Quote which ablation a near-field number refers to.

Limits: as ``x -> 0``, ``J -> (3/4) B / x^3`` (divergent) while ``Gamma_ij -> 1``
(bounded).  That asymmetry -- the exchange shift diverges, the collective decay
does not -- is the central structural fact of the problem.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from kamo.dipole_dipole.green_tensor import (dipole_projection, green_scalar,
                                             green_tensor as dyadic_green_tensor,
                                             spherical_unit_vector)

VARIANTS = ("full", "far", "nonear", "rg", "independent")

#: Variants whose ``Gamma`` is the exact (positive semidefinite) radiative one.
#: Every variant is now in this set; the entry documents the 2026-09-17 change.
EXACT_GAMMA_VARIANTS = ("full", "far", "nonear", "rg", "independent")

__all__ = ["VARIANTS", "EXACT_GAMMA_VARIANTS", "scalar_couplings",
           "scalar_couplings_multi", "near_field_hamiltonian",
           "near_field_pairs", "gamma_matrix", "kernel_pair", "dyadic_green_tensor",
           "spherical_unit_vector", "angular_factors", "near_field_cutoff_radius",
           "max_near_field_factor", "kernel_for_variant"]


def angular_factors(rhat, e_hat) -> Tuple[np.ndarray, np.ndarray]:
    """``(A, B) = (1 - c, 1 - 3c)`` with ``c = |e_hat . r_hat|^2``."""
    c = dipole_projection(e_hat, rhat)
    return 1.0 - c, 1.0 - 3.0 * c


def kernel_for_variant(variant: str) -> str:
    """The kernel a solver variant is built from (``'rg'`` reuses ``'nonear'``)."""
    return "nonear" if variant == "rg" else variant


def kernel_pair(x, c, variant: str = "full"):
    """``(J, Gamma_offdiag)`` for separations ``x = k r`` and projections ``c``.

    Elementwise; ``x`` must be > 0.  ``Gamma`` is the exact radiative overlap for
    every variant (see the module docstring): the variants differ only in ``J``.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    variant = kernel_for_variant(variant)
    if variant == "independent":
        z = np.zeros(np.broadcast(x, c).shape)
        return z, z.copy()
    if variant not in ("full", "far", "nonear"):
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    g = green_scalar(x, c)
    gamma = np.imag(g)                      # exact for every variant
    if variant == "full":
        return -0.5 * np.real(g), gamma
    if variant == "far":
        return -0.75 * (1.0 - c) * np.cos(x) / x, gamma
    # 'nonear': remove only the static (real, divergent) near field.
    return -0.5 * np.real(g) - 0.75 * (1.0 - 3.0 * c) / x ** 3, gamma


def scalar_couplings(positions, k: float, e_hat, variant: str = "full",
                     block: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """``(J, G)``: the exchange and collective-decay matrices, zero on the diagonal.

    Built in row blocks so the ``(N, N, 3)`` separation array is never
    materialised; peak temporary memory is ``block * N * 3 * 8`` bytes.
    """
    pos = np.asarray(positions, dtype=float)
    N = pos.shape[0]
    e_hat = np.asarray(e_hat, dtype=complex)
    J = np.zeros((N, N))
    G = np.zeros((N, N))
    if variant == "independent":
        return J, G
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        d = pos[i0:i1, None, :] - pos[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        # exclude the diagonal and any exactly coincident pair
        rows = np.arange(i0, i1)
        r[rows - i0, rows] = np.inf
        with np.errstate(invalid="ignore", divide="ignore"):
            rhat = d / r[..., None]
        rhat = np.nan_to_num(rhat)
        c = dipole_projection(e_hat, rhat)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            Jb, Gb = kernel_pair(k * r, c, variant)
        Jb = np.nan_to_num(Jb, nan=0.0, posinf=0.0, neginf=0.0)
        Gb = np.nan_to_num(Gb, nan=0.0, posinf=0.0, neginf=0.0)
        J[i0:i1] = Jb
        G[i0:i1] = Gb
    return J, G


def scalar_couplings_multi(positions, k: float, e_hat, variants=("full", "far"),
                           block: int = 512):
    """``{variant: (J, G)}`` for several variants from ONE geometry pass.

    The separations, their norms, the unit vectors and the angular factor ``c``
    do not depend on the variant, and neither does ``Gamma``; only ``J`` does.
    Building them once cuts a four-variant configuration from 202 ms to 71 ms at
    N = 500 (measured 2026-09-17), which matters because the kernel build is 84 %
    of a solve.  ``'rg'`` shares the ``'nonear'`` matrices exactly.
    """
    pos = np.asarray(positions, dtype=float)
    N = pos.shape[0]
    e_hat = np.asarray(e_hat, dtype=complex)
    wanted = {kernel_for_variant(v) for v in variants}
    out = {v: (np.zeros((N, N)), np.zeros((N, N))) for v in wanted}
    if wanted <= {"independent"}:
        return {v: out[kernel_for_variant(v)] for v in variants}
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        d = pos[i0:i1, None, :] - pos[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        rows = np.arange(i0, i1)
        r[rows - i0, rows] = np.inf
        with np.errstate(invalid="ignore", divide="ignore"):
            rhat = np.nan_to_num(d / r[..., None])
        c = dipole_projection(e_hat, rhat)
        x = k * r
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            g = green_scalar(x, c)
            gamma = np.nan_to_num(np.imag(g), nan=0.0, posinf=0.0, neginf=0.0)
            reg = np.nan_to_num(-0.5 * np.real(g), nan=0.0, posinf=0.0, neginf=0.0)
            for v in wanted:
                if v == "independent":
                    continue
                if v == "full":
                    Jb = reg
                elif v == "far":
                    Jb = np.nan_to_num(-0.75 * (1.0 - c) * np.cos(x) / x,
                                       nan=0.0, posinf=0.0, neginf=0.0)
                else:                                    # 'nonear'
                    Jb = reg - np.nan_to_num(0.75 * (1.0 - 3.0 * c) / x ** 3,
                                             nan=0.0, posinf=0.0, neginf=0.0)
                out[v][0][i0:i1] = Jb
                out[v][1][i0:i1] = gamma
    return {v: out[kernel_for_variant(v)] for v in variants}


def gamma_matrix(G: np.ndarray) -> np.ndarray:
    """The full decay matrix ``Gamma = 1 + G`` (unit diagonal)."""
    return G + np.eye(G.shape[0])


def near_field_hamiltonian(positions, k: float, e_hat, block: int = 512) -> np.ndarray:
    """``H^near_ij = (3/4) B / (k r)^3``, the 1/r^3 part of ``J``; zero diagonal."""
    pos = np.asarray(positions, dtype=float)
    N = pos.shape[0]
    H = np.zeros((N, N))
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        d = pos[i0:i1, None, :] - pos[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        rows = np.arange(i0, i1)
        r[rows - i0, rows] = np.inf
        with np.errstate(invalid="ignore", divide="ignore"):
            rhat = np.nan_to_num(d / r[..., None])
        _, B = angular_factors(rhat, e_hat)
        with np.errstate(divide="ignore"):
            H[i0:i1] = np.nan_to_num(0.75 * B / (k * r) ** 3, posinf=0.0, neginf=0.0)
    return H


def max_near_field_factor(e_hat=None) -> float:
    """``max |B| = max |1 - 3 c|`` over directions, for the given dipole vector.

    ``c = |e_hat . rhat|^2`` runs over ``[0, c_max]`` with ``c_max`` the largest
    eigenvalue of ``Re(e e^dagger)``: 1/2 for a circular (sigma) dipole, giving
    ``|B| <= 1``, but 1 for a linear (pi) dipole, giving ``|B| <= 2``.  Passing
    ``None`` returns the conservative linear-dipole bound.
    """
    if e_hat is None:
        return 2.0
    e = np.asarray(e_hat, dtype=complex)
    c_max = float(np.max(np.linalg.eigvalsh(np.real(np.outer(e, np.conj(e))))))
    return float(max(1.0, 3.0 * c_max - 1.0))


def near_field_cutoff_radius(cutoff: float, k: float, e_hat=None) -> float:
    """Largest separation at which ``|H^near| = (3/4)|B|/x^3`` can reach ``cutoff``.

    The bound needs ``max |B|`` for the actual dipole (:func:`max_near_field_factor`):
    1 for a sigma dipole but 2 for a linear one, so a radius computed with
    ``|B| <= 1`` silently misses ``2^{1/3}``-worth of pairs for ``q = 0``.
    """
    return (0.75 * max_near_field_factor(e_hat) / cutoff) ** (1 / 3) / k


def near_field_pairs(positions, k: float, e_hat, cutoff: float):
    """Pairs ``(i, j, H_ij)`` with ``|H^near_ij| >= cutoff`` via a KD-tree.

    Returns ``(i, j, H, Gamma_ij, r)`` arrays with ``i < j``; ``Gamma_ij`` is the
    full collective decay, kept for the tracked RG variant.
    """
    pos = np.asarray(positions, dtype=float)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(near_field_cutoff_radius(cutoff, k, e_hat), output_type="ndarray")
    if pairs.size == 0:
        z = np.zeros(0)
        return np.zeros(0, int), np.zeros(0, int), z, z, z
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[j] - pos[i]
    r = np.linalg.norm(d, axis=1)
    rhat = d / r[:, None]
    c = dipole_projection(e_hat, rhat)
    x = k * r
    H = 0.75 * (1.0 - 3.0 * c) / x ** 3
    keep = np.abs(H) >= cutoff
    _, Gij = kernel_pair(x[keep], c[keep], "full")
    return i[keep], j[keep], H[keep], Gij, r[keep]
