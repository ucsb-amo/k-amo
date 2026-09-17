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

Three variants; the ablation between them is the point of the package:

``'full'``   the expressions above.
``'far'``    only the ``1/x`` terms: ``Gamma = (3/2) A sin(x)/x``,
             ``J = -(3/4) A cos(x)/x``.  Propagation and multiple scattering with
             the near field switched off.  Note that this truncated ``Gamma``
             matrix is NOT positive semidefinite for close pairs (it exceeds 1 as
             ``x -> 0``), so sanity check S2 is expected to fail for it: it is an
             ablation, not a passive medium.
``'rg'``     the far-field kernel with each atom's bare detuning replaced by its
             strong-disorder-RG renormalized resonance (:mod:`kamo.dd_solver.rg`).

Limits: as ``x -> 0``, ``J -> (3/4) B / x^3`` (divergent) while ``Gamma_ij -> 1``
(bounded).  That asymmetry -- the near-field shift diverges, the collective decay
does not -- is the central structural fact of the problem.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from kamo.dipole_dipole.green_tensor import (dipole_projection, green_scalar,
                                             green_tensor as dyadic_green_tensor,
                                             spherical_unit_vector)

VARIANTS = ("full", "far", "rg", "independent")

__all__ = ["VARIANTS", "scalar_couplings", "near_field_hamiltonian", "near_field_pairs",
           "gamma_matrix", "kernel_pair", "dyadic_green_tensor", "spherical_unit_vector",
           "angular_factors", "near_field_cutoff_radius"]


def angular_factors(rhat, e_hat) -> Tuple[np.ndarray, np.ndarray]:
    """``(A, B) = (1 - c, 1 - 3c)`` with ``c = |e_hat . r_hat|^2``."""
    c = dipole_projection(e_hat, rhat)
    return 1.0 - c, 1.0 - 3.0 * c


def kernel_pair(x, c, variant: str = "full"):
    """``(J, Gamma_offdiag)`` for separations ``x = k r`` and projections ``c``.

    Elementwise; ``x`` must be > 0.  ``variant`` is ``'full'`` or ``'far'``
    (``'rg'`` uses the far kernel).
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    if variant in ("full",):
        g = green_scalar(x, c)
        return -0.5 * np.real(g), np.imag(g)
    if variant in ("far", "rg"):
        A = 1.0 - c
        return -0.75 * A * np.cos(x) / x, 1.5 * A * np.sin(x) / x
    if variant == "independent":
        z = np.zeros(np.broadcast(x, c).shape)
        return z, z.copy()
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


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


def near_field_cutoff_radius(cutoff: float, k: float) -> float:
    """Largest separation at which ``|H^near|`` can reach ``cutoff`` (``|B| <= 1``)."""
    return (0.75 / cutoff) ** (1 / 3) / k


def near_field_pairs(positions, k: float, e_hat, cutoff: float):
    """Pairs ``(i, j, H_ij)`` with ``|H^near_ij| >= cutoff`` via a KD-tree.

    Returns ``(i, j, H, Gamma_ij, r)`` arrays with ``i < j``; ``Gamma_ij`` is the
    full collective decay, kept for the tracked RG variant.
    """
    pos = np.asarray(positions, dtype=float)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(near_field_cutoff_radius(cutoff, k), output_type="ndarray")
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
