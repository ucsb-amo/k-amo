"""Free-space resonant dipole-dipole (RDDI) coupling for a two-level transition.

Two identical two-level atoms separated by ``R`` exchange virtual photons.  In the
single-excitation subspace spanned by ``|eg>`` and ``|ge>`` the (rotating-frame)
non-Hermitian Hamiltonian is

    H/hbar = [[ -Delta,  J   ]]        L (decay) = [[ Gamma,    Gamma_12 ]]
             [[  J,     -Delta ]]                  [[ Gamma_12, Gamma    ]]

whose eigenstates are the symmetric / antisymmetric pair states

    |+> = (|eg> + |ge>)/sqrt(2)   ->   shift  +J,  decay  Gamma + Gamma_12
    |-> = (|eg> - |ge>)/sqrt(2)   ->   shift  -J,  decay  Gamma - Gamma_12

Both J and Gamma_12 come from a single contraction of the free-space dyadic Green's
function with the (possibly complex) transition-dipole unit vector.  We use the
dimensionless dyadic

    Gt(xi, rhat) = (3/2) (exp(i xi)/xi) [ (I - rhat rhat)
                                          + (i/xi - 1/xi^2) (I - 3 rhat rhat) ]

with xi = k R, so that

    J        = -(Gamma/2) * Re[ dhat* . Gt . dhat ]      (rad/s)
    Gamma_12 =   Gamma    * Im[ dhat* . Gt . dhat ]      (rad/s)

Limits (both asserted in kamo.dipole_dipole.validation):

* xi -> 0:  J -> (3 Gamma/4) (1 - 3p) / xi^3  and  Gamma_12 -> Gamma, so
  Gamma_+ -> 2 Gamma (superradiant) and Gamma_- -> 0 (subradiant), where
  p = |dhat . rhat|^2.
* For a circular (sigma) dipole about the quantisation axis, p = sin^2(theta)/2, so
  the near-field angular factor is (1 - 1.5 sin^2 theta), which changes sign at the
  magic angle theta = arcsin(sqrt(2/3)) = 54.7356 deg.

Retardation note: for K-39 D2 at detunings of 0.3-10 linewidths the Condon radius
sits at k R ~ 0.4-1.4, so the pure 1/R^3 near-field term is wrong by O((kR)^2), i.e.
20-80%.  Use the full expressions here, not near_field_coupling, for anything
quantitative.
"""

from __future__ import annotations

import numpy as np

# theta at which (1 - 1.5 sin^2 theta) changes sign, for a circular dipole
MAGIC_ANGLE_RAD = float(np.arcsin(np.sqrt(2.0 / 3.0)))
MAGIC_ANGLE_DEG = float(np.degrees(MAGIC_ANGLE_RAD))


def spherical_unit_vector(q: int) -> np.ndarray:
    """Spherical basis vector e_q in Cartesian components (quantisation axis z).

    e_+1 = -(x + i y)/sqrt(2),  e_0 = z,  e_-1 = (x - i y)/sqrt(2).
    """
    if q == 1:
        return np.array([-1.0, -1.0j, 0.0]) / np.sqrt(2.0)
    if q == 0:
        return np.array([0.0, 0.0, 1.0], dtype=complex)
    if q == -1:
        return np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0)
    raise ValueError("q must be -1, 0 or +1 (got {!r}).".format(q))


def dipole_projection(d_hat, r_hat) -> np.ndarray:
    """p = |dhat . rhat|^2, the only angular dependence of the contraction.

    Parameters
    ----------
    d_hat : (..., 3) complex
        Transition-dipole unit vector(s); may be complex (circular polarisation).
    r_hat : (..., 3) float
        Unit vector(s) along the interatomic axis.

    Returns
    -------
    ndarray : real, non-negative, broadcast over the leading axes.
    """
    d_hat = np.asarray(d_hat)
    r_hat = np.asarray(r_hat)
    proj = np.sum(d_hat * r_hat, axis=-1)
    return np.abs(proj) ** 2


def green_scalar(xi, p):
    """Contraction dhat* . Gt . dhat as a complex number.

    Parameters
    ----------
    xi : array_like
        k R, dimensionless separation.  Must be > 0.
    p : array_like
        |dhat . rhat|^2 (see dipole_projection).

    Returns
    -------
    complex ndarray, broadcast over xi and p.
    """
    xi = np.asarray(xi, dtype=float)
    p = np.asarray(p, dtype=float)
    A = 1.0 - p          # coefficient of (I - rhat rhat)
    B = 1.0 - 3.0 * p    # coefficient of (I - 3 rhat rhat)
    inner = A + B * (1.0j / xi - 1.0 / xi ** 2)
    return 1.5 * np.exp(1.0j * xi) / xi * inner


def dd_coupling_scalar(xi, p, Gamma):
    """(J, Gamma_12) in rad/s from the dimensionless separation and angle factor.

    Parameters
    ----------
    xi : array_like
        k R.
    p : array_like
        |dhat . rhat|^2.
    Gamma : float
        Single-atom linewidth in rad/s (that is 1/tau, NOT Gamma/2pi).

    Returns
    -------
    (J, Gamma_12) : tuple of ndarray, both rad/s.
    """
    g = green_scalar(xi, p)
    J = -0.5 * Gamma * np.real(g)
    Gamma_12 = Gamma * np.imag(g)
    return J, Gamma_12


def near_field_coupling(xi, p, Gamma):
    """Near-field (kR << 1) limit: J = (3 Gamma/4)(1 - 3p)/xi^3, Gamma_12 = Gamma.

    Provided for cross-checking only -- see the retardation note in the module
    docstring before using this quantitatively.
    """
    xi = np.asarray(xi, dtype=float)
    p = np.asarray(p, dtype=float)
    J = 0.75 * Gamma * (1.0 - 3.0 * p) / xi ** 3
    Gamma_12 = np.full(np.broadcast(xi, p).shape, float(Gamma))
    return J, Gamma_12


def green_tensor(k: float, R_vec) -> np.ndarray:
    """Full dimensionless dyadic Gt of shape (..., 3, 3).

    Parameters
    ----------
    k : float
        Wavenumber 2 pi / lambda (1/m).
    R_vec : (..., 3) array_like
        Interatomic separation vector(s) in metres.

    Returns
    -------
    (..., 3, 3) complex ndarray

    Notes
    -----
    Kept for validation and for future multi-level extensions.  For a *closed*
    two-level transition every atom shares one dipole orientation, so the many-body
    problem collapses onto the scalar contraction green_scalar and the full tensor
    is not needed.
    """
    R_vec = np.asarray(R_vec, dtype=float)
    R = np.linalg.norm(R_vec, axis=-1)
    if np.any(R <= 0):
        raise ValueError("R_vec must have non-zero length.")
    r_hat = R_vec / R[..., None]
    xi = k * R

    eye = np.eye(3)
    outer = r_hat[..., :, None] * r_hat[..., None, :]
    term_a = eye - outer
    term_b = eye - 3.0 * outer

    pref = np.asarray(1.5 * np.exp(1.0j * xi) / xi)
    coef = np.asarray(1.0j / xi - 1.0 / xi ** 2)
    return pref[..., None, None] * (term_a + coef[..., None, None] * term_b)


def dd_coupling(k: float, R_vec, d_hat, Gamma):
    """(J, Gamma_12) in rad/s from the full tensor.

    Equivalent to dd_coupling_scalar with p = dipole_projection(d_hat, r_hat).
    """
    G = green_tensor(k, R_vec)
    d_hat = np.asarray(d_hat)
    g = np.einsum("...i,...ij,...j->...", np.conj(d_hat), G, d_hat)
    return -0.5 * Gamma * np.real(g), Gamma * np.imag(g)


def pair_branches(xi, p, Gamma):
    """Symmetric / antisymmetric pair-state shifts and decay rates.

    Returns
    -------
    dict with keys V_sym, V_anti (rad/s, shift relative to the bare one-photon
    resonance) and Gamma_sym, Gamma_anti (rad/s, floored to stay positive).
    """
    J, G12 = dd_coupling_scalar(xi, p, Gamma)
    floor = 1e-6 * Gamma
    return {
        "V_sym": J,
        "V_anti": -J,
        "Gamma_sym": np.maximum(Gamma + G12, floor),
        "Gamma_anti": np.maximum(Gamma - G12, floor),
    }
