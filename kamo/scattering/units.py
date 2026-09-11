"""Unit helpers for the scattering subpackage.

Scattering lengths are carried in **Bohr radii (a0)**; energies in **Hz**
(consistent with :mod:`kamo.hamiltonian`, which reports every energy / h).
"""

from __future__ import annotations

import numpy as np

from kamo import constants as c

A0_M = c.a0                      # Bohr radius [m]
HARTREE_HZ = 6.579683920502e15   # 1 Hartree / h [Hz]  (CODATA)


def a0_to_m(a_a0):
    """Bohr radii -> metres."""
    return np.asarray(a_a0) * A0_M


def m_to_a0(a_m):
    """metres -> Bohr radii."""
    return np.asarray(a_m) / A0_M


def reduced_mass_kg():
    """Two-body reduced mass of a K39 pair, ``m_K/2`` [kg]."""
    return c.m_K / 2.0


def vdw_length_a0(C6_au: float) -> float:
    """van der Waals length ``beta6 = (2 mu C6 / hbar^2)^(1/4)`` in a0.

    ``mu = m_K/2``.  Returned in Bohr radii.  ``R_vdW = beta6 / 2``.
    """
    mu = reduced_mass_kg()
    # C6 in SI: E_h * a0^6
    C6_si = C6_au * (HARTREE_HZ * c.h) * (A0_M ** 6)
    beta6_m = (2.0 * mu * C6_si / c.hbar ** 2) ** 0.25
    return float(beta6_m / A0_M)


def mean_scattering_length_a0(C6_au: float) -> float:
    """Gribakin-Flambaum mean scattering length ``abar`` in a0.

    ``abar = [2*pi / Gamma(1/4)^2] * beta6 = 0.4779888... * beta6``.
    """
    from math import gamma, pi
    beta6 = vdw_length_a0(C6_au)
    return (2.0 * pi / gamma(0.25) ** 2) * beta6
