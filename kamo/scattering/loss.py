"""Two-body inelastic loss from a complex scattering length.

For a channel with complex scattering length ``a = alpha - i*beta`` (``beta >= 0``
= loss), the threshold two-body loss-rate coefficient is

    K2 = (2 h / mu) * beta = (4 pi hbar / mu) * beta

(Hutson, New J. Phys. 9, 152 (2007)), with ``mu = m_K/2``.  ``dn/dt = -K2 n^2``
for distinguishable partners; multiply by 2 for two atoms in the *same* state
(identical-particle bosonic enhancement) via ``indistinguishable=True``.
"""

from __future__ import annotations

import numpy as np

from kamo import constants as c
from . import units as u


def k2_from_scattering_length(a_a0, indistinguishable: bool = False):
    """Two-body loss coefficient K2 (m^3/s) from complex ``a`` (a0).

    Parameters
    ----------
    a_a0 : complex or array-like
        Scattering length in Bohr radii, ``alpha - i*beta`` with ``beta >= 0``.
    indistinguishable : bool
        Multiply by 2 for two atoms in the same internal state.

    Returns
    -------
    float or ndarray : K2 in m^3/s (0 where Im(a) == 0).
    """
    a = np.asarray(a_a0, dtype=complex)
    beta_m = np.maximum(-a.imag, 0.0) * u.A0_M       # loss part, metres
    mu = u.reduced_mass_kg()
    K2 = (2.0 * c.h / mu) * beta_m
    if indistinguishable:
        K2 = 2.0 * K2
    return float(K2) if a.ndim == 0 else K2
