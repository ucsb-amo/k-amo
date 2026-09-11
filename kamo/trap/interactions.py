"""Scattering length and mean-field constants for :mod:`kamo.trap`'s solvers.

The only module in kamo.trap that touches :mod:`kamo.scattering`, so the rest of
the package imports without it.  Scattering lengths are carried in **metres**
(the convention of :class:`kamo.BEC_properties.variational.GaussianVariationalCloud`);
:mod:`kamo.scattering` reports Bohr radii and this is the one conversion point.

``Potassium39.get_scattering_length`` is never used: it reads a hard-coded
network path.  :class:`kamo.scattering.ScatteringModel` tabulates only the
intra-state channels ``(1,-1)``, ``(1,0)`` and ``(1,1)``; anything else needs an
explicit ``a_scattering``.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np

import kamo.constants as kc


@lru_cache(maxsize=4)
def _model(B_max: float):
    from kamo.scattering import ScatteringModel
    return ScatteringModel(B_max=B_max)


def scattering_length_m(state, B_gauss: float, a_scattering: Optional[float] = None,
                        B_max: float = 1000.0) -> float:
    """s-wave scattering length (m) for two atoms in ``state`` at ``B_gauss``.

    ``a_scattering`` (m), when given, is returned as is.  Otherwise the ground
    state ``(4, 0, 1/2, F, mF)`` is looked up in :class:`kamo.scattering.ScatteringModel`;
    a lossy channel (complex ``a``) warns and uses ``Re(a)``.
    """
    if a_scattering is not None:
        a = float(a_scattering)
        if not np.isfinite(a):
            raise ValueError(f"a_scattering must be finite (m); got {a_scattering}")
        return a
    if state is None:
        raise ValueError("no state to look the scattering length up for: pass "
                         "a_scattering (m).")
    try:
        n, l, j, F, mF = state
    except (TypeError, ValueError):
        raise ValueError(f"state must be (n, l, j, F, mF); got {state!r}") from None
    if (int(n), int(l), float(j)) != (4, 0, 0.5):
        raise ValueError("kamo.scattering covers the 4S1/2 ground state only; pass "
                         "a_scattering (m).")
    B = float(B_gauss)
    try:
        a_bohr = complex(_model(max(float(B_max), 1.2 * B)).intra((int(F), int(mF)), B))
    except KeyError as err:
        raise ValueError(f"kamo.scattering has no scattering length for |F={F}, mF={mF}> "
                         f"pairs ({err}); pass a_scattering (m).") from None
    if abs(a_bohr.imag) > 1e-6 * max(abs(a_bohr.real), 1.0):
        warnings.warn(f"|F={F}, mF={mF}> at {B} G is a lossy channel (a = {a_bohr:.4g} a0); "
                      "the solvers use Re(a).", UserWarning, stacklevel=2)
    return a_bohr.real * kc.a0


def trap_scattering_length(trap, a_scattering: Optional[float] = None) -> float:
    """:func:`scattering_length_m` for a trap's state and field (a HarmonicTrap
    has neither, so it needs ``a_scattering``)."""
    return scattering_length_m(getattr(trap, "state", None), getattr(trap, "B_gauss", 0.0),
                               a_scattering)


def coupling_g(a_scattering: float, mass: float) -> float:
    """Contact coupling ``g = 4 pi hbar^2 a / m`` (J m^3)."""
    return 4.0 * np.pi * kc.hbar ** 2 * float(a_scattering) / float(mass)


def healing_length(density: float, a_scattering: float) -> float:
    """``xi = 1 / sqrt(8 pi n a)`` (m); NaN for ``a <= 0`` or ``n <= 0``."""
    if a_scattering <= 0 or density <= 0:
        return float("nan")
    return float(1.0 / np.sqrt(8.0 * np.pi * density * a_scattering))
