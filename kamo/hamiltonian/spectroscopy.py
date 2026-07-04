"""Spectroscopic lookup utilities for K-39 multi-level structure.

.. deprecated::
    The standalone functions in this module are kept for backward compatibility.
    Prefer the equivalent **methods** on :class:`~.diagonalize.MagneticSweepResult`
    and :class:`~.diagonalize.LaserSweepResult`::

        res_B = model.magnetic_sweep(B_max=600.0)
        B     = res_B.field_from_splitting(sa, sb, f_hz)
        df_B  = res_B.transition_frequency_shift(sa, sb, at=100.0)

        res_L = model.laser_sweep(beam, model="stark")
        I     = res_L.intensity_from_splitting_shift(sa, sb, df_hz)
        Gamma = res_L.scattering_rate(state, I)
        pairs = res_L.dominant_couplings(n_top=10)
        df_L  = res_L.transition_frequency_shift(sa, sb, at=I)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from kamo import GaussianBeam
from .model import AtomicStructure


def field_from_splitting(
    model: AtomicStructure,
    state_a: tuple,
    state_b: tuple,
    f_measured_hz: float,
    B_max: float = 600.0,
    dB: float = 0.1,
    diamagnetic: bool = False,
    branch: int = 0,
) -> float:
    """Return B (G) where |E_b âˆ’ E_a| equals *f_measured_hz*.

    Wrapper around :meth:`MagneticSweepResult.field_from_splitting`.
    Prefer calling that method on an existing sweep to avoid recomputing.
    """
    return (
        model.magnetic_sweep(B_max=B_max, dB=dB, diamagnetic=diamagnetic)
        .field_from_splitting(state_a, state_b, f_measured_hz, branch=branch)
    )


def intensity_from_splitting_shift(
    model: AtomicStructure,
    state_a: tuple,
    state_b: tuple,
    beam: GaussianBeam,
    df_measured_hz: float,
    B_gauss: float = 0.0,
    n_points: int = 300,
    laser_model: str = "stark",
    polarization: str = "pi",
    I_max: Optional[float] = None,
    branch: int = 0,
) -> float:
    """Return I (W/mÂ²) where the splitting shift equals *df_measured_hz*.

    Wrapper around :meth:`LaserSweepResult.intensity_from_splitting_shift`.
    """
    if I_max is None:
        I_max = beam.I0
    return (
        model.laser_sweep(
            beam, I_max=I_max, n_points=n_points,
            model=laser_model, polarization=polarization, B_gauss=B_gauss,
        ).intensity_from_splitting_shift(state_a, state_b, df_measured_hz, branch=branch)
    )


def scattering_rate(
    model: AtomicStructure,
    ground_state: tuple,
    beam: GaussianBeam,
    intensity_Wpm2,
    polarization: str = "pi",
    B_gauss: float = 0.0,
    delta_hz: float = 0.0,
    weak_probe: bool = True,
):
    """Total scattering rate (sâ»Â¹) for *ground_state*.

    Wrapper around :meth:`LaserSweepResult.scattering_rate`.
    Runs a minimal 2-point laser sweep to populate a ``LaserSweepResult``;
    the RWA coupling and linewidths are evaluated at the beam wavelength.
    """
    res_L = model.laser_sweep(
        beam, I_max=float(np.atleast_1d(intensity_Wpm2).max()) or beam.I0,
        n_points=2, model="rwa", polarization=polarization, B_gauss=B_gauss,
    )
    return res_L.scattering_rate(
        ground_state, intensity_Wpm2, delta_hz=delta_hz, weak_probe=weak_probe
    )


def dominant_couplings(
    model: AtomicStructure,
    beam: GaussianBeam,
    polarization: str = "pi",
    B_gauss: float = 0.0,
    n_top: Optional[int] = None,
) -> List[Tuple[tuple, tuple, float]]:
    """Sorted coupling-strength table.

    Wrapper around :meth:`LaserSweepResult.dominant_couplings`.
    """
    res_L = model.laser_sweep(
        beam, I_max=beam.I0, n_points=2,
        model="rwa", polarization=polarization, B_gauss=B_gauss,
    )
    return res_L.dominant_couplings(n_top=n_top)


def transition_frequency_shift(
    model: AtomicStructure,
    state_a: tuple,
    state_b: tuple,
    *,
    B_gauss: Optional[float] = None,
    beam: Optional[GaussianBeam] = None,
    B_max: float = 600.0,
    dB: float = 0.1,
    diamagnetic: bool = False,
    intensity_Wpm2=None,
    n_points: int = 200,
    laser_model: str = "stark",
    polarization: str = "pi",
    I_max: Optional[float] = None,
    as_sweep: bool = False,
):
    """Net shift of a transition frequency under B or laser field.

    Wrapper around :meth:`SweepResult.transition_frequency_shift`.
    Prefer calling that method on an existing sweep result.
    """
    _has_B = B_gauss is not None
    _has_beam = beam is not None
    _has_L = _has_beam and intensity_Wpm2 is not None

    if as_sweep:
        if _has_beam and _has_B:
            raise ValueError("Supply B_gauss OR beam for as_sweep=True, not both.")
        if not _has_beam and not _has_B:
            raise ValueError(
                "For as_sweep=True supply B_gauss (B-field sweep) "
                "OR beam (laser sweep).")
    else:
        if _has_B == _has_L:
            raise ValueError(
                "Supply exactly one of: B_gauss  OR  (beam, intensity_Wpm2).")

    if _has_B:
        res = model.magnetic_sweep(
            B_max=max(B_max, abs(B_gauss or 0.0) + dB), dB=dB, diamagnetic=diamagnetic
        )
        if as_sweep:
            return res.param, res.transition_frequency_shift(state_a, state_b)
        return res.transition_frequency_shift(state_a, state_b, at=B_gauss)

    if I_max is None:
        I_max = beam.I0
    I_max = max(I_max, float(np.atleast_1d(intensity_Wpm2).max()))
    res = model.laser_sweep(
        beam, I_max=I_max, n_points=n_points,
        model=laser_model, polarization=polarization,
    )
    if as_sweep:
        return res.param, res.transition_frequency_shift(state_a, state_b)
    return res.transition_frequency_shift(state_a, state_b, at=intensity_Wpm2)

