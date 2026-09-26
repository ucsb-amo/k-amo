"""Absorption-imaging cross sections: the numbers, where they came from, and the
recipe that reproduces them.

This is the one home for the cross sections the K-machine analysis (waxa) and the
live viewer (liveOD) divide by; both pick one per shot with the rule in
``waxa.calibrations.cross_section``. The numbers used to be literals there and in
``kexp/util/live_od/gui/analyzer.py``.

Importing this module is cheap and ARC-free on purpose: it is read in analysis
worker processes and by the acquisition server, which must not pay for (or open)
ARC's database. Only :func:`compute_closed_line_cross_section` touches an atom
object, and it imports lazily.

What lives here is *atomic physics*. Which value applies to which shot (the
outer-coil-current threshold, the source tags, the no-record fallback) is
experiment logic and stays in ``waxa.calibrations.cross_section``.

Values
------
K39_D2_CLOSED_HIGH_FIELD_M2
    3 lambda^2 / 2 pi on the closed sigma- D2 line
    (4,0,1/2,-1/2,m_i) -> (4,1,3/2,-3/2,m_i) at the high-field imaging point
    B = 520.583 G. Computed 2026-09-12 with kamo ``Potassium39`` (portal matrix
    elements, 2022-survey hyperfine constants); reproduce it with
    ``compute_closed_line_cross_section(*K39_HIGH_FIELD_IMAGING_LINE)``. Treated
    as field-independent near that point: the line moves ~1.4 MHz/G, i.e. 5e-6
    in lambda^2 per gauss.
K39_LEGACY_LAMBDA_SQUARED_M2
    NOT a physical cross section. It equals lambda_D2^2, a factor 2 pi / 3 = 2.09
    larger than the closed-line value. It is what the analysis historically
    assumed for low-field ("0-field") imaging, so it is kept, by name, for
    low-field shots until the low-field cross section is calibrated against field
    (open transition, optical pumping during the pulse). Until 2026-09-18 liveOD
    divided EVERY shot by it; it now follows the analysis's per-shot rule.
    Atom numbers computed with it are low by that factor relative to a
    closed-line estimate.
K39_LEGACY_D1_M2
    3 lambda^2 / 2 pi at the D1 wavelength; the constant the analysis carried
    before 2026-09-16 (0.9 % above the D2 value). Not used; kept for comparing
    old results.
"""

import math

_C_M_PER_S = 299792458.0

K39_D2_CLOSED_HIGH_FIELD_M2 = 2.80668e-13
K39_LEGACY_LAMBDA_SQUARED_M2 = 5.878324268151581e-13
K39_LEGACY_D1_M2 = 2.8316243e-13

# (ground, excited, B in gauss) for K39_D2_CLOSED_HIGH_FIELD_M2.
# States are (n, l, j, m_j, m_i) in the high-field (Paschen-Back) labelling.
K39_HIGH_FIELD_IMAGING_LINE = ((4, 0, 0.5, -0.5, -0.5),
                               (4, 1, 1.5, -1.5, -0.5),
                               520.583)


def closed_line_cross_section(wavelength_m):
    """Resonant cross section of a closed two-level line, ``3 lambda^2 / 2 pi``
    (m^2), for light of the polarisation that drives it."""
    return 3.0 * wavelength_m ** 2 / (2.0 * math.pi)


def compute_closed_line_cross_section(ground, excited, B_G, atom=None):
    """Recompute a closed-line cross section from atomic structure (m^2).

    Slow: builds a kamo atom (ARC database) unless one is passed in. This is
    the recipe behind ``K39_D2_CLOSED_HIGH_FIELD_M2``; it is here so the
    tabulated number can be checked and regenerated, not for use per shot.

    Args:
        ground, excited: states ``(n, l, j, m_j, m_i)``.
        B_G (float): magnetic field in gauss.
        atom: a kamo atom; defaults to ``kamo.Potassium39()``.
    """
    import numpy as np
    if atom is None:
        from kamo.atom_properties.k39 import Potassium39
        atom = Potassium39()
    f_Hz = atom.get_transition_frequency(ground, excited, B=B_G,
                                         relative_mode="absolute")
    f_Hz = float(np.asarray(f_Hz).ravel()[0])
    return closed_line_cross_section(_C_M_PER_S / f_Hz)
