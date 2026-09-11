"""Calibrated inner-wall corrections for the Falke/Tiemann K2 potentials.

Output of :func:`kamo.scattering.calibration.calibrate` against the
measured s-wave resonance positions of :mod:`.k39_feshbach` (``fit_set()``:
15 elastic resonances in 5 channels), on the default
CoupledChannels grid (h = 0.003 a0).  Regenerate after changing the database
or the grid::

    from kamo.scattering.calibration import calibrate
    res = calibrate(); print(res.summary())

The correction added to each curve is ``delta * (r - r_eq)^2`` for
``r < r_eq`` (Chapurin et al. 2019 form); only the repulsive wall moves.
Generated 2026-09-10.
"""

# inner-wall correction delta * (r - r_eq)^2 for r < r_eq  [Hartree / a0^2]
R_EQ_S_A0 = 7.3160      # Chapurin et al. PRL 123, 233402 (2019), Suppl.
R_EQ_T_A0 = 10.7371
DELTA_S = -1.25979509e-07
DELTA_T = 1.13388175e-09

# fitted singlet/triplet scattering lengths (a0), 1-sigma statistical
A_SINGLET = 138.7150      # grid-converged (h -> 0)
A_TRIPLET = -33.3868
A_SINGLET_UNC = 0.0047
A_TRIPLET_UNC = 0.0109
A_ST_CORR = -0.185
A_SINGLET_GRID = 138.7094   # on the default grid (what the CC actually uses)
A_TRIPLET_GRID = -33.3869
CHI2 = 7.520
DOF = 13
SIGMA_MODEL_G = 0.02

# bare Falke 2008 curves (delta = 0), grid-converged
A_SINGLET_BARE = 138.8085
A_TRIPLET_BARE = -33.3903

# Chapurin's own fitted values (on the Falke 2008 potentials), for reference
DELTA_S_CHAPURIN = 5.3196e-8
DELTA_T_CHAPURIN = -2.4394e-9

# (label, state_a, state_b, B0_exp, B0_unc, B0_cc_bare, B0_cc_calibrated,
#  B0_theory_Etrych2023, used_in_fit, source)   [Gauss]
POSITIONS = [
    ('|1,+1>+|1,+1>', (1, 1), (1, 1), 25.91, 0.06, 25.88, 25.8761, 25.879, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,+1>', (1, 1), (1, 1), 402.74, 0.01, 402.5029, 402.7415, 402.554, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,+1>', (1, 1), (1, 1), 752.3, 0.1, 752.2874, 752.2866, 752.255, True, "D'Errico 2007 NJP 9,223"),
    ('|1,+0>+|1,+0>', (1, 0), (1, 0), 58.97, 0.12, 58.8122, 58.8345, 58.949, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+0>+|1,+0>', (1, 0), (1, 0), 65.57, 0.23, 65.5772, 65.5771, 65.573, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+0>+|1,+0>', (1, 0), (1, 0), 472.33, 0.01, 472.0653, 472.3131, 472.118, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+0>+|1,+0>', (1, 0), (1, 0), 491.17, 0.07, 490.8853, 491.1089, 490.93, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,-1>+|1,-1>', (1, -1), (1, -1), 33.582, 0.0014, 33.5774, 33.5646, 33.568, True, 'Chapurin 2019 PRL 123,233402'),
    ('|1,-1>+|1,-1>', (1, -1), (1, -1), 162.36, 0.02, 162.3402, 162.348, 162.347, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,-1>+|1,-1>', (1, -1), (1, -1), 561.14, 0.02, 560.8774, 561.1252, 560.935, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,+0>', (1, 1), (1, 0), 25.81, 0.06, 25.7798, 25.7762, 25.785, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,+0>', (1, 1), (1, 0), 39.81, 0.06, 39.8366, 39.8328, 39.835, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,+0>', (1, 1), (1, 0), 445.42, 0.03, 445.2443, 445.4787, 445.293, True, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,-1>', (1, 1), (1, -1), 77.6, 0.4, 77.7151, 77.745, 77.725, False, 'Etrych 2023 PRR 5,013174'),
    ('|1,+1>+|1,-1>', (1, 1), (1, -1), 501.6, 0.3, 501.4369, 501.6556, 501.48, False, 'Etrych 2023 PRR 5,013174'),
    ('|1,+0>+|1,-1>', (1, 0), (1, -1), 113.76, 0.01, 113.7554, 113.7658, 113.768, True, 'Tanzi 2018 PRA 98,062712'),
    ('|1,+0>+|1,-1>', (1, 0), (1, -1), 526.16, 0.03, 525.9469, 526.1814, 525.995, True, 'Etrych 2025 PRX 15,021070'),
]
