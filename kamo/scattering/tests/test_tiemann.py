"""Validation of the transcribed Falke/Tiemann K2 potentials (PRA 78, 012503).

Checks the reconstruction reproduces the paper's well depths, is continuous
where the three analytic regions join (a transcription error in any series
coefficient shows up as a step there), and gives the Table VIII scattering
lengths -- which Laskowski & Mehta (arXiv:2307.16654) also reproduce
independently from the same curves (138.808, -33.391 a0).
"""

import numpy as np
import pytest

from kamo.scattering import tiemann as T


def test_well_depths_exact():
    # U(Rm) must equal a0 (the -De well depth) to the cm^-1
    assert abs(T.potential_cm1(T.SINGLET["Rm"], T.SINGLET) - T.SINGLET["a"][0]) < 0.01
    assert abs(T.potential_cm1(T.TRIPLET["Rm"], T.TRIPLET) - T.TRIPLET["a"][0]) < 0.01


def test_potential_minima():
    # true minimum near Rm and equal to -De
    for P, De in [(T.SINGLET, 4450.9), (T.TRIPLET, 255.0)]:
        Rg = np.linspace(P["Rinn"], 8.0, 20000)
        Umin = T.potential_cm1(Rg, P).min()
        assert abs(Umin + De) < 1.0


@pytest.mark.parametrize("P", [T.SINGLET, T.TRIPLET])
@pytest.mark.parametrize("edge", ["Rinn", "Rout"])
def test_continuous_at_region_boundaries(P, edge):
    # the paper sets A, B (inner) and the long-range form for continuity;
    # the old transcription had a 47 cm^-1 step in the triplet at Rout
    R = P[edge]
    lo = T.potential_cm1(np.array([R - 1e-9]), P)[0]
    hi = T.potential_cm1(np.array([R + 1e-9]), P)[0]
    assert abs(hi - lo) < 1e-3          # cm^-1


def test_c6_atomic_units():
    # Falke C6 in cm^-1 A^6 -> ~3922 a.u. (matches literature 3921)
    C6_au = T.SINGLET["C6"] / T.HARTREE_CM1 * (1.0 / T.BOHR_ANG) ** 6
    assert abs(C6_au - 3921.0) < 5.0


def test_scattering_lengths_match_literature():
    # bare curves on the default coupled-channels grid
    from kamo.scattering.coupled_channels import CoupledChannels
    cc = CoupledChannels(B_max=10.0, delta_S=0.0, delta_T=0.0)
    a_S, a_T = cc.singlet_triplet_a()
    assert abs(a_S - 138.808) < 0.02      # Falke Table VIII 138.80; Laskowski 138.808
    assert abs(a_T - (-33.391)) < 0.02    # Falke -33.41; Laskowski -33.391
