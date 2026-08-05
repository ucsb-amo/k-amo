"""Validation of the transcribed Falke/Tiemann K2 potentials (PRA 78, 012503).

Checks the reconstruction reproduces the paper's well depths exactly and the
Table VIII scattering lengths to within transcription precision (~a few %).
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


def test_c6_atomic_units():
    # Falke C6 in cm^-1 A^6 -> ~3922 a.u. (matches literature 3921)
    C6_au = T.SINGLET["C6"] / T.HARTREE_CM1 * (1.0 / T.BOHR_ANG) ** 6
    assert abs(C6_au - 3921.0) < 5.0


@pytest.mark.parametrize("P,rin,target,tol", [
    (T.SINGLET, 4.0, 138.80, 5.0),   # Falke Table VIII case A
    (T.TRIPLET, 6.0, -33.41, 3.0),
])
def test_scattering_lengths(P, rin, target, tol):
    a = T.scattering_length(P, r_in_bohr=rin)
    assert abs(a - target) < tol      # reconstruction within transcription precision
