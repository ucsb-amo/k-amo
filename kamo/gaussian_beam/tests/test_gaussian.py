"""Tests for kamo.gaussian_beam.GaussianBeam.

INTERNAL     self-checking algebra -- the radial/axial curvature factors, the
             intensity/power round trip, and depth <-> power inversion.  These
             use an explicitly supplied polarizability so they never touch ARC.
GROUND-TRUTH pinned against the 1064 nm tweezer operating point used in
             docs/light_shift_intensity_calibration.tex (w0 = 3 um,
             nu_r = 1 kHz).  Requires ARC.

Run: pytest kamo/gaussian_beam/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from kamo import constants as c
from kamo.gaussian_beam.gaussian import GaussianBeam

# alpha_s(4S_1/2) at 1064 nm, in SI (C m^2 / V).  Pinned so the INTERNAL tests
# do not need to instantiate ARC.
ALPHA_1064_SI = 591.2810424316043 * c.convert_polarizability_au_to_SI

WAIST = 3e-6
LAMBDA = 1064e-9


def _beam(power=1e-3):
    return GaussianBeam(waist=WAIST, wavelength=LAMBDA, power=power)


# --------------------------------------------------------------- INTERNAL

def test_axial_to_radial_ratio_is_w0_over_sqrt2_zR():
    """omega_z / omega_r = w0 / (sqrt(2) zR) = lambda / (sqrt(2) pi w0).

    This is the regression test for the axial curvature factor: the axial
    potential is Lorentzian, -U0 / (1 + (z/zR)^2), so its coefficient is 2,
    not the 4 of the Gaussian radial profile.  Using 4 for both makes the
    axial frequency sqrt(2) too high.
    """
    b = _beam()
    ratio = (b.trap_frequency_axial(polarizability=ALPHA_1064_SI)
             / b.trap_frequency_radial(polarizability=ALPHA_1064_SI))
    assert ratio == pytest.approx(b.w0 / (np.sqrt(2) * b.zR), rel=1e-12)
    assert ratio == pytest.approx(LAMBDA / (np.sqrt(2) * np.pi * WAIST), rel=1e-12)


def test_matches_bec_properties_axial_convention():
    """kamo.BEC_properties.BEC.get_axial_trap_frequency must agree."""
    from kamo.BEC_properties.bec import BEC

    b = _beam()
    omega_r = b.trap_frequency_radial(polarizability=ALPHA_1064_SI)
    omega_z = b.trap_frequency_axial(polarizability=ALPHA_1064_SI)
    expected = BEC().get_axial_trap_frequency(omega_r, lmbda=LAMBDA, waist=WAIST)
    assert omega_z == pytest.approx(expected, rel=1e-12)


def test_trap_frequencies_from_depth_directly():
    """omega_r = sqrt(4 U0/(m w0^2)) and omega_z = sqrt(2 U0/(m zR^2))."""
    b = _beam()
    U0 = abs(b.trap_depth(polarizability=ALPHA_1064_SI)) * c.kB  # J
    assert b.trap_frequency_radial(polarizability=ALPHA_1064_SI) == pytest.approx(
        np.sqrt(4 * U0 / (c.m_K * b.w0**2)), rel=1e-12)
    assert b.trap_frequency_axial(polarizability=ALPHA_1064_SI) == pytest.approx(
        np.sqrt(2 * U0 / (c.m_K * b.zR**2)), rel=1e-12)


def test_polarizability_argument_is_honoured():
    """Passing polarizability explicitly must work without ARC and must scale.

    Before the axial fix, trap_frequency() ignored its polarizability argument
    and read self.polarizability_ground_state, which only exists when the beam
    was built with include_trap_properties=True.
    """
    b = _beam()
    w1 = b.trap_frequency_radial(polarizability=ALPHA_1064_SI)
    w2 = b.trap_frequency_radial(polarizability=4 * ALPHA_1064_SI)
    assert w2 / w1 == pytest.approx(2.0, rel=1e-12)


def test_power_for_given_trap_depth_round_trip():
    b = _beam()
    target_K = 0.5e-6
    P = b.power_for_given_trap_depth(target_K, polarizability=ALPHA_1064_SI)
    assert abs(b.trap_depth(P, polarizability=ALPHA_1064_SI)) == pytest.approx(
        target_K, rel=1e-12)


def test_peak_intensity_round_trip():
    b = _beam(power=1e-3)
    assert b.I0 == pytest.approx(2 * 1e-3 / (np.pi * WAIST**2), rel=1e-12)
    assert b.power_from_peak_intensity(b.I0) == pytest.approx(1e-3, rel=1e-12)


# ----------------------------------------------------------- GROUND-TRUTH

def test_tweezer_operating_point():
    """w0 = 3 um, 1064 nm, nu_r = 1 kHz -> 44.2 uW, 0.416 uK, nu_z = 79.8 Hz.

    Pins the numbers quoted in docs/light_shift_intensity_calibration.tex,
    step 2.  Needs ARC for the ground-state polarizability.
    """
    from scipy.optimize import brentq

    b = GaussianBeam(waist=WAIST, wavelength=LAMBDA, power=1e-3,
                     include_trap_properties=True)
    assert (b.polarizability_ground_state
            / c.convert_polarizability_au_to_SI) == pytest.approx(591.28, abs=0.05)

    P = brentq(lambda p: b.trap_frequency_radial(power=p) / 2 / np.pi - 1e3,
               1e-9, 1.0)
    assert P * 1e6 == pytest.approx(44.24, abs=0.02)              # uW
    assert abs(b.trap_depth(P)) * 1e6 == pytest.approx(0.4163, abs=5e-4)  # uK
    assert b.trap_frequency_axial(power=P) / 2 / np.pi == pytest.approx(
        79.83, abs=0.05)                                          # Hz
