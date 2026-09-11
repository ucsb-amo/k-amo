"""Tests for the pair-state lookup behind Potassium39.get_scattering_length."""

import numpy as np
import pytest

from kamo.scattering.lookup import scattering_length, available_pairs, kokkelmans_dir


@pytest.fixture(scope="module")
def cc():
    from kamo.scattering.coupled_channels import CoupledChannels
    return CoupledChannels()


@pytest.mark.parametrize("a,b,B", [
    ((1, -1), (1, -1), 520.58), ((1, 0), (1, 0), 472.40),      # 0.09 G from a pole
    ((1, -1), (1, 0), 56.83), ((2, 0), (2, 0), 100.0),         # inter-state, lossy
    ((1, 1), (1, -1), 77.80), ((1, 0), (1, 0), 824.50),        # lossy pole, narrow pole
])
def test_table_matches_direct_cc(cc, a, b, B):
    t = scattering_length(a, b, B, method="table", return_complex=True)
    d = cc.scattering_length(a, b, B)
    assert abs(t - d) < 1e-3 * max(abs(d), 1.0)


def test_table_poles_are_calibrated_positions():
    from kamo.scattering.tables import table_poles
    from kamo.scattering.data import k39_calibration as kc
    for label, sa, sb, B0, unc, bare, cal, *_ in kc.POSITIONS:
        poles = table_poles(sa, sb).real
        assert np.min(np.abs(poles - cal)) < 1e-3, (label, B0)


def test_table_below_valid_range_raises():
    with pytest.raises(ValueError, match="range"):
        scattering_length((1, 0), None, 0.5)
    assert np.isfinite(scattering_length((1, 0), None, 0.5, method="cc"))


def test_same_state_default_and_symmetry():
    a1 = scattering_length((1, -1), None, 520.58)
    assert isinstance(a1, float)
    assert a1 == scattering_length((1, -1), (1, -1), 520.58)
    # pair order does not matter
    assert scattering_length((1, -1), (1, 0), 56.83) == scattering_length((1, 0), (1, -1), 56.83)


def test_array_shape_and_published_value():
    B = np.array([[54.69, 56.83]])
    a = scattering_length((1, -1), (1, 0), B)
    assert a.shape == B.shape
    assert np.allclose(a, [[-54.2, -53.2]], atol=0.2)   # Hammond 2022 / Lavoine 2021


def test_complex_for_lossy_pair():
    a = scattering_length((2, 0), None, 100.0, return_complex=True)
    assert a.imag < 0 and scattering_length((2, 0), None, 100.0) == a.real


def test_cc_covers_all_36_pairs():
    assert len(available_pairs("cc")) == 36


@pytest.mark.parametrize("a,b,method", [
    ((2, 0), None, "empirical"),          # no measured resonances
    ((1, -1), (1, 0), "kokkelmans"),      # tables are same-state only
])
def test_no_data_raises(a, b, method):
    with pytest.raises(ValueError, match="no .* data"):
        scattering_length(a, b, 100.0, method=method)


@pytest.mark.parametrize("a,b,B", [((1, 2), None, 100.0), ((3, 0), None, 100.0),
                                    ((1, 0), None, 1500.0), ((1, 0), None, -1.0)])
def test_invalid_inputs_raise(a, b, B):
    with pytest.raises(ValueError):
        scattering_length(a, b, B)


def test_potassium39_wrapper():
    from kamo import Potassium39
    atom = Potassium39()
    assert atom.get_scattering_length(1, -1, 520.58) == scattering_length((1, -1), None, 520.58)
    assert atom.get_scattering_length(1, -1, 113.0, 1, 0, method="empirical") == \
        scattering_length((1, -1), (1, 0), 113.0, method="empirical")
    with pytest.raises(ValueError, match="both f2 and mf2"):
        atom.get_scattering_length(1, -1, 100.0, f2=1)


@pytest.mark.skipif(kokkelmans_dir() is None, reason="Kokkelmans tables (G:/B: drive) not reachable")
def test_kokkelmans_close_to_cc_away_from_poles():
    for B in (150.0, 300.0, 620.0):
        k = scattering_length((1, 0), None, B, method="kokkelmans", interp=True)
        assert abs(k - scattering_length((1, 0), None, B)) < 0.3
