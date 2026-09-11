"""Potassium39 with UDel-portal E1 data.

Runs offline: the cache points at an empty directory and the network is
disabled, so the portal data come from the snapshot bundled with kamo.
"""

import numpy as np
import pytest

from kamo.light_shift import udel_portal as up


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("KAMO_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))

    def no_network(*args, **kwargs):
        raise OSError("network disabled for tests")

    mp.setattr(up, "_http", no_network)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def portal_atom(offline):
    from kamo import Potassium39
    with pytest.warns(UserWarning, match="unreachable"):
        return Potassium39()


@pytest.fixture(scope="module")
def arc_atom():
    from kamo import Potassium39
    return Potassium39(use_portal=False)


def test_snapshot_has_k1_data(offline):
    with pytest.warns(UserWarning, match="unreachable"):
        lt = up.lifetimes("K1")
    assert len(lt) == 44
    d2 = lt[lt.state == "4p3/2"].iloc[0]
    assert d2.source == "exp"
    assert d2.tau_s == pytest.approx(26.39e-9)


def test_lifetimes_and_linewidths_are_portal_values(portal_atom):
    """Measured 4P lifetimes (Falke et al. 2006) via the portal."""
    assert portal_atom.getStateLifetime(4, 1, 1.5) == pytest.approx(26.39e-9)
    assert portal_atom.getStateLifetime(4, 1, 0.5) == pytest.approx(26.74e-9)
    gamma_d2 = portal_atom.get_decay_rate(4, 0, 0.5, 4, 1, 1.5) / (2 * np.pi)
    assert gamma_d2 == pytest.approx(6.0309e6, rel=1e-4)


def test_use_portal_false_is_plain_arc(arc_atom):
    assert arc_atom.getStateLifetime(4, 1, 1.5) == pytest.approx(26.504e-9, abs=1e-12)
    assert arc_atom.getReducedMatrixElementJ(4, 0, 0.5, 4, 1, 1.5) == pytest.approx(5.794, abs=1e-3)


@pytest.mark.parametrize("pair, magnitude", [
    ((4, 0, 0.5, 4, 1, 1.5), 5.807),
    ((4, 1, 1.5, 4, 0, 0.5), 5.807),
    ((4, 0, 0.5, 4, 1, 0.5), 4.106),
    ((4, 1, 1.5, 3, 2, 2.5), 10.6949),
])
def test_reduced_elements_take_portal_magnitude_and_arc_sign(portal_atom, arc_atom,
                                                             pair, magnitude):
    value = portal_atom.getReducedMatrixElementJ(*pair)
    assert abs(value) == pytest.approx(magnitude, rel=1e-4)
    assert np.sign(value) == np.sign(arc_atom.getReducedMatrixElementJ(*pair))


def test_dipole_matrix_element_is_rescaled_arc(portal_atom, arc_atom):
    """Angular factor and sign from ARC, magnitude from the portal."""
    cycling = (4, 0, 0.5, 0.5, 4, 1, 1.5, 1.5, 1)
    ratio = (portal_atom.getDipoleMatrixElement(*cycling)
             / arc_atom.getDipoleMatrixElement(*cycling))
    assert ratio == pytest.approx(5.807 / 5.794, rel=1e-6)


def test_transition_rates_are_portal_einstein_a(portal_atom):
    # 3D -> 4P: ARC's own rate is 2.3% low because its 3D energy is off
    assert portal_atom.getTransitionRate(3, 2, 1.5, 4, 1, 0.5) == pytest.approx(2.0016e7, rel=1e-4)
    assert portal_atom.getTransitionRate(4, 1, 0.5, 3, 2, 1.5) == 0.0   # upward, T = 0


def test_finite_temperature_lifetime_uses_portal_elements(portal_atom):
    tau = portal_atom.getStateLifetime(4, 1, 1.5, temperature=300, includeLevelsUpTo=6)
    assert tau == pytest.approx(26.39e-9, rel=1e-3)


def test_energies_are_nist_levels(arc_atom):
    """D lines within 20 MHz of Tiecke (391.016170 / 389.286059 THz)."""
    assert arc_atom.getTransitionFrequency(4, 0, 0.5, 4, 1, 1.5) == pytest.approx(391.016170e12, abs=20e6)
    assert arc_atom.getTransitionFrequency(4, 0, 0.5, 4, 1, 0.5) == pytest.approx(389.286059e12, abs=20e6)
    # 3D3/2 - 4P1/2: NIST 1169.34 nm (ARC's quantum defects gave 1178.59 nm)
    assert abs(arc_atom.getTransitionWavelength(4, 1, 0.5, 3, 2, 1.5)) * 1e9 == pytest.approx(1169.34, abs=0.01)


def test_quantum_defect_energies_still_available():
    from kamo import Potassium39
    qd = Potassium39(use_portal=False, preferQuantumDefects=True)
    assert qd.getTransitionFrequency(4, 0, 0.5, 4, 1, 1.5) == pytest.approx(391.018732e12, abs=1e6)


def test_sign_uncertainty_flags_only_the_small_elements(portal_atom):
    assert portal_atom.portal_sign_uncertain(4, 0, 0.5, 8, 1, 0.5)
    assert not portal_atom.portal_sign_uncertain(4, 0, 0.5, 4, 1, 1.5)
    assert not portal_atom.portal_sign_uncertain(4, 1, 1.5, 3, 2, 2.5)
