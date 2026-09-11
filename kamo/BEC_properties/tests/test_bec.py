"""Tests for kamo.BEC_properties.bec.BEC, now a thin interface over kamo.trap.

INTERNAL  every method reproduces the previous closed-form implementation
          (kept verbatim below as the oracle) for scalars, arrays and
          non-default geometries; the axial frequency really comes from the
          kamo.trap Hessian; construction is silent.

Run: pytest kamo/BEC_properties/tests -q
"""

import warnings

import numpy as np
import pytest

import kamo.constants as c
from kamo.BEC_properties import bec as bec_mod
from kamo.BEC_properties.bec import BEC


class Legacy:
    """The pre-kamo.trap implementation, verbatim: the oracle."""

    def oscillator_len(self, trap_freq):
        return np.sqrt(c.hbar / (c.m_K * trap_freq))

    def get_axial_trap_frequency(self, radial_trap_frequency, lmbda=1064.e-9, waist=3.8e-6):
        return radial_trap_frequency * (lmbda / (np.sqrt(2) * np.pi * waist))

    def chemical_potential(self, n_atoms, a_scattering, radial_trap_frequency):
        a_scat = a_scattering * 5.29177210544e-11
        ax_trap = self.get_axial_trap_frequency(radial_trap_frequency=radial_trap_frequency)
        bar_omega = (radial_trap_frequency * radial_trap_frequency * ax_trap) ** (1 / 3)
        bar_osc_len = self.oscillator_len(bar_omega)
        return 1.4770884695313888 * (((n_atoms * a_scat) / bar_osc_len) ** (2 / 5)) * bar_omega

    def thomas_fermi_radius(self, n_atoms, a_scattering, radial_trap_frequency, trap_frequency):
        chem_pot = c.hbar * self.chemical_potential(n_atoms=n_atoms, a_scattering=a_scattering,
                                                    radial_trap_frequency=radial_trap_frequency)
        return np.sqrt((2 * chem_pot) / (c.m_K * (trap_frequency ** 2)))


OLD = Legacy()
W_R = 2 * np.pi * 3.6e3                           # the tf_radius notebook's trap
REL = 1e-11


def test_construction_is_silent_and_keeps_the_defaults():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = BEC()
    assert (b.lmbda, b.waist) == (1064e-9, 3.8e-6)


def test_oscillator_length():
    assert BEC().oscillator_len(W_R) == OLD.oscillator_len(W_R)


@pytest.mark.parametrize("kw", [{}, {"waist": 3.0e-6}, {"lmbda": 532e-9, "waist": 1.2e-6}])
def test_axial_frequency(kw):
    radial = np.array([2 * np.pi * 600.0, W_R, 2 * np.pi * 13e3])
    assert BEC().get_axial_trap_frequency(W_R, **kw) == pytest.approx(
        OLD.get_axial_trap_frequency(W_R, **kw), rel=REL)
    assert BEC().get_axial_trap_frequency(radial, **kw) == pytest.approx(
        OLD.get_axial_trap_frequency(radial, **kw), rel=REL)


def test_geometry_can_be_set_once_on_the_instance():
    b = BEC(lmbda=780e-9, waist=2.0e-6)
    assert b.get_axial_trap_frequency(W_R) == pytest.approx(
        OLD.get_axial_trap_frequency(W_R, lmbda=780e-9, waist=2.0e-6), rel=REL)


def test_array_waist_broadcasts():
    w = np.array([2e-6, 3e-6, 3.8e-6])
    assert BEC().get_axial_trap_frequency(W_R, waist=w) == pytest.approx(
        OLD.get_axial_trap_frequency(W_R, waist=w), rel=REL)


def test_chemical_potential_and_radii():
    n = np.array([100.0, 500.0, 1000.0])
    assert BEC().chemical_potential(1000, 10, W_R) == pytest.approx(
        OLD.chemical_potential(1000, 10, W_R), rel=REL)
    assert BEC().chemical_potential(n, 10, W_R) == pytest.approx(
        OLD.chemical_potential(n, 10, W_R), rel=REL)
    w_ax = OLD.get_axial_trap_frequency(W_R)
    for w in (W_R, w_ax):
        assert BEC().thomas_fermi_radius(1e3, 10, W_R, w) == pytest.approx(
            OLD.thomas_fermi_radius(1e3, 10, W_R, w), rel=REL)


def test_axial_frequency_comes_from_the_trap_hessian(monkeypatch):
    from kamo.trap.trap import Trap

    def refuse(self):
        raise RuntimeError("routed through Trap.trap_frequencies")

    bec_mod._axial_over_radial.cache_clear()
    monkeypatch.setattr(Trap, "trap_frequencies", refuse)
    with pytest.raises(RuntimeError, match="routed"):
        BEC().get_axial_trap_frequency(W_R, waist=3.3e-6)
    bec_mod._axial_over_radial.cache_clear()
