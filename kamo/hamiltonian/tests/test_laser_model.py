"""Tests for kamo.hamiltonian.laser_model: contribution-based RWA basis and the
automatic RWA / Stark choice.

Numbers pinned here were established 2026-09-13 at the lab operating point
(B = 520.583 G, portal matrix elements, kamo hyperfine constants).
"""

import warnings

import numpy as np
import pytest

import kamo.constants as kc
from kamo import Potassium39
from kamo.hamiltonian import (choose_laser_model, light_shift_basis,
                              photon_indices, state_channels,
                              substructure_spread_Hz)
from kamo.atom_properties.k39 import _own_manifolds

B_LAB = 520.583
GROUND = (4, 0, 0.5, -0.5, -0.5)          # |up> = |1,-1>
EXCITED = (4, 1, 1.5, -1.5, -0.5)         # its sigma- D2 imaging partner
QUBIT_OTHER = (4, 0, 0.5, -0.5, +0.5)     # |dn> = |1,0>
F_1064 = kc.c / 1064e-9


@pytest.fixture(scope="module")
def atom():
    return Potassium39()


@pytest.fixture(scope="module")
def f_d1(atom):
    return atom.getTransitionFrequency(4, 0, 0.5, 4, 1, 0.5)


@pytest.fixture(scope="module")
def f_d2(atom):
    return atom.getTransitionFrequency(4, 0, 0.5, 4, 1, 1.5)


# ------------------------------------------------------------- building blocks

def test_photon_indices_detects_loop():
    """A 4S-4P-5S-5P square cannot be given consistent single-frequency indices."""
    A, Bm, C, D = (4, 0, 0.5), (4, 1, 0.5), (5, 0, 0.5), (5, 1, 0.5)
    energies = {A: 0.0, Bm: 1.0, C: 2.0, D: 3.0}
    idx, bad = photon_indices([A, Bm, C], energies)
    assert idx == {A: 0, Bm: 1, C: 2} and bad == []
    _, bad = photon_indices([A, Bm, C, D], energies)
    assert bad, "5P is one photon up from 4S but three up from 5S: a loop"


def test_substructure_spread_4p32_at_lab_field():
    """Zeeman half-spread g_J J mu_B B/h ~ 1.46 GHz plus a few MHz of hyperfine."""
    W = substructure_spread_Hz((4, 1, 1.5), B_LAB)
    zeeman = 1.334097 * 1.5 * kc.mu_b * B_LAB * 1e-4 / kc.h
    assert W == pytest.approx(zeeman, rel=0.03)      # 6A = 36 MHz of hyperfine on top
    assert W > zeeman                      # hyperfine adds a positive spread
    assert substructure_spread_Hz((4, 0, 0.5), 0.0) == pytest.approx(
        2 * 230.85986e6, rel=1e-6)         # 4S: A (I+1/2) = 2A at B = 0


def test_state_channels_4p32_at_1064(atom):
    """3D and 5S carry the 4P3/2 polarizability at 1064 nm; 4S is a minor channel."""
    sc = state_channels(EXCITED, F_1064, B_gauss=B_LAB)
    by = {c.final: c for c in sc.channels}
    assert sc.alpha_total_au < 0           # anti-trapped: laser is blue of 4P->3D, 4P->5S
    assert by["3d5/2"].share > 0.7
    assert by["3d5/2"].share > by["5s1/2"].share > by["3d3/2"].share > by["4s1/2"].share
    assert by["4s1/2"].share < 0.08
    assert by["3d5/2"].detuning_Hz < 0     # laser blue of the 4P->3D line
    assert by["4s1/2"].detuning_Hz > 0     # laser red of the D2 line
    assert 0.0 < sc.core_share < 0.01
    # channel-level error estimates have the documented forms
    c = by["4s1/2"]
    assert c.eps_rwa == pytest.approx(abs(c.detuning_Hz) / (abs(c.detuning_Hz + F_1064) + F_1064))
    assert c.eps_stark == pytest.approx(c.substructure_Hz / abs(c.detuning_Hz))


# -------------------------------------------------------------- basis choice

def test_light_shift_basis_1064_includes_3d_and_5s(atom):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sel = light_shift_basis((GROUND, EXCITED), F_1064, atom=atom, B_gauss=B_LAB)
    mans = set(sel.manifolds)
    assert {(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)} <= mans     # own + fine-structure partner
    assert {(3, 2, 1.5), (3, 2, 2.5), (5, 0, 0.5)} <= mans     # the 1064 nm channels of 4P
    assert (4, 2, 2.5) not in mans                             # 4P->4D is negligible here
    assert sel.dropped == ()
    assert all(0.0 < c <= 1.0 for c in sel.coverage.values())
    assert sel.min_coverage > 0.98
    assert sel.coverage[(4, 0, 0.5)] == pytest.approx(1 - sel.core_share[(4, 0, 0.5)], abs=2e-3)


def test_light_shift_basis_same_manifold_counts_state_once(atom):
    sel = light_shift_basis((GROUND, QUBIT_OTHER), F_1064, atom=atom, B_gauss=B_LAB)
    assert list(sel.coverage) == [(4, 0, 0.5)]
    assert sel.coverage[(4, 0, 0.5)] <= 1.0
    assert set(sel.manifolds) == {(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)}


def test_light_shift_basis_near_d1_is_just_4s_4p(atom, f_d1):
    sel = light_shift_basis((GROUND, QUBIT_OTHER), f_d1 + 40e9, atom=atom, B_gauss=B_LAB)
    assert set(sel.manifolds) == {(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)}
    assert sel.min_coverage > 0.999


def test_own_manifolds():
    assert _own_manifolds(GROUND, EXCITED) == [(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)]
    assert _own_manifolds(GROUND, QUBIT_OTHER) == [(4, 0, 0.5)]


# -------------------------------------------------------------- model choice

def test_choose_far_detuned_is_stark(atom):
    ch = choose_laser_model((GROUND, EXCITED), F_1064, 1e8, B_gauss=B_LAB, atom=atom)
    assert ch.model == "stark"
    assert ch.eps_stark < 1e-3 < ch.eps_rwa
    assert ch.eta < 1e-2


def test_choose_near_line_is_rwa(atom, f_d1, f_d2):
    ch = choose_laser_model((GROUND, EXCITED), f_d1 + 40e9, 1e6, B_gauss=B_LAB, atom=atom)
    assert ch.model == "rwa"
    assert ch.eps_rwa < ch.eps_stark
    ch = choose_laser_model((GROUND, EXCITED), f_d2 - 300e9, 1e6, B_gauss=B_LAB, atom=atom)
    assert ch.model == "rwa"


def test_choose_crossover_between_300ghz_and_3thz(atom, f_d2):
    """Both models agree to <1e-3 in this window, so the crossover must sit inside it."""
    assert choose_laser_model((GROUND, EXCITED), f_d2 - 300e9, 1e6, B_gauss=B_LAB, atom=atom).model == "rwa"
    assert choose_laser_model((GROUND, EXCITED), f_d2 - 3e12, 1e6, B_gauss=B_LAB, atom=atom).model == "stark"


def test_choose_same_manifold_is_rwa_at_any_detuning(atom, f_d1):
    near = choose_laser_model((GROUND, QUBIT_OTHER), f_d1 + 40e9, 1e6, B_gauss=B_LAB, atom=atom)
    assert near.model == "rwa" and "one manifold" in near.reason
    with pytest.warns(RuntimeWarning, match="only model for a within-manifold"):
        far = choose_laser_model((GROUND, QUBIT_OTHER), F_1064, 1e8, B_gauss=B_LAB, atom=atom)
    assert far.model == "rwa"


def test_choose_non_perturbative_is_rwa(atom, f_d2):
    ch = choose_laser_model((GROUND, EXCITED), f_d2 - 40e9, 1e8, B_gauss=B_LAB, atom=atom)
    assert ch.model == "rwa" and ch.eta > 0.1 and "non-perturbative" in ch.reason
    ch = choose_laser_model((GROUND, EXCITED), f_d2 - 40e9, None, B_gauss=B_LAB, atom=atom)
    assert ch.eta == 0.0 and ch.model == "rwa"     # still rwa: substructure dominates


def test_choose_rejects_wrong_state_count(atom):
    with pytest.raises(ValueError):
        choose_laser_model((GROUND,), F_1064, atom=atom)


# ------------------------------------------------------ through Potassium39

def _kappa(atom, states, f, I, model="auto", **kw):
    df, sw = atom.get_transition_frequency(
        states[0], states[1], B=B_LAB, frequency_Hz=f, intensity=I, polarization="pi",
        laser_model=model, n_points=4, relative_mode="optical", return_sweep=True, **kw)
    return df / I, sw


def test_rwa_with_auto_basis_matches_stark_at_1064(atom):
    """The old n=4-only basis left the RWA 5.5x low here; the contribution-based
    basis brings it to within the counter-rotating error (1.3 % measured)."""
    k_stark, sw_s = _kappa(atom, (GROUND, EXCITED), F_1064, 1e8, "stark")
    k_rwa, sw_r = _kappa(atom, (GROUND, EXCITED), F_1064, 1e8, "rwa")
    assert k_stark * 1e4 / 1e3 == pytest.approx(0.13043, rel=2e-3)   # kHz per W/cm^2
    assert k_rwa == pytest.approx(k_stark, rel=0.03)
    assert {m.nlj for m in sw_r.basis.manifolds} >= {(3, 2, 2.5), (5, 0, 0.5)}
    assert {m.nlj for m in sw_s.basis.manifolds} == {(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)}


def test_auto_default_follows_the_choice(atom, f_d1):
    k_auto, sw = _kappa(atom, (GROUND, EXCITED), F_1064, 1e8)
    k_stark, _ = _kappa(atom, (GROUND, EXCITED), F_1064, 1e8, "stark")
    assert sw.model_choice.model == "stark"
    assert k_auto == k_stark

    k_auto, sw = _kappa(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6)
    k_rwa, _ = _kappa(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6, "rwa")
    assert sw.model_choice.model == "rwa"
    assert k_auto == k_rwa
    assert k_auto * 1.06e6 == pytest.approx(6288.8, rel=1e-3)   # Hz, one Raman beam, pi


def test_default_model_is_auto(atom):
    import inspect
    assert inspect.signature(atom.get_transition_frequency).parameters["laser_model"].default == "auto"
    assert inspect.signature(atom.get_intensity_from_light_shift).parameters["laser_model"].default == "auto"


def test_bad_laser_model_raises(atom):
    with pytest.raises(ValueError, match="laser_model"):
        atom.get_transition_frequency(GROUND, EXCITED, B=B_LAB, frequency_Hz=F_1064,
                                      intensity=1e8, laser_model="floquet")


def test_intensity_inversion_uses_auto(atom):
    target = 1.0e5                                  # Hz light shift on the imaging line
    I = atom.get_intensity_from_light_shift(GROUND, EXCITED, target, B=B_LAB,
                                            wavelength_m=1064e-9, n_points=8)
    k_stark, _ = _kappa(atom, (GROUND, EXCITED), F_1064, 1e8, "stark")
    assert I == pytest.approx(target / k_stark, rel=1e-2)
