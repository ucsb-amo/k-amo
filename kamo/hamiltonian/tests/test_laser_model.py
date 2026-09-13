"""Tests for the light-shift machinery in kamo.hamiltonian:

* ``laser_model``: contribution-based basis and the automatic model choice;
* ``perturbative``: the second-order sum over exact field eigenstates.

Numbers pinned here were established 2026-09-13 at the lab operating point
(B = 520.583 G, portal matrix elements, kamo hyperfine constants).
"""

import inspect
import warnings

import numpy as np
import pytest

import kamo.constants as kc
from kamo import ComputePolarizabilities, GaussianBeam, Potassium39
from kamo.hamiltonian import (AtomicStructure, choose_laser_model, choose_sweep_model,
                              light_shift_basis, photon_indices, state_channels,
                              substructure_spread_Hz, sweep_intensity)
from kamo.hamiltonian.perturbative import (channel_polarizability_components,
                                           dipole_operator, perturbative_stark_operator)
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


def _shift(atom, states, f, I, model="auto", pol="pi", **kw):
    """Light shift (Hz) of the transition and the sweep, at the lab field."""
    return atom.get_transition_frequency(
        states[0], states[1], B=B_LAB, frequency_Hz=f, intensity=I, polarization=pol,
        laser_model=model, n_points=4, relative_mode="optical", return_sweep=True, **kw)


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
    """Zeeman half-spread g_J J mu_B B/h ~ 1.46 GHz plus 6A = 36 MHz of hyperfine."""
    W = substructure_spread_Hz((4, 1, 1.5), B_LAB)
    zeeman = 1.334097 * 1.5 * kc.mu_b * B_LAB * 1e-4 / kc.h
    assert W == pytest.approx(zeeman, rel=0.03)
    assert W > zeeman
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
    assert by["4s1/2"].energy_Hz < 0       # 4S lies below 4P: signed energy is negative
    assert 0.0 < sc.core_share < 0.01
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
    # energy ordered, so laser_rwa_operator's lower-first assumption holds
    energies = [atom.getEnergy(*m) for m in sel.manifolds]
    assert energies == sorted(energies)


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

def test_choose_perturbative_light_is_perturbative(atom, f_d2):
    for f, I in [(F_1064, 1e8), (f_d2 - 300e9, 1e6), (f_d2 - 3e12, 1e6), (kc.c / 850e-9, 1e8)]:
        ch = choose_laser_model((GROUND, EXCITED), f, I, B_gauss=B_LAB, atom=atom)
        assert ch.model == "perturbative", (f, ch.describe())
        assert ch.eps_perturbative == ch.eta ** 2 <= ch.eps_rwa
    far = choose_laser_model((GROUND, EXCITED), F_1064, 1e8, B_gauss=B_LAB, atom=atom)
    assert far.eps_stark < 1e-3 < far.eps_rwa     # the explicit models' errors, for the record


def test_choose_rwa_when_next_order_beats_counter_rotating(atom, f_d1):
    """Raman beams: eta^2 ~ 3e-4 exceeds the 1.6e-4 counter-rotating estimate."""
    ch = choose_laser_model((GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6, B_gauss=B_LAB, atom=atom)
    assert ch.model == "rwa" and "next order" in ch.reason
    assert ch.eta ** 2 > ch.eps_rwa


def test_choose_non_perturbative_is_rwa(atom, f_d2):
    ch = choose_laser_model((GROUND, EXCITED), f_d2 - 40e9, 1e8, B_gauss=B_LAB, atom=atom)
    assert ch.model == "rwa" and ch.eta > 0.1 and "non-perturbative" in ch.reason
    ch = choose_laser_model((GROUND, EXCITED), f_d2 - 40e9, None, B_gauss=B_LAB, atom=atom)
    assert ch.eta == 0.0 and ch.model == "perturbative"     # no intensity: assume perturbative


def test_choose_rejects_wrong_state_count(atom):
    with pytest.raises(ValueError):
        choose_laser_model((GROUND,), F_1064, atom=atom)


# ------------------------------------------------------- perturbative model

def test_channel_components_reproduce_fine_structure_polarizability(atom):
    """Summing the per-channel (s, v, t) over every channel gives kamo's
    compute_fine_structure_polarizability, so the residual is on the same footing."""
    cp = ComputePolarizabilities(atom=atom)
    for nlj in [(4, 0, 0.5), (4, 1, 1.5), (3, 2, 2.5)]:
        for lam in (1064e-9, 770.5e-9):
            f_L = kc.c / lam
            sc = state_channels(nlj, f_L)
            tot = np.array([sc.core_au, 0.0, 0.0])
            for ch in sc.channels:
                tot += channel_polarizability_components(nlj[2], ch.manifold[2], ch.d_au,
                                                         ch.energy_Hz, f_L)
            ref = np.array([float(np.atleast_1d(x)[0]) if x is not None else 0.0
                            for x in cp.compute_fine_structure_polarizability(*nlj, lam)])
            assert tot == pytest.approx(ref, rel=1e-10, abs=1e-9)


def test_dipole_operator_selection_rules(atom):
    """d.eps for sigma+ raises m_j by one from ket to bra, for both directions of the pair."""
    m = AtomicStructure([(4, 0, 0.5), (4, 1, 1.5)], atom=atom)
    A = dipole_operator(m.builder, "sigma+")
    sl = m.basis.state_list
    rows, cols = np.nonzero(np.abs(A) > 0)
    assert len(rows) > 0
    for r, c in zip(rows, cols):
        assert sl[r].m_j == pytest.approx(sl[c].m_j + 1)
        assert sl[r].m_i == pytest.approx(sl[c].m_i)
        assert abs(sl[r].l - sl[c].l) == 1
    # not Hermitian for circular light, Hermitian for pi
    assert not np.allclose(A, A.conj().T)
    A_pi = dipole_operator(m.builder, "pi")
    assert np.allclose(A_pi, A_pi.conj().T)
    # stretched element: 4S m=1/2 -> 4P3/2 m=3/2 is sqrt(3) x the m=-1/2 -> 1/2 element
    i_up = m.basis.index_of(4, 1, 1.5, 1.5, 0.5)
    k_up = m.basis.index_of(4, 0, 0.5, 0.5, 0.5)
    i_lo = m.basis.index_of(4, 1, 1.5, 0.5, 0.5)
    k_lo = m.basis.index_of(4, 0, 0.5, -0.5, 0.5)
    assert abs(A[i_up, k_up]) / abs(A[i_lo, k_lo]) == pytest.approx(np.sqrt(3), rel=1e-9)


def test_perturbative_operator_reproduces_polarizabilities_at_zero_field(atom):
    """At B = 0 the 4S block is one scalar and the 4P3/2 diagonal follows alpha_s + tensor."""
    basis = AtomicStructure([(4, 0, .5), (4, 1, .5), (4, 1, 1.5), (5, 0, .5), (3, 2, 1.5),
                             (3, 2, 2.5), (6, 0, .5)], atom=atom)
    beam = GaussianBeam(waist=1e-6, frequency=F_1064, power=0.0)
    op = perturbative_stark_operator(basis.builder, beam, polarization="pi", I_ref=None)
    pre = kc.ac_stark_shift_J(kc.convert_polarizability_au_to_SI, 1.0) / kc.h
    cp = ComputePolarizabilities(atom=atom)
    diag = np.real(np.diag(op["operator"]))
    for man, sl in basis.basis.manifold_slices():
        if man.nlj not in [(4, 0, .5), (4, 1, 1.5)]:
            continue
        a_s, _, a_t = (float(np.atleast_1d(x)[0]) if x is not None else 0.0
                       for x in cp.compute_fine_structure_polarizability(*man.nlj, 1064e-9))
        j = man.j
        for s in basis.basis.state_list[sl]:
            expect = a_s + (a_t * (3 * s.m_j ** 2 - j * (j + 1)) / (j * (2 * j - 1)) if j > 0.5 else 0.0)
            assert diag[s.index] / pre == pytest.approx(expect, rel=2e-5)
    assert op["eta"] == 0.0


def test_perturbative_matches_stark_far_from_resonance(atom):
    """1064 nm, every polarization: same physics as the fine-structure polarizabilities,
    to the 1e-5 level (the residual difference is the resolved Zeeman substructure)."""
    for pol in ("pi", "sigma-", "sigma+", "linear_perp"):
        p, sw = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "perturbative", pol)
        s, _ = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "stark", pol)
        assert p == pytest.approx(s, rel=5e-5), pol
        assert sw.perturbative["eta"] < 1e-3
    k_pi, _ = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "perturbative", "pi")
    assert k_pi / 1e8 * 1e4 / 1e3 == pytest.approx(0.13043, rel=2e-4)   # kHz per W/cm^2


def test_perturbative_matches_rwa_near_resonance(atom, f_d1):
    """40 GHz from D1: same as the RWA to the counter-rotating / next-order level."""
    p, _ = _shift(atom, (GROUND, EXCITED), f_d1 + 40e9, 1e6, "perturbative")
    r, _ = _shift(atom, (GROUND, EXCITED), f_d1 + 40e9, 1e6, "rwa")
    assert p == pytest.approx(r, rel=3e-4)
    s, _ = _shift(atom, (GROUND, EXCITED), f_d1 + 40e9, 1e6, "stark")
    assert abs(s / r - 1) > 5e-3            # the Stark model is ~1.3 % off here
    # qubit under one Raman beam: the hyperfine-admixture differential
    p, _ = _shift(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6, "perturbative")
    r, _ = _shift(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6, "rwa")
    assert r == pytest.approx(6288.8, rel=1e-3)
    assert p == pytest.approx(r, rel=5e-4)
    s, _ = _shift(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6, "stark")
    assert s == 0.0                         # blind to hyperfine admixture


def test_perturbative_qubit_tweezer_shift_is_linear_and_precise(atom):
    """The block-wise diagonalization resolves a 0.25 Hz differential on top of a
    280 kHz state shift; the eigenvalue route through 4e14 Hz energies cannot."""
    vals = []
    for I in (1e8, 1e7, 3e6):
        p, _ = _shift(atom, (GROUND, QUBIT_OTHER), F_1064, I, "perturbative")
        vals.append(p * 1e8 / I)
    assert vals[0] == pytest.approx(0.2530, rel=1e-3)
    assert max(vals) - min(vals) < 1e-4 * abs(vals[0])


def test_perturbative_warns_when_not_perturbative(atom, f_d2):
    with pytest.warns(RuntimeWarning, match="Rabi/\\(2 detuning\\)"):
        _shift(atom, (GROUND, EXCITED), f_d2 - 40e9, 1e8, "perturbative")


def test_rwa_with_auto_basis_matches_stark_at_1064(atom):
    """The old n=4-only basis left the RWA 5.5x low here; the contribution-based
    basis brings it to within the counter-rotating error (1.3 % measured)."""
    k_stark, sw_s = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "stark")
    k_rwa, sw_r = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "rwa")
    assert k_rwa == pytest.approx(k_stark, rel=0.03)
    assert {m.nlj for m in sw_r.basis.manifolds} >= {(3, 2, 2.5), (5, 0, 0.5)}
    assert {m.nlj for m in sw_s.basis.manifolds} == {(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)}


# ------------------------------------------------------ through Potassium39

def test_auto_default_follows_the_choice(atom, f_d1):
    k_auto, sw = _shift(atom, (GROUND, EXCITED), F_1064, 1e8)
    k_pert, _ = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "perturbative")
    assert sw.model_choice.model == "perturbative"
    assert k_auto == k_pert

    k_auto, sw = _shift(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6)
    k_rwa, _ = _shift(atom, (GROUND, QUBIT_OTHER), f_d1 + 40e9, 1.06e6, "rwa")
    assert sw.model_choice.model == "rwa"
    assert k_auto == k_rwa


def test_default_model_is_auto(atom):
    for meth in (atom.get_transition_frequency, atom.get_intensity_from_light_shift):
        assert inspect.signature(meth).parameters["laser_model"].default == "auto"


def test_bad_laser_model_raises(atom):
    with pytest.raises(ValueError, match="laser_model"):
        atom.get_transition_frequency(GROUND, EXCITED, B=B_LAB, frequency_Hz=F_1064,
                                      intensity=1e8, laser_model="floquet")


def test_magnetic_result_unchanged_by_small_sweep_basis(atom):
    """The bare imaging-line Zeeman shift at the lab field (study value -1032.7 MHz)."""
    f = atom.get_transition_frequency(GROUND, EXCITED, B=B_LAB, relative_mode="magnetic")
    assert f / 1e6 == pytest.approx(-1032.75, abs=0.05)


def test_intensity_inversion_uses_auto(atom):
    target = 1.0e5                                  # Hz light shift on the imaging line
    I = atom.get_intensity_from_light_shift(GROUND, EXCITED, target, B=B_LAB,
                                            wavelength_m=1064e-9, n_points=6)
    k, _ = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "perturbative")
    assert I == pytest.approx(target / (k / 1e8), rel=1e-2)


# ------------------------------------------------- sweep-level auto (no transition)

def _channel_model(atom, states, f):
    sel = light_shift_basis(states, f, atom=atom, B_gauss=B_LAB)
    return AtomicStructure(list(sel.manifolds), atom=atom)


def test_sweep_intensity_defaults_to_auto_and_matches_explicit(atom, f_d2, f_d1):
    """laser_sweep / sweep_intensity choose per basis: perturbative for perturbative
    light, RWA when some pair is driven hard, and the result equals the explicit model."""
    cases = [(F_1064, 1e8, (GROUND, EXCITED), "perturbative"),
             (f_d2 - 40e9, 1e8, (GROUND, EXCITED), "rwa"),
             (f_d1 + 40e9, 1.06e6, (GROUND, QUBIT_OTHER), "rwa")]
    for f, I, sts, expect in cases:
        m = _channel_model(atom, sts, f)
        beam = GaussianBeam(waist=1e-6, frequency=f, power=0.0)
        res = m.laser_sweep(beam, I_max=I, n_points=4, B_gauss=B_LAB)
        assert res.model_choice.model == expect, res.model_choice.describe()
        exp = m.laser_sweep(beam, I_max=I, n_points=4, B_gauss=B_LAB, model=expect)
        assert exp.model_choice is None
        assert res.transition_frequency_shift(*sts, at=I) == pytest.approx(
            exp.transition_frequency_shift(*sts, at=I), rel=1e-12)
        low = sweep_intensity(m.builder, beam, I, n_points=4, B_gauss=B_LAB)
        assert low.model_choice.model == expect
    choice, op = choose_sweep_model(m.builder, beam, 1.06e6, B_gauss=B_LAB)
    assert choice.model == "rwa" and choice.eps_perturbative == choice.eta ** 2 > choice.eps_rwa
    assert "eps_rwa_dominant" in op


def test_magnetic_sweep_chain_and_spectroscopy_default_to_auto(atom):
    from kamo.hamiltonian.spectroscopy import transition_frequency_shift
    m = _channel_model(atom, (GROUND, EXCITED), F_1064)
    beam = GaussianBeam(waist=1e-6, frequency=F_1064, power=0.0)
    res_b = m.magnetic_sweep(B_max=B_LAB + 0.1, dB=0.1)
    res_l = res_b.laser_sweep(beam, B_gauss=B_LAB, I_max=1e8, n_points=4)
    assert res_l.model_choice.model == "perturbative"
    k_pert, _ = _shift(atom, (GROUND, EXCITED), F_1064, 1e8, "perturbative")
    assert res_l.transition_frequency_shift(GROUND, EXCITED, at=1e8) == pytest.approx(k_pert, rel=1e-6)
    df = transition_frequency_shift(m, GROUND, EXCITED, beam=beam, intensity_Wpm2=1e8,
                                    I_max=1e8, n_points=4)          # B = 0, auto
    assert np.isfinite(df) and df > 0


def test_sweep_rejects_unknown_model(atom):
    m = AtomicStructure([(4, 0, 0.5), (4, 1, 1.5)], atom=atom)
    beam = GaussianBeam(waist=1e-6, frequency=F_1064, power=0.0)
    with pytest.raises(ValueError, match="model must be"):
        m.laser_sweep(beam, I_max=1e8, n_points=2, model="floquet")
