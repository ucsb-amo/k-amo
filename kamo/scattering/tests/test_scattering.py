"""Consistency suite for kamo.scattering.

Two tiers:
  * INTERNAL — self-validating physics/algebra that needs no fitted params.
  * GROUND-TRUTH — checks against the literature resonance database
    (:mod:`kamo.scattering.data.k39_feshbach`).
"""

import numpy as np
import pytest

from kamo.scattering import ScatteringModel, k2_from_scattering_length
from kamo.scattering import channels as ch
from kamo.scattering import units as u
from kamo.scattering.channels import PairChannel, singlet_triplet_fractions
from kamo.scattering.thresholds import K39Thresholds
from kamo.scattering.backends.mqdt import MQDTBackend


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def model():
    return ScatteringModel(B_max=600.0, dB=0.1)


@pytest.fixture(scope="module")
def th():
    return K39Thresholds(B_max_gauss=600.0, dB_gauss=0.1)


# ================================================================ INTERNAL
def test_hyperfine_splitting(th):
    # K39 4S1/2 zero-field splitting ~= 461.7 MHz (from kamo hyperfine A).
    assert abs(th.hyperfine_splitting_hz() - 461.7e6) < 1e6


def test_singlet_triplet_normalised():
    for MF in range(-4, 5):
        for c in ch.enumerate_channels(MF):
            fS, fT = singlet_triplet_fractions(c)
            assert abs(fS + fT - 1.0) < 1e-9


def test_stretched_states_pure_triplet():
    for c in (PairChannel((2, 2), (2, 2)), PairChannel((2, -2), (2, -2))):
        fS, fT = singlet_triplet_fractions(c)
        assert fS < 1e-12 and abs(fT - 1.0) < 1e-9


def test_known_singlet_fractions():
    # exact rationals from CG algebra
    assert abs(singlet_triplet_fractions(PairChannel((1, -1), (1, -1)))[0] - 3/16) < 1e-9
    assert abs(singlet_triplet_fractions(PairChannel((1, 0), (1, 0)))[0] - 1/4) < 1e-9


def test_frame_transform_unitary_and_matches_projection():
    mq = MQDTBackend()
    for MF in range(-4, 5):
        for c in ch.enumerate_channels(MF):
            v = mq.frame_transform_vector(c)
            assert abs(float(v @ v) - 1.0) < 1e-9              # unit norm
            fS_frame = mq.singlet_fraction_via_frame(c)
            fS_proj = singlet_triplet_fractions(c)[0]
            assert abs(fS_frame - fS_proj) < 1e-9              # two methods agree


def test_frame_rows_orthonormal():
    # distinct channels at the same M_F are orthogonal in the short-range basis
    mq = MQDTBackend()
    for MF in range(-4, 5):
        U, chans, _ = mq.frame_transform_matrix(MF)
        if len(chans) < 2:
            continue
        G = U @ U.T
        assert np.allclose(G, np.eye(len(chans)), atol=1e-9)


def test_vdw_lengths():
    # R_vdW = beta6/2 ~ 64.6 a0, abar ~ 61.8 a0 for K39 (C6 = 3921 a.u., D'Errico)
    beta6 = u.vdw_length_a0(3921.0)
    assert abs(beta6 / 2 - 64.6) < 0.2
    assert abs(u.mean_scattering_length_a0(3921.0) - 61.8) < 0.2


def test_target_channels_are_elastic(model):
    # second-order Zeeman => mF=0 lowest M_F=0 F=1 pair; the three target
    # channels have no open M_F-conserving loss channel -> not two-body lossy.
    for ent in [((1, -1), (1, -1)), ((1, 0), (1, 0)), ((1, -1), (1, 0))]:
        for B in (33.58, 60.0, 200.0):
            assert model.is_lossy(*ent, B) is False


def test_empirical_returns_real_for_target_channels(model):
    a = model.intra((1, -1), 200.0)
    assert np.isreal(a)


def test_empirical_pole_structure(model):
    # a diverges and flips sign across the |1,-1> resonance near 33.58 G
    a_lo = float(np.real(model.intra((1, -1), 33.4)))
    a_hi = float(np.real(model.intra((1, -1), 33.75)))
    assert abs(a_lo) > 100 and abs(a_hi) > 100
    assert np.sign(a_lo) != np.sign(a_hi)


def test_empirical_vectorised(model):
    B = np.linspace(50, 300, 200)
    a = model.intra((1, -1), B)
    assert a.shape == B.shape and np.all(np.isfinite(a))


def test_k2_zero_when_elastic():
    assert k2_from_scattering_length(-20.0 + 0j) == 0.0


def test_k2_positive_and_linear_in_loss():
    k1 = k2_from_scattering_length(-20.0 - 1.0j)
    k2 = k2_from_scattering_length(-20.0 - 2.0j)
    assert k1 > 0 and abs(k2 / k1 - 2.0) < 1e-9


def test_channel_key_order_independent():
    from kamo.scattering.data import k39_params as kp
    assert kp.channel_key((1, -1), (1, 0)) == kp.channel_key((1, 0), (1, -1))
    assert kp.background_channel((1, -1), (1, -1)) == kp.background_channel((1, -1), (1, -1))


def test_mqdt_gated():
    m = ScatteringModel(B_max=100.0, backend="mqdt")
    with pytest.raises(NotImplementedError):
        m.intra((1, -1), 30.0)


# =========================================================== GROUND-TRUTH
# Literature-verified (Etrych 2023 / Chapurin 2019 / Falke 2008 / D'Errico 2007).
def test_params_verified():
    from kamo.scattering.data import k39_params as kp
    assert kp.PARAMS_VERIFIED is True


def test_singlet_triplet_background():
    from kamo.scattering.data import k39_params as kp
    a_S, a_T, C6 = kp.singlet_triplet()
    assert abs(a_S - 138.49) < 0.12       # Falke 2008
    assert abs(a_T - (-33.48)) < 0.18
    assert abs(C6 - 3921.0) < 8.0         # D'Errico 2007


@pytest.mark.parametrize("channel,B0", [
    (((1, -1), (1, -1)), 33.5820),        # Chapurin anchor
    (((1, -1), (1, -1)), 162.36),
    (((1, -1), (1, -1)), 561.14),
    (((1, 1), (1, 1)), 402.74),
    (((1, 0), (1, 0)), 472.33),
])
def test_resonance_positions_are_poles(model, channel, B0):
    # |a| diverges within +-0.4 G of the tabulated position
    a = np.abs(np.real(model.scattering_length(*channel, B0 + 0.2)))
    assert a > 1000


def test_anchor_position_exact(model):
    res = model.resonances((1, -1), (1, -1))
    assert any(abs(r.B0_gauss - 33.5820) < 0.01 for r in res)   # Chapurin 33.5820(14)


@pytest.mark.parametrize("channel,B_zero", [
    (((1, 1), (1, 1)), 350.4),            # Fattori 2008
    (((1, -1), (1, -1)), 504.9),          # Etrych
    (((1, 0), (1, 0)), 393.2),
    (((1, 0), (1, 0)), 490.1),            # 1 G from the narrow 491.17 G pole
    (((1, 0), (1, 0)), 43.0),             # Roy 2013, 43(2)
])
def test_zero_crossings(model, channel, B_zero):
    from kamo.scattering.resonances import find_features
    _, zeros = find_features(lambda B: model.scattering_length(*channel, B),
                             B_zero - 1.0, B_zero + 1.0, dB=0.01)
    assert zeros and min(abs(z - B_zero) for z in zeros) < 0.1


def test_empirical_interstate_channel(model):
    # |1,0>+|1,-1> is tabulated now (Tanzi 2018 / Etrych 2023)
    a = model.inter((1, -1), (1, 0), np.array([113.70, 113.82]))
    assert np.sign(a[0]) != np.sign(a[1]) and np.all(np.abs(a) > 1000)


def test_empirical_lossy_channel_complex(model):
    a = model.inter((1, 1), (1, -1), 77.6)
    assert np.iscomplexobj(a) and a.imag < 0


def test_resonance_tools_on_analytic_model():
    from kamo.scattering.resonances import find_features, locate_pole, characterize
    # wide pole at 100 G (Delta=-40, zero at 60) + narrow one at 100.8 G
    # (Delta=-0.5) whose zero at 100.3 G is nearer the wide pole than its own
    # zero -- the 472/491 G |1,0> geometry
    abg = -30.0
    a = lambda B: abg * (1 - (-40.0) / (B - 100.0)) * (1 - (-0.5) / (B - 100.8))
    poles, zeros = find_features(a, 50.0, 120.0, dB=0.05)
    assert np.allclose(poles, [100.0, 100.8], atol=1e-5)
    assert np.allclose(zeros, [60.0, 100.3], atol=1e-5)
    assert abs(locate_pole(a, 99.5) - 100.0) < 1e-5
    assert abs(characterize(a, 100.0).zero_crossing - 100.3) < 1e-4    # nearest: the neighbour's
    fit = characterize(a, 100.0, side=-1)                              # sign(Delta) hint
    assert abs(fit.zero_crossing - 60.0) < 1e-4 and abs(fit.width + 40.0) < 1e-4
    assert abs(fit.pole_strength - abg * (-40.0) * (1 + 0.5 / (100.0 - 100.8))) < 0.1
    # lossy pole: residue must survive the inelastic smoothing
    gam = 0.01
    al = lambda B: -30.0 - 900.0 / (B - 80.0 - 0.5j * gam)
    assert abs(characterize(al, 80.0, zero_search=5.0).pole_strength - 900.0) < 20.0
    assert abs(locate_pole(al, 79.7, half_window=0.5) - 80.0) < 1e-4   # Re a is finite there


def test_cc_backend_constructs():
    # regression: _make_backend used `self` inside a staticmethod
    m = ScatteringModel(B_max=100.0, backend="cc")
    assert np.isfinite(m.intra((1, -1), 50.0))


def test_measured_database_consistency():
    from kamo.scattering.data import k39_feshbach as kf
    for r in kf.RESONANCES:
        assert r.B0_unc > 0 and r.partial_wave == "s"
        if r.B0_theory is not None:
            assert abs(r.B0_theory - r.B0) < 0.5
        if r.width_theory is not None and r.a_bg_theory is not None and r.pole_strength:
            # measured pole strength vs theory a_bg*Delta within 15%
            assert abs(r.pole_strength / (r.a_bg_theory * r.width_theory) - 1) < 0.15
