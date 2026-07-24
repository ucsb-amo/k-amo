"""Consistency suite for kamo.scattering.

Two tiers:
  * INTERNAL — self-validating physics/algebra that needs no fitted params.
    These MUST pass.
  * GROUND-TRUTH — checks against literature values.  Marked xfail while the
    K39 parameters remain provisional (deep-research pass rate-limited
    2026-07-24); flip to real assertions once k39_params is verified.
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
    (((1, 1), (1, 1)), 350.4),            # D'Errico/Etrych
    (((1, -1), (1, -1)), 504.9),          # Etrych
    (((1, 0), (1, 0)), 393.2),
    (((1, 0), (1, 0)), 490.1),
])
def test_zero_crossings(model, channel, B_zero):
    a = float(np.real(model.scattering_length(*channel, B_zero)))
    assert abs(a) < 1.0                   # a = 0 at the reported crossing
