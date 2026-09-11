"""Tests for the coupled-channels engine.

Legacy Lennard-Jones tests pass explicit C12 values (skipping the slow
tuning) on a coarse grid.  The Tiemann tests check the calibrated model
against measured resonance positions and other groups' coupled-channels
values.
"""

import numpy as np
import pytest

from kamo.scattering.coupled_channels import (CoupledChannels, _spin_projection_matrices,
                                              segmented_grid)
from kamo.scattering import channels as ch
from kamo.scattering.thresholds import K39Thresholds

# shallow depth_rank=0 tunings (skip tuning); coarse grid for speed
_KW = dict(B_max=1000.0, r_in=9.0, r_out=1000.0, h=0.05,
           C12_S=1.90e12, C12_T=3.00e12, potentials='lj')


@pytest.fixture(scope="module")
def th():
    return K39Thresholds(B_max_gauss=1000.0, dB_gauss=0.05)


@pytest.fixture(scope="module")
def cc(th):
    return CoupledChannels(**_KW, thresholds=th)


@pytest.fixture(scope="module")
def cc_tm(th):
    return CoupledChannels(thresholds=th)            # calibrated Tiemann (default)


def test_spin_matrices_consistent():
    chans = ch.enumerate_channels(-2)
    P_S, P_T, Gram = _spin_projection_matrices(chans)
    assert np.allclose(Gram, np.eye(len(chans)), atol=1e-9)      # channels orthonormal
    assert np.allclose(P_S + P_T, Gram, atol=1e-12)              # P_S + P_T = I
    for i, c in enumerate(chans):
        fS = ch.singlet_triplet_fractions(c)[0]
        assert abs(P_S[i, i] - fS) < 1e-9


def test_segmented_grid_doubles_steps():
    r, i0, hs = segmented_grid(4.5, 2500.0, 0.003)
    assert np.allclose(hs[1:] / hs[:-1], 2.0)
    for s in range(1, len(i0)):
        assert abs((r[i0[s] + 1] - r[i0[s]]) - hs[s]) < 1e-9
    assert r[-1] >= 2500.0 and hs[-1] <= 0.4


def test_step_doubling_is_exact(th):
    # where and how often the step doubles must not change the answer.
    # (A fully uniform tiny-step grid is NOT a good reference: ~3e5 steps of
    # R = 1 + h/r in the flat tail accumulate ~1e-5 relative rounding error.)
    kw = dict(thresholds=th, delta_S=0.0, delta_T=0.0, r_out=1200.0, h=0.004)
    seg = CoupledChannels(**kw)                                  # doubles from 30 a0 to 0.256
    alt = CoupledChannels(**kw, r_fine=120.0, h_max=0.016)       # doubles from 120 a0 to 0.016
    assert len(alt.r) > 5 * len(seg.r)
    assert abs(seg.single_channel_a('S') - alt.single_channel_a('S')) < 1e-4
    assert abs(seg.single_channel_a('T') - alt.single_channel_a('T')) < 1e-4
    a_seg = seg.scattering_length((1, 0), (1, 0), 150.0).real
    a_alt = alt.scattering_length((1, 0), (1, 0), 150.0).real
    assert abs(a_seg - a_alt) < 1e-4


def test_intra_elastic_real(cc):
    a = cc.scattering_length((1, -1), (1, -1), 250.0)
    assert np.isfinite(a) and abs(a.imag) < 1e-6      # single open channel -> real


def test_inter_channel_runs(cc):
    a = cc.scattering_length((1, -1), (1, 0), 250.0)
    assert np.isfinite(a)


def test_high_field_triplet_limit(cc):
    # |1,-1>+|1,-1> -> pure triplet at high field; a -> a_T of the model potential
    from kamo.scattering import potentials as P
    a_T_model = P.scattering_length_1ch(P.C6_AU, 3.00e12)
    a = cc.scattering_length((1, -1), (1, -1), 950.0).real
    assert abs(a - a_T_model) < 15.0


# ------------------------------------------------------------- Tiemann model
def test_stretched_state_is_triplet(cc_tm):
    a_T = cc_tm.single_channel_a('T')
    assert abs(cc_tm.scattering_length((2, 2), (2, 2), 300.0).real - a_T) < 1e-3


@pytest.mark.parametrize("state_a,state_b,B0,tol", [
    ((1, -1), (1, -1), 33.5820, 0.05),     # Chapurin 2019
    ((1, 1), (1, 1), 402.74, 0.05),        # Etrych 2023
    ((1, 0), (1, -1), 113.76, 0.05),       # Tanzi 2018
])
def test_calibrated_resonance_positions(cc_tm, state_a, state_b, B0, tol):
    from kamo.scattering.resonances import locate_pole
    fn = lambda B: cc_tm.scattering_length(state_a, state_b, B).real
    assert abs(locate_pole(fn, B0, half_window=0.3, dB=0.05) - B0) < tol


def test_interstate_matches_published_cc(cc_tm):
    # Lavoine 2021 (Tiemann 2020 potentials): a12 = -53.2 a0 at 56.830 G
    a12 = cc_tm.scattering_length((1, -1), (1, 0), 56.830).real
    assert abs(a12 - (-53.2)) < 0.5


def test_lossy_channel_complex_with_loss_sign(cc_tm):
    # |2,0>+|2,0> can spin-relax into F=1 pairs: a = a_re - i a_im, a_im > 0
    a = cc_tm.scattering_length((2, 0), (2, 0), 100.0)
    assert a.imag < 0


def test_set_scattering_lengths_roundtrip(th):
    c = CoupledChannels(thresholds=th, delta_S=0.0, delta_T=0.0)
    c.set_scattering_lengths(138.7, -33.40)
    a_S, a_T = c.singlet_triplet_a()
    assert abs(a_S - 138.7) < 1e-4 and abs(a_T + 33.40) < 1e-4
