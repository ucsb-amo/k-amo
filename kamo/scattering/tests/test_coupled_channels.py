"""Fast smoke/limit tests for the coupled-channels engine.

Explicit C12 values are passed to skip the (slow, one-time) potential tuning;
a coarse grid keeps these quick.  Physics-accuracy (resonance positions) is a
known model-potential limitation and is NOT asserted here.
"""

import numpy as np
import pytest

from kamo.scattering.coupled_channels import CoupledChannels, _spin_projection_matrices
from kamo.scattering import channels as ch

# shallow depth_rank=0 tunings (skip tuning); coarse grid for speed
_KW = dict(B_max=1000.0, r_in=9.0, r_out=1000.0, h=0.05,
           C12_S=1.90e12, C12_T=3.00e12)


@pytest.fixture(scope="module")
def cc():
    return CoupledChannels(**_KW)


def test_spin_matrices_consistent():
    chans = ch.enumerate_channels(-2)
    P_S, P_T, Gram = _spin_projection_matrices(chans)
    assert np.allclose(Gram, np.eye(len(chans)), atol=1e-9)      # channels orthonormal
    assert np.allclose(P_S + P_T, Gram, atol=1e-12)              # P_S + P_T = I
    # diagonal P_S equals each channel's singlet fraction
    for i, c in enumerate(chans):
        fS = ch.singlet_triplet_fractions(c)[0]
        assert abs(P_S[i, i] - fS) < 1e-9


def test_intra_elastic_real(cc):
    a = cc.scattering_length((1, -1), (1, -1), 250.0)
    assert np.isfinite(a) and abs(a.imag) < 1e-6      # single open channel -> real


def test_inter_channel_runs(cc):
    # empirical backend cannot do this channel; CC can
    a = cc.scattering_length((1, -1), (1, 0), 250.0)
    assert np.isfinite(a)


def test_high_field_triplet_limit():
    # |1,-1>+|1,-1> -> pure triplet at high field; a -> a_T of the model potential
    from kamo.scattering import potentials as P
    a_T_model = P.scattering_length_1ch(P.C6_AU, 3.00e12)      # the C12_T used
    cc = CoupledChannels(**_KW)
    a = cc.scattering_length((1, -1), (1, -1), 950.0).real
    assert abs(a - a_T_model) < 15.0                  # approaching triplet limit
