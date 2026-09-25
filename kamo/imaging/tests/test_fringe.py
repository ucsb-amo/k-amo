"""Tests for kamo.imaging.fringe: the fringe-effective intensity estimator."""

import numpy as np
import pytest

from kamo.imaging.fringe import fringe_effective, intensity_histogram, wrap_phase


def test_uniform_intensity_is_its_own_effective_intensity():
    I, C = fringe_effective(np.full(50, 1.3), np.ones(50), kappa=0.7)
    assert I == pytest.approx(1.3, abs=1e-12)
    assert C == pytest.approx(1.0, abs=1e-12)


def test_narrow_distribution_reduces_to_the_mean():
    rng = np.random.default_rng(0)
    v = 1.0 + 0.02 * rng.standard_normal(20000)
    I, C = fringe_effective(v, np.ones_like(v), kappa=1.5)
    assert I == pytest.approx(v.mean(), abs=2e-4)          # kappa sigma = 0.03: mean to O((k s)^2)
    assert C == pytest.approx(np.exp(-0.5 * (1.5 * 0.02) ** 2), abs=1e-3)


def test_two_point_distribution_matches_closed_form():
    # half the atoms at I = 1, half at I = 3, kappa = 1: z = (e^i + e^3i)/2
    z = 0.5 * (np.exp(1j) + np.exp(3j))
    I, C = fringe_effective([1.0, 3.0], [1.0, 1.0], kappa=1.0)
    assert I == pytest.approx(np.angle(z), abs=1e-12)
    assert C == pytest.approx(abs(z), abs=1e-12)


def test_heavy_tail_drops_out_of_the_phase_and_costs_contrast():
    # 95 % of atoms at I = 1, 5 % spread uniformly over many cycles: they cannot move the
    # phase by more than their weight and they cost ~5 % of contrast
    rng = np.random.default_rng(1)
    v = np.concatenate([np.ones(9500), rng.uniform(20, 200, 500)])
    I, C = fringe_effective(v, np.ones_like(v), kappa=1.0)
    assert abs(I - 1.0) < 0.06
    assert 0.90 < C < 0.97
    assert v.mean() > 5          # the plain mean is useless here


def test_vector_kappa_and_wrapping():
    I, C = fringe_effective([2.0], [1.0], kappa=np.array([0.5, 1.0, 2.0]))
    assert I.shape == (3,)
    # kappa I = 4 rad wraps into [-pi/2, 3pi/2): 4 - 2pi = -2.28 -> I_eff = -1.14
    assert I[2] == pytest.approx(wrap_phase(4.0) / 2.0)
    assert np.allclose(C, 1.0)


def test_intensity_histogram_reproduces_a_density_weighted_mean():
    """A synthetic 3D record with a known density gives the weighted mean of I exactly."""
    class Grid:
        def __init__(self, n):
            ax = np.linspace(-1, 1, n)
            self.Y, self.Z = np.meshgrid(ax, ax, indexing="ij")

    class Result:
        pass

    class Source:
        def __init__(self, dens):
            self._d = dens

        def density(self, x, Y, Z):
            return self._d[int(round(x))]

    n, ns = 12, 3
    rng = np.random.default_rng(2)
    I3 = rng.uniform(0.2, 2.5, size=(ns, n, n)).astype(np.float32)
    dens = rng.uniform(0, 1, size=(ns, n, n))
    res = Result()
    res.intensity_3d, res.window, res.grid, res.x_slices = I3, slice(0, n), Grid(n), np.arange(ns)
    edges = np.linspace(0, 3, 30001)
    hist, centres, overflow = intensity_histogram(res, Source(dens), edges)
    assert overflow == 0.0
    assert hist.sum() == pytest.approx(dens.sum(), rel=1e-12)
    mean_from_hist = (hist * centres).sum() / hist.sum()
    mean_exact = (dens * I3.astype(float)).sum() / dens.sum()
    assert mean_from_hist == pytest.approx(mean_exact, abs=1e-4)   # half a bin width
