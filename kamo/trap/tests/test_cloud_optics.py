"""cloud_optics against the closed forms of a Gaussian cloud."""

import numpy as np
import pytest
from scipy.special import erfc

from kamo.imaging import readout
from kamo.imaging.grid import TransverseGrid
from kamo.imaging.response import TwoLevelResponse
from kamo.trap import cloud_optics as co
from kamo.trap.cloud import TrapCloud
from kamo.trap.grid import TrapGrid

W = np.array([2.6e-6, 0.55e-6, 0.60e-6])        # 1/e density radii
N = 500.0
RESPONSE = TwoLevelResponse(766.700921e-9, 6.035e6)


@pytest.fixture(scope="module")
def cloud():
    grid = TrapGrid.around((0, 0, 0), 4.5 * W, (221, 91, 91))
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    n = N / (np.pi ** 1.5 * np.prod(W)) * np.exp(-(X / W[0]) ** 2 - (Y / W[1]) ** 2 - (Z / W[2]) ** 2)
    return TrapCloud(grid, n, N, None, mode="gp", chemical_potential_J=0.0, V_min_J=0.0)


def _escape_gaussian(tau):
    """<exp(-OD)> of a Gaussian cloud of chord optical depth tau (1D quadrature)."""
    zeta, wz = np.polynomial.legendre.leggauss(400)
    zeta, wz = 7 * zeta, 7 * wz * np.exp(-(7 * zeta) ** 2)
    K = 0.5 * tau * erfc(zeta)
    return float(((-np.expm1(-K) / K) * wz).sum() / np.sqrt(np.pi))


def test_thin_screen_and_chords(cloud):
    ncol = N / (np.pi * W[1] * W[2])
    assert cloud.peak_column_density == pytest.approx(ncol, rel=2e-3)
    D, phi = co.thin_screen(cloud, RESPONSE, [(1.0, 18.26)])
    sigma = RESPONSE.sigma0 / (1 + 18.26 ** 2)
    assert D == pytest.approx(sigma * ncol, rel=2e-3)
    assert phi == pytest.approx(-0.5 * 18.26 * D, rel=1e-9)
    n0 = N / (np.pi ** 1.5 * np.prod(W))
    for axis in range(3):
        assert co.chord_optical_depth(cloud, sigma, axis) == pytest.approx(
            sigma * n0 * np.sqrt(np.pi) * W[axis], rel=2e-3)
    # max |d/dr exp(-r^2/w^2)| = sqrt(2/e) / w, on the tighter axis
    assert co.peak_column_gradient(cloud) == pytest.approx(ncol * np.sqrt(2 / np.e) / W[1], rel=1e-2)


def test_form_factor_out_to_backscatter(cloud):
    k = RESPONSE.k
    nhat = np.array([[1, 0, 0], [0.95, 0.3122, 0], [0.9, 0, 0.4359], [0, 1, 0], [-1, 0, 0]], float)
    q = k * (nhat - np.array([1.0, 0, 0]))
    exact = np.exp(-((q * W) ** 2).sum(-1) / 2)
    got = co.form_factor_sq(cloud, q)
    assert np.allclose(got[:3], exact[:3], rtol=2e-3)
    assert np.all(got[3:] < 1e-9)                       # side and back scatter: nothing coherent


def test_born_far_field_matches_the_analytic_gaussian(cloud):
    grid = TransverseGrid(256, 24e-6, RESPONSE.k)
    numeric = co.born_far_field(grid, cloud)
    analytic = readout.born_far_field(grid, W, N)
    m = analytic > analytic.max() * 1e-4
    # 0.4% at the first decade growing to 1.1% at the fourth, independent of the optical
    # grid: it is the linear resampling of the solver grid (GriddedDensity), whose
    # smoothing grows as (q d)^2.
    assert np.allclose(numeric[m] / numeric.max(), analytic[m] / analytic.max(), rtol=1.5e-2)


def test_escape_probability_depends_only_on_tau_for_a_gaussian(cloud):
    n0 = N / (np.pi ** 1.5 * np.prod(W))
    sigma = 2.0 / (n0 * np.sqrt(np.pi) * W[0])          # tau_x = 2
    nhat = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]])
    got = co.escape_probability(cloud, sigma, nhat, n_u=5, n_phi=5, step=0.12e-6)
    m = np.sqrt(((nhat / np.linalg.norm(nhat, axis=1, keepdims=True)) ** 2 / W ** 2).sum(-1))
    want = [_escape_gaussian(sigma * n0 * np.sqrt(np.pi) / mi) for mi in m]
    assert np.allclose(got[:3], want[:3], rtol=1e-2)
    assert got[3] == pytest.approx(want[3], rel=3e-2)   # interpolated between nodes


def test_stretched_holds_the_column(cloud):
    thin = co.stretched(cloud, 0.1)
    assert thin.peak_column_density == pytest.approx(cloud.peak_column_density, rel=1e-12)
    assert thin.widths[0] == pytest.approx(0.1 * cloud.widths[0], rel=1e-9)
    assert thin.peak_density == pytest.approx(10 * cloud.peak_density, rel=1e-12)


def test_stretched_carries_the_finite_temperature_components(cloud):
    n_c, n_t = 0.7 * cloud.density_grid, 0.3 * cloud.density_grid
    warm = TrapCloud(cloud.grid, n_c + n_t, N, None, mode="gp", chemical_potential_J=0.0, V_min_J=0.0,
                     T_K=50e-9, density_condensate=n_c, density_thermal=n_t)
    thin = co.stretched(warm, 0.5)
    assert np.allclose(thin.density_condensate + thin.density_thermal, thin.density_grid, rtol=1e-12)
    for part, frac in (("condensate", 0.7), ("thermal", 0.3)):
        assert np.max(thin.column_density_grid(part)) == pytest.approx(frac * warm.peak_column_density, rel=1e-9)
