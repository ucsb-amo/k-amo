"""Tests for kamo.trap.imaging_bridge and the kamo.spin / BECCloud hooks.

INTERNAL      effective widths (rms = Gaussian widths exactly; 0.535 R for a
              Thomas-Fermi cloud; containment); a Gaussian cloud gridded and
              driven through GriddedMixture reproduces UniformMixture, and the
              agreement improves as the solver grid is refined; mass
              conservation; the species fast path; slice mismatch; SpinGeometry
              takes the gridded density for a TrapCloud and is unchanged for a
              Gaussian; BECCloud now meets the imaging cloud contract.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, hbar

import kamo.constants as kc
from kamo.BEC_properties.variational import GaussianVariationalCloud
from kamo.imaging.bpm import Propagator, UniformMixture
from kamo.imaging.response import TwoLevelResponse
from kamo.trap.cloud import TrapCloud
from kamo.trap.grid import TrapGrid
from kamo.trap.imaging_bridge import (GriddedMixture, contained_fraction, effective_widths,
                                      propagator_for)
from kamo.trap.thomas_fermi import ThomasFermiSolver
from kamo.trap.trap import HarmonicTrap

M = 38.963706 * atomic_mass
RESPONSE = TwoLevelResponse(766.7e-9, 6.0309e6)
SPECIES = [(0.5, -18.28), (0.5, 18.28)]
F = (80.0, 1000.0, 1000.0)


def harmonic():
    w = 2 * np.pi * np.array(F)
    return HarmonicTrap((0.0, 0.0, 0.0), 0.0, np.diag(M * w ** 2), M)


def gaussian_cloud():
    return GaussianVariationalCloud(500.0, 2 * np.pi * np.array(F), 11.33 * kc.a0, mass=M)


def gridded(var, n):
    """The variational Gaussian sampled onto a TrapGrid, as a TrapCloud."""
    g = TrapGrid.around((0, 0, 0), 5.0 * var.widths, n)
    return TrapCloud(g, var.density(g.X, g.Y, g.Z), var.N, harmonic(), mode="gp",
                     chemical_potential_J=0.0, V_min_J=0.0, a_scattering=11.33 * kc.a0)


@pytest.fixture(scope="module")
def propagator():
    return Propagator.for_cloud(RESPONSE, gaussian_cloud(), n_grid=192, L_box=8e-6,
                                n_slices=60)


class TestWidths:
    def test_rms_widths_of_a_gaussian_are_its_widths(self):
        var = gaussian_cloud()
        assert effective_widths(gridded(var, (96, 48, 48))) == pytest.approx(var.widths, rel=1e-6)
        assert effective_widths(var) is not None

    def test_thomas_fermi_widths_and_containment(self):
        cloud = ThomasFermiSolver(harmonic(), a_scattering=100 * kc.a0).solve(2e4)
        R = cloud.tf_radii()
        assert effective_widths(cloud) == pytest.approx(np.sqrt(2 / 7) * R, rel=3e-3)
        assert contained_fraction(cloud, 3 * cloud.widths) == 1.0
        h = effective_widths(cloud, "containment")
        assert np.all(h <= 1.02 * R) and np.all(h > 0.9 * R)
        assert effective_widths(cloud, "peak")[0] < cloud.widths[0] * 1.5
        with pytest.raises(ValueError):
            effective_widths(cloud, "fwhm")


class TestGriddedMixture:
    def test_reproduces_uniform_mixture_and_converges(self, propagator):
        var = gaussian_cloud()
        ref = propagator.propagate(UniformMixture(var, RESPONSE, SPECIES), saturate=False)
        errors = []
        for n in ((48, 24, 24), (96, 48, 48)):
            src = GriddedMixture.for_propagator(propagator, gridded(var, n), RESPONSE, SPECIES)
            assert abs(src.atom_number_error) < 5e-3
            res = propagator.propagate(src, saturate=False)
            errors.append(np.max(np.abs(res.psi_exit - ref.psi_exit))
                          / np.max(np.abs(ref.psi_exit - 1)))
        assert errors[1] < 3e-3
        assert errors[1] < errors[0]                            # refinement helps

    def test_needs_species_and_the_right_slices(self, propagator):
        cloud = gridded(gaussian_cloud(), (48, 24, 24))
        with pytest.raises(ValueError, match="species"):
            GriddedMixture.for_propagator(propagator, cloud, RESPONSE, [])
        src = GriddedMixture.for_propagator(propagator, cloud, RESPONSE, SPECIES)
        with pytest.raises(ValueError, match="different propagator"):
            src.density(1.2345e-7, None, None)

    def test_propagator_for_warns_on_a_small_box(self):
        cloud = gridded(gaussian_cloud(), (48, 24, 24))
        with pytest.warns(UserWarning):
            propagator_for(cloud, RESPONSE, n_grid=64, L_box=1.0e-6, n_slices=10)


class TestSpinGeometry:
    def test_trap_cloud_density_is_used(self, propagator):
        from kamo.spin import SpinGeometry
        cloud = ThomasFermiSolver(harmonic(), a_scattering=11.33 * kc.a0).solve(500.0)
        geom = SpinGeometry.from_propagator(propagator, cloud)
        X, Y, Z = geom.x[:, None, None], geom.y[None, :, None], geom.z[None, None, :]
        c = cloud.centroid
        assert geom.density == pytest.approx(cloud.density(X + c[0], Y + c[1], Z + c[2]))
        w = geom.widths
        gauss = cloud.N / (np.pi ** 1.5 * np.prod(w)) * np.exp(
            -(X ** 2 / w[0] ** 2 + Y ** 2 / w[1] ** 2 + Z ** 2 / w[2] ** 2))
        assert not np.allclose(geom.density, gauss, rtol=0.05)

    def test_gaussian_cloud_is_unchanged(self, propagator):
        from kamo.spin import SpinGeometry
        var = gaussian_cloud()
        geom = SpinGeometry.from_propagator(propagator, var)
        X, Y, Z = geom.x[:, None, None], geom.y[None, :, None], geom.z[None, None, :]
        w = var.widths
        gauss = var.N / (np.pi ** 1.5 * np.prod(w)) * np.exp(
            -(X ** 2 / w[0] ** 2 + Y ** 2 / w[1] ** 2 + Z ** 2 / w[2] ** 2))
        assert np.array_equal(geom.density, gauss)


class TestBECCloud:
    def test_meets_the_imaging_contract(self):
        from kamo.dipole_dipole.cloud import BECCloud
        c = BECCloud(N=1e5, trap_frequencies_Hz=(150.0, 150.0, 20.0), a_s_bohr=100.0)
        R = c.tf_radii
        assert c.widths == pytest.approx(np.sqrt(2 / 7) * R)
        g = TrapGrid.around((0, 0, 0), 1.01 * R, (64, 64, 64))
        assert g.integrate(c.density(g.X, g.Y, g.Z)) == pytest.approx(c.N, rel=2e-3)
        assert c.density(0.0, 0.0, 0.0) == pytest.approx(c.peak_density)
        UniformMixture(c, RESPONSE, SPECIES)                    # accepted as a cloud
