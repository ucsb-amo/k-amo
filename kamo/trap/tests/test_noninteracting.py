"""Tests for kamo.trap.noninteracting.

INTERNAL      a harmonic trap (E0 = sum hbar omega / 2, sigma = sqrt(hbar / 2 m omega),
              gap = hbar omega_min, all to 1e-8 or better -- sinc-DVR is exact
              for it); a separable sum of Gaussian wells, whose 3D ground energy
              must equal the sum of the 1D sinc-DVR ground energies; the real
              tweezer with gravity -- softer than harmonic, radial degeneracy
              broken by the sag; the per-axis spectra against the
              gaussian_well notebook.  Explicit mass and polarizability.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.constants import atomic_mass, h, hbar

import kamo.constants as kc
from kamo.trap import dvr
from kamo.trap.beams import Tweezer
from kamo.trap.noninteracting import NonInteractingSolver
from kamo.trap.trap import HarmonicTrap, Trap, TrapMinimum, _frequencies_from_hessian

M = 38.963706 * atomic_mass
W0, LAM = 3.0e-6, 1064e-9
ALPHA = 599.3005266591443 * kc.convert_polarizability_au_to_SI


def harmonic(f_Hz):
    w = 2 * np.pi * np.asarray(f_Hz, dtype=float)
    return HarmonicTrap((0.0, 0.0, 0.0), -1e-30, np.diag(M * w ** 2), M)


class SumOfGaussians:
    """V = -sum_i V_i exp(-2 x_i^2 / w_i^2): separable, so 3D = sum of 1D."""

    def __init__(self, V, w):
        self.V, self.w, self.mass = np.asarray(V, float), np.asarray(w, float), M

    def potential_J(self, X, Y, Z):
        return -sum(Vi * np.exp(-2 * c ** 2 / wi ** 2) for Vi, wi, c in zip(self.V, self.w, (X, Y, Z)))

    def minimum(self):
        return TrapMinimum(np.zeros(3), -float(self.V.sum()), True)

    def trap_frequencies(self):
        return _frequencies_from_hessian(np.diag(4 * self.V / self.w ** 2), np.zeros(3), M)

    def trap_depth_J(self):
        return float(self.V.min())


def tweezer_trap(gravity=True):
    t = Tweezer.from_trap_frequency(1e3, polarizability_SI=ALPHA, mass=M, waist=W0,
                                    wavelength_m=LAM)
    return Trap(t, polarizability_SI=ALPHA, mass=M, gravity=gravity)


@pytest.fixture(scope="module")
def harmonic_cloud():
    return NonInteractingSolver(harmonic((80.0, 1000.0, 700.0)), n_per_axis=32).solve(10.0)


@pytest.fixture(scope="module")
def tweezer_cloud():
    return NonInteractingSolver(tweezer_trap()).solve()


class TestHarmonic:
    def test_ground_state_is_exact(self, harmonic_cloud):
        cloud = harmonic_cloud
        w = 2 * np.pi * np.array([80.0, 1000.0, 700.0])
        info = cloud.info
        assert info.zero_point_J == pytest.approx(0.5 * hbar * w.sum(), rel=1e-8)
        assert info.anharmonicity == pytest.approx(0.0, abs=1e-8)
        assert info.gap_J == pytest.approx(hbar * w.min(), rel=1e-7)
        assert cloud.sigma == pytest.approx(np.sqrt(hbar / (2 * M * w)), rel=1e-6)
        assert cloud.atom_number_error == pytest.approx(0.0, abs=1e-12)
        assert info.separability_index < 1e-12

    def test_cloud_bookkeeping(self, harmonic_cloud):
        cloud = harmonic_cloud
        assert cloud.mode == "noninteracting" and cloud.N == 10.0
        assert cloud.chemical_potential == cloud.info.energies_3d[0]
        assert np.isnan(cloud.healing_length)
        norms = np.sum(cloud.info.psi_3d ** 2, axis=(1, 2, 3)) * cloud.grid.dV
        assert norms == pytest.approx(np.ones(4), rel=1e-12)

    def test_axis_spectra_are_the_oscillator(self):
        solver = NonInteractingSolver(harmonic((80.0, 1000.0, 700.0)))
        w = np.sort(2 * np.pi * np.array([80.0, 1000.0, 700.0]))[::-1]
        for spec, wi in zip(solver.axis_spectra(n_grid_max=801), w):
            assert spec.ground_energy - spec.V_min == pytest.approx(0.5 * hbar * wi, rel=1e-8)


class TestSeparable:
    def test_3d_ground_energy_is_the_sum_of_1d(self):
        V = h * np.array([6e3, 9e3, 12e3])
        w = np.array([3.0e-6, 2.5e-6, 2.0e-6])
        trap = SumOfGaussians(V, w)
        info = NonInteractingSolver(trap, n_per_axis=40).solve().info
        with warnings.catch_warnings():        # a near-threshold top state is box-limited
            warnings.simplefilter("ignore", UserWarning)
            E1d = sum(dvr.solve_axis(lambda u, Vi=Vi, wi=wi: -Vi * np.exp(-2 * u ** 2 / wi ** 2),
                                     length_scale=wi, mass=M).ground_energy
                      for Vi, wi in zip(V, w))
        assert info.energies_3d[0] == pytest.approx(E1d, rel=1e-8)
        assert info.separability_index < 1e-10
        assert info.anharmonicity < 0                        # softer than harmonic


class TestRealTweezer:
    def test_softer_than_harmonic(self, tweezer_cloud):
        cloud = tweezer_cloud
        info = cloud.info
        assert -0.06 < info.anharmonicity < -0.03             # a few percent below the ladder
        assert info.residual < 1e-6

    def test_transverse_zero_point_softens_the_axial_confinement(self, tweezer_cloud):
        """The first excitation is axial, ~7% below the Hessian's axial frequency.
        Born-Oppenheimer: the transverse zero-point energy hbar (w_y + w_z) / 2 scales
        with the local depth V0(s) = V0 / (1 + s^2 / zR^2), like the potential, so it
        cancels part of the axial well: w_eff = w_ax sqrt(1 - hbar (w_y + w_z) / 2 V0)."""
        trap = tweezer_cloud.trap
        w = 2 * np.pi * trap.trap_frequencies().frequencies_Hz          # (y, z, x)
        V0 = -float(trap.light_potential_J(*trap.minimum().position))
        w_eff = w[2] * np.sqrt(1 - hbar * (w[0] + w[1]) / (2 * V0))
        gap = tweezer_cloud.info.gap_J
        assert 0.90 < gap / (hbar * w[2]) < 0.95
        assert gap / (hbar * w_eff) == pytest.approx(1.0, abs=0.015)

    def test_gravity_breaks_the_radial_degeneracy(self, tweezer_cloud):
        cloud = tweezer_cloud
        sig = cloud.sigma
        assert 1.003 < sig[2] / sig[1] < 1.03                  # vertical wider than horizontal
        assert cloud.centroid[2] < cloud.trap.minimum().position[2]   # hangs below the minimum
        assert abs(cloud.centroid[1]) < 1e-12
        harm = np.sqrt(hbar / (2 * M * cloud.trap.trap_frequencies().omega))  # axes y, z, x
        assert np.all(sig[[1, 2, 0]] > harm)                   # every width above harmonic

    def test_not_separable(self, tweezer_cloud):
        cloud = tweezer_cloud
        """The transverse potential is a product of Gaussians, not a sum."""
        assert cloud.info.separability_index > 1e-3

    def test_horizontal_axis_spectrum_is_the_notebook(self):
        """Without gravity the radial cut through the focus is the notebook's
        eta = 300.92 well: 14 bound states, E0 = -283.949 E_w."""
        specs = NonInteractingSolver(tweezer_trap(gravity=False)).axis_spectra(n_grid_max=1601)
        radial = specs[0]
        E_w = hbar ** 2 / (M * W0 ** 2)
        assert radial.n_bound == 14
        assert radial.ground_energy / E_w == pytest.approx(-283.949, abs=5e-4)
