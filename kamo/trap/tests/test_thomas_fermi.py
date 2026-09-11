"""Tests for kamo.trap.grid, kamo.trap.cloud and kamo.trap.thomas_fermi.

INTERNAL      TrapGrid construction and moments; the Thomas-Fermi solver against
              the harmonic closed forms (mu, n0 = mu/g, integral n^2, mean density,
              radii, rms widths R/sqrt(7)) at two very different N / omega -- the
              discretization error is scale-invariant; the V_extra seam; argument
              errors; the real tweezer (flatter than harmonic, TF invalid at
              N = 500, too shallow for 1e7 atoms).  Explicit mass,
              polarizability and scattering length: no ARC, no network.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, h, hbar

import kamo.constants as kc
from kamo.trap.beams import Tweezer
from kamo.trap.cloud import TrapTooShallowError
from kamo.trap.grid import TrapGrid
from kamo.trap.thomas_fermi import ThomasFermiSolver
from kamo.trap.trap import HarmonicTrap, Trap

M = 38.963706 * atomic_mass
W0, LAM = 3.0e-6, 1064e-9
ALPHA = 599.3005266591443 * kc.convert_polarizability_au_to_SI
A_OP = 11.33 * kc.a0                 # |1,-1> near 520.6 G, as in kamo.imaging's tests


def harmonic(f_Hz):
    w = 2 * np.pi * np.asarray(f_Hz, dtype=float)
    return HarmonicTrap((0.0, 0.0, 0.0), 0.0, np.diag(M * w ** 2), M)


def closed_form(N, a, f_Hz):
    w = 2 * np.pi * np.asarray(f_Hz, dtype=float)
    wbar = np.prod(w) ** (1 / 3)
    mu = 0.5 * hbar * wbar * (15 * N * a / np.sqrt(hbar / (M * wbar))) ** 0.4
    return mu, np.sqrt(2 * mu / (M * w ** 2))


def tweezer_trap():
    t = Tweezer.from_trap_frequency(1e3, polarizability_SI=ALPHA, mass=M, waist=W0,
                                    wavelength_m=LAM)
    return Trap(t, polarizability_SI=ALPHA, mass=M, gravity=True)


# ------------------------------------------------------------------ grid

class TestGrid:
    def test_midpoint_box(self):
        g = TrapGrid.around((1e-6, 0, 0), (2e-6, 1e-6, 1e-6), (8, 4, 4))
        assert g.shape == (8, 4, 4)
        assert np.allclose(g.center, [1e-6, 0, 0])
        assert np.allclose(g.half_widths, [2e-6, 1e-6, 1e-6])
        assert np.all(np.abs(g.x - 1e-6) > 0)                   # no node on the centre
        assert g.dV == pytest.approx(np.prod(g.d))

    def test_moments_of_a_gaussian(self):
        g = TrapGrid.around((0, 0, 0), 10e-6, (96, 96, 96))   # tails < 1e-20 at the faces
        s = np.array([1.0e-6, 0.7e-6, 0.5e-6])
        c = np.array([0.3e-6, -0.2e-6, 0.1e-6])
        n = np.exp(-((g.X - c[0]) ** 2 / (2 * s[0] ** 2) + (g.Y - c[1]) ** 2 / (2 * s[1] ** 2)
                     + (g.Z - c[2]) ** 2 / (2 * s[2] ** 2)))
        N, cen, var = g.moments(n)
        assert N == pytest.approx((2 * np.pi) ** 1.5 * np.prod(s), rel=1e-10)
        assert cen == pytest.approx(c, abs=1e-15)
        assert np.sqrt(var) == pytest.approx(s, rel=1e-10)

    def test_validation(self):
        with pytest.raises(ValueError, match="uniform"):
            TrapGrid([0, 1, 3], [0, 1], [0, 1])
        with pytest.raises(ValueError):
            TrapGrid.around(0, -1.0, 8)


# --------------------------------------------------------- Thomas-Fermi

class TestHarmonicClosedForm:
    @pytest.mark.parametrize("N, a, f", [(1e6, 100 * kc.a0, (150.0, 150.0, 20.0)),
                                         (500.0, A_OP, (79.8, 1000.0, 1000.0))])
    def test_matches_the_closed_form(self, N, a, f):
        cloud = ThomasFermiSolver(harmonic(f), a_scattering=a).solve(N)
        mu, R = closed_form(N, a, f)
        g = 4 * np.pi * hbar ** 2 * a / M
        n0 = mu / g
        assert cloud.chemical_potential_offset == pytest.approx(mu, rel=1e-5)
        assert cloud.peak_density == pytest.approx(cloud.chemical_potential_offset / g, rel=1e-14)
        assert cloud.atom_number_error == pytest.approx(0.0, abs=1e-12)
        assert cloud.density_squared_integral == pytest.approx(
            32 * np.pi / 105 * n0 ** 2 * np.prod(R), rel=5e-5)
        assert cloud.mean_density == pytest.approx(4 * n0 / 7, rel=5e-5)
        assert cloud.tf_radii() == pytest.approx(R, rel=5e-6)
        assert cloud.sigma == pytest.approx(R / np.sqrt(7), rel=2e-3)
        assert cloud.widths == pytest.approx(np.sqrt(2 / 7) * R, rel=2e-3)   # 0.535 R
        assert cloud.centroid == pytest.approx(np.zeros(3), abs=1e-12 * R.max())
        assert cloud.grid.face_max(cloud.density_grid) == 0.0

    def test_density_is_the_closed_form(self):
        f, N, a = (150.0, 150.0, 20.0), 1e6, 100 * kc.a0
        cloud = ThomasFermiSolver(harmonic(f), a_scattering=a).solve(N)
        mu = cloud.chemical_potential
        g = 4 * np.pi * hbar ** 2 * a / M
        w = 2 * np.pi * np.array(f)
        x = np.linspace(-1, 1, 7) * 1e-5
        expect = np.clip((mu - 0.5 * M * w[0] ** 2 * x ** 2) / g, 0, None)
        assert cloud.density(x, 0.0, 0.0) == pytest.approx(expect, rel=1e-12, abs=1e-6 * expect.max())

    def test_V_extra_shifts_mu_not_the_density(self):
        ht, a = harmonic((150.0, 150.0, 20.0)), 100 * kc.a0
        base = ThomasFermiSolver(ht, a_scattering=a).solve(1e6)
        shift = 1e-31
        moved = ThomasFermiSolver(ht, a_scattering=a).solve(
            1e6, V_extra=lambda X, Y, Z: shift + 0 * X * Y * Z)
        assert moved.chemical_potential - base.chemical_potential == pytest.approx(shift, rel=1e-9)
        assert moved.density_grid == pytest.approx(base.density_grid, rel=1e-9, abs=1e-6 * base.peak_density)

    @pytest.mark.parametrize("a, match", [(-10 * kc.a0, "mode='gp'"), (0.0, "noninteracting")])
    def test_no_tf_limit(self, a, match):
        with pytest.raises(ValueError, match=match):
            ThomasFermiSolver(harmonic((100, 100, 100)), a_scattering=a).solve(1e4)

    def test_harmonic_trap_needs_a_scattering_length(self):
        with pytest.raises(ValueError, match="a_scattering"):
            ThomasFermiSolver(harmonic((100, 100, 100))).solve(1e4)


@pytest.fixture(scope="module")
def cloud():
    return ThomasFermiSolver(tweezer_trap(), a_scattering=A_OP).solve(500.0)


class TestRealTweezer:
    def test_anharmonic_well_holds_a_lower_mu(self, cloud):
        mu_h = cloud.info.mu_offset_harmonic_J
        assert 0.9 < cloud.chemical_potential_offset / mu_h < 1.0
        assert cloud.harmonic_reference().N == 500.0

    def test_tf_is_not_valid_at_the_operating_point(self, cloud):
        assert not cloud.thomas_fermi_valid
        omega_max = np.max(cloud.trap.trap_frequencies().omega)
        assert cloud.chemical_potential_offset / (hbar * omega_max) < 1.0

    def test_cloud_sits_below_the_focus(self, cloud):
        assert cloud.centroid[2] < 0 and abs(cloud.centroid[1]) < 1e-12
        assert cloud.atom_number_error == pytest.approx(0.0, abs=1e-12)
        assert "thomas-fermi" in cloud.summary()

    def test_too_many_atoms_for_the_depth(self):
        with pytest.raises(TrapTooShallowError, match="escape"):
            ThomasFermiSolver(tweezer_trap(), a_scattering=100 * kc.a0).solve(1e7)
