"""Tests for kamo.trap.dvr, the sinc-DVR axis solver.

INTERNAL      the harmonic oscillator (exact to 1e-10), parity, the kinetic
              matrix, box logic and argument errors.
GROUND-TRUTH  the 1D Gaussian well of the K-team tweezer (w0 = 3 um, 1 kHz
              radial, 1064 nm), pinned to the printed outputs of
              analysis/atomic_physics/gaussian_well_wavefunctions.ipynb, which
              this module productizes.  Tolerances are the notebook's print
              rounding.

Pure numpy/scipy with an explicit mass -- no ARC, no network.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, g as G, h, hbar

from kamo.trap import dvr

M = 38.963706 * atomic_mass                 # the notebook's K-39 mass
W0 = 3.0e-6
OMEGA = 2 * np.pi * 1.0e3
V0 = M * OMEGA ** 2 * W0 ** 2 / 4           # depth from the 1 kHz radial frequency
E_W = hbar ** 2 / (M * W0 ** 2)

# Section 2 of the notebook: E_n / E_w, printed to 3 decimals.
NOTEBOOK_E = [-283.949, -250.777, -219.171, -189.181, -160.865, -134.290, -109.537,
              -86.700, -65.899, -47.286, -31.063, -17.522, -7.131, -0.835]


def well(u):
    return -V0 * np.exp(-2.0 * u ** 2 / W0 ** 2)


def tilted(u):
    """Vertical radial cut: gravity pulls toward -u (the notebook's section 7)."""
    return well(u) + M * G * u


@pytest.fixture(scope="module")
def flat():
    return dvr.solve_axis(well, length_scale=W0, mass=M)


@pytest.fixture(scope="module")
def notebook_grid():
    return dvr.solve_axis(well, half_width=8 * W0, n_grid=1601, mass=M)


@pytest.fixture(scope="module")
def vertical():
    return dvr.solve_axis(tilted, length_scale=W0, mass=M)


# --------------------------------------------------------------- INTERNAL

class TestInternal:
    def test_kinetic_matrix_symmetric_positive(self):
        T = dvr.sinc_dvr_kinetic(64, 1e-8, M)
        assert np.allclose(T, T.T, rtol=0, atol=0)
        assert np.all(np.linalg.eigvalsh(T) > 0)

    def test_harmonic_oscillator_is_exact(self):
        a = np.sqrt(hbar / (M * OMEGA))
        with pytest.warns(UserWarning, match="reach the edge"):   # the box-limited top states
            s = dvr.solve_axis(lambda u: 0.5 * M * OMEGA ** 2 * u ** 2,
                               half_width=10 * a, n_grid=401, mass=M)
        E = s.bound_energies[:11] / (hbar * OMEGA)
        assert E == pytest.approx(np.arange(11) + 0.5, abs=1e-10)
        assert s.frequency_Hz == pytest.approx(1.0e3, rel=1e-7)
        assert s.anharmonicity == pytest.approx(0.0, abs=1e-10)
        assert s.sigma == pytest.approx(a / np.sqrt(2), rel=1e-9)
        assert s.sigma_harmonic == pytest.approx(a / np.sqrt(2), rel=1e-7)

    def test_eigenfunctions_normalized(self, flat):
        assert np.sum(flat.psi ** 2, axis=0) * flat.du == pytest.approx(
            np.ones(flat.n_bound), abs=1e-12)

    def test_parity_alternates_in_a_symmetric_well(self, flat):
        for n in range(flat.n_bound):
            overlap = np.sum(flat.psi[::-1, n] * flat.psi[:, n]) * flat.du
            assert overlap == pytest.approx((-1) ** n, abs=1e-6)

    def test_needs_a_length_scale_or_explicit_grid(self):
        with pytest.raises(ValueError, match="length_scale"):
            dvr.solve_axis(well, mass=M)
        with pytest.raises(ValueError, match="n_grid"):
            dvr.solve_axis(well, bounds=(-W0, W0), mass=M)
        with pytest.raises(ValueError, match="not both"):
            dvr.solve_axis(well, bounds=(-W0, W0), half_width=W0, n_grid=11, mass=M)

    def test_gravity_beyond_critical_has_no_well(self):
        """4 g exceeds gamma_c = 2 eta / sqrt(e) (~3.7 g here): nothing to solve."""
        with pytest.raises(ValueError, match="no local minimum"):
            dvr.solve_axis(lambda u: well(u) + 4 * M * G * u, length_scale=W0, mass=M)

    def test_check_convergence(self, flat):
        out = dvr.check_convergence(well, flat)
        assert out["n_bound"] == (14, 14)
        assert out["max_dE"] / E_W < 1e-6


# ----------------------------------------------------------- GROUND-TRUTH

class TestNotebookUntilted:
    def test_dimensionless_numbers(self):
        eta = dvr.gaussian_eta(V0, W0, M)
        assert eta == pytest.approx(300.92, abs=0.005)
        assert dvr.waist_energy(W0, M) / h == pytest.approx(28.823, abs=5e-4)
        assert dvr.gravity_gamma(W0, M) == pytest.approx(99.67, abs=0.005)
        assert dvr.critical_gamma(eta) == pytest.approx(365.0, abs=0.5)
        assert dvr.gravity_gamma(W0, M) / dvr.critical_gamma(eta) == pytest.approx(0.273, abs=5e-4)

    def test_spectrum_on_the_notebook_grid(self, notebook_grid):
        assert notebook_grid.n_bound == 14
        assert notebook_grid.bound_energies / E_W == pytest.approx(NOTEBOOK_E, abs=5e-4)

    def test_default_box_matches_the_notebook_grid(self, flat, notebook_grid):
        assert flat.n_bound == 14
        assert np.max(np.abs(flat.bound_energies - notebook_grid.bound_energies)) / E_W < 1e-6

    def test_small_box_loses_the_top_state_unless_grown(self, notebook_grid):
        """The notebook's L = 6 row finds 13 states; the edge test is right to
        drop the weakly bound 14th, and growing the box recovers it."""
        with pytest.warns(UserWarning, match="reach the edge"):
            small = dvr.solve_axis(well, half_width=6 * W0, n_grid=1201, mass=M)
        assert small.n_bound == 13 and small.n_unresolved == 1
        dE = np.abs(small.bound_energies - notebook_grid.bound_energies[:13])
        assert np.max(dE) / E_W < 1e-9
        grown = dvr.solve_axis(well, half_width=6 * W0, n_grid=1201, mass=M, auto_box=True)
        assert grown.n_bound == 14 and grown.box_growths >= 1

    def test_departure_from_harmonic(self, flat):
        assert flat.frequency_Hz == pytest.approx(1.0e3, rel=1e-6)
        assert flat.anharmonicity == pytest.approx(-0.0108, abs=5e-5)          # -1.08%
        assert flat.spacings_over_hbar_omega[0] == pytest.approx(0.9561, abs=6e-5)  # -4.39%
        assert flat.spacings[-1] / h == pytest.approx(181.0, abs=0.5)          # 0.181 kHz
        assert flat.sigma > flat.sigma_harmonic                               # softened well


class TestNotebookGravity:
    def test_sag_and_escape_lip(self, vertical):
        assert vertical.has_lip and vertical.u_lip < vertical.u_min
        assert vertical.u_min / W0 == pytest.approx(-0.0840, abs=5e-5)
        assert vertical.u_lip / W0 == pytest.approx(-1.146, abs=5e-4)
        assert vertical.E_escape / h == pytest.approx(-3919.0, abs=0.5)
        assert vertical.usable_depth / h == pytest.approx(4870.0, abs=5.0)

    def test_trapped_states_and_frequency(self, vertical):
        assert vertical.n_bound == 6
        assert np.all(vertical.bound_energies < vertical.E_escape)
        assert vertical.frequency_Hz == pytest.approx(978.9, abs=0.05)
        assert vertical.ground_energy / h == pytest.approx(-8316.0, abs=0.5)

    def test_gravity_unbinds_the_top_of_the_well(self, flat, vertical):
        assert vertical.n_bound < flat.n_bound
        assert vertical.frequency_Hz < flat.frequency_Hz
