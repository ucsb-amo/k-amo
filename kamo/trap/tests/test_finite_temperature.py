"""Tests for kamo.trap.finite_temperature: the hybrid finite-temperature solver.

INTERNAL      the momentum-integral kernel against the Bose function, a direct
              quadrature, the closed-form Boltzmann tail and the infinite-escape
              cap; the product spectrum on a harmonic trap (analytic ladder, the
              exact Mehler density) and on a rotated one (both density paths agree);
              the calibrated separable basis of the real tweezer (the axial ladder
              rescaled by the Born-Oppenheimer factor, radial axes untouched);
              the ideal harmonic oracle: N_0(T) against the exact finite-N series,
              converging with e_cut; g -> 0 reduces the GP path to the ideal gas;
              T -> 0 reproduces the ground-state cloud and T_K = 0 is the old path;
              the components, moments, column densities and normalization audit;
              the closure (every level above mu; mu(N_0) reduces to the ideal
              relation); the attractive state; the gates warn and never raise on
              eta; the basin leak regression and the point budget; solve_all and
              comparison_table at T > 0; the spectrum cache across a T sweep.
GROUND-TRUTH  (slow) the K-team operating point, N = 500, a = 11.33 a0, at 30.4 nK
              (eta 6.8) and 45.6 nK (eta 4.6): condensate fraction, thermal widths,
              passes, the numbers that justify the one-way default; the reference
              T_c values.  Frozen 2026-09-13 from the first run of this solver.

Run: pytest kamo/trap/tests -q            (the tweezer class: -m slow)
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pytest
from scipy.constants import atomic_mass
from scipy.integrate import quad
from scipy.special import gammainc

import kamo.constants as kc
from kamo.BEC_properties.thermal import (IdealHarmonicBoseGas, bose_g, harmonic_thermal_density,
                                         ideal_harmonic_condensate, critical_temperature_K)
from kamo.BEC_properties.variational import CollapseError
from kamo.trap.beams import Tweezer
from kamo.trap.cloud import TrapCloud
from kamo.trap.finite_temperature import (FiniteTemperatureResult,
                                          FiniteTemperatureSolver, ModelValidityWarning,
                                          ProductSpectrum, critical_temperature,
                                          semiclassical_density,
                                          temperature_from_condensate_fraction, weyl_count)
from kamo.trap.grid import TrapGrid, basin_mask, touches_face
from kamo.trap.gross_pitaevskii import GPResult
from kamo.trap.solvers import canonical_mode, comparison_table, solve, solve_all
from kamo.trap.trap import HarmonicTrap, Trap

M = 38.963706 * atomic_mass
W0, LAM = 3.0e-6, 1064e-9
ALPHA = 599.3005266591443 * kc.convert_polarizability_au_to_SI
A_OP = 11.33 * kc.a0
F_LAB = (78.706, 992.97, 978.87)          # the lab tweezer's principal frequencies, lab-axis order
N = 500.0
FAST = dict(points_per_scale=2.0)         # condensate solver options for the harmonic tests
HARMONIC = dict(condensate_options=FAST, n_thermal_widths=4.5)   # keeps the harmonic boxes small


def harmonic(f_Hz=F_LAB, rotate=False):
    w = 2 * np.pi * np.asarray(f_Hz, dtype=float)
    H = np.diag(M * w ** 2)
    if rotate:
        th = 0.3
        R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        H = R @ H @ R.T
    return HarmonicTrap((0.0, 0.0, 0.0), 0.0, H, M)


def tweezer_trap():
    t = Tweezer.from_trap_frequency(1e3, polarizability_SI=ALPHA, mass=M, waist=W0, wavelength_m=LAM)
    return Trap(t, polarizability_SI=ALPHA, mass=M, gravity=True)


def ideal_solver(trap, **kw):
    opts = dict(condensate="noninteracting", **HARMONIC)
    opts.update(kw)
    return FiniteTemperatureSolver(trap, a_scattering=0.0, **opts)


def quiet_solve(solver, N_, T):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return solver.solve(N_, T)


# ------------------------------------------------------------------ kernel

class TestKernel:
    T, m = 60e-9, M

    def test_matches_the_bose_function(self):
        kT = kc.kB * self.T
        lam = kc.h / np.sqrt(2 * np.pi * self.m * kT)
        d = np.array([0.0, 0.5, 2.0, 5.0, 12.0])
        n = semiclassical_density(np.zeros(5), d * kT, 0.0, self.T, self.m)
        assert n * lam ** 3 == pytest.approx(bose_g(np.exp(-d), 1.5), rel=1e-11)
        assert n[0] * lam ** 3 == pytest.approx(2.6123753486854883, rel=1e-13)

    def test_truncated_against_a_direct_quadrature(self):
        kT = kc.kB * self.T
        lam = kc.h / np.sqrt(2 * np.pi * self.m * kT)
        for d in (0.05, 0.5, 3.0):
            for s_max in (1.0, 2.0, 4.0):
                V_esc = s_max ** 2 * kT                     # V = 0, V_eff = d kT, escape at s_max
                n = float(semiclassical_density(0.0, d * kT, 0.0, self.T, self.m, V_escape=V_esc))
                ref, _ = quad(lambda s: s * s / np.expm1(s * s + d), 0.0, s_max, epsrel=1e-12)
                assert n * lam ** 3 == pytest.approx(4 / np.sqrt(np.pi) * ref, rel=1e-8)

    def test_boltzmann_is_the_incomplete_gamma_closed_form(self):
        kT = kc.kB * self.T
        lam = kc.h / np.sqrt(2 * np.pi * self.m * kT)
        eta, v = 8.0, 1.3                                   # z e^{-v} P(3/2, eta - v) with z = 1
        n = float(semiclassical_density(v * kT, v * kT, 0.0, self.T, self.m, V_escape=eta * kT,
                                        statistics="boltzmann"))
        assert n * lam ** 3 == pytest.approx(np.exp(-v) * gammainc(1.5, eta - v), rel=1e-10)

    def test_infinite_escape_is_capped_not_nan(self):
        n = semiclassical_density(np.zeros(3), np.array([0.0, 1.0, 50.0]) * kc.kB * self.T, 0.0,
                                  self.T, self.m, V_escape=np.inf)
        assert np.all(np.isfinite(n)) and n[0] > n[1] > n[2] >= 0.0

    def test_e_cut_makes_a_tail(self):
        kT = kc.kB * self.T
        full = float(semiclassical_density(0.0, 2.0 * kT, 0.0, self.T, self.m))
        tail = float(semiclassical_density(0.0, 2.0 * kT, 0.0, self.T, self.m, E_cut=1.5 * kT))
        assert 0.0 < tail < full


# ------------------------------------------------------- product spectrum

class TestProductSpectrum:
    def test_harmonic_ladder_is_analytic(self):
        trap = harmonic()
        spec = ProductSpectrum.harmonic(trap)
        w = 2 * np.pi * np.array(F_LAB)
        assert spec.E_0 == pytest.approx(0.5 * kc.hbar * w.sum(), rel=1e-14)
        E, q = spec.levels(spec.E_0 + 2.5 * kc.hbar * w.max())
        wk = np.array([lad.omega for lad in spec.ladders])       # principal order (descending)
        assert E == pytest.approx(spec.E_0 + kc.hbar * (q @ wk), rel=1e-12)
        assert np.all(np.diff(E) >= -1e-30)

    def test_density_matches_the_mehler_kernel(self):
        trap = harmonic()
        spec = ProductSpectrum.harmonic(trap)
        w = np.array([lad.omega for lad in spec.ladders])
        T = 50e-9
        kT = kc.kB * T
        mu = -0.5 * kc.hbar * w.min()
        g = TrapGrid.around((0, 0, 0), (25e-6, 2.2e-6, 2.2e-6), (120, 30, 30))
        E_max = spec.E_0 + 14 * kT
        E, q = spec.levels(E_max)
        occ = 1.0 / np.expm1((E - spec.E_0 - mu) / kT)
        occ[0] = 0.0
        rho = spec.density(g, E_max, occ)
        u = spec.coordinates(g)                                    # 1D arrays (aligned axes)
        u3 = []
        for k, (i_lab, _) in enumerate(spec.aligned()):
            shape = [1, 1, 1]
            shape[i_lab] = -1
            u3.append(np.broadcast_to(u[k].reshape(shape), g.shape))
        ref = harmonic_thermal_density(u3[0], u3[1], u3[2], T, mu, w, M)
        assert np.max(np.abs(rho - ref)) < 2e-3 * ref.max()
        assert g.integrate(rho) == pytest.approx(occ.sum(), rel=2e-3)

    def test_rotated_axes_take_the_general_path(self):
        F = (300.0, 400.0, 500.0)
        spec_a = ProductSpectrum.harmonic(harmonic(F))
        spec_r = ProductSpectrum.harmonic(harmonic(F, rotate=True))
        assert spec_a.aligned() is not None and spec_r.aligned() is None
        E_max = spec_a.E_0 + 1.5 * kc.hbar * 2 * np.pi * max(F)
        Ea, _ = spec_a.levels(E_max)
        Er, _ = spec_r.levels(E_max)
        assert Ea == pytest.approx(Er, rel=1e-9)
        g = TrapGrid.around((0, 0, 0), (6e-6, 6e-6, 6e-6), (56, 56, 56))
        occ = np.exp(-(Ea - spec_a.E_0) / (kc.kB * 40e-9))
        occ[0] = 0.0
        W = np.broadcast_to(np.exp(-(g.X ** 2 + g.Y ** 2 + g.Z ** 2) / (2 * (1e-6) ** 2)), g.shape)
        # a rotation about z leaves N and <z^2> and any isotropic-in-xy expectation unchanged
        Na = g.integrate(spec_a.density(g, E_max, occ))
        Nr = g.integrate(spec_r.density(g, E_max, occ))
        assert Na == pytest.approx(occ.sum(), rel=3e-3) and Nr == pytest.approx(Na, rel=3e-3)
        ea = spec_a.diagonal_expectation(g, E_max, W)
        er = spec_r.diagonal_expectation(g, E_max, W)
        assert ea[0] == pytest.approx(er[0], rel=5e-3)

    def test_level_cap(self):
        spec = ProductSpectrum.harmonic(harmonic())
        with pytest.raises(ValueError, match="cap"):
            spec.levels(spec.E_0 + 400 * kc.hbar * 2 * np.pi * max(F_LAB))
        with pytest.raises(ValueError):
            spec.levels(np.inf)


@pytest.fixture(scope="module")
def tweezer_spectrum():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ProductSpectrum.from_trap(tweezer_trap())


class TestCalibration:
    """The real tweezer: the axial ladder is 6% too stiff in the separable picture and
    the 4-state 3D solve calibrates it; the radial axes are beyond its reach and stay
    at 1 (their Born-Oppenheimer correction is negligible)."""

    def test_axial_factor_is_the_born_oppenheimer_one(self, tweezer_spectrum):
        cal = tweezer_spectrum.calibration
        soft = int(np.argmin([lad.eps[1] for lad in tweezer_spectrum.ladders]))
        assert cal.factors[soft] == pytest.approx(0.9377, abs=2e-3)
        assert cal.bo_prediction == pytest.approx(0.9406, abs=2e-3)
        assert cal.matched_states[soft] == 1 and cal.overlaps[soft] > 0.9
        for k in range(3):
            if k != soft:
                assert cal.factors[k] == 1.0 and k in cal.defaulted_axes
        assert tweezer_spectrum.separability_index > 1.0

    def test_anchor_and_counts(self, tweezer_spectrum):
        spec = tweezer_spectrum
        trap = tweezer_trap()
        V_min = trap.minimum().potential_J
        assert (spec.E_0 - V_min) / kc.h == pytest.approx(991.9, abs=1.0)
        V_esc = V_min + trap.trap_depth_J()
        assert 150 < spec.count_below(V_esc) < 260
        E_cut = spec.E_0 + 2 * kc.hbar * 2 * np.pi * max(F_LAB)
        assert 45 < spec.count_below(E_cut) < 80


# --------------------------------------------------------- harmonic oracle

@pytest.fixture(scope="module")
def ideal_curve():
    """N_0(T) of the ideal gas in the lab-frequency harmonic trap at three temperatures
    and three seams, once."""
    trap = harmonic()
    w = 2 * np.pi * np.array(F_LAB)
    T0 = critical_temperature_K(N, w)
    out = {}
    for t in (0.3, 0.6):
        for e_cut in (1.0, 2.0, 3.0):
            s = ideal_solver(trap, e_cut_hbar_omega=e_cut)
            out[(t, e_cut)] = quiet_solve(s, N, t * T0)
    return T0, out


class TestHarmonicOracle:
    def test_condensate_number_against_the_exact_series(self, ideal_curve):
        T0, clouds = ideal_curve
        w = 2 * np.pi * np.array(F_LAB)
        for t in (0.3, 0.6):
            N_0, mu = ideal_harmonic_condensate(N, t * T0, w)
            errs = [abs(clouds[(t, e)].N_0 / N_0 - 1) for e in (1.0, 2.0, 3.0)]
            assert errs[1] < 5e-3                                     # the default seam
            assert clouds[(t, 2.0)].chemical_potential - clouds[(t, 2.0)].info.E_0_J == pytest.approx(mu, rel=5e-3)

    def test_the_seam_does_not_matter_much(self, ideal_curve):
        """The hybrid's spread over e_cut is its model uncertainty: sub-percent here."""
        T0, clouds = ideal_curve
        for t in (0.3, 0.6):
            vals = [clouds[(t, e)].N_0 for e in (1.0, 2.0, 3.0)]
            assert max(vals) / min(vals) - 1 < 5e-3
            assert clouds[(t, 3.0)].info.quantum_fraction > clouds[(t, 1.0)].info.quantum_fraction

    def test_closure_reduces_to_the_ideal_relation(self, ideal_curve):
        T0, clouds = ideal_curve
        c = clouds[(0.3, 2.0)]
        kT = kc.kB * 0.3 * T0
        assert c.info.mu_GP_J == pytest.approx(c.info.E_0_J, rel=1e-12)
        assert c.chemical_potential == pytest.approx(c.info.E_0_J - kT * np.log1p(1 / c.N_0), rel=1e-12)
        hw_min = kc.hbar * 2 * np.pi * min(F_LAB)
        assert c.info.min_level_margin == pytest.approx((hw_min + kT * np.log1p(1 / c.N_0)) / kT, rel=1e-6)
        assert c.info.passes == 1 and c.info.converged and c.info.condensate_model == "noninteracting"

    def test_boltzmann_limit(self):
        F = (300.0, 400.0, 500.0)
        trap = harmonic(F)
        w = 2 * np.pi * np.array(F)
        T = 20 * critical_temperature_K(N, w)
        sig = np.sqrt(kc.kB * T / (M * w ** 2))            # a classical cloud: pin a coarse box
        g = TrapGrid.around((0, 0, 0), 5.0 * sig, 48)
        s = ideal_solver(trap, thermal="boltzmann", grid=g)
        with pytest.warns(ModelValidityWarning):
            c = s.solve(N, T)
        assert c.condensate_fraction == 0.0 and c.info.condensate_model == "none"
        kT = kc.kB * T
        assert c.sigma_thermal == pytest.approx(np.sqrt(kT / (M * w ** 2)), rel=2e-3)
        mu_classical = kT * np.log(N * (kc.hbar * float(np.prod(w)) ** (1 / 3) / kT) ** 3)
        assert c.chemical_potential_offset == pytest.approx(mu_classical, rel=1e-3)
        assert c.atom_number_error == pytest.approx(0.0, abs=1e-6)

    def test_lda_is_the_semiclassical_answer_and_warns(self):
        trap = harmonic()
        w = 2 * np.pi * np.array(F_LAB)
        T = 0.5 * critical_temperature_K(N, w)
        with pytest.warns(ModelValidityWarning, match="lda"):
            c = ideal_solver(trap, thermal="lda").solve(N, T)
        ref = IdealHarmonicBoseGas(N, w, T, mass=M)
        assert c.N_th == pytest.approx(ref.N_th, rel=0.02)         # the textbook saturated LDA
        exact = ideal_harmonic_condensate(N, T, w)[0]
        assert c.N_0 > exact                                          # ... which under-counts the thermal cloud


# --------------------------------------------------------------- reductions

class TestReductions:
    def test_g_to_zero_is_the_ideal_gas(self):
        trap = harmonic((300.0, 400.0, 500.0))
        T = 0.4 * critical_temperature_K(N, 2 * np.pi * np.array([300.0, 400.0, 500.0]))
        gp = quiet_solve(FiniteTemperatureSolver(trap, a_scattering=0.0, condensate="gp", **HARMONIC),
                         N, T)
        ni = quiet_solve(ideal_solver(trap), N, T)
        assert gp.condensate_fraction == pytest.approx(ni.condensate_fraction, rel=1e-8)
        assert gp.density_thermal == pytest.approx(ni.density_thermal, rel=1e-8, abs=1e-8 * ni.peak_density_thermal)

    def test_T_to_zero_reproduces_the_ground_state_cloud(self):
        trap = harmonic((300.0, 400.0, 500.0))
        cold = solve(trap, N, "gp", a_scattering=50 * kc.a0, **FAST)
        warm = quiet_solve(FiniteTemperatureSolver(trap, a_scattering=50 * kc.a0, **HARMONIC), N, 1e-12)
        assert warm.condensate_fraction > 1 - 1e-9
        assert warm.info.mu_GP_J == pytest.approx(cold.chemical_potential, rel=1e-7)
        assert warm.sigma_condensate == pytest.approx(cold.sigma, rel=1e-6)

    def test_T_K_zero_is_the_old_path(self):
        cloud = solve(harmonic((300.0, 400.0, 500.0)), N, "gp", a_scattering=50 * kc.a0, T_K=0.0, **FAST)
        assert isinstance(cloud.info, GPResult)
        assert cloud.density_thermal is None and cloud.condensate_fraction == 1.0
        assert cloud.density_condensate is cloud.density_grid
        assert not cloud.is_finite_temperature and cloud.model_label == "gp"
        with pytest.raises(ValueError, match="T = 0"):
            cloud.density(0.0, 0.0, 0.0, component="thermal")


# -------------------------------------------------------------- bookkeeping

@pytest.fixture(scope="module")
def warm():
    trap = harmonic((300.0, 400.0, 500.0))
    T = 0.4 * critical_temperature_K(N, 2 * np.pi * np.array([300.0, 400.0, 500.0]))
    s = FiniteTemperatureSolver(trap, a_scattering=50 * kc.a0, **HARMONIC)
    return quiet_solve(s, N, T), s


class TestBookkeeping:
    def test_components_sum_and_audit(self, warm):
        c, _ = warm
        g = c.grid
        assert c.density_condensate + c.density_thermal == pytest.approx(c.density_grid)
        assert c.N_0 + c.N_th == pytest.approx(N, rel=1e-12)
        assert abs(c.atom_number_error) < 1e-5 and abs(c.info.normalization_error) < 1e-5
        assert g.integrate(c.density_condensate) == pytest.approx(c.N_0, rel=1e-6)
        assert 0.0 < c.condensate_fraction < 1.0 and c.is_finite_temperature
        col = c.column_density_grid("condensate") + c.column_density_grid("thermal")
        assert col == pytest.approx(c.column_density_grid(), rel=1e-12)
        assert np.all(c.sigma_thermal > c.sigma_condensate)
        assert c.sigma_condensate == pytest.approx(np.sqrt(g.moments(c.density_condensate)[2]))
        assert c.density(0, 0, 0, component="condensate") + c.density(0, 0, 0, component="thermal") \
            == pytest.approx(c.density(0, 0, 0), rel=1e-9)

    def test_result_and_summary(self, warm):
        c, _ = warm
        info = c.info
        assert isinstance(info, FiniteTemperatureResult) and info.converged
        assert len(info.residual_history) == info.passes and info.residual_history[-1] == min(info.residual_history)
        assert info.passes <= 4 and info.warm_start_fallbacks == 0
        assert info.min_level_margin > 0 and 0.0 < info.quantum_fraction < 1.0
        assert np.isinf(info.eta) and info.truncation_fraction == 0.0
        assert abs(info.normalization_error) < 1e-6 and info.edge_fraction_thermal < 1e-4
        assert not info.basin_face_contact                    # (4.5 widths: the 1e-6 edge gate is not met)
        assert "nK" in c.summary() and "N0/N" in repr(c) and "thermal" in c.model_label
        assert "passes" in info.summary()
        assert c.harmonic_reference().N_0 > 0 and c.harmonic_reference(0.0).N == N
        assert c.healing_length == pytest.approx(1 / np.sqrt(8 * np.pi * c.peak_density_condensate * c.a_scattering))

    def test_solve_all_and_table(self, warm):
        c, _ = warm
        trap = harmonic((300.0, 400.0, 500.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clouds = solve_all(trap, N, a_scattering=50 * kc.a0, T_K=c.T_K,
                               options={"gp": HARMONIC, "noninteracting": HARMONIC})
        assert isinstance(clouds["thomas-fermi"], ValueError)
        assert isinstance(clouds["gp"], TrapCloud) and clouds["gp"].T_K == c.T_K
        table = comparison_table(clouds)
        assert "T (nK)" in table and "N0/N" in table
        assert "T (nK)" not in comparison_table({"gp": solve(trap, N, "gp", a_scattering=50 * kc.a0, **FAST)})

    def test_spectrum_and_grid_are_reused_across_temperatures(self, warm):
        c, s = warm
        spec = s.spectrum()
        c2 = quiet_solve(s, N, 0.8 * c.T_K)                    # a colder cloud fits the same box
        assert s.spectrum() is spec and c2.condensate_fraction > c.condensate_fraction
        assert c2.grid is c.grid and c2.info.warm_start_fallbacks == 0
        c3 = quiet_solve(s, N, 1.15 * c.T_K)                   # a hotter one needs a bigger box
        assert c3.grid.shape != c.grid.shape and c3.condensate_fraction < c.condensate_fraction

    def test_above_the_crossover_is_thermal_only(self):
        trap = harmonic((300.0, 400.0, 500.0))
        w = 2 * np.pi * np.array([300.0, 400.0, 500.0])
        T = 1.6 * critical_temperature_K(N, w)
        c = quiet_solve(ideal_solver(trap, n_thermal_widths=4.0), N, T)
        assert c.condensate_fraction < 1e-3 and "thermal only" in c.model_label
        assert c.info.condensate_model == "ideal-ground-state"
        assert c.chemical_potential < c.info.E_0_J


# ---------------------------------------------------------- closure, guards

class TestGuards:
    def test_attractive_thermal_cloud_and_collapse_on_N0(self):
        trap = harmonic((300.0, 400.0, 500.0))
        w = 2 * np.pi * np.array([300.0, 400.0, 500.0])
        a = -73.70 * kc.a0
        var_crit = FiniteTemperatureSolver(trap, a_scattering=a, **HARMONIC)
        from kamo.BEC_properties.variational import GaussianVariationalCloud
        n_crit = GaussianVariationalCloud(100.0, w, a, mass=M).critical_atom_number()
        T = 0.5 * critical_temperature_K(0.6 * n_crit, w)
        c = quiet_solve(var_crit, 0.6 * n_crit, T)
        assert 0.0 < c.condensate_fraction < 1.0 and c.info.min_level_margin > 0
        with pytest.raises(CollapseError):
            quiet_solve(FiniteTemperatureSolver(trap, a_scattering=a, **HARMONIC), 1.5 * n_crit, 0.05 * T)

    def test_modes_and_arguments(self):
        trap = harmonic((300.0, 400.0, 500.0))
        with pytest.raises(ValueError, match="thomas-fermi"):
            solve(trap, N, "thomas-fermi", a_scattering=50 * kc.a0, T_K=50e-9)
        with pytest.raises(ValueError, match="T_K"):
            canonical_mode("hartree-fock")
        with pytest.raises(ValueError, match="thermal"):
            FiniteTemperatureSolver(trap, thermal="exact")
        with pytest.raises(ValueError, match="T_K"):
            solve(trap, N, "gp", a_scattering=50 * kc.a0, T_K=-1e-9)
        with pytest.raises(ValueError, match="T_K > 0"):
            solve(trap, N, "gp", a_scattering=50 * kc.a0, thermal="lda")
        with pytest.raises(ValueError, match="e_cut"):
            FiniteTemperatureSolver(trap, e_cut_hbar_omega=0.2)
        with pytest.raises(TypeError):
            FiniteTemperatureSolver(trap, points_per_scale=3.0)

    def test_point_budget(self):
        trap = harmonic((300.0, 400.0, 500.0))
        s = FiniteTemperatureSolver(trap, a_scattering=0.0, condensate="noninteracting",
                                    n_max_total=1000, condensate_options=FAST)
        with pytest.raises(ValueError, match="n_max_total"):
            s.solve(N, 50e-9)


# -------------------------------------------------------- the real tweezer

class TestGrid:
    def test_basin_leak_regression(self):
        """At exactly the saddle energy the mask leaks through the saddle and runs down
        the gravity slope to the box face; the epsilon cut bounds it."""
        trap = tweezer_trap()
        mn = trap.minimum()
        V_esc = mn.potential_J + trap.trap_depth_J()
        g = TrapGrid.around(mn.position, (36e-6, 2.5e-6, 3.5e-6), (201, 41, 61))
        V = np.broadcast_to(np.asarray(trap.potential_J(g.X, g.Y, g.Z), float), g.shape)
        leaky = basin_mask(V, g, mn.position, V_esc)
        bounded = basin_mask(V, g, mn.position, V_esc, epsilon=1e-2)
        assert not touches_face(bounded)
        assert bounded.sum() <= leaky.sum()
        vol = bounded.sum() * g.dV
        assert 380e-18 < vol < 560e-18                        # ~518 um^3 measured on a finer grid
        assert weyl_count(V_esc, V[bounded], g.dV, M) == pytest.approx(301.0, rel=0.05)


class _OperatingPoint(NamedTuple):
    solver: FiniteTemperatureSolver
    c30: TrapCloud
    c45: TrapCloud
    messages: list


@pytest.fixture(scope="module")
def tweezer_30nK():
    s = FiniteTemperatureSolver(tweezer_trap(), a_scattering=A_OP,
                                condensate_options=dict(points_per_scale=3.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c30 = s.solve(N, 30.4e-9)
        c45 = s.solve(N, 45.6e-9)
    return _OperatingPoint(s, c30, c45, [str(x.message) for x in w])


@pytest.mark.slow
class TestOperatingPoint:
    """N = 500, a = +11.33 a0, the 1064 nm / 3 um / 1 kHz tweezer with gravity.
    Frozen 2026-09-13 from this solver's first run (grid 310x42x54, e_cut = 2):
    N0/N = 0.944 at 30.4 nK (eta 6.84), 0.884 at 45.6 nK (eta 4.56); the exact
    truncated harmonic sum of the round tables gave 0.946 and 0.896."""

    def test_thirty_nanokelvin(self, tweezer_30nK):
        c = tweezer_30nK.c30
        assert c.info.converged and c.info.passes <= 4
        assert c.eta == pytest.approx(6.84, abs=0.05)
        assert c.condensate_fraction == pytest.approx(0.944, abs=0.01)
        assert 0.85 < c.info.quantum_fraction < 0.95
        assert c.info.calibration.factors[2] == pytest.approx(0.9377, abs=2e-3)
        assert c.sigma_thermal[0] > 4e-6 and 0.4e-6 < c.sigma_thermal[1] < 0.55e-6
        assert c.sigma_condensate == pytest.approx([1.734e-6, 0.382e-6, 0.387e-6], rel=0.02)
        kT = kc.kB * c.T_K
        assert 2 * c.coupling_g * c.peak_density_thermal / kT < 0.02   # the one-way default's justification
        assert abs(c.info.normalization_error) < 1e-5
        assert c.info.trustworthy

    def test_forty_five_nanokelvin_warns(self, tweezer_30nK):
        c30, c, msgs = tweezer_30nK.c30, tweezer_30nK.c45, tweezer_30nK.messages
        assert c.eta == pytest.approx(4.56, abs=0.05)
        assert c.condensate_fraction == pytest.approx(0.884, abs=0.015)
        assert c.condensate_fraction < c30.condensate_fraction
        assert 0.1 < c.info.truncation_fraction < 0.25
        assert any("eta" in m for m in msgs)
        assert c.info.passes <= 3

    def test_thermal_wings_are_the_imaging_systematic(self, tweezer_30nK):
        c = tweezer_30nK.c30
        frac_col = float(np.max(c.column_density_grid("thermal"))) / c.peak_column_density
        assert c.thermal_fraction > 0.04 and frac_col < c.thermal_fraction   # wings, not a peak

    def test_critical_temperature_reference(self):
        ref = critical_temperature(tweezer_trap(), N, a_scattering=A_OP)
        assert ref.T_c_ideal_nK == pytest.approx(152.08, rel=2e-3)
        assert ref.T_c_finite_size_nK == pytest.approx(129.6, rel=3e-3)
        assert 135 < ref.T_c_exact_series_nK < 150
        assert ref.T_c_interaction_shift == pytest.approx(-0.0029, abs=3e-4)
        assert ref.eta_at_T_c == pytest.approx(1.37, abs=0.02)
        assert "no sharp transition" in ref.summary()

    def test_temperature_from_fraction_round_trips(self, tweezer_30nK):
        T = temperature_from_condensate_fraction(tweezer_trap(), N, 0.90, solver=tweezer_30nK.solver,
                                                 bracket_K=(28e-9, 48e-9), rtol=2e-2)
        assert 30e-9 < T < 46e-9
