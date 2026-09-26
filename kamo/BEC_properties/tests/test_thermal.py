"""Tests for kamo.BEC_properties.thermal: Bose functions, the finite-N series and
IdealHarmonicBoseGas.

INTERNAL   bose_g against mpmath's polylog on both branches (series and Robinson)
           and at z = 1; the finite-N series against a brute-force sum over product
           states; T_c closed forms and the reported shifts; the condensate-fraction
           curve and its N -> large limit; the classical (Boltzmann) limit; both
           component densities integrate to their atom numbers; the Mehler thermal
           density against the series; continuity at T = 0; the inverse question;
           the bimodal column density; the imaging cloud contract (N, widths,
           density).  Explicit mass; no trap, no portal.

Run: pytest kamo/BEC_properties/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass

import kamo.constants as kc
from kamo.BEC_properties.thermal import (IdealHarmonicBoseGas, ZETA_3, ZETA_3_2, bose_g,
                                         critical_temperature_K, finite_size_shift,
                                         harmonic_thermal_density, ideal_harmonic_atom_number,
                                         ideal_harmonic_condensate, mean_field_shift)

M = 38.963706 * atomic_mass
OMEGA = 2 * np.pi * np.array([78.706, 992.97, 978.87])     # the lab tweezer's principal frequencies (Hz)
N = 500.0


class TestBoseFunction:
    Z = np.array([0.0, 1e-8, 1e-3, 0.1, 0.3, 0.36787, 0.368, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 1.0])

    @pytest.mark.parametrize("order", [0.5, 1.5, 2.0, 2.5, 3.0, 4.0])
    def test_against_mpmath(self, order):
        mp = pytest.importorskip("mpmath")
        mp.mp.dps = 30
        got = bose_g(self.Z, order)
        for z, g in zip(self.Z, got):
            if z == 1.0:
                ref = float(mp.zeta(order)) if order > 1 else np.inf
            else:
                ref = float(mp.polylog(order, z))
            if np.isinf(ref):
                assert np.isinf(g)
            else:
                tol = 1e-8 if order in (2.0, 3.0) else 1e-12
                assert g == pytest.approx(ref, rel=tol, abs=1e-300)

    def test_unity(self):
        assert float(bose_g(1.0, 1.5)) == pytest.approx(ZETA_3_2, rel=1e-15)
        assert float(bose_g(1.0, 3.0)) == pytest.approx(ZETA_3, rel=1e-12)

    def test_domain(self):
        with pytest.raises(ValueError):
            bose_g(1.5, 1.5)
        with pytest.raises(ValueError):
            bose_g(0.5, 1.0)


class TestSeries:
    def test_series_matches_a_brute_force_sum(self):
        T, mu = 60e-9, -0.7 * kc.hbar * OMEGA[0]
        got = ideal_harmonic_atom_number(T, mu, OMEGA)
        n1 = np.arange(400)[:, None, None]
        n2 = np.arange(60)[None, :, None]
        n3 = np.arange(60)[None, None, :]
        E = kc.hbar * (OMEGA[0] * n1 + OMEGA[1] * n2 + OMEGA[2] * n3)
        brute = np.sum(1.0 / np.expm1((E - mu) / (kc.kB * T)))
        assert got == pytest.approx(brute, rel=1e-10)
        thermal = ideal_harmonic_atom_number(T, mu, OMEGA, thermal_only=True)
        assert got - thermal == pytest.approx(1.0 / np.expm1(-mu / (kc.kB * T)), rel=1e-10)

    def test_condensate_closes_the_budget(self):
        T = 60e-9
        N_0, mu = ideal_harmonic_condensate(N, T, OMEGA)
        assert mu < 0
        total = 1.0 / np.expm1(-mu / (kc.kB * T)) + ideal_harmonic_atom_number(T, mu, OMEGA, thermal_only=True)
        assert total == pytest.approx(N, rel=1e-12)
        assert 0.7 < N_0 / N < 0.9                            # 402 of 500 at 60 nK, measured

    def test_mehler_density_matches_the_series(self):
        T, mu = 60e-9, -0.7 * kc.hbar * OMEGA[0]
        x = np.linspace(-40e-6, 40e-6, 401)
        y = np.linspace(-4e-6, 4e-6, 81)
        n = harmonic_thermal_density(x[:, None, None], y[None, :, None], y[None, None, :], T, mu, OMEGA, M)
        assert np.all(n >= -1e-9 * n.max())
        got = n.sum() * (x[1] - x[0]) * (y[1] - y[0]) ** 2
        assert got == pytest.approx(ideal_harmonic_atom_number(T, mu, OMEGA, thermal_only=True), rel=2e-3)


class TestCriticalTemperature:
    def test_closed_forms(self):
        wbar = float(np.prod(OMEGA)) ** (1 / 3)
        T0 = critical_temperature_K(N, OMEGA)
        assert T0 == pytest.approx(kc.hbar * wbar * (N / ZETA_3) ** (1 / 3) / kc.kB, rel=1e-14)
        assert T0 * 1e9 == pytest.approx(152.08, rel=1e-3)      # the lab tweezer, N = 500
        assert finite_size_shift(N, OMEGA) == pytest.approx(-0.1476, abs=5e-4)
        assert mean_field_shift(N, OMEGA, 11.33 * kc.a0, M) == pytest.approx(-0.0029, abs=2e-4)

    def test_exact_series_is_below_the_finite_size_formula(self):
        from scipy.optimize import brentq
        T0 = critical_temperature_K(N, OMEGA)
        f = lambda T: ideal_harmonic_condensate(N, T, OMEGA)[0] / N - 0.01
        T_exact = brentq(f, 0.3 * T0, 1.5 * T0)
        assert 0.85 < T_exact / T0 < 1.0                       # 142.5 / 152.1 nK at N0/N = 1%

    def test_fraction_curve_approaches_one_minus_t_cubed(self):
        """Finite N sits below the thermodynamic 1 - t^3 (A: 0.68 vs 0.875 at t = 0.5,
        N = 500) and approaches it as N grows."""
        for n, tol in ((N, 0.30), (1e5, 0.04)):
            T0 = critical_temperature_K(n, OMEGA)
            for t in (0.3, 0.6):
                N_0, _ = ideal_harmonic_condensate(n, t * T0, OMEGA)
                assert N_0 / n < 1.0 - t ** 3
                assert N_0 / n == pytest.approx(1.0 - t ** 3, abs=tol)


@pytest.fixture(scope="module")
def gas():
    return IdealHarmonicBoseGas(N, OMEGA, 50e-9, mass=M, a_scattering=11.33 * kc.a0)


class TestIdealHarmonicBoseGas:
    def test_fraction_and_reference_numbers(self, gas):
        assert gas.condensate_fraction == pytest.approx(1 - (50 / 152.08) ** 3, rel=1e-3)
        assert gas.fugacity == 1.0 and gas.chemical_potential == 0.0
        assert gas.T_c_nK == pytest.approx(152.08, rel=1e-3)
        assert gas.T_c_finite_size_nK == pytest.approx(129.6, rel=2e-3)
        assert gas.exact_condensate_fraction() < gas.condensate_fraction

    def test_densities_integrate_to_their_numbers(self, gas):
        x = np.linspace(-60e-6, 60e-6, 601)
        y = np.linspace(-5e-6, 5e-6, 101)
        X, Y, Z = x[:, None, None], y[None, :, None], y[None, None, :]
        dV = (x[1] - x[0]) * (y[1] - y[0]) ** 2
        assert gas.density(X, Y, Z, component="thermal").sum() * dV == pytest.approx(gas.N_th, rel=2e-3)
        assert gas.density(X, Y, Z, component="condensate").sum() * dV == pytest.approx(gas.N_0, rel=1e-4)
        assert gas.density(X, Y, Z).sum() * dV == pytest.approx(gas.N, rel=1e-3)
        col = gas.column_density(Y[0], Z[0])
        assert col.sum() * (y[1] - y[0]) ** 2 == pytest.approx(gas.N, rel=1e-3)
        assert gas.column_density(0.0, 0.0, component="thermal") < 0.1 * gas.peak_column_density

    def test_classical_limit(self, gas):
        hot = gas.with_temperature(20 * gas.T_c_K)
        assert hot.fugacity < 1e-3 and hot.N_0 == pytest.approx(0.0, abs=1e-6)
        boltz = np.sqrt(hot.kT / (M * OMEGA ** 2))
        assert hot.sigma_thermal == pytest.approx(boltz, rel=1e-4)
        assert hot.chemical_potential == pytest.approx(hot.kT * np.log(hot.fugacity))
        # Bose statistics pile atoms into the low states: narrower than Boltzmann
        assert gas.sigma_thermal[0] < np.sqrt(gas.kT / (M * OMEGA[0] ** 2))

    def test_zero_temperature_is_the_oscillator(self):
        cold = IdealHarmonicBoseGas(N, OMEGA, 0.0, mass=M)
        assert cold.condensate_fraction == 1.0 and cold.exact_condensate_fraction() == 1.0
        assert cold.sigma == pytest.approx(np.sqrt(kc.hbar / (2 * M * OMEGA)))
        assert np.isinf(cold.thermal_wavelength_m)
        assert cold.density(0, 0, 0) == pytest.approx(N / ((2 * np.pi) ** 1.5 * np.prod(cold.sigma)))

    def test_inverse_question(self, gas):
        T = gas.temperature_for_condensate_fraction(0.5)
        assert gas.with_temperature(T).condensate_fraction == pytest.approx(0.5, abs=1e-12)

    def test_imaging_cloud_contract(self, gas):
        assert gas.widths.shape == (3,) and np.all(gas.widths > gas.widths_condensate)
        assert gas.density(np.zeros(3), np.zeros(3), np.zeros(3)).shape == (3,)
        assert "nK" in gas.summary() and "N0/N" in repr(gas)
        with pytest.raises(ValueError):
            gas.density(0, 0, 0, component="half")

    def test_from_trap(self):
        from kamo.trap.trap import HarmonicTrap
        w = 2 * np.pi * np.array([300.0, 400.0, 500.0])
        trap = HarmonicTrap((0, 0, 0), 0.0, np.diag(M * w ** 2), M)
        g = IdealHarmonicBoseGas.from_trap(trap, 1000, 40e-9)
        assert g.omega == pytest.approx(w, rel=1e-9)
