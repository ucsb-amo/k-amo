"""Tests for kamo.trap.solvers, the solve() front door, and the package exports.

INTERNAL only: mode aliases, dispatch to the three solvers, the finite-T
wiring (the solver itself is tested in test_finite_temperature.py), solve_all /
comparison_table, and that ``from kamo import Tweezer, Trap, solve`` works.
Harmonic traps with explicit mass and a.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass

import kamo.constants as kc
from kamo.trap.cloud import TrapCloud
from kamo.trap.solvers import canonical_mode, comparison_table, solve, solve_all
from kamo.trap.trap import HarmonicTrap

M = 38.963706 * atomic_mass
A = 50 * kc.a0


def harmonic():
    w = 2 * np.pi * np.array([300.0, 400.0, 500.0])
    return HarmonicTrap((0.0, 0.0, 0.0), 0.0, np.diag(M * w ** 2), M)


FAST = {"noninteracting": dict(n_per_axis=24), "thomas-fermi": dict(n_per_axis=48),
        "gp": dict(points_per_scale=2.0)}


class TestModes:
    @pytest.mark.parametrize("alias, mode", [("NI", "noninteracting"), ("ideal", "noninteracting"),
                                             ("tf", "thomas-fermi"), ("Thomas_Fermi", "thomas-fermi"),
                                             ("gpe", "gp"), ("gross-pitaevskii", "gp")])
    def test_aliases(self, alias, mode):
        assert canonical_mode(alias) == mode

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            canonical_mode("hartree")

    def test_finite_temperature_is_wired_up(self):
        import warnings
        from kamo.trap.finite_temperature import FiniteTemperatureResult
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cloud = solve(harmonic(), 300.0, "gp", a_scattering=A, T_K=30e-9, n_thermal_widths=4.5,
                          condensate_options=dict(points_per_scale=2.0))
        assert isinstance(cloud, TrapCloud) and isinstance(cloud.info, FiniteTemperatureResult)
        assert 0.0 < cloud.condensate_fraction < 1.0 and cloud.T_K == 30e-9

    def test_temperature_is_not_a_mode(self):
        with pytest.raises(ValueError, match="T_K"):
            canonical_mode("hartree-fock")
        with pytest.raises(ValueError, match="thomas-fermi"):
            solve(harmonic(), 1000.0, "tf", a_scattering=A, T_K=50e-9)
        with pytest.raises(ValueError, match="T_K"):
            solve(harmonic(), 1000.0, "gp", a_scattering=A, T_K=-1.0)
        with pytest.raises(ValueError, match="T_K > 0"):
            solve(harmonic(), 1000.0, "gp", a_scattering=A, condensate_options={})
        with pytest.raises(TypeError):                      # a GP option at the top level at T > 0
            solve(harmonic(), 1000.0, "gp", a_scattering=A, T_K=50e-9, points_per_scale=2.0)

    def test_noninteracting_takes_no_V_extra(self):
        with pytest.raises(ValueError, match="V_extra"):
            solve(harmonic(), 1000.0, "ni", V_extra=0.0)


class TestDispatch:
    @pytest.mark.parametrize("mode", ["noninteracting", "thomas-fermi", "gp"])
    def test_each_mode(self, mode):
        cloud = solve(harmonic(), 1000.0, mode, a_scattering=A, **FAST[mode])
        assert isinstance(cloud, TrapCloud) and cloud.mode == mode
        assert cloud.atom_number_error == pytest.approx(0.0, abs=1e-9)

    def test_solve_all_and_table(self):
        clouds = solve_all(harmonic(), 1000.0, a_scattering=0.0, options=FAST)
        assert isinstance(clouds["thomas-fermi"], ValueError)          # no TF limit at a = 0
        assert isinstance(clouds["gp"], TrapCloud)
        assert clouds["gp"].chemical_potential_offset == pytest.approx(
            clouds["noninteracting"].chemical_potential_offset, rel=1e-6)
        table = comparison_table(clouds)
        assert "noninteracting" in table and "ValueError" in table
        with pytest.raises(ValueError):
            solve_all(harmonic(), 1000.0, a_scattering=0.0, options=FAST, skip_errors=False)


def test_top_level_exports():
    import kamo
    from kamo import Crossed, LightSheet, Trap, TrapCloud as TC, Tweezer, solve as s
    import kamo.trap
    assert s is kamo.trap.solve and TC is TrapCloud
    assert Tweezer.__module__ == "kamo.trap.beams" and Trap.__module__ == "kamo.trap.trap"
    assert kamo.GaussianBeam.__module__ == "kamo.trap.gaussian"
