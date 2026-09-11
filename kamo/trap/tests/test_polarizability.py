"""Tests for kamo.trap.polarizability.

INTERNAL      the lab-frame combination against compute_complete_polarizability's
              own formula (components stubbed -- no ARC), the F guards, state
              validation, and that nothing touches ARC until asked.
GROUND-TRUTH  offline (network disabled, fresh cache, so the snapshot bundled
              with kamo is read): alpha_s(4S1/2, 1064 nm) = 599.3005 a.u. as
              pinned in test_gaussian.py; the K tune-out wavelength
              768.9712(15) nm (Holmgren et al., PRL 109, 243004 (2012)), which
              both sources hit (it pins the D-line energies, not the matrix-
              element source), plus the rounded-eV-constant regression; the term-by-term
              breakdown, provenance, uncertainty and the cache contract.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from kamo.trap import frames as fr
from kamo.trap import polarizability as pz

RNG = np.random.default_rng(11)
LAM = 1064e-9
ALPHA_1064_AU = 599.3005266591443        # kamo/trap/tests/test_gaussian.py
TUNE_OUT_NM, TUNE_OUT_UNC_NM = 768.9712, 0.0015


# --------------------------------------------------------------- INTERNAL

class TestCombination:
    def _oracle(self, components, F, mF, pol_frame):
        """compute_complete_polarizability with stubbed components (no ARC)."""
        from kamo.light_shift.compute_polarizabilities import ComputePolarizabilities
        cp = object.__new__(ComputePolarizabilities)
        a = [np.array([x]) for x in components]
        cp.compute_polarizability = lambda *args, **kw: tuple(a)
        return float(np.real(cp.compute_complete_polarizability(
            4, 1, 1.5, F, mF, LAM, polarization=pol_frame)[0]))

    @pytest.mark.parametrize("F, mF", [(1, -1), (1, 0), (2, 2), (2, -1), (3, 1)])
    def test_matches_light_shift_formula(self, F, mF):
        comps = (500.0, -40.0, 25.0)
        for _ in range(20):
            eps = fr.as_unit_complex_vector(RNG.normal(size=3) + 1j * RNG.normal(size=3))
            bhat = fr.as_unit_real_vector(RNG.normal(size=3))
            f1, f2, _ = fr.orthonormal_frame(bhat)
            pol_frame = np.array([bhat @ eps, f1 @ eps, f2 @ eps])   # index 0 = B axis
            beta, gamma = fr.polarization_geometry(eps, bhat)
            assert pz.combine_au(comps, F, mF, beta, gamma) == pytest.approx(
                self._oracle(comps, F, mF, pol_frame), rel=1e-12)

    def test_guards(self):
        comps = (100.0, 10.0, 5.0)
        assert pz.combine_au(comps, 0, 0, 1.0, 1.0) == 100.0          # F = 0: scalar only
        assert pz.combine_au(comps, 0.5, 0.5, 0.0, 1.0) == 100.0      # F = 1/2: no tensor

    @pytest.mark.parametrize("state", [(4, 0, 0.5, 1), (4, 4, 0.5, 1, 0), (4, 0, 1.5, 1, 0),
                                       (4, 0, 0.5, 1, -2), (4, 0, 0.5, 1, 0.5)])
    def test_state_validation(self, state):
        with pytest.raises(ValueError):
            pz.StatePolarizability(state)

    def test_source_validation(self):
        with pytest.raises(ValueError, match="source"):
            pz.StatePolarizability((4, 0, 0.5, 1, -1), source="nist")

    def test_construction_is_lazy(self):
        pz.clear_caches()
        pz.StatePolarizability((4, 0, 0.5, 1, -1)).for_state((4, 0, 0.5, 2, 2))
        assert pz._CP == {}


# ----------------------------------------------------------- GROUND-TRUTH

@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    from kamo.light_shift import udel_portal as up
    mp = pytest.MonkeyPatch()
    mp.setenv("KAMO_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))

    def no_network(*args, **kwargs):
        raise OSError("network disabled for tests")

    mp.setattr(up, "_http", no_network)
    pz.clear_caches()
    with pytest.warns(UserWarning, match="unreachable"):
        pz.compute_polarizabilities("portal")
    yield
    pz.clear_caches()
    mp.undo()


def _alpha_s(lam_m, source):
    return pz.hyperfine_components_au(4, 0, 0.5, 1, float(lam_m), source)[0]


class TestPortalGroundTruth:
    def test_alpha_1064_matches_the_pinned_value(self, offline):
        assert _alpha_s(LAM, "portal") == pytest.approx(ALPHA_1064_AU, rel=1e-6)

    def test_breakdown_reproduces_the_calculator(self, offline):
        b = pz.scalar_breakdown(4, 0, 0.5, LAM, "portal")
        assert b.total_au == pytest.approx(_alpha_s(LAM, "portal"), rel=1e-12)
        assert b.core_au == pytest.approx(5.457)
        assert b.n_portal >= 2 and 0 <= b.arc_share < 0.01

    @pytest.mark.parametrize("source", ["portal", "arc"])
    def test_tune_out_wavelength(self, offline, source):
        """The scalar zero between D1 and D2, set by the D-line energies and the
        D1/D2 ratio.  Both sources land inside the 1.5 pm measurement (portal
        -0.10 pm, ARC -0.56 pm).  Before 2026-09 ARC missed by 20 pm -- 11 pm
        from a rounded eV constant, 8 pm from its quantum-defect levels -- so
        this pins the energies, not the choice of matrix elements."""
        lam = brentq(lambda L: _alpha_s(L * 1e-9, source), 767.2, 769.8, xtol=1e-7)
        assert abs(lam - TUNE_OUT_NM) < TUNE_OUT_UNC_NM

    def test_tune_out_catches_a_rounded_eV_constant(self, offline, monkeypatch):
        """kamo.constants used to carry e = 1.6022e-19 (14.6 ppm off); on the ARC
        path that moves the tune-out ~11 pm, which the test above must catch."""
        import kamo.constants as kc_mod
        pz.clear_caches()
        try:
            monkeypatch.setattr(kc_mod, "convert_joules_per_electronvolt", 1.6022e-19)
            lam = brentq(lambda L: _alpha_s(L * 1e-9, "arc"), 767.2, 769.8, xtol=1e-7)
            assert abs(lam - TUNE_OUT_NM) > 5 * TUNE_OUT_UNC_NM
        finally:
            monkeypatch.undo()
            pz.clear_caches()

    def test_provenance_offline_is_the_snapshot(self, offline):
        sp = pz.StatePolarizability((4, 0, 0.5, 1, -1))
        p = sp.provenance()
        assert p["origin"] == "snapshot" and p["fetched_at"].startswith("2026-09-10")
        assert "snapshot" in sp.describe(LAM)
        assert pz.provenance("arc")["origin"] == "arc"

    def test_uncertainty_is_the_propagated_d1_d2_errors(self, offline):
        sp = pz.StatePolarizability((4, 0, 0.5, 1, -1))
        b = sp.breakdown(LAM)
        by_state = {t.final: t for t in b.transitions}
        d1, d2 = by_state["4p1/2"], by_state["4p3/2"]
        assert (d1.d_au, d1.d_unc_au) == pytest.approx((4.106, 0.002))
        assert (d2.d_au, d2.d_unc_au) == pytest.approx((5.807, 0.003))
        leading = np.hypot(2 * d1.scalar_au * 0.002 / 4.106, 2 * d2.scalar_au * 0.003 / 5.807)
        assert leading <= sp.uncertainty_au(LAM) < 1.01 * leading
        assert pz.StatePolarizability((4, 0, 0.5, 1, -1), source="arc").uncertainty_au(LAM) != \
            pz.StatePolarizability((4, 0, 0.5, 1, -1), source="arc").uncertainty_au(LAM)  # NaN

    def test_vector_shift_is_mF_and_helicity_dependent(self, offline):
        sp = pz.StatePolarizability((4, 0, 0.5, 1, -1))
        z = [0, 0, 1]
        sig_p = -np.array([1, 1j, 0]) / np.sqrt(2)
        sig_m = np.array([1, -1j, 0]) / np.sqrt(2)
        _, a_v, _ = sp.components_au(LAM)
        diff = sp.alpha_au(LAM, sig_p, z) - sp.alpha_au(LAM, sig_m, z)
        assert diff == pytest.approx(2 * (-1) / (2 * 1) * a_v, rel=1e-12)
        assert a_v != 0.0

    def test_default_polarization_with_vertical_B_has_no_vector_shift(self, offline):
        eps, z = [0, 1, 1j], [0, 0, 1]
        up = pz.StatePolarizability((4, 0, 0.5, 1, +1)).alpha_au(LAM, eps, z)
        dn = pz.StatePolarizability((4, 0, 0.5, 1, -1)).alpha_au(LAM, eps, z)
        assert up == dn
        along_x = pz.StatePolarizability((4, 0, 0.5, 1, -1)).alpha_au(LAM, eps, [1, 0, 0])
        assert along_x != dn                     # sigma+ about x: the vector shift is on

    def test_cache_contract(self, offline):
        sp = pz.StatePolarizability((4, 0, 0.5, 1, -1))
        sp.alpha_au(LAM, [0, 1, 0], [0, 0, 1])
        misses = pz.hyperfine_components_au.cache_info().misses
        sp.for_state((4, 0, 0.5, 1, +1)).alpha_au(LAM, [0, 1, 1j], [1, 0, 0])
        sp.alpha_au(LAM, [0, 0, 1], fr.rotation_about([1, 1, 0], 0.3) @ [0, 0, 1])
        assert pz.hyperfine_components_au.cache_info().misses == misses

    def test_near_resonance_guard(self, offline):
        sp = pz.StatePolarizability((4, 0, 0.5, 1, -1))
        with pytest.warns(UserWarning, match="close to resonance"):
            sp.alpha_au(766.75e-9, [0, 1, 0], [0, 0, 1])
