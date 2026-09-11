"""kamo.atom_properties.hyperfine: potassium A and B constants.

SURVEY TABLE   the bundled CSV is self-consistent (j = l +- 1/2, no B for J = 1/2).
PINNED VALUES  the 4S/4P constants the Hamiltonian uses, with their sources.
SELECTION      a state's own measurement or theory beats a model; models fill gaps.
THEORY UNC     the uncertainty given to portal theory covers its miss vs experiment.
EXTRAPOLATION  A n*^3 reproduces a held-out state and gives Rydberg values.
ISOTOPES       scaling 39K by nuclear-moment ratios reproduces measured 40K/41K.
CONSUMERS      constants wrappers, g_J, builder h0 and Potassium39.getHFSCoefficients.

Runs offline: the portal cache is an empty directory and the network is off,
so theory comes from the snapshot bundled with kamo.
"""
import math

import numpy as np
import pytest

from kamo import constants as c
from kamo.atom_properties import hyperfine as hfs
from kamo.light_shift import udel_portal as up

H = hfs.hyperfine_constants
pytestmark = pytest.mark.filterwarnings("ignore:UDel portal unreachable")


def _clear_caches():
    for f in (hfs._portal_theory, hfs._hyperfine_constants, hfs.nuclear_moments):
        f.cache_clear()


@pytest.fixture
def theory_without(monkeypatch):
    """Drop some theory states, to test extrapolation against held-out values."""
    full = dict(hfs._portal_theory())

    def drop(*states):
        kept = {k: v for k, v in full.items() if k not in states}
        monkeypatch.setattr(hfs, "_portal_theory", lambda: kept)
        hfs._hyperfine_constants.cache_clear()
        return full

    yield drop
    hfs._hyperfine_constants.cache_clear()


@pytest.fixture(scope="module", autouse=True)
def offline(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("KAMO_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))

    def no_network(*args, **kwargs):
        raise OSError("network disabled for tests")

    mp.setattr(up, "_http", no_network)
    _clear_caches()
    yield
    mp.undo()
    _clear_caches()


# ============================================================== SURVEY TABLE
def test_survey_table_is_consistent():
    df = hfs.survey_table()
    assert set(df.iso) == {39, 40, 41}
    assert not df.duplicated(["iso", "n", "l", "twoj"]).any()
    assert ((df.twoj - 2 * df.l).abs() == 1).all()
    assert (df.A_unc_MHz > 0).all()
    assert df.loc[df.twoj == 1, ["B_MHz", "B_unc_MHz"]].isna().all().all()
    # a B value always comes with its uncertainty
    assert (df.B_MHz.notna() == df.B_unc_MHz.notna()).all()


# ============================================================== PINNED VALUES
@pytest.mark.parametrize("state, A, B", [
    ((4, 0, 0.5, 39), 230.8598601, 0.0),
    ((4, 1, 0.5, 39), 27.793, 0.0),
    ((4, 1, 1.5, 39), 6.084, 2.842),
    ((4, 0, 0.5, 40), -285.7308, 0.0),
    ((4, 1, 1.5, 40), -7.585, -3.445),
    ((4, 0, 0.5, 41), 127.0069352, 0.0),
    ((4, 1, 0.5, 41), 15.245, 0.0),            # ARC has the sign wrong
    ((4, 1, 1.5, 41), 3.342, 3.242),
])
def test_ground_and_D_line_constants(state, A, B):
    hc = H(*state)
    assert hc.A_MHz == A and hc.B_MHz == B
    assert hc.A_source == "measured" and "Allegrini" in hc.A_ref


def test_4S_and_4P_need_no_portal_or_arc(monkeypatch):
    """The precisely measured states never read theory or ARC energies."""
    _clear_caches()

    def boom(*a, **k):
        raise AssertionError("should not be called")

    for name in ("_portal_theory", "_effective_n", "nuclear_moments", "_arc"):
        monkeypatch.setattr(hfs, name, boom)
    for iso in (39, 40, 41):
        for state in [(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)]:
            assert H(*state, iso=iso).A_source == "measured"
    hfs._hyperfine_constants.cache_clear()


# ================================================================= SELECTION
@pytest.mark.parametrize("state, source", [
    ((8, 0, 0.5), "measured"),         # 5.99(8): a measurement beats interpolation
    ((5, 1, 0.5), "measured"),
    ((9, 0, 0.5), "theory"),           # never measured well
    ((7, 1, 1.5), "theory"),           # theory 3% is > 2x better than the 8% measurement
    ((7, 1, 0.5), "measured"),         # 2.18(5): theory 2.14(4) is not 2x better
    ((4, 2, 1.5), "theory"),
    ((5, 2, 2.5), "measured"),         # -0.24(7): theory -0.167(50) is not 2x better
    ((4, 2, 2.5), "theory"),
    ((11, 0, 0.5), "extrapolated"),    # 2.1(9) measured, no theory
    ((8, 1, 1.5), "extrapolated"),
    ((20, 2, 2.5), "extrapolated"),
])
def test_source_selection(state, source):
    assert H(*state).A_source == source


def test_d5halves_are_inverted():
    for n in range(3, 9):
        assert H(n, 2, 2.5).A_MHz < 0
    assert H(3, 2, 2.5).A_MHz == -0.62


def test_B_is_zero_for_J_half_and_extrapolated_where_unmeasured():
    assert H(5, 0, 0.5).B_source.startswith("exact") and H(5, 0, 0.5).B_MHz == 0
    b7 = H(7, 1, 1.5)                  # B listed as 0 (not measured) in the survey
    assert b7.B_source == "extrapolated" and 0.15 < b7.B_MHz < 0.30


def test_B_bound():
    hc = H(3, 2, 2.5)                  # |B| < 0.3 MHz
    assert (hc.B_MHz, hc.B_unc_MHz, hc.B_source) == (0.0, 0.3, "bound")
    assert H(3, 2, 2.5, iso=40).B_source == "bound"      # scaled bound stays a bound


def test_no_data_for_f_states_and_core_orbitals():
    hc = H(4, 3, 3.5)
    assert not hc.has_A and hc.A_MHz == 0 and hc.B_source == "none"
    assert c.get_hyperfine_constant(3, 3.5) is None
    core = H(3, 0, 0.5)                # 3s is a core orbital of K
    assert not core.has_A and "core" in core.A_ref
    assert c.get_hyperfine_constant(0, 0.5, n=3) is None


@pytest.mark.parametrize("args", [(4, 1, 2.5), (4, 0, -0.5), (4.5, 0, 0.5),
                                  (2, 2, 1.5), (4, 0, 0.5, 42)])
def test_bad_arguments(args):
    with pytest.raises(ValueError):
        H(*args)


# ================================================================ THEORY UNC
def test_theory_uncertainty_covers_miss_vs_measurement():
    theory = hfs._portal_theory()
    assert len(theory) == 24                  # 5-10s, 4-7p, 3-7d (none for 4s)
    n_pairs = 0
    for (l, twoj), frac in hfs.THEORY_FRAC_UNC.items():
        devs = []
        for (n, ll, tj), a_th in theory.items():
            meas = hfs._measured(39, n, ll, tj, "A")
            if (ll, tj) != (l, twoj) or meas is None:
                continue
            n_pairs += 1
            # every pair agrees within 2 sigma of the combined uncertainty
            assert abs(a_th - meas.value) < 2 * math.hypot(frac * a_th, meas.unc), (n, l, twoj)
            if meas.frac <= 0.05:          # well measured: the miss is theory's
                devs.append((a_th - meas.value) / meas.value)
        if devs:
            assert math.sqrt(np.mean(np.square(devs))) <= frac, (l, twoj, devs)
    assert n_pairs == 20


def test_portal_6p3halves_disagrees_with_survey():
    """The portal lists 0.866(8); the survey (and ARC) 0.886(8). Theory's
    systematic low bias along 4-6p3/2 (1.6%, 1.9%, 2.3%) fits 0.886."""
    df = up.hyperfine_constants()
    portal = df[(df.iso == 39) & (df.state == "6p3/2")].A_exp_MHz.item()
    assert portal == pytest.approx(0.866) and H(6, 1, 1.5).A_MHz == 0.886


# ============================================================= EXTRAPOLATION
@pytest.mark.parametrize("held_out, twoj, l", [((10,), 1, 0), ((9, 10), 1, 0),
                                               ((7,), 3, 2), ((6, 7), 3, 2),
                                               ((7,), 3, 1), ((7,), 5, 2)])
def test_extrapolation_reproduces_held_out_theory(theory_without, held_out, twoj, l):
    """Fit the trend without the top theory states, then predict them. The
    d3/2 series is still rising at 7d; a plain n*^3 average came out 16% low."""
    full = theory_without(*[(n, l, twoj) for n in held_out])
    for n in held_out:
        ex = hfs._extrapolated_theory(n, l, twoj)
        true = full[(n, l, twoj)]
        assert abs(ex.value - true) < 2 * ex.unc, (n, ex.value, true)
        # and the trend, not a flat average (16% low for d3/2): within 5%
        assert abs(ex.value / true - 1) < 0.05, (n, ex.value, true)


def test_rydberg_values_follow_the_theory_trend():
    a59 = H(59, 0, 0.5)
    assert a59.A_source == "extrapolated" and "n = 8, 9, 10" in a59.A_ref
    assert 6.2e3 < a59.A_Hz < 6.7e3            # ~6.48 kHz
    ns = hfs._effective_n(59, 0, 1)
    assert a59.A_MHz * ns ** 3 == pytest.approx(1188.5, rel=2e-3)   # theory C_inf
    for l, twoj in [(0, 1), (1, 1), (1, 3), (2, 3), (2, 5)]:
        A = [H(n, l, twoj / 2).A_MHz for n in range(8, 40)]
        assert all(abs(a) > abs(b) > 0 for a, b in zip(A, A[1:])), (l, twoj)


def _full_rule(iso, n, l, twoj, which):
    """The selection rule with every candidate evaluated (no lazy shortcut)."""
    if which == "B" and twoj == 1:
        return hfs._Candidate(0.0, 0.0, hfs._EXACT, "")
    meas = hfs._measured(iso, n, l, twoj, which)
    if iso != 39:
        return hfs._keep_measured(meas, hfs._isotope_scaled(iso, n, l, twoj, which))
    own = hfs._keep_measured(meas, hfs._theory(39, n, l, twoj, which))
    if hfs._is_good(own, which):
        return own
    return hfs._keep_measured(own, hfs._extrapolated(n, l, twoj, which))


def test_lazy_shortcut_matches_the_full_rule():
    for iso, n_max in [(39, 14), (40, 8), (41, 8)]:
        for l in range(3):
            for twoj in sorted({abs(2 * l - 1), 2 * l + 1}):
                for n in list(range(hfs.lowest_valence_n(l), n_max + 1)) + [20, 59]:
                    for which in "AB":
                        lazy = hfs._best(iso, n, l, twoj, which)
                        full = _full_rule(iso, n, l, twoj, which)
                        assert (lazy is None) == (full is None)
                        if lazy is not None:
                            assert (lazy.value, lazy.source) == (full.value, full.source),                                 (iso, n, l, twoj, which)


def test_extrapolation_only_beyond_theory():
    assert hfs._extrapolated(3, 2, 3, "A") is None
    assert hfs._extrapolated(6, 2, 3, "A") is None


# ================================================================== ISOTOPES
@pytest.mark.parametrize("iso, state", [(40, (4, 1, 1)), (41, (4, 1, 1)),
                                        (40, (4, 1, 3)), (41, (4, 1, 3)),
                                        (41, (5, 1, 3)), (41, (6, 0, 1))])
def test_isotope_scaling_reproduces_measurements(iso, state):
    scaled = hfs._isotope_scaled(iso, *state, "A")
    meas = hfs._measured(iso, *state, "A")
    assert abs(scaled.value - meas.value) < 2 * math.hypot(scaled.unc, meas.unc)


def test_isotope_scaling_fills_gaps():
    hc = H(7, 0, 0.5, iso=41)              # measured 6.5(10), a poor value
    assert hc.A_source == "isotope-scaled"
    assert hc.A_MHz == pytest.approx(10.79 * 127.0069352 / 230.8598601)
    assert H(5, 0, 0.5, iso=40).A_MHz < 0


def test_isotope_scaling_competes_like_theory():
    """Scaled 39K values replace a measurement only when 2x more precise."""
    assert H(6, 0, 0.5, iso=41).A_source == "isotope-scaled"   # 12.043(50) vs 12.03(40)
    assert H(5, 1, 0.5, iso=40).A_source == "isotope-scaled"   # -11.20(22) vs -12.0(9)
    assert H(3, 2, 1.5, iso=41).A_source == "measured"         # 0.55(3) vs 0.527(22)
    assert H(4, 1, 1.5, iso=40).B_source == "measured"         # -3.445(90) vs -3.535(89)


# ================================================================= CONSUMERS
def test_get_hyperfine_constant_wrapper():
    assert c.get_hyperfine_constant(0, 0.5) == pytest.approx(c.h * 230.8598601e6)
    assert c.get_hyperfine_constant(1, 1.5, n=4) == pytest.approx(c.h * 6.084e6)
    assert c.get_hyperfine_constant(2, 2.5) == pytest.approx(c.h * -0.62e6)  # n -> 3
    # used to return the 4S value for any n
    a40_5s = c.get_hyperfine_constant(0, 0.5, iso=40, n=5) / c.h
    assert -80e6 < a40_5s < -60e6


def test_arc_hfs_table_is_not_used(monkeypatch):
    import arc.alkali_atom_functions as aaf

    def boom(*a, **k):
        raise AssertionError("ARC's HFS table must not be used")

    monkeypatch.setattr(aaf.AlkaliAtom, "getHFSCoefficients", boom)
    _clear_caches()
    hfs.hyperfine_table(39, n_max=12)
    hfs.hyperfine_table(41, n_max=8)
    _clear_caches()


@pytest.mark.parametrize("l, j, g", [(0, 0.5, 2.00229421), (1, 0.5, 0.665875),
                                     (1, 1.5, 1.334097), (2, 1.5, 0.799519),
                                     (2, 2.5, 1.200453)])
def test_g_J(l, j, g):
    assert c.get_total_electronic_g_factor(l, j) == pytest.approx(g, abs=1e-6)


def test_builder_uses_A_and_B():
    from kamo.hamiltonian import Basis
    from kamo.hamiltonian.builder import HamiltonianBuilder

    class _Atom:                    # h0 needs only the fine-structure energy
        def getEnergy(self, n, l, j):
            return 0.0

    man = (4, 1, 1.5)
    E = np.linalg.eigvalsh(HamiltonianBuilder(Basis([man]), atom=_Atom()).h0())
    hc = H(*man)
    expected = sorted(F_E for F in (0, 1, 2, 3)
                      for F_E in [hfs.hyperfine_energy(F, 1.5, 1.5, hc.A_Hz, hc.B_Hz)]
                      * (2 * F + 1))
    np.testing.assert_allclose(E, expected, atol=1.0)       # Hz


def test_F_order_follows_the_sign_of_A():
    from kamo.hamiltonian.basis import Manifold
    assert Manifold(4, 1, 1.5).F_order() == [0.0, 1.0, 2.0, 3.0]
    assert Manifold(3, 2, 2.5).F_order() == [4.0, 3.0, 2.0, 1.0]      # A < 0
    assert Manifold(4, 3, 3.5).F_order() == [2.0, 3.0, 4.0, 5.0]      # no data


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")   # ARC's np.matrix
def test_potassium39_breitRabi_has_the_full_quadrupole_term():
    """ARC's breitRabi halves B; the override matches the analytic levels."""
    from kamo import Potassium39
    k = Potassium39()
    hc = H(4, 1, 1.5)
    levels = {F: hfs.hyperfine_energy(F, 1.5, 1.5, hc.A_Hz, hc.B_Hz) for F in range(4)}
    E, F, mF = k.breitRabi(4, 1, 1.5, np.array([0.0, 0.05]))
    assert E.shape == (2, 16) and sorted(set(F)) == [0, 1, 2, 3]
    expected = sorted(levels[f] for f in range(4) for _ in range(2 * f + 1))
    np.testing.assert_allclose(E[0], expected, atol=1.0)
    E_arc = Potassium39.__mro__[1].breitRabi(k, 4, 1, 1.5, np.array([0.0]))[0][0]
    assert abs(E_arc.min() - levels[0]) > 1e6               # ARC: 1.8 MHz off at F'=0
    # at 500 G the m_J = +3/2 levels use the Landé g_J with g_S
    top = E[1].max()
    assert top > levels[3] + 0.9 * 1.334 * 1.5 * c.mu_b * 0.05 / c.h


def test_potassium39_breitRabi_labels_small_A_manifolds():
    """Labels come from a field scaled to the hyperfine gap: a 1e-4 T probe
    would F-mix 59p3/2 (A = 364 Hz)."""
    from kamo import Potassium39
    k = Potassium39()
    for state in [(59, 1, 1.5), (59, 2, 2.5), (100, 0, 0.5), (5, 1, 1.5)]:
        _, F, mF = k.breitRabi(*state, np.array([0.0]))
        assert np.all(F == np.round(F)), state
        for f in set(F):
            assert np.sum(F == f) == 2 * f + 1, state
    with pytest.raises(ValueError):
        k.breitRabi(4, 3, 3.5, np.array([0.0]))
    assert Potassium39.gI == c.g_I != 0                  # ARC leaves it at 0


def test_make_nlj_basis_skips_core_orbitals():
    from kamo.hamiltonian import make_nlj_basis
    assert make_nlj_basis(4, 0, n_range=1) == [
        (4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5), (5, 0, 0.5), (5, 1, 0.5), (5, 1, 1.5)]
    assert (3, 1, 1.5) not in make_nlj_basis(3, 2)
    assert (3, 2, 1.5) in make_nlj_basis(3, 2)


def test_potassium39_getHFSCoefficients():
    from kamo import Potassium39
    k = Potassium39.__new__(Potassium39)      # skip ARC's database for this path
    k.use_portal = True
    A, B = k.getHFSCoefficients(4, 1, 0.5)
    assert (A, B) == (pytest.approx(27.793e6), 0.0)
    with pytest.raises(ValueError):
        k.getHFSCoefficients(4, 3, 3.5)
