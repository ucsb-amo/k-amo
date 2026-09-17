"""Tests for kamo.light_shift.udel_portal.

INTERNAL tests run on an inline fixture (the K 4s/4p rows as the portal returns
them) and never touch the network. NETWORK tests query the live portal and are
skipped when it cannot be reached.
"""

import json
import urllib.request
import warnings

import numpy as np
import pandas as pd
import pytest

from kamo.light_shift import udel_portal as up

# Two K1 rows exactly as the GraphQL API returns them: the same transition,
# once in each direction.
_ROWS = [
    {"stateOneConfiguration": "4p", "stateOneTerm": "2P", "stateOneJ": "3/2",
     "stateTwoConfiguration": "4s", "stateTwoTerm": "2S", "stateTwoJ": "1/2",
     "wavelength": 766.7009, "wavelengthUncertainty": 0,
     "matrixElement": 5.807, "matrixElementUncertainty": 0.003,
     "matrixElementRef": "St. Falke et al., J. Chem. Phys. 125, 224303 (2006)"},
    {"stateOneConfiguration": "4s", "stateOneTerm": "2S", "stateOneJ": "1/2",
     "stateTwoConfiguration": "4p", "stateTwoTerm": "2P", "stateTwoJ": "3/2",
     "wavelength": 766.7009, "wavelengthUncertainty": 0,
     "matrixElement": 5.807, "matrixElementUncertainty": 0.003,
     "matrixElementRef": "St. Falke et al., J. Chem. Phys. 125, 224303 (2006)"},
    {"stateOneConfiguration": "5p", "stateOneTerm": "2P", "stateOneJ": "1/2",
     "stateTwoConfiguration": "4s", "stateTwoTerm": "2S", "stateTwoJ": "1/2",
     "wavelength": 404.8356, "wavelengthUncertainty": 0.0001,
     "matrixElement": 0.2758298, "matrixElementUncertainty": 0.0108746,
     "matrixElementRef": ""},
]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the cache at tmp_path and pre-populate it with the fixture rows."""
    monkeypatch.setenv("KAMO_CACHE_DIR", str(tmp_path))
    path = up.cache_dir() / "K1_matrix_elements.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"data": {"matrixElements": _ROWS}}))
    return path


# --------------------------------------------------------------- INTERNAL

def test_parse_display_value():
    assert up.parse_display_value("0.29025(25)") == pytest.approx((0.29025, 0.00025))
    assert up.parse_display_value("4961(22)") == pytest.approx((4961, 22))
    assert up.parse_display_value("-12.3") == (-12.3, pytest.approx(np.nan, nan_ok=True))
    assert np.isnan(up.parse_display_value("")[0])


def test_matrix_elements_parses_cached_rows(cache):
    df = up.matrix_elements("K1")
    first = df.iloc[0]
    assert (first.state1, first.state2) == ("4p3/2", "4s1/2")
    assert (first.n1, first.l1, first.J1) == (4, 1, 1.5)
    assert (first.n2, first.l2, first.J2) == (4, 0, 0.5)
    assert first.d_au == 5.807
    assert list(df.source) == ["exp", "exp", "theory"]


def test_dedupe_pairs_collapses_both_directions(cache):
    df = up.matrix_elements("K1")
    assert len(df) == 3
    assert len(up.dedupe_pairs(df)) == 2


def test_legacy_table_has_both_directions(cache):
    legacy = up.to_legacy_table(up.matrix_elements("K1"))
    assert list(legacy.columns) == ["Initial", "Final",
                                    "Matrix element (a.u.)", "Wavelength (nm)"]
    pairs = set(zip(legacy.Initial, legacy.Final))
    # 5p1/2 -> 4s1/2 was listed in one direction only; the reverse is added
    assert pairs == {("4p3/2", "4s1/2"), ("4s1/2", "4p3/2"),
                     ("5p1/2", "4s1/2"), ("4s1/2", "5p1/2")}


def test_legacy_table_rejects_multivalence():
    multi = pd.DataFrame({"state1": ["5s2.5p1/2"], "state2": ["5s2.6s1/2"],
                          "n1": [np.nan], "n2": [np.nan],
                          "d_au": [1.0], "wavelength_nm": [500.0]})
    with pytest.raises(ValueError):
        up.to_legacy_table(multi)


def test_cache_write_and_offline_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("KAMO_CACHE_DIR", str(tmp_path))
    assert up._cached("x", lambda: {"a": 1}) == {"a": 1}
    stored = json.loads((up.cache_dir() / "x.json").read_text())
    assert stored["data"] == {"a": 1} and "fetched_at" in stored and stored["citation"]

    def offline():
        raise OSError("no network")

    with pytest.warns(UserWarning):
        assert up._cached("x", offline, refresh=True) == {"a": 1}
    with pytest.raises(ConnectionError):
        up._cached("y", offline)


# ------------------------------------------------- INTERNAL: normalisation

# The Cs hyperfine table has one row whose configuration is upper case ("9D")
# where every other row is lower case; unnormalised it parses to n = NaN.
_HFS_ROWS = [
    {"isotopeMassNumber": "133", "stateConfiguration": "6s", "stateTerm": "2S",
     "stateJ": "1/2", "hyperfineTheory": 2298.2, "hyperfineTheoryRef": "theory",
     "hyperfineExperiment": 2298.157, "hyperfineExperimentUncertainty": None,
     "hyperfineExperimentRef": "exp"},
    {"isotopeMassNumber": "", "stateConfiguration": "9D", "stateTerm": "2D",
     "stateJ": "3/2", "hyperfineTheory": None, "hyperfineTheoryRef": None,
     "hyperfineExperiment": 2.35, "hyperfineExperimentUncertainty": "0.02",
     "hyperfineExperimentRef": "exp"},
]


def test_uppercase_configuration_is_normalized():
    assert up.normalize_config("9D") == "9d"
    assert up.normalize_config(None) == ""
    assert up._valence_nl("9D") == (9, 2)
    assert up.state_label("9D", "3/2") == "9d3/2"
    rec = up._state_record(1, "9D", "2D", "3/2")
    assert (rec["state1"], rec["config1"], rec["n1"], rec["l1"]) == ("9d3/2", "9d", 9, 2)


def test_hyperfine_constants_normalizes_uppercase_row(tmp_path, monkeypatch):
    monkeypatch.setenv("KAMO_CACHE_DIR", str(tmp_path))
    path = up.cache_dir() / "Cs1_hyperfine_constants.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"data": {"hyperfineConstants": _HFS_ROWS}}))
    df = up.hyperfine_constants("Cs1")
    row = df[df.state == "9d3/2"].iloc[0]
    assert (row.config, row.n, row.l, row.J, row.iso) == ("9d", 9, 2, 1.5, 133)
    assert not df.n.isna().any()


# ------------------------------------------------------ INTERNAL: species ids

def test_element_symbol():
    assert up.element_symbol("K1") == "K"
    assert up.element_symbol("Cs1") == "Cs"
    assert up.element_symbol("Ca2") == "Ca"
    with pytest.raises(ValueError):
        up.element_symbol("4p3/2")


def test_species_for_element():
    assert up.species_for_element("K") == "K1"
    assert up.species_for_element("cs") == "Cs1"
    assert up.species_for_element("Ca", ion_stage=1) == "Ca2"
    with pytest.raises(ValueError):
        up.species_for_element("K", ion_stage=-1)
    with pytest.raises(ValueError):
        up.species_for_element("K1")


@pytest.mark.parametrize("species", ["Li1", "Na1", "K1", "Rb1", "Cs1"])
def test_species_id_roundtrip(species):
    assert up.species_for_element(up.element_symbol(species)) == species


# ---------------------------------------------------- INTERNAL: bundled path

_ACCESSORS = [up.matrix_elements, up.transition_rates, up.lifetimes, up.energies,
              up.static_polarizabilities, up.hyperfine_constants, up.nuclear_data]
_ALKALIS = ["Li1", "Na1", "K1", "Rb1", "Cs1"]


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Empty cache, no network: only the bundled snapshot can answer."""
    monkeypatch.setenv("KAMO_CACHE_DIR", str(tmp_path))

    def dead(*args, **kwargs):
        raise OSError("no network (test)")

    monkeypatch.setattr(up, "_http", dead)


@pytest.mark.parametrize("species", _ALKALIS)
@pytest.mark.parametrize("accessor", _ACCESSORS, ids=lambda f: f.__name__)
def test_bundled_accessor_needs_no_cache_or_network(offline, accessor, species):
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # bundled=True must not warn
        df = accessor(species, bundled=True)
    assert len(df) > 0


@pytest.mark.parametrize("species", _ALKALIS)
@pytest.mark.parametrize("accessor", _ACCESSORS, ids=lambda f: f.__name__)
def test_offline_fallback_reaches_the_snapshot(offline, accessor, species):
    with pytest.warns(UserWarning, match="unreachable"):
        df = accessor(species)
    assert len(df) > 0


@pytest.mark.parametrize("accessor", _ACCESSORS, ids=lambda f: f.__name__)
def test_bundled_missing_snapshot_names_species_and_path(offline, accessor):
    with pytest.raises(KeyError) as exc:
        accessor("Xx9", bundled=True)
    message = str(exc.value)
    assert "Xx9" in message and "write_snapshot" in message
    assert str(up._SNAPSHOT_DIR.name) in message


def test_connection_error_names_dataset_and_suggests_write_snapshot(offline):
    with pytest.raises(ConnectionError) as exc:
        up.matrix_elements("Xx9")
    message = str(exc.value)
    assert "Xx9_matrix_elements" in message and "write_snapshot('Xx9')" in message


# --------------------------------------------------- INTERNAL: write_snapshot

def test_write_snapshot_defaults_include_energies():
    import inspect

    datasets = inspect.signature(up.write_snapshot).parameters["datasets"].default
    assert set(datasets) >= {"matrix_elements", "transition_rates", "energies",
                             "hyperfine_constants", "nuclears"}
    # static polarizabilities stay opt-in
    assert "static_polarizabilities" not in datasets


def test_write_snapshot_rejects_unknown_dataset():
    with pytest.raises(KeyError):
        up.write_snapshot("K1", ("not_a_dataset",))


def test_write_snapshot_offline_raises_connection_error(offline, monkeypatch):
    """Offline it must not fall back to the snapshot and rewrite it in place."""
    written = []
    monkeypatch.setattr(up.Path, "write_text",
                        lambda self, *a, **k: written.append(self))
    with pytest.raises(ConnectionError, match="K1"):
        up.write_snapshot("K1", ("matrix_elements",))
    assert not [p for p in written if up._SNAPSHOT_DIR in p.parents]


def test_write_snapshot_offline_raises_for_unknown_species(offline):
    with pytest.raises(ConnectionError):
        up.write_snapshot("Xx9", ("matrix_elements",))


# ---------------------------------------------------------------- NETWORK

def _portal_reachable():
    try:
        urllib.request.urlopen(up.GRAPHQL_URL.rsplit("/", 1)[0], timeout=5)
    except urllib.error.HTTPError:
        return True  # the host answered (its root is a 404)
    except OSError:
        return False
    return True


network = pytest.mark.skipif(not _portal_reachable(), reason="UDel portal unreachable")


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAMO_CACHE_DIR", str(tmp_path))


def _static_alpha0_au(df, energies, state, j, core=5.457):
    """Sum over states for the static scalar polarizability, in a.u."""
    E = dict(zip(energies.state, energies.energy_cm))
    hartree_cm = 219474.6313632
    total = core
    for r in up.dedupe_pairs(df).itertuples():
        if state not in (r.state1, r.state2):
            continue
        other = r.state2 if r.state1 == state else r.state1
        total += 2 / (3 * (2 * j + 1)) * r.d_au**2 / ((E[other] - E[state]) / hartree_cm)
    return total


@network
def test_k1_live_matrix_elements(fresh_cache):
    df = up.matrix_elements("K1")
    assert len(up.dedupe_pairs(df)) == len(df) // 2
    d1 = df[(df.state1 == "4p1/2") & (df.state2 == "4s1/2")].d_au.item()
    assert d1 == pytest.approx(4.106, abs=0.01)


@network
def test_k1_static_polarizabilities_reproduce_portal(fresh_cache):
    """Portal: 290.25(25) a.u. (4s) and 4961(22) a.u. (5s)."""
    df, E = up.matrix_elements("K1"), up.energies("K1")
    assert _static_alpha0_au(df, E, "4s1/2", 0.5) == pytest.approx(290.25, abs=0.5)
    assert _static_alpha0_au(df, E, "5s1/2", 0.5) == pytest.approx(4961, abs=22)


@network
def test_compute_polarizabilities_matches_portal_calculator(fresh_cache):
    """kamo's portal path vs the portal's own calculator, K 4s at 1064 nm."""
    from kamo.light_shift import ComputePolarizabilities

    dp = up.dynamic_polarizability("K1", "4s_2S_1-2")
    portal = float(np.interp(1064, dp.wavelength_nm, dp.alpha_au))
    cp = ComputePolarizabilities(force_arc=False, portal_species="K1")
    alpha_s = cp.compute_fine_structure_polarizability(4, 0, 0.5, 1064e-9)[0][0]
    assert alpha_s == pytest.approx(portal, abs=1.0)


@network
def test_4p32_scalar_and_tensor_match_portal_calculator(fresh_cache):
    """Recover alpha0, alpha2 of 4P3/2 from the portal at theta = 0 and 90.

    Without |m| the portal evaluates the tensor term at m = 0, so
    alpha(theta) = alpha0 - (5/4) alpha2 (3 cos^2 theta - 1)/2 for j = 3/2.
    """
    from kamo.light_shift import ComputePolarizabilities

    cp = ComputePolarizabilities(force_arc=False, portal_species="K1")
    th0 = up.dynamic_polarizability("K1", "4p_2P_3-2", theta=0)
    th90 = up.dynamic_polarizability("K1", "4p_2P_3-2", theta=90)
    for L in (800.0, 1064.0, 1550.0):
        a0 = np.interp(L, th0.wavelength_nm, th0.alpha_au)
        a90 = np.interp(L, th90.wavelength_nm, th90.alpha_au)
        alpha2 = (a0 - a90) / (1.5 * -1.25)
        alpha0 = a0 + 1.25 * alpha2
        s, _, t = (float(x[0]) for x in
                   cp.compute_fine_structure_polarizability(4, 1, 1.5, L * 1e-9))
        assert s == pytest.approx(alpha0, rel=0.01)
        assert t == pytest.approx(alpha2, rel=0.01)
