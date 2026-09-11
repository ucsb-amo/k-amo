"""Client for the University of Delaware atomic data portal (https://www.udel.edu/atom).

The portal website is a static front end over an open (undocumented) GraphQL API,
so no HTML scraping is needed:

- ``GRAPHQL_URL`` serves energies, reduced E1 matrix elements, transition rates,
  static polarizabilities, hyperfine constants, ... for ~40 atoms and ions.
  Species titles are ``<symbol><ion stage + 1>``: ``K1`` (neutral K), ``Ca2`` (Ca+).
- ``POL_URL`` is the portal's on-the-fly dynamic polarizability calculator,
  useful as an independent check.

Gotchas:

- Every matrix-element row is listed twice, once per direction, with identical
  values. :func:`matrix_elements` keeps both (the ``Initial``/``Final`` lookup in
  :class:`~kamo.light_shift.PortalDataParser` needs that); use
  :func:`dedupe_pairs` before summing over states directly.
- Matrix elements are magnitudes (no sign), which is fine for polarizabilities.
- Static polarizabilities come back only as display strings, and the portal
  shows them in units of 10^3 a.u. for some species (e.g. K 4s: "0.29025(25)").

Responses are cached as JSON under ``$KAMO_CACHE_DIR`` (default
``~/.cache/kamo/udel_portal``); pass ``refresh=True`` to re-fetch. If the portal
is unreachable and nothing is cached, the snapshot bundled in
``kamo/light_shift/data/udel_portal`` is used (regenerate with
:func:`write_snapshot`).

Data are CC BY 4.0. Cite: P. Barakhshan et al., "Portal for high-precision
atomic data and computation", Comput. Phys. Commun. (2025), arXiv:2506.08170.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://atom.ece.udel.edu/graphql"
POL_URL = "https://atom.ece.udel.edu/get_pol/get_pol_data"
CITATION = ("P. Barakhshan et al., Portal for high-precision atomic data and "
            "computation, Comput. Phys. Commun. (2025), arXiv:2506.08170; "
            "https://www.udel.edu/atom (CC BY 4.0)")

_SNAPSHOT_DIR = Path(__file__).parent / "data" / "udel_portal"
_TIMEOUT_S = 60
_HEADERS = {"User-Agent": "kamo-udel-portal-client (UCSB Weld lab)",
            "Content-Type": "application/json"}
_L_LETTERS = "spdfghik"

_MATRIX_ELEMENT_FIELDS = """
    stateOneConfiguration stateOneTerm stateOneJ
    stateTwoConfiguration stateTwoTerm stateTwoJ
    wavelength wavelengthUncertainty
    matrixElement matrixElementUncertainty matrixElementRef
"""


# --------------------------------------------------------------------- transport

def cache_dir():
    """Directory holding cached portal responses."""
    root = os.environ.get("KAMO_CACHE_DIR")
    root = Path(root) if root else Path.home() / ".cache" / "kamo"
    return root / "udel_portal"


def _http(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=_HEADERS,
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.load(resp)


def _graphql(query, variables=None):
    out = _http(GRAPHQL_URL, {"query": query, "variables": variables or {}})
    if out.get("errors"):
        raise RuntimeError(f"UDel portal GraphQL error: {out['errors']}")
    return out["data"]


def _cached(key, fetch, refresh=False, source=GRAPHQL_URL, query=None):
    """Return ``fetch()``'s result, cached as ``<cache_dir>/<key>.json``."""
    path = cache_dir() / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())["data"]
    try:
        t0 = time.perf_counter()
        data = fetch()
        logger.info("UDel portal: fetched %s in %.2f s", key, time.perf_counter() - t0)
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        for fallback in (path, _SNAPSHOT_DIR / f"{key}.json"):
            if fallback.exists():
                warnings.warn(f"UDel portal unreachable ({err}); using {fallback}")
                return json.loads(fallback.read_text())["data"]
        raise ConnectionError(
            f"Could not reach the UDel portal ({err}), and there is no cache at "
            f"{path} and no bundled snapshot for {key!r}.") from err
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "query": query, "citation": CITATION, "data": data}))
    return data


def _element(species, fields, key, refresh=False, bundled=False):
    """``element(title)`` fields, from cache/network, or with ``bundled=True``
    straight from the snapshot shipped with kamo (no cache, no network)."""
    if bundled:
        path = _SNAPSHOT_DIR / f"{species}_{key}.json"
        if not path.exists():
            raise KeyError(f"No bundled snapshot {path.name}; see write_snapshot().")
        return json.loads(path.read_text())["data"]
    query = f'query($t: String) {{ element(title: $t) {{ {fields} }} }}'
    data = _cached(f"{species}_{key}",
                   lambda: _graphql(query, {"t": species})["element"],
                   refresh=refresh, query=query)
    if data is None:
        raise KeyError(f"No species {species!r} on the UDel portal; see list_species().")
    return data


# ---------------------------------------------------------------------- parsing

def _j_to_float(j):
    num, _, den = str(j).partition("/")
    return float(num) / float(den) if den else float(num)


def state_label(configuration, j):
    """Portal configuration + J -> kamo label, e.g. ("4p", "3/2") -> "4p3/2"."""
    return f"{configuration}{j}"


def _valence_nl(configuration):
    """(n, l) of a single-valence configuration like "4p"; (nan, nan) otherwise."""
    m = re.fullmatch(r"(\d+)([a-z])", configuration or "")
    if m is None or m.group(2) not in _L_LETTERS:
        return np.nan, np.nan
    return int(m.group(1)), _L_LETTERS.index(m.group(2))


def _state_record(i, configuration, term, j):
    """Columns describing one state of a transition row, suffixed with ``i``."""
    n, l = _valence_nl(configuration)
    return {f"state{i}": state_label(configuration, j), f"config{i}": configuration,
            f"term{i}": term, f"J{i}": _j_to_float(j), f"n{i}": n, f"l{i}": l}


def parse_display_value(text):
    """Parse a portal display string "0.29025(25)" -> (0.29025, 0.00025)."""
    m = re.fullmatch(r"\s*(-?[\d.]+)(?:\((\d+)\))?\s*", text or "")
    if m is None:
        return np.nan, np.nan
    value = m.group(1)
    if m.group(2) is None:
        return float(value), np.nan
    decimals = len(value.split(".")[1]) if "." in value else 0
    return float(value), int(m.group(2)) * 10.0 ** -decimals


# --------------------------------------------------------------------- datasets

def list_species(refresh=False):
    """DataFrame of portal species and which datasets each one has."""
    query = ("{ elements { title titleDisplay isMatrixElements isPolarizabilities "
             "isStaticPolarizabilities isEnergies isHCI isHidden } }")
    rows = _cached("species", lambda: _graphql(query)["elements"],
                   refresh=refresh, query=query)
    df = pd.DataFrame(rows)
    df["name"] = df.pop("titleDisplay").str.replace(r"</?sup>", "", regex=True)
    return df.sort_values("title", ignore_index=True)


def matrix_elements(species="K1", refresh=False):
    """Reduced E1 matrix elements (a.u.) for ``species``, both directions.

    Columns: ``state1, state2`` (labels like "4p1/2"), ``config1, term1, J1``,
    ``config2, term2, J2``, ``n1, l1, n2, l2`` (NaN for multi-valence
    configurations), ``d_au, d_unc_au, wavelength_nm, wavelength_unc_nm``,
    ``source`` ("exp" if the portal cites a measurement, else "theory"), ``ref``.
    """
    rows = _element(species, f"matrixElements {{ {_MATRIX_ELEMENT_FIELDS} }}",
                    "matrix_elements", refresh)["matrixElements"]
    if not rows:
        raise KeyError(f"The UDel portal has no matrix elements for {species!r}.")
    records = []
    for r in rows:
        rec = {}
        for i, side in ((1, "One"), (2, "Two")):
            rec.update(_state_record(i, r[f"state{side}Configuration"],
                                     r[f"state{side}Term"], r[f"state{side}J"]))
        ref = r["matrixElementRef"] or ""
        rec.update(d_au=r["matrixElement"], d_unc_au=r["matrixElementUncertainty"],
                   wavelength_nm=r["wavelength"],
                   wavelength_unc_nm=r["wavelengthUncertainty"],
                   source="exp" if ref else "theory", ref=ref)
        records.append(rec)
    return pd.DataFrame(records)


def dedupe_pairs(df):
    """Collapse the portal's two-direction listing to one row per unordered pair."""
    key = df.apply(lambda r: tuple(sorted((r.state1, r.state2))), axis=1)
    return df.loc[~key.duplicated()].reset_index(drop=True)


def to_legacy_table(df):
    """Convert :func:`matrix_elements` output to the ``PortalDataParser`` schema.

    Returns columns ``Initial, Final, Matrix element (a.u.), Wavelength (nm)``,
    with both directions present.
    """
    if df[["n1", "n2"]].isna().any().any():
        raise ValueError("to_legacy_table only supports single-valence species "
                         "(states like '4p3/2').")
    legacy = df.rename(columns={"state1": "Initial", "state2": "Final",
                                "d_au": "Matrix element (a.u.)",
                                "wavelength_nm": "Wavelength (nm)"})
    legacy = legacy[["Initial", "Final", "Matrix element (a.u.)", "Wavelength (nm)"]]
    # the portal already lists both directions, but make that a guarantee
    flipped = legacy.rename(columns={"Initial": "Final", "Final": "Initial"})
    both = pd.concat([legacy, flipped], ignore_index=True)
    return both.drop_duplicates(subset=["Initial", "Final"], ignore_index=True)


def transition_rates(species="K1", refresh=False):
    """Einstein A coefficients and upper-state lifetimes for ``species``.

    One row per decay channel ``state1 -> state2`` (state 1 is the upper,
    decaying state). Columns: the state columns of :func:`matrix_elements`,
    ``A_s, A_unc_s`` (s^-1), ``branching_ratio``, ``wavelength_nm``, ``d_au``,
    and the upper-state lifetime ``tau_s, tau_unc_s, tau_ref`` (the portal
    quotes a measured lifetime where one exists; ``tau_ref`` is then non-empty).
    """
    rows = _element(species,
                    "transitionRates { stateOneConfiguration stateOneTerm stateOneJ "
                    "stateTwoConfiguration stateTwoTerm stateTwoJ wavelength "
                    "matrixElement transitionRate transitionRateUncertainty "
                    "branchingRatio lifetime lifetimeUncertainty lifetimeRef }",
                    "transition_rates", refresh)["transitionRates"]
    if not rows:
        raise KeyError(f"The UDel portal has no transition rates for {species!r}.")
    ns = 1e-9
    records = []
    for r in rows:
        rec = {}
        for i, side in ((1, "One"), (2, "Two")):
            rec.update(_state_record(i, r[f"state{side}Configuration"],
                                     r[f"state{side}Term"], r[f"state{side}J"]))
        rec.update(A_s=r["transitionRate"], A_unc_s=r["transitionRateUncertainty"],
                   branching_ratio=r["branchingRatio"], wavelength_nm=r["wavelength"],
                   d_au=r["matrixElement"],
                   tau_s=None if r["lifetime"] is None else r["lifetime"] * ns,
                   tau_unc_s=None if r["lifetimeUncertainty"] is None
                   else r["lifetimeUncertainty"] * ns,
                   tau_ref=r["lifetimeRef"] or "")
        records.append(rec)
    return pd.DataFrame(records)


def lifetimes(species="K1", refresh=False):
    """Radiative lifetime of every decaying state (one row per state).

    Columns: ``state, config, J, n, l, tau_s, tau_unc_s, linewidth_Hz``
    (Gamma / 2 pi = 1 / (2 pi tau)), ``source`` ("exp" or "theory"), ``ref``.
    """
    tr = transition_rates(species, refresh)
    up = tr.drop_duplicates("state1").rename(columns={
        "state1": "state", "config1": "config", "J1": "J", "n1": "n", "l1": "l",
        "tau_ref": "ref"})
    up = up[["state", "config", "J", "n", "l", "tau_s", "tau_unc_s", "ref"]].copy()
    up["linewidth_Hz"] = 1 / (2 * np.pi * up.tau_s)
    up["source"] = np.where(up.ref != "", "exp", "theory")
    return up.reset_index(drop=True)


def hyperfine_constants(species="K1", refresh=False, bundled=False):
    """Magnetic-dipole hyperfine A constants (MHz), one row per (isotope, state).

    ``bundled=True`` reads the snapshot shipped with kamo, which gives the same
    answer on every machine.

    Columns: ``iso`` (mass number), ``state, config, J, n, l``,
    ``A_theory_MHz, theory_ref``, ``A_exp_MHz, A_exp_unc_MHz, exp_ref``.

    Gotchas in the raw rows, handled here:

    - The isotope is filled in only on the first row of each isotope group,
      and the groups are separated by blank rows. Both are fixed up here.
    - There is no electric-quadrupole B constant.
    - Theory values (for K: U. I. Safronova & M. S. Safronova, PRA 78, 052504
      (2008)) exist only for the main isotope.
    - The experimental column drops the sign of measurements that give only
      |A|: the portal lists 39K 3d5/2 as +0.62 and 40K 3d3/2 as +1.07, but both
      are negative. :mod:`kamo.atom_properties.hyperfine` therefore takes
      measured values from its own survey table and uses this for theory.
    """
    rows = _element(species,
                    "hyperfineConstants { isotopeMassNumber stateConfiguration "
                    "stateTerm stateJ hyperfineTheory hyperfineTheoryRef "
                    "hyperfineExperiment hyperfineExperimentUncertainty "
                    "hyperfineExperimentRef }",
                    "hyperfine_constants", refresh, bundled)["hyperfineConstants"]
    if not rows:
        raise KeyError(f"The UDel portal has no hyperfine constants for {species!r}.")
    records, iso = [], None
    for r in rows:
        if r["isotopeMassNumber"]:
            iso = int(r["isotopeMassNumber"])
        if not r["stateConfiguration"]:          # separator row
            continue
        n, l = _valence_nl(r["stateConfiguration"])
        unc = r["hyperfineExperimentUncertainty"]
        records.append(dict(
            iso=iso, state=state_label(r["stateConfiguration"], r["stateJ"]),
            config=r["stateConfiguration"], J=_j_to_float(r["stateJ"]), n=n, l=l,
            A_theory_MHz=r["hyperfineTheory"], theory_ref=r["hyperfineTheoryRef"] or "",
            A_exp_MHz=r["hyperfineExperiment"],
            A_exp_unc_MHz=float(unc) if unc else np.nan,
            exp_ref=r["hyperfineExperimentRef"] or ""))
    return pd.DataFrame(records)


def nuclear_data(species="K1", refresh=False, bundled=False):
    """Nuclear spin, magnetic dipole moment (nuclear magnetons) and electric
    quadrupole moment for every isotope the portal lists.

    Columns: ``iso, I, mu_N, mu_unc, Q, Q_unc, abundance, half_life``.
    The moments are the portal's values. Use them for isotope ratios only: the
    portal's Q unit is not stated, and ``mu`` has no diamagnetic-shielding
    correction.
    """
    rows = _element(species,
                    "nuclears { isotopeMassNumber nuclearSpin magneticMoment "
                    "magneticMomentUncertainty quadrupoleMoment "
                    "quadrupoleMomentUncertainty naturalAbundance halfLife }",
                    "nuclears", refresh, bundled)["nuclears"]
    if not rows:
        raise KeyError(f"The UDel portal has no nuclear data for {species!r}.")
    return pd.DataFrame([dict(
        iso=int(r["isotopeMassNumber"]), I=_j_to_float(r["nuclearSpin"]),
        mu_N=r["magneticMoment"], mu_unc=r["magneticMomentUncertainty"],
        Q=r["quadrupoleMoment"], Q_unc=r["quadrupoleMomentUncertainty"],
        abundance=r["naturalAbundance"], half_life=r["halfLife"]) for r in rows])


def write_snapshot(species="K1", datasets=("matrix_elements", "transition_rates",
                                           "hyperfine_constants", "nuclears")):
    """Re-fetch ``datasets`` for ``species`` and copy them into the bundled
    snapshot directory (the offline fallback shipped with kamo)."""
    fetchers = {"matrix_elements": matrix_elements, "transition_rates": transition_rates,
                "energies": energies, "static_polarizabilities": static_polarizabilities,
                "hyperfine_constants": hyperfine_constants, "nuclears": nuclear_data}
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name in datasets:
        fetchers[name](species, refresh=True)
        src = cache_dir() / f"{species}_{name}.json"
        dst = _SNAPSHOT_DIR / src.name
        dst.write_text(src.read_text())
        written.append(dst)
    return written


def energies(species="K1", refresh=False):
    """Level energies in cm^-1 above the ground state."""
    rows = _element(species,
                    "energies { stateConfiguration stateTerm stateJ energy "
                    "energyUncertainty isFromTheory }",
                    "energies", refresh)["energies"]
    df = pd.DataFrame(rows)
    df.insert(0, "state", [state_label(c_, j) for c_, j in
                           zip(df.stateConfiguration, df.stateJ)])
    return df.rename(columns={"stateConfiguration": "config", "stateTerm": "term",
                              "stateJ": "J", "energy": "energy_cm",
                              "energyUncertainty": "energy_unc_cm"})


def static_polarizabilities(species="K1", refresh=False):
    """Static scalar/tensor polarizabilities as displayed on the portal.

    Values are parsed from the display strings and are in the portal's display
    units, which are 10^3 a.u. for some species (check against a known value).
    """
    rows = _element(species,
                    "staticPolarizabilities { stateConfiguration stateJ "
                    "alpha0Display alpha2Display }",
                    "static_polarizabilities", refresh)["staticPolarizabilities"]
    records = []
    for r in rows:
        a0, a0_unc = parse_display_value(r["alpha0Display"])
        a2, a2_unc = parse_display_value(r["alpha2Display"])
        records.append(dict(state=state_label(r["stateConfiguration"], r["stateJ"]),
                            alpha0=a0, alpha0_unc=a0_unc,
                            alpha2=a2, alpha2_unc=a2_unc))
    return pd.DataFrame(records)


def dynamic_polarizability(species="K1", state="4s_2S_1-2", theta=0, refresh=False):
    """The portal calculator's alpha(lambda) in a.u., 600-1800 nm.

    ``state`` uses the portal's "new notation" (``<config>_<term>_<2J>-2``,
    e.g. "4p_2P_3-2"). Points the portal leaves blank (at resonances) are dropped.

    The server rejects the "|m|=..." state forms its own front end lists (HTTP
    500). Without |m| it evaluates the tensor term at m = 0:
    ``alpha = alpha0 - j(j+1)/(j(2j-1)) * alpha2 * (3 cos^2 theta - 1)/2``.
    That is -5/4 alpha2 at theta = 0 for j = 3/2, which is not a physical
    sublevel. Combine theta = 0 and 90 to recover alpha0 and alpha2.
    """
    params = urllib.parse.urlencode({"element": species, "state": state, "theta": theta})
    url = f"{POL_URL}?{params}"
    rows = _cached(f"{species}_dynpol_{state}_theta{theta}",
                   lambda: _http(url), refresh=refresh, source=url)
    df = pd.DataFrame(rows)
    df = df[pd.to_numeric(df.alpha_pol, errors="coerce").notna()]
    return (df.astype({"alpha_pol": float, "wavelength": float})
              .rename(columns={"wavelength": "wavelength_nm", "alpha_pol": "alpha_au"})
              .sort_values("wavelength_nm", ignore_index=True)
              [["wavelength_nm", "alpha_au"]])
