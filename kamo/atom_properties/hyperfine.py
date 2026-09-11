"""Hyperfine A and B constants of potassium, with uncertainties and provenance.

:func:`hyperfine_constants` returns the magnetic-dipole constant A and the
electric-quadrupole constant B for any valence ``(n, l, j)`` state of 39K, 40K
or 41K. A and B are chosen separately. A value for the state itself is
preferred:

1. **measured**: the recommended value from Table 3 of Allegrini, Arimondo &
   Orozco, J. Phys. Chem. Ref. Data 51, 043102 (2022), bundled as
   ``data/k_hfs_survey2022.csv``. That survey supersedes the Arimondo 1977
   review that ARC's ``getHFSCoefficients`` still uses.
2. **theory** (A only, 39K): the all-order values of U. I. Safronova &
   M. S. Safronova, PRA 78, 052504 (2008) for 5-10s, 4-7p and 3-7d. They are
   read from the UDel portal snapshot bundled with kamo (see
   :mod:`kamo.light_shift.udel_portal`), so every machine gets the same
   numbers. Each value is given the fractional uncertainty in
   :data:`THEORY_FRAC_UNC`.

A measurement is kept unless another candidate is at least
:data:`REPLACE_FACTOR` times more precise. Beyond the states theory covers,
model-derived values fill in wherever no measurement is better than
:data:`GOOD_FRAC_UNC`, under the same rule:

3. **extrapolated** (39K). ``A n*^3`` tends to a constant along a Rydberg
   series. ``C(n*) = C_inf + a / n*^2`` is fitted to the three highest-n
   theory values of the ``(l, j)`` series and scaled to the target, with
   ``n*`` taken from ARC's NIST energies. This follows a series that has not
   yet converged (d3/2 is still rising at 7d). The theory uncertainty is kept.
   B has no theory, so ``B n*^3`` is averaged over the highest-n measured
   values instead.
4. **isotope-scaled** (40K, 41K): the 39K value scaled by the ratio of nuclear
   moments. For A that ratio is the measured 4S ratio for s states (it already
   contains the s-state hyperfine anomaly) and mu/I for l >= 1. For B it is the
   ratio of quadrupole moments. These compete with the isotope's own
   measurement the same way 39K theory does.

B is exactly zero for J = 1/2. A measured bound ``|B| < x`` is reported as
``0 +- x`` with source ``"bound"``. If there is no candidate at all, the value
is 0 with a NaN uncertainty and source ``"none"``: A for l >= 3 and for core
states, B for unmeasured d5/2.

Candidates are formed lazily, so the precisely measured 4S and 4P constants
never touch the portal snapshot or ARC. Units are MHz (the ``*_Hz`` properties
convert).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SURVEY_CSV = Path(__file__).parent / "data" / "k_hfs_survey2022.csv"
SURVEY_REF = ("Allegrini, Arimondo & Orozco, J. Phys. Chem. Ref. Data 51, "
              "043102 (2022), Table 3")
THEORY_REF = ("U. I. Safronova & M. S. Safronova, PRA 78, 052504 (2008), "
              "via the UDel portal")

SUPPORTED_ISOTOPES = (39, 40, 41)
_PORTAL_SPECIES = "K1"

#: Lowest valence n of each l in K (4s, 4p, 3d; then n = l + 1). Lower n are
#: core orbitals.
LOWEST_VALENCE_N = {0: 4, 1: 4, 2: 3}

#: Fractional 1-sigma uncertainty given to the portal theory A, keyed by
#: ``(l, 2j)``. Chosen to cover how far theory falls from the survey's
#: measurements (a test checks this):
#:
#: - s states: rms 1.8%; the 10s value is 3.4% off.
#: - p1/2 states: rms 1.1%.
#: - p3/2 states: rms 1.9%, with every one of them low.
#: - d3/2 states: 3d3/2 is 16% off.
#: - d5/2 states: A comes from a near-cancelling core polarisation and
#:   changes sign between theory orders, so these get 30%.
THEORY_FRAC_UNC = {(0, 1): 0.025, (1, 1): 0.02, (1, 3): 0.03,
                   (2, 3): 0.15, (2, 5): 0.30}

#: A measurement this good (fractional 1-sigma) is used as it stands and can
#: anchor the B n*^3 average. Worse ones give way to a model-derived value that
#: is more precise. B values are few and imprecise, so B uses a looser cut.
GOOD_FRAC_UNC = {"A": 0.10, "B": 0.50}

#: A measurement of the state is replaced by theory, extrapolation or isotope
#: scaling only if that value's uncertainty is this many times smaller.
REPLACE_FACTOR = 2.0

#: Smallest fractional uncertainty an extrapolated value may claim.
EXTRAP_FRAC_FLOOR = 0.02

#: Fractional uncertainty of the isotope scaling ratios:
#:
#: - s-state A: the 4S ratio carries the anomaly to 1e-4.
#: - l >= 1 A: mu/I misses the p/d anomaly (the measured 4P ratios agree
#:   within 0.3%).
#: - B: the Q ratio from the portal's moments.
#:
#: The survey's 41K 4P3/2 B, 3.242(22), is 4 sigma off 39K times the Q ratio;
#: Falke's 3.351(71) is not. See the CSV note.
ISOTOPE_RATIO_FRAC_UNC = {"A_s": 1e-4, "A_l": 5e-3, "B": 0.02}

_EXACT = "exact (J = 1/2)"
_L_LETTERS = "spdfghik"


@dataclass(frozen=True)
class HyperfineConstants:
    """Hyperfine constants of one state. Values in MHz, 1-sigma uncertainties.

    ``*_source`` is one of ``"measured"``, ``"bound"``, ``"theory"``,
    ``"extrapolated"``, ``"isotope-scaled"``, ``"exact (J = 1/2)"`` (B only)
    or ``"none"``. ``*_ref`` says where the number comes from.
    """

    iso: int
    n: int
    l: int
    j: float
    A_MHz: float
    A_unc_MHz: float
    A_source: str
    A_ref: str
    B_MHz: float
    B_unc_MHz: float
    B_source: str
    B_ref: str

    @property
    def A_Hz(self) -> float:
        return self.A_MHz * 1e6

    @property
    def B_Hz(self) -> float:
        return self.B_MHz * 1e6

    @property
    def A_unc_Hz(self) -> float:
        return self.A_unc_MHz * 1e6

    @property
    def B_unc_Hz(self) -> float:
        return self.B_unc_MHz * 1e6

    @property
    def has_A(self) -> bool:
        return self.A_source != "none"

    def __str__(self) -> str:
        label = f"{self.iso}K {self.n}{_L_LETTERS[self.l]}{_jstr(self.j)}"
        a = f"A = {self.A_MHz:.9g}({_unc(self.A_unc_MHz)}) MHz [{self.A_source}]"
        b = f"B = {self.B_MHz:.6g}({_unc(self.B_unc_MHz)}) MHz [{self.B_source}]"
        return f"{label}: {a}, {b}"


@dataclass(frozen=True)
class _Candidate:
    value: float
    unc: float
    source: str
    ref: str

    @property
    def frac(self) -> float:
        return abs(self.unc / self.value) if self.value else math.inf


def _jstr(j: float) -> str:
    return f"{int(round(2 * j))}/2"


def _unc(u: float) -> str:
    return "nan" if not np.isfinite(u) else f"{u:.2g}"


def lowest_valence_n(l: int) -> int:
    """Lowest valence principal quantum number of orbital ``l`` in K."""
    return LOWEST_VALENCE_N.get(l, l + 1)


# --------------------------------------------------------------------- data

@functools.lru_cache(maxsize=None)
def _survey() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_CSV, comment="#", skipinitialspace=True)
    df["twoj"] = (2 * df.j).round().astype(int)
    return df.fillna({"sign_from": "", "note": ""})


def survey_table() -> pd.DataFrame:
    """The bundled survey table (MHz), one row per measured state (a copy)."""
    return _survey().copy()


@functools.lru_cache(maxsize=None)
def _survey_index() -> dict:
    return {(int(r.iso), int(r.n), int(r.l), int(r.twoj)): r
            for r in _survey().itertuples()}


@functools.lru_cache(maxsize=None)
def _portal_theory() -> dict:
    """``{(n, l, 2j): A_theory_MHz}`` for 39K from the bundled portal snapshot."""
    from kamo.light_shift import udel_portal
    df = udel_portal.hyperfine_constants(_PORTAL_SPECIES, bundled=True)
    df = df[(df.iso == 39) & df.A_theory_MHz.notna() & df.n.notna()]
    return {(int(r.n), int(r.l), int(round(2 * r.J))): float(r.A_theory_MHz)
            for r in df.itertuples()}


@functools.lru_cache(maxsize=None)
def nuclear_moments(iso: int) -> dict:
    """``{'I', 'mu', 'Q'}`` for ``iso`` from the bundled portal snapshot. Use for
    ratios only (see :func:`kamo.light_shift.udel_portal.nuclear_data`)."""
    from kamo.light_shift import udel_portal
    row = udel_portal.nuclear_data(_PORTAL_SPECIES, bundled=True).query("iso == @iso")
    if row.empty:
        raise KeyError(f"No nuclear data for {iso}K in the portal snapshot.")
    r = row.iloc[0]
    return {"I": float(r.I), "mu": float(r.mu_N), "Q": float(r.Q)}


@functools.lru_cache(maxsize=None)
def _arc():
    import arc
    return arc.Potassium39(preferQuantumDefects=False)


@functools.lru_cache(maxsize=None)
def _effective_n(n: int, l: int, twoj: int) -> float:
    """``n* = sqrt(Ry_K / E_bind)`` from ARC's NIST-level energies."""
    atom = _arc()
    energy_eV = atom.getEnergy(n, l, twoj / 2)        # relative to the ionisation limit
    return math.sqrt(atom.scaledRydbergConstant / -energy_eV)


# --------------------------------------------------------------- candidates

def _measured(iso, n, l, twoj, which):
    row = _survey_index().get((iso, n, l, twoj))
    if row is None:
        return None
    value, unc = getattr(row, f"{which}_MHz"), getattr(row, f"{which}_unc_MHz")
    if not (np.isfinite(value) and np.isfinite(unc)):
        return None
    ref = f"{SURVEY_REF}: {row.ref}"
    if which == "A" and row.sign_from:
        ref += f" (sign from {row.sign_from})"
    return _Candidate(float(value), float(unc), "bound" if value == 0 else "measured", ref)


def _theory(iso, n, l, twoj, which):
    frac = THEORY_FRAC_UNC.get((l, twoj))
    if which != "A" or iso != 39 or frac is None:
        return None
    value = _portal_theory().get((n, l, twoj))
    if value is None:
        return None
    return _Candidate(value, abs(value) * frac, "theory", THEORY_REF)


def _more_precise(*cands):
    cands = [c for c in cands if c is not None]
    return min(cands, key=lambda c: c.unc) if cands else None


def _keep_measured(meas, *others):
    """``meas`` unless another candidate is REPLACE_FACTOR times more precise."""
    other = _more_precise(*others)
    if meas is None or meas.source not in ("measured", "bound"):
        return _more_precise(meas, other)
    if other is not None and other.unc * REPLACE_FACTOR < meas.unc:
        return other
    return meas


def _is_good(cand, which):
    return cand is not None and (cand.source == "theory"
                                 or cand.frac <= GOOD_FRAC_UNC[which])


def _extrapolated(n, l, twoj, which):
    if which == "A" and sum(k[1:] == (l, twoj) for k in _portal_theory()) >= 3:
        return _extrapolated_theory(n, l, twoj)
    return _extrapolated_measured(n, l, twoj, which)


def _extrapolated_theory(n, l, twoj):
    """``A n*^3 = C_inf + a/n*^2`` fitted to the top three theory values."""
    pts = sorted((m, a) for (m, ll, tj), a in _portal_theory().items()
                 if (ll, tj) == (l, twoj))
    if n <= pts[-1][0]:
        return None                           # inside the range theory covers
    top = pts[-3:]
    x = np.array([_effective_n(m, l, twoj) ** -2 for m, _ in top])
    C = np.array([a * _effective_n(m, l, twoj) ** 3 for m, a in top])
    slope, c_inf = np.polyfit(x, C, 1)
    resid = float(np.sqrt(np.mean((C - (c_inf + slope * x)) ** 2)))
    ns = _effective_n(n, l, twoj)
    C_t = c_inf + slope * ns ** -2
    # Half of the fit's own step beyond the last theory point, as a model error.
    model = 0.5 * abs(C_t - C[-1])
    frac = math.hypot(THEORY_FRAC_UNC[(l, twoj)], model / abs(C_t), resid / abs(C_t))
    value = C_t / ns ** 3
    return _Candidate(value, abs(value) * frac, "extrapolated",
                      f"A n*^3 = C_inf + a/n*^2 fitted to theory at 39K n = "
                      f"{', '.join(str(m) for m, _ in top)} [{THEORY_REF}]")


def _extrapolated_measured(n, l, twoj, which):
    """Weighted mean of ``X n*^3`` over the three highest-n good measurements."""
    known_n = sorted({k[1] for k in _survey_index() if k[0] == 39 and k[2:] == (l, twoj)},
                     reverse=True)
    anchors = [(m, c) for m in known_n if m != n
               for c in [_measured(39, m, l, twoj, which)] if _is_good(c, which)][:3]
    if len(anchors) < 2 or n <= min(m for m, _ in anchors):
        return None
    scaled = [(c.value * _effective_n(m, l, twoj) ** 3, c.unc * _effective_n(m, l, twoj) ** 3)
              for m, c in anchors]
    C = np.array([s[0] for s in scaled])
    sC = np.array([s[1] for s in scaled])
    w = 1 / sC ** 2
    c_mean = float(np.sum(w * C) / np.sum(w))
    # The anchors' errors are correlated (one method), so the scatter is not
    # averaged down. Allow for the series to keep drifting by as much as it
    # did across the anchor window.
    c_unc = math.hypot(float(sC.min()), float(C[0] - C[-1]))
    ns3 = _effective_n(n, l, twoj) ** 3
    value = c_mean / ns3
    unc = max(c_unc / ns3, abs(value) * EXTRAP_FRAC_FLOOR)
    return _Candidate(value, unc, "extrapolated",
                      f"{which} n*^3 averaged over measured 39K n = "
                      f"{', '.join(str(m) for m, _ in anchors)} [{SURVEY_REF}]")


def _isotope_scaled(iso, n, l, twoj, which):
    base = _best_39(n, l, twoj, which)
    if base is None or base.source == _EXACT:
        return None
    ref39, refi = nuclear_moments(39), nuclear_moments(iso)
    if which == "B":
        ratio, frac, how = refi["Q"] / ref39["Q"], ISOTOPE_RATIO_FRAC_UNC["B"], "Q ratio"
    elif l == 0:
        ratio = _measured(iso, 4, 0, 1, "A").value / _measured(39, 4, 0, 1, "A").value
        frac, how = ISOTOPE_RATIO_FRAC_UNC["A_s"], "measured 4S A ratio"
    else:
        ratio = (refi["mu"] / refi["I"]) / (ref39["mu"] / ref39["I"])
        frac, how = ISOTOPE_RATIO_FRAC_UNC["A_l"], "mu/I ratio"
    value = base.value * ratio + 0.0         # + 0.0: no "-0" from a zero bound
    unc = math.hypot(base.unc * ratio, value * frac)
    source = "bound" if base.source == "bound" else "isotope-scaled"
    return _Candidate(value, abs(unc), source,
                      f"39K {base.source} value x {how} {ratio:.6f} [{base.ref}]")


def _best_39(n, l, twoj, which):
    if which == "B" and twoj == 1:
        return _Candidate(0.0, 0.0, _EXACT, "")
    meas = _measured(39, n, l, twoj, which)
    # theory is never better than THEORY_FRAC_UNC, so a good measurement
    # within REPLACE_FACTOR of that is kept without the snapshot being read
    floor = THEORY_FRAC_UNC.get((l, twoj), math.inf) if which == "A" else math.inf
    if meas is not None and meas.frac <= min(REPLACE_FACTOR * floor, GOOD_FRAC_UNC[which]):
        return meas
    own = _keep_measured(meas, _theory(39, n, l, twoj, which))
    if _is_good(own, which):
        return own
    return _keep_measured(own, _extrapolated(n, l, twoj, which))


def _best(iso, n, l, twoj, which):
    if iso == 39:
        return _best_39(n, l, twoj, which)
    if which == "B" and twoj == 1:
        return _Candidate(0.0, 0.0, _EXACT, "")
    meas = _measured(iso, n, l, twoj, which)
    floor = (ISOTOPE_RATIO_FRAC_UNC["B"] if which == "B" else
             ISOTOPE_RATIO_FRAC_UNC["A_s" if l == 0 else "A_l"])
    if meas is not None and meas.frac <= REPLACE_FACTOR * floor:
        return meas                           # scaling can't do better enough
    return _keep_measured(meas, _isotope_scaled(iso, n, l, twoj, which))


# ----------------------------------------------------------------- public API

def hyperfine_constants(n: int, l: int, j: float, iso: int = 39) -> HyperfineConstants:
    """A and B for state ``(n, l, j)`` of ``iso``K. See the module docstring
    for how the values are chosen. Core states (n below
    :func:`lowest_valence_n`) have source ``"none"``."""
    if n != int(n) or l != int(l) or int(n) <= int(l) or l < 0:
        raise ValueError(f"Invalid (n, l) = ({n}, {l}).")
    n, l, twoj = int(n), int(l), int(round(2 * j))
    if twoj <= 0 or abs(twoj - 2 * l) != 1 or abs(2 * j - twoj) > 1e-9:
        raise ValueError(f"j = {j} is not l +- 1/2 for l = {l}.")
    if iso not in SUPPORTED_ISOTOPES:
        raise ValueError(f"iso must be one of {SUPPORTED_ISOTOPES}, got {iso}.")
    return _hyperfine_constants(n, l, twoj, int(iso))


@functools.lru_cache(maxsize=None)
def _hyperfine_constants(n, l, twoj, iso):
    if n < lowest_valence_n(l):
        none = _Candidate(0.0, math.nan, "none", "core orbital, not a valence state")
        return HyperfineConstants(iso, n, l, twoj / 2, *vars(none).values(),
                                  *vars(none).values())
    A = _best(iso, n, l, twoj, "A") or _Candidate(0.0, math.nan, "none", "no data")
    B = (_best(iso, n, l, twoj, "B")
         or _Candidate(0.0, math.nan, "none", "no data; B taken as 0"))
    return HyperfineConstants(iso, n, l, twoj / 2, A.value, A.unc, A.source, A.ref,
                              B.value, B.unc, B.source, B.ref)


def hyperfine_energy(F: float, I: float, J: float, A: float, B: float = 0.0) -> float:
    """Zero-field hyperfine energy of level ``F``, in the units of A and B:
    ``A K/2 + B [3/2 K(K+1) - 2 I(I+1) J(J+1)] / [4 I(2I-1) J(2J-1)]``, with
    ``K = F(F+1) - I(I+1) - J(J+1)``."""
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    E = A * K / 2
    if B and I > 0.5 and J > 0.5:
        E += B * (1.5 * K * (K + 1) - 2 * I * (I + 1) * J * (J + 1)) / (
            4 * I * (2 * I - 1) * J * (2 * J - 1))
    return E


def hyperfine_table(iso: int = 39, n_max: int = 10, l_max: int = 2) -> pd.DataFrame:
    """Every valence state up to ``n_max`` and ``l_max``, one row each."""
    rows = []
    for l in range(l_max + 1):
        for twoj in sorted({abs(2 * l - 1), 2 * l + 1}):
            for n in range(lowest_valence_n(l), n_max + 1):
                hc = hyperfine_constants(n, l, twoj / 2, iso)
                rows.append(dict(state=f"{n}{_L_LETTERS[l]}{_jstr(twoj / 2)}",
                                 A_MHz=hc.A_MHz, A_unc_MHz=hc.A_unc_MHz,
                                 A_source=hc.A_source, B_MHz=hc.B_MHz,
                                 B_unc_MHz=hc.B_unc_MHz, B_source=hc.B_source))
    return pd.DataFrame(rows)
