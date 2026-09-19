"""ARC alkali atoms with UDel-portal E1 data and kamo's structure machinery.

:class:`PortalAlkali` is the species-agnostic layer. Mixed in front of an ARC
alkali class it supplies

* portal E1 matrix elements, rates and lifetimes (``use_portal=True``);
* the atom facts kamo's other modules read: nuclear spin ``I``, the shielded
  nuclear g-factor ``gI``, the electronic g-factor :meth:`g_J`, the ground
  state, the lowest valence n of each l, hyperfine constants;
* every kamo structure method (Zeeman shifts, Breit-Rabi, transition
  frequencies, light shifts, cross sections, ...) written against those facts.

One concrete class per isotope is defined at the bottom of this module
(``Lithium6`` ... ``Caesium``); :class:`~kamo.atom_properties.k39.Potassium39`
lives in its own module because it carries potassium-only extras (the curated
hyperfine module, scattering lengths). :func:`atom` builds one from a species
string (``"Rb87"``).

All quantities that ARC builds from its radial matrix element
(``getDipoleMatrixElement``, ``getReducedMatrixElementJ``, Rabi couplings,
...) use the portal's recommended magnitude; the sign still comes from ARC,
which the portal does not provide. At zero temperature ``getTransitionRate``
and ``getStateLifetime`` return the portal's rates and lifetimes (measured
where they exist). Transitions or states the portal lacks fall back to ARC.
``use_portal=False`` gives ARC's own E1 and hyperfine data.

Energies are ARC's tabulated NIST levels (``preferQuantumDefects=False``).
That matters only for Na and K, where ARC's default computes every level from
Rydberg quantum defects instead (K: D2 2.55 GHz and D1 4.71 GHz too high, 3D
67 cm^-1 too low). Li, Rb and Cs always use NIST levels below
``minQuantumDefectN``. ARC's level tables are per element, so isotope shifts
are absent: Li6 D lines are 10.8 GHz off, K40/K41 0.1-0.2 GHz.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Optional

import arc
from arc.wigner import Wigner6j, Wigner3j
import numpy as np
import kamo.constants as c

# ARC's sign is trusted when its magnitude is within this factor of the portal's.
_PORTAL_SIGN_TOLERANCE = 1.25

# kamo.hamiltonian (exact diagonalization with nuclear spin) is used for states
# with n below groundStateN + this; pairinteraction (no nuclear spin) above.
_HAMILTONIAN_N_ABOVE_GROUND = 6

_L_LETTERS = "spdfghik"


def _angular_factor(l1, j1, l2, j2, s=0.5):
    """|<j1||er||j2>| / |radial element|, in ARC's convention."""
    return abs(np.sqrt((2*j1 + 1) * (2*j2 + 1) * (2*l1 + 1) * (2*l2 + 1))
               * Wigner6j(j1, 1, j2, l2, s, l1) * Wigner3j(l1, 1, l2, 0, 0, 0))


def _own_manifolds(*states):
    """The ``(n, l, j)`` manifolds of ``states`` plus their fine-structure
    partners (same n and l, other j), in first-seen order."""
    out = []
    for st in states:
        n, l = int(st[0]), int(st[1])
        js = (float(st[2]),) if l == 0 else (l - 0.5, l + 0.5)
        for j in js:
            if (n, l, j) not in out:
                out.append((n, l, j))
    return out


def _crossings(x, y, target):
    """Every ``x`` where the sampled curve ``y(x)`` equals ``target``.

    Roots are linear interpolations across each sign change of
    ``y - target``; grid points landing exactly on the target are included
    too.  Returns a sorted, de-duplicated ndarray (empty when the target is
    never reached).
    """
    x = np.asarray(x, dtype=float)
    diff = np.asarray(y, dtype=float) - float(target)
    k = np.where(np.diff(np.sign(diff)) != 0)[0]
    roots = []
    for i in k:
        x0, x1, y0, y1 = x[i], x[i + 1], diff[i], diff[i + 1]
        roots.append(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0))
    roots.extend(x[j] for j in np.where(diff == 0)[0])
    return np.unique(np.round(roots, 9))


# ------------------------------------------------------------ isotope table

#: Ionic-core static polarizabilities (a.u.): Safronova, Johnson & Derevianko,
#: PRA 60, 4476 (1999), Table I. Treated as static; the core resonances lie
#: near 20 eV, so the frequency dependence is negligible above ~300 nm.
CORE_POLARIZABILITY_AU = {"Li": 0.1894, "Na": 0.9457, "K": 5.457,
                          "Rb": 9.076, "Cs": 15.81}
CORE_POLARIZABILITY_REF = "Safronova, Johnson & Derevianko, PRA 60, 4476 (1999)"

_STECK_RB87 = "D. A. Steck, Rubidium 87 D Line Data (rev. 2.3.3, 2024)"
_STECK_RB85 = "D. A. Steck, Rubidium 85 D Line Data (rev. 2.3.3, 2024)"
_STECK_CS = "D. A. Steck, Cesium D Line Data (rev. 2.3.3, 2024)"
_STECK_NA = "D. A. Steck, Sodium D Line Data (rev. 2.3.3, 2024)"
_GEHM_LI6 = "M. E. Gehm, Properties of 6Li (2003)"
_ARIMONDO = "Arimondo, Inguscio & Violino, Rev. Mod. Phys. 49, 31 (1977)"
_TIECKE_K = "T. G. Tiecke, Properties of Potassium (v1.03, 2019)"


@dataclass(frozen=True)
class IsotopeData:
    """Reference constants for one alkali isotope.

    ``g_I`` is the total nuclear g-factor in the convention
    ``H = mu_B B (g_J J_z + g_I I_z)``, the atomic (diamagnetically shielded)
    value; it is what a Breit-Rabi field calibration needs. ``g_J_ground`` is
    the measured ground-state electronic g-factor. Both carry their reference.
    """

    species: str          #: e.g. "Rb87"
    element: str          #: e.g. "Rb"
    isotope: int          #: mass number
    arc_class: str        #: name of the class in the ``arc`` package
    g_I: float
    g_I_ref: str
    g_J_ground: float
    g_J_ref: str

    @property
    def portal_species(self) -> str:
        return f"{self.element}1"

    @property
    def core_polarizability_au(self) -> float:
        return CORE_POLARIZABILITY_AU[self.element]


#: Reference constants per isotope, keyed by species string.
ISOTOPES = {d.species: d for d in (
    IsotopeData("Li6", "Li", 6, "Lithium6", -0.0004476540, _GEHM_LI6,
                2.0023010, _GEHM_LI6),
    IsotopeData("Li7", "Li", 7, "Lithium7", -0.0011822130, _ARIMONDO,
                2.0023010, _ARIMONDO),
    IsotopeData("Na23", "Na", 23, "Sodium", -0.00080461080, _STECK_NA,
                2.0022960, _STECK_NA),
    IsotopeData("K39", "K", 39, "Potassium39", -0.00014193489, _ARIMONDO,
                2.00229421, _ARIMONDO),
    IsotopeData("K40", "K", 40, "Potassium40", +0.000176490, _TIECKE_K,
                2.00229421, _TIECKE_K),
    IsotopeData("K41", "K", 41, "Potassium41", -0.00007790600, _TIECKE_K,
                2.00229421, _TIECKE_K),
    IsotopeData("Rb85", "Rb", 85, "Rubidium85", -0.00029364000, _STECK_RB85,
                2.00233113, _STECK_RB85),
    IsotopeData("Rb87", "Rb", 87, "Rubidium87", -0.0009951414, _STECK_RB87,
                2.00233113, _STECK_RB87),
    IsotopeData("Cs133", "Cs", 133, "Caesium", -0.00039885395, _STECK_CS,
                2.00254032, _STECK_CS),
)}

#: Accepted spellings of a species string, e.g. ``"Rb-87"``, ``"87Rb"``, ``"rb87"``.
_ALIASES = {"Cs": "Cs133", "Na": "Na23", "Cesium": "Cs133", "Caesium": "Cs133",
            "Sodium": "Na23"}


def normalize_species(species: str) -> str:
    """``"87Rb"``, ``"Rb-87"``, ``"rb87"``, ``"Rubidium87"`` -> ``"Rb87"``."""
    s = str(species).strip().replace("-", "").replace("_", "").replace(" ", "")
    if s in ISOTOPES:
        return s
    if s in _ALIASES:
        return _ALIASES[s]
    digits = "".join(ch for ch in s if ch.isdigit())
    letters = "".join(ch for ch in s if ch.isalpha())
    for d in ISOTOPES.values():
        if digits == str(d.isotope) and letters.lower() in (
                d.element.lower(), d.arc_class.lower().rstrip("0123456789")):
            return d.species
    raise KeyError(f"Unknown alkali species {species!r}; "
                   f"known: {', '.join(ISOTOPES)}.")


def lowest_valence_n(atom_or_cls, l):
    """Lowest valence principal quantum number of orbital ``l`` for an atom or
    atom class (K: 4s, 4p, 3d, 4f, ...), from ARC's class-level
    ``groundStateN`` and ``extraLevels``. Lower n are core orbitals."""
    l = int(l)
    extra = [int(n) for n, ll, _j in atom_or_cls.extraLevels if int(ll) == l]
    if extra:
        return min(extra)
    return max(int(atom_or_cls.groundStateN), l + 1)


def lande_g_j(l, j, s=0.5, gL=1.0, gS=c.g_S):
    """Landé g_J (positive), ``g_L [J(J+1)-S(S+1)+L(L+1)]/2J(J+1) +
    g_S [J(J+1)+S(S+1)-L(L+1)]/2J(J+1)``."""
    jj, ss, ll = j * (j + 1), s * (s + 1), l * (l + 1)
    return (gL * (jj - ss + ll) + gS * (jj + ss - ll)) / (2 * jj)


# ------------------------------------------------------- duck-typed facts
#
# kamo.hamiltonian and kamo.trap accept any ARC-like atom. These helpers read
# the facts they need from a PortalAlkali, a plain ARC atom, or a test stub.

def nuclear_spin(atom) -> float:
    return float(atom.I)


def nuclear_g(atom) -> float:
    return float(atom.gI)


def electronic_g(atom, l, j, n=None, s=0.5) -> float:
    """``atom.g_J`` when it has one, else Landé with the atom's gL and gS."""
    g = getattr(atom, "g_J", None)
    if callable(g):
        return float(g(l, j, n=n, s=s))
    return lande_g_j(l, j, s, getattr(atom, "gL", 1.0), getattr(atom, "gS", c.g_S))


def hyperfine(atom, n, l, j):
    """Hyperfine constants of ``(n, l, j)`` as an object with ``A_MHz``,
    ``B_MHz``, ``A_Hz``, ``B_Hz`` and ``has_A``: ``atom.hyperfine_constants``
    when the atom has it, else ARC's ``getHFSCoefficients`` (``has_A`` False
    when ARC has no entry)."""
    f = getattr(atom, "hyperfine_constants", None)
    if callable(f):
        return f(n, l, j)
    try:
        A_Hz, B_Hz = atom.getHFSCoefficients(n, l, j)
    except (ValueError, KeyError, AttributeError):
        return HFS.none(n, l, j)
    return HFS(n, l, float(j), A_Hz / 1e6, float("nan"), "arc",
               B_Hz / 1e6, float("nan"), "arc")


def species_of(atom) -> str:
    """A hashable token identifying the atom's species: ``atom.species`` when
    present, else the class name (``"Rubidium87"`` for a plain ARC atom)."""
    return str(getattr(atom, "species", None) or type(atom).__name__)


_DEFAULT = {}


def default_atom():
    """The process-wide default atom (a :class:`~kamo.atom_properties.k39.Potassium39`),
    built on first use. Used wherever kamo takes ``atom=None``."""
    if "atom" not in _DEFAULT:
        from kamo.atom_properties.k39 import Potassium39
        _DEFAULT["atom"] = Potassium39()
    return _DEFAULT["atom"]


def atom_class(species: str):
    """The kamo class for a species string (``"Rb87"`` -> ``Rubidium87``)."""
    name = ISOTOPES[normalize_species(species)].arc_class
    if name == "Potassium39":
        from kamo.atom_properties.k39 import Potassium39
        return Potassium39
    return globals()[name]


def atom(species: str, **kwargs):
    """Build an atom from its species string: ``atom("Rb87")``.

    ``kwargs`` go to the class (``use_portal``, ``preferQuantumDefects``).
    """
    return atom_class(species)(**kwargs)


_SHARED = {}


def shared_atom(species: str):
    """One cached default-configured instance per species, for code that
    needs a manifold's hyperfine order or nuclear spin but is handed only a
    species token (label caches)."""
    key = normalize_species(species)
    if key not in _SHARED:
        _SHARED[key] = atom(key)
    return _SHARED[key]


# ------------------------------------------------------- hyperfine record

@dataclass(frozen=True)
class HFS:
    """Hyperfine ``A`` and ``B`` of one state (MHz) with their provenance.

    Same read-only surface as :class:`kamo.atom_properties.hyperfine.HyperfineConstants`
    (``A_Hz``, ``B_Hz``, ``has_A``, ...). ``*_source`` is ``"portal"``,
    ``"arc"``, ``"exact (J = 1/2)"`` or ``"none"``.
    """

    n: int
    l: int
    j: float
    A_MHz: float
    A_unc_MHz: float
    A_source: str
    B_MHz: float
    B_unc_MHz: float
    B_source: str
    A_ref: str = ""
    B_ref: str = ""
    iso: int = 0

    @classmethod
    def none(cls, n, l, j, iso=0):
        return cls(n, l, float(j), 0.0, float("nan"), "none",
                   0.0, float("nan"), "none", iso=iso)

    @property
    def A_Hz(self):
        return self.A_MHz * 1e6

    @property
    def B_Hz(self):
        return self.B_MHz * 1e6

    @property
    def A_unc_Hz(self):
        return self.A_unc_MHz * 1e6

    @property
    def B_unc_Hz(self):
        return self.B_unc_MHz * 1e6

    @property
    def has_A(self):
        return self.A_source != "none"

    def __str__(self):
        j2 = int(round(2 * self.j))
        label = f"{self.n}{_L_LETTERS[self.l]}{j2}/2"
        return (f"{label}: A = {self.A_MHz:.9g} MHz [{self.A_source}], "
                f"B = {self.B_MHz:.6g} MHz [{self.B_source}]")


class PortalAlkali:
    """Mixin in front of an ARC alkali class; see the module docstring.

    Subclasses set ``species`` (a key of :data:`ISOTOPES`); everything else
    (``gI``, ``element``, ``isotope``, ``portal_species``, ``g_J_ground``,
    ``core_polarizability_au``) is filled in from the table at class creation,
    as class attributes, so an instance made with ``__new__`` alone still has
    them.
    """

    species: str = ""
    element: str = ""
    isotope: int = 0
    portal_species: str = ""
    g_J_ground: float = float("nan")
    g_J_ground_ref: str = ""
    gI_ref: str = ""
    core_polarizability_au: float = float("nan")
    use_portal: bool = True
    _portal_data = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.species:
            d = ISOTOPES[cls.species]
            cls.element = d.element
            cls.isotope = d.isotope
            cls.portal_species = d.portal_species
            # ARC leaves gI at 0 for Li, Na and K, which drops the nuclear
            # Zeeman term from its getLandegfExact and breitRabi; for Rb and
            # Cs it has the same (Steck) values. Same sign convention
            # (H = mu_B B (g_J J_z + g_I I_z)).
            cls.gI = d.g_I
            cls.gI_ref = d.g_I_ref
            cls.g_J_ground = d.g_J_ground
            cls.g_J_ground_ref = d.g_J_ref
            cls.core_polarizability_au = d.core_polarizability_au

    def __init__(self, use_portal=True, portal_species=None, preferQuantumDefects=False):
        self.use_portal = use_portal
        if portal_species is not None:
            self.portal_species = portal_species
        self._portal_data = None
        super().__init__(preferQuantumDefects=preferQuantumDefects)
        self.cross_section = self.get_cross_section()

    def __repr__(self):
        return (f"{type(self).__name__}(use_portal={self.use_portal}, "
                f"portal_species={self.portal_species!r})")

    # ------------------------------------------------------------ atom facts

    @property
    def ground_state(self):
        """``(n, l, j)`` of the ground state."""
        return (int(self.groundStateN), 0, 0.5)

    def lowest_valence_n(self, l):
        """Lowest valence principal quantum number of orbital ``l`` (K: 4s,
        4p, 3d, 4f, ...). Lower n are core orbitals. From ARC's
        ``groundStateN`` and ``extraLevels``."""
        return lowest_valence_n(self, l)

    def is_valence(self, n, l):
        return int(n) >= self.lowest_valence_n(l)

    def g_J(self, l, j, n=None, s=0.5):
        """Electronic g-factor g_J (positive; H_Z = mu_B B (g_J m_j + g_I m_i)).

        The ground state (``l = 0`` with ``n`` the ground n or None) returns
        the measured value from the isotope table. Every other state uses the
        Landé formula with ARC's g_S and g_L = 1 - m_e/M. Relativistic and QED
        corrections for excited states are of order 1e-5 to 1e-4 and are left
        out.
        """
        if l == 0 and n in (None, int(self.groundStateN)) and np.isfinite(self.g_J_ground):
            return float(self.g_J_ground)
        return lande_g_j(l, j, s, self.gL, self.gS)

    @property
    def half_integer_I(self) -> bool:
        """True when I is half-integer (F and m_F integer, m_I half-integer)."""
        return abs(2 * self.I - round(2 * self.I)) < 1e-9 and round(2 * self.I) % 2 == 1

    def coupled_qn(self, F, mF):
        """``(F, mF)`` in the type kamo's state-tuple convention expects for
        this atom: ints when I is half-integer, half-integer floats when I is
        an integer (see :func:`kamo.hamiltonian.state_labels.is_coupled`)."""
        if self.half_integer_I:
            return int(round(float(F))), int(round(float(mF)))
        return round(2 * float(F)) / 2, round(2 * float(mF)) / 2

    def ground_qn(self, F, mF):
        """The 5-tuple ``(n, l, j, F, mF)`` of a ground-state sublevel."""
        return self.ground_state + self.coupled_qn(F, mF)

    @property
    def hamiltonian_n_threshold(self):
        """States with n below this use kamo.hamiltonian exact diagonalization
        (with nuclear spin); at or above, pairinteraction (m_j only)."""
        return int(self.groundStateN) + _HAMILTONIAN_N_ABOVE_GROUND

    @property
    def cycling_transition(self):
        """``((n, l, j, F), (n', l', j', F'))`` of the stretched D2 cycling
        transition ``nS_1/2 F=I+1/2 -> nP_3/2 F'=I+3/2``."""
        n0 = int(self.groundStateN)
        return ((n0, 0, 0.5, self.I + 0.5), (n0, 1, 1.5, self.I + 1.5))

    # ------------------------------------------------------------ hyperfine

    def hyperfine_constants(self, n, l, j):
        """Hyperfine ``A`` and ``B`` of ``(n, l, j)`` (an :class:`HFS`).

        ``A`` is the portal's value for this isotope when the portal lists the
        state (measured where it has one, else theory); otherwise ARC's table.
        ``B`` always comes from ARC (the portal has no quadrupole constants);
        it is exactly zero for J = 1/2. A state neither source knows has
        ``has_A`` False. ``use_portal=False`` gives ARC's table for both.
        """
        return self._hyperfine_cached(int(n), int(l), round(2 * float(j)))

    def _hyperfine_cached(self, n, l, twoj):
        cache = self.__dict__.setdefault("_hfs_cache", {})
        key = (n, l, twoj, bool(self.use_portal))
        if key not in cache:
            cache[key] = self._resolve_hyperfine(n, l, twoj / 2)
        return cache[key]

    def _arc_hyperfine(self, n, l, j):
        """ARC's ``(A_MHz, B_MHz)`` for the state, or ``None``."""
        try:
            A_Hz, B_Hz = super().getHFSCoefficients(n, l, j)
        except (ValueError, KeyError):
            return None
        return A_Hz / 1e6, B_Hz / 1e6

    def _portal_hyperfine_A(self, n, l, j):
        """The portal's A (MHz, uncertainty, source, ref) for this isotope, or ``None``."""
        rows = self._portal_hfs_table()
        if rows is None:
            return None
        sel = rows[(rows.iso == self.isotope) & (rows.n == n) & (rows.l == l)
                   & (np.abs(rows.J - j) < 1e-9)]
        if sel.empty:
            return None
        r = sel.iloc[0]
        if r.get("A_exp_MHz") is not None and np.isfinite(r.get("A_exp_MHz", np.nan)):
            return (float(r.A_exp_MHz), float(r.get("A_exp_unc_MHz", np.nan)),
                    "portal (measured)", str(r.get("A_exp_ref", "")))
        if r.get("A_theory_MHz") is not None and np.isfinite(r.get("A_theory_MHz", np.nan)):
            return (float(r.A_theory_MHz), float("nan"),
                    "portal (theory)", str(r.get("A_theory_ref", "")))
        return None

    def _portal_hfs_table(self):
        if "_hfs_rows" not in self.__dict__:
            from kamo.light_shift import udel_portal
            try:
                rows = udel_portal.hyperfine_constants(self.portal_species, bundled=True)
            except KeyError:
                try:
                    rows = udel_portal.hyperfine_constants(self.portal_species)
                except (KeyError, ConnectionError, OSError):
                    rows = None
            self.__dict__["_hfs_rows"] = rows
        return self.__dict__["_hfs_rows"]

    def _resolve_hyperfine(self, n, l, j):
        arc_ab = self._arc_hyperfine(n, l, j)
        B_MHz, B_src = (0.0, "exact (J = 1/2)") if j == 0.5 else (
            (arc_ab[1], "arc") if arc_ab is not None else (0.0, "none"))
        if self.use_portal:
            p = self._portal_hyperfine_A(n, l, j)
            if p is not None:
                A, A_unc, A_src, A_ref = p
                return HFS(n, l, j, A, A_unc, A_src, B_MHz, float("nan"), B_src,
                           A_ref=A_ref, iso=self.isotope)
        if arc_ab is None:
            return HFS.none(n, l, j, iso=self.isotope)
        return HFS(n, l, j, arc_ab[0], float("nan"), "arc", B_MHz, float("nan"), B_src,
                   iso=self.isotope)

    # ------------------------------------------------------------ scattering (K39 only)

    def get_scattering_length(self, *args, **kwargs):
        """s-wave scattering lengths are tabulated for 39K only
        (:mod:`kamo.scattering`); see ``Potassium39.get_scattering_length``."""
        raise NotImplementedError(
            f"kamo.scattering covers 39K only; {type(self).__name__} has no "
            "scattering-length data.")

    # ------------------------------------------------------------ portal data

    @staticmethod
    def _key(n, l, j):
        return (round(n), round(l), round(2 * j))

    def _portal(self):
        """Portal radial elements, rates and lifetimes, loaded on first use."""
        if self._portal_data is None:
            from kamo.light_shift import udel_portal
            me = udel_portal.dedupe_pairs(udel_portal.matrix_elements(self.portal_species))
            me = me[me.n1.notna() & me.n2.notna()]
            radial = {
                frozenset((self._key(r.n1, r.l1, r.J1), self._key(r.n2, r.l2, r.J2))):
                    r.d_au / _angular_factor(round(r.l1), r.J1, round(r.l2), r.J2)
                for r in me.itertuples()}
            tr = udel_portal.transition_rates(self.portal_species)
            tr = tr[tr.n1.notna() & tr.n2.notna()]
            rates = {(self._key(r.n1, r.l1, r.J1), self._key(r.n2, r.l2, r.J2)): r.A_s
                     for r in tr.itertuples()}
            lifetimes = {self._key(r.n1, r.l1, r.J1): r.tau_s for r in tr.itertuples()}
            self._portal_data = {"radial": radial, "rates": rates, "lifetimes": lifetimes,
                                 "signed": {}, "sign_uncertain": set()}
        return self._portal_data

    def getRadialMatrixElement(self, n1, l1, j1, n2, l2, j2, s=0.5, useLiterature=True):
        """ARC's radial element, rescaled to the portal's magnitude.

        ``useLiterature=False`` returns ARC's own numerical integral.
        """
        if not (self.use_portal and useLiterature):
            return super().getRadialMatrixElement(n1, l1, j1, n2, l2, j2, s=s,
                                                  useLiterature=useLiterature)
        portal = self._portal()
        pair = frozenset((self._key(n1, l1, j1), self._key(n2, l2, j2)))
        if pair not in portal["signed"]:
            arc_value = super().getRadialMatrixElement(n1, l1, j1, n2, l2, j2, s=s)
            magnitude = portal["radial"].get(pair)
            if magnitude is None or arc_value == 0:
                value = arc_value
            else:
                value = float(np.copysign(magnitude, arc_value))
                ratio = abs(arc_value) / magnitude
                if not 1 / _PORTAL_SIGN_TOLERANCE < ratio < _PORTAL_SIGN_TOLERANCE:
                    portal["sign_uncertain"].add(pair)
            portal["signed"][pair] = value
        return portal["signed"][pair]

    def portal_sign_uncertain(self, n1, l1, j1, n2, l2, j2):
        """True if ARC's sign for this transition should not be trusted.

        That is the case when ARC's magnitude is off from the portal's by more
        than ``_PORTAL_SIGN_TOLERANCE``. For K these are all small elements
        (< 0.25 a.u.): 4S-nP with n >= 8, and 4P-4D. Signs matter only where
        different matrix elements interfere (Raman couplings, D1/D2 in one
        coupling); polarizabilities use |d|^2 alone.
        """
        if not self.use_portal:
            return False
        self.getRadialMatrixElement(n1, l1, j1, n2, l2, j2)
        pair = frozenset((self._key(n1, l1, j1), self._key(n2, l2, j2)))
        return pair in self._portal()["sign_uncertain"]

    def getTransitionRate(self, n1, l1, j1, n2, l2, j2, temperature=0.0, s=0.5):
        """Rate (s^-1) from state 1 to state 2.

        At zero temperature this is the portal's Einstein A when it lists the
        decay channel. At finite temperature it is ARC's formula (portal matrix
        element, ARC transition frequency).
        """
        if self.use_portal and not temperature:
            rate = self._portal()["rates"].get((self._key(n1, l1, j1), self._key(n2, l2, j2)))
            if rate is not None:
                return rate
        return super().getTransitionRate(n1, l1, j1, n2, l2, j2, temperature=temperature, s=s)

    def getStateLifetime(self, n, l, j, temperature=0, includeLevelsUpTo=0, s=0.5):
        """Lifetime (s). At zero temperature this is the portal's recommended
        value when it has one; otherwise ARC's sum over decay rates."""
        if self.use_portal and temperature < 0.1:
            tau = self._portal()["lifetimes"].get(self._key(n, l, j))
            if tau is not None:
                return tau
        return super().getStateLifetime(n, l, j, temperature=temperature,
                                        includeLevelsUpTo=includeLevelsUpTo, s=s)

    def getHFSCoefficients(self, n, l, j, s=None):
        """Hyperfine ``(A, B)`` in Hz from :meth:`hyperfine_constants`.

        ARC's hyperfine helpers (``getHFSEnergyShift`` users) call this
        method, so they pick up kamo's values too. ``use_portal=False`` gives
        ARC's table back.
        """
        if not self.use_portal:
            return super().getHFSCoefficients(n, l, j, s=s)
        hc = self.hyperfine_constants(n, l, j)
        if not hc.has_A:
            raise ValueError(f"No hyperfine data for state ({n}, {l}, {j}).")
        return hc.A_Hz, hc.B_Hz

    def breitRabi(self, n, l, j, B):
        """Zeeman + hyperfine energies (Hz) of manifold ``(n, l, j)`` at fields
        ``B`` (tesla), in ARC's return format ``(energies, F, mF)``.

        ARC's version halves the quadrupole term: its B denominator is
        ``2I(2I-1) 2J(2J-1)`` instead of ``2I(2I-1) J(2J-1)``. That puts the
        4P_3/2 F'=0 level 1.8 MHz off. This one is built from kamo.hamiltonian,
        so it uses the same A, B, g_J and g_I as every other kamo Zeeman
        calculation (no diamagnetic term, as in ARC). As in ARC, each row of
        energies is sorted ascending (not tracked). F and mF label the columns
        by their order at a field whose Zeeman energy is 1e-4 of the smallest
        hyperfine gap. ARC always labels at 1e-4 T, where small-A manifolds
        (5P_3/2 and up) are already F-mixed and come out with half-integer F.
        A manifold with no hyperfine A (l >= 3) raises ``ValueError``, as
        ``getHFSCoefficients`` does. ``use_portal=False`` gives ARC's original.
        """
        if not self.use_portal:
            return super().breitRabi(n, l, j, B)
        from kamo.atom_properties.hyperfine import hyperfine_energy
        from kamo.hamiltonian.basis import Basis
        from kamo.hamiltonian.builder import HamiltonianBuilder
        hc = self.hyperfine_constants(n, l, j)
        if not hc.has_A:
            raise ValueError(f"No hyperfine data for state ({n}, {l}, {j}).")
        builder = HamiltonianBuilder(Basis([(n, l, j)], atom=self), atom=self)
        h0 = builder.h0()
        zeeman = builder.zeeman_operator() * 1e4          # Hz/G -> Hz/T
        B = np.atleast_1d(np.asarray(B, dtype=float))
        energies = np.array([np.linalg.eigvalsh(h0 + b * zeeman) for b in B])
        levels = np.sort([hyperfine_energy(F, builder.I, j, hc.A_Hz, hc.B_Hz)
                          for F in builder.basis.manifolds[0].allowed_F()])
        gap = np.diff(levels)
        gap = gap[gap > 0].min() if np.any(gap > 0) else abs(hc.A_Hz)
        B_label = 1e-4 * gap / np.abs(zeeman).max()       # tesla
        _, vecs = np.linalg.eigh(h0 + B_label * zeeman)
        IJ = builder._ij_operator(j, builder.I)
        F2 = (j * (j + 1) + builder.I * (builder.I + 1)) * np.eye(len(IJ)) + 2 * IJ
        f2 = np.einsum("ik,ij,jk->k", vecs, F2, vecs)
        F = np.round(-1 + np.sqrt(1 + 4 * f2)) / 2
        mF = np.round(2 * (np.array([s.m_f for s in builder.basis]) @ vecs ** 2)) / 2 + 0.0
        return energies, F, mF

    # def init_pairinteraction(self):
    #     if pi.Database.get_global_database() is None:
    #         pi.Database.initialize_global_database(download_missing=True)

    def get_magnetic_field_from_ground_state_transition_frequency(self,
                                                                f1, mf1, f2, mf2, transition_frequency_Hz,
                                                                B_bounds_G=(0., 600.),
                                                                N_interp=10000,
                                                                B_guess=None):
        """Returns the magnetic field(s) (in G) at which the transition from
        (f1,mf1) to (f2,mf2) would occur at frequency 'transition_frequency_Hz'.

        Args:
            f1 (int): State 1 quantum number F.
            mf1 (int): State 1 quantum number mF.
            f2 (int): State 2 quantum number F.
            mf2 (int): State 2 quantum number mF.
            transition_frequency_Hz (float or array-like): Measured transition frequency
                between (f1,mF1) and (f2,mF2).
            B_bounds_G (tuple, optional): Bounds used for field finding. Only
                limited to save time computing all the possible transition frequencies. Defaults to (0.,600.).
            N_interp (int, optional): Number of points used for interpolation. Defaults to 10000.
            B_guess (float, optional): Field (G) used to pick a branch when the
                splitting is non-monotonic and a target has several solutions.

        Raises:
            ValueError: If a target is never reached within the bounds, or if it
                has multiple branches and no ``B_guess`` was given.

        Returns:
            float or np.ndarray: the magnetic field(s) in G.
        """
        # Ground-state special case of the general splitting inverter;
        # both states share a manifold, so a single sweep serves all targets.
        return self.get_magnetic_field_from_splitting(
            self.ground_qn(f1, mf1),
            self.ground_qn(f2, mf2),
            transition_frequency_Hz,
            B_bounds_G=B_bounds_G,
            n_points=N_interp,
            B_guess=B_guess,
        )
    
    def get_ground_state_transition_sensitivity(self,f1,mf1,f2,mf2,B):
        """Returns the ground state transition sensitivity in MHz/G for (f1,mf1) to
        (f2,mf2) at field B.

        Args:
            f1 (int): State 1 quantum number F.
            mf1 (int): State 1 quantum number mF.
            f2 (int): State 2 quantum number F.
            mf2 (int): State 2 quantum number mF.
            B (float): Magnetic field in G.

        Returns:
            float: ground state transition sensitivity in MHz/G.
        """
        # Ground-state special case of get_transition_sensitivity.  That one
        # differentiates the signed E2 - E1; this function has always been the
        # slope of the positive splitting |E2 - E1|, so flip the sign where
        # the signed energy is negative.
        s1, s2 = self.ground_qn(f1, mf1), self.ground_qn(f2, mf2)
        sens = self.get_transition_sensitivity(s1, s2, B)
        sign = np.where(self.get_transition_energy(s1, s2, B) < 0, -1.0, 1.0)
        return sign * sens

    def _zeeman_hamiltonian_multi(self, states, B_gauss, B_sweep_steps=500, dB=None):
        """Run one kamo.hamiltonian sweep covering all requested states.

        Parameters
        ----------
        states : list of ``(n, l, j, m_j, m_i)`` tuples
        B_gauss : scalar or 1-D array
        B_sweep_steps : int
            Minimum number of sweep steps up to ``max(B_gauss)`` (the step is
            also capped at 0.01 G).  Ignored when ``dB`` is given.
        dB : float, optional
            Explicit sweep step in Gauss.

        Returns
        -------
        energies : list of ndarray (MHz), one per entry in *states*
        sweep : MagneticSweepResult
        """
        from kamo.hamiltonian import AtomicStructure
        B_arr = np.atleast_1d(np.asarray(B_gauss, dtype=float))
        B_max = max(float(np.max(B_arr)), 0.01)
        if dB is None:
            dB = min(B_max / B_sweep_steps, 0.01)
        else:
            dB = float(dB)
            if dB <= 0:
                raise ValueError(f"dB must be positive; got {dB}.")
            B_max = max(B_max, dB)          # at least one sweep step
        if dB > 0.1:
            print(f"Sweep steps are large ({dB:1.2e} G per step). Consider increasing sampling if adiabatic state detection suffers.")

        # Collect unique (n, l, j) fine-structure levels spanning all requests
        njl_levels = list(dict.fromkeys((s[0], s[1], s[2]) for s in states))
        model = AtomicStructure(njl_levels, atom=self)
        res = model.magnetic_sweep(B_max=B_max, dB=dB, diamagnetic=True, include_quadrupole=True)

        energies = [res.get_energy(n, l, j, m_j, m_i, at=B_arr) / 1e6
                    for n, l, j, m_j, m_i in states]
        return energies, res

    def _zeeman_hamiltonian(self, n, l, j, m_j, m_i, B_gauss):
        """Energy track (MHz) via kamo.hamiltonian exact diagonalization.

        Uses the adiabatic (Paschen-Back) connection: ``m_j`` and ``m_i`` are
        the high-field limiting quantum numbers, resolved to a tracked state by
        :meth:`~kamo.hamiltonian.SweepResult.get_energy`.

        ``B_gauss`` may be a scalar or 1-D array; returns ``(energies_MHz, sweep_result)``.
        """
        (energy,), res = self._zeeman_hamiltonian_multi([(n, l, j, m_j, m_i)], B_gauss)
        return energy, res

    def _zeeman_pairinteraction(self, n, l, j, m_j, B_gauss):
        """Zeeman *shift* from zero field (MHz) via pairinteraction.

        Only uses ``m_j`` (pairinteraction has no nuclear spin).
        ``B_gauss`` may be a scalar or 1-D array; returns matching ndarray.
        """
        import pairinteraction as pi
        B_arr = np.atleast_1d(np.asarray(B_gauss, dtype=float))
        shifts = np.zeros(len(B_arr))
        ket = pi.KetAtom(self.element, n=n, l=l, j=j, m=m_j)
        zero_field_energy = ket.get_energy()
        l_max = min(l + 2, n - 1)
        basis = pi.BasisAtom(self.element, n=(n - 3, n + 3), l=(l, l_max))
        for idx, b_val in enumerate(B_arr):
            system = pi.SystemAtom(basis)
            system.set_diamagnetism_enabled(True)
            system.set_magnetic_field([0.0, 0.0, float(b_val)], unit="gauss")
            pi.diagonalize([system])
            shifted = system.get_corresponding_energy(ket)
            shifts[idx] = (shifted - zero_field_energy).to("J").magnitude / c.h / 1e6
        return shifts

    def get_semiclassical_polarizability(self,n1,l1,j1,n2,l2,j2,detuning_Hz):
        """See Grimm 1999 equation 8.
        """        
        f0 = np.abs(self.getTransitionFrequency(n1,l1,j1,n2,l2,j2))
        omega0 = 2 * np.pi * f0
        omega = 2 * np.pi * (f0 + detuning_Hz)
        linewidth = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        return 6 * np.pi * c.epsilon0 * c.c**3 * \
            ( linewidth / omega0**2 ) / ( omega0**2 - omega**2 - 1j * (omega**3/omega0**2) * linewidth )
    
    def get_scattering_rate(self,
                            n1,l1,j1,
                            n2,l2,j2,
                            intensity,
                            detuning_Hz=100.e6):
        """Photon scattering rate in s^-1.  See Grimm 1999 equation 9::

            Gamma_sc = Im(alpha) * I / (hbar * eps0 * c)

        The result is a genuine rate (photons per second), not an angular
        frequency: `get_decay_rate` supplies Gamma = 1/tau as a population
        decay constant in s^-1, and eq. 9 carries that through unchanged.
        There is no 1/(2 pi) in Grimm's expression -- dividing by 2 pi here
        would under-report the rate by a factor of 2 pi (it would, for
        example, cap the resonant saturated rate at Gamma/(4 pi) instead of
        the correct Gamma/2).
        """
        alpha = self.get_semiclassical_polarizability(n1,l1,j1,n2,l2,j2,detuning_Hz)
        return 1/(c.hbar * c.epsilon0 * c.c) * np.imag(alpha) * intensity

    # def get_off_resonant_scattering_rate(self,
    #                         n1,l1,j1,
    #                         n2,l2,j2,
    #                         intensity,
    #                         detuning_Hz=100.e6):
    #     omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
    #     linewidth = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
    #     Delta = 2 * np.pi * detuning_Hz
    #     return 3 * np.pi * c.c**2 / (2 * c.hbar * omega0**3) * (linewidth/Delta)**2 * intensity

    def get_decay_rate(self,n1,l1,j1,n2,l2,j2):
        '''
        Returns spontaneous emission rate for the higher of two states in 1/s.
        '''
        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            Gamma = 1/self.getStateLifetime(n2,l2,j2)
        else:
            Gamma = 1/self.getStateLifetime(n1,l1,j1)
        return Gamma

    def get_saturation_intensity(self,
                                 n1,l1,j1,
                                 n2,l2,j2,
                                 detuning_Hz=0.,
                                 convert_to_mW_per_cm2=False):
        '''
        Returns the off-resonant (effective) saturation intensity in W/m^2 for
        the transition between the two given states.

        The on-resonance two-level value is

            I_sat = pi h c Gamma / (3 lambda^3),

        with Gamma the spontaneous decay rate of the upper state (from
        `get_decay_rate`) and lambda the transition wavelength.  A laser detuned
        by Delta saturates the transition more slowly, so the intensity at which
        the excited-state population reaches 1/4 grows as

            I_sat(Delta) = I_sat * ( 1 + (2 Delta / Gamma)^2 ).

        This is the quantity that makes the scattering rate

            R = (Gamma/2) * (I / I_sat(Delta)) / (1 + I / I_sat(Delta))

        equal to the usual detuned two-level result.

        Parameters
        ----------
        n1,l1,j1 : lower/upper state quantum numbers (order does not matter;
            the higher-lying state supplies the linewidth)
        n2,l2,j2 : the other state of the pair
        detuning_Hz: float
            The laser detuning from resonance, as an ordinary frequency in Hz
            (not angular).  Default 0., which gives the resonant I_sat.  Only
            the magnitude matters.
        convert_to_mW_per_cm2: bool
            If True, converts the output to mW/cm^2 before returning.

        Returns
        -------
        float

        Notes
        -----
        This is the two-level (cycling-transition) saturation intensity: it
        carries no Clebsch-Gordan factor for a particular m_F -> m_F' pair, and
        no polarization dependence.  The states are used only to fix Gamma and
        the transition wavelength.
        '''
        Gamma = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        f0 = np.abs(self.getTransitionFrequency(n1,l1,j1,n2,l2,j2))
        wavelength = c.c / f0

        saturation_intensity = np.pi * c.h * c.c * Gamma / (3 * wavelength**3)
        detuning_factor = 1 + (2 * 2 * np.pi * np.asarray(detuning_Hz) / Gamma)**2

        convert_W_per_m2_to_mW_per_cm2 = 0.1
        if convert_to_mW_per_cm2:
            convert = convert_W_per_m2_to_mW_per_cm2
        else:
            convert = 1

        return saturation_intensity * detuning_factor * convert
        
    def lineshape(self,n1=None,l1=0,j1=1/2,n2=None,l2=1,j2=3/2,detuning_Hz=0):
        '''
        Returns the lineshape evaluated at a given detuning for a two-level system. Does not work for excited states.
        ``n1`` and ``n2`` default to the ground n (the D2 line).
        '''
        n0 = int(self.groundStateN)
        n1 = n0 if n1 is None else n1
        n2 = n0 if n2 is None else n2
        if n2 > n0 or n1 > n0 or l1 > 1 or l2 > 1:
            print(f"Lineshape not accurate for excited states with n>{n0}.")
        gamma = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        # transition_omega = np.abs( self.getTransitionFrequency(n1,l1,j1,n2,l2,j2) ) / 2 / np.pi
        detuning_omega = 2 * np.pi * detuning_Hz
        return (1/(2*np.pi)) * gamma / ( detuning_omega**2 + gamma**2 / 4 )

    def get_cross_section(self,n1=None,l1=0,j1=1/2,F1=None,n2=None,l2=1,j2=3/2,F2=None,detuning_Hz=0):
        """Resonant scattering cross section (m^2) of the ``(n1 l1 j1 F1) ->
        (n2 l2 j2 F2)`` transition. Defaults to the stretched D2 cycling
        transition of this atom (:attr:`cycling_transition`; K39: 4S1/2 F=2 ->
        4P3/2 F=3)."""
        (g_n, _, _, g_F), (e_n, _, _, e_F) = self.cycling_transition
        n1 = g_n if n1 is None else n1
        n2 = e_n if n2 is None else n2
        F1 = g_F if F1 is None else F1
        F2 = e_F if F2 is None else F2

        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            A21 = 2*np.pi*self.getTransitionRate(n2,l2,j2,n1,l1,j1,temperature=0.0) 
        else:
            A21 = 2*np.pi*self.getTransitionRate(n1,l1,j1,n2,l2,j2,temperature=0.0)

        g2 = 2*F2 + 1
        g1 = 2*F1 + 1

        # if ordered:
        #     g_ratio = g2/g1
        # else:
        #     g_ratio = g1/g2
        g_ratio = 1

        omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
        lineshape = self.lineshape(n1,l1,j1,n2,l2,j2,detuning_Hz=detuning_Hz)
        scattering_cross_section = g_ratio * np.pi**2 * c.c**2 / omega0**2 * A21 * lineshape
        return scattering_cross_section
    
    def get_zeeman_shift(self, n, l, j, m_j, m_i=None,
                         B=0, return_sweep=False):
        """Return the Zeeman energy (MHz) for state |n l j; m_j [m_i]> at B (Gauss).

        Routing:
        - ``n < hamiltonian_n_threshold`` (ground n + 6; 10 for K): kamo.hamiltonian
          exact diagonalization.  ``m_i`` required.
          Quantum numbers use the **adiabatic (Paschen-Back) convention**: ``m_j``
          and ``m_i`` are the high-field limiting values.
        - ``n >= hamiltonian_n_threshold``: pairinteraction.  Only ``m_j`` is used; ``m_i`` is ignored.

        ``B`` may be a scalar or array; returns matching shape.

        Parameters
        ----------
        return_sweep : bool, optional
            If True, return a ``(energy, sweep)`` tuple where ``sweep`` is the
            :class:`~kamo.hamiltonian.MagneticSweepResult` (low n only; ``None``
            for the pairinteraction path).
        """
        B_arr = np.atleast_1d(np.asarray(B, dtype=float))
        scalar_in = np.ndim(B) == 0

        n_thr = self.hamiltonian_n_threshold
        if n < n_thr:
            if m_i is None:
                raise ValueError(
                    f"m_i must be provided for n < {n_thr} "
                    "(kamo.hamiltonian includes nuclear Zeeman)."
                )
            result, sweep = self._zeeman_hamiltonian(n, l, j, m_j, m_i, B_arr)
        else:
            result = self._zeeman_pairinteraction(n, l, j, m_j, B_arr)
            sweep = None

        energy = float(result[0]) if scalar_in else result
        return (energy, sweep) if return_sweep else energy

    @staticmethod
    def _check_state_tuples(*states):
        """Return ``states`` as tuples, or raise if one is not a 5-tuple."""
        out = []
        for st in states:
            st = tuple(st)
            if len(st) != 5:
                raise ValueError(
                    "Each state must be a 5-tuple (n, l, j, a, b) where (a, b) "
                    "are either (F, mF) ints or (m_j, m_i) half-integer floats; "
                    f"got {st!r}.")
            out.append(st)
        return out

    def _transition_energy_mhz(self, state1, state2, B=0, dB=None):
        """Signed ``E2 − E1`` (MHz) versus field, vectorized over ``B`` (Gauss).

        The engine behind :meth:`get_transition_energy` (see there for the
        state convention and the meaning of the energy).  ``B`` may be a
        scalar or 1-D array (the return matches its shape).  When both states
        are low-n (below :attr:`hamiltonian_n_threshold`) a single magnetic
        sweep covers both manifolds and the states are followed
        adiabatically; high-n states fall back to per-state
        :meth:`get_zeeman_shift`.  ``dB`` is the sweep step (Gauss); ``None``
        picks the :meth:`_zeeman_hamiltonian_multi` default.
        """
        state1, state2 = self._check_state_tuples(state1, state2)
        n1, l1, j1, m_j1, m_i1 = state1
        n2, l2, j2, m_j2, m_i2 = state2
        B_arr = np.atleast_1d(np.asarray(B, dtype=float))
        scalar_in = np.ndim(B) == 0

        n_thr = self.hamiltonian_n_threshold
        if n1 < n_thr and n2 < n_thr:
            # One magnetic sweep up to the largest requested field covers both
            # manifolds; each requested field is read back by interpolation
            # (no per-B re-diagonalization).
            [e1_arr, e2_arr], _ = self._zeeman_hamiltonian_multi(
                [state1, state2], B_arr, dB=dB
            )
            result = e2_arr - e1_arr
        else:
            e1 = self.get_zeeman_shift(n1, l1, j1, m_j1, m_i1, B_arr)
            e2 = self.get_zeeman_shift(n2, l2, j2, m_j2, m_i2, B_arr)
            result = np.atleast_1d(e2 - e1)

        return float(result[0]) if scalar_in else result

    def _splitting_mhz(self, state1, state2, B=0):
        """Return |E2 − E1| (MHz) versus field, vectorized over ``B`` (Gauss).

        The unsigned form of :meth:`_transition_energy_mhz`, used by
        :meth:`get_microwave_transition_frequency` and
        :meth:`get_magnetic_field_from_splitting`, which need array-valued ``B``
        and MHz units that the scalar, Hz-valued
        :meth:`get_transition_frequency` state-tuple API does not provide.
        """
        return np.abs(self._transition_energy_mhz(state1, state2, B))

    def get_transition_energy(self, state1, state2, B=0, B_ref=None, dB=None):
        """Transition energy ``E2 − E1`` (MHz) between two states at field ``B``
        (Gauss), optionally relative to its value at a reference field.

        This is the general form of
        :meth:`get_ground_state_transition_frequency` (any pair of states,
        signed) and of :meth:`get_transition_shift` (``B_ref=0``).  ``B`` may
        be a scalar or a 1-D array; the result has the same shape, and every
        field is read from a single magnetic sweep.

        Each state is a 5-tuple ``(n, l, j, a, b)`` in the standard ``kamo``
        convention: ``(a, b)`` are coupled ``(F, mF)`` low-field labels when
        both are ints, or uncoupled adiabatic ``(m_j, m_i)`` Paschen-Back
        labels when both are half-integer floats.  Either way the state is
        followed adiabatically through the field sweep.

        The energy of each state is its full field-free energy (fine
        structure relative to a common reference, plus hyperfine) plus its
        Zeeman energy, so:

        * within one ``(n, l, j)`` manifold the result is the hyperfine +
          Zeeman splitting (the RF / microwave transition frequency);
        * across manifolds it is the optical transition frequency in MHz
          (~3.9e8 MHz for a potassium D line), hyperfine and Zeeman resolved.
          Pass ``B_ref`` to keep only the field-dependent part.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) tuples
            Lower/upper states; the result is ``E2 − E1``, signed, so it is
            negative when ``state2`` lies below ``state1``.
        B : float or 1-D array
            Magnetic field(s) in Gauss (default 0).
        B_ref : float, optional
            Reference field in Gauss.  When given, return
            ``f(B) − f(B_ref)`` instead of ``f(B)``; ``B_ref=0`` gives the
            differential Zeeman shift of the transition.
        dB : float, optional
            Magnetic-sweep step in Gauss.  ``None`` (default) uses at least
            500 steps up to ``max(B)`` and never coarser than 0.01 G.

        Returns
        -------
        float or np.ndarray
            ``E2 − E1`` in MHz, matching the shape of ``B``.
        """
        if B_ref is None:
            return self._transition_energy_mhz(state1, state2, B, dB=dB)
        B_arr = np.atleast_1d(np.asarray(B, dtype=float))
        scalar_in = np.ndim(B) == 0
        # One sweep serves both the requested fields and the reference.
        e = self._transition_energy_mhz(
            state1, state2, np.append(B_arr, float(B_ref)), dB=dB)
        result = e[:-1] - e[-1]
        return float(result[0]) if scalar_in else result

    def get_transition_sensitivity(self, state1, state2, B, dB=None,
                                   sweep_dB=None):
        """Magnetic sensitivity ``d(E2 − E1)/dB`` (MHz/G) of a transition at
        field ``B`` (Gauss).

        The general form of :meth:`get_ground_state_transition_sensitivity`:
        any pair of states in the 5-tuple convention of
        :meth:`get_transition_energy`, and ``B`` may be a scalar or a 1-D
        array (the result matches its shape).  The derivative is the forward
        difference ``(f(B + dB) − f(B)) / dB`` of :meth:`get_transition_energy`,
        signed like it: reverse the two states to flip the sign.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) tuples
            The two states (see :meth:`get_transition_energy`).
        B : float or 1-D array
            Magnetic field(s) in Gauss.
        dB : float or array, optional
            Finite-difference step in Gauss.  ``None`` (default) uses
            ``0.001 * |B|``, but at least 0.001 G (so ``B = 0`` works).
        sweep_dB : float, optional
            Magnetic-sweep step in Gauss (``dB`` of
            :meth:`get_transition_energy`).

        Returns
        -------
        float or np.ndarray
            Sensitivity in MHz/G, matching the shape of ``B``.
        """
        B_arr = np.atleast_1d(np.asarray(B, dtype=float))
        scalar_in = np.ndim(B) == 0
        if dB is None:
            dB_arr = np.maximum(1e-3 * np.abs(B_arr), 1e-3)
        else:
            dB_arr = np.broadcast_to(np.asarray(dB, dtype=float), B_arr.shape)
            if np.any(dB_arr <= 0):
                raise ValueError("dB must be positive.")
        # Both fields of every pair come from a single sweep.
        e = self._transition_energy_mhz(
            state1, state2, np.concatenate([B_arr, B_arr + dB_arr]), dB=sweep_dB)
        n = B_arr.size
        result = (e[n:] - e[:n]) / dB_arr
        return float(result[0]) if scalar_in else result

    def get_microwave_transition_frequency(self, n, l, j, m_j1, m_i1, m_j2, m_i2, B=0):
        """|E2 − E1| (MHz) for the ``(m_j1,m_i1) → (m_j2,m_i2)`` transition in
        manifold ``(n, l, j)`` at field ``B`` (Gauss).

        ``B`` is normally a scalar.  Passing a 1-D array is **deprecated**: it
        still works — a single magnetic sweep is run up to ``max(B)`` and each
        requested field is interpolated from that sweep — but prefer a scalar
        ``B``, or drive a :meth:`~kamo.hamiltonian.model.AtomicStructure.magnetic_sweep`
        result directly (e.g. ``SweepResult.field_energy`` / ``get_energy``) for
        full control over the field grid.
        """
        if np.ndim(B) > 0:
            import warnings
            warnings.warn(
                "Supplying a vector B to get_microwave_transition_frequency is "
                "deprecated: one magnetic sweep is run up to max(B) and each "
                "requested field is interpolated from that sweep. Pass a "
                "scalar B, or use a magnetic_sweep result "
                "(SweepResult.field_energy / get_energy) for full control.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._splitting_mhz(
            (n, l, j, m_j1, m_i1), (n, l, j, m_j2, m_i2), B)

    def get_magnetic_field_from_splitting(
        self,
        state1,
        state2,
        transition_frequency_Hz,
        B_bounds_G=(0.0, 600.0),
        n_points=10000,
        B_guess=None,
    ):
        """Return the field(s) (Gauss) at which |E2 − E1| equals ``transition_frequency_Hz``.

        A **single** magnetic sweep of the two states' manifold(s) is run over
        ``B_bounds_G`` (``n_points`` samples) and inverted, so the cost is
        independent of how many target frequencies are requested.

        Monotonic vs non-monotonic
        --------------------------
        When the splitting increases (or decreases) monotonically over
        ``B_bounds_G`` the inversion is a single interpolation.  If the splitting
        curve turns over, a given target can be reached at **several** fields
        (branches).  In that case:

        * pass ``B_guess`` (Gauss) and the branch nearest to it is returned;
        * omit ``B_guess`` and a :class:`ValueError` lists the branch fields so
          you can pick one via ``B_guess``.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) 5-tuples
            The two states, with ``(a, b)`` either ``(F, mF)`` ints or
            ``(m_j, m_i)`` half-integer floats (standard ``kamo`` convention).
        transition_frequency_Hz : float or array-like
            Target splitting |E2 − E1| in Hz.  Array-like returns an array of
            fields (one per target).
        B_bounds_G : (float, float)
            Field-search bounds in Gauss (default ``(0, 600)``).  Only the range
            searched; widen it if the target lies outside.
        n_points : int
            Sweep / interpolation samples across ``B_bounds_G`` (default 10000).
        B_guess : float, optional
            Field (Gauss) used to disambiguate multiple branches: the crossing
            nearest ``B_guess`` is returned.  Required only when the splitting is
            non-monotonic and a target has more than one solution.

        Returns
        -------
        float or np.ndarray
            The field(s) in Gauss.

        Raises
        ------
        ValueError
            If a target is never reached within ``B_bounds_G`` (widen the
            bounds), or if it has multiple branches and no ``B_guess`` was given
            (the error lists the branch fields).
        """
        target_hz = np.atleast_1d(np.asarray(transition_frequency_Hz, dtype=float))
        b = np.linspace(B_bounds_G[0], B_bounds_G[1], n_points)
        freq_MHz = self._splitting_mhz(state1, state2, b)   # one sweep

        diffs = np.diff(freq_MHz)
        monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)

        if monotonic and B_guess is None:
            # single branch → one fast, vectorized interpolation over all targets
            b_grid, f_grid = b, freq_MHz
            if f_grid[0] > f_grid[-1]:                 # np.interp needs ascending x
                f_grid, b_grid = f_grid[::-1], b_grid[::-1]
            B_G = np.interp(target_hz / 1e6, f_grid, b_grid)
            if np.any((B_G == B_bounds_G[0]) | (B_G == B_bounds_G[1])):
                raise ValueError(
                    "One or more target frequencies fall on a bound of "
                    f"B_bounds_G {B_bounds_G} G.  Widen the bounds and re-run."
                )
        else:
            # non-monotonic (or an explicit B_guess): resolve each target from
            # its actual crossing(s) of the splitting curve.
            B_G = np.array([
                self._select_branch(b, freq_MHz, t / 1e6, B_guess, B_bounds_G)
                for t in target_hz
            ])

        return float(B_G[0]) if B_G.size == 1 else B_G

    _crossings = staticmethod(_crossings)

    @staticmethod
    def _select_branch(b, freq_MHz, target_MHz, B_guess, B_bounds_G):
        """Return the field where ``freq_MHz(b) == target_MHz``.

        Finds every crossing (linear root of ``freq_MHz − target_MHz``) in the
        swept range.  With one crossing it is returned directly; with several,
        the branch nearest ``B_guess`` is chosen, or a :class:`ValueError`
        listing the branch fields is raised when ``B_guess`` is ``None``.
        """
        roots = _crossings(b, freq_MHz, target_MHz)

        if len(roots) == 0:
            raise ValueError(
                f"Splitting never equals {target_MHz:.6f} MHz within "
                f"B_bounds_G {B_bounds_G} G.  Widen the bounds and re-run."
            )
        if len(roots) == 1:
            return float(roots[0])
        if B_guess is None:
            branch_str = ", ".join(f"{r:.3f}" for r in roots)
            raise ValueError(
                f"Splitting equals {target_MHz:.6f} MHz at multiple fields "
                f"(non-monotonic curve): branches at [{branch_str}] G.  Pass "
                "B_guess (Gauss) to select the branch nearest a known field."
            )
        return float(roots[np.argmin(np.abs(roots - B_guess))])

    def get_ground_state_transition_frequency(self,f1,m_f1,f2,m_f2,B=0):
        '''
        Returns the ground-state transition frequency |E2 − E1| (MHz) between
        (f1,m_f1) and (f2,m_f2) under external magnetic field B (in Gauss).
        B may be a scalar or 1-D array.  Both states are read from a single
        magnetic sweep.  Ground-state, unsigned special case of
        :meth:`get_transition_energy`.
        '''
        return np.abs(self.get_transition_energy(
            self.ground_qn(f1, m_f1),
            self.ground_qn(f2, m_f2), B))

    def get_transition_shift(self, *args, B=0, **kwargs):
        """Differential Zeeman shift of a transition at field B (MHz).

        Returns ``f(B) − f(0)`` for the transition ``state1 -> state2``, i.e.
        ``ΔE(state2, B) − ΔE(state1, B)`` with ``ΔE(state, B) = E(state, B) −
        E(state, 0)``.  The ``B_ref=0`` special case of
        :meth:`get_transition_energy`; ``B`` may be a scalar or 1-D array.

        Accepts either two state 5-tuples::

            atom.get_transition_shift(state1, state2, B=520)

        or the original ten flat quantum numbers::

            atom.get_transition_shift(n1, l1, j1, m_j1, m_i1,
                                      n2, l2, j2, m_j2, m_i2, B=520)

        In both forms ``(m_j, m_i)`` follow the standard ``kamo`` convention
        (ints for ``(F, mF)``, half-integer floats for ``(m_j, m_i)``).
        """
        names = ("n1", "l1", "j1", "m_j1", "m_i1",
                 "n2", "l2", "j2", "m_j2", "m_i2")
        if len(args) == 2 and not kwargs:
            state1, state2 = args
        elif len(args) == 3 and not kwargs:
            state1, state2, B = args
        else:
            if len(args) == 11:                 # B given positionally
                *args, B = args
            if any(isinstance(x, (tuple, list, np.ndarray)) for x in args):
                raise TypeError(
                    "get_transition_shift with state tuples takes only "
                    f"(state1, state2, B=0); got extra {sorted(kwargs)!r}.")
            # flat form, possibly with some quantum numbers given by keyword
            flat = list(args)
            for name in names[len(flat):]:
                if name not in kwargs:
                    raise TypeError(
                        "get_transition_shift takes either (state1, state2) "
                        "or the ten quantum numbers "
                        f"{', '.join(names)}; missing {name!r}.")
                flat.append(kwargs.pop(name))
            if kwargs or len(flat) != 10:
                bad = list(kwargs) or flat[10:]
                raise TypeError(
                    f"get_transition_shift got unexpected arguments {bad!r}.")
            state1, state2 = tuple(flat[:5]), tuple(flat[5:])
        # dB=0.1 G is the sweep step this method has always used.
        return self.get_transition_energy(state1, state2, B, B_ref=0.0, dB=0.1)
    
    def get_transition_frequency(
        self,
        state1,
        state2,
        B=0.0,
        beam=None,
        frequency_Hz=None,
        intensity=None,
        polarization="pi",
        laser_model="auto",
        basis=None,
        n_points=200,
        dB=0.1,
        diamagnetic=True,
        relative_mode=None,
        return_sweep=False,
    ):
        """Transition frequency (Hz) between two states at field ``B``, with an
        optional laser light shift.

        Each state is a 5-tuple ``(n, l, j, a, b)`` whose last two numbers use
        the standard ``kamo`` convention:

        * both **int**   -> coupled-basis ``(F, mF)`` low-field labels;
        * both **float** (half-integer) -> uncoupled adiabatic ``(m_j, m_i)``
          (Paschen-Back) labels.

        The calculation uses ``kamo.hamiltonian`` exact diagonalization with
        eigenshuffle state tracking, so states are followed adiabatically (the
        same magnetic-sweep connection used everywhere else) through avoided
        crossings.

        How the two fields combine
        --------------------------
        * **Magnetic only** (no laser): a magnetic sweep 0 -> ``B`` gives the
          bare transition frequency at ``B``.
        * **Magnetic + laser**: the bare transition at ``B`` is computed from
          the magnetic sweep first; then a laser-intensity sweep *at the same
          field* ``B`` provides the light shift of the transition.  The light
          shift is taken as an intensity *difference* (``f(I) - f(0)``), so it
          is a true lab-frame shift regardless of the rotating frame used
          internally by the RWA model.

        Reference (``relative_mode``)
        -----------------------------
        Let ``f(B, I)`` be the absolute transition frequency ``E2 - E1``.

        * ``"absolute"`` -- return ``f(B, I)`` (the full transition frequency).
        * ``"magnetic"`` -- return ``f(B, I) - f(0, 0)``, i.e. relative to the
          zero-magnetic-field, zero-intensity transition frequency (the Zeeman
          shift of the transition when no laser is present).
        * ``"optical"`` -- return ``f(B, I) - f(B, 0)``, i.e. relative to zero
          intensity at the given (high) magnetic field (the pure light shift).
        * ``None`` (default) -- ``"optical"`` when a laser is supplied,
          otherwise ``"magnetic"``.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) tuples
            Lower/upper states of the transition (result is ``E2 - E1``, signed).
        B : float, optional
            Static magnetic field in Gauss (default 0).
        beam : kamo.GaussianBeam, optional
            Laser beam producing the light shift.  Supplies the laser frequency
            and, unless ``intensity`` is given, the peak intensity ``beam.I0``.
            Provide *either* ``beam`` *or* (``frequency_Hz`` + ``intensity``),
            not both.
        frequency_Hz : float, optional
            Laser frequency (Hz).  Requires ``intensity``.
        intensity : float, optional
            Laser intensity in W/m^2.  Required with ``frequency_Hz``; overrides
            ``beam.I0`` when supplied alongside ``beam``.
        polarization : str, optional
            Laser polarization: "pi", "sigma+", or "sigma-" (default "pi").
        laser_model : {"auto", "perturbative", "rwa", "stark"}, optional
            Light-shift model (default "auto").

            * "perturbative": second-order sum over the exact eigenstates of
              ``h0 + B * Zeeman`` with both rotating terms
              (:mod:`kamo.hamiltonian.perturbative`).  Hyperfine and Zeeman
              resolved, so it gives the qubit's differential shift, and it
              matches the fine-structure polarizabilities far from resonance.
              Linear in intensity.
            * "rwa": rotating-wave dipole coupling, non-perturbative, but
              without the counter-rotating terms (error ~ detuning / 2 f_0).
            * "stark": fine-structure polarizabilities in a diagonal operator;
              fast, no channel manifolds needed, but blind to the substructure
              of the coupled manifolds (error ~ their hyperfine + Zeeman
              spread over the detuning; zero for any within-manifold
              differential).  Never chosen by "auto".
            * "auto" calls :func:`kamo.hamiltonian.choose_laser_model`: "rwa"
              when Rabi/(2 detuning) exceeds 0.1 or when its square (the next
              order of the perturbative sum) exceeds the RWA's counter-rotating
              error estimate; "perturbative" otherwise.  The decision is
              attached to the returned sweep as ``sweep.model_choice``
              (``return_sweep=True``).
        basis : AtomicStructure, optional
            Override the atomic-structure basis.  When omitted, a basis is built
            automatically: just the two states' own manifolds (and their
            fine-structure partners) for a pure magnetic calculation or the
            Stark model, whose operator is diagonal; for the RWA and
            perturbative models, :func:`kamo.hamiltonian.light_shift_basis`
            adds every manifold that carries at least 1e-3 of either state's
            polarizability at the laser wavelength (so 3D and 5S appear for a
            4P state at 1064 nm).
        n_points : int, optional
            Intensity steps for the laser sweep (default 200).
        dB : float, optional
            Magnetic-sweep step in Gauss (default 0.1).
        diamagnetic : bool, optional
            Include the diamagnetic term in the magnetic sweep (default True).
        relative_mode : {None, "absolute", "magnetic", "optical"}, optional
            Reference for the returned frequency (see above).  Default None.
        return_sweep : bool, optional
            If True, also return the underlying sweep result (the laser sweep
            when a laser is supplied, otherwise the magnetic sweep).

        Returns
        -------
        float, or (float, SweepResult) when ``return_sweep`` is True.
        """
        from kamo import GaussianBeam
        from kamo.hamiltonian import (AtomicStructure, choose_laser_model,
                                      light_shift_basis)

        s1 = tuple(state1)
        s2 = tuple(state2)
        if len(s1) != 5 or len(s2) != 5:
            raise ValueError(
                "Each state must be a 5-tuple (n, l, j, a, b) where (a, b) are "
                "either (F, mF) ints or (m_j, m_i) half-integer floats.")

        # ---- validate the laser specification ----
        has_beam = beam is not None
        has_freq = frequency_Hz is not None
        if has_beam and has_freq:
            raise ValueError("Provide either `beam` or `frequency_Hz`, not both.")
        if has_freq and intensity is None:
            raise ValueError("`frequency_Hz` requires `intensity` (W/m^2).")
        use_light = has_beam or has_freq

        # ---- resolve the reference mode ----
        if relative_mode is None:
            relative_mode = "optical" if use_light else "magnetic"
        if relative_mode not in ("absolute", "magnetic", "optical"):
            raise ValueError(
                "relative_mode must be None, 'absolute', 'magnetic', or "
                f"'optical'; got {relative_mode!r}.")

        # ---- choose the laser model ----
        model_choice = None
        if use_light:
            f_laser = beam.frequency() if has_beam else float(frequency_Hz)
            I_choice = (float(intensity) if intensity is not None
                        else (float(beam.I0) if has_beam else None))
            if laser_model == "auto":
                model_choice = choose_laser_model(
                    (s1, s2), f_laser, I_choice, B_gauss=float(B), atom=self)
                laser_model = model_choice.model
            elif laser_model not in ("rwa", "stark", "perturbative"):
                raise ValueError(
                    "laser_model must be 'auto', 'rwa', 'perturbative' or 'stark'; "
                    f"got {laser_model!r}.")

        # ---- build / accept the basis ----
        # The magnetic sweep only needs the states' own manifolds and their
        # fine-structure partners (Zeeman and hyperfine do not couple
        # different (n, l); the diamagnetic l +/- 2 coupling is ~1e-12 at
        # 520 G), so it runs in that small basis unless the caller supplied
        # one.  The RWA and perturbative laser models also need the channel
        # manifolds, which can multiply the dimension by five; the laser sweep
        # identifies its states from the field eigenstates on its own, so it
        # gets the larger basis.
        model_B = model = AtomicStructure(_own_manifolds(s1, s2), atom=self)
        if basis is not None:
            model_B = model = basis
        elif use_light and laser_model in ("rwa", "perturbative"):
            # these models only see channels present in the basis: take every
            # manifold carrying >= 1e-3 of either state's polarizability.  The
            # photon-index loop check only matters for the RWA.
            sel = model_choice.basis if model_choice is not None else None
            if sel is None or (laser_model == "perturbative" and sel.dropped):
                sel = light_shift_basis((s1, s2), f_laser, atom=self, B_gauss=float(B),
                                        check_loops=(laser_model == "rwa"))
            model = AtomicStructure(list(sel.manifolds), atom=self)

        # ---- bare transition frequency at B (magnetic sweep, lab frame) ----
        B = float(B)
        B_max = max(B + dB, dB)
        resB = model_B.magnetic_sweep(B_max=B_max, dB=dB, diamagnetic=diamagnetic)
        f_B0 = resB.get_transition_frequency(s1, s2, at=B)      # f(B, 0)

        # ---- add the laser light shift at the same field ----
        df_light = 0.0
        resL = None
        if use_light:
            if has_beam:
                I_max = (float(intensity) if intensity is not None
                         else float(beam.I0))
            else:
                beam = GaussianBeam(waist=1e-6, frequency=float(frequency_Hz),
                                    power=0.0)
                I_max = float(intensity)
            resL = model.laser_sweep(
                beam, I_max=I_max, n_points=n_points,
                model=laser_model, polarization=polarization, B_gauss=B,
            )
            # intensity difference cancels the RWA rotating-frame offset,
            # leaving the true lab-frame light shift of the transition.
            df_light = resL.transition_frequency_shift(s1, s2, at=I_max)
            resL.model_choice = model_choice

        f_BI = f_B0 + df_light                                  # f(B, I)

        # ---- apply the requested reference ----
        if relative_mode == "absolute":
            result = f_BI
        elif relative_mode == "optical":
            result = f_BI - f_B0                                # pure light shift
        else:  # "magnetic"
            f_00 = resB.get_transition_frequency(s1, s2, at=0.0)  # f(0, 0)
            result = f_BI - f_00

        sweep = resL if use_light else resB
        return (result, sweep) if return_sweep else result

    def get_intensity_from_light_shift(
        self,
        state1,
        state2,
        light_shift_Hz,
        B=0.0,
        beam=None,
        frequency_Hz=None,
        wavelength_m=None,
        polarization="pi",
        laser_model="auto",
        basis=None,
        n_points=200,
        I_max=None,
        I_guess=None,
        max_expansions=8,
        relative_mode=True,
        dB=0.1,
        diamagnetic=True,
        return_sweep=False,
    ):
        """Return the intensity (W/m^2) at which the ``state1 -> state2``
        transition reaches ``light_shift_Hz`` at field ``B``.

        A **single** laser-intensity sweep is run at the fixed magnetic field
        ``B`` and inverted for the requested target(s), so the cost is
        independent of how many targets are requested.

        Reference (``relative_mode``)
        -----------------------------
        ``relative_mode`` selects what ``light_shift_Hz`` means, mirroring the
        ``True``/``False`` behaviour of ``SweepResult.plot``'s
        ``plot_differential``:

        * ``True`` (default) — differential.  Each state's energy at the
          *start* of the intensity sweep is subtracted first, so the target is
          the light **shift** of the transition,

              ``df(I) = f(B, I) - f(B, 0)``     with    ``f = E2 - E1``.

          Taking the shift as an intensity difference cancels the RWA
          rotating-frame offset, so ``df`` is a true lab-frame shift.  This
          inverts :meth:`get_transition_frequency` with
          ``relative_mode="optical"``.
        * ``False`` — absolute.  The target is the full transition frequency
          ``f(B, I)`` itself.  The bare transition frequency ``f(B, 0)`` is
          obtained from a magnetic sweep at the same field (as in
          :meth:`get_transition_frequency`) and added to the light shift, so
          the inverted curve is a genuine lab-frame absolute frequency rather
          than an RWA rotating-frame value.  This inverts
          :meth:`get_transition_frequency` with ``relative_mode="absolute"``.

        The strings ``"optical"`` and ``"absolute"`` are accepted as aliases
        for ``True`` and ``False``, matching
        :meth:`get_transition_frequency`'s vocabulary.

        Sign convention
        ---------------
        ``light_shift_Hz`` is **signed** and refers to ``E2 - E1``.  In
        differential mode a positive value means the laser pushes the two
        levels apart, a negative value that it pulls them together; swapping
        ``state1``/``state2`` flips the sign.

        Unreachable targets
        -------------------
        A target the sweep never reaches is returned as ``NaN`` (rather than
        raising) and a :class:`RuntimeWarning` reports how many were dropped,
        their values, and the span actually covered — usually a sign or
        detuning mistake, or a target beyond ``I_max``.  Because every target
        is inverted from the same sweep, one unreachable entry never discards
        the rest of the vector.

        Intensity range
        ---------------
        The sweep runs over ``I in [0, I_max]``.  When ``I_max`` is omitted it
        is taken from ``beam.I0`` (if the beam carries power), else estimated
        from the low-intensity slope of ``df`` (which is linear in ``I`` for a
        far-detuned laser).  The range is then doubled — up to
        ``max_expansions`` times — while at least one missing target still lies
        beyond the end of the shift curve, since that is the only case the
        doubling can fix.

        Monotonic vs non-monotonic
        --------------------------
        Far from resonance ``df(I)`` is monotonic and the inversion is a single
        interpolation.  Close to resonance the RWA dressed-state curve can turn
        over, so one target can be produced at **several** intensities
        (branches).  In that case:

        * pass ``I_guess`` (W/m^2) and the branch nearest to it is returned;
        * omit ``I_guess`` and a :class:`ValueError` lists the branch
          intensities so you can pick one via ``I_guess``.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) 5-tuples
            Lower/upper states of the transition, with ``(a, b)`` either
            ``(F, mF)`` ints or ``(m_j, m_i)`` half-integer floats (standard
            ``kamo`` convention).  The target is referenced to ``E2 - E1``.
        light_shift_Hz : float or array-like
            Target(s) in Hz: a signed light shift when ``relative_mode`` is
            ``True``, or an absolute transition frequency when it is ``False``.
            Array-like returns an array of intensities (one per target), all
            read from the same sweep; unreachable targets come back as ``NaN``
            (see *Unreachable targets*).
        B : float, optional
            Static magnetic field in Gauss (default 0).  The sweep is run at
            this field, so the returned intensity accounts for the Zeeman
            structure of the two states.
        beam : kamo.GaussianBeam, optional
            Laser beam; supplies the laser frequency and, unless ``I_max`` is
            given, the sweep's upper intensity ``beam.I0``.
        frequency_Hz : float, optional
            Laser frequency (Hz), as an alternative to ``beam``.
        wavelength_m : float, optional
            Laser wavelength (m), as an alternative to ``beam``.  Exactly one
            of ``beam``, ``frequency_Hz``, ``wavelength_m`` must be given.
        polarization : str, optional
            Laser polarization: "pi", "sigma+", or "sigma-" (default "pi").
        laser_model : {"auto", "perturbative", "rwa", "stark"}, optional
            "auto" chooses as in :meth:`get_transition_frequency`, using
            ``I_max`` (or the beam's ``I0``) for the perturbativity check.
            Light-shift model (default "rwa").
        basis : AtomicStructure, optional
            Override the atomic-structure basis.  When omitted, the two states'
            manifolds plus their dipole-coupled (Delta l = +-1) neighbours are
            used, as required for a light shift.
        n_points : int, optional
            Intensity steps in the sweep (default 200).  More points give a
            finer inversion grid.
        I_max : float, optional
            Upper intensity of the sweep in W/m^2 (see *Intensity range*).
        I_guess : float, optional
            Intensity (W/m^2) used to disambiguate multiple branches: the
            crossing nearest ``I_guess`` is returned.
        max_expansions : int, optional
            Maximum number of ``I_max`` doublings (default 8).
        relative_mode : bool or {"optical", "absolute"}, optional
            Reference for ``light_shift_Hz`` (see above).  Default ``True``
            (differential / light shift).
        dB : float, optional
            Magnetic-sweep step in Gauss (default 0.1).  Only used when
            ``relative_mode`` is ``False``, where a magnetic sweep supplies the
            bare transition frequency ``f(B, 0)``.
        diamagnetic : bool, optional
            Include the diamagnetic term in that magnetic sweep (default True).
            Only used when ``relative_mode`` is ``False``.
        return_sweep : bool, optional
            If True, also return the underlying
            :class:`~kamo.hamiltonian.LaserSweepResult`.

        Returns
        -------
        float or np.ndarray
            The intensity/intensities in W/m^2, ``NaN`` where a target was not
            reachable, or ``(intensity, sweep)`` when ``return_sweep`` is True.

        Warns
        -----
        RuntimeWarning
            When one or more targets were unreachable and returned as ``NaN``.

        Raises
        ------
        ValueError
            If a target has multiple branches and no ``I_guess`` was given (the
            error lists the branch intensities).

        Examples
        --------
        >>> atom = Potassium39()
        >>> # 30 kHz measured shift of the |1,-1> -> |1,0> clock transition
        >>> # from a 780 nm beam at 100 G:
        >>> atom.get_intensity_from_light_shift(
        ...     (4, 0, 0.5, 1, -1), (4, 0, 0.5, 1, 0),
        ...     light_shift_Hz=30e3, B=100.0, wavelength_m=780e-9)
        >>> # a whole vector of shifts, inverted from one sweep:
        >>> atom.get_intensity_from_light_shift(
        ...     (4, 0, 0.5, 1, -1), (4, 0, 0.5, 1, 0),
        ...     light_shift_Hz=np.linspace(10e3, 50e3, 9),
        ...     B=100.0, wavelength_m=780e-9)
        >>> # the same, but targeting an absolute transition frequency:
        >>> atom.get_intensity_from_light_shift(
        ...     (4, 0, 0.5, 1, -1), (4, 0, 0.5, 1, 0),
        ...     light_shift_Hz=461.75e6, B=100.0, wavelength_m=780e-9,
        ...     relative_mode=False)

        See Also
        --------
        get_transition_frequency : forward direction (intensity -> frequency),
            whose ``relative_mode="optical"`` / ``"absolute"`` this inverts.
        kamo.hamiltonian.LaserSweepResult.intensity_from_splitting_shift :
            same inversion on an existing sweep, using ``|df|``.
        """
        from kamo import GaussianBeam
        from kamo.hamiltonian import (AtomicStructure, choose_laser_model,
                                      light_shift_basis)

        s1 = tuple(state1)
        s2 = tuple(state2)
        if len(s1) != 5 or len(s2) != 5:
            raise ValueError(
                "Each state must be a 5-tuple (n, l, j, a, b) where (a, b) are "
                "either (F, mF) ints or (m_j, m_i) half-integer floats.")

        # ---- validate / build the laser ----
        n_spec = sum(x is not None for x in (beam, frequency_Hz, wavelength_m))
        if n_spec != 1:
            raise ValueError(
                "Provide exactly one of `beam`, `frequency_Hz`, or "
                f"`wavelength_m`; got {n_spec}.")
        if beam is None:
            f_laser = (float(frequency_Hz) if frequency_Hz is not None
                       else c.c / float(wavelength_m))
            # power=0 -> I0=0; the sweep's intensity range is set below.
            beam = GaussianBeam(waist=1e-6, frequency=f_laser, power=0.0)

        # ---- resolve the reference mode ----
        if relative_mode is True or relative_mode == "optical":
            differential = True
        elif relative_mode is False or relative_mode == "absolute":
            differential = False
        else:
            raise ValueError(
                "relative_mode must be True/'optical' (target is a light "
                "shift) or False/'absolute' (target is a full transition "
                f"frequency); got {relative_mode!r}.")

        B = float(B)
        targets = np.atleast_1d(np.asarray(light_shift_Hz, dtype=float))
        scalar_in = np.ndim(light_shift_Hz) == 0

        # ---- choose the laser model ----
        model_choice = None
        I_choice = float(I_max) if I_max is not None else float(getattr(beam, "I0", 0.0))
        if laser_model == "auto":
            model_choice = choose_laser_model(
                (s1, s2), beam.frequency(), I_choice if I_choice > 0 else None,
                B_gauss=B, atom=self)
            laser_model = model_choice.model
        elif laser_model not in ("rwa", "stark", "perturbative"):
            raise ValueError(
                "laser_model must be 'auto', 'rwa', 'perturbative' or 'stark'; "
                f"got {laser_model!r}.")

        # ---- build / accept the basis (see get_transition_frequency) ----
        model_B = model = AtomicStructure(_own_manifolds(s1, s2), atom=self)
        if basis is not None:
            model_B = model = basis
        elif laser_model in ("rwa", "perturbative"):
            sel = model_choice.basis if model_choice is not None else None
            if sel is None or (laser_model == "perturbative" and sel.dropped):
                sel = light_shift_basis((s1, s2), beam.frequency(), atom=self, B_gauss=B,
                                        check_loops=(laser_model == "rwa"))
            model = AtomicStructure(list(sel.manifolds), atom=self)

        def _sweep(i_max, npts):
            return model.laser_sweep(
                beam, I_max=i_max, n_points=npts, model=laser_model,
                polarization=polarization, B_gauss=B,
            )

        # ---- reduce the targets to light shifts ----
        # The laser sweep only ever yields the *shift* df(I) = f(B, I) - f(B, 0)
        # as a true lab-frame quantity (the intensity difference cancels the RWA
        # rotating-frame offset).  In absolute mode the bare transition
        # frequency f(B, 0) comes from a magnetic sweep at the same field, and
        # subtracting it turns the absolute targets into shift targets, so the
        # inversion below is identical in both modes.
        f_B0 = 0.0
        if not differential:
            resB = model_B.magnetic_sweep(
                B_max=max(B + dB, dB), dB=dB, diamagnetic=diamagnetic)
            f_B0 = float(resB.get_transition_frequency(s1, s2, at=B))
        targets_shift = targets - f_B0
        quantity = "Light shift" if differential else "Transition frequency"

        # ---- choose the initial intensity range ----
        I_max = float(I_max) if I_max is not None else float(getattr(beam, "I0", 0.0))
        if I_max <= 0.0:
            # No intensity scale supplied: estimate one from the low-intensity
            # slope of df (linear in I for a far-detuned laser).
            I_probe = 1.0e2                     # 10 mW/cm^2 — safely perturbative
            slope = (_sweep(I_probe, 2)
                     .transition_frequency_shift(s1, s2)[-1] / I_probe)
            scale = float(np.max(np.abs(targets_shift)))
            I_max = 2.0 * scale / abs(slope) if (slope and scale) else I_probe

        # ---- sweep and invert, expanding the range until every target is hit ----
        # All targets share one sweep; the range is doubled only while at least
        # one still-missing target lies beyond the end of the (same-signed)
        # shift curve, since that is the only case doubling can fix.  Anything
        # still missing at the end is unreachable and comes back as NaN.
        resL = None
        roots = []
        n_doublings = max(0, int(max_expansions))
        for attempt in range(n_doublings + 1):
            resL = _sweep(I_max, n_points)
            shift = resL.transition_frequency_shift(s1, s2)      # signed, Hz
            roots = [self._crossings(resL.param, shift, t) for t in targets_shift]
            missing = [k for k, r in enumerate(roots) if len(r) == 0]
            if not missing:
                break
            df_end = float(shift[-1])
            expandable = [k for k in missing
                          if targets_shift[k] * df_end > 0
                          and abs(targets_shift[k]) > abs(df_end)]
            if not expandable or attempt == n_doublings:
                break
            I_max *= 2.0

        # ---- pick a branch per target (NaN where none was found) ----
        out = np.full(len(targets_shift), np.nan)
        unreachable = []
        for k, (t, r) in enumerate(zip(targets, roots)):
            if len(r) == 0:
                unreachable.append(t)
            elif len(r) == 1:
                out[k] = float(r[0])
            elif I_guess is not None:
                out[k] = float(r[np.argmin(np.abs(r - float(I_guess)))])
            else:
                shown = ", ".join(f"{v:.4e}" for v in r[:8])
                if len(r) > 8:
                    shown += f", ... ({len(r)} total)"
                hint = ""
                if len(r) > 4:
                    hint = (
                        f"  So many branches usually means {t:.4e} Hz is below "
                        "the numerical resolution of the sweep (the two states "
                        "shift almost identically), in which case no intensity "
                        "is well determined."
                    )
                raise ValueError(
                    f"{quantity} of {t:.4e} Hz occurs at multiple "
                    f"intensities (non-monotonic curve): branches at "
                    f"[{shown}] W/m^2.  Pass I_guess (W/m^2) to select the "
                    f"branch nearest a known intensity.{hint}"
                )

        if unreachable:
            import warnings
            miss_str = ", ".join(f"{t:.4e}" for t in unreachable[:8])
            if len(unreachable) > 8:
                miss_str += f", ... ({len(unreachable)} total)"
            warnings.warn(
                f"{len(unreachable)} of {len(targets)} target(s) returned as "
                f"NaN: {quantity.lower()}(s) [{miss_str}] Hz never occur for I "
                f"in [0, {I_max:.4e}] W/m^2, where the reachable range is "
                f"[{np.min(shift) + f_B0:.4e}, {np.max(shift) + f_B0:.4e}] Hz "
                f"after {attempt} doubling(s).  Check the sign of "
                "light_shift_Hz (it is "
                + ("f(B, I) - f(B, 0)" if differential else "f(B, I)")
                + " with f = E2 - E1) and the laser detuning; if the target "
                "is simply beyond the swept range, pass a larger I_max or "
                "raise max_expansions.",
                RuntimeWarning,
                stacklevel=2,
            )

        I_out = out
        result = float(I_out[0]) if scalar_in else I_out
        return (result, resL) if return_sweep else result

    def state_lookup(self, n, l, j, m1, m2):
        """Both label sets of one state (deprecated).

        Use :func:`kamo.hamiltonian.state_label` for labels, or
        ``Manifold(n, l, j).state_for(F, mF)`` / ``.label_for(m_j, m_i)`` to
        convert quantum numbers.

        Args:
            n, l, j: fine-structure quantum numbers.
            m1, m2: ``(F, mF)`` integers or ``(m_J, m_I)`` half-integers.

        Returns:
            dict: ``"hf"`` ``(m_J, m_I)``, ``"lf"`` ``(F, mF)`` (the two ends
            of the Paschen-Back adiabatic connection), and ``"hf_str"`` /
            ``"lf_str"``, the matching TeX kets.
        """
        from kamo.hamiltonian.state_labels import _manifold, state_label, is_coupled
        warnings.warn(
            f"{type(self).__name__}.state_lookup is deprecated; use "
            "kamo.hamiltonian.state_label or Manifold.state_for/label_for.",
            DeprecationWarning, stacklevel=2)
        man = _manifold(n, l, j, atom=self)
        try:
            if is_coupled(m1, m2, self.I):
                F, mF = self.coupled_qn(m1, m2)
                m_j, m_i = man.state_for(F, mF)
            else:
                m_j, m_i = float(m1), float(m2)
                F, mF = man.label_for(m_j, m_i)
        except KeyError as err:
            raise ValueError(err.args[0]) from None
        F, mF = self.coupled_qn(F, mF)
        return {
            "hf": (float(m_j), float(m_i)),
            "lf": (F, mF),
            "hf_str": state_label(n, l, j, float(m_j), float(m_i), term=False, atom=self),
            "lf_str": state_label(n, l, j, F, mF, term=False, atom=self),
        }

    def state_label(self,
                    n,l,j,
                    m1=None,m2=None,
                    skip_njl = False,
                    force_hf_lf = None,
                    force_skip_spin = False,
                    tex_formatting=True):
        r"""Spectroscopic label of a state, e.g. ``4S_{1/2}|F=1, m_F=-1\rangle``.

        Thin wrapper around :func:`kamo.hamiltonian.state_label`, returned
        *without* ``$`` delimiters so it can sit inside a larger math string.

        Args:
            n, l, j: fine-structure quantum numbers.
            m1, m2 (optional): ``(F, mF)`` integers or ``(m_J, m_I)``
                half-integers.  Give only ``m1`` for a one-number ket
                (``|F=2>`` or ``|m_J=+1/2>``).
            skip_njl (bool): leave out the term symbol ``nL_J``.
            force_hf_lf ({None, 'hf', 'lf'}): print the uncoupled ``(m_J, m_I)``
                ('hf') or coupled ``(F, mF)`` ('lf') ket, converting through the
                Paschen-Back adiabatic connection.  None keeps the input's basis.
            force_skip_spin (bool): leave out the ket.
            tex_formatting (bool): TeX (default) or plain text.

        Returns:
            str: e.g. ``4S_{1/2}|m_J=-1/2, m_I=+3/2\rangle``.
        """
        from kamo.hamiltonian import state_label
        bases = {None: None, 'hf': 'uncoupled', 'lf': 'coupled'}
        if force_hf_lf not in bases:
            raise ValueError(
                f"force_hf_lf must be None, 'hf', or 'lf'; got {force_hf_lf!r}.")
        spins = () if force_skip_spin else tuple(m for m in (m1, m2) if m is not None)
        return state_label(n, l, j, *spins,
                           basis=bases[force_hf_lf] if len(spins) == 2 else None,
                           math=False, term=not skip_njl, tex=tex_formatting,
                           atom=self)


# ------------------------------------------------------------ concrete atoms
#
# Potassium39 is in kamo.atom_properties.k39 (it adds the curated potassium
# hyperfine module and scattering lengths).

class Lithium6(PortalAlkali, arc.Lithium6):
    """6Li (I = 1). Integer nuclear spin: F and m_F are half-integers,
    m_I integers. ARC's Li levels are the 7Li ones (D lines 10.8 GHz off)."""
    species = "Li6"


class Lithium7(PortalAlkali, arc.Lithium7):
    """7Li (I = 3/2)."""
    species = "Li7"


class Sodium(PortalAlkali, arc.Sodium):
    """23Na (I = 3/2)."""
    species = "Na23"


class Potassium40(PortalAlkali, arc.Potassium40):
    """40K (I = 4). Integer nuclear spin: F and m_F are half-integers, m_I
    integers. ARC's K levels are the 39K ones (D lines 0.11 GHz off)."""
    species = "K40"


class Potassium41(PortalAlkali, arc.Potassium41):
    """41K (I = 3/2). ARC's K levels are the 39K ones (D lines 0.22 GHz off)."""
    species = "K41"


class Rubidium85(PortalAlkali, arc.Rubidium85):
    """85Rb (I = 5/2)."""
    species = "Rb85"


class Rubidium87(PortalAlkali, arc.Rubidium87):
    """87Rb (I = 3/2)."""
    species = "Rb87"


class Caesium(PortalAlkali, arc.Caesium):
    """133Cs (I = 7/2)."""
    species = "Cs133"


Cesium = Caesium
Sodium23 = Sodium
Caesium133 = Caesium
