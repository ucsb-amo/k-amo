"""kamo.atom_properties.alkali: the species-agnostic alkali layer.

TABLE        every ISOTOPES entry carries a reference for g_I and g_J, and
             resolves to a class; species strings normalize from any spelling.
CLASS ATTRS  the atom facts are class attributes: an instance built with
             __new__ alone (no ARC database) already has them.
ATOM FACTS   I, g_I (with its sign convention), g_J, ground state, lowest
             valence n of each l, cycling transition, Hamiltonian threshold.
HYPERFINE    ground-state A is the portal's measured value for all eight
             portal species (39K keeps its own curated module); B is ARC's or
             exactly zero for J = 1/2; an unknown state has has_A False.
BREIT-RABI   zero-field ground splittings against the literature, including
             the half-integer-F sets of the integer-spin isotopes 6Li and 40K.
THREADING    kamo.hamiltonian, kamo.constants and kamo.trap.polarizability all
             take the atom through (nuclear spin, g_I, species-keyed caches).

Runs offline: the cache points at an empty directory and the network is
disabled, so the portal data come from the snapshot bundled with kamo.

Run: pytest kamo/atom_properties/tests/test_alkali.py -q
"""
from __future__ import annotations

import inspect
import warnings

import arc
import numpy as np
import pytest
import scipy.constants as scon

from kamo.atom_properties import alkali
from kamo.atom_properties.alkali import (ISOTOPES, Rubidium87, atom, atom_class,
                                         lande_g_j, normalize_species)
from kamo.light_shift import udel_portal as up

pytestmark = pytest.mark.filterwarnings("ignore:UDel portal unreachable")

SPECIES = list(ISOTOPES)                       # the nine isotopes
PORTAL_SPECIES = [s for s in SPECIES if s != "K39"]   # 39K has its own module

#: Lowest valence n of each l, per element (lower n are core orbitals).
LOWEST_VALENCE_N = {
    "Li": {0: 2, 1: 2, 2: 3, 3: 4},
    "Na": {0: 3, 1: 3, 2: 3, 3: 4},
    "K":  {0: 4, 1: 4, 2: 3, 3: 4},
    "Rb": {0: 5, 1: 5, 2: 4, 3: 4},
    "Cs": {0: 6, 1: 6, 2: 5, 3: 4},
}

#: Measured ground-state hyperfine A (MHz).
GROUND_A_MHZ = {"Rb87": 3417.341, "Cs133": 2298.158, "Na23": 885.813,
                "Li7": 401.752, "Li6": 152.137, "K40": -285.731,
                "K41": 127.007, "Rb85": 1011.911}

#: Ground-state zero-field hyperfine splitting (MHz).
GROUND_SPLITTING_MHZ = {"Rb87": 6834.6826, "Cs133": 9192.6318, "Li6": 228.2053,
                        "K40": 1285.7886, "Na23": 1771.6261, "Li7": 803.5041,
                        "Rb85": 3035.7324, "K41": 254.0139}

#: A lab-frame pi polarization: eps parallel to the quantization axis.
#  (kamo.trap takes lab-frame 3-vectors, not the "pi"/"sigma+" names that
#  kamo.light_shift and kamo.hamiltonian.builder use.)
PI_POL = (0.0, 0.0, 1.0)
Z_HAT = (0.0, 0.0, 1.0)


@pytest.fixture(scope="module", autouse=True)
def offline(tmp_path_factory):
    """Portal data from the bundled snapshot only: empty cache, no network."""
    mp = pytest.MonkeyPatch()
    mp.setenv("KAMO_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))

    def no_network(*args, **kwargs):
        raise OSError("network disabled for tests")

    mp.setattr(up, "_http", no_network)
    yield
    mp.undo()
    # module-wide caches keyed by species were filled from the temp cache dir
    from kamo.hamiltonian.state_labels import clear_manifold_cache
    from kamo.trap import polarizability as pol
    pol.clear_caches()
    clear_manifold_cache()


@pytest.fixture(scope="module")
def atoms(offline):
    """One default-configured instance per species (~1 s for all nine)."""
    return {sp: atom(sp) for sp in SPECIES}


@pytest.fixture(scope="module")
def arc_atoms(offline):
    """One ``use_portal=False`` instance per portal species."""
    return {sp: atom(sp, use_portal=False) for sp in PORTAL_SPECIES}


# ===================================================================== TABLE
def test_table_covers_nine_isotopes():
    assert len(ISOTOPES) == 9
    assert set(ISOTOPES) == {"Li6", "Li7", "Na23", "K39", "K40", "K41",
                             "Rb85", "Rb87", "Cs133"}


@pytest.mark.parametrize("sp", SPECIES)
def test_every_isotope_entry_is_referenced(sp):
    d = ISOTOPES[sp]
    assert d.species == sp
    assert d.g_I_ref.strip(), f"{sp} has no reference for g_I"
    assert d.g_J_ref.strip(), f"{sp} has no reference for g_J"
    assert np.isfinite(d.g_I) and np.isfinite(d.g_J_ground)
    assert d.portal_species == f"{d.element}1"
    assert np.isfinite(d.core_polarizability_au)


@pytest.mark.parametrize("sp", SPECIES)
def test_atom_class_resolves(sp):
    cls = atom_class(sp)
    assert cls.__name__ == ISOTOPES[sp].arc_class
    assert issubclass(cls, alkali.PortalAlkali)
    assert cls.species == sp


@pytest.mark.parametrize("spelling", ["Rb87", "87Rb", "Rb-87", "rb87",
                                      "Rubidium87", " Rb_87 "])
def test_normalize_species_accepts_spellings(spelling):
    assert normalize_species(spelling) == "Rb87"


@pytest.mark.parametrize("spelling, expected", [
    ("Cs", "Cs133"), ("Caesium", "Cs133"), ("Cesium", "Cs133"),
    ("Na", "Na23"), ("Sodium", "Na23"), ("K-40", "K40"),
    ("Potassium41", "K41"), ("6Li", "Li6"),
])
def test_normalize_species_aliases(spelling, expected):
    assert normalize_species(spelling) == expected


@pytest.mark.parametrize("bad", ["Xe131", "Rb86", "", "argon"])
def test_normalize_species_rejects_unknown(bad):
    with pytest.raises(KeyError):
        normalize_species(bad)


# =============================================================== CLASS ATTRS
def test_class_attributes_exist_without_init():
    """__new__ alone: no ARC database, but every atom fact is already there."""
    a = Rubidium87.__new__(Rubidium87)
    assert a.species == "Rb87"
    assert a.element == "Rb"
    assert a.isotope == 87
    assert a.portal_species == "Rb1"
    assert a.gI == pytest.approx(-9.951414e-4)
    assert a.g_J_ground == pytest.approx(2.00233113)
    assert a.core_polarizability_au == pytest.approx(9.076)
    assert a.use_portal is True
    assert a.gI_ref and a.g_J_ground_ref


# ================================================================ ATOM FACTS
@pytest.mark.parametrize("sp", SPECIES)
def test_nuclear_spin_is_arcs(atoms, sp):
    a = atoms[sp]
    assert float(a.I) == float(getattr(arc, ISOTOPES[sp].arc_class).I)


@pytest.mark.parametrize("sp", SPECIES)
def test_nuclear_g_factor_is_the_table_value(atoms, sp):
    assert atoms[sp].gI == ISOTOPES[sp].g_I


def test_nuclear_g_factor_sign_convention(atoms):
    """H = mu_B B (g_J J_z + g_I I_z): g_I is negative except for 40K."""
    assert atoms["Rb87"].gI == pytest.approx(-9.951414e-4, rel=1e-9)
    assert atoms["K40"].gI == pytest.approx(+1.76490e-4, rel=1e-9)
    assert all(atoms[sp].gI < 0 for sp in SPECIES if sp != "K40")


@pytest.mark.parametrize("sp", SPECIES)
def test_g_j_ground_is_measured_and_excited_is_lande(atoms, sp):
    a = atoms[sp]
    n0 = int(a.groundStateN)
    assert a.g_J(0, 0.5) == ISOTOPES[sp].g_J_ground
    assert a.g_J(0, 0.5, n=n0) == ISOTOPES[sp].g_J_ground
    assert a.g_J(1, 1.5) == pytest.approx(lande_g_j(1, 1.5, 0.5, a.gL, a.gS), rel=1e-12)
    assert a.g_J(1, 1.5) == pytest.approx(4 / 3, rel=1e-3)
    # an excited S state is not the measured ground value
    assert a.g_J(0, 0.5, n=n0 + 1) == pytest.approx(lande_g_j(0, 0.5, 0.5, a.gL, a.gS))


@pytest.mark.parametrize("sp", SPECIES)
def test_ground_state_and_threshold(atoms, sp):
    a = atoms[sp]
    n0 = int(a.groundStateN)
    assert a.ground_state == (n0, 0, 0.5)
    assert a.hamiltonian_n_threshold == n0 + 6


@pytest.mark.parametrize("sp", SPECIES)
def test_lowest_valence_n(atoms, sp):
    a = atoms[sp]
    expected = LOWEST_VALENCE_N[ISOTOPES[sp].element]
    assert {l: a.lowest_valence_n(l) for l in expected} == expected
    assert a.is_valence(expected[0], 0)
    assert not a.is_valence(expected[0] - 1, 0)


@pytest.mark.parametrize("sp", SPECIES)
def test_cycling_transition_is_the_stretched_d2(atoms, sp):
    a = atoms[sp]
    n0 = int(a.groundStateN)
    assert a.cycling_transition == ((n0, 0, 0.5, a.I + 0.5),
                                    (n0, 1, 1.5, a.I + 1.5))


# ================================================================= HYPERFINE
@pytest.mark.parametrize("sp", PORTAL_SPECIES)
def test_ground_state_a_is_the_portal_measurement(atoms, sp):
    hc = atoms[sp].hyperfine_constants(*atoms[sp].ground_state)
    assert hc.A_source == "portal (measured)"
    assert hc.A_MHz == pytest.approx(GROUND_A_MHZ[sp], abs=1e-3)
    assert hc.has_A


@pytest.mark.parametrize("sp", PORTAL_SPECIES)
def test_quadrupole_b_is_arcs_or_exactly_zero(atoms, sp):
    a = atoms[sp]
    n0 = int(a.groundStateN)
    for state in [(n0, 0, 0.5), (n0, 1, 0.5), (n0, 1, 1.5)]:
        hc = a.hyperfine_constants(*state)
        assert hc.B_source in ("arc", "exact (J = 1/2)"), (sp, state)
        if state[2] == 0.5:
            assert hc.B_source == "exact (J = 1/2)"
            assert hc.B_MHz == 0.0


def test_rubidium87_nP3halves_constants(atoms):
    hc = atoms["Rb87"].hyperfine_constants(5, 1, 1.5)
    assert hc.A_MHz == pytest.approx(84.72, abs=1e-3)
    assert hc.B_MHz == pytest.approx(12.50, abs=1e-2)
    assert hc.A_Hz == hc.A_MHz * 1e6 and hc.B_Hz == hc.B_MHz * 1e6


def test_state_neither_source_knows_has_no_a(atoms):
    a = atoms["Rb87"]
    hc = a.hyperfine_constants(int(a.groundStateN), 3, 3.5)
    assert hc.has_A is False
    assert hc.A_source == "none" and hc.A_MHz == 0.0


@pytest.mark.parametrize("sp", PORTAL_SPECIES)
def test_use_portal_false_is_arcs_table(arc_atoms, sp):
    a = arc_atoms[sp]
    hc = a.hyperfine_constants(*a.ground_state)
    assert hc.A_source == "arc"


def test_get_hfs_coefficients_is_hz(atoms):
    a = atoms["Rb87"]
    hc = a.hyperfine_constants(5, 1, 1.5)
    assert a.getHFSCoefficients(5, 1, 1.5) == (hc.A_Hz, hc.B_Hz)
    assert a.getHFSCoefficients(5, 0, 0.5)[0] == pytest.approx(3417.341e6, abs=1e3)


def test_get_hfs_coefficients_raises_without_data(atoms):
    with pytest.raises(ValueError, match="No hyperfine data"):
        atoms["Rb87"].getHFSCoefficients(5, 3, 3.5)


# ================================================================ BREIT-RABI
@pytest.mark.parametrize("sp", PORTAL_SPECIES)
def test_ground_state_zero_field_splitting(atoms, sp):
    a = atoms[sp]
    energies, _, _ = a.breitRabi(*a.ground_state, np.array([0.0]))
    split_MHz = (energies.max() - energies.min()) / 1e6
    assert split_MHz == pytest.approx(GROUND_SPLITTING_MHZ[sp], rel=1e-6)


@pytest.mark.parametrize("sp", SPECIES)
def test_ground_state_f_labels_and_multiplicities(atoms, sp):
    a = atoms[sp]
    _, F, _ = a.breitRabi(*a.ground_state, np.array([0.0]))
    expected = {a.I - 0.5, a.I + 0.5}
    assert set(np.round(F, 9)) == {round(f, 9) for f in expected}
    for f in expected:
        assert np.count_nonzero(np.isclose(F, f)) == int(round(2 * f + 1))


@pytest.mark.parametrize("sp, F_set", [("Li6", {0.5, 1.5}), ("K40", {3.5, 4.5})])
def test_integer_nuclear_spin_has_half_integer_f(atoms, sp, F_set):
    a = atoms[sp]
    _, F, mF = a.breitRabi(*a.ground_state, np.array([0.0]))
    assert set(np.round(F, 9)) == F_set
    assert all(abs(2 * m - round(2 * m)) < 1e-9 and round(2 * m) % 2 == 1 for m in mF)


# ================================================== THREADING: kamo.hamiltonian
def test_atomic_structure_takes_the_atoms_nuclear_spin(atoms):
    from kamo.hamiltonian import AtomicStructure
    model = AtomicStructure([(6, 0, 0.5)], atom=atoms["Cs133"])
    assert model.basis.manifolds[0].i_nuclear == 3.5
    assert model.basis.dim == 16


def test_builder_rejects_a_basis_built_for_another_spin(atoms):
    """A default Basis is I = 3/2; Cs has I = 7/2."""
    from kamo.hamiltonian.basis import Basis
    from kamo.hamiltonian.builder import HamiltonianBuilder
    with pytest.raises(ValueError, match="nuclear spin"):
        HamiltonianBuilder(Basis([(6, 0, 0.5)]), atom=atoms["Cs133"])


def test_zeeman_operator_uses_the_atoms_nuclear_g(atoms):
    from kamo.hamiltonian.basis import Basis
    from kamo.hamiltonian.builder import HamiltonianBuilder
    import kamo.constants as c
    a = atoms["Rb87"]
    basis = Basis([(5, 0, 0.5)], atom=a)
    diag = np.diag(HamiltonianBuilder(basis, atom=a).zeeman_operator())
    lo = basis.index_of(5, 0, 0.5, -0.5, -0.5)
    hi = basis.index_of(5, 0, 0.5, -0.5, +0.5)
    per_gauss = c.mu_b * 1e-4 / c.h                     # Hz/G per unit g m
    assert diag[hi] - diag[lo] == pytest.approx(-9.951414e-4 * per_gauss, rel=1e-9)


def test_make_nlj_basis_uses_the_atoms_core_shells():
    from kamo.hamiltonian import make_nlj_basis
    assert make_nlj_basis(4, 2, n_range=0, l_range=0, atom=Rubidium87) == [
        (4, 2, 1.5), (4, 2, 2.5)]
    assert make_nlj_basis(4, 0, n_range=0, l_range=0, atom=Rubidium87) == []  # 4s is core


# ==================================================== THREADING: kamo.constants
def test_m_K_is_silent_and_is_arcs_mass():
    import kamo.constants as c
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = c.m_K
    assert value == arc.Potassium39.mass
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("name, expected", [
    ("g_I", -0.00014193489),
    ("g_L", 1 - scon.m_e / (38.9637064864
                            * scon.physical_constants["atomic mass constant"][0])),
    ("g_J_4S", 2.00229421),
])
def test_deprecated_constants_warn_and_keep_their_values(name, expected):
    import kamo.constants as c
    with pytest.deprecated_call():
        value = getattr(c, name)
    assert value == pytest.approx(expected, rel=1e-12)


def test_deprecated_hyperfine_helper():
    import kamo.constants as c
    with pytest.deprecated_call():
        A_J = c.get_hyperfine_constant(0, 0.5)
    assert A_J == pytest.approx(c.h * 230.8598601e6, rel=1e-12)


def test_deprecated_g_factor_helper():
    import kamo.constants as c
    with pytest.deprecated_call():
        g = c.get_total_electronic_g_factor(1, 1.5)
    assert g == pytest.approx(1.334097, rel=1e-6)


# ================================================================ 39K UNCHANGED
def test_potassium39_positional_signature(offline):
    from kamo import Potassium39
    params = list(inspect.signature(Potassium39.__init__).parameters)
    assert params == ["self", "use_portal", "portal_species", "preferQuantumDefects"]
    a = Potassium39(True, "K1", False)               # positional, as before
    assert a.use_portal is True and a.portal_species == "K1"


def test_potassium39_keeps_its_curated_hyperfine(atoms):
    a = atoms["K39"]
    assert a.hyperfine_constants(4, 1, 0.5).A_MHz == pytest.approx(27.793)
    assert a.hyperfine_constants(4, 0, 0.5).A_MHz == pytest.approx(230.8598601)


def test_potassium39_imports_are_unchanged():
    from kamo import Potassium39                          # noqa: F401
    from kamo.atom_properties.k39 import _own_manifolds   # noqa: F401
    assert _own_manifolds((4, 1, 1.5)) == [(4, 1, 0.5), (4, 1, 1.5)]


def test_default_atom_is_potassium39(atoms):
    from kamo import Potassium39
    from kamo.atom_properties.alkali import default_atom, species_of
    assert isinstance(default_atom(), Potassium39)
    assert species_of(atoms["K39"]) == "K39"
    assert species_of(default_atom()) == "K39"


def test_plain_arc_atom_still_works(offline):
    """A bare ARC atom duck-types: nuclear spin and ARC's own hyperfine."""
    from kamo.atom_properties.alkali import hyperfine, species_of
    from kamo.hamiltonian import AtomicStructure
    plain = arc.Rubidium87()
    model = AtomicStructure([(5, 0, 0.5)], atom=plain)
    assert model.basis.manifolds[0].i_nuclear == 1.5
    hc = hyperfine(plain, 5, 0, 0.5)
    assert hc.A_source == "arc"
    assert hc.A_MHz == pytest.approx(3417.341, rel=1e-6)
    assert species_of(plain) == "Rubidium87"


# ==================================================== THREADING: kamo.trap
def test_state_polarizability_takes_the_atom(atoms):
    from kamo.trap.polarizability import StatePolarizability
    sp = StatePolarizability((5, 0, 0.5, 2, -2), atom=atoms["Rb87"])
    assert sp.species == "Rb87"
    assert sp.nuclear_spin == 1.5
    alpha = sp.alpha_au(1064e-9, PI_POL, Z_HAT)
    assert alpha > 0
    # 87Rb 5S1/2 scalar polarizability at 1064 nm, ~687 a.u. (Safronova)
    assert alpha == pytest.approx(687.0, rel=0.15)


def test_compute_polarizabilities_is_keyed_by_species():
    from kamo.trap.polarizability import compute_polarizabilities
    cp = compute_polarizabilities("portal", "Rb87")
    assert isinstance(cp.atom, Rubidium87)
    assert cp.atom.portal_species == "Rb1"


def test_state_polarizability_defaults_to_39K():
    from kamo.trap.polarizability import StatePolarizability
    sp = StatePolarizability((4, 0, 0.5, 1, -1))
    assert sp.species == "K39"
    assert sp.nuclear_spin == 1.5
    assert sp.alpha_au(1064e-9, PI_POL, Z_HAT) > 0
