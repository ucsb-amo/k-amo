"""Label conventions for integer nuclear spin (6Li, I = 1; 40K, I = 4).

For half-integer I (39K, 87Rb, ...) the familiar rule holds: integers are the
coupled (F, m_F), half-integers the uncoupled (m_J, m_I).  For integer I it is
the reverse -- F and m_F are half-integers and m_I an integer -- so every label
function has to read the basis from the parity of the numbers *and* the atom's
nuclear spin.

BASIS SNIFF   is_coupled follows the parity of I, and rejects mixed pairs.
FORMATTER     state_label prints and converts half-integer F for 6Li.
LABEL MAP     40K's 4S1/2 map is a bijection over 2(2I+1) = 18 states.
QUANTUM NRS   coupled_qn keeps ints for half-integer I, halves for integer I;
              the deprecated state_lookup round-trips for both.

Runs offline: the portal cache is an empty directory and the network is off.

Run: pytest kamo/hamiltonian/tests/test_integer_spin.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from kamo.hamiltonian.basis import Manifold
from kamo.hamiltonian.state_labels import is_coupled, state_label
from kamo.light_shift import udel_portal as up

pytestmark = pytest.mark.filterwarnings("ignore:UDel portal unreachable")


@pytest.fixture(scope="module", autouse=True)
def offline(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("KAMO_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))

    def no_network(*args, **kwargs):
        raise OSError("network disabled for tests")

    mp.setattr(up, "_http", no_network)
    yield
    mp.undo()
    from kamo.hamiltonian.state_labels import clear_manifold_cache
    clear_manifold_cache()


@pytest.fixture(scope="module")
def li6(offline):
    from kamo.atom_properties.alkali import Lithium6
    return Lithium6()


@pytest.fixture(scope="module")
def k40(offline):
    from kamo.atom_properties.alkali import Potassium40
    return Potassium40()


@pytest.fixture(scope="module")
def rb87(offline):
    from kamo.atom_properties.alkali import Rubidium87
    return Rubidium87()


# ================================================================ BASIS SNIFF
@pytest.mark.parametrize("a, b, I, coupled", [
    (1, -1, 1.5, True),            # (F, m_F) for half-integer I
    (0.5, -1.5, 1.5, False),       # (m_J, m_I) for half-integer I
    (1.5, -0.5, 1.0, True),        # (F, m_F) for integer I
    (0.5, 1, 1.0, False),          # (m_J, m_I) for integer I
    (0.5, -4, 4.0, False),         # 40K: m_I is an integer
    (4.5, -4.5, 4.0, True),        # 40K: F and m_F are half-integers
])
def test_is_coupled_follows_the_parity_of_I(a, b, I, coupled):
    assert is_coupled(a, b, I) is coupled


@pytest.mark.parametrize("a, b, I", [
    (1, -0.5, 1.5),                # integer with a half-integer, half-integer I
    (1, -0.5, 1.0),                # ... and integer I
    (0.5, 1, 1.5),
    (1.25, 0.5, 1.5),              # not a (half-)integer at all
])
def test_is_coupled_rejects_mixed_parity(a, b, I):
    with pytest.raises(ValueError):
        is_coupled(a, b, I)


# =================================================================== FORMATTER
def test_state_label_prints_half_integer_f_for_lithium6(li6):
    assert state_label(2, 0, 0.5, 1.5, -1.5, atom=li6) == \
        r"$2S_{1/2}|F=3/2, m_F=-3/2\rangle$"


def test_state_label_converts_to_uncoupled_for_lithium6(li6):
    assert state_label(2, 0, 0.5, 1.5, -1.5, atom=li6, basis="uncoupled") == \
        r"$2S_{1/2}|m_J=-1/2, m_I=-1\rangle$"


def test_state_label_uses_the_default_atom_convention_without_an_atom():
    """Same numbers, half-integer I (39K): (3/2, -3/2) reads as (m_J, m_I)."""
    with pytest.raises(ValueError):           # m_J = 3/2 is not in a j = 1/2 manifold
        state_label(4, 0, 0.5, 1.5, -1.5)


# =================================================================== LABEL MAP
def test_potassium40_ground_label_map_is_a_bijection(k40):
    man = Manifold(4, 0, 0.5, atom=k40)
    assert man.i_nuclear == 4.0
    assert man.dim == int(2 * (2 * man.i_nuclear + 1)) == 18

    fwd = man.label_map
    assert len(fwd) == 18
    assert len(set(fwd.values())) == 18                     # injective
    assert set(man.reverse_label_map) == set(fwd.values())  # and onto

    F_values = [F for F, _ in fwd.values()]
    assert set(F_values) == {3.5, 4.5}
    for F in (3.5, 4.5):
        assert F_values.count(F) == int(round(2 * F + 1))
    # m_I integer, m_J half-integer; F, m_F half-integer
    for (m_j, m_i), (F, mF) in fwd.items():
        assert float(m_i) == round(float(m_i))
        assert abs(m_j) == 0.5
        assert round(2 * F) % 2 == 1 and round(2 * mF) % 2 == 1
        assert mF == pytest.approx(m_j + m_i)


def test_label_map_round_trips(k40):
    man = Manifold(4, 0, 0.5, atom=k40)
    for (m_j, m_i), (F, mF) in man.label_map.items():
        assert man.state_for(F, mF) == (m_j, m_i)
        assert man.label_for(m_j, m_i) == (F, mF)


# ================================================================== QUANTUM NRS
def test_coupled_qn_types(rb87, k40, li6):
    assert rb87.half_integer_I and not k40.half_integer_I
    F, mF = rb87.coupled_qn(2, -2)
    assert (F, mF) == (2, -2) and isinstance(F, int) and isinstance(mF, int)
    F, mF = k40.coupled_qn(4.5, -4.5)
    assert (F, mF) == (4.5, -4.5) and isinstance(F, float) and isinstance(mF, float)
    F, mF = li6.coupled_qn(1.5, 0.5)
    assert (F, mF) == (1.5, 0.5) and isinstance(F, float)


@pytest.mark.parametrize("atom_name, state, hf, lf", [
    ("rb87", (5, 0, 0.5, 2, -2), (-0.5, -1.5), (2, -2)),
    ("k40", (4, 0, 0.5, 4.5, -4.5), (-0.5, -4.0), (4.5, -4.5)),
])
def test_state_lookup_round_trips(request, atom_name, state, hf, lf):
    atom = request.getfixturevalue(atom_name)
    with pytest.deprecated_call():
        d = atom.state_lookup(*state)
    assert d["hf"] == hf
    assert d["lf"] == lf
    with pytest.deprecated_call():
        back = atom.state_lookup(*state[:3], *d["hf"])
    assert back["lf"] == lf
    assert back["hf"] == hf


def test_potassium40_ground_transition_is_the_hyperfine_splitting(k40):
    """|F=9/2, m_F=-9/2> -> |F=7/2, m_F=-7/2> at B = 0 is (I + 1/2) |A|."""
    hc = k40.hyperfine_constants(4, 0, 0.5)
    expected = abs(hc.A_MHz) * (k40.I + 0.5)
    got = k40.get_ground_state_transition_frequency(4.5, -4.5, 3.5, -3.5, B=0)
    assert got == pytest.approx(expected, rel=1e-6)
    assert got == pytest.approx(1285.7886, rel=1e-6)

    energies, _, _ = k40.breitRabi(4, 0, 0.5, np.array([0.0]))
    assert (energies.max() - energies.min()) / 1e6 == pytest.approx(got, rel=1e-6)
