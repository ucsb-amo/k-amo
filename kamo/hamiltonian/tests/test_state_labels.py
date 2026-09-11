"""State labels: the Paschen-Back label map, tracked-state labels, and the one
formatter every label goes through.

LABEL MAP     Manifold.label_map is the real adiabatic connection: the F a
              tracked state has at B = 0 and the (m_J, m_I) it has at high
              field, read off a sweep, for S, P and D manifolds, including the
              inverted (A < 0) nD5/2.
NO HYPERFINE  a manifold with no A constant (l >= 3) is fully degenerate at
              B = 0.  Its tracked states used to be labelled from arbitrary
              zero-field eigenvectors (9 of 16 wrong for a 16-state
              manifold); they must get their true (m_J, m_I).
MISTRACK      two tracks of one m_F block crossing (an eigenshuffle swap) warns.
LASER SWEEP   labels at I = 0 agree with the magnetic sweep to the same field.
FORMATTER     exact strings from state_label and the Potassium39 wrapper,
              including the 4P3/2 m_J = +3/2 states the old hand-typed dicts
              got wrong.

Run: pytest kamo/hamiltonian/tests -q
"""
from __future__ import annotations

import numpy as np
import pytest

from kamo import constants as c
from kamo.hamiltonian import AtomicStructure, state_label, rs_state_label
from kamo.hamiltonian import state_labels
from kamo.hamiltonian.builder import _clebsch
from kamo.hamiltonian.diagonalize import MagneticSweepResult

MANIFOLDS_WITH_A = [(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5), (5, 0, 0.5),
                    (5, 1, 0.5), (5, 1, 1.5), (6, 0, 0.5), (6, 1, 1.5),
                    (3, 2, 1.5), (3, 2, 2.5), (4, 2, 2.5)]    # nD5/2: A < 0
NO_HYPERFINE = (4, 3, 3.5)          # l >= 3: kamo has no A constant


@pytest.fixture(scope="module")
def atom():
    from kamo import Potassium39
    return Potassium39(use_portal=False)


@pytest.fixture
def fresh_warnings(monkeypatch):
    """Reset the once-per-manifold 'no hyperfine structure' warning."""
    monkeypatch.setattr(state_labels, "_WARNED_NO_HYPERFINE", set())


def _pure_high_field(sweep, i):
    """(m_j, m_i) of tracked state i at the last step, which must be nearly pure."""
    w = np.abs(sweep.vectors[-1][:, i]) ** 2
    assert w.max() > 0.9, "sweep does not reach the Paschen-Back regime"
    s = sweep.basis[int(np.argmax(w))]
    return float(s.m_j), float(s.m_i)


# ================================================================= LABEL MAP
@pytest.mark.filterwarnings("error::RuntimeWarning")      # no false mistracks
@pytest.mark.parametrize("nlj", MANIFOLDS_WITH_A)
def test_label_map_is_the_adiabatic_connection(atom, nlj):
    n, l, j = nlj
    A_MHz = abs(c.get_hyperfine_constant(l, j, n=n)) / c.h / 1e6
    assert A_MHz > 0
    # reach Paschen-Back, in steps fine enough for eigenshuffle to follow the
    # Breit-Rabi region (width ~ A / (1.4 MHz/G))
    B_max = max(100.0, 8.0 * A_MHz)
    dB = min(2.0, B_max / 300, 0.1 * A_MHz / 1.4)
    sweep = AtomicStructure([nlj], atom=atom).magnetic_sweep(B_max=B_max, dB=dB)
    man = sweep.basis.manifolds[0]

    for i in range(sweep.energies.shape[1]):
        m_j, m_i = _pure_high_field(sweep, i)
        # F at B = 0: weight on each whole F multiplet (blind to the arbitrary
        # rotation inside a degenerate multiplet)
        w_F = {}
        for F in (int(round(f)) for f in man.allowed_F()):
            psi = np.array([[_clebsch(j, s.m_j, man.i_nuclear, s.m_i, F, mF)
                             for s in sweep.basis.state_list]
                            for mF in range(-F, F + 1)])
            w_F[F] = float(np.sum(np.abs(psi @ sweep.vectors[0][:, i]) ** 2))
        F = max(w_F, key=w_F.get)
        assert w_F[F] > 0.99
        assert man.label_map[(m_j, m_i)] == (F, int(round(m_j + m_i))), (i, w_F)
        assert sweep.adiabatic_state(i) == (n, l, j, m_j, m_i)


# ============================================================== NO HYPERFINE
@pytest.mark.filterwarnings("ignore:No hyperfine A constant:UserWarning")
@pytest.mark.parametrize("manifolds", [[NO_HYPERFINE],
                                       [(4, 0, 0.5), NO_HYPERFINE]])
def test_no_hyperfine_manifold_gets_its_true_labels(atom, manifolds):
    sweep = AtomicStructure(manifolds, atom=atom).magnetic_sweep(B_max=50.0, dB=0.5)
    idxs = sweep.indices_for(*NO_HYPERFINE)
    assert len(idxs) == 32
    for i in idxs:
        assert sweep.adiabatic_state(i)[3:] == _pure_high_field(sweep, i)
        assert sweep._resolve_states(sweep.adiabatic_state(i)) == [i]
        assert "F=" not in sweep.label(i)


def test_no_hyperfine_coupled_label_falls_back(fresh_warnings):
    with pytest.warns(UserWarning, match="no hyperfine structure"):
        lbl = state_label(*NO_HYPERFINE, -1.5, -0.5, basis="coupled")
    assert lbl == r"$4F_{7/2}|m_J=-3/2, m_I=-1/2\rangle$"


# ================================================================== MISTRACK
def test_crossing_inside_an_mF_block_warns(atom):
    good = AtomicStructure([(4, 0, 0.5)], atom=atom).magnetic_sweep(B_max=100.0, dB=0.5)
    pair = [i for i in range(8) if good._tracked_mF(i) == 0]
    assert len(pair) == 2

    # swap the two m_F = 0 tracks halfway, as a too-coarse dB would
    k = len(good.param) // 2
    E, V = good.energies.copy(), good.vectors.copy()
    E[k:, pair] = E[k:, pair[::-1]]
    V[k:, :, pair] = V[k:, :, pair[::-1]]
    bad = MagneticSweepResult(good.param, good.param_name, E, V, good.basis)
    with pytest.warns(RuntimeWarning, match="change energy order"):
        bad.adiabatic_state(pair[0])


# =============================================================== LASER SWEEP
def test_laser_sweep_labels_match_magnetic_sweep(atom):
    from kamo import GaussianBeam
    model = AtomicStructure([(4, 0, 0.5), (4, 1, 1.5)], atom=atom)
    B = 100.0
    resB = model.magnetic_sweep(B_max=B, dB=0.5)
    beam = GaussianBeam(waist=50e-6, wavelength=766.0e-9, power=1e-3)
    resL = model.laser_sweep(beam, I_max=beam.I0, n_points=3, B_gauss=B)

    # step 0 of the laser sweep is the Zeeman eigenbasis at B
    overlap = np.abs(resB.vectors[-1].conj().T @ resL.vectors[0]) ** 2
    for iL in range(resL.energies.shape[1]):
        iB = int(np.argmax(overlap[:, iL]))
        assert overlap[iB, iL] > 0.99
        assert resL.adiabatic_state(iL) == resB.adiabatic_state(iB)


# ================================================================= FORMATTER
@pytest.mark.parametrize("args, kwargs, expected", [
    ((4, 0, 0.5), {}, r"$4S_{1/2}$"),
    ((4, 2, 2.5), {}, r"$4D_{5/2}$"),
    ((4, 0, 0.5, 1, -1), {}, r"$4S_{1/2}|F=1, m_F=-1\rangle$"),
    ((4, 0, 0.5, 1, 0), {}, r"$4S_{1/2}|F=1, m_F=0\rangle$"),
    ((4, 0, 0.5, 1, -1), {"basis": "uncoupled"}, r"$4S_{1/2}|m_J=-1/2, m_I=-1/2\rangle$"),
    ((4, 0, 0.5, 0.5, -1.5), {"basis": "coupled"}, r"$4S_{1/2}|F=2, m_F=-1\rangle$"),
    ((4, 1, 1.5, 2), {}, r"$4P_{3/2}|F=2\rangle$"),
    ((4, 1, 1.5, 0.5), {}, r"$4P_{3/2}|m_J=+1/2\rangle$"),
    (((4, 0, 0.5, -0.5, 1.5),), {"term": False, "math": False},
     r"|m_J=-1/2, m_I=+3/2\rangle"),
    ((4, 0, 0.5, 1, -1), {"tex": False}, "4S_1/2|F=1, m_F=-1⟩"),
])
def test_state_label_strings(args, kwargs, expected):
    assert state_label(*args, **kwargs) == expected


@pytest.mark.parametrize("mF, m_i", [(0, -1.5), (1, -0.5), (2, 0.5), (3, 1.5)])
def test_4P3halves_F3_connects_to_mJ_plus_3halves(mF, m_i):
    """The old hand-typed dicts mapped these four to m_J = -1/2."""
    assert state_label(4, 1, 1.5, 3, mF, basis="uncoupled") == (
        rf"$4P_{{3/2}}|m_J=+3/2, m_I={state_labels._frac(m_i)}\rangle$")


def test_rs_state_label_is_state_label():
    assert rs_state_label(4, 1, 1.5, 0.5, -1.5) == r"$4P_{3/2}|m_J=+1/2, m_I=-3/2\rangle$"
    assert rs_state_label((4, 1, 1.5, 2, -2)) == r"$4P_{3/2}|F=2, m_F=-2\rangle$"


@pytest.mark.parametrize("args, kwargs", [
    ((4, 0, 0.5, 1, 0.5), {}),                  # mixed integer / half-integer
    ((4, 0, 0.5, 3, 0), {}),                    # no F = 3 in 4S1/2
    ((4, 0, 0.5, -0.5, 2.5), {}),               # m_I out of range
    ((4, 0, 0.5, 1, -1), {"basis": "hf"}),      # unknown basis name
    ((4, 1, 1.5, 2), {"basis": "uncoupled"}),   # can't convert one number
    ((4, 0, 1.5, 1, -1), {}),                   # j incompatible with l
])
def test_state_label_rejects(args, kwargs):
    with pytest.raises(ValueError):
        state_label(*args, **kwargs)


# ======================================================= Potassium39 wrapper
@pytest.fixture(scope="module")
def k39():
    from kamo import Potassium39
    # the label methods use no ARC data, so skip ARC's database load
    return Potassium39.__new__(Potassium39)


@pytest.mark.parametrize("args, kwargs, expected", [
    ((4, 0, 0.5, 1, -1), {}, r"4S_{1/2}|F=1, m_F=-1\rangle"),
    ((4, 0, 0.5, 1, -1), {"force_hf_lf": "hf"}, r"4S_{1/2}|m_J=-1/2, m_I=-1/2\rangle"),
    ((4, 1, 1.5, 1.5, 0.5), {"force_hf_lf": "lf"}, r"4P_{3/2}|F=3, m_F=+2\rangle"),
    ((5, 1, 1.5, -1.5, -0.5), {"skip_njl": True}, r"|m_J=-3/2, m_I=-1/2\rangle"),
    ((4, 0, 0.5, 1, -1), {"force_skip_spin": True}, r"4S_{1/2}"),
    ((60, 0, 0.5), {}, r"60S_{1/2}"),
    ((4, 2, 2.5), {}, r"4D_{5/2}"),                          # used to raise
    ((4, 0, 0.5), {"tex_formatting": False}, "4S_1/2"),      # used to raise
])
def test_potassium39_state_label(k39, args, kwargs, expected):
    assert k39.state_label(*args, **kwargs) == expected


def test_potassium39_state_label_rejects_bad_force(k39):
    with pytest.raises(ValueError):
        k39.state_label(4, 0, 0.5, 1, -1, force_hf_lf="high")


def test_state_lookup_shim(k39):
    with pytest.warns(DeprecationWarning):
        d = k39.state_lookup(4, 1, 1.5, 3, 3)
    assert d["hf"] == (1.5, 1.5) and d["lf"] == (3, 3)
    assert d["hf_str"] == r"$|m_J=+3/2, m_I=+3/2\rangle$"
    assert d["lf_str"] == r"$|F=3, m_F=+3\rangle$"
    with pytest.warns(DeprecationWarning):
        assert k39.state_lookup(4, 0, 0.5, -0.5, 0.5)["lf"] == (1, 0)
