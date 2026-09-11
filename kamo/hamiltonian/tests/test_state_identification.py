"""Tests that a tracked state's label is the state the filter actually picked.

INTERNAL      every tracked state of a magnetic sweep round-trips through
              label -> filter -> index, and no two states share a label.
GROUND-TRUTH  the label is the state's true high-field (Paschen-Back)
              character, read off the sweep at 1.5 kG.

The regression these guard: labels used to come from the CG-dominant uncoupled
component at B = 0, while every ``states=`` filter used the Paschen-Back
adiabatic convention.  Plotting ``states=(4, 1, 1.5, -3/2)`` then produced a
legend in which three of the four lines carried a wrong (m_J, m_I) - most of
them repeating m_I = -3/2 - even though the four lines themselves were right.

Run: pytest kamo/hamiltonian/tests -q
"""
from __future__ import annotations

import pytest

from kamo.hamiltonian import AtomicStructure
from kamo.hamiltonian.state_labels import _frac

MANIFOLDS = [(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)]
B_PASCHEN_BACK = 1500.0     # well past the 4S1/2 and 4P3/2 crossovers
DB = 2.0                    # fine enough that eigenshuffle keeps every thread


@pytest.fixture(scope="module")
def sweep():
    model = AtomicStructure(MANIFOLDS)
    return model.magnetic_sweep(B_max=B_PASCHEN_BACK, dB=DB)


# ========================================================= the reported case
def test_mJ_filter_labels_span_all_mI(sweep):
    """states=(4, 1, 3/2, -3/2) must give four lines labelled m_J = -3/2 with
    the four distinct m_I, not four copies of m_I = -3/2."""
    idxs = sweep._resolve_states((4, 1, 1.5, -1.5))
    assert len(idxs) == 4

    labels = [sweep.adiabatic_state(i)[3:] for i in idxs]
    assert all(m_j == -1.5 for m_j, _ in labels)
    assert sorted(m_i for _, m_i in labels) == [-1.5, -0.5, 0.5, 1.5]

    # and the rendered legend entries say the same thing
    for (m_j, m_i), i in zip(labels, idxs):
        assert f"m_I={_frac(m_i)}" in sweep.tex_label(i)
        assert f"m_j={_frac(m_j)}" in sweep.label(i)


# ================================================================== INTERNAL
def test_label_round_trips_to_its_own_index(sweep):
    """The label of tracked state i must resolve back to exactly [i] — the
    labelling and selection conventions cannot disagree."""
    for i in range(sweep.energies.shape[1]):
        label = sweep.adiabatic_state(i)
        assert sweep._resolve_states(label) == [i], (i, label)


def test_labels_are_unique(sweep):
    labels = [sweep.adiabatic_state(i) for i in range(sweep.energies.shape[1])]
    assert len(set(labels)) == len(labels)


def test_label_is_step_independent(sweep):
    """Eigenshuffle tracks each state continuously, so its label is a property
    of the whole sweep — evaluating it at a later step must not change it."""
    last = len(sweep.param) - 1
    for i in range(sweep.energies.shape[1]):
        assert sweep.adiabatic_state(i, 0) == sweep.adiabatic_state(i, last)


def test_F_filter_returns_2F_plus_1_states(sweep):
    for F in (0, 1, 2, 3):
        idxs = sweep._resolve_states((4, 1, 1.5, F))
        assert len(idxs) == 2 * F + 1
        assert all(f"(F={F}," in sweep.label(i) for i in idxs)


# ============================================================== GROUND-TRUTH
def test_label_is_the_high_field_character(sweep):
    """At 1.5 kG the manifolds are Paschen-Back, so the dominant uncoupled
    component *is* the adiabatic label — that is what the label must report."""
    last = len(sweep.param) - 1
    for i in range(sweep.energies.shape[1]):
        s = sweep.dominant_state(i, last)
        assert sweep.adiabatic_state(i) == (
            s.n, s.l, s.j, float(s.m_j), float(s.m_i)), i


def test_zero_field_degeneracy_does_not_alias_states(sweep):
    """The (2F+1) states of an F multiplet are exactly degenerate at B = 0, so
    ``eigh`` returns them mixed across m_F; identification must not rely on a
    per-(F, mF) overlap there.  Two 4P3/2 states used to be aliased this way."""
    for F in (0, 1, 2, 3):
        for mF in range(-F, F + 1):
            i = sweep._tracked_index_F_mF(4, 1, 1.5, F, mF, step=0)
            assert sweep._manifold(4, 1, 1.5).label_for(
                *sweep.adiabatic_state(i)[3:]) == (F, mF)
