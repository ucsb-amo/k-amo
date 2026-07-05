"""kamo.hamiltonian — multi-manifold basis construction & diagonalization for K39.

Build a basis from any number of (n, l, j) fine-structure manifolds, assemble
Hamiltonian terms (fine + hyperfine, paramagnetic Zeeman, optional diamagnetic,
and laser fields from a GaussianBeam via RWA or AC-Stark), and solve by direct
diagonalization.  Field and intensity sweeps use eigenshuffle to track states
through avoided crossings.

Quick start
-----------
>>> from kamo.hamiltonian import AtomicStructure
>>> model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)])
>>> res = model.magnetic_sweep(B_max=600.0)     # 0.1 G steps by default
"""

from .basis import Basis, BasisState, Manifold
from .builder import HamiltonianBuilder
from .diagonalize import (MagneticSweepResult, LaserSweepResult,
                          SweepResult, diagonalize, eigenshuffle, sweep_field,
                          sweep_intensity)
from .model import AtomicStructure, make_nlj_basis
from .state_labels import uncoupled_label, coupled_label, both_labels, format_state
from .spectroscopy import (
    field_from_splitting,
    intensity_from_splitting_shift,
    scattering_rate,
    dominant_couplings,
    transition_frequency_shift,
)

__all__ = [
    "Basis",
    "BasisState",
    "Manifold",
    "HamiltonianBuilder",
    "AtomicStructure",
    "make_nlj_basis",
    "diagonalize",
    "eigenshuffle",
    "sweep_field",
    "sweep_intensity",
    "SweepResult",
    "MagneticSweepResult",
    "LaserSweepResult",
    "uncoupled_label",
    "coupled_label",
    "both_labels",
    "format_state",
    "field_from_splitting",
    "intensity_from_splitting_shift",
    "scattering_rate",
    "dominant_couplings",
    "transition_frequency_shift",
]
