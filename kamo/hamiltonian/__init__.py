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
from .diagonalize import (SweepResult, diagonalize, eigenshuffle, sweep_field,
                          sweep_intensity)
from .model import AtomicStructure
from .state_labels import uncoupled_label, coupled_label, both_labels, format_state

__all__ = [
    "Basis",
    "BasisState",
    "Manifold",
    "HamiltonianBuilder",
    "AtomicStructure",
    "diagonalize",
    "eigenshuffle",
    "sweep_field",
    "sweep_intensity",
    "SweepResult",
    "uncoupled_label",
    "coupled_label",
    "both_labels",
    "format_state",
]
