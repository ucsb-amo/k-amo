"""kamo.hamiltonian — multi-manifold basis construction & diagonalization.

Species-agnostic: every entry point takes ``atom=`` (any alkali from
:mod:`kamo.atom_properties.alkali`, e.g. ``atom("Rb87")``) and reads the nuclear
spin, g-factors, hyperfine constants and levels from it.  The default is kamo's
default atom, 39K, so existing K39 code is unchanged.

Build a basis from any number of (n, l, j) fine-structure manifolds, assemble
Hamiltonian terms (fine + hyperfine, paramagnetic Zeeman, optional diamagnetic,
and laser fields from a GaussianBeam via RWA or AC-Stark), and solve by direct
diagonalization.  Field and intensity sweeps use eigenshuffle to track states
through avoided crossings.

Quick start
-----------
A worked tour (manifolds, magnetic and laser sweeps, the plot arguments,
state labels and the adiabatic connection, other atoms) is the notebook
``kamo/hamiltonian/examples/structure_demo.ipynb``.

>>> from kamo.hamiltonian import AtomicStructure
>>> model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)])   # 39K
>>> res = model.magnetic_sweep(B_max=600.0)     # 0.1 G steps by default

Another alkali is the same call with its own manifolds and ``atom=``::

>>> from kamo import atom
>>> rb = atom("Rb87")
>>> model = AtomicStructure([(5, 0, 0.5), (5, 1, 0.5), (5, 1, 1.5)], atom=rb)

Light shifts: three laser models, ``"rwa"`` (non-perturbative, rotating frame),
``"perturbative"`` (second-order sum over the exact field eigenstates, both
rotating terms; ``perturbative.py``) and ``"stark"`` (fine-structure
polarizabilities).  ``"auto"`` is the default everywhere: ``choose_laser_model``
picks between the first two for a transition (any kamo atom), ``choose_sweep_model``
for a whole-basis sweep (``sweep_intensity``, ``laser_sweep``), and
``light_shift_basis`` builds the basis from the channels that carry the
polarizability at the laser wavelength (see ``laser_model.py``).
"""

from .basis import Basis, BasisState, Manifold
from .builder import HamiltonianBuilder
from .diagonalize import (MagneticSweepResult, LaserSweepResult,
                          SweepResult, diagonalize, eigenshuffle, sweep_field,
                          sweep_intensity)
from .model import AtomicStructure, make_nlj_basis
from .laser_model import (Channel, StateChannels, BasisSelection, LaserModelChoice,
                          state_channels,
                          channel_weights, light_shift_basis,
                          choose_laser_model, photon_indices,
                          substructure_spread_Hz)
from .perturbative import (perturbative_stark_operator, sweep_intensity_perturbative,
                           choose_sweep_model, SweepModelChoice)
from .state_labels import (state_label, uncoupled_label, coupled_label,
                           both_labels, format_state, rs_state_label,
                           is_coupled, clear_manifold_cache,
                           StateLabelMixin)
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
    "Channel",
    "StateChannels",
    "state_channels",
    "BasisSelection",
    "LaserModelChoice",
    "channel_weights",
    "light_shift_basis",
    "choose_laser_model",
    "photon_indices",
    "substructure_spread_Hz",
    "perturbative_stark_operator",
    "sweep_intensity_perturbative",
    "choose_sweep_model",
    "SweepModelChoice",
    "diagonalize",
    "eigenshuffle",
    "sweep_field",
    "sweep_intensity",
    "SweepResult",
    "MagneticSweepResult",
    "LaserSweepResult",
    "state_label",
    "is_coupled",
    "clear_manifold_cache",
    "uncoupled_label",
    "coupled_label",
    "both_labels",
    "format_state",
    "rs_state_label",
    "StateLabelMixin",
    "field_from_splitting",
    "intensity_from_splitting_shift",
    "scattering_rate",
    "dominant_couplings",
    "transition_frequency_shift",
]
