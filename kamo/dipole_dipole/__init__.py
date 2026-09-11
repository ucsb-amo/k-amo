"""Resonant dipole-dipole physics for near-resonant light on a dense K-39 gas.

The package answers one question: how fast does a dense cloud lose atoms when
near-resonant light is applied on a closed sigma transition at high field?

Layers, bottom to top:

* :mod:`~kamo.dipole_dipole.green_tensor` -- free-space dyadic Green's function and
  the symmetric/antisymmetric pair-state shifts and decay rates.
* :mod:`~kamo.dipole_dipole.transition` -- the driven two-level transition,
  parameterised from :class:`kamo.hamiltonian.AtomicStructure` at a given field.
* :mod:`~kamo.dipole_dipole.pair` -- molecular potential curves, Condon surfaces,
  and the classical inward/outward flight after excitation.
* :mod:`~kamo.dipole_dipole.cloud` -- Thomas-Fermi BEC profile, pair-separation
  statistics and relative-velocity scale.
* :mod:`~kamo.dipole_dipole.light_assisted_collisions` -- the two-body loss rate
  coefficient beta(detuning, intensity; trap depth).
* :mod:`~kamo.dipole_dipole.coupled_dipole` -- N-atom microscopic model, used to
  bound the pair approximation and to give the collective line shift.
* :mod:`~kamo.dipole_dipole.dynamics` -- integrate dn/dt = -beta n^2 over the cloud
  and a pulse train to get N(t).
* :mod:`~kamo.dipole_dipole.validation` -- analytic-limit and regression checks.

Quick start
-----------
>>> from kamo.dipole_dipole import CyclingTransition, PairPotential, BECCloud
>>> from kamo.dipole_dipole import QuasiStaticLZModel
>>> t = CyclingTransition.at_field(520.6)
>>> print(t.summary())
>>> pot = PairPotential(t)
>>> cloud = BECCloud(N=1e5, trap_frequencies_Hz=(150., 150., 20.), a_s_bohr=100.)
>>> model = QuasiStaticLZModel(t, pot, cloud)
>>> beta = model.beta(detuning_Hz=-6e6, intensity=1.0, trap_depth_K=50e-6)

For a worked end-to-end run, and for the regression suite:

    python -m kamo.dipole_dipole.example
    python -m kamo.dipole_dipole.validation
"""

# NOTE: the dyadic itself is deliberately not re-exported here -- a function
# named green_tensor at package level would shadow the green_tensor submodule.
from .green_tensor import (MAGIC_ANGLE_DEG, MAGIC_ANGLE_RAD, dd_coupling,
                           dd_coupling_scalar, dipole_projection, green_scalar,
                           near_field_coupling, pair_branches,
                           spherical_unit_vector)
from .transition import CompetingChannel, CyclingTransition
from .pair import PairPotential
from .cloud import BECCloud
from .light_assisted_collisions import LossChannels, QuasiStaticLZModel
from .coupled_dipole import CoupledDipole, lorentz_lorenz_shift_Hz
from .dynamics import integrate_loss

__all__ = [
    "MAGIC_ANGLE_DEG",
    "MAGIC_ANGLE_RAD",
    "dd_coupling",
    "dd_coupling_scalar",
    "dipole_projection",
    "green_scalar",
    "near_field_coupling",
    "pair_branches",
    "spherical_unit_vector",
    "CompetingChannel",
    "CyclingTransition",
    "PairPotential",
    "BECCloud",
    "LossChannels",
    "QuasiStaticLZModel",
    "CoupledDipole",
    "lorentz_lorenz_shift_Hz",
    "integrate_loss",
]
