# Lazy top-level imports.  Eager imports here trigger compute_polarizabilities.py
# which instantiates Potassium39() (-> ARC SQLite) at class-definition time.
# Worker subprocesses importing kamo.constants must not touch ARC.
# Python 3.7+ module __getattr__ defers these until first access.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .atom_properties.k39 import Potassium39
    from .gaussian_beam import GaussianBeam
    from .light_shift import ComputeLightShift, ComputePolarizabilities
    from .BEC_properties import bec
    from .hamiltonian import AtomicStructure
    from .dipole_dipole import (BECCloud, CoupledDipole, CyclingTransition,
                                PairPotential, QuasiStaticLZModel)
    from .BEC_properties.variational import GaussianVariationalCloud
    from .imaging import ProbeBeam, Propagator, TwoLevelResponse, UniformMixture
    from .spin import SpinField, SpinGeometry, Sequence

_lazy = {
    'Potassium39':             '.atom_properties.k39',
    'GaussianBeam':            '.gaussian_beam',
    'ComputeLightShift':       '.light_shift',
    'ComputePolarizabilities': '.light_shift',
    'bec':                     '.BEC_properties',
    'AtomicStructure':         '.hamiltonian',
    'CyclingTransition':       '.dipole_dipole',
    'PairPotential':           '.dipole_dipole',
    'BECCloud':                '.dipole_dipole',
    'QuasiStaticLZModel':      '.dipole_dipole',
    'CoupledDipole':           '.dipole_dipole',
    'GaussianVariationalCloud': '.BEC_properties.variational',
    'ProbeBeam':               '.imaging',
    'Propagator':              '.imaging',
    'TwoLevelResponse':        '.imaging',
    'UniformMixture':          '.imaging',
    'SpinField':               '.spin',
    'SpinGeometry':            '.spin',
    'Sequence':                '.spin',
}

def __getattr__(name):
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache so subsequent accesses skip __getattr__
        return val
    raise AttributeError(f"module 'kamo' has no attribute {name!r}")