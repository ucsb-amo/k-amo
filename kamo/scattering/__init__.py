"""kamo.scattering — s-wave scattering length a(B) for K39 ground-state collisions.

Compute the complex s-wave scattering length ``a(B) = a_re - i a_im`` (Bohr) for
a colliding pair of K39 4S1/2 atoms vs magnetic field, for any Zeeman channel.

Quick start
-----------
>>> from kamo.scattering import ScatteringModel
>>> m = ScatteringModel(B_max=600.0)
>>> m.intra((1, -1), 33.6)              # |1,-1>+|1,-1>
>>> m.inter((1, -1), (1, 0), 60.0)      # |1,-1>+|1,0>
>>> m.is_lossy((1, 0), (1, 0), 60.0)    # open inelastic channel?

.. warning::
   K39 resonance / singlet-triplet parameters are PROVISIONAL and not yet
   literature-verified — see :mod:`kamo.scattering.data.k39_params`.

Layers: :mod:`.thresholds` (reuses kamo.hamiltonian), :mod:`.channels`
(pair-channel algebra + frame projection), :mod:`.backends` (engines),
:mod:`.loss` (K2), :mod:`.plotting`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scattering import ScatteringModel
    from .thresholds import K39Thresholds
    from .channels import PairChannel, enumerate_channels
    from .loss import k2_from_scattering_length
    from .coupled_channels import CoupledChannels

_lazy = {
    "ScatteringModel":          ".scattering",
    "K39Thresholds":            ".thresholds",
    "PairChannel":              ".channels",
    "enumerate_channels":       ".channels",
    "k2_from_scattering_length": ".loss",
    "CoupledChannels":          ".coupled_channels",
}


def __getattr__(name):
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'kamo.scattering' has no attribute {name!r}")
