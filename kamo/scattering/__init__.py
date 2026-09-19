"""kamo.scattering — s-wave scattering length a(B) for K39 ground-state collisions.

Compute the complex s-wave scattering length ``a(B) = a_re - i a_im`` (Bohr) for
a colliding pair of K39 4S1/2 atoms vs magnetic field, for any Zeeman channel.

**39K only.**  Unlike the rest of kamo (which works for every alkali in
:mod:`kamo.atom_properties.alkali`), the potentials, measured resonances and
coupled-channels tables here are 39K data.  The entry points take ``atom=`` only
to check it: :class:`ScatteringModel`, :class:`~.thresholds.K39Thresholds`,
:class:`~.coupled_channels.CoupledChannels` and
:func:`~.lookup.scattering_length` raise ``NotImplementedError`` for any other
species, and states outside the 4S1/2 ``|F, mF>`` ladder raise ``ValueError``.

Quick start
-----------
>>> from kamo.scattering import ScatteringModel
>>> m = ScatteringModel(B_max=600.0)             # empirical (fast, 6 channels)
>>> m.intra((1, -1), 33.6)                      # |1,-1>+|1,-1>
>>> m.inter((1, -1), (1, 0), 60.0)              # |1,-1>+|1,0>
>>> cc = ScatteringModel(B_max=600.0, backend="cc")   # coupled channels, any channel
>>> cc.intra((2, 0), 100.0)                     # complex: F=2 pairs spin-relax

Two engines, one literature database
------------------------------------
* :mod:`.data.k39_feshbach` — measured resonance positions / zero crossings /
  dimer energies collected from the literature (Etrych 2023, Chapurin 2019,
  Roy 2013, D'Errico 2007, Tanzi 2018, Tiemann 2020, ...).
* ``backend="empirical"`` — sum-of-poles a(B) from that table.
* ``backend="cc"`` — coupled channels on the Falke/Tiemann K2 potentials with
  inner walls calibrated to the measured positions (:mod:`.calibration`).

Layers: :mod:`.thresholds` (reuses kamo.hamiltonian), :mod:`.channels`
(pair-channel algebra + frame projection), :mod:`.coupled_channels`,
:mod:`.backends`, :mod:`.resonances` (pole / zero finding), :mod:`.loss` (K2),
:mod:`.plotting`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scattering import ScatteringModel
    from .thresholds import K39Thresholds
    from .channels import PairChannel, enumerate_channels
    from .loss import k2_from_scattering_length
    from .coupled_channels import CoupledChannels
    from .calibration import calibrate
    from .resonances import find_features, locate_pole, characterize
    from .lookup import scattering_length
    from .thresholds import require_k39

_lazy = {
    "ScatteringModel":          ".scattering",
    "K39Thresholds":            ".thresholds",
    "PairChannel":              ".channels",
    "enumerate_channels":       ".channels",
    "k2_from_scattering_length": ".loss",
    "CoupledChannels":          ".coupled_channels",
    "calibrate":                ".calibration",
    "find_features":            ".resonances",
    "locate_pole":              ".resonances",
    "characterize":             ".resonances",
    "scattering_length":        ".lookup",
    "require_k39":              ".thresholds",
}


def __getattr__(name):
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'kamo.scattering' has no attribute {name!r}")
