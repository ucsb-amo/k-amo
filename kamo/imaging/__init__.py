"""kamo.imaging -- propagating a probe through a cold cloud, and reading it out.

Forward model for dispersive (phase-contrast) and absorption imaging of a
BEC, in the regime where the cloud is a strong lens rather than a weak scatterer:
the probe is propagated slice by slice through the three-dimensional density with
the full complex polarizability, and the resulting field is refocused, apertured
and imaged the way a real microscope would.

Quick start
-----------
>>> import kamo.constants as kc
>>> from kamo import Potassium39
>>> from kamo.BEC_properties.variational import GaussianVariationalCloud
>>> from kamo.imaging import ProbeBeam, Propagator, UniformMixture, readout
>>>
>>> atom = Potassium39()
>>> B = 520.58
>>> cloud = GaussianVariationalCloud.from_tweezer(
...     N=500., a_scattering=atom.get_scattering_length(1, -1, B) * kc.a0,
...     f_radial_Hz=1.0e3, waist=3.0e-6)
>>> probe = ProbeBeam.from_midpoint(
...     atom, B_gauss=B,
...     ground_up=(4, 0, 1/2, -1/2, -1/2), excited_up=(4, 1, 3/2, -3/2, -1/2),
...     ground_dn=(4, 0, 1/2, -1/2, +1/2), excited_dn=(4, 1, 3/2, -3/2, +1/2),
...     s0_incident=0.335)
>>> prop = Propagator.for_cloud(probe.response, cloud)
>>> res = prop.propagate(UniformMixture(cloud, probe.response, probe.species(xi=1.)),
...                      s0_incident=probe.s0_incident)
>>> psi = readout.refocus(res)
>>> readout.recovered_phase(psi, res.grid), readout.optical_depth(psi)

Layers
------
:mod:`.response`  two-level complex polarizability, susceptibility, cross section,
                  light shift -- one closed form behind every observable.
:mod:`.probe`     the laser and the two ground-state transitions it drives.
:mod:`.grid`      transverse sampling and angular spectrum.
:mod:`.bpm`       split-step propagation; susceptibility sources.
:mod:`.readout`   refocus, far field, phase contrast, signal inversion.
:mod:`.farfield`  the incoherent channel: reabsorption and Born form factors.

Geometry convention: x is the probe propagation axis, y and z transverse, z the
quantization axis.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .response import TwoLevelResponse
    from .probe import ProbeBeam
    from .grid import TransverseGrid
    from .bpm import (Propagator, PropagationResult, SusceptibilitySource,
                      UniformMixture)
    from .farfield import Sky

_lazy = {
    "TwoLevelResponse":      ".response",
    "ProbeBeam":             ".probe",
    "TransverseGrid":        ".grid",
    "Propagator":            ".bpm",
    "PropagationResult":     ".bpm",
    "SusceptibilitySource":  ".bpm",
    "UniformMixture":        ".bpm",
    "Sky":                   ".farfield",
}

_lazy_modules = ("response", "probe", "grid", "bpm", "readout", "farfield", "plotting")


def __getattr__(name):
    import importlib
    if name in _lazy:
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name in _lazy_modules:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'kamo.imaging' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_lazy) | set(_lazy_modules))
