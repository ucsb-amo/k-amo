"""kamo.spin -- spatially resolved spin states, and what imaging does to them.

Sits on top of :mod:`kamo.imaging`.  A :class:`SpinField` holds a mean spin vector
per voxel of the propagation grid; operations rotate it, decohere it, and image it,
and a :class:`Sequence` composes them into an experiment.

Quick start
-----------
>>> import numpy as np
>>> from kamo.spin import SpinField, SpinGeometry, ramsey
>>> geom = SpinGeometry.from_propagator(prop, cloud)
>>> field = SpinField.spin_coherent(geom, Sz_total=0.0)   # equal superposition
>>> out = ramsey(prop, t_pulse=5e-6).run(field, probe=probe)
>>> print(out.summary())

The one thing to understand first
---------------------------------
A z-rotation commutes with ``S_z``, so the differential light shift an imaging
pulse applies does NOT change the refractive index -- rotate-then-image reproduces
the same picture.  Converting the imprinted phase into something a probe can see
takes a closing pi/2 pulse; :func:`ramsey` is that sequence.  See
:mod:`kamo.spin.operations` for the full argument and its consequences.

Approximations
--------------
- Mean-field Bloch vectors: no quantum correlations, hence no spin squeezing.
  Shot-to-shot projection noise is available stochastically via
  :meth:`SpinField.sample_shot`.  The extension point for a Gaussian-moments state
  is documented on :meth:`SpinField._apply_rotation` /
  :meth:`SpinField._apply_decoherence` -- every operator is written through those
  two primitives, so a covariance-carrying subclass needs no changes above them.
- One shared spatial density profile; spin weights the polarizability but does not
  reshape the cloud.  Exact for a pulse too short for the density to respond.
- Spontaneous scattering fully decoheres per photon (the two transitions are split
  by far more than a linewidth, so an emitted photon is which-path information),
  but does not repopulate: optical pumping is not modelled.
- RF/microwave rotations are instantaneous and uniform unless given an
  ``inhomogeneity`` map.
- Mean-field interaction shifts are local-density diagonal terms only: no
  spin-changing collisions, no back-action on the density.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .field import SpinField, SpinGeometry, SpinFieldSource
    from .operations import FreeEvolve, ImagePulse, Operation, Rotate
    from .result import ImagingResult, SequenceResult
    from .sequence import Sequence, ramsey

_lazy = {
    "SpinField":       ".field",
    "SpinGeometry":    ".field",
    "SpinFieldSource": ".field",
    "Operation":       ".operations",
    "Rotate":          ".operations",
    "FreeEvolve":      ".operations",
    "ImagePulse":      ".operations",
    "ImagingResult":   ".result",
    "SequenceResult":  ".result",
    "Sequence":        ".sequence",
    "ramsey":          ".sequence",
}

_lazy_modules = ("field", "operations", "result", "sequence", "plotting")


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
    raise AttributeError(f"module 'kamo.spin' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_lazy) | set(_lazy_modules))
