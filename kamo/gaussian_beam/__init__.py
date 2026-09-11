"""Deprecated location: GaussianBeam and the thin-lens helpers moved to kamo.trap.

``from kamo import GaussianBeam`` and ``from kamo.gaussian_beam import
GaussianBeam`` keep working silently and give the same class object as
``kamo.trap.GaussianBeam``.  The legacy submodules
(``kamo.gaussian_beam.gaussian``, ``kamo.gaussian_beam.thin_lens_gaussian``)
still import, with a DeprecationWarning.
"""
from kamo.trap.gaussian import GaussianBeam

__all__ = ["GaussianBeam"]
