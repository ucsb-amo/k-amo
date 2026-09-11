"""Deprecated shim: moved to :mod:`kamo.trap.thin_lens`."""
import warnings

from kamo.trap.gaussian import GaussianBeam
from kamo.trap.thin_lens import Objective, ThinLensGaussian

warnings.warn("kamo.gaussian_beam.thin_lens_gaussian has moved to kamo.trap.thin_lens.",
              DeprecationWarning, stacklevel=2)

__all__ = ["GaussianBeam", "Objective", "ThinLensGaussian"]
