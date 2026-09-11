"""Deprecated shim: moved to :mod:`kamo.trap.gaussian`."""
import warnings

from kamo.trap.gaussian import GaussianBeam

warnings.warn("kamo.gaussian_beam.gaussian has moved to kamo.trap.gaussian; "
              "import GaussianBeam from kamo or kamo.trap instead.",
              DeprecationWarning, stacklevel=2)

__all__ = ["GaussianBeam"]
