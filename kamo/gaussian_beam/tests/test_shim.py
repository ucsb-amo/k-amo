"""The kamo.gaussian_beam shim (GaussianBeam moved to kamo.trap).

INTERNAL: every legacy import spelling still works and yields the same class
object; only the explicit legacy submodules warn.

Run: pytest kamo/gaussian_beam/tests -q
"""
import importlib
import sys
import warnings

import pytest


def test_same_class_object_everywhere():
    import kamo
    import kamo.gaussian_beam
    from kamo.trap.gaussian import GaussianBeam
    assert kamo.gaussian_beam.GaussianBeam is GaussianBeam
    assert kamo.GaussianBeam is GaussianBeam


def test_top_level_and_package_imports_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from kamo import GaussianBeam  # noqa: F401
        from kamo.gaussian_beam import GaussianBeam  # noqa: F401,F811


@pytest.mark.parametrize("legacy, names", [
    ("kamo.gaussian_beam.gaussian", ("GaussianBeam",)),
    ("kamo.gaussian_beam.thin_lens_gaussian", ("ThinLensGaussian", "Objective")),
])
def test_legacy_submodules_warn_and_reexport(legacy, names):
    sys.modules.pop(legacy, None)
    with pytest.warns(DeprecationWarning, match="kamo.trap"):
        mod = importlib.import_module(legacy)
    import kamo.trap.gaussian, kamo.trap.thin_lens
    for name in names:
        home = kamo.trap.gaussian if name == "GaussianBeam" else kamo.trap.thin_lens
        assert getattr(mod, name) is getattr(home, name)
