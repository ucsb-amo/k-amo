"""Tests for kamo.trap.plotting.

INTERNAL only: the plane geometry and default extents, the sampled data
against direct evaluation, argument errors, and that every plot builds and
returns (fig, ax).  Agg backend; explicit mass and polarizability -- no ARC.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from scipy.constants import atomic_mass  # noqa: E402

import kamo.constants as kc  # noqa: E402
from kamo.trap import plotting as tp  # noqa: E402
from kamo.trap.beams import LightSheet, Tweezer  # noqa: E402
from kamo.trap.trap import Trap  # noqa: E402

M = 38.963706 * atomic_mass
W0, LAM = 3.0e-6, 1064e-9
ALPHA = 599.3005266591443 * kc.convert_polarizability_au_to_SI


@pytest.fixture(scope="module")
def tweezer():
    return Tweezer.from_trap_frequency(1e3, polarizability_SI=ALPHA, mass=M,
                                       waist=W0, wavelength_m=LAM)


@pytest.fixture(scope="module")
def trap(tweezer):
    return Trap(tweezer, polarizability_SI=ALPHA, mass=M, gravity=True)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class TestGeometry:
    @pytest.mark.parametrize("normal, e1, e2", [
        ((0, 1, 0), (1, 0, 0), (0, 0, 1)),        # x right, z up
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),        # x right, y up
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),        # y right, z up
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),       # orientation ignores the normal's sign
    ])
    def test_readable_orientation(self, normal, e1, e2):
        a, b, n = tp._plane_axes(normal)
        assert np.allclose(a, e1) and np.allclose(b, e2)
        assert abs(a @ n) < 1e-15 and abs(b @ n) < 1e-15 and abs(a @ b) < 1e-15

    def test_in_plane_axis_sets_the_horizontal(self):
        a, b, _ = tp._plane_axes((0, 0, 1), in_plane_axis=(1, 1, 0))
        assert np.allclose(a, np.array([1, 1, 0]) / np.sqrt(2))
        assert b[1] > 0 or b[2] > 0

    def test_default_extent_interpolates_waist_and_rayleigh_range(self, tweezer):
        cut = tp.plane_cut(tweezer, normal=(0, 1, 0), n=(11, 11))
        assert cut.a1[-1] == pytest.approx(2.5 * tweezer.zR)      # along the beam
        assert cut.a2[-1] == pytest.approx(2.5 * W0)              # across it

    def test_labels(self):
        assert tp._axis_label([0, 0, 1]).startswith("$z$")
        assert tp._axis_label([-1, 0, 0]).startswith("$-x$")
        assert tp._direction_label([1, 1, 0] / np.sqrt(2)) == "(0.71, 0.71, 0.00)"


class TestData:
    def test_plane_cut_is_the_field(self, tweezer):
        cut = tp.plane_cut(tweezer, normal=(0, 1, 1), center=(1e-6, 0, 0), n=(21, 17))
        X, Y, Z = cut.lab_points()
        assert cut.values.shape == (21, 17)
        assert cut.values == pytest.approx(tweezer.intensity(X, Y, Z), rel=1e-15)
        assert np.allclose((X - 1e-6) * cut.normal[0] + Y * cut.normal[1] + Z * cut.normal[2],
                           0.0, atol=1e-20)

    def test_potential_cut_through_the_minimum(self, trap):
        cut = tp.plane_cut(trap, normal=(0, 1, 0), n=(41, 41), units="kHz")
        assert cut.quantity == "potential" and cut.units == "kHz"
        assert np.allclose(cut.center, trap.minimum().position)
        assert cut.values.min() == pytest.approx(trap.minimum().potential_J / kc.h / 1e3,
                                                 rel=1e-3)

    def test_line_cuts_meet_the_harmonic_expansion_at_the_minimum(self, trap):
        for c in tp.line_cuts(trap, n=201):
            mid = len(c["s"]) // 2
            assert c["U"][mid] == pytest.approx(c["U_harmonic"][mid], rel=1e-12)

    @pytest.mark.parametrize("kw, match", [
        (dict(quantity="potential"), "Trap"),
        (dict(quantity="density"), "quantity"),
        (dict(units="mK"), "units"),
        (dict(center=(0, 0)), "center"),
    ])
    def test_errors(self, tweezer, kw, match):
        with pytest.raises(ValueError, match=match):
            tp.plane_cut(tweezer, **kw)


class TestPlots:
    def test_plane_cut_plot_of_a_beam(self, tweezer):
        fig, ax = tp.plot_plane_cut(tweezer, normal=(0, 1, 0), n=(61, 61))
        assert ax.get_images() and ax.get_xlabel().startswith("$x$")
        assert len(ax.collections) >= 1                         # 1/e^2 and 1/e contours

    def test_plane_cut_plot_of_a_trap_marks_escape_and_minimum(self, trap):
        fig, ax = tp.plot_plane_cut(trap, normal=(0, 1, 0), n=(81, 81), half_width=(35e-6, 6e-6))
        assert len(ax.collections) >= 2                         # grey levels + escape contour
        assert any(line.get_marker() == "+" for line in ax.get_lines())
        assert any(t.get_text() == "g" for t in ax.texts)

    def test_off_plane_minimum_is_hollow(self, trap):
        fig, ax = tp.plot_plane_cut(trap, normal=(0, 0, 1), center=(0, 0, 1e-6), n=(41, 41))
        assert any(line.get_marker() == "o" for line in ax.get_lines())

    def test_line_cuts_plot(self, trap):
        fig, ax = tp.plot_line_cuts(trap, n=101)
        assert len(ax.get_lines()) == 3 + 3 + 1                  # cuts, harmonic, escape

    def test_beam_profile_and_summary(self, trap):
        s = LightSheet(waist=(8e-6, 120e-6), wavelength_m=LAM, power=0.2)
        fig, ax = tp.plot_beam_profile(s)
        assert len(ax.get_lines()) >= 2
        fig, axes = tp.plot_trap_summary(trap)
        assert axes.shape == (2, 2)
