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

    def test_elliptical_beam_uses_the_right_waist_and_the_smaller_zR(self):
        sheet = LightSheet(waist=(8e-6, 120e-6), wavelength_m=LAM, power=1.0,
                           propagation_direction=(0, 0, 1), transverse_axis=(1, 0, 0),
                           polarization=(0, 1, 0))
        assert tp._extent_along(sheet, np.array([1.0, 0, 0])) == pytest.approx(2.5 * 8e-6)
        assert tp._extent_along(sheet, np.array([0, 1.0, 0])) == pytest.approx(2.5 * 120e-6)
        s = tp._extent_along(sheet, np.array([0, 0, 1.0]))
        drop = float(sheet.intensity(0.0, 0.0, s) / sheet.intensity(0.0, 0.0, 0.0))
        assert drop == pytest.approx(1.0 / 7.25, rel=1e-9)          # a round beam's at 2.5 zR
        assert 2.5 * sheet.rayleigh_range_u < s < 0.1 * sheet.rayleigh_range_v

    def test_crossed_beams_frame_the_crossing(self, tweezer):
        """A tweezer along x through a sheet propagating along z, thin along x: the
        window is the crossing, set per axis by the tighter beam."""
        sheet = LightSheet(waist=(8e-6, 120e-6), wavelength_m=LAM, power=40.0,
                           propagation_direction=(0, 0, 1), transverse_axis=(1, 0, 0),
                           polarization=(1, 0, 0))
        both = tweezer.with_power(1.0) + sheet
        x, y, z = (tp._extent_along(both, np.eye(3)[i]) for i in range(3))
        assert x == pytest.approx(2.5 * 8e-6)         # the sheet's thin axis beats the tweezer's zR
        assert y == pytest.approx(2.5 * W0) and z == pytest.approx(2.5 * W0)
        cut = tp.plane_cut(both, normal=(0, 1, 0), n=(101, 101))
        assert cut.values.max() == pytest.approx(float(both.intensity(0.0, 0.0, 0.0)), rel=1e-3)

    def test_spread_origins_are_all_in_the_window(self, tweezer):
        a = tweezer.moved_to((0.0, -50e-6, 0.0))
        b = tweezer.moved_to((0.0, +50e-6, 0.0))
        from kamo.trap.beams import InterferenceWarning
        with pytest.warns(InterferenceWarning):         # same colour, same polarization
            both = a + b
        cut = tp.plane_cut(both, normal=(0, 0, 1), n=(11, 11))
        assert cut.vertical[0] < -50e-6 - 2 * W0 and cut.vertical[-1] > 50e-6 + 2 * W0

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
            assert c["values"][mid] == pytest.approx(c["harmonic"][mid], rel=1e-12)

    @pytest.mark.parametrize("kw, match", [
        (dict(quantity="potential"), "Trap"),
        (dict(quantity="phase"), "quantity"),
        (dict(quantity="density"), "cloud"),
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


# ------------------------------------------------- windows, aspect, densities

@pytest.fixture(scope="module")
def cloud():
    from kamo.trap.thomas_fermi import ThomasFermiSolver
    from kamo.trap.trap import HarmonicTrap
    w = 2 * np.pi * np.array([300.0, 400.0, 500.0])
    ht = HarmonicTrap((0.0, 0.0, 1e-6), 0.0, np.diag(M * w ** 2), M)   # off-origin centroid
    return ThomasFermiSolver(ht, a_scattering=100 * kc.a0, n_per_axis=48).solve(2000.0)


class TestWindows:
    def test_limits_are_lab_coordinates_either_end_defaulting(self, trap):
        z0 = trap.minimum().position[2]
        cut = tp.plane_cut(trap, normal=(0, 1, 0), xlim=(-10e-6, 30e-6), ylim=(-4e-6, None),
                           n=(11, 9))
        assert cut.horizontal[0] == pytest.approx(-10e-6) and cut.horizontal[-1] == pytest.approx(30e-6)
        assert cut.vertical[0] == pytest.approx(-4e-6)
        assert cut.vertical[-1] == pytest.approx(z0 + 2.5 * W0)          # default upper end
        X, Y, Z = cut.lab_points()                                        # and they are lab z
        assert Z[0, 0] == pytest.approx(-4e-6) and X[-1, 0] == pytest.approx(30e-6)

    def test_limits_override_half_width(self, trap):
        cut = tp.plane_cut(trap, normal=(0, 0, 1), xlim=(0.0, 5e-6), half_width=1e-6, n=5)
        assert (cut.horizontal[0], cut.horizontal[-1]) == pytest.approx((0.0, 5e-6))
        assert cut.vertical[-1] - cut.vertical[0] == pytest.approx(2e-6)   # half_width kept on y

    @pytest.mark.parametrize("kw", [dict(xlim=(1e-6, 0.0)), dict(xlim=(0, 1, 2)),
                                    dict(ylim="wide"), dict(xlim=(0.0, np.inf)),
                                    dict(half_width=-1e-6)])
    def test_bad_windows(self, tweezer, kw):
        with pytest.raises(ValueError):
            tp.plane_cut(tweezer, **kw)

    def test_ticks_and_minimum_marker_are_lab_coordinates(self, trap):
        fig, ax = tp.plot_plane_cut(trap, normal=(0, 1, 0), xlim=(-8e-6, 8e-6),
                                    ylim=(-3e-6, 1e-6), n=(41, 41))
        assert ax.get_images()[0].get_extent() == pytest.approx([-8, 8, -3, 1])
        plus = next(l for l in ax.get_lines() if l.get_marker() == "+")
        r = trap.minimum().position * 1e6
        assert (plus.get_xdata()[0], plus.get_ydata()[0]) == pytest.approx((r[0], r[2]))

    def test_line_cut_windows(self, trap):
        cuts = tp.line_cuts(trap, [(1, 0, 0)], xlim=(-5e-6, 20e-6), n=51)
        assert (cuts[0]["s"][0], cuts[0]["s"][-1]) == pytest.approx((-5e-6, 20e-6))
        fig, ax = tp.plot_line_cuts(trap, n=51, ylim=(None, 0.0))
        assert ax.get_ylim()[1] == 0.0 and ax.get_ylim()[0] < 0.0

    def test_beam_profile_window(self, tweezer):
        fig, ax = tp.plot_beam_profile(tweezer, xlim=(0.0, 50e-6), ylim=(2e-6, 10e-6))
        assert ax.get_xlim() == pytest.approx((0.0, 50.0))
        assert ax.get_ylim() == pytest.approx((2.0, 10.0))

    def test_summary_limits_by_lab_axis(self, trap):
        fig, axes = tp.plot_trap_summary(trap, limits={"x": (-20e-6, 20e-6), "z": (-2e-6, 1e-6)})
        xy, xz, yz = (a.get_images()[0].get_extent() for a in axes.flat[:3])
        assert xy[:2] == pytest.approx([-20, 20]) and xz[:2] == pytest.approx([-20, 20])
        assert xz[2:] == pytest.approx([-2, 1]) and yz[2:] == pytest.approx([-2, 1])
        with pytest.raises(ValueError, match="lab axes"):
            tp.plot_trap_summary(trap, limits={"r": (0, 1)})


class TestAspect:
    def test_maps_default_equal(self, trap, cloud):
        assert tp.plot_plane_cut(trap, n=21)[1].get_aspect() == 1.0
        assert tp.plot_column_density(cloud, n=21)[1].get_aspect() == 1.0
        _, axes = tp.plot_trap_summary(trap)
        assert all(a.get_aspect() == 1.0 for a in axes.flat[:3])

    def test_selectable(self, trap, cloud):
        assert tp.plot_plane_cut(trap, n=21, aspect="auto")[1].get_aspect() == "auto"
        assert tp.plot_column_density(cloud, n=21, aspect=2.0)[1].get_aspect() == 2.0


class TestDensity:
    def test_density_cut_is_the_cloud(self, cloud):
        cut = tp.plane_cut(cloud, normal=(1, 0, 0), n=(31, 25))
        assert cut.quantity == "density" and cut.units == "1/m^3"
        assert np.allclose(cut.center, cloud.centroid)
        X, Y, Z = cut.lab_points()
        assert cut.values == pytest.approx(cloud.density(X, Y, Z), rel=1e-15)
        assert cut.horizontal[-1] - cloud.centroid[1] == pytest.approx(4 * cloud.sigma[1])

    def test_column_density_along_x_is_the_cloud_method(self, cloud):
        cut = tp.column_density_map(cloud, "x", n=(41, 37))
        Y, Z = np.meshgrid(cut.horizontal, cut.vertical, indexing="ij")
        assert cut.values == pytest.approx(cloud.column_density(Y, Z), rel=1e-12, abs=1e-6)
        assert cut.values.max() == pytest.approx(cloud.peak_column_density, rel=0.02)

    @pytest.mark.parametrize("axis, h, v", [("x", 1, 2), ("y", 0, 2), ("z", 0, 1), (2, 0, 1)])
    def test_column_density_along_any_axis_holds_every_atom(self, cloud, axis, h, v):
        g = cloud.grid.half_widths
        c = cloud.centroid
        cut = tp.column_density_map(cloud, axis, n=(301, 301),
                                    xlim=(c[h] - g[h], c[h] + g[h]), ylim=(c[v] - g[v], c[v] + g[v]))
        dA = np.diff(cut.horizontal[:2])[0] * np.diff(cut.vertical[:2])[0]
        assert np.sum(cut.values) * dA == pytest.approx(cloud.N, rel=2e-2)
        assert tp._lab_axis_name(cut.e1) == "xyz"[h] and tp._lab_axis_name(cut.e2) == "xyz"[v]

    def test_errors(self, cloud, tweezer):
        with pytest.raises(TypeError, match="TrapCloud"):
            tp.column_density_map(tweezer)
        with pytest.raises(ValueError, match="axis"):
            tp.column_density_map(cloud, "r")
        with pytest.raises(ValueError, match="cloud"):
            tp.plane_cut(tweezer, quantity="density")
        with pytest.raises(ValueError, match="no intensity"):
            tp.plane_cut(cloud, quantity="intensity")

    def test_plots_build_with_a_shared_colour_scale(self, cloud):
        fig, ax = tp.plot_plane_cut(cloud, normal=(0, 1, 0), n=41, vmax=1e14)
        assert ax.get_images()[0].get_clim()[1] == 1e14
        assert "cm$^{-3}$" in fig.axes[-1].get_ylabel()
        fig, ax = tp.plot_column_density(cloud, "x", n=41, vmax=5e10, contours=True)
        assert ax.get_images()[0].get_clim() == (0.0, 5e10)
        assert ax.get_xlabel().startswith("$y$") and ax.get_ylabel().startswith("$z$")


# ------------------------------------------------------------- live plots

from io import BytesIO  # noqa: E402


def _live_map(ax):
    return next(im for im in ax.get_images() if hasattr(im, "n_evaluations"))


def _live_lines(ax):
    return next(a for a in ax.get_children() if type(a).__name__ == "LiveLineCuts")


def _sheet(power=2.0):
    """Propagating along z, thin along x: crosses the x-tweezer."""
    return LightSheet(waist=(8e-6, 120e-6), wavelength_m=LAM, power=power,
                      propagation_direction=(0, 0, 1), transverse_axis=(1, 0, 0),
                      polarization=(1, 0, 0))


class TestLive:
    def test_a_new_view_is_resampled_once_and_exactly(self, tweezer):
        both = tweezer.with_power(1e-3) + _sheet()
        fig, ax = tp.plot_plane_cut(both, normal=(0, 1, 0))
        im = _live_map(ax)
        fig.savefig(BytesIO(), dpi=100)
        k = im.n_evaluations
        ax.set_xlim(-100, 100)
        ax.set_ylim(-40, 40)
        assert im.n_evaluations == k                      # nothing happens until a draw
        fig.savefig(BytesIO(), dpi=100)
        assert im.n_evaluations == k + 1                  # x and y together: one sampling
        assert im.get_extent() == pytest.approx([-100, 100, -40, 40])
        n2, n1 = im.get_array().shape
        ref = tp.plane_cut(both, normal=(0, 1, 0), xlim=(-100e-6, 100e-6),
                           ylim=(-40e-6, 40e-6), n=(n1, n2))
        # the view is in um, so its edges are one ulp off the SI literals: compare to 1e-12
        assert np.asarray(im.get_array()) == pytest.approx(ref.values.T, rel=1e-12)
        assert im.cut.horizontal[0] == pytest.approx(-100e-6)
        fig.savefig(BytesIO(), dpi=100)
        assert im.n_evaluations == k + 1                  # unchanged view: cached

    def test_one_sample_per_pixel_capped_or_fixed(self, tweezer, monkeypatch):
        fig, ax = tp.plot_plane_cut(tweezer, normal=(0, 1, 0), aspect="auto")
        im = _live_map(ax)
        fig.canvas.draw()
        bb = ax.get_window_extent()
        assert im.get_array().shape == (round(bb.height), round(bb.width))
        fig.savefig(BytesIO(), dpi=3 * fig.dpi)            # the output dpi sets the resolution
        assert im.get_array().shape[1] > 2 * round(bb.width)
        monkeypatch.setattr(tp, "MAX_SAMPLES", 10_000)
        fig.canvas.draw()
        assert im.get_array().size <= 10_000
        fig, ax = tp.plot_plane_cut(tweezer, normal=(0, 1, 0), n=(40, 30))
        ax.set_xlim(-5, 5)
        fig.canvas.draw()
        assert _live_map(ax).get_array().shape == (30, 40)

    def test_colour_scale_is_frozen_unless_view(self, tweezer):
        clims = {}
        for mode in ("first", "view"):
            fig, ax = tp.plot_plane_cut(tweezer, normal=(0, 1, 0), autoscale=mode)
            im = _live_map(ax)
            fig.canvas.draw()
            before = im.get_clim()
            ax.set_xlim(40, 60)
            ax.set_ylim(5, 7)                               # a dim corner
            fig.canvas.draw()
            clims[mode] = (before, im.get_clim())
        assert clims["first"][1] == clims["first"][0]
        assert clims["view"][1][1] < 0.5 * clims["view"][0][1]
        fig, ax = tp.plot_plane_cut(tweezer, normal=(0, 1, 0), vmin=0.0, vmax=1.0,
                                    autoscale="view")
        assert _live_map(ax).get_clim() == (0.0, 1.0)
        with pytest.raises(ValueError, match="autoscale"):
            tp.plot_plane_cut(tweezer, autoscale="sometimes")

    def test_contours_follow_the_view(self, trap):
        fig, ax = tp.plot_plane_cut(trap, normal=(0, 1, 0))
        im = _live_map(ax)
        fig.canvas.draw()
        assert len(im.escape_lines.get_segments()) > 0
        ax.set_xlim(-40, 0)
        ax.set_ylim(-6, 2)
        fig.canvas.draw()
        escape = np.concatenate(im.escape_lines.get_segments())
        assert escape[:, 0].min() >= -40 - 1e-9 and escape[:, 0].max() <= 1e-9
        assert len(im.contour_lines.get_segments()) > 0

    def test_intensity_contours_are_fractions_of_the_overall_peak(self, tweezer):
        fig, ax = tp.plot_plane_cut(tweezer, normal=(0, 1, 0))
        im = _live_map(ax)
        peak = float(tweezer.intensity(0.0, 0.0, 0.0))
        assert im._levels == pytest.approx(peak * np.exp([-2.0, -1.0]), rel=1e-12)
        ax.set_xlim(40, 60)                                 # zoom away from the peak
        fig.canvas.draw()
        assert im._levels == pytest.approx(peak * np.exp([-2.0, -1.0]), rel=1e-12)

    def test_cloud_plots_are_sampled_once(self, cloud):
        fig, ax = tp.plot_plane_cut(cloud, normal=(1, 0, 0))
        assert not any(hasattr(im, "n_evaluations") for im in ax.get_images())
        fig, ax = tp.plot_line_cuts(cloud)
        assert not any(type(a).__name__ == "LiveLineCuts" for a in ax.get_children())

    def test_line_cuts_resample_on_zoom(self, trap):
        fig, ax = tp.plot_line_cuts(trap)
        fig.canvas.draw()
        live = _live_lines(ax)
        k = live.n_evaluations
        ax.set_xlim(-200, 150)
        fig.canvas.draw()
        assert live.n_evaluations == k + 1
        for line in ax.get_lines()[:6]:                     # three cuts and their harmonics
            x = line.get_xdata()
            assert x[0] == pytest.approx(-200) and x[-1] == pytest.approx(150)


class TestLineCutsOfAnything:
    def test_beams_along_one_direction(self, tweezer):
        """The reported failure: line cuts of a Crossed, one direction as a flat tuple."""
        both = tweezer.with_power(1e-3) + _sheet()
        cuts = tp.line_cuts(both, (0, 0, 1), n=101)
        assert len(cuts) == 1
        cut = cuts[0]
        assert cut["quantity"] == "intensity" and cut["harmonic"] is None
        assert cut["values"] == pytest.approx(both.intensity(0.0, 0.0, cut["s"]), rel=1e-15)
        fig, ax = tp.plot_line_cuts(both, directions=(0, 0, 1))
        assert "W/m" in ax.get_ylabel() and len(ax.get_lines()) == 1

    def test_default_directions_and_center(self, tweezer, trap, cloud):
        assert [tuple(c["direction"]) for c in tp.line_cuts(tweezer, n=5)] == \
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        assert len(tp.line_cuts(trap, (1, 0, 0), n=5)) == 1
        c = tp.line_cuts(cloud, (0, 0, 1), n=11)[0]
        X = cloud.centroid
        assert c["values"] == pytest.approx(cloud.density(X[0], X[1], X[2] + c["s"]), rel=1e-15)
        assert c["units"] == "1/m^3"

    def test_errors(self, tweezer):
        with pytest.raises(ValueError, match="directions"):
            tp.line_cuts(tweezer, [(1, 0)])
        with pytest.raises(ValueError, match="Trap"):
            tp.line_cuts(tweezer, quantity="potential")
