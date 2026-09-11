"""Tests for kamo.trap.trap.

INTERNAL      the single-beam curvatures of GaussianBeam.trap_frequency; the
              analytic gradient; gravity -- the sag, closed-form frequencies at
              the sagged point of a Gaussian beam; the vertical escape lip
              against kamo.trap.dvr and the gaussian_well notebook, and the
              lower 3D escape saddle displaced along the beam axis; the
              principal axes of a crossed trap; the harmonic expansion; argument
              handling.  Explicit mass and polarizability: no ARC, no network.
GROUND-TRUTH  the 1 kHz operating point with the UDel-portal polarizability,
              read offline from the snapshot bundled with kamo.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, g as G, h, hbar

import kamo.constants as kc
from kamo.trap import dvr
from kamo.trap import polarizability as pz
from kamo.trap.beams import Crossed, Tweezer
from kamo.trap.gaussian import GaussianBeam
from kamo.trap.trap import Trap
from kamo.trap.tests.test_polarizability import offline  # noqa: F401  (fixture)

M = 38.963706 * atomic_mass
W0, LAM = 3.0e-6, 1064e-9
ALPHA = 599.3005266591443 * kc.convert_polarizability_au_to_SI
F_R = 1.0e3


def _tweezer(**kw):
    return Tweezer.from_trap_frequency(F_R, polarizability_SI=ALPHA, mass=M,
                                       waist=W0, wavelength_m=LAM, **kw)


def _trap(beam=None, **kw):
    kw.setdefault("gravity", False)
    return Trap(_tweezer() if beam is None else beam, polarizability_SI=ALPHA, mass=M, **kw)


def _vertical_cut(trap):
    """The trap's vertical line through the focus, solved by kamo.trap.dvr."""
    U0 = -kc.ac_stark_shift_J(ALPHA, trap.field.I0)
    return dvr.solve_axis(lambda u: -U0 * np.exp(-2 * u ** 2 / W0 ** 2) + M * G * u,
                          length_scale=W0, mass=M)


@pytest.fixture(scope="module")
def flat():
    return _trap()


@pytest.fixture(scope="module")
def sagged():
    return _trap(gravity=True)


# --------------------------------------------------------------- INTERNAL

class TestSingleBeam:
    def test_frequencies_match_gaussian_beam_formulas(self, flat):
        """The curvature-factor-4 radial / factor-2 axial relations of
        GaussianBeam.trap_frequency, reproduced by the numerical Hessian."""
        t = flat.field
        U0 = -kc.ac_stark_shift_J(ALPHA, t.I0)
        w_r = np.sqrt(4 * U0 / (M * W0 ** 2))
        w_z = np.sqrt(2 * U0 / (M * t.zR ** 2))
        f = flat.trap_frequencies()
        assert f.frequencies_Hz == pytest.approx(np.array([w_r, w_r, w_z]) / (2 * np.pi), rel=1e-8)
        assert f.frequencies_Hz[0] == pytest.approx(f.frequencies_Hz[1], rel=1e-10)
        assert f.frequencies_Hz[0] == pytest.approx(F_R, rel=1e-10)
        assert abs(f.axes[2] @ [1, 0, 0]) == pytest.approx(1.0, abs=1e-10)  # weak axis = beam

    def test_potential_at_focus_is_the_gaussian_beam_depth(self, flat):
        gb = GaussianBeam(waist=W0, wavelength=LAM, power=flat.field.power)
        assert flat.potential_K(0.0, 0.0, 0.0) == pytest.approx(
            gb.trap_depth(polarizability=ALPHA), rel=1e-14)
        assert flat.potential_J(0.0, 0.0, 0.0) < 0

    def test_crossed_of_one_beam_is_the_beam(self, flat):
        one = _trap(Crossed([flat.field]))
        assert np.array_equal(one.trap_frequencies().frequencies_Hz,
                              flat.trap_frequencies().frequencies_Hz)

    def test_minimum_without_gravity_is_the_focus(self, flat):
        m = flat.minimum()
        assert m.converged and np.all(m.position == 0.0)
        assert np.all(flat.sag == 0.0)

    def test_gradient_is_analytic(self, sagged):
        rng = np.random.default_rng(3)
        step = 1e-10
        for p in rng.normal(size=(20, 3)) * np.array([10e-6, 2e-6, 2e-6]):
            g = np.array(sagged.gradient(*p))
            fd = np.array([(sagged.potential_J(*(p + e)) - sagged.potential_J(*(p - e))) / (2 * step)
                           for e in np.eye(3) * step])
            assert g == pytest.approx(fd, rel=1e-5, abs=1e-9 * np.max(np.abs(fd)))


class TestGravity:
    def test_sag_matches_the_notebook_and_dvr(self, sagged):
        r = sagged.minimum().position
        assert r[0] == pytest.approx(0.0, abs=1e-15) and r[1] == pytest.approx(0.0, abs=1e-15)
        assert r[2] / W0 == pytest.approx(-0.0840, abs=5e-5)          # notebook section 7
        assert r[2] == pytest.approx(_vertical_cut(sagged).u_min, rel=1e-7)
        assert sagged.sag_along_gravity == pytest.approx(-r[2], rel=1e-12)

    def test_frequencies_at_the_sagged_point_are_closed_form(self, sagged):
        """At depth z below the focus of a Gaussian beam (u = z / w0):
        f_horizontal = f_r e^{-u^2},  f_vertical = f_r e^{-u^2} sqrt(1 - 4u^2),
        f_axial = f_ax e^{-u^2} sqrt(1 - 2u^2)."""
        u = sagged.minimum().position[2] / W0
        f_ax0 = F_R * W0 / (np.sqrt(2) * sagged.field.zR)
        expect = np.exp(-u ** 2) * np.array([F_R, F_R * np.sqrt(1 - 4 * u ** 2),
                                             f_ax0 * np.sqrt(1 - 2 * u ** 2)])
        f = sagged.trap_frequencies()
        assert f.frequencies_Hz == pytest.approx(expect, rel=1e-7)
        assert f.frequencies_Hz[1] == pytest.approx(978.9, abs=0.05)  # notebook
        assert np.allclose(np.abs(f.axes), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), atol=1e-10)

    def test_vertical_escape_matches_dvr_and_the_notebook(self, sagged):
        """Straight down, the barrier is the 1D lip of the notebook's vertical cut."""
        down = sagged.escape_barrier_J([0, 0, -1])
        assert down / h == pytest.approx(4873.86, abs=0.05)           # notebook: 4.87 kHz
        assert down == pytest.approx(_vertical_cut(sagged).usable_depth, rel=1e-6)

    def test_3d_escape_is_through_saddles_along_the_beam_axis(self, sagged):
        """A horizontal tweezer loses atoms off-axis.  About one Rayleigh range
        along the beam the focus is weaker and wider, so the gravity lip there is
        lower; the point straight below the focus is a second-order stationary
        point, and the 1D vertical cut overestimates the depth by 11%."""
        depth = sagged.trap_depth_J()
        s, r0 = sagged.escape_saddle, sagged.minimum().position
        assert depth / h == pytest.approx(4330.69, abs=0.05)
        assert s is not None
        assert abs(s[0]) * 1e6 == pytest.approx(28.435, abs=0.01)
        assert s[2] * 1e6 == pytest.approx(-3.107, abs=0.005)
        assert abs(s[1]) < 1e-12
        assert int(np.sum(np.linalg.eigvalsh(sagged.hessian(s)) < 0)) == 1   # first order
        assert np.linalg.norm(sagged.gradient(*s)) < 1e-8 * abs(sagged.minimum().potential_J) / W0
        assert depth == pytest.approx(sagged.potential_J(*s) - sagged.minimum().potential_J,
                                      rel=1e-12)
        assert depth < sagged.escape_barrier_J([0, 0, -1])
        assert sagged.escape_barrier_J(s - r0) >= depth * (1 - 1e-12)  # a ray only overestimates
        below = np.array([0.0, 0.0, -3.4388e-6])                       # straight below the focus
        assert int(np.sum(np.linalg.eigvalsh(sagged.hessian(below)) < 0)) == 2

    def test_depth_without_gravity_is_the_full_well(self, flat, sagged):
        assert flat.trap_depth_J() == pytest.approx(-flat.minimum().potential_J, rel=1e-5)
        assert flat.escape_saddle is None
        assert sagged.trap_depth_J() < flat.trap_depth_J()
        assert flat.is_bound() and sagged.is_bound()

    def test_sag_is_g_over_omega_squared_when_stiff(self):
        stiff = _trap(_tweezer().scaled(100.0), gravity=True)
        w_vert = 2 * np.pi * stiff.trap_frequencies().frequencies_Hz[1]
        assert stiff.sag_along_gravity == pytest.approx(G / w_vert ** 2, rel=1e-3)

    def test_too_weak_to_hold_against_gravity(self):
        """20% of the power: gamma exceeds gamma_c (the notebook's 0.27 P)."""
        weak = _trap(_tweezer().scaled(0.2), gravity=True)
        with pytest.warns(UserWarning, match="no minimum"):
            f = weak.trap_frequencies()
        assert np.all(np.isnan(f.frequencies_Hz))
        assert not weak.is_bound() and weak.trap_depth_J() == 0.0


class TestCrossed:
    def test_principal_axes_follow_the_beams(self):
        a_hat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        b_hat = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        a = Tweezer(waist=4e-6, wavelength_m=1064e-9, power=20e-3,
                    propagation_direction=a_hat, polarization=(0, 0, 1))
        b = Tweezer(waist=6e-6, wavelength_m=1070e-9, power=60e-3,
                    propagation_direction=b_hat, polarization=(0, 0, 1))
        tr = Trap(a + b, polarizability_SI=ALPHA, mass=M, gravity=False)
        f = tr.trap_frequencies()
        for axis in (a_hat, b_hat, np.array([0.0, 0.0, 1.0])):
            assert np.max(np.abs(f.axes @ axis)) == pytest.approx(1.0, abs=1e-8)
        Ua = -kc.ac_stark_shift_J(ALPHA, a.I0)
        Ub = -kc.ac_stark_shift_J(ALPHA, b.I0)
        curv = np.array([2 * Ua / a.zR ** 2 + 4 * Ub / b.waist ** 2,   # along a: a axial + b radial
                         4 * Ua / a.waist ** 2 + 2 * Ub / b.zR ** 2,   # along b
                         4 * Ua / a.waist ** 2 + 4 * Ub / b.waist ** 2])
        expect = np.sort(np.sqrt(curv / M) / (2 * np.pi))[::-1]
        assert f.frequencies_Hz == pytest.approx(expect, rel=1e-7)


class TestHarmonic:
    def test_harmonic_expansion(self, sagged):
        hm = sagged.harmonic()
        m = sagged.minimum()
        for d, lo, hi in [(0.01 * W0, 0.0, 1e-3), (0.5 * W0, 1e-2, np.inf)]:
            p = m.position + np.array([0.0, d, 0.0])
            full, quad = sagged.potential_J(*p), hm.potential_J(*p)
            rel = abs(quad - full) / abs(full - m.potential_J)
            assert lo <= rel < hi
        g = np.array(hm.gradient(*(m.position + [1e-8, -2e-8, 3e-8])))
        assert g == pytest.approx(hm.hessian_matrix @ [1e-8, -2e-8, 3e-8], rel=1e-12)
        assert hm.trap_frequencies().frequencies_Hz == pytest.approx(
            sagged.trap_frequencies().frequencies_Hz, rel=1e-12)
        assert hm.minimum().potential_J == m.potential_J and hm.trap_depth_J() == np.inf


class TestConstruction:
    def test_polarizability_per_beam(self):
        a = _tweezer()
        b = _tweezer(propagation_direction=(0, 1, 0), polarization=(1, 0, 0))
        tr = Trap(a + b, polarizability_SI=[ALPHA, 2 * ALPHA], mass=M, gravity=False)
        assert tr.alphas() == (ALPHA, 2 * ALPHA)
        with pytest.raises(ValueError, match="one per beam"):
            Trap(a + b, polarizability_SI=[ALPHA] * 3, mass=M)
        assert isinstance(Trap([a, b], polarizability_SI=ALPHA, mass=M).field, Crossed)

    def test_field_argument_forms(self):
        t = _tweezer()
        tr = Trap(t, B_gauss=[0, 3.0, 4.0], polarizability_SI=ALPHA, mass=M)
        assert tr.B_gauss == pytest.approx(5.0) and np.allclose(tr.B_hat, [0, 0.6, 0.8])
        tr2 = Trap(t, B_gauss=-2.0, B_direction=(1, 0, 0), polarizability_SI=ALPHA, mass=M)
        assert tr2.B_gauss == 2.0 and np.allclose(tr2.B_hat, [-1, 0, 0])
        with pytest.raises(ValueError, match="not both"):
            Trap(t, B_gauss=[0, 0, 1.0], B_direction=(1, 0, 0), polarizability_SI=ALPHA, mass=M)

    def test_construction_never_builds_an_atom(self):
        pz.clear_caches()
        tr = Trap(_tweezer(), B_gauss=520.0, mass=M, gravity=False)
        tr.for_state((4, 0, 0.5, 1, 1)).without_gravity().with_field(10.0)
        assert pz._CP == {}

    def test_suggested_axes(self, sagged):
        x, y, z = sagged.suggested_axes(n=(33, 17, 17), n_widths=4.0)
        r0 = sagged.minimum().position
        assert (x.size, y.size, z.size) == (33, 17, 17)
        assert (x[16], y[8], z[8]) == pytest.approx(tuple(r0), abs=1e-15)
        sig = np.sqrt(hbar / (2 * M * sagged.trap_frequencies().omega))   # axes y, z, x
        assert x[-1] - x[16] == pytest.approx(4 * sig[2], rel=1e-6)
        assert y[-1] - y[8] == pytest.approx(4 * sig[0], rel=1e-6)
        assert z[-1] - z[8] == pytest.approx(4 * sig[1], rel=1e-6)

    def test_rescaled_to_frequency(self, flat):
        tr = flat.rescaled_to_frequency(2.0e3)
        assert tr.trap_frequencies().frequencies_Hz[0] == pytest.approx(2.0e3, rel=1e-8)
        assert tr.field.power == pytest.approx(4 * flat.field.power, rel=1e-8)

    def test_summary(self, sagged):
        s = sagged.summary()
        assert "sag" in s and "escape depth" in s and "escape saddle" in s


# ----------------------------------------------------------- GROUND-TRUTH

class TestPortalOperatingPoint:
    def test_portal_polarizability_gives_one_kilohertz(self, offline):
        """Default polarization (0, 1, i) with B along z: beta = 0, so alpha is
        the scalar 599.30 a.u. and the pinned power gives 1 kHz."""
        tr = Trap(_tweezer(), B_gauss=520.594, mass=M, gravity=False)
        assert tr.alphas()[0] / kc.convert_polarizability_au_to_SI == pytest.approx(
            599.3005, rel=1e-6)
        assert tr.trap_frequencies().frequencies_Hz[0] == pytest.approx(F_R, rel=1e-6)
        assert "snapshot" in tr.summary()

    def test_vector_shift_with_the_field_along_the_beam(self, offline):
        t = _tweezer()                                        # (0, 1, i): sigma+ about x
        along_x = Trap(t, B_gauss=520.594, B_direction=(1, 0, 0), mass=M, gravity=False)
        assert along_x.alphas()[0] != along_x.for_state((4, 0, 0.5, 1, 1)).alphas()[0]
        along_z = Trap(t, B_gauss=520.594, B_direction=(0, 0, 1), mass=M, gravity=False)
        assert along_z.alphas()[0] == along_z.for_state((4, 0, 0.5, 1, 1)).alphas()[0]
