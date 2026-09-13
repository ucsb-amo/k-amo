"""Tests for kamo.trap.beams.

INTERNAL only: specification groups, geometry conversions, the lab-frame
intensity against the legacy GaussianBeam, rotation invariance, Crossed
composition and array genericity.  No ARC, no network.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from scipy.constants import atomic_mass

import kamo.constants as kc
from kamo.trap import frames as fr
from kamo.trap.beams import (Beam, Crossed, InterferenceWarning, LightSheet,
                             Tweezer, DEFAULT_POLARIZATION)
from kamo.trap.gaussian import GaussianBeam

W0, LAM, P = 3.0e-6, 1064e-9, 1e-3
M = 38.963706 * atomic_mass
ALPHA_1064_SI = 599.3005266591443 * kc.convert_polarizability_au_to_SI
RNG = np.random.default_rng(7)


def _tw(**kw):
    base = dict(waist=W0, wavelength_m=LAM, power=P)
    base.update(kw)
    return Tweezer(**base)


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ------------------------------------------------------------ specification

class TestSpecification:
    def test_defaults(self):
        t = _tw()
        assert np.allclose(t.propagation_direction, [1, 0, 0])
        assert np.allclose(t.polarization, np.array([0, 1, 1j]) / np.sqrt(2))
        assert np.allclose(t.origin, 0)
        assert Tweezer(waist=W0, wavelength_m=LAM).power == 0.0

    @pytest.mark.parametrize("kw, match", [
        (dict(waist=W0), "colour"),
        (dict(waist=W0, wavelength_m=LAM, frequency_Hz=2.8e14), "colour"),
        (dict(wavelength_m=LAM), "size"),
        (dict(waist=W0, NA=0.2, wavelength_m=LAM), "size"),
        (dict(waist=W0, wavelength_m=LAM, power=1e-3, peak_intensity=1e8), "strength"),
        (dict(waist=-W0, wavelength_m=LAM), "positive"),
        (dict(waist=W0, wavelength_m=LAM, power=-1.0), "power"),
        (dict(NA=1.2, wavelength_m=LAM), "n_medium"),
        (dict(waist=W0, wavelength_m=50e-9), "outside"),
        (dict(waist=(W0, 2 * W0), wavelength_m=LAM), "radially symmetric"),
    ])
    def test_invalid_combinations(self, kw, match):
        with pytest.raises(ValueError, match=match):
            Tweezer(**kw)

    def test_error_names_the_offending_keywords(self):
        with pytest.raises(ValueError) as err:
            Tweezer(waist=W0, NA=0.2, wavelength_m=LAM)
        assert "NA" in str(err.value) and "waist" in str(err.value)

    def test_tweezer_has_no_transverse_axis(self):
        with pytest.raises(TypeError):
            Tweezer(waist=W0, wavelength_m=LAM, transverse_axis=(0, 0, 1))

    def test_frequency_is_equivalent_to_wavelength(self):
        a = _tw()
        b = Tweezer(waist=W0, frequency_Hz=kc.c / LAM, power=P)
        assert b.wavelength_m == pytest.approx(a.wavelength_m, rel=1e-15)

    def test_default_polarization_needs_the_default_direction(self):
        with pytest.raises(ValueError, match="default polarization"):
            _tw(propagation_direction=(0, 1, 0))
        t = _tw(propagation_direction=(0, 1, 0), polarization=(1, 0, 0))
        assert np.allclose(t.polarization, [1, 0, 0])

    def test_transversality_modes_pass_through(self):
        with pytest.raises(ValueError, match="not transverse"):
            _tw(polarization=(1, 1, 0))
        t = _tw(polarization=(1, 1, 0), transversality="project")
        assert abs(np.dot(t.propagation_direction, t.polarization)) < 1e-15

    def test_immutable(self):
        t = _tw()
        with pytest.raises(AttributeError):
            t.power = 2.0
        with pytest.raises(ValueError):
            t._k[0] = 0.0                     # the stored arrays are read-only
        pol = t.polarization
        pol[0] = 5.0
        assert t.polarization[0] == 0.0       # getters hand out copies


class TestGeometry:
    def test_matches_gaussian_beam(self):
        t, gb = _tw(), GaussianBeam(waist=W0, wavelength=LAM, power=P)
        assert t.rayleigh_range == pytest.approx(gb.rayleigh_range, rel=1e-15)
        assert t.peak_intensity == pytest.approx(gb.I0, rel=1e-14)
        assert t.I0 == t.peak_intensity and t.wavelength == LAM
        assert t.divergence_angle_u == pytest.approx(gb.divergence_angle, rel=1e-15)

    @pytest.mark.parametrize("key", ["NA", "rayleigh_range", "divergence_angle"])
    def test_size_round_trips(self, key):
        t = _tw()
        value = {"NA": t.NA, "rayleigh_range": t.rayleigh_range,
                 "divergence_angle": t.divergence_angle_u}[key]
        back = Tweezer(wavelength_m=LAM, power=P, **{key: value})
        assert back.waist == pytest.approx(W0, rel=1e-14)

    def test_power_and_peak_intensity_round_trip(self):
        t = _tw()
        u = Tweezer(waist=W0, wavelength_m=LAM, peak_intensity=t.peak_intensity)
        assert u.power == pytest.approx(P, rel=1e-14)
        assert t.with_peak_intensity(2 * t.I0).power == pytest.approx(2 * P, rel=1e-14)

    def test_lightsheet_axes(self):
        """Default sheet (2026-09-13): k along +z, thin axis along x, wide axis and
        polarization along y; it crosses the x tweezer."""
        s = LightSheet(waist=(8e-6, 120e-6), wavelength_m=LAM, power=0.2)
        u, v, k = s.frame
        assert np.allclose(k, [0, 0, 1]) and np.allclose(u, [1, 0, 0]) and np.allclose(v, [0, 1, 0])
        assert np.allclose(s.polarization, [0, 1, 0])
        assert s.rayleigh_range_u / s.rayleigh_range_v == pytest.approx((8 / 120) ** 2)
        assert s.peak_intensity == pytest.approx(2 * 0.2 / (np.pi * 8e-6 * 120e-6))
        assert s.intensity(8e-6, 0.0, 0.0) / s.I0 == pytest.approx(np.exp(-2.0))    # thin: x
        assert s.intensity(0.0, 120e-6, 0.0) / s.I0 == pytest.approx(np.exp(-2.0))  # wide: y
        assert s.intensity(0.0, 0.0, s.rayleigh_range_u) / s.I0 > 0.5                # k: z

    def test_lightsheet_rejects_bad_axes(self):
        with pytest.raises(ValueError, match="parallel"):
            LightSheet(waist=W0, wavelength_m=LAM, transverse_axis=(0, 0, 1))   # along k
        with pytest.raises(NotImplementedError):
            LightSheet(waist=W0, wavelength_m=LAM, waist_offset_v=1e-6)
        with pytest.raises(ValueError, match="pair"):
            LightSheet(waist=(1e-6, 2e-6, 3e-6), wavelength_m=LAM)

    def test_right_handed_frame(self):
        for _ in range(20):
            k = RNG.normal(size=3)
            pol = np.cross(k, RNG.normal(size=3))
            b = LightSheet(waist=(2e-6, 5e-6), wavelength_m=LAM, propagation_direction=k,
                           polarization=pol, transverse_axis=RNG.normal(size=3))
            u, v, kk = b.frame
            assert np.allclose(np.cross(u, v), kk, atol=1e-14)


# -------------------------------------------------------------- the field

class TestIntensity:
    def test_matches_gaussian_beam_profile(self):
        """Propagation along x: lab (x, y, z) = (beam z, r, 0)."""
        t, gb = _tw(), GaussianBeam(waist=W0, wavelength=LAM, power=P)
        zs = np.linspace(-3, 3, 13) * t.zR
        rs = np.linspace(0, 3, 13) * W0
        Zs, Rs = np.meshgrid(zs, rs, indexing="ij")
        assert t.intensity(Zs, Rs, 0.0) == pytest.approx(gb.intensity(P, r=Rs, z=Zs), rel=1e-13)

    def test_lightsheet_of_equal_waists_is_the_tweezer(self):
        """Given the tweezer's frame and polarization (the defaults differ), a
        round sheet is the tweezer, bit for bit."""
        t = _tw()
        s = LightSheet(waist=W0, wavelength_m=LAM, power=P, transverse_axis=t.frame[0],
                       propagation_direction=t.frame[2], polarization=t.polarization)
        X, Y, Z = (RNG.normal(size=(50,)) * W0 for _ in range(3))
        assert np.array_equal(s.intensity(X, Y, Z), t.intensity(X, Y, Z))

    def test_rotation_invariance(self):
        b = LightSheet(waist=(2e-6, 4e-6), wavelength_m=LAM, power=P,
                       propagation_direction=(1, 1, 0), polarization=(0, 0, 1),
                       transverse_axis=(0, 0, 1), origin=(1e-6, -2e-6, 0.5e-6))
        pts = RNG.normal(size=(200, 3)) * 5e-6
        for _ in range(5):
            R = _random_rotation(RNG)
            rb = b.rotated(R)
            assert rb.intensity_at(pts @ R.T) == pytest.approx(b.intensity_at(pts), rel=1e-12)
            assert np.allclose(rb.polarization, R @ b.polarization)

    def test_moved_to_translates(self):
        t = _tw()
        d = np.array([1e-6, 2e-6, -3e-6])
        pts = RNG.normal(size=(20, 3)) * W0
        assert t.moved_to(d).intensity_at(pts + d) == pytest.approx(t.intensity_at(pts), rel=1e-13)

    def test_power_is_linear(self):
        t = _tw()
        assert t.scaled(3.0).intensity(0.0, 1e-6, 0.0) == pytest.approx(
            3 * t.intensity(0.0, 1e-6, 0.0), rel=1e-15)

    def test_intensity_gradient_is_analytic(self):
        b = LightSheet(waist=(2e-6, 5e-6), wavelength_m=LAM, power=P,
                       propagation_direction=(1, 1, 0.3), polarization=(0, 0, 1),
                       transversality="project", origin=(1e-6, 0, -1e-6))
        c = b + _tw(wavelength_m=1070e-9, polarization=(0, 1, 0))
        h = 1e-10
        for field in (b, c):
            for p in RNG.normal(size=(20, 3)) * 4e-6:
                g = np.array(field.intensity_gradient(*p))
                fd = np.array([(field.intensity(*(p + e)) - field.intensity(*(p - e))) / (2 * h)
                               for e in np.eye(3) * h])
                assert g == pytest.approx(fd, rel=1e-6, abs=1e-7 * np.max(np.abs(fd)))

    def test_field_amplitude(self):
        t = _tw()
        E = t.electric_field_amplitude(0.0, 0.0, 0.0)
        assert 0.5 * kc.c * kc.epsilon0 * E ** 2 == pytest.approx(t.I0, rel=1e-14)

    def test_broadcasting_shapes(self):
        t = _tw()
        assert np.ndim(t.intensity(0.0, 0.0, 0.0)) == 0
        x = np.linspace(-1, 1, 5)[:, None, None] * W0
        y = np.linspace(-1, 1, 4)[None, :, None] * W0
        z = np.linspace(-1, 1, 3)[None, None, :] * W0
        assert t.intensity(x, y, z).shape == (5, 4, 3)
        assert t.intensity_at(np.zeros((7, 3))).shape == (7,)

    def test_array_generic_source(self):
        for fn in (Beam.intensity, Beam._beam_coords, Beam.intensity_gradient):
            assert "asarray" not in inspect.getsource(fn)

    def test_torch_tensors(self):
        torch = pytest.importorskip("torch")
        t = LightSheet(waist=(2e-6, 4e-6), wavelength_m=LAM, power=P,
                       propagation_direction=(1, 1, 0), polarization=(0, 0, 1))
        pts = RNG.normal(size=(3, 100)) * 3e-6
        ref = t.intensity(*pts)
        got = t.intensity(*(torch.tensor(p) for p in pts))
        assert isinstance(got, torch.Tensor)
        assert got.numpy() == pytest.approx(ref, rel=1e-12)


# ------------------------------------------------------------- composition

class TestCrossed:
    def test_one_beam_is_the_beam(self):
        t = _tw()
        pts = RNG.normal(size=(30, 3)) * W0
        assert np.array_equal(Crossed([t]).intensity_at(pts), t.intensity_at(pts))

    def test_flattening_and_sum(self):
        a = _tw(label="a")
        b = _tw(frequency_Hz=None, wavelength_m=1064.5e-9, label="b")
        c = _tw(wavelength_m=1065e-9, label="c")
        assert len(a + b + c) == 3 and len(Crossed([a + b, c])) == 3
        s = sum([a, b, c])
        assert isinstance(s, Crossed) and len(s) == 3
        pts = RNG.normal(size=(10, 3)) * W0
        assert s.intensity_at(pts) == pytest.approx(
            a.intensity_at(pts) + b.intensity_at(pts) + c.intensity_at(pts), rel=1e-15)
        assert s.characteristic_size == W0

    def test_validation(self):
        with pytest.raises(ValueError, match="at least one"):
            Crossed([])
        with pytest.raises(ValueError, match="Beam or Crossed"):
            Crossed([_tw(), "beam"])

    def test_interference_warning(self):
        a = _tw(propagation_direction=(1, 0, 0), polarization=(0, 0, 1))
        b = _tw(propagation_direction=(0, 1, 0), polarization=(0, 0, 1))
        with pytest.warns(InterferenceWarning, match="interfere"):
            Crossed([a, b])
        with pytest.raises(ValueError, match="interfere"):
            Crossed([a, b], strict=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InterferenceWarning)
            Crossed([a, _tw(propagation_direction=(0, 1, 0), polarization=(1, 0, 0))])  # crossed pol
            Crossed([a, _tw(propagation_direction=(0, 1, 0), polarization=(0, 0, 1),
                            wavelength_m=None, frequency_Hz=kc.c / LAM + 80e6)])                          # AOM offset


class TestConstructors:
    def test_from_gaussian_beam(self):
        gb = GaussianBeam(waist=W0, wavelength=LAM, power=P)
        t = Tweezer.from_gaussian_beam(gb)
        assert t.waist == W0 and t.power == P and t.wavelength_m == LAM

    def test_from_trap_frequency_inverts_the_curvature(self):
        t = Tweezer.from_trap_frequency(1.0e3, polarizability_SI=ALPHA_1064_SI, mass=M,
                                        waist=W0, wavelength_m=LAM)
        U0 = -kc.ac_stark_shift_J(ALPHA_1064_SI, t.I0)
        assert np.sqrt(4 * U0 / (M * W0 ** 2)) / (2 * np.pi) == pytest.approx(1.0e3, rel=1e-13)
        # the operating point pinned in kamo/trap/tests/test_gaussian.py
        assert t.power * 1e6 == pytest.approx(43.65, abs=0.02)

    def test_from_trap_frequency_rejects_a_strength(self):
        with pytest.raises(ValueError, match="strength"):
            Tweezer.from_trap_frequency(1e3, polarizability_SI=ALPHA_1064_SI, mass=M,
                                        waist=W0, wavelength_m=LAM, power=1e-3)
        with pytest.raises(ValueError, match="red-detuned"):
            Tweezer.from_trap_frequency(1e3, polarizability_SI=-ALPHA_1064_SI, mass=M,
                                        waist=W0, wavelength_m=LAM)

    def test_from_trap_frequency_on_an_instance_reuses_its_geometry(self):
        base = Tweezer(waist=W0, wavelength_m=LAM, propagation_direction=(1, 0, 0),
                       polarization=(0, 0, 1), label="1064")
        t = base.from_trap_frequency(1.0e3, polarizability_SI=ALPHA_1064_SI, mass=M)
        ref = Tweezer.from_trap_frequency(1.0e3, polarizability_SI=ALPHA_1064_SI, mass=M,
                                          waist=W0, wavelength_m=LAM,
                                          propagation_direction=(1, 0, 0),
                                          polarization=(0, 0, 1), label="1064")
        assert t.power == pytest.approx(ref.power, rel=1e-14)
        assert t.waist == W0 and t.label == "1064"
        assert np.allclose(t.propagation_direction, (1, 0, 0))
        assert base.power == 0.0                                  # immutable
        with pytest.raises(ValueError, match="reuses its geometry"):
            base.from_trap_frequency(1e3, polarizability_SI=ALPHA_1064_SI, waist=W0)

    def test_repr(self):
        assert "Tweezer" in repr(_tw()) and "LightSheet" in repr(
            LightSheet(waist=(1e-6, 2e-6), wavelength_m=LAM))
