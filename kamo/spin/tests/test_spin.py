"""Tests for kamo.spin.

INTERNAL     Bloch algebra, the SpinField <-> UniformMixture cross-check, and the
             invariance that decides the whole design (z-rotations do not move S_z
             and therefore cannot change an image).
GROUND-TRUTH the Ramsey demonstration at the K-39 operating point.

Run: pytest kamo/spin/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import kamo.constants as kc
from kamo.BEC_properties.variational import GaussianVariationalCloud
from kamo.imaging import ProbeBeam, Propagator, UniformMixture, readout
from kamo.spin import (ImagePulse, Rotate, Sequence, SpinField, SpinGeometry,
                       ramsey)

B_GAUSS = 520.5830387285664
A_UPUP_A0 = 11.333713034704147
N_ATOMS = 500.0

G_UP, E_UP = (4, 0, 1 / 2, -1 / 2, +1 / 2), (4, 1, 3 / 2, -3 / 2, +1 / 2)
G_DN, E_DN = (4, 0, 1 / 2, -1 / 2, -1 / 2), (4, 1, 3 / 2, -3 / 2, -1 / 2)


@pytest.fixture(scope="module")
def cloud():
    return GaussianVariationalCloud.from_tweezer(
        N=N_ATOMS, a_scattering=A_UPUP_A0 * kc.a0, f_radial_Hz=1.0e3, waist=3.0e-6)


@pytest.fixture(scope="module")
def probe():
    from kamo import Potassium39
    p = ProbeBeam.from_midpoint(Potassium39(), B_gauss=B_GAUSS,
                                ground_up=G_UP, excited_up=E_UP,
                                ground_dn=G_DN, excited_dn=E_DN)
    return ProbeBeam.from_offresonant_saturation(p, 0.001, which="up")


@pytest.fixture(scope="module")
def prop(probe, cloud):
    return Propagator.for_cloud(probe.response, cloud, n_grid=256, n_slices=80)


@pytest.fixture(scope="module")
def geom(prop, cloud):
    return SpinGeometry.from_propagator(prop, cloud)


# ============================================================ INTERNAL: state


class TestSpinField:
    def test_spin_coherent_purity(self, geom):
        for xi in (-1.0, -0.5, 0.0, 0.3, 1.0):
            f = SpinField.spin_coherent(geom, xi)
            assert f.Sz_total == pytest.approx(xi, abs=1e-12)
            assert f.purity == pytest.approx(np.ones(geom.shape), rel=1e-12)
            assert f.coherence == pytest.approx(
                np.full(geom.shape, np.sqrt(1 - xi**2)), rel=1e-10)

    def test_polarized_state_has_no_coherence(self, geom):
        f = SpinField.spin_coherent(geom, 1.0)
        assert f.contrast == pytest.approx(0.0, abs=1e-12)

    def test_balanced_state_has_full_coherence(self, geom):
        f = SpinField.spin_coherent(geom, 0.0)
        assert f.contrast == pytest.approx(1.0, rel=1e-12)

    def test_rejects_out_of_range(self, geom):
        with pytest.raises(ValueError):
            SpinField.spin_coherent(geom, 1.5)

    def test_rotations_compose(self, geom):
        a = SpinField.spin_coherent(geom, 1.0)
        a.rotate_y(np.pi / 2).rotate_y(np.pi / 2)
        b = SpinField.spin_coherent(geom, 1.0).rotate_y(np.pi)
        assert a.sz == pytest.approx(b.sz, abs=1e-12)
        assert a.sx == pytest.approx(b.sx, abs=1e-12)

    def test_pi_pulse_inverts(self, geom):
        f = SpinField.spin_coherent(geom, 1.0).rotate_y(np.pi)
        assert f.Sz_total == pytest.approx(-1.0, abs=1e-12)

    def test_rotation_preserves_length(self, geom):
        f = SpinField.spin_coherent(geom, 0.3)
        before = f.purity.copy()
        f.rotate((1.0, 2.0, -0.5), 0.7)
        assert f.purity == pytest.approx(before, rel=1e-12)

    def test_z_rotation_preserves_sz(self, geom):
        """The invariance the whole module is designed around."""
        f = SpinField.spin_coherent(geom, 0.0)
        before = f.sz.copy()
        f.rotate_z(np.random.default_rng(0).normal(size=geom.shape))
        assert f.sz == pytest.approx(before, abs=1e-15)

    def test_decoherence_preserves_populations(self, geom):
        f = SpinField.spin_coherent(geom, 0.2)
        before = f.sz.copy()
        f.decohere(0.3)
        assert f.sz == pytest.approx(before, abs=1e-15)
        assert f.contrast == pytest.approx(0.3 * np.sqrt(1 - 0.04), rel=1e-10)

    def test_inhomogeneous_dephasing_kills_contrast_not_coherence(self, geom):
        """A cloud can be everywhere locally coherent and still show no fringe."""
        f = SpinField.spin_coherent(geom, 0.0)
        rng = np.random.default_rng(1)
        f.rotate_z(rng.uniform(-np.pi, np.pi, geom.shape))
        assert f.coherence == pytest.approx(np.ones(geom.shape), rel=1e-10)
        assert f.contrast < 0.1                      # but the vector sum cancels

    def test_collapse_to_columns_uniform(self, geom):
        f = SpinField.spin_coherent(geom, 0.4)
        xi_col, Z_col = f.collapse_to_columns()
        assert xi_col == pytest.approx(np.full(xi_col.shape, 0.4), rel=1e-10)
        assert np.abs(Z_col) == pytest.approx(
            np.full(Z_col.shape, np.sqrt(1 - 0.16)), rel=1e-10)

    def test_column_collapse_detects_axial_dephasing(self, geom):
        """Opposite phases at the two ends of each column must cancel in Z."""
        f = SpinField.spin_coherent(geom, 0.0)
        sign = np.where(geom.x[:, None, None] > 0, 1.0, -1.0)
        f.rotate_z(sign * np.pi / 2)
        _, Z_col = f.collapse_to_columns()
        assert np.abs(Z_col).max() < 0.2             # nearly complete cancellation
        assert f.coherence.min() == pytest.approx(1.0, rel=1e-10)

    def test_sample_shot_scales_as_one_over_sqrt_N(self, geom):
        f = SpinField.spin_coherent(geom, 0.0)
        rng = np.random.default_rng(2)
        xis = np.array([f.sample_shot(rng).Sz_total for _ in range(1500)])
        target = 1.0 / np.sqrt(geom.cloud.N)
        assert abs(xis.mean()) < 5 * target / np.sqrt(1500)
        assert xis.std() == pytest.approx(target, rel=0.08)

    def test_projection_noise_is_grid_independent(self, probe, cloud):
        """Shot noise is physics; refining the OPTICAL grid must not change it.

        Drawing per pixel column fails this: on a fine grid a column holds order
        one atom, the Gaussian is wider than the physical range of xi, and clipping
        silently eats the variance -- 36% of it at 768^2.  Hence the noise cells of
        SpinField.noise_block_size.
        """
        target = 1.0 / np.sqrt(cloud.N)
        seen = []
        for n_grid in (192, 384):
            p = Propagator.for_cloud(probe.response, cloud, n_grid=n_grid,
                                     n_slices=40)
            g = SpinGeometry.from_propagator(p, cloud)
            f = SpinField.spin_coherent(g, 0.0)
            rng = np.random.default_rng(7)
            seen.append(np.array([f.sample_shot(rng).Sz_total
                                  for _ in range(1200)]).std())
        for sd in seen:
            assert sd == pytest.approx(target, rel=0.10)
        assert seen[0] == pytest.approx(seen[1], rel=0.10)

    def test_polarized_state_has_no_projection_noise(self, geom):
        """Var(xi) = (1 - xi^2)/N vanishes at full polarization: nothing to project."""
        f = SpinField.spin_coherent(geom, 1.0)
        rng = np.random.default_rng(5)
        xis = np.array([f.sample_shot(rng).Sz_total for _ in range(50)])
        assert xis.std() == 0.0

    def test_sample_shot_keeps_states_pure(self, geom):
        f = SpinField.spin_coherent(geom, 0.0)
        s = f.sample_shot(np.random.default_rng(3))
        assert s.purity == pytest.approx(np.ones(geom.shape), rel=1e-9)


# ================================================== INTERNAL: optics bridge


class TestSusceptibilityBridge:
    def test_uniform_spin_field_matches_uniform_mixture(self, prop, cloud, probe,
                                                        geom):
        """A SpinField with uniform xi must reproduce UniformMixture EXACTLY.

        The two layers compute the same susceptibility by different routes; if they
        ever disagree the spin layer has silently changed the optics.
        """
        r = probe.response
        for xi in (0.0, 1.0, -0.6):
            field = SpinField.spin_coherent(geom, xi)
            a = prop.propagate(field.susceptibility_source(r, probe),
                               s0_incident=probe.s0_incident)
            b = prop.propagate(UniformMixture(cloud, r, probe.species(xi)),
                               s0_incident=probe.s0_incident)
            assert np.abs(a.psi_exit - b.psi_exit).max() < 1e-9

    def test_structured_field_differs_from_its_column_average(self, prop, probe,
                                                              geom):
        """Axial structure in S_z is not equivalent to its column mean.

        This is the entire justification for a 3D spin state: if the propagation
        could not tell them apart, a 2D column field would do.
        """
        r = probe.response
        structured = SpinField.spin_coherent(geom, 0.0)
        sign = np.where(geom.x[:, None, None] > 0, 1.0, -1.0)
        structured.sz = 0.5 * sign * np.ones(geom.shape)     # +1 / -1 halves
        structured.sx = np.zeros(geom.shape)
        structured.sy = np.zeros(geom.shape)

        xi_col, _ = structured.collapse_to_columns()
        assert np.abs(xi_col).max() < 1e-9                   # column mean is zero
        flat = SpinField.spin_coherent(geom, 0.0)            # ... same column mean

        a = prop.propagate(structured.susceptibility_source(r, probe),
                           s0_incident=probe.s0_incident)
        b = prop.propagate(flat.susceptibility_source(r, probe),
                           s0_incident=probe.s0_incident)
        assert np.abs(a.psi_exit - b.psi_exit).max() > 1e-3


# ================================================ INTERNAL: the key invariance


class TestZRotationInvariance:
    def test_z_rotation_preserves_sz_exactly(self, geom):
        """Not approximately: bit-for-bit.  See SpinField._apply_rotation."""
        f = SpinField.spin_coherent(geom, 0.3)
        before = f.sz.copy()
        for _ in range(20):
            f.rotate_z(np.random.default_rng(9).uniform(-np.pi, np.pi, geom.shape))
        assert np.abs(f.sz - before).max() == 0.0

    def test_z_rotation_does_not_change_the_image(self, prop, probe, geom):
        """Rotate about z by anything at all; the image is bit-for-bit unchanged.

        A z-rotation commutes with S_z, the susceptibility depends only on S_z, so
        no amount of imprinted phase is visible to a probe until something converts
        it into population.  This is why the module exposes sequences rather than a
        single 'imprint then re-image' call.
        """
        r = probe.response
        f = SpinField.spin_coherent(geom, 0.3)
        before = prop.propagate(f.susceptibility_source(r, probe),
                                s0_incident=probe.s0_incident)
        f.rotate_z(np.random.default_rng(4).uniform(-np.pi, np.pi, geom.shape))
        after = prop.propagate(f.susceptibility_source(r, probe),
                               s0_incident=probe.s0_incident)
        assert np.abs(after.psi_exit - before.psi_exit).max() == 0.0

    def test_back_action_false_leaves_the_field_alone(self, prop, probe, geom):
        f = SpinField.spin_coherent(geom, 0.0)
        seq = Sequence([ImagePulse(prop, t_pulse=5e-6, back_action=False)])
        out = seq.run(f, probe=probe)
        assert out.field.sx == pytest.approx(f.sx, abs=1e-15)
        assert out.field.contrast == pytest.approx(f.contrast, rel=1e-12)

    def test_image_pulse_rotates_and_decoheres(self, prop, probe, geom):
        f = SpinField.spin_coherent(geom, 0.0)
        out = Sequence([ImagePulse(prop, t_pulse=5e-6)]).run(f, probe=probe)
        res = out[0]
        # phi_z is NEGATIVE at the midpoint: nu_diff = nu_up - nu_dn with
        # delta_up < 0 makes the up state shift down and the dn state up.
        assert np.abs(res.phi_z).max() > 0.5             # it rotated
        assert out.field.Sz_total == pytest.approx(0.0, abs=1e-12)   # ... about z
        assert out.field.contrast < f.contrast           # and it decohered
        assert 0.0 < res.photons_per_atom < 1.0


# ============================================================== GROUND TRUTH


class TestRamsey:
    """The demonstration this module was built for, at the operating point."""

    @pytest.fixture(scope="class")
    def out(self, prop, probe, geom):
        field = SpinField.spin_coherent(geom, Sz_total=1.0)
        return ramsey(prop, t_pulse=5e-6, NA=0.42).run(field, probe=probe)

    def test_first_image_sees_a_balanced_cloud(self, out):
        """After the opening pi/2 the cloud is balanced: no lens, tiny phase."""
        first = out[0]
        assert abs(first.recovered_phase) < 0.02
        assert first.propagation.mean_intensity < 1.0       # absorbs, does not focus
        assert first.propagation.s_peak < 1.1 * 0.34

    def test_first_image_still_imprints(self, out):
        """Invisible to itself, but it writes a multi-radian phase grating.

        The grating is only mildly non-uniform -- ~15% across the cloud -- and that
        is a consequence of imaging the BALANCED state: with Re alpha cancelled the
        cloud is a pure absorber, so the intensity an atom sees varies only through
        absorption, not lensing.  A polarized cloud at the same detuning
        concentrates the probe several-fold and would imprint a correspondingly
        steeper grating.  The mildness is why the sequence keeps its contrast.
        """
        first = out[0]
        mag = np.abs(first.phi_z)
        assert mag.max() > 1.0                           # multi-radian
        assert 1.05 < mag.max() / mag.min() < 1.5        # structured, but gently

    def test_second_image_sees_what_the_first_wrote(self, out):
        """The closing pi/2 converts phase to population, and the signal jumps."""
        first, second = out[0], out[1]
        assert abs(second.signal) > 5 * abs(first.signal)

    def test_population_became_structured(self, out):
        """S_z after the sequence is -cos(phi_z), hence spatially varying."""
        assert out.final_contrast > 0.5
        assert abs(out.final_Sz) < 0.5

    def test_contrast_is_lost_to_scattering(self, out):
        """Two 5 us pulses cost real coherence -- this is not a footnote."""
        total = sum(r.photons_per_atom for r in out.images)
        assert 0.05 < total < 1.0
        assert out.final_contrast < np.exp(-0.5 * total) + 0.1

    def test_ramsey_needs_a_polarized_start(self, prop, probe, geom):
        """Starting balanced drives the opening pi/2 to the POLE instead.

        Guards the documented usage: the first pulse is what creates the
        superposition, so handing it one already made produces a fully polarized
        cloud -- which lenses hard and gives a large spurious first signal.
        """
        bad = ramsey(prop, t_pulse=5e-6).run(
            SpinField.spin_coherent(geom, Sz_total=0.0), probe=probe)
        assert abs(bad[0].recovered_phase) > 1.0        # NOT a balanced cloud
        good = ramsey(prop, t_pulse=5e-6).run(
            SpinField.spin_coherent(geom, Sz_total=1.0), probe=probe)
        assert abs(good[0].recovered_phase) < 0.02
