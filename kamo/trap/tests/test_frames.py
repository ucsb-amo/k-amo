"""Tests for kamo.trap.frames.

INTERNAL only: vector validation, frames, and the lab-frame (beta, gamma)
invariants.  Pure numpy -- no ARC, no network.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from kamo.trap import frames as fr

RNG = np.random.default_rng(20260910)
SQ2 = np.sqrt(2.0)

# Spherical basis about z, the convention of kamo.hamiltonian.builder.
E_SPH = {+1: -np.array([1.0, 1.0j, 0.0]) / SQ2,
         -1: np.array([1.0, -1.0j, 0.0]) / SQ2,
         0: np.array([0.0, 0.0, 1.0])}


def _random_rotation(rng, proper=True):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if (np.linalg.det(q) < 0) == proper:          # flip to the requested handedness
        q[:, 0] = -q[:, 0]
    return q


def _random_polarization(rng):
    return rng.normal(size=3) + 1j * rng.normal(size=3)


# ------------------------------------------------------------------ casting

class TestCasting:
    def test_list_is_cast_normalized_and_copied(self):
        src = np.array([0.0, 2.0, 2.0j])
        eps = fr.as_unit_complex_vector(src)
        assert eps.dtype == np.complex128
        assert np.sum(np.abs(eps) ** 2) == pytest.approx(1.0, rel=1e-15)
        eps[0] = 99.0
        assert src[0] == 0.0                          # no aliasing
        assert np.allclose(fr.as_unit_complex_vector([0, 1, 1j]), [0, 1 / SQ2, 1j / SQ2])

    def test_normalization_is_idempotent(self):
        eps = fr.as_unit_complex_vector(_random_polarization(RNG))
        assert np.allclose(fr.as_unit_complex_vector(eps), eps, rtol=0, atol=1e-15)  # ~ulp

    def test_global_phase_is_kept(self):
        eps = fr.as_unit_complex_vector([0, 1j, 1j])
        assert eps[1] == pytest.approx(1j / SQ2)

    def test_real_direction(self):
        k = fr.as_unit_real_vector((3, 0, 4))
        assert k.dtype == np.float64 and np.allclose(k, [0.6, 0, 0.8])

    @pytest.mark.parametrize("bad, match", [
        ([1.0, 1.0j], "length-2"),
        ([1, 0, 0, 0], "3-vector"),
        ([[1, 0, 0]], "3-vector"),
        ([0, 0, 0], "zero vector"),
        ([np.nan, 1, 0], "NaN"),
        ([np.inf, 1, 0], "NaN or inf"),
        (["a", "b", "c"], "numeric"),
        ([True, False, False], "numeric"),
        (None, "numeric"),
    ])
    def test_bad_polarizations(self, bad, match):
        with pytest.raises(ValueError, match=match):
            fr.as_unit_complex_vector(bad)

    def test_length_2_hint_is_only_for_polarizations(self):
        with pytest.raises(ValueError) as err:
            fr.as_unit_real_vector([1.0, 0.0])
        assert "length-2" not in str(err.value)

    def test_complex_direction_rejected_roundoff_accepted(self):
        with pytest.raises(ValueError, match="must be real"):
            fr.as_unit_real_vector([1, 1e-3j, 0])
        k = fr.as_unit_real_vector(np.array([1, 0, 0]) + 1e-15j)
        assert k.dtype == np.float64 and np.allclose(k, [1, 0, 0])


class TestTransversality:
    def test_default_beam_is_transverse(self):
        eps = fr.check_transverse([0, 1, 1j], [1, 0, 0])
        assert np.allclose(eps, [0, 1 / SQ2, 1j / SQ2])

    def test_longitudinal_polarization_raises_by_default(self):
        with pytest.raises(ValueError, match="not transverse"):
            fr.check_transverse([1, 0, 0], [1, 0, 0])

    def test_warn_returns_unchanged(self):
        with pytest.warns(UserWarning, match="not transverse"):
            eps = fr.check_transverse([0.05, 1, 0], [1, 0, 0], mode="warn")
        assert abs(eps[0]) > 0.04

    def test_project_removes_the_longitudinal_part(self):
        k = fr.as_unit_real_vector([1, 1, 0])
        eps = fr.check_transverse([1, 0, 0.3j], k, mode="project")
        assert abs(np.dot(k, eps)) < 1e-15
        assert np.sum(np.abs(eps) ** 2) == pytest.approx(1.0, rel=1e-15)

    def test_project_of_a_parallel_vector_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            fr.check_transverse([0, 0, 2], [0, 0, 1], mode="project")

    def test_ignore_is_silent(self):
        eps = fr.check_transverse([1, 0, 0], [1, 0, 0], mode="ignore")
        assert np.allclose(eps, [1, 0, 0])

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="transversality"):
            fr.check_transverse([0, 1, 0], [1, 0, 0], mode="fix")


# ------------------------------------------------------------------- frames

class TestFrames:
    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 0, 1], [0, 0, -1], [1, 2, 3],
                                      [0.1, 0.0, 0.995], [-2, 5, 0.3]])
    def test_right_handed_orthonormal(self, axis):
        e1, e2, e3 = fr.orthonormal_frame(axis)
        E = np.array([e1, e2, e3])
        assert np.allclose(E @ E.T, np.eye(3), atol=1e-14)
        assert np.allclose(np.cross(e1, e2), e3, atol=1e-14)
        assert np.allclose(e3, np.asarray(axis) / np.linalg.norm(axis))

    def test_default_matches_cone_quadrature_seed_rule(self):
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
        seed = np.array([0.0, 0.0, 1.0])              # |axis_z| = 0.80 < 0.9
        e1 = np.cross(seed, axis)
        e1 /= np.linalg.norm(e1)
        assert np.allclose(fr.orthonormal_frame(axis)[0], e1, atol=1e-15)

    def test_first_sets_the_in_plane_orientation(self):
        e1, e2, e3 = fr.orthonormal_frame([0, 1, 0], first=[0.2, 0.3, 1.0])
        assert np.allclose(e3, [0, 1, 0])
        assert np.allclose(e1, np.array([0.2, 0, 1.0]) / np.hypot(0.2, 1.0))
        assert np.allclose(np.cross(e1, e2), e3)

    def test_first_parallel_to_axis_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            fr.orthonormal_frame([0, 0, 1], first=[0, 0, -3])

    def test_rotation_about(self):
        R = fr.rotation_about([0, 0, 1], np.pi / 2)
        assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-15)
        assert fr.as_rotation_matrix(R) is not R
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_rotation_validation(self):
        with pytest.raises(ValueError, match="orthogonal"):
            fr.as_rotation_matrix(2 * np.eye(3))
        with pytest.raises(ValueError, match="reflection"):
            fr.as_rotation_matrix(np.diag([1.0, 1.0, -1.0]))
        assert np.allclose(fr.as_rotation_matrix(np.diag([1.0, 1.0, -1.0]), proper=False),
                           np.diag([1.0, 1.0, -1.0]))
        with pytest.raises(ValueError, match="3x3"):
            fr.as_rotation_matrix(np.eye(2))


# ------------------------------------------------------ polarization geometry

class TestPolarizationGeometry:
    @pytest.mark.parametrize("name, beta, gamma", [
        ("pi", 0.0, 1.0), ("sigma+", -1.0, -0.5), ("sigma-", 1.0, -0.5),
        ("linear_perp", 0.0, -0.5)])
    def test_reference_table(self, name, beta, gamma):
        from kamo.hamiltonian.builder import _POLARIZATIONS, _polarization_geometry
        pol = _POLARIZATIONS[name]
        eps = sum(c * E_SPH[q] for q, c in pol.items())
        got = fr.polarization_geometry(eps, [0, 0, 1])
        assert got == pytest.approx((beta, gamma), abs=1e-15)
        assert got == pytest.approx(_polarization_geometry(pol), abs=1e-15)

    def test_matches_spherical_implementation_for_random_states(self):
        from kamo.hamiltonian.builder import _polarization_geometry
        for _ in range(50):
            c = RNG.normal(size=3) + 1j * RNG.normal(size=3)
            pol = {-1: c[0], 0: c[1], +1: c[2]}
            eps = sum(a * E_SPH[q] for q, a in pol.items())
            assert fr.polarization_geometry(eps, [0, 0, 1]) == pytest.approx(
                _polarization_geometry(pol), abs=1e-13)

    def test_user_default_depends_on_the_quantization_axis(self):
        """The same lab vector is sigma+ about x but has no vector shift about z."""
        eps = [0, 1, 1j]
        assert fr.polarization_geometry(eps, [1, 0, 0]) == pytest.approx((-1.0, -0.5), abs=1e-15)
        assert fr.polarization_geometry(eps, [0, 0, 1]) == pytest.approx((0.0, 0.25), abs=1e-15)

    def test_matches_light_shift_cartesian_convention(self):
        """compute_complete_polarizability's formula (index 0 = quantization axis),
        applied to eps expressed in a right-handed frame (Bhat, e1, e2)."""
        for _ in range(50):
            eps = fr.as_unit_complex_vector(_random_polarization(RNG))
            bhat = fr.as_unit_real_vector(RNG.normal(size=3))
            f1, f2, _ = fr.orthonormal_frame(bhat)            # (f1, f2, bhat) right-handed
            pol = np.array([bhat @ eps, f1 @ eps, f2 @ eps])  # (bhat, f1, f2) right-handed
            beta_ls = np.imag(np.cross(pol, np.conj(pol)))[0]
            gamma_ls = np.real((3 * np.conj(pol[0]) * pol[0] - 1) / 2)
            assert fr.polarization_geometry(eps, bhat) == pytest.approx(
                (beta_ls, gamma_ls), abs=1e-13)

    def test_invariant_under_proper_rotations(self):
        for _ in range(50):
            eps, bhat = _random_polarization(RNG), RNG.normal(size=3)
            R = _random_rotation(RNG, proper=True)
            assert fr.polarization_geometry(R @ eps, R @ bhat) == pytest.approx(
                fr.polarization_geometry(eps, bhat), abs=1e-13)

    def test_beta_is_a_pseudoscalar(self):
        """Reflecting eps and Bhat together flips beta and leaves gamma alone --
        the check that catches a left-handed frame."""
        for _ in range(50):
            eps, bhat = _random_polarization(RNG), RNG.normal(size=3)
            R = _random_rotation(RNG, proper=False)
            b0, g0 = fr.polarization_geometry(eps, bhat)
            b1, g1 = fr.polarization_geometry(R @ eps, R @ bhat)
            assert b1 == pytest.approx(-b0, abs=1e-13)
            assert g1 == pytest.approx(g0, abs=1e-13)

    def test_bounds(self):
        for _ in range(200):
            b, g = fr.polarization_geometry(_random_polarization(RNG), RNG.normal(size=3))
            assert -1 - 1e-12 <= b <= 1 + 1e-12
            assert -0.5 - 1e-12 <= g <= 1 + 1e-12
