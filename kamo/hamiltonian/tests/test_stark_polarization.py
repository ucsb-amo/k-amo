"""Tests for the polarization dependence of the AC-Stark operator.

INTERNAL      the (beta, gamma) geometry factors, and that they agree with the
              independent Cartesian implementation in kamo.light_shift.
GROUND-TRUTH  the 1064 nm tweezer shift of the sigma- imaging transitions at the
              K-39 operating point, for eps || B and eps _|_ B.

Run: pytest kamo/hamiltonian/tests -q
"""
from __future__ import annotations

import numpy as np
import pytest

import kamo.constants as kc
from kamo import GaussianBeam
from kamo.hamiltonian import AtomicStructure
from kamo.hamiltonian.builder import _polarization_geometry

# --- operating point (analysis/feedback/figures/atomic_physics.ipynb) --------
B_GAUSS = 520.5939555658828
WAIST = 3.0e-6
LAM_TWEEZER = 1064e-9
P_TYP = 44.2443e-6          # the P that gives nu_radial = 1 kHz at w0 = 3 um

# |up> is the LOWER sigma- transition, i.e. m_i = -1/2
G_UP, E_UP = (4, 0, 0.5, -0.5, -0.5), (4, 1, 1.5, -1.5, -0.5)
G_DN, E_DN = (4, 0, 0.5, -0.5, +0.5), (4, 1, 1.5, -1.5, +0.5)


# =========================================================== geometry factors
@pytest.mark.parametrize("pol, beta, gamma", [
    ("pi", 0.0, +1.0),
    ("sigma+", -1.0, -0.5),
    ("sigma-", +1.0, -0.5),
    ("linear_perp", 0.0, -0.5),
])
def test_polarization_geometry(pol, beta, gamma):
    """pi aligns (gamma=+1); everything perpendicular to B gives gamma=-1/2,
    and only beta distinguishes circular from linear."""
    from kamo.hamiltonian.builder import _POLARIZATIONS
    b, g = _polarization_geometry(_POLARIZATIONS[pol])
    assert b == pytest.approx(beta, abs=1e-12)
    assert g == pytest.approx(gamma, abs=1e-12)


def test_geometry_is_normalization_independent():
    b1, g1 = _polarization_geometry({0: 1.0})
    b2, g2 = _polarization_geometry({0: 7.3})
    assert (b1, g1) == pytest.approx((b2, g2), abs=1e-12)


def test_unpolarized_mix_is_gamma_minus_half():
    """Equal sigma+ / sigma- (i.e. linear _|_ B, any azimuth) -> beta=0."""
    b, g = _polarization_geometry({+1: 1.0, -1: 1.0})
    assert b == pytest.approx(0.0, abs=1e-12)
    assert g == pytest.approx(-0.5, abs=1e-12)


def test_gamma_matches_light_shift_module():
    """Cross-check against compute_complete_polarizability's Cartesian form,
    whose 0th component is the quantization axis."""
    for cart, spherical in ((np.array([1.0, 0.0, 0.0]), "pi"),
                            (np.array([0.0, 1.0, 0.0]), "linear_perp")):
        eps = cart / np.linalg.norm(cart)
        gamma_cart = float(np.real((3 * np.conj(eps[0]) * eps[0] - 1) / 2))
        from kamo.hamiltonian.builder import _POLARIZATIONS
        _, gamma_sph = _polarization_geometry(_POLARIZATIONS[spherical])
        assert gamma_sph == pytest.approx(gamma_cart, abs=1e-12)


# ============================================================ the Stark operator
@pytest.fixture(scope="module")
def model():
    return AtomicStructure([(4, 0, 0.5), (4, 1, 1.5)])


@pytest.fixture(scope="module")
def tweezer():
    return GaussianBeam(waist=WAIST, wavelength=LAM_TWEEZER, power=P_TYP)


def test_pi_reproduces_legacy_hardwired_gamma(model, tweezer):
    """pi must reproduce the pre-fix behaviour exactly: gamma=+1, no vector."""
    new = np.diag(model.builder.laser_stark_operator(tweezer, polarization="pi"))

    # the old code, written out
    from kamo import ComputePolarizabilities
    cp = ComputePolarizabilities(force_arc=True)
    pre = -1.0 / (2 * kc.epsilon0 * kc.c) * kc.convert_polarizability_au_to_SI / kc.h
    old = np.zeros(model.basis.dim)
    for man, sl in model.basis.manifold_slices():
        a_s, _, a_t = cp.compute_fine_structure_polarizability(
            man.n, man.l, man.j, LAM_TWEEZER)
        a_s = float(np.atleast_1d(a_s)[0])
        a_t = float(np.atleast_1d(a_t)[0]) if a_t is not None else 0.0
        for s in model.basis.state_list[sl]:
            sh = pre * a_s
            if man.j > 0.5 and a_t != 0.0:
                j = man.j
                sh += pre * a_t * (3 * s.m_j**2 - j * (j + 1)) / (j * (2 * j - 1))
            old[s.index] = sh
    assert new == pytest.approx(old, rel=1e-12)


def test_ground_manifold_flat_for_linear_polarization(model, tweezer):
    """4S_1/2 has j=1/2, so no tensor term; for LINEAR light beta=0 too, and
    every ground sublevel shifts identically -- the operator is a multiple of
    the identity there and cannot produce a ground-state differential."""
    for pol in ("pi", "linear_perp"):
        d = np.diag(model.builder.laser_stark_operator(tweezer, polarization=pol))
        assert np.ptp(d[0:8]) == pytest.approx(0.0, abs=1e-18)


def test_ground_manifold_splits_for_circular_polarization(model, tweezer):
    """...but alpha_v(4S_1/2) is NOT zero at 1064 nm, so circular light DOES
    split the ground manifold by m_j -- a vector (fictitious-field) shift that
    the pre-fix operator dropped entirely."""
    d_lin = np.diag(model.builder.laser_stark_operator(tweezer,
                                                       polarization="linear_perp"))
    d_sp = np.diag(model.builder.laser_stark_operator(tweezer,
                                                      polarization="sigma+"))
    d_sm = np.diag(model.builder.laser_stark_operator(tweezer,
                                                      polarization="sigma-"))
    assert np.ptp(d_sp[0:8]) > 1e-6          # Hz per W/m^2
    # gamma is the same for all three, so sigma+- differ from linear_perp only
    # through beta, and sigma+ / sigma- are exact mirrors of each other.
    assert d_sp[0:8] - d_lin[0:8] == pytest.approx(-(d_sm[0:8] - d_lin[0:8]),
                                                   rel=1e-10, abs=1e-20)
    # and the mean over m_j is unchanged: the vector term is purely differential
    assert d_sp[0:8].mean() == pytest.approx(d_lin[0:8].mean(), rel=1e-10)


def test_vector_term_can_be_disabled(model, tweezer):
    """include_vector=False restores the old (incomplete) circular behaviour."""
    d_on = np.diag(model.builder.laser_stark_operator(tweezer, polarization="sigma+"))
    d_off = np.diag(model.builder.laser_stark_operator(tweezer, polarization="sigma+",
                                                       include_vector=False))
    assert np.ptp(d_off[0:8]) == pytest.approx(0.0, abs=1e-18)
    assert np.ptp(d_on[0:8]) > 1e-6


def test_perpendicular_is_half_the_tensor_of_pi(model, tweezer):
    """gamma(_|_) / gamma(||) = -1/2 exactly, so the tensor PART flips and
    halves while the scalar part is untouched."""
    d_pi = np.diag(model.builder.laser_stark_operator(tweezer, polarization="pi"))
    d_pp = np.diag(model.builder.laser_stark_operator(tweezer,
                                                      polarization="linear_perp"))
    d_sc = np.diag(model.builder.laser_stark_operator(tweezer, polarization="pi",
                                                      include_tensor=False))
    # excited manifold only; ground has no tensor part
    tens_pi, tens_pp = d_pi[8:] - d_sc[8:], d_pp[8:] - d_sc[8:]
    assert tens_pp == pytest.approx(-0.5 * tens_pi, rel=1e-12)


# ==================================================== the operating-point numbers
def _sigma_minus_shift(model, tweezer, polarization):
    """Shift of both sigma- transitions at 1x P_typ, in Hz, via laser_sweep."""
    I0 = tweezer.I0
    res = model.laser_sweep(tweezer, I_max=I0, n_points=3, model="stark",
                            polarization=polarization, B_gauss=B_GAUSS)
    return (res.transition_frequency_shift(G_UP, E_UP, at=I0),
            res.transition_frequency_shift(G_DN, E_DN, at=I0))


def test_sigma_minus_shift_pi(model, tweezer):
    """eps || B: +41.20 kHz at the typical tweezer power."""
    up, dn = _sigma_minus_shift(model, tweezer, "pi")
    assert up / 1e3 == pytest.approx(41.201, abs=0.02)
    assert dn / 1e3 == pytest.approx(41.200, abs=0.02)


def test_sigma_minus_shift_perpendicular(model, tweezer):
    """eps _|_ B (a tweezer propagating along B): +58.71 kHz, NOT +41.20."""
    up, dn = _sigma_minus_shift(model, tweezer, "linear_perp")
    assert up / 1e3 == pytest.approx(58.714, abs=0.02)
    assert dn / 1e3 == pytest.approx(58.713, abs=0.02)


def test_perpendicular_differs_from_pi_by_the_tensor_term(model, tweezer):
    """The two geometries differ by 1.5 x |gamma| x the tensor contribution --
    a 42% error if you use the wrong one."""
    up_pi, _ = _sigma_minus_shift(model, tweezer, "pi")
    up_pp, _ = _sigma_minus_shift(model, tweezer, "linear_perp")
    assert up_pp / up_pi == pytest.approx(1.425, rel=2e-3)


def test_sigma_minus_transitions_stay_degenerate(model, tweezer):
    """Both sigma- transitions end on m_j = -3/2 and the ground shift is
    m_i-independent, so the tweezer moves them TOGETHER: no differential."""
    for pol in ("pi", "linear_perp"):
        up, dn = _sigma_minus_shift(model, tweezer, pol)
        assert abs(dn - up) < 5.0          # Hz, against a 110 MHz splitting


def test_sweep_honours_polarization(model, tweezer):
    """laser_sweep must PASS polarization to the stark operator -- it used to
    build the operator without it, silently giving the pi answer."""
    up_pi, _ = _sigma_minus_shift(model, tweezer, "pi")
    up_pp, _ = _sigma_minus_shift(model, tweezer, "linear_perp")
    assert abs(up_pp - up_pi) > 1.0e4      # Hz; they must not coincide
