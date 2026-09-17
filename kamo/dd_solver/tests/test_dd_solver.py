"""Tests for kamo.dd_solver.

Numbered as in the build specification (T1-T21) plus the sanity checks S1-S8.

THEOREMS   T1-T9, T15, T16, T21 and S1-S4, S7, S8 are exact identities and are
           tested to machine precision.  If one fails the CODE is wrong.
LIMITS     T4, T7, T10-T13, T17, T18 are analytic limits with stated tolerances.
GROUND     T14, T19, T20 pin the operating-point physics (excess law, coherent-
           field convergence, guiding sign); they are statistical and their
           tolerances come from the calibration run of 2026-09-16 recorded in
           k-jam/jpagett/dd_solver/tests_and_logs/.

Run: pytest kamo/dd_solver -q      (about 3 minutes; the heavy tests use joblib)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from kamo.dd_solver import (Configuration, GaussianProfile, OperatingPoint,
                            sample_configuration, solve)
from kamo.dd_solver import detect, ensemble, fields, rg, stats, vector
from kamo.dd_solver.cloud import uniform_sphere_configuration
from kamo.dd_solver.kernel import (kernel_pair, near_field_hamiltonian, scalar_couplings,
                                   gamma_matrix)
from kamo.dd_solver.solver import (SanityWarning, independent_solution,
                                   single_atom_cross_section)
from kamo.dd_solver.system import GaussianBeam, PlaneWave

XI_L = 1 / (6 * np.pi * np.sqrt(3.0))
XI_C = 1 / (12 * np.pi * np.sqrt(3.0))
N_JOBS = 8


@pytest.fixture(scope="module")
def op():
    return OperatingPoint.nominal()


@pytest.fixture(scope="module")
def prof500():
    return GaussianProfile.operating_point(500)


@pytest.fixture(scope="module")
def spec500():
    return GaussianProfile.spec_reference(500)


@pytest.fixture(scope="module")
def res300(op, prof500):
    cfg = sample_configuration(prof500, theta=np.pi / 2, seed=11, N=300)
    return solve(cfg, op, "full", positivity=True)


# ============================================================ theorems


def test_T1_independent_atoms(op, prof500):
    cfg = sample_configuration(prof500, theta=np.pi / 3, seed=1, N=100)
    r = solve(cfg, op, "independent")
    # alpha = -(f/2)/(delta + i/2): f is the ABSOLUTE oscillator strength of the
    # driven line (0.977 at 520.583 G), not 1.
    expect = -0.5 * op.strengths(cfg.spins) * r.Omega / (op.detunings(cfg.spins) + 0.5j)
    assert np.max(np.abs(r.beta - expect)) < 1e-15
    assert np.max(np.abs(independent_solution(cfg, op) - expect)) < 1e-15
    ideal = OperatingPoint.nominal().__class__(op.linewidth_Hz, op.wavelength, op.delta_up,
                                               op.delta_dn)
    r1 = solve(cfg, ideal, "independent")
    assert np.max(np.abs(r1.beta / r.beta - 1 / op.strengths(cfg.spins))) < 1e-12


def test_T2_S1_optical_theorem(res300):
    assert res300.checks.optical_theorem < 1e-12
    assert res300.checks.passed


def test_S1_catches_the_double_linewidth_bug(op, prof500):
    """M built with i*Gamma (unit-diagonal Gamma in the coupling) on the diagonal."""
    from kamo.dd_solver.solver import build_matrix, sanity_checks, solve_linear
    cfg = sample_configuration(prof500, theta=0.0, seed=2, N=150)
    det = op.detunings(cfg.spins)
    M, J, G = build_matrix(cfg.positions, det, op, "full")
    Mbad = M.copy()
    Mbad[np.arange(150), np.arange(150)] = det + 1.0j           # Gamma instead of Gamma/2
    Om = PlaneWave(op.k).drive(cfg.positions, op.e_hat)
    beta, _ = solve_linear(Mbad, -0.5 * Om)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = sanity_checks(Mbad, G, beta, Om, warn=False)
    assert rep.optical_theorem > 0.1
    assert not rep.passed


def test_T3_S7_angular_quadrature_closure(res300):
    quad, qf, rel = fields.far_field_power(res300, n_u=200, n_phi=256)
    assert rel < 1e-8
    # the quadrature resolution is a convergence parameter: a coarse one is worse
    _, _, rel_coarse = fields.far_field_power(res300, n_u=12, n_phi=16)
    assert rel_coarse > rel


def test_T4_two_atom_limits():
    for x in (1e-2, 1e-3):
        for c in (0.0, 0.5, 1 / 3):
            B = 1 - 3 * c
            J, G = kernel_pair(x, c, "full")
            assert abs(J * x ** 3 / 0.75 - B) < 5e-4 if B != 0 else abs(J * x ** 3) < 5e-4
            assert abs(G - 1.0) < 5e-4
    # pair eigen-decays 1 +- Gamma_12 -> 2, 0
    op = OperatingPoint.nominal()
    pos = np.array([[0, 0, 0], [0, 0, 5e-9]])
    J, G = scalar_couplings(pos, op.k, op.e_hat, "full")
    ev = np.linalg.eigvalsh(gamma_matrix(G))
    assert abs(ev.max() - 2.0) < 1e-3 and abs(ev.min()) < 1e-3


def test_T5_S8_self_consistency(res300):
    assert fields.self_consistency_residual(res300) < 1e-12


def test_T6_analytic_single_dipole(op):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(300, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    for kr in (0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        pts = d * kr / op.k + 1e-7
        ref = fields.analytic_dipole_field(pts, np.full(3, 1e-7), 0.3 - 0.7j, op.k, op.e_hat)
        got = fields.scattered_field(pts, np.full((1, 3), 1e-7), np.array([0.3 - 0.7j]), op.k, op.e_hat)
        assert np.max(np.abs(ref - got)) / np.max(np.abs(ref)) < 1e-13


def test_T7_unlike_spin_repulsion():
    for J in (0.1, 1.0, 5.0, 20.0):
        ev = np.linalg.eigvalsh(np.array([[9.1, J], [J, -9.1]]))
        assert np.all(np.abs(ev) >= 9.1)
        wp, wm = rg.pair_eigenvalues(9.1, -9.1, J)
        assert min(abs(wp), abs(wm)) >= 9.1


def test_T8_vector_scalar_reduction(op, spec500):
    cfg = sample_configuration(spec500, theta=np.pi / 2, seed=3, N=150)
    chk = vector.reduction_check(cfg, op)
    assert chk["beta_difference"] < 1e-13
    assert chk["out_of_plane"] < 1e-13


def test_T8b_channel_amplitude_ratios(op):
    """alpha_pi / alpha_- and alpha_+ / alpha_- from the operating point's channels."""
    a_minus = abs(op.polarizability_scalar(op.delta_up))
    ratios = {ch.q: abs(0.5 * ch.strength / (ch.detuning + 0.5j)) / a_minus for ch in op.channels_up}
    # spec: 0.0357 and 0.0092 with the bare 2/3, 1/3 strengths at -170/-331 Gamma;
    # kamo's convention puts them at -152/-312 with dressed strengths
    assert 0.03 < ratios[0] < 0.045
    assert 0.007 < ratios[1] < 0.012


def test_T9_rg_on_a_pair(op):
    pos = np.array([[0, 0, 0], [0, 0, 60e-9]])
    H = near_field_hamiltonian(pos, op.k, op.e_hat)[0, 1]
    for w0 in ([9.14, 9.14], [9.14, -9.14], [-9.14, 9.14], [2.0, -5.0]):
        r = rg.renormalize(pos, np.array(w0), op.k, op.e_hat, cutoff=1e-3)
        wp, wm = rg.pair_eigenvalues(w0[0], w0[1], H)
        assert abs(max(r.omega) - wp) < 1e-9 and abs(min(r.omega) - wm) < 1e-9
        assert r.n_steps == 1


def test_T10_rg_tracked_sum_rules(op, prof500):
    cfg = sample_configuration(prof500, theta=np.pi / 4, seed=4)
    r = rg.renormalize_configuration(cfg, op, tracked=True)
    assert r.tracked and r.n_steps > 100
    assert abs(np.mean(np.abs(r.drive) ** 2) - 0.5) < 1e-9          # |Omega|^2 = 1/2 for y light
    assert abs(np.mean(r.gamma) - 1.0) < 1e-9


def test_T11_rg_cutoff_convergence(op, prof500):
    cfg = sample_configuration(prof500, theta=0.0, seed=5)
    a = rg.renormalize_configuration(cfg, op, cutoff=1e-2)
    b = rg.renormalize_configuration(cfg, op, cutoff=1e-3)
    assert np.max(np.abs(a.omega - b.omega)) < 0.05
    ea = solve(cfg, op, "rg", rg_cutoff=1e-2, keep_matrices=False).excitation
    eb = solve(cfg, op, "rg", rg_cutoff=1e-3, keep_matrices=False).excitation
    assert abs(ea / eb - 1) < 1e-2


def test_T12_gp_regression():
    """kamo's Gaussian-variational cloud against an independent minimisation of the
    correct functional (kinetic hbar^2/8m sigma^2 per axis), and the build spec's
    table against its own (doubled-kinetic) functional -- see cloud.py."""
    import kamo.constants as kc
    from scipy.optimize import minimize
    m, hb = kc.m_K, kc.hbar
    g = 4 * np.pi * hb ** 2 * 11.3 * kc.a0 / m
    wr, wx = 2 * np.pi * 1170.0, 2 * np.pi * 93.0
    table = {500: (401, 1.64, 19.2), 1000: (405, 1.83, 33.7), 2000: (411, 2.13, 56.4),
             3500: (418, 2.45, 82.7)}
    lam = OperatingPoint.nominal().wavelength
    for N, (sr_t, sx_t, eta_t) in table.items():
        def E(p):
            sr, sx = np.exp(p)
            return (hb ** 2 / (8 * m) * (2 / sr ** 2 + 1 / sx ** 2)
                    + 0.5 * m * (2 * wr ** 2 * sr ** 2 + wx ** 2 * sx ** 2)
                    + g * N / (16 * np.pi ** 1.5 * sr ** 2 * sx))
        r = minimize(E, np.log([3.5e-7, 1.6e-6]), method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-40, maxiter=5000))
        sr, sx = np.exp(r.x)
        p = GaussianProfile.operating_point(N)
        assert abs(p.sigma[1] / sr - 1) < 1e-5 and abs(p.sigma[0] / sx - 1) < 1e-5
        q = GaussianProfile.spec_reference(N)
        assert abs(q.sigma[1] * 1e9 / sr_t - 1) < 0.01
        assert abs(q.sigma[0] * 1e6 / sx_t - 1) < 0.01
        assert abs(q.eta_eff(lam) / eta_t - 1) < 0.01
        # the non-interacting limit of the correct functional is the textbook one
    p0 = GaussianProfile.operating_point(1, a_bohr=0.0)
    assert abs(p0.sigma[1] / np.sqrt(hb / (2 * m * wr)) - 1) < 1e-6


def test_T13_xi_tail(op):
    """RG on a uniform sphere: fraction with |shift| > W is xi eta / (pi W)."""
    lam = op.wavelength
    for eta in (100.0, 500.0):
        n = eta / lam ** 3
        for e_hat, xi in ((np.array([0.0, 0.0, 1.0]), XI_L), (op.e_hat, XI_C)):
            fr = []
            W = 20.0 if eta == 100.0 else 50.0
            for seed in range(3):
                cfg = uniform_sphere_configuration(3000, n, seed=seed)
                r = rg.renormalize(cfg.positions, np.zeros(3000), op.k, e_hat, cutoff=1e-2)
                R = np.linalg.norm(cfg.positions, axis=1)
                core = R < 0.8 * R.max()                     # surface atoms have fewer neighbours
                fr.append(np.mean(np.abs(r.shifts[core]) > W))
            assert abs(np.mean(fr) / (xi * eta / (np.pi * W)) - 1) < 0.15
    # circular is exactly half of linear in the analytic angular factor
    u = np.linspace(-1, 1, 200001)
    assert abs(np.trapezoid(np.abs(1 - 3 * u ** 2), u) / 2 - 4 / (3 * np.sqrt(3))) < 1e-6
    assert abs(np.trapezoid(np.abs(1 - 1.5 * (1 - u ** 2)), u) / 2 - 2 / (3 * np.sqrt(3))) < 1e-6


def test_T14_excess_law(op):
    """Near-field excess: its SCALING, which is the robust content of the law.

    The excess is defined on the EXCITATION sum|beta|^2, against the literature
    ablation 'nonear' (exact Gamma, static near field removed).  Recalibrated
    2026-09-17; the old test used the unphysical 'far' kernel.

    What is tested is what the resonant-shell argument actually predicts and what
    a measurement can resolve at 40 configurations:

    1. linear in eta_eff -- the coefficient xi is the same at two densities a
       factor 2.7 apart;
    2. the angular factor 1 - sin^2(theta)/2, i.e. the like-pair fraction.

    The COEFFICIENT is recorded, not asserted against the published xi_circ: it
    measures at xi_circ / 2 (ensemble.XI_MEASURED).  The tempting explanation --
    that one detuning sign only reaches half the angular distribution -- is
    refuted by test_pair_has_two_resonances, so the factor of two stays open and
    the measured value is what the package quotes.  See ensemble.excess_law.
    """
    xis = {}
    for scale in (1.0, 1.4):
        prof = GaussianProfile.spec_reference(1000).scaled(scale)
        eta = prof.eta_eff(op.wavelength)
        ex = {}
        for theta in (0.0, np.pi):
            e = ensemble.run_ensemble(prof, op, theta, 40, seed0=0,
                                      variants=("full", "nonear", "independent"),
                                      n_jobs=N_JOBS)
            assert e.checks_passed()
            ex[theta] = ((e.excitation("full") - e.excitation("nonear"))
                         / e.excitation("independent"))
        both = 0.5 * (ex[0.0] + ex[np.pi])            # same seeds: pair the signs
        xis[scale] = stats.trimmed_mean(both, 0.1) / eta
        assert xis[scale] > 0
    assert abs(xis[1.4] / xis[1.0] - 1) < 0.35, xis   # linear in eta_eff
    assert abs(xis[1.0] / ensemble.XI_MEASURED - 1) < 0.35, xis
    assert xis[1.0] < 0.8 * XI_C, xis                 # and it is NOT the published xi


def test_T14b_excess_angular_factor(op):
    """The excess follows the like-pair fraction 1 - sin^2(theta)/2."""
    prof = GaussianProfile.spec_reference(1000)
    val = {}
    for th in (0.0, np.pi / 4):
        ex = {}
        for theta in (th, np.pi - th):
            e = ensemble.run_ensemble(prof, op, theta, 40, seed0=0,
                                      variants=("full", "nonear", "independent"),
                                      n_jobs=N_JOBS)
            ex[theta] = ((e.excitation("full") - e.excitation("nonear"))
                         / e.excitation("independent"))
        val[th] = stats.trimmed_mean(0.5 * (ex[th] + ex[np.pi - th]), 0.1)
    expect = (1 - 0.5 * np.sin(np.pi / 4) ** 2) / 1.0          # 0.75
    assert abs(val[np.pi / 4] / val[0.0] / expect - 1) < 0.25, val


def test_T15_S4_reciprocity(res300):
    assert res300.checks.reciprocity < 1e-13


def test_T16_S2_positivity(res300, op, prof500):
    """EVERY variant is a passive medium, because Im G has no near-field part.

    Corrected 2026-09-17: the 'far' variant used to truncate Im g as well, which
    put 141 of 500 eigenvalues of its Gamma below zero (min -1.51) and gave the
    solver 97 gain modes.  Truncating the imaginary part is not a near-field
    ablation -- see test_gamma_is_the_radiation_pattern_overlap."""
    assert res300.checks.positivity_min_eig > -1e-10
    assert abs(res300.checks.decay_sum - 1.0) < 1e-12
    cfg = sample_configuration(prof500, theta=0.0, seed=6, N=200)
    for variant in ("full", "far", "nonear"):
        r = solve(cfg, op, variant, positivity=True, warn=False)
        assert r.checks.positivity_min_eig > -1e-10, variant
        assert r.checks.passed, variant
        # no gain modes: every eigenvalue of M sits in the upper half plane
        M, _, _ = __import__("kamo.dd_solver.solver", fromlist=["x"]).build_matrix(
            cfg.positions, op.detunings(cfg.spins), op, variant,
            strengths=op.strengths(cfg.spins))
        assert np.linalg.eigvals(M).imag.min() > 0.0, variant


def test_gamma_is_the_radiation_pattern_overlap(op):
    """Gamma_ij = (3/8pi) int dOmega [1 - |n.e|^2] e^{i x n.rhat} for EVERY variant.

    This is the statement that the decay matrix is purely radiative and band
    limited (hence positive semidefinite by Bochner's theorem), so there is no
    near-field term in it to ablate."""
    e = op.e_hat
    ct, w = np.polynomial.legendre.leggauss(200)
    ph = 2 * np.pi * np.arange(256) / 256
    st = np.sqrt(1 - ct ** 2)
    n = np.stack(np.broadcast_arrays(st[:, None] * np.cos(ph), st[:, None] * np.sin(ph),
                                     ct[:, None] * np.ones(256)), -1)
    ne = n @ e
    rng = np.random.default_rng(0)
    for _ in range(8):
        r = rng.normal(size=3)
        r /= np.linalg.norm(r)
        x = float(rng.uniform(0.05, 6.0))
        c = float(abs(np.conj(e) @ r) ** 2)
        quad = float(np.real(3 / (8 * np.pi) * ((1 - np.abs(ne) ** 2)
                                                * np.exp(1j * x * (n @ r))
                                                * w[:, None]).sum() * (2 * np.pi / 256)))
        for variant in ("full", "far", "nonear"):
            assert abs(kernel_pair(x, c, variant)[1] - quad) < 1e-12, variant
    # and the x -> 0 limit is 1, not (3/2)(1 - c)
    for c in (0.0, 0.25, 0.5):
        assert abs(kernel_pair(1e-4, c, "full")[1] - 1.0) < 1e-6


def test_nonear_removes_only_the_static_near_field(op):
    """'nonear' is J_full - (3/4)B/x^3 (Andreoli Eq. A.3), 'far' is the 1/x term.

    At the operating point's typical neighbour distance the two ablations differ
    by more than the term they are meant to isolate, so they are not
    interchangeable."""
    x, c = 1.23, 0.0                       # k r_nn at the operating density, polar pair
    J_full, g_full = kernel_pair(x, c, "full")
    J_non, g_non = kernel_pair(x, c, "nonear")
    J_far, g_far = kernel_pair(x, c, "far")
    assert abs((J_full - J_non) - 0.75 * (1 - 3 * c) / x ** 3) < 1e-12
    assert abs(J_far + 0.75 * (1 - c) * np.cos(x) / x) < 1e-12
    assert g_full == g_non == g_far        # exact Gamma in all three
    # 'far' throws away a further 0.199 on top of the 0.403 static term: the two
    # ablations differ by half the near-field term itself, so they are not
    # interchangeable and a "near-field" number must say which one it used.
    assert abs(J_non - J_far) > 0.4 * abs(J_full - J_non)


def test_T17_S5_low_density_convergence(op, prof500):
    prev = None
    for s in (2.0, 4.0, 8.0, 16.0):
        p = prof500.scaled(s)
        dev = []
        for seed in range(3):
            cfg = sample_configuration(p, theta=np.pi / 2, seed=seed, N=200)
            dev.append([abs(solve(cfg, op, v, keep_matrices=False).excess - 1)
                        for v in ("full", "far", "rg")])
        dev = np.mean(dev, axis=0)
        if prev is not None:
            assert np.all(dev < prev + 1e-3)
        prev = dev
    assert np.all(prev < 0.02)


def test_T18_S6_beer_lambert(op):
    """Transversely wide pancake, sigma- plane wave along z, delta = 2 Gamma:
    the coherent transmitted field is exp(-OD/2 - i delta OD), NOT the Born
    1 - (OD/2)(1 + 2 i delta) (which has |t| > 1 here)."""
    delta = 2.0
    op2 = OperatingPoint(op.linewidth_Hz, op.wavelength, delta, delta)
    inc = PlaneWave(op.k, khat=(0, 0, 1), polarization=op.e_hat)
    pk = GaussianProfile(1000, (3e-6, 3e-6, 0.3e-6))
    pts = np.array([[x, y, 1.2e-6] for x in (-0.4e-6, 0, 0.4e-6) for y in (-0.4e-6, 0, 0.4e-6)])
    OD = op.sigma0 / (1 + 4 * delta ** 2) * pk.column_density(pts[:, 0], pts[:, 1], axis=2)
    t_bl = np.exp(-OD / 2 - 1j * delta * OD)
    res = [solve(sample_configuration(pk, 0.0, seed=s), op2, "full", incident=inc,
                 keep_matrices=False) for s in range(60)]
    cf = fields.coherent_field(res, pts, R_exc=150e-9)
    t = (cf.mean @ np.conj(op.e_hat)) / (cf.E_inc @ np.conj(op.e_hat))
    assert 0.25 < OD[4] < 0.35
    assert np.mean(np.abs(t - t_bl)) < 0.07                   # 0.044 on 2026-09-16
    t_born = 1 - OD / 2 * (1 + 2j * delta)
    assert np.mean(np.abs(t - t_born)) > 2 * np.mean(np.abs(t - t_bl))


def test_T19_coherent_field_convergence(op, spec500):
    """<E> on fixed points converges in n_config and R_exc, and the converged map
    is independent of R_exc over a few interparticle spacings."""
    sx = spec500.sigma[0]
    pts = np.array([[-3 * sx, 0, 0], [0, 0, 0], [3 * sx, 0, 0], [0, 1e-6, 0]])
    e = ensemble.run_ensemble(spec500, op, np.pi, 40, seed0=0, variants=("full",), n_jobs=N_JOBS)
    res = e.results["full"]
    I = {}
    sem = {}
    for n in (10, 40):
        for R in (0.0, 150e-9, 250e-9):
            cf = fields.coherent_field(res[:n], pts, R_exc=R)
            I[n, R] = cf.intensity_coherent
            sem[n, R] = cf.relative_sem()
    assert sem[40, 150e-9] < sem[10, 150e-9]                  # converges in n_config
    assert sem[40, 150e-9] < sem[40, 0.0]                     # exclusion removes variance
    assert np.max(np.abs(I[40, 150e-9] / I[40, 250e-9] - 1)) < 0.08   # R_exc independent
    assert np.max(np.abs(I[40, 0.0] / I[40, 150e-9] - 1)) < 0.25     # unbiased, noisier


def test_T20_guiding_sign(op, spec500):
    """Red-detuned (index > 1) concentrates on-axis intensity; blue depletes it."""
    sx = spec500.sigma[0]
    pts = np.array([[-3 * sx, 0, 0], [0, 0, 0], [3 * sx, 0, 0], [0, 1e-6, 0]])
    out = {}
    for theta in (np.pi, 0.0):          # pi: all |dn> (red, guiding); 0: all |up> (blue)
        e = ensemble.run_ensemble(spec500, op, theta, 40, seed0=0, variants=("full",), n_jobs=N_JOBS)
        out[theta] = fields.coherent_field(e.results["full"], pts, R_exc=150e-9).intensity_coherent
    red, blue = out[np.pi], out[0.0]
    assert red[1] > 1.2 and red[2] > red[1]                    # centre bright, exit brighter
    assert blue[1] < 0.8 and blue[2] < blue[1]
    assert red[1] > red[3] and blue[1] < blue[3]               # on-axis vs 1 um off-axis
    # the specification's reference values (its profile, 40 configs, R_exc = 150 nm)
    assert abs(red[1] / 1.45 - 1) < 0.25 and abs(blue[1] / 0.49 - 1) < 0.35


def test_T21_far_field_is_the_plane_wave_spectrum(op, spec500):
    """detect.scattered_spectrum (from F(n)) against the FFT of the microscopic
    field on a downstream plane; agreement limited by the finite box (~1%)."""
    import scipy.fft as sfft
    from kamo.imaging.grid import TransverseGrid
    cfg = sample_configuration(spec500, theta=np.pi, seed=5, N=80)
    r = solve(cfg, op, "full")
    g = TransverseGrid(256, 40e-6, op.k)
    xp = 6e-6
    pf = fields.transmitted_plane(r, xp, g.axis, g.axis)
    Et_plane = (sfft.fft2(sfft.ifftshift(pf.E - pf.E_inc, axes=(0, 1)), axes=(0, 1)) * g.d ** 2
                * np.exp(-1j * g.KX * xp)[..., None])
    Et_F = detect.scattered_spectrum(cfg.positions, r.beta, op.k, op.e_hat, g, NA=0.42)
    m = g.na_mask(0.42)
    assert np.max(np.abs(Et_plane[m] - Et_F[m])) / np.max(np.abs(Et_F[m])) < 0.03


# ============================================================ sanity S3


def test_S3_single_atom_cross_section(op):
    assert abs(single_atom_cross_section(op, 0.0) / op.sigma0 - 1) < 1e-12
    assert abs(single_atom_cross_section(op, 0.0, polarization=(0, 1, 0)) / op.sigma0 - 0.5) < 1e-12
    # Lorentzian at delta = 3 Gamma: sigma0 / (1 + 4 delta^2)
    assert abs(single_atom_cross_section(op, 3.0) / (op.sigma0 / 37) - 1) < 1e-12
    assert abs(op.sigma0 - 3 * op.wavelength ** 2 / (2 * np.pi)) < 1e-30
    # physical number: the D2 value at the high-field transition (CLAUDE.md)
    assert abs(op.sigma0 / 2.8067e-13 - 1) < 1e-3


# ============================================================ solvers


def test_solver_paths_agree(op, prof500):
    cfg = sample_configuration(prof500, theta=np.pi / 2, seed=8, N=300)
    ref = solve(cfg, op, "full").beta
    scale = np.max(np.abs(ref))
    for kw in (dict(precision="mixed"), dict(method="gmres")):
        b = solve(cfg, op, "full", **kw).beta
        assert np.max(np.abs(b - ref)) / scale < 1e-8, kw
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu:
        for prec in ("double", "mixed"):
            b = solve(cfg, op, "full", backend="gpu", precision=prec).beta
            assert np.max(np.abs(b - ref)) / scale < 1e-8
        E1 = fields.exciting_field_at_atoms(solve(cfg, op, "full"))
        E2 = fields.exciting_field_at_atoms(solve(cfg, op, "full"), backend="gpu")
        assert np.max(np.abs(E1 - E2)) / np.max(np.abs(E1)) < 1e-10


def test_ensemble_parallel_matches_serial(op, prof500):
    a = ensemble.run_ensemble(prof500, op, 0.5, 4, seed0=3, variants=("full",), n_jobs=1, N=100)
    b = ensemble.run_ensemble(prof500, op, 0.5, 4, seed0=3, variants=("full",), n_jobs=2, N=100)
    for ra, rb in zip(a.results["full"], b.results["full"]):
        # workers run one BLAS thread, the serial path many: rounding differs at 1e-16
        assert np.allclose(ra.beta, rb.beta, rtol=0, atol=1e-12)
        assert ra.config.seed == rb.config.seed


# ============================================================ sampling


def test_sampling_api(prof500):
    c = sample_configuration(prof500, theta=1.0, seed=1)
    c2 = sample_configuration(prof500, theta=1.0, seed=1)
    assert np.array_equal(c.positions, c2.positions) and np.array_equal(c.spins, c2.spins)
    s = c.resample_spins(spin_seed=99)
    assert np.array_equal(s.positions, c.positions) and not np.array_equal(s.spins, c.spins)
    p = c.resample_positions(prof500, seed=99)
    assert np.array_equal(p.spins, c.spins) and not np.array_equal(p.positions, c.positions)
    assert abs(c.S_z - 0.5 * (c.n_up - c.n_dn)) == 0
    # spins iid with p_up = cos^2(theta/2)
    big = sample_configuration(prof500, theta=1.0, seed=2, N=20000)
    assert abs(big.n_up / big.N - np.cos(0.5) ** 2) < 0.01
    # positions Gaussian with the profile's widths
    assert np.allclose(big.positions.std(axis=0), prof500.sigma, rtol=0.03)


def test_profile_contract_for_imaging(prof500):
    """The profile satisfies kamo.imaging's cloud contract (N, widths, density)."""
    assert np.allclose(prof500.widths, np.sqrt(2) * prof500.sigma)
    n0 = prof500.N / (np.pi ** 1.5 * np.prod(prof500.widths))
    assert abs(n0 / prof500.peak_density - 1) < 1e-12
    assert abs(prof500.density(0, 0, 0) / n0 - 1) < 1e-12
    assert abs(prof500.eta_eff(1.0) / (n0 / 2 ** 1.5) - 1) < 1e-12


def test_incident_fields(op):
    pw = PlaneWave(op.k)
    pts = np.array([[1e-6, 2e-6, -3e-6]])
    assert np.allclose(pw.field(pts)[0], [0, np.exp(1j * op.k * 1e-6), 0])
    assert abs(pw.sigma_projection - 0.5) < 1e-15
    gb = GaussianBeam(op.k, waist=50e-6)
    # over a 0.4 um cloud the intensity varies by < 0.1%
    I = np.sum(np.abs(gb.field(np.array([[0, 0.4e-6, 0], [0, 0, 0]]))) ** 2, 1)
    assert abs(I[0] / I[1] - 1) < 1e-3
    with pytest.raises(ValueError):
        PlaneWave(op.k, khat=(1, 0, 0), polarization=(1, 0, 0))


# ============================================================ stats and detection


def test_stats_helpers():
    x = np.r_[np.ones(20), 50.0]
    s = stats.robust_summary(x)
    assert s["median"] == 1.0 and s["trimmed_mean"] < s["mean"]
    with pytest.warns(stats.PrecisionWarning):
        stats.check_precision(x, 0.01, "x")
    fits = stats.fit_within_groups([0, 0, 0, 0, 1, 1], [-1, 0, 1, 2, 0, 1], [1, 2, 3, 4, 5, 6])
    assert abs(fits[0.0]["slope"] - 1.0) < 1e-12
    assert np.isnan(fits[1.0]["slope"])                 # two points: no fit
    assert stats.near_field_noise([1, 2, 3], [1, 2, 3]) == 0.0


def test_detection_units(op, spec500):
    """Independent atoms read S_z exactly; the local slope is finite; the result
    carries both the detected signal and the old proxy."""
    cfg = sample_configuration(spec500, theta=np.pi / 2, seed=7, N=200)
    sysm = detect.ImagingSystem(NA=0.42, n_grid=128, L_box=30e-6)
    ri = solve(cfg, op, "independent", keep_matrices=False)
    d = detect.detect(ri, sysm)
    assert abs(d.atoms_on_axis - cfg.S_z) < 1e-9 and abs(d.deviation_atoms_on_axis) < 1e-9
    assert np.isfinite(d.slope_on_axis) and d.slope_on_axis != 0
    rf = solve(cfg, op, "full", keep_matrices=False)
    df = detect.detect(rf, sysm)
    assert abs(df.forward_amplitude - rf.forward_amplitude()) < 1e-12
    assert np.isfinite(df.atoms_on_axis)


def test_coarse_grain_warns(op, spec500):
    cfg = sample_configuration(spec500, theta=0.0, seed=9, N=100)
    r = solve(cfg, op, "full", keep_matrices=False)
    ax = np.linspace(-1e-6, 1e-6, 9)
    fs = fields.field_on_grid(r, ax, ax, ax, mask_radius=10e-9)
    assert fs.masked.shape == (9, 9, 9)
    with pytest.warns(fields.CoarseGrainWarning):
        fields.coarse_grain(fs.E, (ax, ax, ax), 300e-9, positions=cfg.positions)


# ================================ added 2026-09-17 by the physics audit


def test_T22_forward_amplitude_optical_theorem(op, prof500):
    """extinction = (4 pi / k) Im[pol* . f(0)] / sigma0 with f = (3/2k) F(khat).

    Unlike S1, which is an algebraic identity of the linear system and drops any
    real symmetric J, this ties together the 3/2 prefactor of the Green dyadic,
    the -Omega/2 right-hand side, the +i/2 on the diagonal and the e^{-i k n.r}
    sign convention of the far-field amplitude.  A sign error in any one of them
    breaks it."""
    cfg = sample_configuration(prof500, theta=np.pi / 3, seed=5, N=120)
    r = solve(cfg, op, "full")
    F = fields.far_field_amplitude(r.incident.khat[None, :], cfg.positions, r.beta,
                                   op.k, op.e_hat)[0]
    f0 = 1.5 / op.k * F                                   # scattering amplitude (length)
    forward = 4 * np.pi / op.k * np.imag(np.conj(r.incident.polarization) @ f0) / op.sigma0
    assert abs(forward / r.extinction - 1) < 1e-12
    # and the same number from the quadratic form, up to the Raman leak
    assert abs((r.radiated_power + r.raman_leak) / r.extinction - 1) < 1e-12
    assert r.raman_leak > 0                               # f < 1 at this operating point


def test_T23_ab_window_orientation_on_an_anisotropic_cloud(op):
    """The A/B window is y/z-symmetric for the real cloud, so a transposed or
    swapped axis would be invisible.  Check the orientation on a deliberately
    anisotropic profile instead: the microscopic coherent field and the BPM must
    be elongated along the SAME transverse axis."""
    from kamo.dd_solver import compare_bpm
    prof = GaussianProfile(200, (1.2e-6, 1.6e-6, 0.5e-6))     # sigma_y >> sigma_z
    pt = compare_bpm.compare_at(prof, op, theta=np.pi, n_config=6, variants=("full",),
                                window=5e-6, n_grid=192, L_box=24e-6, n_slices=60)

    def aspect(psi):
        w = np.abs(psi - 1.0)
        c = psi.shape[0] // 2
        return float(w[c, :].sum() / w[:, c].sum())

    a_bpm, a_mic = aspect(pt.psi_bpm), aspect(pt.psi["full"])
    # axis 0 of psi is y (sigma_y = 1.6 um), axis 1 is z (sigma_z = 0.5 um), so
    # the imprint is WIDER along axis 0 and this ratio is below 1 for both codes
    assert a_bpm < 0.85, a_bpm
    assert abs(a_mic / a_bpm - 1) < 0.2, (a_mic, a_bpm)


def test_driving_intensity_is_what_excites_the_atom(op, prof500):
    """|beta_j|^2 = |alpha_j|^2 |conj(e).E_exc|^2, and the TOTAL |E_exc|^2 is a
    different (much larger, differently distributed) number."""
    tot_all, rel_all = [], []
    for seed in range(4):
        cfg = sample_configuration(prof500, theta=np.pi, seed=seed, N=500)
        r = solve(cfg, op, "full", keep_matrices=False)
        E = fields.exciting_field_at_atoms(r)
        drive = fields.driving_intensity(E, op.e_hat)
        alpha = op.polarizability_scalar(r.detunings, op.strengths(cfg.spins))
        assert np.max(np.abs(np.abs(r.beta) ** 2 - np.abs(alpha) ** 2 * drive)) <             1e-14 * np.max(np.abs(r.beta) ** 2)
        total = np.sum(np.abs(E) ** 2, axis=1)
        assert np.all(total >= drive - 1e-12)      # a projection is never larger
        tot_all.append(total)
        rel_all.append(fields.driving_intensity(E, op.e_hat, incident=r.incident))
    total, rel = np.concatenate(tot_all), np.concatenate(rel_all)
    # the TAIL is where they part company: a near-field spike is mostly in the
    # e_+ and z components, which this line cannot absorb
    assert np.percentile(total, 99.9) > 2.0 * np.percentile(rel, 99.9)
    assert np.mean(total > 10) > 1.5 * np.mean(rel > 10)
    assert 0.3 < np.median(rel) < 3.0


def test_light_shift_uses_the_projected_intensity(op):
    """One atom in the probe: the light shift must equal delta * Gamma * rho_ee
    with rho_ee = (s0/2)|beta|^2 -- i.e. built on the sigma- projection of the
    field, not on the total intensity."""
    s0 = 0.30
    cfg = Configuration(np.zeros((1, 3)), np.array([-1], dtype=np.int8))
    r = solve(cfg, op, "full", keep_matrices=False)
    rho_ee = 0.5 * s0 * float(np.abs(r.beta[0]) ** 2)
    expect = op.delta_dn * op.linewidth_Hz * rho_ee
    # an isolated atom in the y-polarized Voigt probe has driving intensity 1/2
    drive = fields.driving_intensity(r.incident.field(np.zeros((1, 3))), op.e_hat)[0]
    assert abs(drive - 0.5) < 1e-12
    got = float(fields.light_shift_landscape(drive, op, op.delta_dn, s0))
    assert abs(got / expect - 1) < 1e-12
    assert got < 0                                         # red detuning lowers the ground state
    # the total-intensity mistake this replaced would give exactly 2x
    assert abs(fields.light_shift_landscape(1.0, op, op.delta_dn, s0) / got - 2.0) < 1e-12


def test_detection_unit_refuses_a_useless_slope(op, spec500):
    """Near the dark fringe the local slope passes through zero and changes sign;
    the atom-equivalent unit must refuse rather than return a large number."""
    cfg = sample_configuration(spec500, theta=0.0, seed=4, N=500)     # all up
    sysm = detect.ImagingSystem(NA=0.42, n_grid=128, L_box=30e-6)
    r = solve(cfg, op, "full", keep_matrices=False)
    d = detect.detect(r, sysm)
    assert not d.unit_usable_on_axis
    with pytest.warns(detect.DetectionUnitWarning):
        assert np.isnan(d.deviation_atoms_on_axis)
    # at balance the same machinery works and independent atoms read S_z
    cfgb = sample_configuration(spec500, theta=np.pi / 2, seed=4, N=500)
    db = detect.detect(solve(cfgb, op, "independent", keep_matrices=False), sysm)
    assert db.unit_usable_on_axis
    assert abs(db.atoms_on_axis - cfgb.S_z) < 1e-9


def test_near_field_cutoff_radius_covers_linear_dipoles(op, prof500):
    """The KD-tree neighbour list must not miss pairs for a pi dipole, where
    |B| reaches 2 rather than 1."""
    from kamo.dd_solver.kernel import near_field_pairs, near_field_hamiltonian
    cfg = sample_configuration(prof500, theta=0.0, seed=8, N=400)
    for q in (-1, 0):
        e = __import__("kamo.dipole_dipole.green_tensor",
                       fromlist=["x"]).spherical_unit_vector(q)
        i, j, H, _, _ = near_field_pairs(cfg.positions, op.k, e, 1e-2)
        Hd = near_field_hamiltonian(cfg.positions, op.k, e)
        iu = np.triu_indices(cfg.N, 1)
        n_dense = int(np.sum(np.abs(Hd[iu]) >= 1e-2))
        assert len(i) == n_dense, (q, len(i), n_dense)


def test_tracked_rg_passes_the_optical_theorem(op, prof500):
    """The brightness-tracked RG puts gamma_j on the diagonal; S1 must test the
    identity the solved matrix actually obeys, not the unit-diagonal one."""
    cfg = sample_configuration(prof500, theta=np.pi / 2, seed=1, N=300)
    r = solve(cfg, op, "rg", rg_tracked=True)
    assert r.checks.passed, r.checks
    assert r.checks.optical_theorem < 1e-12



def test_gridded_kamo_cloud_round_trip(op):
    """A kamo.trap GP cloud goes straight into the microscopic solver AND into
    kamo.imaging, as the same object (the integration added 2026-09-17)."""
    from kamo.trap import Trap, Tweezer, solve as trap_solve
    from kamo.dd_solver.cloud import GridProfile, profile_from_kamo
    trap = Trap(Tweezer(waist=3e-6, wavelength_m=1064e-9), state=(4, 0, 0.5, 1, -1),
                B_gauss=520.583, B_direction=(0, 0, 1)).rescaled_to_frequency(1.0e3)
    tc = trap_solve(trap, N=500, mode="gp")
    prof = profile_from_kamo(tc)
    assert isinstance(prof, GridProfile)
    # the profile reproduces kamo's own moments and int n^2
    assert np.allclose(prof.sigma, np.asarray(tc.sigma), rtol=2e-3)
    assert abs(prof.density_squared_integral / tc.density_squared_integral - 1) < 1e-9
    # eta_eff from int n^2, NOT the Gaussian closed form
    eta = prof.eta_eff(op.wavelength)
    gauss_form = prof.peak_density * op.wavelength ** 3 / 2 ** 1.5
    assert 18 < eta < 21
    assert gauss_form < 0.95 * eta                     # the GP shape is flatter
    # the lab cloud is markedly less dense than the package's harmonic default
    assert eta < 0.8 * GaussianProfile.operating_point(500).eta_eff(op.wavelength)
    # sampling reproduces the moments, and the residual bias is the documented one
    pos = prof.sample_positions(60000, np.random.default_rng(0))
    assert np.allclose(pos.std(axis=0), prof.sigma, rtol=0.02)
    assert np.allclose(pos.mean(axis=0), prof.origin, atol=0.03 * prof.sigma)
    up = GridProfile(tc, upsample=2)
    assert abs(up.eta_eff(op.wavelength) / eta - 1) < 1e-6
    assert up.density_grid.min() >= 0.0                # sqrt-FFT refinement stays positive
    # and it solves
    cfg = sample_configuration(prof, theta=np.pi, seed=1, N=400)
    r = solve(cfg, op, "full", keep_matrices=False)
    assert r.checks.passed


def test_from_variational_axis_order_guard(op):
    """kamo.trap hands back principal-axis order (weak axis LAST); copying it
    verbatim would put the long axis along B."""
    from kamo.BEC_properties.variational import GaussianVariationalCloud
    import kamo.constants as kc
    omega = 2 * np.pi * np.array([1000.0, 986.0, 79.3])          # principal order
    cloud = GaussianVariationalCloud(500, omega, 10.96 * kc.a0)
    naive = GaussianProfile.from_variational(cloud)
    assert naive.sigma[2] > naive.sigma[0]                        # long axis along z: wrong
    axes = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    fixed = GaussianProfile.from_variational(cloud, axes=axes)
    assert fixed.sigma[0] > 3 * fixed.sigma[1]                    # long axis along the probe
    assert abs(fixed.sigma[0] - naive.sigma[2]) < 1e-12
    with pytest.raises(ValueError):
        GaussianProfile.from_variational(cloud, axes=np.full((3, 3), 0.577))


def test_pair_has_two_resonances(op):
    """A close pair is resonant at delta = +J through its bright mode AND at
    delta = -J through its dark one, whenever the pair axis has a component
    along the probe.

    This is what refutes the "one detuning sign only sees half the angular
    distribution" reading of the excess law (ensemble.excess_law), and it is
    the same structure that makes the excess an excited-POPULATION effect rather
    than a scattering one: the subradiant peak is ~200x taller in population."""
    k = op.k
    r = 0.435 / k                                     # the Condon radius

    def peaks(axis):
        pos = np.stack([np.zeros(3), np.asarray(axis, dtype=float) * r])
        J = scalar_couplings(pos, k, op.e_hat, "full")[0][0, 1]
        cfg = Configuration(pos, np.array([1, 1], dtype=np.int8))
        d = np.linspace(-12, 12, 2401)
        e = np.array([solve(cfg, OperatingPoint(op.linewidth_Hz, op.wavelength,
                                                delta_up=float(x), delta_dn=float(x)),
                            "full", keep_matrices=False, checks=False).excitation for x in d])
        loc = [i for i in range(1, len(d) - 1) if e[i] > e[i - 1] and e[i] > e[i + 1]]
        return J, [(d[i], e[i]) for i in loc]

    # axis perpendicular to k: only the bright mode is driven, at delta = J
    for axis in ([0, 0, 1.0], [0, 1.0, 0]):
        J, pk = peaks(axis)
        assert len(pk) == 1, (axis, pk)
        assert abs(pk[0][0] - J) < 0.3, (axis, J, pk)
    # axis along k: the dark mode is driven too, at delta = -J, and dominates
    J, pk = peaks([1.0, 0, 0])
    assert len(pk) == 2, pk
    lo, hi = sorted(pk, key=lambda t: t[1])
    assert abs(lo[0] - J) < 0.3 and abs(hi[0] + J) < 0.3, (J, pk)
    assert hi[1] > 50 * lo[1], pk                     # the subradiant peak dominates


def test_excess_law_angular_constant():
    """The published xi_circ is pi^2 <|B|> / (2 pi)^3 with <|B|> over the whole
    sphere, and each sign of B carries exactly half of that.  Recorded because
    the halving is the natural (but, per test_pair_has_two_resonances, wrong)
    explanation of why the measurement comes out at xi_circ / 2."""
    from scipy.integrate import quad
    B = lambda u: (3 * u ** 2 - 1) / 2            # noqa: E731
    kink = [1 / np.sqrt(3)]                        # B changes sign at the magic angle
    both, _ = quad(lambda u: abs(B(u)), 0, 1, points=kink)
    pos, _ = quad(lambda u: max(B(u), 0.0), 0, 1, points=kink)
    neg, _ = quad(lambda u: max(-B(u), 0.0), 0, 1, points=kink)
    assert abs(both - 2 / (3 * np.sqrt(3))) < 1e-10
    assert abs(pos - neg) < 1e-10 and abs(pos - both / 2) < 1e-10
    # xi = pi^2 <B_res> / (2 pi)^3
    assert abs(np.pi ** 2 * both / (8 * np.pi ** 3) - ensemble.XI_CIRC) < 1e-12
    assert abs(np.pi ** 2 * pos / (8 * np.pi ** 3) - ensemble.XI_CIRC / 2) < 1e-12
    assert ensemble.excess_law(10.0, 0.0) == 1 + ensemble.XI_CIRC * 10.0
