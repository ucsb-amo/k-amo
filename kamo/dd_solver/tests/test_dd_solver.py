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
    expect = -0.5 * r.Omega / (op.detunings(cfg.spins) + 0.5j)
    assert np.max(np.abs(r.beta - expect)) < 1e-15
    assert np.max(np.abs(independent_solution(cfg, op) - expect)) < 1e-15


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
    """Near-field excess, excitation(full)/excitation(far), at the specification's
    N = 1000 profile.  Calibrated 2026-09-16: the law holds for the average over the
    two detuning signs; the red-detuned (guiding) side sits above it and the
    blue side below, because close pairs live at the cloud centre where the
    coherent intensity is 1.4x / 0.5x the incident one."""
    prof = GaussianProfile.spec_reference(1000)
    eta = prof.eta_eff(op.wavelength)
    for th in (0.0, np.pi / 4):
        ex = {}
        for theta in (th, np.pi - th):
            e = ensemble.run_ensemble(prof, op, theta, 40, seed0=0,
                                      variants=("full", "far", "independent"), n_jobs=N_JOBS)
            assert e.checks_passed()
            ex[theta] = (e.excitation("full") - e.excitation("far")) / e.excitation("independent")
        law = XI_C * eta * (1 - 0.5 * np.sin(th) ** 2)
        both = 0.5 * (ex[th] + ex[np.pi - th])           # same seeds: pair the signs
        # the MEAN is dominated by rare near-dark resonant pairs (sem ~ 30% at 40
        # configurations); the trimmed mean is the stable statistic (stats.py)
        assert abs(stats.trimmed_mean(both, 0.1) / law - 1) < 0.4
        assert np.median(ex[np.pi - th]) > np.median(ex[th])      # red side above blue


def test_T15_S4_reciprocity(res300):
    assert res300.checks.reciprocity < 1e-13


def test_T16_S2_positivity(res300, op, prof500):
    assert res300.checks.positivity_min_eig > -1e-10
    assert abs(res300.checks.decay_sum - 1.0) < 1e-12
    # the far-field-only kernel is NOT a passive medium: document, do not hide
    cfg = sample_configuration(prof500, theta=0.0, seed=6, N=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SanityWarning)
        r = solve(cfg, op, "far", positivity=True, warn=False)
    assert r.checks.positivity_min_eig < -1e-3


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
