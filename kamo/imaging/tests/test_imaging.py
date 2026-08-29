"""Tests for kamo.imaging.

Two tiers, following kamo.scattering.tests:

INTERNAL     self-checking algebra and numerics -- optical theorem, splitting
             convergence, limits the method must reproduce by construction.
             These must always pass.
GROUND-TRUTH pinned against the operating point of the K-39 tweezer BEC these
             modules were written for (B = 520.58 G, N = 500, NA 0.42).  They
             encode physics conclusions, so a change here is a claim about the
             experiment, not a refactor.

Run: pytest kamo/imaging/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import kamo.constants as kc
from kamo.BEC_properties.variational import (CollapseError,
                                             GaussianVariationalCloud)
from kamo.imaging import ProbeBeam, Propagator, UniformMixture, readout
from kamo.imaging.grid import TransverseGrid
from kamo.imaging.response import TwoLevelResponse

# Operating point.  Field from the measured Raman/clock frequency, 119.4639 MHz.
B_GAUSS = 520.5830387285664
N_ATOMS = 500.0
A_UPUP_A0 = 11.333713034704147      # kamo Potassium39.get_scattering_length(1,-1,B)
A_DNDN_A0 = -73.69949836325802      # ...(1, 0, B)
NA = 0.42

#: Spin imbalance of the POLARIZED, LENSING cloud.  Re alpha ~ -delta, so the
#: RED-detuned state focuses; |up> is the lower-frequency transition and is
#: therefore BLUE-detuned and defocusing.  The cloud that lenses is |dn>, xi = -1.
#: Named rather than written inline so these physics tests say what they mean.
XI_LENS = -1.0

# |up> is the state whose transition to the excited state is LOWER in frequency,
# so a probe at the midpoint is BLUE of |up> (delta_up > 0) and RED of |dn>.
# kamo gives f(m_i=-1/2) < f(m_i=+1/2) by 110.22 MHz at this field, so |up> is
# m_i = -1/2.  Corrected 2026-08-28; these were previously swapped, which mirrors
# the lensing (Re alpha ~ -delta, so the RED-detuned state focuses).
G_UP, E_UP = (4, 0, 1 / 2, -1 / 2, -1 / 2), (4, 1, 3 / 2, -3 / 2, -1 / 2)
G_DN, E_DN = (4, 0, 1 / 2, -1 / 2, +1 / 2), (4, 1, 3 / 2, -3 / 2, +1 / 2)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def cloud():
    return GaussianVariationalCloud.from_tweezer(
        N=N_ATOMS, a_scattering=A_UPUP_A0 * kc.a0, f_radial_Hz=1.0e3, waist=3.0e-6)


@pytest.fixture(scope="module")
def atom():
    from kamo import Potassium39
    return Potassium39()


@pytest.fixture(scope="module")
def probe(atom):
    p = ProbeBeam.from_midpoint(atom, B_gauss=B_GAUSS,
                                ground_up=G_UP, excited_up=E_UP,
                                ground_dn=G_DN, excited_dn=E_DN)
    return ProbeBeam.from_offresonant_saturation(p, 0.001, which="up")


@pytest.fixture(scope="module")
def response(probe):
    return probe.response


# =========================================================== INTERNAL: response


class TestResponse:
    """Algebraic identities the atomic response must satisfy exactly."""

    def test_optical_theorem(self):
        r = TwoLevelResponse.from_transition_frequency(391.018e12, 6.005e6)
        for delta in (0.0, -18.36, +18.36, 3.0):
            for s in (0.0, 0.7, 3.0):
                lhs = r.k * r.polarizability(delta, s).imag / kc.epsilon_0
                assert lhs == pytest.approx(r.cross_section(delta, s), rel=1e-12)

    def test_sigma0_is_three_lambda_squared_over_two_pi(self):
        r = TwoLevelResponse.from_transition_frequency(391.018e12, 6.005e6)
        assert r.sigma0 == pytest.approx(3 * r.wavelength**2 / (2 * np.pi), rel=1e-14)
        # and 6 pi / k^2 is the same statement
        assert r.sigma0 == pytest.approx(6 * np.pi / r.k**2, rel=1e-14)

    def test_slice_operator_matches_closed_form(self, response):
        """chi/2 form is algebra, not new physics -- pin it before anything uses it."""
        delta, n, s, dx = -18.36, 1.0e19, 0.7, 1e-9
        got = response.slice_operator(n, delta, dx, s)
        want = np.exp(-(1 + 1j * delta) * response.sigma0 * n * dx
                      / (2 * (1 + delta**2 + s)))
        assert got == pytest.approx(want, rel=1e-12)

    def test_saturation_broadens_both_quadratures_together(self, response):
        """Re and Im alpha share one denominator, so their RATIO is s-independent."""
        delta = -18.36
        ratios = [(response.polarizability(delta, s).real
                   / response.polarizability(delta, s).imag) for s in (0.0, 1.0, 5.0)]
        assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)
        assert ratios[0] == pytest.approx(ratios[2], rel=1e-12)
        assert ratios[0] == pytest.approx(-delta, rel=1e-12)

    def test_red_detuning_converges(self, response):
        """delta < 0 -> Re alpha > 0 -> n_ref > 1 -> converging lens."""
        assert response.polarizability(-18.36).real > 0
        assert response.polarizability(+18.36).real < 0
        assert response.index_minus_one(
            response.susceptibility(1e19, -18.36)).real > 0

    def test_absorption_even_dispersion_odd(self, response):
        """The entire midpoint readout scheme rests on these two parities."""
        a_m, a_p = response.polarizability(-18.36), response.polarizability(+18.36)
        assert a_m.imag == pytest.approx(a_p.imag, rel=1e-14)      # even
        assert a_m.real == pytest.approx(-a_p.real, rel=1e-14)     # odd

    def test_offresonant_saturation_roundtrip(self, response):
        delta = -18.36
        s0 = response.saturation_from_offresonant(0.001, delta)
        assert (1 + s0 + delta**2) == pytest.approx(
            (1 + delta**2) * (1 + 0.001), rel=1e-12)

    def test_local_field_reduces_to_dilute(self):
        """Clausius-Mossotti -> chi/2 as chi -> 0."""
        dilute = TwoLevelResponse(766.7e-9, 6.005e6, local_field=False)
        lorentz = TwoLevelResponse(766.7e-9, 6.005e6, local_field=True)
        for chi in (1e-8, 1e-6):
            assert lorentz.index_minus_one(chi) == pytest.approx(
                dilute.index_minus_one(chi), rel=1e-5)

    def test_local_field_enhances(self):
        """At the operating density LL is a several-percent enhancement, not zero."""
        lorentz = TwoLevelResponse(766.7e-9, 6.005e6, local_field=True)
        dilute = TwoLevelResponse(766.7e-9, 6.005e6, local_field=False)
        chi = 0.26 + 0.0143j          # |chi| ~ 0.26 at n0 = 1.38e14 cm^-3
        enh = abs(lorentz.index_minus_one(chi) / dilute.index_minus_one(chi))
        assert 1.02 < enh < 1.25


# =========================================================== INTERNAL: cloud


class TestVariationalCloud:
    def test_noninteracting_limit(self):
        """As a -> 0 the ansatz must return the harmonic ground state exactly."""
        c = GaussianVariationalCloud.from_tweezer(
            N=1.0, a_scattering=0.0, f_radial_Hz=1.0e3, waist=3.0e-6)
        assert c.sigma == pytest.approx(c.sigma_noninteracting, rel=1e-6)

    def test_repulsive_swells_attractive_shrinks(self):
        kw = dict(N=500.0, f_radial_Hz=1.0e3, waist=3.0e-6)
        free = GaussianVariationalCloud.from_tweezer(a_scattering=0.0, **kw)
        rep = GaussianVariationalCloud.from_tweezer(a_scattering=50 * kc.a0, **kw)
        att = GaussianVariationalCloud.from_tweezer(a_scattering=-5 * kc.a0, **kw)
        assert np.all(rep.sigma > free.sigma)
        assert np.all(att.sigma < free.sigma)

    def test_thomas_fermi_limit(self):
        """At large chi the Gaussian mu approaches a FIXED multiple of Thomas-Fermi.

        Dropping the kinetic term, the stationarity condition gives
        ``lambda_i^2 u_i^2 / 2 = I`` for every axis, so ``prod(u) = (2I)^{3/2}``
        and ``I = (chi / (4 sqrt(pi)))^{2/5}``, whence ``mu = (7/2) I``.  Against
        ``mu_TF = (1/2)(15 chi)^{2/5}`` that is

            mu_gauss / mu_TF -> 7 (60 sqrt(pi))^{-2/5} = 1.0824...

        The Gaussian sits ABOVE Thomas-Fermi because a Gaussian is not a parabola:
        TF is the true minimizer of the kinetic-free functional, and any restricted
        family costs energy.  Convergence to 1 would mean the ansatz had somehow
        beaten the exact result -- so this constant, not unity, is the correct test.
        """
        ratio_exact = 7 * (60 * np.sqrt(np.pi)) ** (-2 / 5)
        assert ratio_exact == pytest.approx(1.0824, abs=1e-4)
        for N in (1e7, 1e9):
            c = GaussianVariationalCloud.from_tweezer(
                N=N, a_scattering=100 * kc.a0, f_radial_Hz=1.0e3, waist=3.0e-6)
            assert c.chi > 1e4
            assert c.chemical_potential / c.chemical_potential_tf == pytest.approx(
                ratio_exact, rel=1e-3)

    def test_chemical_potential_scales_as_chi_two_fifths(self):
        """The TF scaling exponent itself, independent of the prefactor above."""
        kw = dict(a_scattering=100 * kc.a0, f_radial_Hz=1.0e3, waist=3.0e-6)
        a = GaussianVariationalCloud.from_tweezer(N=1e7, **kw)
        b = GaussianVariationalCloud.from_tweezer(N=1e9, **kw)
        assert (b.chemical_potential / a.chemical_potential) == pytest.approx(
            100 ** 0.4, rel=1e-3)

    def test_peak_density_consistent_with_widths(self, cloud):
        w = cloud.widths
        assert cloud.peak_density == pytest.approx(
            cloud.N / (np.pi**1.5 * np.prod(w)), rel=1e-12)
        assert cloud.peak_column_density == pytest.approx(
            cloud.peak_density * np.sqrt(np.pi) * w[0], rel=1e-12)

    def test_density_integrates_to_N(self, cloud):
        w = cloud.widths
        ax = [np.linspace(-6 * wi, 6 * wi, 401) for wi in w]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        dV = np.prod([a[1] - a[0] for a in ax])
        assert cloud.density(X, Y, Z).sum() * dV == pytest.approx(cloud.N, rel=1e-4)

    def test_sigma_is_w_over_root_two(self, cloud):
        assert cloud.sigma == pytest.approx(cloud.widths / np.sqrt(2), rel=1e-14)

    def test_attractive_cloud_collapses(self):
        c = GaussianVariationalCloud.from_tweezer(
            N=N_ATOMS, a_scattering=A_DNDN_A0 * kc.a0,
            f_radial_Hz=1.0e3, waist=3.0e-6)
        assert c.collapsed
        with pytest.raises(CollapseError):
            _ = c.widths
        assert 50 < c.critical_atom_number() < 200

    def test_axial_ratio_from_beam_geometry(self):
        c = GaussianVariationalCloud.from_tweezer(
            N=1.0, a_scattering=0.0, f_radial_Hz=1.0e3,
            waist=3.0e-6, wavelength=1064e-9)
        assert c.omega[0] / c.omega[1] == pytest.approx(
            1064e-9 / (np.sqrt(2) * np.pi * 3.0e-6), rel=1e-12)

    def test_waist_and_axial_frequency_agree(self):
        a = GaussianVariationalCloud.from_tweezer(
            N=1.0, a_scattering=0.0, f_radial_Hz=1.0e3, waist=3.0e-6)
        b = GaussianVariationalCloud.from_tweezer(
            N=1.0, a_scattering=0.0, f_radial_Hz=1.0e3,
            f_axial_Hz=a.omega[0] / (2 * np.pi))
        assert a.omega == pytest.approx(b.omega, rel=1e-12)

    def test_requires_exactly_one_of_waist_or_axial(self):
        with pytest.raises(ValueError):
            GaussianVariationalCloud.from_tweezer(N=1., a_scattering=0., f_radial_Hz=1e3)
        with pytest.raises(ValueError):
            GaussianVariationalCloud.from_tweezer(
                N=1., a_scattering=0., f_radial_Hz=1e3, waist=3e-6, f_axial_Hz=80.)


# ============================================================ INTERNAL: grid


class TestGrid:
    def test_propagator_unitary_on_propagating_modes(self):
        g = TransverseGrid(64, 36e-6, 2 * np.pi / 766.7e-9)
        H = g.propagator(1e-6)
        assert np.abs(H[g.propagating]) == pytest.approx(1.0, abs=1e-12)

    def test_evanescent_modes_decay_forward(self):
        g = TransverseGrid(64, 2e-6, 2 * np.pi / 766.7e-9)   # small box -> evanescent
        assert (~g.propagating).any()
        H = g.propagator(1e-6)
        assert np.all(np.abs(H[~g.propagating]) < 1.0)

    def test_back_propagator_inverts_on_propagating_modes(self):
        g = TransverseGrid(64, 36e-6, 2 * np.pi / 766.7e-9)
        prod = g.propagator(2e-6) * g.back_propagator(2e-6)
        assert prod[g.propagating] == pytest.approx(1.0, abs=1e-12)

    def test_dipole_weight_normalized_over_sphere(self):
        """(3/16pi)(1+cos^2) integrates to 1 -- check on a direct sphere quadrature."""
        u, wu = np.polynomial.legendre.leggauss(200)
        total = ((3 / (16 * np.pi)) * (1 + u**2) * wu).sum() * 2 * np.pi
        assert total == pytest.approx(1.0, rel=1e-12)


# ======================================================= INTERNAL: propagation


@pytest.fixture(scope="module")
def prop(response, cloud):
    return Propagator.for_cloud(response, cloud, n_grid=384, n_slices=120)


class TestPropagation:
    def test_vacuum_is_transparent(self, response, cloud, prop):
        """Zero density must leave psi = 1 exactly (tests the vacuum reference)."""
        src = UniformMixture(cloud, response, ((1.0, -18.36),), density_scale=0.0)
        res = prop.propagate(src)
        assert np.abs(res.psi_exit - 1.0).max() < 1e-10

    def test_small_phase_limit_matches_born(self, response, cloud, prop, probe):
        """VALIDATION 1: the weak-phase far field IS the analytic Gaussian form factor.

        Pins the propagator, the dipole weighting and the q_x handling at once.
        """
        src = UniformMixture(cloud, response, probe.species(1.0), density_scale=0.002)
        res = prop.propagate(src, saturate=False)
        W = readout.far_field(res)
        Born = readout.born_far_field(res.grid, cloud.widths, cloud.N)
        m = Born > Born.max() * 1e-4                 # four decades of dynamic range
        ratio = (W[m] / W.max()) / (Born[m] / Born.max())
        assert np.median(ratio) == pytest.approx(1.0, abs=2e-3)
        assert np.abs(ratio - 1).max() < 2e-2

    def test_into_NA_converges(self, response, cloud, prop, probe):
        """VALIDATION 2: halving slices or grid must not move the collected fraction."""
        src = lambda: UniformMixture(cloud, response, probe.species(1.0))
        base = readout.into_NA(readout.far_field(
            prop.propagate(src(), s0_incident=probe.s0_incident)), prop.grid, NA)
        for kw in (dict(n_slices=60), dict(n_grid=192)):
            p2 = prop.rescaled(**kw)
            r2 = p2.propagate(src(), s0_incident=probe.s0_incident)
            assert readout.into_NA(readout.far_field(r2), p2.grid, NA) == \
                pytest.approx(base, rel=2e-3)

    def test_thin_cloud_recovers_thin_screen(self, response, cloud, probe):
        """VALIDATION 1c: squeezing the cloud at fixed COLUMN density must restore
        the thin-screen phase.  A coding error would leave the ratio pinned near
        0.5 regardless of cloud length; a real depth-of-field effect climbs to 1."""
        _, phi_thin = response.thin_screen(cloud.peak_column_density,
                                           probe.species(XI_LENS))
        ratios = []
        for f in (1.0, 0.3, 0.1):
            p = Propagator.for_cloud(response, cloud, n_grid=384, n_slices=120,
                                     wx_scale=f)
            r = p.propagate(UniformMixture(cloud, response, probe.species(XI_LENS),
                                           wx_scale=f), saturate=False)
            ratios.append(readout.recovered_phase(readout.refocus(r), r.grid)
                          / phi_thin)
        assert ratios[0] < ratios[1] < ratios[2]      # monotone towards 1
        assert ratios[0] < 0.65                        # long cloud: big deficit
        assert ratios[2] > 0.85                        # short cloud: nearly recovered

    def test_beer_law_in_the_extended_limit(self, response, cloud, probe):
        """A transversely EXTENDED absorbing cloud must reproduce Beer's law.

        Beer needs two things: no refraction, and no diffraction across the
        cloud's own transverse structure while the light is inside it.  The first
        is arranged exactly by the balanced mixture (Re alpha cancels); the second
        is the Fresnel number ``F = k w_perp^2 / L``, which the real cloud fails --
        it is 0.47, so the measured optical depth is about HALF the Beer value.
        Widening w_perp at fixed column density must drive the ratio to 1; if it
        did not, the propagator would be wrong rather than the cloud unusual.
        """
        from kamo.imaging.grid import TransverseGrid
        sp = probe.species(xi=0.0)            # pure absorber, no lens
        sigma = response.cross_section(probe.delta_up, probe.s0_incident)
        L = 2 * cloud.widths[0]

        ratios = []
        for f in (1.0, 4.0, 16.0):
            L_box = max(36e-6, 12 * cloud.widths[1] * f)
            grid = TransverseGrid(384, L_box, response.k)
            p = Propagator(response, grid, x_edge=3 * cloud.widths[0], n_slices=120)
            src = UniformMixture(cloud, response, sp, w_scale=[1., f, f],
                                 density_scale=f**2)
            psi = readout.refocus(p.propagate(src, s0_incident=probe.s0_incident))
            n_col_peak = src.peak_density * np.sqrt(np.pi) * src.widths[0]
            ratios.append(readout.optical_depth(psi) / (sigma * n_col_peak))

        assert ratios[0] == pytest.approx(0.51, abs=0.05)   # the real cloud: half
        assert ratios[1] > 0.97                              # extended: nearly Beer
        assert ratios[2] == pytest.approx(1.0, abs=0.01)     # extended: Beer
        assert ratios[0] < ratios[1] < ratios[2]

    def test_real_cloud_is_not_extended(self, response, cloud):
        """State the geometry that makes Beer's law inapplicable here."""
        F = response.k * cloud.widths[1] ** 2 / (2 * cloud.widths[0])
        assert F < 1.0                                       # NOT an extended cloud

    def test_forward_mode_scales_as_N_squared(self, response, cloud):
        """Coherent forward power ~ N^2; every other mode ~ N.

        <|sum exp(i q.r)|^2> = N + N(N-1)|f(q)|^2.  At q = 0 the form factor is 1
        and the two terms make N^2 (amplitude linear in N); outside the lobe |f|^2
        vanishes and only the N incoherent photons survive.  The forward mode
        therefore leads by a factor of order N, which is what lets a dispersive
        measurement work on a collection axis pointed at the dim lobe of the
        dipole pattern.
        """
        from kamo.imaging.farfield import Sky
        sky = Sky(cloud.widths, cloud.N, response.k)

        def nhat(deg):
            t = np.radians(deg)
            return np.array([np.cos(t), np.sin(t), 0.0])

        assert sky.form_factor_sq(nhat(0.0)) == pytest.approx(1.0, rel=1e-12)
        assert sky.form_factor_sq(nhat(40.0)) < 1e-5

        N = np.logspace(2, 5, 25)
        for deg, want in ((0.0, 2.0), (40.0, 1.0)):
            f2 = float(sky.form_factor_sq(nhat(deg)))
            P = N + N * (N - 1) * f2
            slope = np.polyfit(np.log(N), np.log(P), 1)[0]
            assert slope == pytest.approx(want, abs=0.02)

    def test_bpm_coherent_power_is_quadratic_in_density(self, response, cloud,
                                                        prop, probe):
        """The propagation carries only the COHERENT field, so P ~ N^2 exactly.

        Scaling the density at fixed widths is N at fixed geometry.  In the
        weak-phase limit the exit amplitude is linear in it, so the far-field power
        is quadratic.  At the operating point it is NOT: the departure is upward,
        about 20% -- a strong lens deflects more light out of the unscattered mode
        than a linear response would, so the scattered power runs ahead of N^2
        rather than saturating below it.
        """
        weak = []
        for sc in (0.001, 0.002, 0.005, 0.01):
            r = prop.propagate(UniformMixture(cloud, response, probe.species(XI_LENS),
                                              density_scale=sc), saturate=False)
            weak.append(float(readout.far_field(r).sum()) / sc**2)
        assert max(weak) / min(weak) < 1.02          # quadratic to better than 2%

        strong = prop.propagate(
            UniformMixture(cloud, response, probe.species(XI_LENS)), saturate=False)
        departure = float(readout.far_field(strong).sum()) / np.mean(weak)
        assert departure > 1.1                       # and it departs upward

    def test_no_wraparound(self, response, cloud, prop, probe):
        res = prop.propagate(UniformMixture(cloud, response, probe.species(1.0)),
                             s0_incident=probe.s0_incident)
        assert res.edge_power_fraction() < 1e-8

    def test_lensing_raises_local_saturation(self, response, cloud, prop, probe):
        """A converging cloud concentrates the probe above the incident intensity."""
        res = prop.propagate(UniformMixture(cloud, response, probe.species(XI_LENS)),
                             s0_incident=probe.s0_incident)
        assert res.s_peak > 3 * probe.s0_incident
        assert res.mean_intensity > 1.0

    def test_record_shapes(self, response, cloud, prop, probe):
        src = UniformMixture(cloud, response, probe.species(1.0))
        r = prop.propagate(src, record="plane")
        assert r.intensity_plane.shape == (prop.n_slices, prop.grid.n)
        r3 = prop.propagate(src, record="3d")
        nw = r3.window.stop - r3.window.start
        assert r3.intensity_3d.shape == (prop.n_slices, nw, nw)
        assert r3.intensity_3d.dtype == np.float32

    def test_record_does_not_change_the_field(self, response, cloud, prop, probe):
        src = UniformMixture(cloud, response, probe.species(1.0))
        a = prop.propagate(src, s0_incident=probe.s0_incident)
        b = prop.propagate(src, s0_incident=probe.s0_incident, record="3d")
        assert np.abs(a.psi_exit - b.psi_exit).max() < 1e-14


# =========================================================== INTERNAL: readout


class TestReadout:
    def test_refocus_is_identity_at_the_exit_plane(self, response, cloud, prop, probe):
        """Zero-distance refocus returns the exit field, up to evanescent content.

        It is not bit-exact: refocusing is a projection onto propagating modes
        (see TransverseGrid.back_propagator), so the evanescent part of
        ``psi_exit - 1`` is discarded.  That part carries ~1e-14 of the power here,
        i.e. ~1e-7 of the amplitude, which bounds the discrepancy below.
        """
        res = prop.propagate(UniformMixture(cloud, response, probe.species(1.0)))
        psi = readout.refocus(res, x_focus=res.x_edge)      # zero distance
        Ek = np.fft.fft2(res.psi_exit - 1.0)
        evanescent_amp = np.sqrt((np.abs(Ek[~res.grid.propagating]) ** 2).sum()
                                 / (np.abs(Ek) ** 2).sum())
        assert evanescent_amp < 1e-6
        assert np.abs(psi - res.psi_exit).max() < 1e-5

    def test_phase_contrast_weak_limit(self, response, cloud, probe):
        """With a pi/2 plate and small phi, I/I0 -> 1 + 2 phi."""
        p = Propagator.for_cloud(response, cloud, n_grid=256, n_slices=80)
        res = p.propagate(UniformMixture(cloud, response, probe.species(1.0),
                                         density_scale=0.01), saturate=False)
        psi = readout.refocus(res)
        img = readout.phase_contrast(psi, res.grid)
        phi = readout.recovered_phase(psi, res.grid)
        assert readout.signal_on_axis(img) == pytest.approx(2 * phi, rel=0.1)

    def test_phase_plate_with_zero_retardation_is_flat(self, response, cloud,
                                                       prop, probe):
        res = prop.propagate(UniformMixture(cloud, response, probe.species(1.0)))
        psi = readout.refocus(res)
        img = readout.phase_contrast(psi, res.grid, theta=0.0)
        assert img == pytest.approx(np.abs(psi) ** 2, abs=1e-12)

    def test_invert_signal_roundtrip(self):
        xi = np.linspace(-1, 1, 21)
        sig = 2.5 * xi                          # a monotone synthetic curve
        assert readout.invert_signal([-1.25, 0.0, 1.25], (xi, sig)) == \
            pytest.approx([-0.5, 0.0, 0.5], abs=1e-9)

    def test_invert_signal_refuses_folded_curve(self):
        xi = np.linspace(-1, 1, 21)
        sig = 1.0 - xi**2                       # folds: two xi per signal
        with pytest.raises(ValueError, match="not monotonic"):
            readout.invert_signal(0.5, (xi, sig))
        # strict=False falls back to the monotonic branch containing xi = 0, which
        # is the operating point: sig = 1 - xi^2 rises on xi < 0, so 0.5 -> -1/sqrt(2)
        got = readout.invert_signal(0.5, (xi, sig), strict=False)
        assert np.isfinite(got).all()
        assert float(got) == pytest.approx(-1 / np.sqrt(2), abs=0.05)


# ===================================================== INTERNAL: spin symmetry


class TestMidpointSymmetry:
    """The readout scheme's two load-bearing claims, stated as tests."""

    def test_midpoint_makes_detunings_opposite(self, probe):
        assert probe.at_midpoint
        assert probe.delta_dn == pytest.approx(-probe.delta_up, rel=1e-12)

    def test_optical_depth_is_spin_blind(self, response, cloud, probe):
        Ds = [response.thin_screen(cloud.peak_column_density, probe.species(xi))[0]
              for xi in (-1.0, -0.3, 0.0, 0.5, 1.0)]
        assert Ds == pytest.approx([Ds[0]] * len(Ds), rel=1e-12)

    def test_phase_is_proportional_to_imbalance(self, response, cloud, probe):
        xis = np.array([-1.0, -0.5, 0.0, 0.25, 1.0])
        phis = np.array([response.thin_screen(cloud.peak_column_density,
                                              probe.species(x))[1] for x in xis])
        assert phis[2] == pytest.approx(0.0, abs=1e-14)     # balanced -> no phase
        assert phis == pytest.approx(xis * phis[-1], rel=1e-12)

    def test_balanced_cloud_does_not_lens(self, response, cloud, probe):
        """Re alpha cancels at 50/50, so the probe passes without concentrating."""
        p = Propagator.for_cloud(response, cloud, n_grid=256, n_slices=80)
        bal = p.propagate(UniformMixture(cloud, response, probe.species(0.0)),
                          s0_incident=probe.s0_incident)
        pol = p.propagate(UniformMixture(cloud, response, probe.species(XI_LENS)),
                          s0_incident=probe.s0_incident)
        # A balanced cloud is a pure ABSORBER, so it cannot concentrate the probe.
        # It is not perfectly flat either: a Gaussian absorbing mask diffracts, and
        # the shadow overshoots by ~2% -- an amplitude effect, not refraction.
        assert bal.s_peak / probe.s0_incident == pytest.approx(1.0, abs=0.05)
        assert bal.mean_intensity < 1.0            # net absorption, no focusing
        assert pol.s_peak > 3 * bal.s_peak          # the polarized cloud DOES lens

        # The REFRACTIVE phase cancels exactly (thin_screen gives 0 to machine
        # precision, tested above).  The propagated phase does not quite: a pure
        # absorber still modulates amplitude, and that modulation diffracts, which
        # carries phase of its own.  That residual is the floor on how well phase
        # contrast can read "balanced" -- here 0.05% of the polarized signal, i.e.
        # far below the 1/sqrt(N) ~ 4.5% projection noise it has to resolve.
        phi_bal = readout.recovered_phase(readout.refocus(bal), bal.grid)
        phi_pol = readout.recovered_phase(readout.refocus(pol), pol.grid)
        assert abs(phi_bal) < 0.01
        assert abs(phi_bal / phi_pol) < 1e-3
        assert abs(phi_bal) < abs(phi_pol) / np.sqrt(N_ATOMS)   # under shot noise


# ============================================================== GROUND TRUTH


class TestOperatingPoint:
    """Pinned against the K-39 tweezer BEC at B = 520.58 G, N = 500.

    These encode physics conclusions.  Changing them is a claim about the
    experiment, not a refactor.
    """

    def test_cloud_widths(self, cloud):
        assert cloud.widths[0] * 1e6 == pytest.approx(2.393, abs=0.01)
        assert cloud.widths[1] * 1e6 == pytest.approx(0.522, abs=0.01)
        assert cloud.peak_density * 1e-6 == pytest.approx(1.379e14, rel=0.01)
        assert cloud.peak_column_density * 1e-4 == pytest.approx(5.85e10, rel=0.01)

    def test_probe_geometry(self, probe):
        assert probe.splitting_Hz / 1e6 == pytest.approx(110.2, abs=0.5)
        assert probe.response.sigma0 == pytest.approx(2.8066e-13, rel=1e-3)
        # |up> is the LOWER-frequency transition, so a probe at the midpoint is
        # BLUE of |up> and RED of |dn>.  This is the sign that fixes which state
        # lenses, so assert the convention itself, not just its magnitude.
        assert probe.delta_up == pytest.approx(+18.36, abs=0.05)
        assert probe.delta_dn == pytest.approx(-18.36, abs=0.05)
        assert probe.f_up < probe.f_dn
        # ... and therefore the differential shift nu_dn - nu_up is NEGATIVE
        assert probe.with_saturation(0.3).differential_light_shift_Hz() < 0

    def test_cloud_is_a_strong_phase_object(self, response, cloud, probe):
        D, phi = response.thin_screen(cloud.peak_column_density, probe.species(XI_LENS))
        assert D == pytest.approx(0.486, abs=0.01)      # optically THIN
        assert phi == pytest.approx(4.46, abs=0.05)     # but a strong phase object
        assert abs(phi) > 4                              # far outside weak-phase

    def test_depth_of_field_costs_about_forty_percent(self, response, cloud, probe):
        """A focused, aberration-free system recovers ~58% of the projected phase."""
        p = Propagator.for_cloud(response, cloud, n_grid=384, n_slices=120)
        res = p.propagate(UniformMixture(cloud, response, probe.species(XI_LENS)),
                          s0_incident=probe.s0_incident)
        _, phi_thin = response.thin_screen(cloud.peak_column_density,
                                           probe.species(XI_LENS))
        recovered = readout.recovered_phase(readout.refocus(res), res.grid)
        assert recovered / phi_thin == pytest.approx(0.58, abs=0.03)

    def test_collection_efficiency_survives_the_lensing(self, response, cloud, probe):
        """Born gets the collected fraction right even though it gets phi wrong."""
        p = Propagator.for_cloud(response, cloud, n_grid=384, n_slices=120)
        res = p.propagate(UniformMixture(cloud, response, probe.species(XI_LENS)),
                          s0_incident=probe.s0_incident)
        eta = readout.into_NA(readout.far_field(res), res.grid, NA)
        assert eta == pytest.approx(0.979, abs=0.005)

    def test_probe_is_concentrated_inside_the_cloud(self, response, cloud, probe):
        p = Propagator.for_cloud(response, cloud, n_grid=384, n_slices=120)
        res = p.propagate(UniformMixture(cloud, response, probe.species(XI_LENS)),
                          s0_incident=probe.s0_incident)
        assert res.s_peak / probe.s0_incident == pytest.approx(5.75, rel=0.1)
        assert res.s_peak == pytest.approx(1.94, rel=0.1)
