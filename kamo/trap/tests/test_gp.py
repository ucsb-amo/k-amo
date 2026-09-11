"""Tests for kamo.trap.gross_pitaevskii.

INTERNAL      the a -> 0 limit (harmonic exactly; the real tweezer against the
              non-interacting solver); the variational upper bound in a harmonic
              trap; mu = dE_tot/dN on a fixed grid; the virial theorem; the
              polish is load-bearing (the split-step fixed point alone misses the
              residual gate); collapse for attractive interactions; single
              precision needs an opt-in; CPU/GPU agreement when a GPU is present.
GROUND-TRUTH  the K-team operating point (1064 nm, 3 um, 1 kHz tweezer with
              gravity, N = 500, a = 11.33 a0): the real anharmonic condensate
              against the Gaussian-variational one kamo used before -- lower mu,
              every width larger, the radial degeneracy broken, and a markedly
              lower peak and peak-column density.

Run: pytest kamo/trap/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, hbar

import kamo.constants as kc
from kamo.BEC_properties.variational import CollapseError, GaussianVariationalCloud
from kamo.trap.beams import Tweezer
from kamo.trap.gross_pitaevskii import GrossPitaevskiiSolver
from kamo.trap.noninteracting import NonInteractingSolver
from kamo.trap.trap import HarmonicTrap, Trap

M = 38.963706 * atomic_mass
W0, LAM = 3.0e-6, 1064e-9
ALPHA = 599.3005266591443 * kc.convert_polarizability_au_to_SI
A_OP = 11.33 * kc.a0
F_H = (300.0, 400.0, 500.0)                 # mildly anisotropic: well conditioned
N_H, A_H = 300.0, 10 * kc.a0                # weakly interacting: a 32^3 grid, well under a second


def harmonic(f_Hz=F_H):
    w = 2 * np.pi * np.asarray(f_Hz, dtype=float)
    return HarmonicTrap((0.0, 0.0, 0.0), 0.0, np.diag(M * w ** 2), M)


def tweezer_trap():
    t = Tweezer.from_trap_frequency(1e3, polarizability_SI=ALPHA, mass=M, waist=W0,
                                    wavelength_m=LAM)
    return Trap(t, polarizability_SI=ALPHA, mass=M, gravity=True)


def solve(trap, N, a, **kw):
    kw.setdefault("points_per_scale", 2.5)
    return GrossPitaevskiiSolver(trap, a_scattering=a, **kw).solve(N)


@pytest.fixture(scope="module")
def interacting():
    return solve(harmonic(), N_H, A_H, points_per_scale=3.0)


class TestLimits:
    def test_noninteracting_limit_is_the_oscillator(self):
        cloud = solve(harmonic(), N_H, 0.0, points_per_scale=3.0)
        w = 2 * np.pi * np.array(F_H)
        assert cloud.chemical_potential_offset == pytest.approx(0.5 * hbar * w.sum(), rel=1e-8)
        assert cloud.info.residual < 1e-8 and cloud.info.edge_fraction < 1e-8
        assert cloud.atom_number_error == pytest.approx(0.0, abs=1e-12)
        assert cloud.sigma == pytest.approx(np.sqrt(hbar / (2 * M * w)), rel=1e-6)

    def test_a_to_zero_matches_the_noninteracting_solver(self):
        trap = tweezer_trap()
        gp = solve(trap, 500.0, 0.0)
        ni = NonInteractingSolver(trap).solve()
        assert gp.chemical_potential_offset == pytest.approx(ni.info.zero_point_J, rel=1e-6)
        assert gp.centroid == pytest.approx(ni.centroid, abs=1e-4 * W0)

    def test_variational_upper_bound(self, interacting):
        """In a harmonic trap the Gaussian ansatz is a trial function of the same
        functional, so E_GP <= E_variational; and interactions cost energy."""
        var = GaussianVariationalCloud(N_H, 2 * np.pi * np.array(F_H), A_H, mass=M)
        E_gp = interacting.energy_per_atom - interacting.V_min
        assert E_gp <= var.energy_per_atom
        assert E_gp > 0.5 * hbar * 2 * np.pi * sum(F_H)
        assert interacting.info.residual < 1e-8

    def test_mu_is_dE_dN(self, interacting):
        dN = 20.0
        grid = interacting.grid                       # same grid: no discretization change
        up, dn = (solve(harmonic(), N_H + s * dN, A_H, grid=grid) for s in (+1, -1))
        E = lambda c: c.N * (c.energy_per_atom - c.V_min)
        assert (E(up) - E(dn)) / (2 * dN) == pytest.approx(interacting.chemical_potential_offset,
                                                           rel=1e-5)

    def test_virial_theorem(self, interacting):
        assert abs(interacting.info.virial) < 1e-5

    def test_polish_is_load_bearing(self, interacting):
        """The split-step fixed point is O(dtau^2) off: without the polish the
        residual gate is missed and mu is measurably wrong."""
        rough = solve(harmonic(), N_H, A_H, points_per_scale=3.0, max_polish=0, strict=False)
        assert rough.info.residual > 1e-6
        assert abs(rough.chemical_potential - interacting.chemical_potential) > \
            1e-7 * interacting.chemical_potential_offset


class TestAttractive:
    def test_collapse_above_the_critical_number(self):
        a = -20 * kc.a0
        n_crit = GaussianVariationalCloud(1.0, 2 * np.pi * np.array(F_H), a,
                                          mass=M).critical_atom_number()
        with pytest.raises(CollapseError):
            solve(harmonic(), 3 * n_crit, a)
        cloud = solve(harmonic(), 0.4 * n_crit, a)
        w = 2 * np.pi * np.array(F_H)
        assert np.all(cloud.sigma < np.sqrt(hbar / (2 * M * w)))   # attraction squeezes
        assert cloud.info.residual < 1e-8


class TestBackend:
    def test_single_precision_needs_an_opt_in(self):
        with pytest.raises(ValueError, match="allow_single"):
            GrossPitaevskiiSolver(harmonic(), a_scattering=0.0, precision="single")

    def test_gpu_matches_cpu(self, interacting):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        gpu = solve(harmonic(), N_H, A_H, points_per_scale=3.0, backend="gpu")
        assert gpu.chemical_potential_offset == pytest.approx(
            interacting.chemical_potential_offset, rel=1e-9)


# ----------------------------------------------------------- GROUND-TRUTH

@pytest.fixture(scope="module")
def operating_point():
    trap = tweezer_trap()
    cloud = solve(trap, 500.0, A_OP, points_per_scale=3.0)
    var = GaussianVariationalCloud(500.0, trap.trap_frequencies().omega, A_OP, mass=M)
    return cloud, var


class TestOperatingPoint:
    def test_converged(self, operating_point):
        cloud, _ = operating_point
        assert cloud.info.residual < 1e-8 and cloud.info.edge_fraction < 1e-8
        assert abs(cloud.info.virial) < 1e-4
        assert not cloud.thomas_fermi_valid

    def test_softer_well_lower_mu_larger_cloud(self, operating_point):
        cloud, var = operating_point
        ratio = cloud.chemical_potential_offset / var.chemical_potential
        assert 0.93 < ratio < 0.97
        axes = cloud.trap.trap_frequencies().axes
        var_lab = np.sqrt((axes ** 2).T @ var.sigma ** 2)
        assert np.all(cloud.sigma > var_lab)
        assert cloud.sigma[2] > cloud.sigma[1]                       # gravity breaks y/z
        assert cloud.centroid[2] < cloud.trap.minimum().position[2]

    def test_the_imaging_systematic(self, operating_point):
        """The headline: the real condensate is markedly less dense at its peak,
        and lower in peak column density, than the Gaussian-variational cloud of
        the harmonic expansion -- a direct systematic in any column-density
        (imaging-phase) calculation built on the latter."""
        cloud, var = operating_point
        axes = cloud.trap.trap_frequencies().axes
        w_lab = np.sqrt((axes ** 2).T @ var.widths ** 2)
        column_var = cloud.N / (np.pi * w_lab[1] * w_lab[2])          # along lab x
        assert 0.75 < cloud.peak_density / var.peak_density < 0.85
        assert 0.88 < cloud.peak_column_density / column_var < 0.95
