"""N-atom microscopic coupled-dipole model and its continuum baseline.

This is the companion to the pair model.  Its jobs are:

1. say where the two-body picture stops being valid, by counting how many
   neighbours sit inside the Condon surface;
2. give the collective (cooperative) line shift and broadening that renormalise the
   detuning seen by a pair;
3. expose the eigenvalue spectrum, i.e. the super- and subradiant decay rates.

Because the transition is closed and every atom shares one dipole orientation, the
vector problem collapses onto an N x N scalar matrix built from the same Green
function contraction used for the pair potentials.

The continuum baseline is the Lorentz-Lorenz local-field shift, which for a
two-level medium is  Delta_LL = -pi (rho / k^3) Gamma.  At 1e14 cm^-3 on the K-39 D2
line rho/k^3 = 0.18, so the shift is about -0.57 linewidths -- not a small
correction inside a +/-10 linewidth window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .green_tensor import dd_coupling_scalar, dipole_projection


def lorentz_lorenz_shift_Hz(density, transition):
    """Lorentz-Lorenz local-field resonance shift, in Hz.

    Delta_LL = -pi (rho / k^3) Gamma, from requiring the Clausius-Mossotti
    denominator 1 - rho alpha / 3 to vanish for a two-level polarisability.
    """
    rho = np.asarray(density, dtype=float)
    return -np.pi * rho / transition.k ** 3 * transition.linewidth_Hz


class CoupledDipole:
    """Linear-optics coupled-dipole model for N atoms at a given density.

    Atoms are placed uniformly in a box (or sphere) sized to hold ``n_atoms`` at
    ``density``.  Sampling from the full Thomas-Fermi profile instead would
    reproduce the cloud *shape* but not its density -- a few hundred sampled atoms
    spread over a 10^5-atom condensate sit orders of magnitude further apart than
    real neighbours -- so the microscopic model is run on a representative volume
    at the local density.  Use :meth:`from_cloud` to take that density from a
    BECCloud.

    Parameters
    ----------
    transition : CyclingTransition
    density : float
        Number density in m^-3.
    n_atoms : int
        Number of atoms.  The matrix is dense and N x N, so a few thousand is the
        practical ceiling.
    geometry : {"box", "sphere"}
    k_hat : 3-sequence
        Beam direction, used for the drive phase pattern.
    cloud : BECCloud, optional
        Only used for the pair correlation g2 if one is supplied.
    rng : numpy Generator, optional
    """

    def __init__(self, transition, density, n_atoms: int = 1500,
                 geometry: str = "box", k_hat=(1.0, 0.0, 0.0), cloud=None,
                 rng=None):
        self.transition = transition
        self.density = float(density)
        self.n_atoms = int(n_atoms)
        self.geometry = geometry
        self.k_hat = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
        self.cloud = cloud
        self.rng = np.random.default_rng() if rng is None else rng
        self.positions = self._sample()
        self._matrix = None

    @classmethod
    def from_cloud(cls, transition, cloud, n_atoms: int = 1500, use_peak=True,
                   **kwargs):
        """Build at the peak (or density-weighted mean) density of a cloud."""
        density = cloud.peak_density if use_peak else cloud.mean_density
        return cls(transition, density, n_atoms=n_atoms, cloud=cloud, **kwargs)

    def _sample(self):
        volume = self.n_atoms / self.density
        if self.geometry == "box":
            L = volume ** (1.0 / 3.0)
            self.extent = L
            return self.rng.uniform(-0.5 * L, 0.5 * L, size=(self.n_atoms, 3))
        if self.geometry == "sphere":
            radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            self.extent = 2.0 * radius
            pts = self.rng.normal(size=(self.n_atoms, 3))
            pts /= np.linalg.norm(pts, axis=1)[:, None]
            return pts * radius * self.rng.uniform(size=(self.n_atoms, 1)) ** (1.0 / 3.0)
        raise ValueError("geometry must be box or sphere.")

    # ---------------------------------------------------------------- matrix
    def couplings(self):
        """Pairwise (J, Gamma_12) matrices in rad/s, with zero diagonal."""
        pos = self.positions
        diff = pos[:, None, :] - pos[None, :, :]
        R = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(R, np.inf)                     # avoid the self term
        with np.errstate(invalid="ignore"):
            r_hat = diff / R[..., None]
        p = dipole_projection(self.transition.d_hat, r_hat)
        with np.errstate(divide="ignore", invalid="ignore"):
            J, G12 = dd_coupling_scalar(self.transition.k * R, p,
                                        self.transition.Gamma)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        G12 = np.nan_to_num(G12, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(J, 0.0)
        np.fill_diagonal(G12, 0.0)
        return J, G12

    def matrix(self):
        """Non-Hermitian coupling matrix M = J - i Gamma_12 / 2, with the single-atom
        -i Gamma / 2 on the diagonal.  Cached."""
        if self._matrix is None:
            J, G12 = self.couplings()
            M = J - 0.5j * G12
            np.fill_diagonal(M, -0.5j * self.transition.Gamma)
            self._matrix = M
        return self._matrix

    def spectrum(self):
        """Collective mode shifts and decay rates.

        Returns
        -------
        (shift_Hz, decay_rate) : each of length n_atoms.  ``decay_rate`` is in
        rad/s; values above Gamma are superradiant, below are subradiant.
        """
        ev = np.linalg.eigvals(self.matrix())
        return np.real(ev) / (2.0 * np.pi), -2.0 * np.imag(ev)

    def steady_state(self, detuning_Hz, intensity=None):
        """Weak-drive steady-state dipole amplitudes.

        Solves  [ (i Delta - Gamma/2) delta_jl + i M_jl (off-diagonal) ] b = i Omega/2
        with the beam phase exp(i k . r_j).  Valid in the linear-optics limit; the
        saturating physics lives in the pair model, this is a diagnostic.
        """
        Gamma = self.transition.Gamma
        Delta = 2.0 * np.pi * float(detuning_Hz)
        J, G12 = self.couplings()
        A = 1.0j * J - 0.5 * G12
        np.fill_diagonal(A, 1.0j * Delta - 0.5 * Gamma)
        Omega = 1.0 if intensity is None else self.transition.rabi(intensity)
        phase = np.exp(1.0j * self.transition.k
                       * (self.positions @ self.k_hat))
        return np.linalg.solve(A, 1.0j * 0.5 * Omega * phase)

    def lineshape(self, detunings_Hz, intensity=None):
        """Total scattered intensity (arbitrary units) versus detuning."""
        det = np.atleast_1d(np.asarray(detunings_Hz, dtype=float))
        return np.array([float(np.sum(np.abs(self.steady_state(d, intensity)) ** 2))
                         for d in det])

    def collective_shift_Hz(self, detunings_Hz, intensity=None):
        """Peak of the computed lineshape, i.e. the cooperative line shift in Hz.

        Compare with lorentz_lorenz_shift_Hz for the continuum baseline.
        """
        det = np.atleast_1d(np.asarray(detunings_Hz, dtype=float))
        y = self.lineshape(det, intensity)
        return float(det[int(np.argmax(y))])

    # ----------------------------------------------------------- diagnostics
    def neighbour_statistics(self, detuning_Hz):
        """How badly the two-body picture is violated at this detuning.

        An atom is counted as being inside a Condon surface when the dipole-dipole
        shift to a neighbour exceeds the laser detuning, so that the pair is at or
        beyond resonance.

        Returns
        -------
        dict with
            mean_neighbours       -- mean number of such neighbours per atom
            fraction_with_one     -- fraction of atoms with at least one
            fraction_with_two     -- fraction with at least two (pair picture fails)
        """
        J, _ = self.couplings()
        delta = 2.0 * np.pi * abs(float(detuning_Hz))
        inside = np.abs(J) >= delta
        np.fill_diagonal(inside, False)
        counts = inside.sum(axis=1)
        return {
            "mean_neighbours": float(counts.mean()),
            "fraction_with_one": float(np.mean(counts >= 1)),
            "fraction_with_two": float(np.mean(counts >= 2)),
        }

    def summary(self, detuning_Hz=-6e6) -> str:
        """Diagnostic dump at one detuning."""
        stats = self.neighbour_statistics(detuning_Hz)
        shifts, decays = self.spectrum()
        Gamma = self.transition.Gamma
        ll = lorentz_lorenz_shift_Hz(self.density, self.transition)
        return "\n".join([
            "CoupledDipole  N = %d   density = %.3e cm^-3   box = %.2f um"
            % (self.n_atoms, self.density / 1e6, self.extent * 1e6),
            "  detuning              = %.3f MHz" % (detuning_Hz / 1e6),
            "  mean neighbours in R_C= %.3f" % stats["mean_neighbours"],
            "  fraction with >= 1    = %.3f" % stats["fraction_with_one"],
            "  fraction with >= 2    = %.3f  (pair picture fails above ~0.1)"
            % stats["fraction_with_two"],
            "  max decay rate        = %.2f Gamma (superradiant)"
            % (decays.max() / Gamma),
            "  min decay rate        = %.2e Gamma (subradiant)"
            % (decays.min() / Gamma),
            "  Lorentz-Lorenz shift  = %.3f MHz (%.2f linewidths)"
            % (ll / 1e6, ll / self.transition.linewidth_Hz),
        ])

    def __repr__(self) -> str:
        return ("CoupledDipole(n_atoms=%d, density=%.3e cm^-3)"
                % (self.n_atoms, self.density / 1e6))
