"""The non-interacting ground state on the real (anharmonic) trap potential.

The harmonic ground state is what every existing kamo cloud assumes.  The real
well is softer: for the 3 um / 1 kHz tweezer its levels sit below the harmonic
ladder, its widths are larger, and gravity sags it *and* breaks the radial
degeneracy.  This solver finds the actual ground state (and the lowest excited
states) in 3D.

Method
------
A uniform midpoint :class:`~kamo.trap.grid.TrapGrid` on the lab axes (x stays the
imaging axis), sized from the harmonic widths along each lab axis.  The
Hamiltonian is the tensor-product sinc-DVR one -- the Colbert-Miller kinetic
matrix of :mod:`kamo.trap.dvr` along each axis plus the diagonal potential --
applied matrix-free as three ``tensordot`` contractions, so it handles
non-separable potentials (a Gaussian beam is a product, not a sum), crossed
traps and gravity alike.  The lowest ``n_states`` eigenpairs come from LOBPCG
with an FFT kinetic preconditioner, seeded with the harmonic ground state and
its first excitations along the principal axes.

Outside the basin of the minimum (the connected region below the escape
energy) the potential is flattened to the escape energy, so the box corner that
gravity pulls downhill cannot host a lower, spurious state.

Diagnostics (:class:`NonInteractingResult`): the 3D energies, the gap
``E1 - E0`` (what the GP solver's step count scales with), the separability
index ``max |V - V_separable| / hbar omega_max`` over the cloud, the residual,
and the eigenfunctions themselves -- the finite-temperature seam: a thermal
density is ``sum_n exp(-E_n / kT) |psi_n|^2 / Z``, and these are the ``psi_n``.
:meth:`NonInteractingSolver.axis_spectra` adds the 1D sinc-DVR spectrum along
each principal axis (bound-state count, anharmonicity, the escape lip) -- the
gaussian_well notebook, per axis, on demand.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg

import kamo.constants as kc

from . import dvr
from .cloud import ConvergenceError, TrapCloud, TrapTooShallowError
from .grid import TrapGrid, basin_mask


@dataclass
class NonInteractingResult:
    """Diagnostics of a non-interacting solve."""

    energies_3d: np.ndarray       #: lowest eigenvalues, ascending (J)
    psi_3d: np.ndarray            #: (n_states, nx, ny, nz), each normalized to 1 (1/m^3/2)
    gap_J: float                  #: E1 - E0 (J); NaN with one state
    separability_index: float     #: max |V - V_sep| / hbar omega_max where the cloud is
    residual: float               #: max ||H psi - E psi|| / hbar omega_max over the states
    V_min_J: float                #: potential at the minimum (J)
    V_escape_J: float             #: escape energy the basin is cut at (J)
    harmonic_zero_point_J: float  #: sum hbar omega_i / 2 at the minimum (J)

    @property
    def zero_point_J(self) -> float:
        """``E0 - V_min`` (J)."""
        return float(self.energies_3d[0] - self.V_min_J)

    @property
    def anharmonicity(self) -> float:
        """``(E0 - V_min) / (sum hbar omega_i / 2) - 1``: negative for a softer well."""
        return self.zero_point_J / self.harmonic_zero_point_J - 1.0


def _axis_length_scale(trap, e, tf) -> float:
    """The natural width of the trap along ``e``: waist across a beam, Rayleigh
    range along it; twice the oscillator length for a harmonic trap."""
    beams = getattr(trap, "beams", None) or ()
    if beams:
        L = 0.0
        for b in beams:
            c = min(1.0, abs(float(e @ b.propagation_direction)))
            L = max(L, c * max(b.rayleigh_range_u, b.rayleigh_range_v)
                    + np.sqrt(1.0 - c * c) * max(b.waist_u, b.waist_v))
        return L
    k = int(np.argmax(np.abs(tf.axes @ e)))
    return 2.0 * float(np.sqrt(kc.hbar / (tf.mass * tf.omega[k])))


class NonInteractingSolver:
    """Ideal-gas ground state of ``trap`` (a Trap or HarmonicTrap).

    Parameters
    ----------
    trap : Trap or HarmonicTrap
    n_per_axis : int or (3,)
        Grid points per lab axis (default 40).
    n_widths : float
        Box half-width in harmonic ground-state widths (x1.1 for the softer well).
    n_states : int
        Eigenpairs to compute (default 4: the ground state and one excitation
        per principal axis).
    tol, maxiter
        LOBPCG stopping criteria (residual in units of hbar omega_max).
    """

    def __init__(self, trap, *, n_per_axis=40, n_widths: float = 7.0, n_states: int = 4,
                 tol: float = 1e-8, maxiter: int = 3000):
        self.trap = trap
        self.n_per_axis = n_per_axis
        self.n_widths = float(n_widths)
        self.n_states = int(n_states)
        if self.n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.tol = float(tol)
        self.maxiter = int(maxiter)

    # ------------------------------------------------------------ helpers
    def _bound(self):
        mn = self.trap.minimum()
        if not mn.converged:
            raise TrapTooShallowError("the trap has no minimum to hold a cloud")
        tf = self.trap.trap_frequencies()
        if not tf.is_bound:
            raise TrapTooShallowError("the trap has a negative curvature at its minimum")
        return mn, tf

    def _seed(self, grid: TrapGrid, r0, tf, sig_k) -> np.ndarray:
        D = (grid.X - r0[0], grid.Y - r0[1], grid.Z - r0[2])
        p = [tf.axes[k, 0] * D[0] + tf.axes[k, 1] * D[1] + tf.axes[k, 2] * D[2]
             for k in range(3)]
        gauss = np.exp(-sum(p[k] ** 2 / (4.0 * sig_k[k] ** 2) for k in range(3)))
        gauss = np.broadcast_to(gauss, grid.shape)
        cols = [gauss]
        for k in np.argsort(sig_k)[::-1]:                  # softest axis first
            cols.append(np.broadcast_to(p[k] / sig_k[k] * gauss, grid.shape))
        rng = np.random.default_rng(0)
        while len(cols) < self.n_states:
            cols.append(gauss * rng.normal(size=grid.shape))
        return np.stack([c.reshape(-1) for c in cols[: self.n_states]], axis=1)

    # -------------------------------------------------------------- solve
    def solve(self, N: float = 1.0) -> TrapCloud:
        """The ground state, with ``N`` atoms in it (default 1: the single-particle
        density)."""
        N = float(N)
        if not N > 0:
            raise ValueError(f"N must be positive; got {N}")
        trap = self.trap
        m = trap.mass
        mn, tf = self._bound()
        r0, V_min = mn.position, mn.potential_J
        E_s = kc.hbar * float(np.max(tf.omega))
        sig_k = np.sqrt(kc.hbar / (2.0 * m * tf.omega))
        sig_lab = np.sqrt((tf.axes ** 2).T @ sig_k ** 2)
        grid = TrapGrid.around(r0, 1.1 * self.n_widths * sig_lab, self.n_per_axis)

        V = np.broadcast_to(np.asarray(trap.potential_J(grid.X, grid.Y, grid.Z), dtype=float),
                            grid.shape)
        V_esc = V_min + float(trap.trap_depth_J())
        basin = basin_mask(V, grid, r0, V_esc)
        Vn = (np.where(basin, V, V_esc if np.isfinite(V_esc) else V) - V_min) / E_s
        Tn = [dvr.sinc_dvr_kinetic(n, d, m) / E_s for n, d in zip(grid.shape, grid.d)]
        shape, size = grid.shape, grid.size

        def H(v):
            psi = v.reshape(shape)
            out = np.tensordot(Tn[0], psi, axes=([1], [0]))
            out += np.tensordot(psi, Tn[1], axes=([1], [1])).transpose(0, 2, 1)
            out += np.tensordot(psi, Tn[2], axes=([2], [1]))
            out += Vn * psi
            return out.reshape(-1)

        precond = 1.0 / (kc.hbar ** 2 * grid.k_squared() / (2.0 * m) / E_s + 1.0)

        def P(v):
            return np.fft.ifftn(np.fft.fftn(v.reshape(shape)) * precond).real.reshape(-1)

        def columns(f):
            return lambda X: np.column_stack([f(X[:, i]) for i in range(X.shape[1])])

        A = LinearOperator((size, size), matvec=H, matmat=columns(H), dtype=float)
        M = LinearOperator((size, size), matvec=P, matmat=columns(P), dtype=float)
        def residual_of(lam, vecs):
            return max(float(np.linalg.norm(H(vecs[:, i]) - lam[i] * vecs[:, i])
                             / np.linalg.norm(vecs[:, i])) for i in range(len(lam)))

        # LOBPCG in short chunks, keeping the best iterate.  Left to itself the block
        # iteration can lose orthogonality once it is nearly converged and blow up
        # (a residual of 1e-7 turning into 1e+1 within 60 iterations was observed);
        # measuring the residual here and stopping on any 10x rise makes it robust.
        X, best, best_res, done = self._seed(grid, r0, tf, sig_k), None, np.inf, 0
        chunk = 25
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            while done < self.maxiter:
                lam, vecs = lobpcg(A, X, M=M, largest=False, tol=self.tol,
                                   maxiter=min(chunk, self.maxiter - done))
                done += chunk
                res = residual_of(lam, vecs)
                if res < best_res:
                    best, best_res = (lam, vecs), res
                if res < self.tol or res > 10.0 * best_res:
                    break
                X = vecs
        lam, vecs = best
        order = np.argsort(lam)
        lam, vecs = lam[order], vecs[:, order]
        residual = best_res
        if residual > 1e4 * self.tol:
            raise ConvergenceError(f"LOBPCG stopped at residual {residual:.2e} (hbar omega_max "
                                   f"units); raise maxiter or check the grid.")

        psi = vecs.T.reshape((len(lam),) + shape).copy()
        psi /= np.sqrt(np.sum(psi ** 2, axis=(1, 2, 3)) * grid.dV)[:, None, None, None]
        if psi[0].sum() < 0:
            psi[0] = -psi[0]
        energies = V_min + lam * E_s
        rho0 = psi[0] ** 2
        if grid.face_max(rho0) > 1e-8 * rho0.max():
            warnings.warn("the ground state reaches the box face (> 1e-8 of its peak); "
                          "raise n_widths.", UserWarning, stacklevel=2)

        # separability: V against the sum of its cuts along the principal axes
        D = (grid.X - r0[0], grid.Y - r0[1], grid.Z - r0[2])
        V_sep = V_min
        for e in tf.axes:
            s = e[0] * D[0] + e[1] * D[1] + e[2] * D[2]
            V_sep = V_sep + (np.asarray(trap.potential_J(r0[0] + s * e[0], r0[1] + s * e[1],
                                                         r0[2] + s * e[2])) - V_min)
        where = rho0 > 1e-6 * rho0.max()
        sep = float(np.max(np.abs(np.broadcast_to(V - V_sep, shape)[where]))) / E_s

        info = NonInteractingResult(
            energies_3d=energies, psi_3d=psi,
            gap_J=float(energies[1] - energies[0]) if len(energies) > 1 else float("nan"),
            separability_index=sep, residual=residual, V_min_J=V_min, V_escape_J=V_esc,
            harmonic_zero_point_J=0.5 * kc.hbar * float(np.sum(tf.omega)))
        return TrapCloud(grid, N * rho0, N, trap, mode="noninteracting",
                         chemical_potential_J=float(energies[0]), V_min_J=V_min,
                         energy_per_atom_J=float(energies[0]), a_scattering=0.0, info=info)

    def axis_spectra(self, n_grid_max: int = 1601) -> tuple:
        """1D sinc-DVR spectra along each principal axis through the minimum
        (:class:`~kamo.trap.dvr.AxisSpectrum`): bound-state counts, level
        spacings, anharmonicity, the escape lip.  A 1D cut is exact only for a
        separable trap -- see ``separability_index``."""
        mn, tf = self._bound()
        r0 = mn.position
        out = []
        for e in tf.axes:
            def cut(u, e=e):
                return self.trap.potential_J(r0[0] + u * e[0], r0[1] + u * e[1],
                                             r0[2] + u * e[2])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out.append(dvr.solve_axis(cut, length_scale=_axis_length_scale(self.trap, e, tf),
                                          mass=self.trap.mass, n_grid_max=n_grid_max))
        return tuple(out)
