"""Thomas-Fermi condensates on the real trap potential.

``n(r) = max(0, (mu - V(r)) / g)``, with ``mu`` fixed by ``integral n = N``.  The
harmonic closed form ``mu = (hbar wbar / 2)(15 N a / a_ho)^(2/5)`` is only the
*starting guess* and the cross-check: the solver integrates the actual potential,
so an anharmonic (flatter) well gives a lower ``mu`` and a longer cloud.

Algorithm
---------
1. Size a midpoint :class:`~kamo.trap.grid.TrapGrid` from the harmonic TF radii,
   ``pad`` x larger (the TF density is exactly zero beyond the radius, so there is
   nothing to gain from padding by oscillator widths); grow it 1.3x until no atom
   sits on a face.
2. Keep only the **basin**: the connected component of ``{V < V_escape}`` holding
   the minimum (``scipy.ndimage.label``), so a gravity-tilted box corner that dips
   below ``mu`` is never filled.
3. Root-find in the **offset** ``s = mu - V_min`` (never in ``mu``, which is
   dominated by the trap depth): ``F(s) = sum(clip(s + V_min - V, 0)) dV / g - N`` is
   monotone, bracketed by ``[0, mu_harmonic]`` with doubling, and solved with
   ``brentq`` to ~1e-14.
4. ``s`` above the escape energy raises :class:`~kamo.trap.cloud.TrapTooShallowError`.

``energy_per_atom`` is the Thomas-Fermi energy (potential plus mean field; the
kinetic term is dropped by construction).  The closed form makes ``density()`` exact (no interpolation error) and the peak
density analytic, ``(mu - V_min) / g``; the grid is only for moments and for the
imaging bridge.

Validity: ``thomas_fermi_valid`` (``(mu - V_min) / hbar omega_max > 10``).  At the
K-team operating point (N = 500, a = 11.3 a0) it is ~0.4: TF is not a cheap
approximation there, it is wrong by a factor ~2 in the radial size -- use the GP
solver.  ``a < 0`` has no TF limit (use GP below collapse), ``a = 0`` neither (use
the non-interacting solver).

``V_extra`` (J; callable ``V_extra(X, Y, Z)`` or an array on the grid) is added to
the potential.  Thomas-Fermi is refused at ``T > 0`` (see
:mod:`kamo.trap.finite_temperature`): its radius is already ~2x wrong at the
operating point, and a Hartree-Fock cloud on top of it would inherit that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

import kamo.constants as kc

from . import interactions as ia
from .cloud import ConvergenceError, TrapCloud, TrapTooShallowError
from .grid import TrapGrid, basin_mask


def harmonic_chemical_potential_J(N, a_scattering, omega_bar, mass):
    """Thomas-Fermi chemical potential in a harmonic trap (J), measured from the
    trap bottom: ``mu = (hbar wbar / 2)(15 N a / a_ho)^(2/5)`` with
    ``a_ho = sqrt(hbar / (m wbar))``.  ``a_scattering`` in metres, ``omega_bar``
    the geometric-mean angular frequency (rad/s).  Broadcasts."""
    a_ho = np.sqrt(kc.hbar / (mass * omega_bar))
    return 0.5 * kc.hbar * omega_bar * (15.0 * N * a_scattering / a_ho) ** 0.4


def harmonic_tf_radius(mu_J, omega, mass):
    """Thomas-Fermi radius ``sqrt(2 mu / (m omega^2))`` (m) along an axis of
    angular frequency ``omega`` (rad/s).  Broadcasts."""
    return np.sqrt(2.0 * mu_J / (mass * omega ** 2))


@dataclass
class ThomasFermiInfo:
    """Diagnostics of a Thomas-Fermi solve."""

    mu_offset_harmonic_J: float  #: closed-form mu - V_min in the harmonic expansion (J)
    box_growths: int             #: times the box was enlarged to contain the cloud
    basin_fraction: float        #: fraction of grid nodes inside the basin
    escape_energy_J: float       #: V_escape the cloud must stay below (J)


class ThomasFermiSolver:
    """Thomas-Fermi ground state of ``trap`` (a Trap or HarmonicTrap).

    Parameters
    ----------
    trap : Trap or HarmonicTrap
    a_scattering : float, optional
        Scattering length (m); looked up from the trap's state and field if omitted.
    n_per_axis : int
        Grid points per axis (default 128: mu to ~2e-6 relative).
    pad : float
        Box half-width in units of the harmonic TF radius.
    max_growth : int
        Maximum 1.3x box enlargements.
    """

    def __init__(self, trap, *, a_scattering: Optional[float] = None, n_per_axis: int = 128,
                 pad: float = 1.4, max_growth: int = 6):
        self.trap = trap
        self.a_scattering = a_scattering
        self.n_per_axis = int(n_per_axis)
        self.pad = float(pad)
        self.max_growth = int(max_growth)

    def solve(self, N: float, V_extra=None) -> TrapCloud:
        N = float(N)
        if not N > 0:
            raise ValueError(f"N must be positive; got {N}")
        trap = self.trap
        a = ia.trap_scattering_length(trap, self.a_scattering)
        if a < 0:
            raise ValueError(
                "Thomas-Fermi needs a > 0: with g < 0, max(0, (mu - V)/g) is not a "
                "density.  Use mode='gp' with N below the collapse threshold "
                "(GaussianVariationalCloud.critical_atom_number estimates it).")
        if a == 0:
            raise ValueError("g = 0: the Thomas-Fermi limit does not exist; use "
                             "mode='noninteracting'.")
        m = trap.mass
        g = ia.coupling_g(a, m)
        mn = trap.minimum()
        if not mn.converged:
            raise TrapTooShallowError("the trap has no minimum to hold a cloud")
        tf = trap.trap_frequencies()
        omega, obar = tf.omega, tf.omega_bar
        mu_h = harmonic_chemical_potential_J(N, a, obar, m)
        R_k = harmonic_tf_radius(mu_h, omega, m)
        half = self.pad * np.sqrt((tf.axes ** 2).T @ R_k ** 2)   # bounding box of the ellipsoid
        depth = float(trap.trap_depth_J())
        V_min = mn.potential_J
        V_esc = V_min + depth

        for growth in range(self.max_growth + 1):
            grid = TrapGrid.around(mn.position, half, self.n_per_axis, midpoint=True)
            V = np.broadcast_to(np.asarray(trap.potential_J(grid.X, grid.Y, grid.Z),
                                           dtype=float), grid.shape)
            if V_extra is not None:
                extra = V_extra(grid.X, grid.Y, grid.Z) if callable(V_extra) else V_extra
                V = V + np.broadcast_to(np.asarray(extra, dtype=float), grid.shape)
            try:
                basin = basin_mask(V, grid, mn.position, V_esc)
            except ValueError as err:
                raise ConvergenceError(str(err)) from None
            Vb = np.where(basin, V, np.inf)

            def F(s):
                return float(np.sum(np.clip(s + V_min - Vb, 0.0, None),
                                    dtype=np.float64)) * grid.dV / g - N

            hi = mu_h
            while F(hi) < 0 and hi < depth:
                hi *= 2.0
            if hi >= depth:
                if F(depth) < 0:
                    raise TrapTooShallowError(
                        f"mu exceeds the escape energy: {N:.4g} atoms do not fit below "
                        f"the {depth / kc.h / 1e3:.3f} kHz depth of this trap.")
                hi = depth
            s = brentq(F, 0.0, hi, xtol=1e-16 * mu_h, rtol=1e-14, maxiter=500)
            n = np.clip(s + V_min - Vb, 0.0, None) / g
            if grid.face_max(n) == 0.0:
                break
            half = half * 1.3
        else:
            raise ConvergenceError("the Thomas-Fermi cloud kept reaching the box face")

        mu = V_min + s
        lo, hi_box = grid.center - grid.half_widths, grid.center + grid.half_widths

        def density_fn(x, y, z):
            x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
            inside = ((x >= lo[0]) & (x <= hi_box[0]) & (y >= lo[1]) & (y <= hi_box[1])
                      & (z >= lo[2]) & (z <= hi_box[2]))
            return np.where(inside, np.clip((mu - trap.potential_J(x, y, z)) / g, 0.0, None),
                            0.0)

        # Thomas-Fermi energy: potential + mean field, no kinetic term by construction.
        E_per_atom = (grid.integrate(V * n) + 0.5 * g * grid.integrate(n * n)) / N
        info = ThomasFermiInfo(mu_offset_harmonic_J=mu_h, box_growths=growth,
                               basin_fraction=float(np.mean(basin)), escape_energy_J=V_esc)
        return TrapCloud(grid, n, N, trap, mode="thomas-fermi", chemical_potential_J=mu,
                         V_min_J=V_min, energy_per_atom_J=E_per_atom, a_scattering=a,
                         density_fn=density_fn, peak_density=s / g, info=info)
