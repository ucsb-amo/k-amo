"""The result every kamo.trap density solver returns: :class:`TrapCloud`.

Three solvers, one cloud.  The non-interacting, Thomas-Fermi and GP solvers share
no parameters, but their results share everything downstream -- moments, the
imaging bridge, normalization audits, the ``kamo.imaging`` duck-typing -- so one
class carries a gridded density whatever produced it.

:class:`DensityCloud` is the contract :mod:`kamo.imaging` duck-types on (``N``,
``widths``, ``density(x, y, z)``); ``GaussianVariationalCloud`` already meets it
and ``TrapCloud`` does too.  ``widths`` are rms widths ``w = sqrt(2 <x^2>)``: exact
for a Gaussian, and generous for a Thomas-Fermi profile (``w = 0.535 R``), which is
what the propagator's box-sizing needs.

The energy that matters is the **offset** ``chemical_potential_offset = mu - V_min``:
``V_min/h`` is ~ -8.8 kHz at the operating point while ``mu - V_min`` is ~1 kHz, so
any ratio or tolerance built on ``mu`` itself is swamped by the trap depth.

Finite-T seam: ``T_K``, ``condensate_fraction`` and ``density_thermal`` are here
(0, 1, None in this iteration); ``density_grid`` is the total the imaging bridge
and every moment use, so a thermal component added later needs no change
downstream.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from scipy.optimize import brentq

import kamo.constants as kc
from kamo.BEC_properties.variational import CollapseError  # noqa: F401  (re-exported)

from . import interactions as ia
from .grid import TrapGrid


class TrapTooShallowError(RuntimeError):
    """The requested cloud does not fit below the trap's escape energy."""


class ConvergenceError(RuntimeError):
    """A solver did not reach its tolerance (or its box kept clipping the cloud)."""


class DensityCloud(Protocol):
    """What kamo.imaging's propagator and SpinGeometry duck-type on."""

    N: float
    widths: np.ndarray

    def density(self, x, y, z): ...


class TrapCloud:
    """A ground-state density on a :class:`~kamo.trap.grid.TrapGrid`.

    Parameters
    ----------
    grid : TrapGrid
    density_grid : array (grid.shape)
        Atoms per m^3 -- condensate plus any thermal part.
    N : float
        The atom number the solver was asked for.
    trap : Trap or HarmonicTrap
    mode : str
        "noninteracting", "thomas-fermi" or "gp".
    chemical_potential_J, V_min_J : float
        Absolute chemical potential and trap minimum (J).
    energy_per_atom_J : float, optional
    a_scattering : float
        Scattering length used (m).
    density_fn : callable, optional
        Exact ``n(x, y, z)`` where the mode has one (Thomas-Fermi); otherwise
        :meth:`density` interpolates the grid.
    peak_density : float, optional
        Analytic peak density; otherwise the grid maximum.
    info : dataclass or dict, optional
        Mode-specific diagnostics.
    """

    def __init__(self, grid: TrapGrid, density_grid, N: float, trap, *, mode: str,
                 chemical_potential_J: float, V_min_J: float,
                 energy_per_atom_J: float = float("nan"), a_scattering: float = 0.0,
                 density_fn=None, peak_density: Optional[float] = None, info=None,
                 T_K: float = 0.0):
        self.grid = grid
        self.density_grid = np.asarray(density_grid, dtype=float)
        if self.density_grid.shape != grid.shape:
            raise ValueError(f"density_grid shape {self.density_grid.shape} != grid {grid.shape}")
        self.N = float(N)
        self.trap = trap
        self.mode = mode
        self.chemical_potential = float(chemical_potential_J)
        self.V_min = float(V_min_J)
        self.energy_per_atom = float(energy_per_atom_J)
        self.a_scattering = float(a_scattering)
        self._density_fn = density_fn
        self._peak = None if peak_density is None else float(peak_density)
        self.info = info
        self.T_K = float(T_K)
        self.condensate_fraction = 1.0
        self.density_thermal = None
        self._N_grid, self.centroid, var = grid.moments(self.density_grid)
        self.sigma = np.sqrt(var)
        self._interp = None
        self._column = None

    # ----------------------------------------------------------- basics
    @property
    def mass(self) -> float:
        return self.trap.mass

    @property
    def widths(self) -> np.ndarray:
        """rms 1/e widths ``sqrt(2 <x_i^2>)`` about the centroid, lab axes (m)."""
        return np.sqrt(2.0) * self.sigma

    @property
    def chemical_potential_offset(self) -> float:
        """``mu - V_min`` (J): the physically meaningful chemical potential."""
        return self.chemical_potential - self.V_min

    @property
    def chemical_potential_offset_Hz(self) -> float:
        """``(mu - V_min) / h`` (Hz)."""
        return self.chemical_potential_offset / kc.h

    @property
    def energy_offset_Hz(self) -> float:
        """``(E/N - V_min) / h`` (Hz); NaN if the solver reports no energy."""
        return (self.energy_per_atom - self.V_min) / kc.h

    @property
    def coupling_g(self) -> float:
        return ia.coupling_g(self.a_scattering, self.mass)

    @property
    def atom_number_error(self) -> float:
        """``integral(n) / N - 1``: the normalization audit."""
        return self._N_grid / self.N - 1.0

    @property
    def peak_density(self) -> float:
        return self._peak if self._peak is not None else float(np.max(self.density_grid))

    def column_density_grid(self) -> np.ndarray:
        """Column density along x (the probe axis), ``(ny, nz)`` (1/m^2)."""
        if self._column is None:
            self._column = np.sum(self.density_grid, axis=0) * self.grid.d[0]
        return self._column

    @property
    def peak_column_density(self) -> float:
        return float(np.max(self.column_density_grid()))

    @property
    def healing_length(self) -> float:
        """At the peak density, ``1 / sqrt(8 pi n0 a)`` (m)."""
        return ia.healing_length(self.peak_density, self.a_scattering)

    @property
    def density_squared_integral(self) -> float:
        """``integral n^2 dV`` (1/m^3): what two-body loss goes as."""
        return self.grid.integrate(self.density_grid ** 2)

    @property
    def mean_density(self) -> float:
        """Density-weighted mean density ``integral n^2 / N`` (1/m^3)."""
        return self.density_squared_integral / self.N

    @property
    def thomas_fermi_valid(self) -> bool:
        """``(mu - V_min) / (hbar omega_max) > 10`` -- the rule GaussianVariationalCloud uses."""
        omega_max = float(np.nanmax(self.trap.trap_frequencies().omega))
        return bool(self.chemical_potential_offset / (kc.hbar * omega_max) > 10.0)

    # -------------------------------------------------------- evaluation
    def density(self, x, y, z):
        """``n(x, y, z)`` (1/m^3): exact where the mode has a closed form,
        trilinear on the grid otherwise; 0 outside the box.  numpy only."""
        if self._density_fn is not None:
            return self._density_fn(x, y, z)
        if self._interp is None:
            self._interp = self.grid.interpolator(self.density_grid)
        X, Y, Z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                                      np.asarray(z, dtype=float))
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        return self._interp(pts).reshape(X.shape)

    def column_density(self, y, z):
        """Column density along x at ``(y, z)`` (1/m^2), bilinear; 0 outside the box."""
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.grid.y, self.grid.z), self.column_density_grid(),
                                    method="linear", bounds_error=False, fill_value=0.0)
        Y, Z = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        return f(np.stack([Y.ravel(), Z.ravel()], axis=-1)).reshape(Y.shape)

    def tf_radii(self) -> np.ndarray:
        """Half-widths of ``{V < mu}`` along the lab axes through the centroid (m);
        the harmonic ``sqrt(2 mu / (m omega^2))`` in a harmonic trap."""
        out = np.full(3, np.nan)
        c = self.centroid
        h = self.grid.half_widths * 4.0
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1.0
            f = lambda t: float(self.trap.potential_J(*(c + t * e))) - self.chemical_potential
            if f(0.0) >= 0:
                continue
            ends = []
            for sign in (+1.0, -1.0):
                t_hi = sign * h[i]
                if f(t_hi) <= 0:
                    break
                ends.append(brentq(f, 0.0, t_hi, xtol=1e-15 * h[i], rtol=1e-14))
            if len(ends) == 2:
                out[i] = 0.5 * (ends[0] - ends[1])
        return out

    def harmonic_reference(self):
        """The Gaussian-variational cloud at the same N and a in the harmonic
        expansion of the trap -- what kamo used before kamo.trap."""
        from kamo.BEC_properties.variational import GaussianVariationalCloud
        return GaussianVariationalCloud(self.N, self.trap.trap_frequencies().omega,
                                        self.a_scattering, mass=self.mass)

    # ------------------------------------------------------------ report
    def summary(self) -> str:
        h = kc.h
        s, c = self.sigma * 1e6, self.centroid * 1e6
        lines = [f"TrapCloud ({self.mode}): N = {self.N:.0f}, a = {self.a_scattering / kc.a0:+.3f} a0",
                 f"  grid {self.grid!r}, normalization error {self.atom_number_error:+.1e}",
                 f"  (mu - V_min)/h = {self.chemical_potential_offset / h:.3f} Hz"
                 + (f", (E/N - V_min)/h = {(self.energy_per_atom - self.V_min) / h:.3f} Hz"
                    if np.isfinite(self.energy_per_atom) else ""),
                 f"  centroid ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}) um, "
                 f"sigma ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}) um",
                 f"  n0 = {self.peak_density * 1e-6:.4e} cm^-3, "
                 f"peak column {self.peak_column_density * 1e-4:.4e} cm^-2"]
        if self.a_scattering > 0:
            lines.append(f"  healing length {self.healing_length * 1e6:.4f} um, Thomas-Fermi "
                         f"{'valid' if self.thomas_fermi_valid else 'NOT valid'} here")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"TrapCloud({self.mode}, N={self.N:.0f}, "
                f"mu-Vmin={self.chemical_potential_offset / kc.h:.1f} Hz)")
