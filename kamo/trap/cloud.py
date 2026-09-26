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

Finite temperature: ``density_grid`` is the **total**, and every moment, column
density and the imaging bridge run off it unchanged.  A cloud from
:mod:`kamo.trap.finite_temperature` also carries ``density_condensate`` and
``density_thermal`` on the same grid (they sum to the total), ``T_K``, ``N_0`` /
``N_th`` / ``condensate_fraction`` (computed from the grid, so the audit
``N_0 + N_th = N`` means something), per-component ``sigma_*`` / ``widths_*`` /
``centroid_*`` / ``peak_density_*``, and ``component="total" | "condensate" |
"thermal"`` on :meth:`density`, :meth:`column_density` and
:meth:`column_density_grid`.  At ``T_K = 0`` nothing changes: ``density_thermal``
is None, ``condensate_fraction`` is 1 and the condensate is the total.

``widths`` stays the rms width of the *total* density -- it is the containment
contract of :mod:`kamo.imaging`, and a truncated thermal cloud with a hard edge at
the escape saddle is what the box has to hold (``widths_condensate`` sizes a box
for the condensate alone).  ``chemical_potential`` at ``T > 0`` is the single
equilibrium ``mu`` both components share, ``mu_GP - kT ln(1 + 1/N_0)`` (the
condensate as one Bose level at the GP eigenvalue); ``healing_length`` and
``thomas_fermi_valid`` read the condensate peak.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator
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
    T_K : float
        Temperature (K).  With ``T_K > 0`` give both components:
    density_condensate, density_thermal : arrays (grid.shape), optional
        Atoms per m^3 in each; they must sum to ``density_grid``.
    """

    def __init__(self, grid: TrapGrid, density_grid, N: float, trap, *, mode: str,
                 chemical_potential_J: float, V_min_J: float,
                 energy_per_atom_J: float = float("nan"), a_scattering: float = 0.0,
                 density_fn=None, peak_density: Optional[float] = None, info=None,
                 T_K: float = 0.0, density_condensate=None, density_thermal=None):
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
        if not self.T_K >= 0:
            raise ValueError(f"T_K must be >= 0; got {T_K}")
        self._N_grid, self.centroid, var = grid.moments(self.density_grid)
        self.sigma = np.sqrt(var)
        if density_thermal is None and density_condensate is None:
            self.density_condensate = self.density_grid
            self.density_thermal = None
            self.condensate_fraction = 1.0
        else:
            if density_thermal is None or density_condensate is None:
                raise ValueError("give both density_condensate and density_thermal, or neither")
            self.density_condensate = np.asarray(density_condensate, dtype=float)
            self.density_thermal = np.asarray(density_thermal, dtype=float)
            if (self.density_condensate.shape != grid.shape
                    or self.density_thermal.shape != grid.shape):
                raise ValueError("the component densities must live on the cloud's grid")
            total = self.density_condensate + self.density_thermal
            scale = max(float(np.max(np.abs(self.density_grid))), 1e-300)
            if float(np.max(np.abs(total - self.density_grid))) > 1e-9 * scale:
                raise ValueError("density_condensate + density_thermal != density_grid")
            N_0 = grid.integrate(self.density_condensate)
            self.condensate_fraction = float(N_0 / self._N_grid) if self._N_grid > 0 else 0.0
        self._interp = {}
        self._column = {}
        self._column_interp = {}

    # ----------------------------------------------------------- basics
    @property
    def mass(self) -> float:
        return self.trap.mass

    @property
    def widths(self) -> np.ndarray:
        """rms 1/e widths ``sqrt(2 <x_i^2>)`` about the centroid, lab axes (m) --
        of the **total** density (the imaging box-sizing contract)."""
        return np.sqrt(2.0) * self.sigma

    # ---------------------------------------------------------- components
    @property
    def is_finite_temperature(self) -> bool:
        return self.density_thermal is not None

    @property
    def T_nK(self) -> float:
        return self.T_K * 1e9

    @property
    def N_0(self) -> float:
        """Condensate atoms, ``condensate_fraction * N``."""
        return self.condensate_fraction * self.N

    @property
    def N_th(self) -> float:
        """Thermal atoms, ``N - N_0``."""
        return self.N - self.N_0

    @property
    def thermal_fraction(self) -> float:
        return 1.0 - self.condensate_fraction

    @property
    def model_label(self) -> str:
        """``"gp"``, or ``"gp + hybrid thermal, T = 30.0 nK"`` (``"thermal only"`` above
        the crossover)."""
        if not self.is_finite_temperature:
            return self.mode
        thermal = getattr(self.info, "thermal_model", "thermal")
        state = "thermal only" if self.condensate_fraction < 1e-3 else f"T = {self.T_nK:.1f} nK"
        return f"{self.mode} + {thermal} thermal, {state}"

    def component_density_grid(self, component: str = "total") -> np.ndarray:
        """The gridded density of one component (``"total"``, ``"condensate"`` or
        ``"thermal"``); a thermal component needs ``T_K > 0``."""
        if component == "total":
            return self.density_grid
        if component == "condensate":
            return self.density_condensate
        if component == "thermal":
            if self.density_thermal is None:
                raise ValueError("this cloud is at T = 0: it has no thermal component")
            return self.density_thermal
        raise ValueError(f"component must be 'total', 'condensate' or 'thermal'; got {component!r}")

    def _moments_of(self, component: str):
        return self.grid.moments(self.component_density_grid(component))

    @property
    def sigma_condensate(self) -> np.ndarray:
        """rms widths of the condensate alone (m); ``sigma`` at T = 0."""
        return np.sqrt(self._moments_of("condensate")[2])

    @property
    def sigma_thermal(self) -> np.ndarray:
        """rms widths of the thermal cloud alone (m); NaN at T = 0."""
        if self.density_thermal is None:
            return np.full(3, np.nan)
        return np.sqrt(self._moments_of("thermal")[2])

    @property
    def widths_condensate(self) -> np.ndarray:
        return np.sqrt(2.0) * self.sigma_condensate

    @property
    def widths_thermal(self) -> np.ndarray:
        return np.sqrt(2.0) * self.sigma_thermal

    @property
    def centroid_condensate(self) -> np.ndarray:
        return self._moments_of("condensate")[1]

    @property
    def centroid_thermal(self) -> np.ndarray:
        if self.density_thermal is None:
            return np.full(3, np.nan)
        return self._moments_of("thermal")[1]

    @property
    def peak_density_condensate(self) -> float:
        """Peak condensate density (1/m^3); the analytic peak where the mode has one."""
        if self.density_thermal is None:
            return self.peak_density
        return float(np.max(self.density_condensate))

    @property
    def peak_density_thermal(self) -> float:
        return 0.0 if self.density_thermal is None else float(np.max(self.density_thermal))

    @property
    def thermal_wavelength_m(self) -> float:
        """``h / sqrt(2 pi m k T)`` (m); inf at T = 0."""
        if self.T_K == 0:
            return float("inf")
        return kc.h / np.sqrt(2.0 * np.pi * self.mass * kc.kB * self.T_K)

    @property
    def eta(self) -> float:
        """Trap depth over ``k T``; inf at T = 0 or for a HarmonicTrap."""
        if self.T_K == 0:
            return float("inf")
        return float(self.trap.trap_depth_J()) / (kc.kB * self.T_K)

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
        """Peak of the total density (1/m^3); analytic where the mode has one."""
        return self._peak if self._peak is not None else float(np.max(self.density_grid))

    def column_density_grid(self, component: str = "total") -> np.ndarray:
        """Column density along x (the probe axis), ``(ny, nz)`` (1/m^2)."""
        if component not in self._column:
            rho = self.component_density_grid(component)
            self._column[component] = np.sum(rho, axis=0) * self.grid.d[0]
        return self._column[component]

    @property
    def peak_column_density(self) -> float:
        return float(np.max(self.column_density_grid()))

    @property
    def healing_length(self) -> float:
        """At the condensate's peak density, ``1 / sqrt(8 pi n0 a)`` (m)."""
        return ia.healing_length(self.peak_density_condensate, self.a_scattering)

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
    def density(self, x, y, z, *, component: str = "total"):
        """``n(x, y, z)`` (1/m^3): exact where the mode has a closed form,
        trilinear on the grid otherwise; 0 outside the box.  numpy only.
        ``component`` selects the condensate or the thermal part of a
        finite-temperature cloud."""
        if self._density_fn is not None and component == "total":
            return self._density_fn(x, y, z)
        if component not in self._interp:
            self._interp[component] = self.grid.interpolator(self.component_density_grid(component))
        X, Y, Z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                                      np.asarray(z, dtype=float))
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        return self._interp[component](pts).reshape(X.shape)

    def column_density(self, y, z, *, component: str = "total"):
        """Column density along x at ``(y, z)`` (1/m^2), bilinear; 0 outside the box."""
        if component not in self._column_interp:
            self._column_interp[component] = RegularGridInterpolator(
                (self.grid.y, self.grid.z), self.column_density_grid(component),
                method="linear", bounds_error=False, fill_value=0.0)
        f = self._column_interp[component]
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

    def harmonic_reference(self, T_K: Optional[float] = None):
        """The reference cloud in the harmonic expansion of the trap: at T = 0 the
        Gaussian-variational cloud at the same N and a (what kamo used before
        kamo.trap); at ``T_K > 0`` (default: this cloud's temperature) the closed-form
        :class:`~kamo.BEC_properties.thermal.IdealHarmonicBoseGas`, which drops the
        interactions -- a reference for the thermal cloud and the condensate
        fraction, not for the condensate's size."""
        T = self.T_K if T_K is None else float(T_K)
        omega = self.trap.trap_frequencies().omega
        if T > 0:
            from kamo.BEC_properties.thermal import IdealHarmonicBoseGas
            return IdealHarmonicBoseGas(self.N, omega, T, mass=self.mass,
                                        a_scattering=self.a_scattering)
        from kamo.BEC_properties.variational import GaussianVariationalCloud
        return GaussianVariationalCloud(self.N, omega, self.a_scattering, mass=self.mass)

    # ------------------------------------------------------------ report
    def summary(self) -> str:
        h = kc.h
        s, c = self.sigma * 1e6, self.centroid * 1e6
        lines = [f"TrapCloud ({self.model_label}): N = {self.N:.0f}, a = {self.a_scattering / kc.a0:+.3f} a0",
                 f"  grid {self.grid!r}, normalization error {self.atom_number_error:+.1e}",
                 f"  (mu - V_min)/h = {self.chemical_potential_offset / h:.3f} Hz"
                 + (f", (E/N - V_min)/h = {(self.energy_per_atom - self.V_min) / h:.3f} Hz"
                    if np.isfinite(self.energy_per_atom) else ""),
                 f"  centroid ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}) um, "
                 f"sigma ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}) um",
                 f"  n0 = {self.peak_density * 1e-6:.4e} cm^-3, "
                 f"peak column {self.peak_column_density * 1e-4:.4e} cm^-2"]
        if self.is_finite_temperature:
            sc, st = self.sigma_condensate * 1e6, self.sigma_thermal * 1e6
            lines.append(f"  T = {self.T_nK:.2f} nK, eta = U/kT = {self.eta:.2f}, "
                         f"lambda_dB = {self.thermal_wavelength_m * 1e6:.3f} um")
            lines.append(f"  condensate N0 = {self.N_0:.1f} ({100 * self.condensate_fraction:.1f}%), "
                         f"sigma ({sc[0]:.4f}, {sc[1]:.4f}, {sc[2]:.4f}) um, "
                         f"peak {self.peak_density_condensate * 1e-6:.3e} cm^-3")
            lines.append(f"  thermal    Nth = {self.N_th:.1f} ({100 * self.thermal_fraction:.1f}%), "
                         f"sigma ({st[0]:.4f}, {st[1]:.4f}, {st[2]:.4f}) um, "
                         f"peak {self.peak_density_thermal * 1e-6:.3e} cm^-3")
        if self.a_scattering > 0:
            lines.append(f"  healing length {self.healing_length * 1e6:.4f} um, Thomas-Fermi "
                         f"{'valid' if self.thomas_fermi_valid else 'NOT valid'} here")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.is_finite_temperature:
            return (f"TrapCloud({self.mode}, N={self.N:.0f}, T={self.T_nK:.1f} nK, "
                    f"N0/N={self.condensate_fraction:.3f}, "
                    f"mu-Vmin={self.chemical_potential_offset / kc.h:.1f} Hz)")
        return (f"TrapCloud({self.mode}, N={self.N:.0f}, "
                f"mu-Vmin={self.chemical_potential_offset / kc.h:.1f} Hz)")
