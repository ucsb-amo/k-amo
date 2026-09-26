"""The front door: ``solve(trap, N, mode)`` returns a :class:`~kamo.trap.cloud.TrapCloud`.

>>> from kamo.trap import Tweezer, Trap, solve
>>> trap = Trap(Tweezer(waist=3e-6, wavelength_m=1064e-9, power=43.65e-6),
...             B_gauss=520.594, B_direction=(0, 0, 1))
>>> cloud = solve(trap, N=500, mode="gp")      # or "thomas-fermi", "noninteracting"
>>> print(cloud.summary())

Modes (aliases in brackets):

* ``"noninteracting"`` [``"ni"``, ``"ideal"``] --
  :class:`~kamo.trap.noninteracting.NonInteractingSolver`;
* ``"thomas-fermi"`` [``"tf"``, ``"thomas_fermi"``] --
  :class:`~kamo.trap.thomas_fermi.ThomasFermiSolver`;
* ``"gp"`` [``"gpe"``, ``"gross-pitaevskii"``] --
  :class:`~kamo.trap.gross_pitaevskii.GrossPitaevskiiSolver` (the default).

Solver options pass through as keyword arguments.  :func:`solve_all` runs every
mode on the same trap and :func:`comparison_table` lays them side by side --
which, at the K-team operating point, is the quickest way to see that
Thomas-Fermi is off by a factor ~2 and the harmonic Gaussian by ~20% in peak
density.

Finite temperature: ``T_K`` (kelvin) is orthogonal to the mode.  ``mode`` still
names the condensate model and the thermal cloud comes from
:class:`~kamo.trap.finite_temperature.FiniteTemperatureSolver`:

============================  =====================================  ===============================================
``mode``                      T = 0                                  T > 0
============================  =====================================  ===============================================
``"noninteracting"``          3D single-particle ground state        ideal Bose gas on the real potential
``"gp"`` (default)            GP ground state                        GP condensate + Hartree-Fock thermal cloud
``"thomas-fermi"``            ``n = max(0, (mu - V) / g)``           refused (use ``"gp"``)
============================  =====================================  ===============================================

``thermal`` picks the thermal model: ``"hybrid"`` (default: calibrated discrete
levels below ``E_0 + 2 hbar w_max`` plus a truncated semiclassical tail),
``"lda"`` (the semiclassical model; 1.7-6.5x low in ``N_th`` at the K-team
operating point, warns) or ``"boltzmann"`` (the classical limit, no condensate).
Keyword arguments then go to the finite-temperature solver, and the condensate
solver's own options travel in ``condensate_options=`` (their names collide).
There is no ``"hartree-fock"`` mode: temperature is not a mode.

>>> cloud = solve(trap, N=500, mode="gp", T_K=30e-9)
>>> cloud.condensate_fraction, cloud.sigma_thermal, cloud.info.eta
>>> print(cloud.summary())
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import kamo.constants as kc

from .cloud import ConvergenceError, TrapCloud, TrapTooShallowError

MODES = ("noninteracting", "thomas-fermi", "gp")
_ALIASES = {"noninteracting": "noninteracting", "ni": "noninteracting",
            "ideal": "noninteracting", "thomas-fermi": "thomas-fermi",
            "thomas_fermi": "thomas-fermi", "tf": "thomas-fermi", "gp": "gp", "gpe": "gp",
            "gross-pitaevskii": "gp", "gross_pitaevskii": "gp"}
_NOT_MODES = ("hartree-fock", "hartree_fock", "hf", "popov", "hfb", "finite-temperature")


def canonical_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in _NOT_MODES:
        raise ValueError(f"{mode!r} is not a mode: temperature is orthogonal to the condensate "
                         "model.  Pass T_K to any mode, e.g. solve(trap, N, 'gp', T_K=30e-9), and "
                         "choose the thermal model with thermal='hybrid' | 'lda' | 'boltzmann'.")
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES} (aliases: "
                         f"{sorted(set(_ALIASES) - set(MODES))})") from None


def solve(trap, N: float, mode: str = "gp", *, a_scattering: Optional[float] = None,
          T_K: float = 0.0, thermal: str = "hybrid", V_extra=None,
          condensate_options: Optional[dict] = None, **solver_kwargs) -> TrapCloud:
    """Equilibrium density of ``N`` atoms in ``trap`` (a Trap or HarmonicTrap).

    ``mode`` names the condensate model; ``T_K`` (kelvin) adds a thermal cloud
    (see the module docstring).  ``a_scattering`` (m) overrides the
    kamo.scattering lookup for the trap's state and field; the non-interacting
    mode ignores it.  At ``T_K = 0`` (the default) this is one call to the
    ground-state solver, unchanged; at ``T_K > 0`` the finite-temperature solver
    runs, ``thermal`` selects its thermal model and ``condensate_options`` carries
    the condensate solver's own keyword arguments.
    """
    m = canonical_mode(mode)
    T_K = float(T_K)
    if T_K < 0 or not np.isfinite(T_K):
        raise ValueError(f"T_K must be finite and >= 0; got {T_K}")
    if T_K > 0:
        from .finite_temperature import FiniteTemperatureSolver
        if V_extra is not None:
            raise ValueError("V_extra is not supported at T > 0 (the thermal mean field is the "
                             "solver's own V_extra)")
        return FiniteTemperatureSolver(trap, a_scattering=a_scattering, condensate=m,
                                       thermal=thermal, condensate_options=condensate_options,
                                       **solver_kwargs).solve(N, T_K)
    if thermal != "hybrid" or condensate_options is not None:
        raise ValueError("thermal= and condensate_options= only apply at T_K > 0")
    if m == "noninteracting":
        if V_extra is not None:
            raise ValueError("the non-interacting solver takes no V_extra")
        from .noninteracting import NonInteractingSolver
        return NonInteractingSolver(trap, **solver_kwargs).solve(N)
    if m == "thomas-fermi":
        from .thomas_fermi import ThomasFermiSolver
        return ThomasFermiSolver(trap, a_scattering=a_scattering,
                                 **solver_kwargs).solve(N, V_extra=V_extra)
    from .gross_pitaevskii import GrossPitaevskiiSolver
    return GrossPitaevskiiSolver(trap, a_scattering=a_scattering,
                                 **solver_kwargs).solve(N, V_extra=V_extra)


def solve_all(trap, N: float, *, a_scattering: Optional[float] = None, modes=MODES,
              skip_errors: bool = True, options: Optional[dict] = None, T_K: float = 0.0,
              thermal: str = "hybrid") -> dict:
    """Every mode on the same trap: ``{mode: TrapCloud}``.

    With ``skip_errors`` a mode that has no answer here (Thomas-Fermi for
    ``a <= 0`` or at ``T_K > 0``, a collapse, a trap too shallow) maps to its
    exception instead of stopping the others.  ``options`` maps a mode to its
    solver keyword arguments (at ``T_K > 0``: the finite-temperature solver's,
    with ``condensate_options`` inside).
    """
    from kamo.BEC_properties.variational import CollapseError
    options = options or {}
    out = {}
    for mode in modes:
        m = canonical_mode(mode)
        kw = dict(options.get(m, {}))
        if T_K > 0:
            kw.update(T_K=T_K, thermal=thermal)
        try:
            out[m] = solve(trap, N, m, a_scattering=a_scattering, **kw)
        except (ValueError, TrapTooShallowError, CollapseError, ConvergenceError) as err:
            if not skip_errors:
                raise
            out[m] = err
    return out


def comparison_table(clouds: dict, axes=None) -> str:
    """Side-by-side table of the clouds from :func:`solve_all`, plus any others,
    e.g. ``{"harmonic Gaussian": cloud.harmonic_reference()}``.

    Widths and the peak column density are along the **lab** axes (column along
    x, the probe axis).  A ``GaussianVariationalCloud`` carries its widths along
    the trap's principal axes, so pass ``axes=trap.principal_axes`` to map them
    to lab marginals (``sigma_lab^2 = sum_k axes[k]^2 sigma_k^2``); without
    ``axes`` such a row is printed in its own axis order and flagged.
    """
    thermal = any(getattr(c, "T_K", 0.0) > 0 for c in clouds.values() if not isinstance(c, Exception))
    head = f"{'mode':<20}" + (f"{'T (nK)':>8}{'N0/N':>7}" if thermal else "")
    rows = [head + f"{'(mu-Vmin)/h Hz':>16}{'n0 cm^-3':>12}{'column cm^-2':>14}"
            f"{'sigma x,y,z (um)':>26}"]
    for name, c in clouds.items():
        if isinstance(c, Exception):
            rows.append(f"{name:<20}  -- {type(c).__name__}: {str(c)[:60]}")
            continue
        flag = ""
        tcols = (f"{getattr(c, 'T_K', 0.0) * 1e9:>8.2f}{getattr(c, 'condensate_fraction', 1.0):>7.3f}"
                 if thermal else "")
        if isinstance(c, TrapCloud):
            mu, col, s = c.chemical_potential_offset / kc.h, c.peak_column_density, c.sigma
        else:                                   # a GaussianVariationalCloud or IdealHarmonicBoseGas
            mu = c.chemical_potential / kc.h
            if axes is not None:
                A2 = np.asarray(axes, dtype=float) ** 2
                s = np.sqrt(A2.T @ c.sigma ** 2)
                w = np.sqrt(A2.T @ c.widths ** 2)
                col = c.N / (np.pi * w[1] * w[2])
            else:
                s, col, flag = c.sigma, c.peak_column_density, "  (own axis order)"
        s = s * 1e6
        rows.append(f"{name:<20}{tcols}{mu:>16.2f}{c.peak_density * 1e-6:>12.3e}{col * 1e-4:>14.3e}"
                    f"{s[0]:>9.4f}{s[1]:>8.4f}{s[2]:>8.4f}{flag}")
    return "\n".join(rows)
