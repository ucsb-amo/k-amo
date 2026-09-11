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

Finite temperature is not implemented yet.  ``T_K > 0`` raises and names the
four seams it will attach to: ``TrapCloud.T_K`` / ``condensate_fraction`` /
``density_thermal``; the solvers' ``V_extra`` (for ``2 g n_thermal`` in a
Popov/Hartree-Fock loop); ``NonInteractingResult.psi_3d`` (excited states for a
Boltzmann sum); and a ``"hartree-fock"`` entry in this dispatch table.
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


def canonical_mode(mode: str) -> str:
    try:
        return _ALIASES[str(mode).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES} (aliases: "
                         f"{sorted(set(_ALIASES) - set(MODES))})") from None


def solve(trap, N: float, mode: str = "gp", *, a_scattering: Optional[float] = None,
          T_K: float = 0.0, V_extra=None, **solver_kwargs) -> TrapCloud:
    """Ground-state density of ``N`` atoms in ``trap`` (a Trap or HarmonicTrap).

    ``a_scattering`` (m) overrides the kamo.scattering lookup for the trap's
    state and field; the non-interacting mode ignores it.
    """
    if T_K:
        raise NotImplementedError(
            "finite temperature is not implemented yet.  It attaches to TrapCloud.T_K / "
            "condensate_fraction / density_thermal, the solvers' V_extra (2 g n_thermal "
            "in a Popov/Hartree-Fock loop), NonInteractingResult.psi_3d (excited states "
            "for a Boltzmann sum) and a 'hartree-fock' mode in kamo.trap.solvers.")
    m = canonical_mode(mode)
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
              skip_errors: bool = True, options: Optional[dict] = None) -> dict:
    """Every mode on the same trap: ``{mode: TrapCloud}``.

    With ``skip_errors`` a mode that has no answer here (Thomas-Fermi for
    ``a <= 0``, a collapse, a trap too shallow) maps to its exception instead of
    stopping the others.  ``options`` maps a mode to its solver keyword
    arguments.
    """
    from kamo.BEC_properties.variational import CollapseError
    options = options or {}
    out = {}
    for mode in modes:
        m = canonical_mode(mode)
        try:
            out[m] = solve(trap, N, m, a_scattering=a_scattering, **options.get(m, {}))
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
    rows = [f"{'mode':<20}{'(mu-Vmin)/h Hz':>16}{'n0 cm^-3':>12}{'column cm^-2':>14}"
            f"{'sigma x,y,z (um)':>26}"]
    for name, c in clouds.items():
        if isinstance(c, Exception):
            rows.append(f"{name:<20}  -- {type(c).__name__}: {str(c)[:60]}")
            continue
        flag = ""
        if isinstance(c, TrapCloud):
            mu, col, s = c.chemical_potential_offset / kc.h, c.peak_column_density, c.sigma
        else:                                   # a GaussianVariationalCloud
            mu = c.chemical_potential / kc.h
            if axes is not None:
                A2 = np.asarray(axes, dtype=float) ** 2
                s = np.sqrt(A2.T @ c.sigma ** 2)
                w = np.sqrt(A2.T @ c.widths ** 2)
                col = c.N / (np.pi * w[1] * w[2])
            else:
                s, col, flag = c.sigma, c.peak_column_density, "  (own axis order)"
        s = s * 1e6
        rows.append(f"{name:<20}{mu:>16.2f}{c.peak_density * 1e-6:>12.3e}{col * 1e-4:>14.3e}"
                    f"{s[0]:>9.4f}{s[1]:>8.4f}{s[2]:>8.4f}{flag}")
    return "\n".join(rows)
