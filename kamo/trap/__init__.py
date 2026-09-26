"""kamo.trap -- optical dipole traps: beams, potentials, and the BEC they hold.

Beam geometry -> trap potential -> condensate density, on the real (anharmonic,
gravity-tilted) potential rather than its harmonic approximation.

Quick start
-----------
>>> from kamo.trap import (Tweezer, LightSheet, Trap, solve, solve_all, comparison_table,
...                        critical_temperature)
>>> from kamo.trap.plotting import plot_trap_summary, plot_plane_cut
>>>
>>> # a 3 um, 1064 nm tweezer along x; polarization and B set the light shift
>>> tw = Tweezer(waist=3e-6, wavelength_m=1064e-9, polarization=(0, 1, 1j))
>>> trap = Trap(tw, state=(4, 0, 0.5, 1, -1), B_gauss=520.6, B_direction=(0, 0, 1))
>>>
>>> # any kamo atom works -- pass atom= and that atom's own state tuple
>>> from kamo import atom
>>> rb_trap = Trap(tw, atom=atom("Rb87"), state=(5, 0, 0.5, 2, -2), B_gauss=520.6)
>>> trap = trap.rescaled_to_frequency(1.0e3)     # the measured 1 kHz radial frequency
>>> trap.frequencies_Hz, trap.sag_along_gravity, trap.depth_uK
>>> print(trap.summary())
>>>
>>> cloud = solve(trap, N=500, mode="gp")        # or "thomas-fermi", "noninteracting"
>>> cloud.sigma, cloud.peak_density, cloud.chemical_potential_offset_Hz
>>> print(comparison_table(solve_all(trap, 500)))
>>> plot_trap_summary(trap); plot_plane_cut(trap, normal=(0, 1, 0))
>>>
>>> warm = solve(trap, N=500, mode="gp", T_K=30e-9)   # condensate + thermal cloud
>>> warm.condensate_fraction, warm.sigma_thermal, warm.info.eta
>>> critical_temperature(trap, 500).summary()          # reference T_c values and eta

Units: SI throughout (m, W, J, s); frequencies in Hz with a ``_Hz`` suffix,
wavelengths ``wavelength_m``, energies as ``_J`` (with ``_Hz``, ``_K``, ``_uK``
views where people read them that way), polarizabilities ``_au`` / ``_SI``,
fields in Gauss.  Lengths and powers are bare SI, as in the rest of kamo.

Modules
-------
:mod:`.beams`      ``Tweezer``, ``LightSheet``, ``Crossed``: lab-frame polarization
                   and propagation direction, 3D intensity and its gradient.
:mod:`.trap`       ``Trap``: beams + state + field + gravity -> the potential, its
                   sagged minimum, principal frequencies, escape depth and saddle.
:mod:`.polarizability`  state polarizabilities from UDel-portal matrix elements
                   (ARC fill-in), lab-frame geometry, provenance, uncertainty.
:mod:`.solvers`    ``solve`` / ``solve_all``: the three density modes, at T = 0 or T > 0.
:mod:`.noninteracting`, :mod:`.thomas_fermi`, :mod:`.gross_pitaevskii`  the ground-state solvers.
:mod:`.finite_temperature`  ``FiniteTemperatureSolver``: condensate + Hartree-Fock
                   thermal cloud on the real potential (hybrid discrete + semiclassical
                   spectrum, truncated at the escape energy); ``critical_temperature``,
                   ``temperature_from_condensate_fraction``.
:mod:`.cloud`      ``TrapCloud``: what every solver returns.
:mod:`.imaging_bridge`  ``GriddedMixture``: any cloud through kamo.imaging.
:mod:`.cloud_optics`  thin-screen phase, Born form factor and far field, chord optical
                   depths and reabsorption of a gridded cloud, no Gaussian assumed.
:mod:`.plotting`   plane cuts (any normal), line cuts, beam profiles, summaries.
:mod:`.dvr`        the exact 1D eigensolver for one trap axis.
:mod:`.frames`     vector validation and the polarization invariants (beta, gamma).
:mod:`.gaussian`, :mod:`.thin_lens`  the legacy ``GaussianBeam`` and lens helpers,
                   moved here from :mod:`kamo.gaussian_beam` (now a shim).

Geometry convention: lab ``(x, y, z)`` as in :mod:`kamo.imaging` -- x the default
propagation axis, z the default quantization axis and the vertical.

Species: ``Trap(..., atom=...)`` takes any :mod:`kamo.atom_properties.alkali`
atom (polarizabilities, mass and light shifts all come from it); the default is
kamo's default atom, 39K, which is what the examples above use.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .beams import Beam, Crossed, InterferenceWarning, LightSheet, Tweezer
    from .dvr import AxisSpectrum, solve_axis
    from .polarizability import StatePolarizability
    from .trap import HarmonicTrap, Trap, TrapFrequencies, TrapMinimum
    from .grid import TrapGrid
    from .cloud import ConvergenceError, TrapCloud, TrapTooShallowError
    from .thomas_fermi import ThomasFermiSolver
    from .noninteracting import NonInteractingSolver
    from .gross_pitaevskii import GrossPitaevskiiSolver
    from .solvers import comparison_table, solve, solve_all
    from .finite_temperature import (FiniteTemperatureSolver, FiniteTemperatureResult,
                                     CriticalTemperature, ModelValidityWarning, ProductSpectrum,
                                     critical_temperature, semiclassical_density,
                                     temperature_from_condensate_fraction)
    from .imaging_bridge import GriddedMixture, effective_widths, propagator_for
    from .gaussian import GaussianBeam
    from .thin_lens import Objective, ThinLensGaussian

_lazy = {
    "Beam":                ".beams",
    "Tweezer":             ".beams",
    "LightSheet":          ".beams",
    "Crossed":             ".beams",
    "InterferenceWarning": ".beams",
    "StatePolarizability": ".polarizability",
    "Trap":                ".trap",
    "TrapMinimum":         ".trap",
    "TrapFrequencies":     ".trap",
    "HarmonicTrap":        ".trap",
    "TrapGrid":            ".grid",
    "TrapCloud":           ".cloud",
    "TrapTooShallowError": ".cloud",
    "ConvergenceError":    ".cloud",
    "ThomasFermiSolver":   ".thomas_fermi",
    "NonInteractingSolver": ".noninteracting",
    "GrossPitaevskiiSolver": ".gross_pitaevskii",
    "solve":               ".solvers",
    "solve_all":           ".solvers",
    "comparison_table":    ".solvers",
    "FiniteTemperatureSolver": ".finite_temperature",
    "FiniteTemperatureResult": ".finite_temperature",
    "CriticalTemperature":  ".finite_temperature",
    "ModelValidityWarning": ".finite_temperature",
    "ProductSpectrum":      ".finite_temperature",
    "critical_temperature": ".finite_temperature",
    "semiclassical_density": ".finite_temperature",
    "temperature_from_condensate_fraction": ".finite_temperature",
    "GriddedMixture":      ".imaging_bridge",
    "effective_widths":    ".imaging_bridge",
    "propagator_for":      ".imaging_bridge",
    "AxisSpectrum":     ".dvr",
    "solve_axis":       ".dvr",
    "GaussianBeam":     ".gaussian",
    "ThinLensGaussian": ".thin_lens",
    "Objective":        ".thin_lens",
}

_lazy_modules = ("frames", "beams", "polarizability", "trap", "plotting", "dvr",
                 "grid", "interactions", "cloud", "thomas_fermi", "noninteracting",
                 "gross_pitaevskii", "solvers", "finite_temperature", "imaging_bridge",
                 "cloud_optics", "gaussian", "thin_lens")


def __getattr__(name):
    import importlib
    if name in _lazy:
        val = getattr(importlib.import_module(_lazy[name], __name__), name)
        globals()[name] = val
        return val
    if name in _lazy_modules:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'kamo.trap' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_lazy) | set(_lazy_modules))
