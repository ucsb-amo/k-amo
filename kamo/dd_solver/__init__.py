"""kamo.dd_solver -- microscopic coupled-dipole field solver for a dispersively probed K-39 BEC.

A discrete, N-atom, linear-optics solver for the field of a dense cloud, valid
where the smooth-susceptibility propagation of :mod:`kamo.imaging` is not.  It
produces the dipole amplitudes of every atom AND the electromagnetic field
everywhere, so it can be compared with, and where the smooth treatment fails
replace, the beam-propagation model on the same density profile and incident beam.

Quick start
-----------
>>> from kamo.dd_solver import OperatingPoint, GaussianProfile, sample_configuration, solve
>>> from kamo.dd_solver import fields
>>> op = OperatingPoint.nominal()                       # 520.583 G, kamo's numbers
>>> profile = GaussianProfile.operating_point(N=500)    # kamo's Gaussian-variational GP cloud
>>> cfg = sample_configuration(profile, theta=0.0, seed=1)
>>> res = solve(cfg, op, variant="full")                # sanity checks run automatically
>>> print(res.checks, res.excess)
>>> E = fields.field_at(res, points)                    # microscopic field, anywhere
>>> E_exc = fields.exciting_field_at_atoms(res)         # what each atom sees
>>> cf = fields.coherent_field([solve(sample_configuration(profile, 0., s), op) for s in range(40)],
...                            points, R_exc=150e-9)   # the macroscopic field

Modules
-------
:mod:`.system`     operating point (detunings, wavelength, channels), dipole basis,
                   incident fields (plane wave, Gaussian beam).
:mod:`.cloud`      Gaussian-variational profile (kamo's), eta_eff, iid sampling of
                   positions and spins, resampling one at fixed other.
:mod:`.kernel`     J_ij and Gamma_ij from the vacuum Green tensor; full / far / RG.
:mod:`.solver`     the N x N system, LU / mixed / GMRES / GPU, sanity checks S1-S4.
:mod:`.rg`         strong-disorder RG of the near-field shifts (naive; tracked behind a flag).
:mod:`.fields`     field_at, field_on_grid, exciting_field_at_atoms, far field,
                   transmitted plane, configuration-averaged coherent field.
:mod:`.vector`     3N x 3N solve with the pi and sigma+ channels; exact reduction check.
:mod:`.detect`     detected-mode projection: collection NA, phase plate, atom-equivalent units.
:mod:`.stats`      heavy-tail statistics: median, trimmed mean, SE with sample-size warnings.
:mod:`.ensemble`   many configurations in parallel; CSS-angle scans and the excess law.
:mod:`.compare_bpm` A/B against kamo.imaging's propagator on one profile and beam.
:mod:`.benchmark`  build / LU / mixed / GMRES / GPU timings and the LU-GMRES crossover.

Units: detunings in Gamma (blue positive), lengths in m, fields in units of the
incident amplitude, dipoles ``beta`` in units of ``6 pi eps0 E0 / k^3`` (see
:mod:`.system`).

Assumptions and limitations (each one real)
--------------------------------------------
1. **Linear response.**  Weak excitation, no saturation.  Resonant pairs see
   ``s ~ 0.13`` (inferred from the photon budget, not measured) and saturate at
   the 12% level, so the computed near-field excess is a low-saturation upper
   bound.  A nonlinear Bloch-vector solver is the natural extension; the hook is
   the per-atom polarizability in :func:`solver.build_matrix` (diagonal) and
   :func:`vector.polarizability_tensors`.
2. **Frozen positions.**  Atoms move ~0.03 nm per excited-state lifetime, ~0.3 nm
   with a recoil kick, against ~130 nm spacing; the steady state is reached in
   ``1/Gamma = 26 ns``, far inside the ~5 us pulse, so a steady-state solve is the
   right object.
3. **Two-level per transition**, closed sigma- cycling.  Leakage ~1e-2 per
   scattered photon is not modelled.
4. **One-body observables only.**  The solver gives field moments for a GIVEN
   configuration ensemble.  It does not decide whether configuration-derived
   noise is a genuine quantum measurement noise floor -- that needs a two-body
   (Bethe-Salpeter or second-order cumulant) treatment and is out of scope.
5. **Gaussian-variational GP ansatz**, which slightly overestimates the peak
   density at this ``N a / a_r``; and one ``a = 11.3 a0`` profile for every spin
   composition (the ``|1,0>`` state is attractive and would not even be stable
   as a pure N = 500 cloud).
6. **``a_s = 11.3 a0`` is unverified** at 520.6 G (kamo's table says 10.96);
   ``eta_eff`` scales roughly as ``a_s^{-3/5}``.
7. **Plane-wave incident field** by default: over a cloud of 0.4 um radius a
   ``w0 >> 20 um`` beam varies by < 0.1%.  :class:`system.GaussianBeam` makes
   that checkable.
8. **Condensate statistics**, ``g2(0) = 1 - 1/N``; a thermal cloud needs
   correlated sampling.
9. **Voigt polarization.**  The lab probe (along x, polarized y, ``B || z``)
   projects onto the sigma- dipole with ``|e.y|^2 = 1/2``; only that half of the
   field is scattered, and the forward-scattered y amplitude is ``alpha/2`` per
   atom.  A scalar propagation with the full ``sigma0`` corresponds instead to a
   sigma- circular wave along B.  :func:`compare_bpm` scales the BPM's cross
   section by the projection to compare like with like.

Conventions that differ from the build specification (2026-09-16)
------------------------------------------------------------------
* ``|up> = |1,-1>`` is the LOWER-frequency line and BLUE-detuned
  (``delta_up = +9.14``); the specification had it red.  Guiding results are
  therefore stated by detuning sign.
* The GP regression table in the specification came from a doubled kinetic
  term; kamo's (correct) profile is denser: ``eta_eff = 27.3`` not 19.2 at N=500.
  See :mod:`.cloud`.
* Splitting 110.23 MHz = 18.28 Gamma (kamo), not 106.3 MHz.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import OperatingPoint, PlaneWave, GaussianBeam, Channel
    from .cloud import GaussianProfile, Configuration, sample_configuration
    from .solver import solve, SolveResult, SanityReport
    from .rg import renormalize, renormalize_configuration, RGResult

_lazy = {
    "OperatingPoint": ".system", "PlaneWave": ".system", "GaussianBeam": ".system",
    "Channel": ".system", "SPIN_UP": ".system", "SPIN_DN": ".system",
    "GaussianProfile": ".cloud", "Configuration": ".cloud",
    "sample_configuration": ".cloud", "uniform_sphere_configuration": ".cloud",
    "solve": ".solver", "SolveResult": ".solver", "SanityReport": ".solver",
    "renormalize": ".rg", "renormalize_configuration": ".rg", "RGResult": ".rg",
}
_lazy_modules = ("system", "cloud", "kernel", "solver", "rg", "fields", "vector", "detect",
                 "stats", "ensemble", "compare_bpm", "benchmark")


def __getattr__(name):
    import importlib
    if name in _lazy:
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name in _lazy_modules:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'kamo.dd_solver' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_lazy) | set(_lazy_modules))
