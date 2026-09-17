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
:mod:`.cloud`      density profiles and sampling: ``GaussianProfile`` (kamo's
                   Gaussian-variational GP cloud) and ``GridProfile`` (ANY gridded
                   kamo density -- a ``kamo.trap`` GP or Thomas-Fermi cloud -- fed to
                   both this solver and kamo.imaging as the same object); eta_eff,
                   iid sampling of positions and spins, resampling one at fixed other.
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
3. **Two-level per transition, and the line is NOT perfectly closed.**  At
   520.583 G the ground states carry a 2.3-2.5 % ``m_J = +1/2`` admixture, so the
   driven line has absolute oscillator strength ``f = 0.9774`` (up) and
   ``0.9786`` (dn): every polarizability, and hence every absolute amplitude, is
   2.3 % below the ideal closed-line value, and 2.2-2.3 % of scattering events
   are Raman into a state 1.3-1.5 GHz away that is dark to the probe.  Both are
   carried (``OperatingPoint.strength_up/dn``, :attr:`SolveResult.raman_leak`);
   the depumping this causes during a 5 us pulse is 0.07-0.11 % of the atoms and
   is not modelled.  Corrected 2026-09-17; the strength is a LARGER correction
   than the 0.98 % sigma+ background below.
4. **One-body observables only.**  The solver gives field moments for a GIVEN
   configuration ensemble.  It does not decide whether configuration-derived
   noise is a genuine quantum measurement noise floor -- that needs a two-body
   (Bethe-Salpeter or second-order cumulant) treatment and is out of scope.
5. **Gaussian-variational GP ansatz.**  Against kamo's true 3D GP solver in the
   SAME harmonic trap it overestimates the peak density by 13 % and
   ``int n^2`` by 3.5 %; on the real (anharmonic, gravity-tilted) tweezer
   potential the gaps are 18 % and 12 %, and the GP shape is flatter than
   Gaussian (axial excess kurtosis -0.35).  ``GaussianProfile.eta_eff``'s closed
   form ``n_peak lambda^3 / 2^{3/2}`` is therefore Gaussian-only: for any other
   profile compute ``eta_eff`` from ``int n^2`` directly.  One ``a`` profile is
   used for every spin composition (the ``|1,0>`` state is attractive and would
   not be stable as a pure N = 500 cloud).
6. **Trap and scattering length are the largest density uncertainty, not ``a``.**
   The package default (1170 / 93 Hz, 11.3 a0, Gaussian ansatz) gives
   ``eta_eff = 27.3`` at N = 500.  The LAB trap (1064 nm, 3 um waist, measured
   1.0 kHz radial, gravity along -z) with kamo's ``a = 10.96 a0`` and the true 3D
   GP gives ``eta_eff = 19.3``, a factor 1.41 lower -- see
   :func:`cloud.GaussianProfile.lab_operating_point`.  The whole 10.96-11.53 a0
   backend spread moves ``eta_eff`` by 1.2 %, because the cloud is
   kinetic-energy dominated radially and the exponent is
   ``d ln eta_eff / d ln a = -0.24``, not the Thomas-Fermi -3/5.
7. **Plane-wave incident field** by default: over a cloud of 0.4 um radius a
   ``w0 >> 20 um`` beam varies by < 0.1%.  :class:`system.GaussianBeam` makes
   that checkable.
8. **Condensate statistics.**  Within GP the N-body state is a product, so iid
   sampling of exactly N atoms is EXACT and reproduces ``g2 = 1 - 1/N``
   identically.  What it omits is beyond-GP: for ``r << xi`` the pair density
   carries the two-body correlation ``(1 - a/r)^2``, which at the 42-53 nm
   resonant-pair distance is 0.97-0.98 for ``|up>`` pairs but 1.15-1.20 for
   ``|dn>`` pairs.  A thermal cloud (``g2(0) = 2``) would need correlated
   sampling and this sampler must not be carried there.
9. **Voigt polarization.**  The lab probe (along x, polarized y, ``B || z``)
   projects onto the sigma- dipole with ``|e.y|^2 = 1/2``; only that half of the
   field is scattered, and the forward-scattered y amplitude is ``alpha/2`` per
   atom.  A scalar propagation with the full ``sigma0`` corresponds instead to a
   sigma- circular wave along B.  :func:`compare_bpm` scales the BPM's cross
   section by the projection AND by ``f`` to compare like with like.
10. **Inside the cloud the configuration-averaged field is the LORENTZ LOCAL
    field**, ``E_Maxwell + P/(3 eps0)``, not the Maxwell field a susceptibility
    propagation reports, and an exclusion radius ``R_exc`` biases it further by a
    calculable, ``R_exc``-dependent amount.  Both are ~10 % at the operating
    density.  The A/B is therefore done OUTSIDE the cloud.  See :mod:`.fields`.

Conventions that differ from the build specification (2026-09-16)
------------------------------------------------------------------
* ``|up> = |1,-1>`` is the LOWER-frequency line and BLUE-detuned
  (``delta_up = +9.14``); the specification had it red.  Guiding results are
  therefore stated by detuning sign.
* The GP regression table in the specification came from a doubled kinetic
  term; kamo's (correct) Gaussian-variational profile in the SAME trap is
  denser: ``eta_eff = 27.3`` not 19.2 at N = 500.  The lab trap's true GP cloud
  is 19.3, so the specification's number is right by accident, through two
  cancelling errors.  See :mod:`.cloud`.
* Splitting 110.23 MHz = 18.28 Gamma (kamo), not 106.3 MHz.
* The near-field ablation is ``'nonear'`` (exact ``Gamma``, static ``1/r^3``
  removed), following Andreoli et al., PRX 11, 011026 (2021), not the
  ``1/x``-only ``'far'`` kernel.  ``Im G`` has no near-field part to remove, and
  truncating it breaks positivity (2026-09-17).
* The measured excess coefficient against that ablation is 0.0078-0.0087, i.e.
  ``xi_circ / 2`` with ``xi_circ = 1/(12 pi sqrt3)``.  The specification's
  constant is quoted as measured rather than assumed; the factor 2 is a
  convention in the resonant-shell derivation that the specification does not
  pin down.
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
    "GaussianProfile": ".cloud", "GridProfile": ".cloud",
    "profile_from_kamo": ".cloud", "Configuration": ".cloud",
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
