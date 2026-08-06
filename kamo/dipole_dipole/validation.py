"""Analytic-limit and regression checks for the dipole-dipole model.

k-amo has no pytest infrastructure, so these are plain functions.  Run them with

    python -m kamo.dipole_dipole.validation

Every check either compares against a closed-form limit or against a number
measured from kamo/ARC while the module was written.  A failure in the atomic-
structure checks means kamo or the ARC database changed underneath the model.
"""

from __future__ import annotations

import numpy as np

from kamo import constants as c
from . import green_tensor as gt  # submodule, not the dyadic function
from .cloud import BECCloud
from .light_assisted_collisions import QuasiStaticLZModel
from .pair import ANTISYMMETRIC, SYMMETRIC, PairPotential
from .transition import CyclingTransition

# Numbers measured from kamo/ARC at B = 520.6 G during development.
REF_B_GAUSS = 520.6
REF_DIPOLE_EA0 = 2.8639
REF_LINEWIDTH_MHZ = 6.0050
REF_WAVELENGTH_NM = 766.696
REF_GROUND_PURITY = 0.97544
REF_NEAREST_CHANNEL_MHZ = 969.98


class CheckFailed(AssertionError):
    pass


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = "  [%s] %s" % (status, name)
    if detail:
        line += " -- " + detail
    print(line)
    return bool(ok)


def check_green_tensor_limits():
    """Near-field limit, superradiant and subradiant rates, and the magic angle."""
    print("green tensor limits")
    ok = True
    Gamma = 1.0
    xi = 1e-3
    for theta in (0.0, np.pi / 4, np.pi / 2):
        p = np.sin(theta) ** 2 / 2.0
        J, G12 = gt.dd_coupling_scalar(xi, p, Gamma)
        Jn, _ = gt.near_field_coupling(xi, p, Gamma)
        ok &= _check("near field at theta=%.0f deg" % np.degrees(theta),
                     abs(J / Jn - 1.0) < 1e-4, "ratio %.8f" % (J / Jn))
        ok &= _check("Gamma_12 -> Gamma at theta=%.0f deg" % np.degrees(theta),
                     abs(G12 / Gamma - 1.0) < 1e-4)
    br = gt.pair_branches(1e-4, 0.0, Gamma)
    ok &= _check("Gamma_sym -> 2 Gamma", abs(br["Gamma_sym"] / Gamma - 2.0) < 1e-4)
    ok &= _check("Gamma_anti -> 0", br["Gamma_anti"] / Gamma < 1e-4)
    ok &= _check("magic angle = 54.7356 deg",
                 abs(gt.MAGIC_ANGLE_DEG - 54.7356) < 1e-3,
                 "%.4f deg" % gt.MAGIC_ANGLE_DEG)
    # tensor and scalar contractions must agree
    e = gt.spherical_unit_vector(-1)
    rv = np.array([0.4, -0.7, 0.9])
    J1, g1 = gt.dd_coupling(2.0, rv, e, Gamma)
    p = gt.dipole_projection(e, rv / np.linalg.norm(rv))
    J2, g2 = gt.dd_coupling_scalar(2.0 * np.linalg.norm(rv), p, Gamma)
    ok &= _check("dyadic == scalar contraction",
                 np.allclose([J1, g1], [J2, g2], rtol=1e-12))
    return ok


def check_atomic_structure(transition=None):
    """Regression against the kamo/ARC numbers at 520.6 G."""
    print("atomic structure at %.1f G" % REF_B_GAUSS)
    t = transition or CyclingTransition.at_field(REF_B_GAUSS)
    ok = True
    ok &= _check("dipole = %.4f e a0" % REF_DIPOLE_EA0,
                 abs(t.d_ea0 / REF_DIPOLE_EA0 - 1.0) < 1e-3,
                 "%.4f" % t.d_ea0)
    ok &= _check("linewidth = %.4f MHz" % REF_LINEWIDTH_MHZ,
                 abs(t.linewidth_Hz / 1e6 / REF_LINEWIDTH_MHZ - 1.0) < 1e-3,
                 "%.4f MHz" % (t.linewidth_Hz / 1e6))
    ok &= _check("wavelength = %.3f nm" % REF_WAVELENGTH_NM,
                 abs(t.wavelength_m * 1e9 / REF_WAVELENGTH_NM - 1.0) < 1e-5,
                 "%.4f nm" % (t.wavelength_m * 1e9))
    ok &= _check("ground purity = %.5f" % REF_GROUND_PURITY,
                 abs(t.ground_purity - REF_GROUND_PURITY) < 1e-3,
                 "%.6f" % t.ground_purity)
    strong = [ch for ch in t.channels
              if abs(ch.detuning_Hz) > 1e6 and ch.relative_strength > 1e-3]
    nearest = min(strong, key=lambda ch: abs(ch.detuning_Hz))
    ok &= _check("nearest strong channel near %.0f MHz" % REF_NEAREST_CHANNEL_MHZ,
                 abs(abs(nearest.detuning_Hz) / 1e6 / REF_NEAREST_CHANNEL_MHZ - 1.0) < 1e-2,
                 "%s at %.2f MHz" % (nearest.polarization, nearest.detuning_Hz / 1e6))
    # the hyperfine-admixture leak inside the +/-10 linewidth window must stay tiny
    inside = [ch for ch in t.channels
              if 1e3 < abs(ch.detuning_Hz) < 10 * t.linewidth_Hz]
    worst = max((ch.relative_strength for ch in inside), default=0.0)
    ok &= _check("in-window leakage channels are negligible",
                 worst < 1e-3, "strongest is %.2e of the driven channel" % worst)
    ok &= _check("I_sat within 5 percent of 1.75 mW/cm^2",
                 abs(t.I_sat / 10.0 / 1.75 - 1.0) < 0.05,
                 "%.4f mW/cm^2" % (t.I_sat / 10.0))
    return ok


def check_beam_geometry(transition=None):
    """Polarisation projection onto the driven sigma channel."""
    print("beam geometry")
    t = transition or CyclingTransition.at_field(REF_B_GAUSS)
    ok = True
    perp = t.drive_projection((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ok &= _check("beam perp to B, eps perp to both -> half on sigma",
                 abs(perp[-1] - 0.5) < 1e-12, "%.6f" % perp[-1])
    ok &= _check("same geometry -> no pi", abs(perp[0]) < 1e-12)
    along_B = t.drive_projection((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    ok &= _check("eps along B -> pure pi, no sigma",
                 abs(along_B[-1]) < 1e-12 and abs(along_B[0] - 1.0) < 1e-12)
    pure = t.drive_projection((0.0, 0.0, 1.0), gt.spherical_unit_vector(-1))
    ok &= _check("beam along B with sigma-minus light -> all on sigma",
                 abs(pure[-1] - 1.0) < 1e-12, "%.6f" % pure[-1])
    return ok


def check_condon_radii(transition=None, potential=None):
    """Condon radii, and the size of the retardation correction."""
    print("Condon radii")
    t = transition or CyclingTransition.at_field(REF_B_GAUSS)
    pot = potential or PairPotential(t)
    ok = True
    expected_nm = {0.3: 165.6, 1.0: 110.9, 3.0: 76.9, 10.0: 51.5}
    for x, nm in expected_nm.items():
        det = -x * t.linewidth_Hz
        # theta = 0: the symmetric branch is repulsive, so red detuning finds the
        # antisymmetric one
        R_nf = float(pot.condon_radius_near_field(det, 0.0, ANTISYMMETRIC))
        R = float(pot.condon_radius(det, 0.0, ANTISYMMETRIC)[()] if
                  np.ndim(pot.condon_radius(det, 0.0, ANTISYMMETRIC)) == 0
                  else pot.condon_radius(det, 0.0, ANTISYMMETRIC)[0])
        ok &= _check("near-field R_C at %.1f linewidths = %.1f nm" % (x, nm),
                     abs(R_nf * 1e9 / nm - 1.0) < 0.02, "%.1f nm" % (R_nf * 1e9))
        kR = t.k * R_nf
        rel = abs(R / R_nf - 1.0)
        ok &= _check("retardation correction at kR = %.2f is O(kR^2)" % kR,
                     0.05 * kR ** 2 < rel < 3.0 * kR ** 2,
                     "%.1f percent (kR^2 = %.2f)" % (100 * rel, kR ** 2))
    return ok


def _default_system():
    t = CyclingTransition.at_field(REF_B_GAUSS)
    pot = PairPotential(t)
    cloud = BECCloud(N=1e5, trap_frequencies_Hz=(150.0, 150.0, 20.0), a_s_bohr=100.0)
    model = QuasiStaticLZModel(t, pot, cloud)
    return t, pot, cloud, model


def check_flight_time(potential=None, transition=None):
    """Analytic incomplete-beta fall time against direct quadrature."""
    print("flight time")
    t = transition or CyclingTransition.at_field(REF_B_GAUSS)
    pot = potential or PairPotential(t)
    ok = True
    R0 = 111e-9
    W = 2.0 * c.kB * 50e-6
    t_fast = float(pot.flight_time(R0, 0.0, ANTISYMMETRIC, W))
    t_slow = float(pot.flight_time(R0, 0.0, ANTISYMMETRIC, W, exact=True))
    ok &= _check("analytic vs quadrature (inward)",
                 abs(t_fast / t_slow - 1.0) < 2e-3,
                 "%.3f ns vs %.3f ns" % (t_fast * 1e9, t_slow * 1e9))
    # a repulsive branch can release at most its own potential energy
    A = float(pot.C3(0.0, SYMMETRIC))
    E0 = abs(A) / R0 ** 3
    t_inf = float(pot.flight_time(R0, 0.0, SYMMETRIC, 2.0 * E0))
    ok &= _check("repulsive branch cannot release more than |A|/R^3",
                 not np.isfinite(t_inf))
    return ok


def check_beta_scalings(model=None, transition=None):
    """Quasistatic wing exponent, saturation, trap-depth trend and shielding."""
    print("beta scalings")
    if model is None:
        transition, _, _, model = _default_system()
    t = transition or model.transition
    ok = True
    G = t.linewidth_Hz
    I_weak = 0.01 * t.I_sat

    # Excitation only (no survival factor) must follow the analytic C3 quasistatic
    # wing, integrated excitation rate proportional to Delta^-2.
    x = np.array([3.0, 5.0, 7.0, 10.0])
    exc = np.array([model.beta_quasistatic(-xi * G, I_weak, 50e-6,
                                           include_survival=False).beta for xi in x])
    slope = float(np.polyfit(np.log(x), np.log(exc), 1)[0])
    ok &= _check("weak-drive quasistatic excitation wing exponent = -2",
                 abs(slope + 2.0) < 0.15, "%.3f" % slope)

    # beta itself is shallower than Delta^-2, because the survival factor rises
    # with detuning: the pair converts 2U into kinetic energy faster at large
    # detuning, where the Condon radius is smaller and the gradient steeper.
    b = np.array([model.beta(-xi * G, I_weak, 50e-6, regime="quasistatic")
                  for xi in x])
    slope_b = float(np.polyfit(np.log(x), np.log(b), 1)[0])
    ok &= _check("beta wing is shallower than the excitation wing",
                 -2.0 < slope_b < 0.0, "%.3f versus %.3f" % (slope_b, slope))

    ints = t.I_sat * np.array([1e-3, 1e-2, 1e-1])
    bl = np.array([model.beta(-3 * G, I, 50e-6, regime="quasistatic") for I in ints])
    lin = float(np.polyfit(np.log(ints), np.log(bl), 1)[0])
    ok &= _check("beta is linear in intensity well below saturation",
                 abs(lin - 1.0) < 0.1, "%.3f" % lin)
    ints_hi = t.I_sat * np.array([1e1, 1e2, 1e3])
    bh = np.array([model.beta(-3 * G, I, 50e-6, regime="quasistatic")
                   for I in ints_hi])
    sat = float(np.polyfit(np.log(ints_hi), np.log(bh), 1)[0])
    ok &= _check("beta saturates well above I_sat", sat < 0.7,
                 "log slope %.3f" % sat)

    depths = np.array([10e-6, 50e-6, 200e-6, 500e-6])
    bd = np.array([model.beta(-3 * G, I_weak, U, regime="quasistatic")
                   for U in depths])
    ok &= _check("beta decreases monotonically with trap depth",
                 bool(np.all(np.diff(bd) < 0)),
                 "%s cm^3/s" % np.array2string(bd, precision=2))

    # Optical shielding is a deep-trap effect: it appears once 2U approaches
    # hbar |Delta|, not generically.
    U_deep = 500e-6
    red_deep = model.beta(-1 * G, I_weak, U_deep, regime="quasistatic")
    blue_deep = model.beta(+1 * G, I_weak, U_deep, regime="quasistatic")
    ok &= _check("blue is shielded in a deep trap", red_deep / blue_deep > 2.0,
                 "red/blue = %.2f at U = %.0f uK" % (red_deep / blue_deep,
                                                     U_deep * 1e6))
    U_shallow = 10e-6
    red_sh = model.beta(-1 * G, I_weak, U_shallow, regime="quasistatic")
    blue_sh = model.beta(+1 * G, I_weak, U_shallow, regime="quasistatic")
    ok &= _check("shielding disappears in a shallow trap",
                 red_sh / blue_sh < 1.0,
                 "red/blue = %.2f at U = %.0f uK" % (red_sh / blue_sh,
                                                     U_shallow * 1e6))
    return ok


def check_coupled_dipole(transition=None, cloud=None, n_atoms=400):
    """Dilute limit reduces to a single-atom Lorentzian; dense limit matches
    the Lorentz-Lorenz continuum shift in sign and order of magnitude."""
    print("coupled dipole")
    from .coupled_dipole import CoupledDipole, lorentz_lorenz_shift_Hz
    t = transition or CyclingTransition.at_field(REF_B_GAUSS)
    ok = True

    dilute_density = 1e16       # m^-3, rho/k^3 ~ 2e-5
    cd = CoupledDipole(t, dilute_density, n_atoms=n_atoms)
    G = t.linewidth_Hz
    det = np.linspace(-4 * G, 4 * G, 81)
    y = cd.lineshape(det)
    y = y / y.max()
    peak = int(np.argmax(y))
    half = np.interp(0.5, y[: peak + 1], det[: peak + 1])
    fwhm = 2.0 * abs(det[peak] - half)
    ok &= _check("dilute lineshape has the single-atom linewidth",
                 abs(fwhm / G - 1.0) < 0.15, "FWHM = %.3f linewidths" % (fwhm / G))

    dense = cloud or BECCloud(N=1e5, trap_frequencies_Hz=(150.0, 150.0, 20.0),
                              a_s_bohr=100.0)
    ll = lorentz_lorenz_shift_Hz(dense.peak_density, t)
    ok &= _check("Lorentz-Lorenz shift is red and of order the linewidth",
                 ll < 0 and 0.05 < abs(ll / G) < 5.0,
                 "%.3f linewidths at %.2e cm^-3" % (ll / G, dense.peak_density / 1e6))

    cdd = CoupledDipole.from_cloud(t, dense, n_atoms=n_atoms)
    near = cdd.neighbour_statistics(-0.3 * G)
    far = cdd.neighbour_statistics(-10.0 * G)
    ok &= _check("more neighbours inside the Condon surface close to resonance",
                 near["mean_neighbours"] > far["mean_neighbours"],
                 "%.3f at 0.3 linewidths vs %.3f at 10"
                 % (near["mean_neighbours"], far["mean_neighbours"]))
    return ok


def run_all_checks(verbose=True):
    """Run every check.  Returns True if all pass.

    Builds the atomic structure once and shares it, since the ARC lookup dominates
    the runtime.
    """
    t, pot, cloud, model = _default_system()
    results = [
        ("green tensor", check_green_tensor_limits()),
        ("atomic structure", check_atomic_structure(t)),
        ("beam geometry", check_beam_geometry(t)),
        ("Condon radii", check_condon_radii(t, pot)),
        ("flight time", check_flight_time(pot, t)),
        ("beta scalings", check_beta_scalings(model, t)),
        ("coupled dipole", check_coupled_dipole(t, cloud)),
    ]
    passed = all(r for _, r in results)
    if verbose:
        print("")
        for name, r in results:
            print("%-20s %s" % (name, "PASS" if r else "FAIL"))
        print("")
        print("ALL PASS" if passed else "SOME CHECKS FAILED")
    return passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all_checks() else 1)
