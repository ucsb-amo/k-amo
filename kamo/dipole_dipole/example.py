"""End-to-end example: light-assisted loss for the k-team K-39 BEC at 520.6 G.

Run with

    python -m kamo.dipole_dipole.example

Prints the transition parameters, the cloud, the loss rate coefficient across
+/-10 linewidths at three intensities and three trap depths, the Condon-lobe
geometry, and a predicted atom-number decay.  Edit DEFAULTS below, or import
build_system() and drive it yourself.
"""

from __future__ import annotations

import numpy as np

from .cloud import BECCloud
from .coupled_dipole import CoupledDipole
from .dynamics import initial_loss_rate, integrate_loss
from .light_assisted_collisions import QuasiStaticLZModel
from .pair import ANTISYMMETRIC, SYMMETRIC, PairPotential
from .transition import CyclingTransition

DEFAULTS = dict(
    B_gauss=520.6,
    N=1e5,
    trap_frequencies_Hz=(150.0, 150.0, 20.0),
    a_s_bohr=100.0,
    # beam perpendicular to B, polarised along (k x B): half the intensity on sigma
    k_hat=(1.0, 0.0, 0.0),
    eps=(0.0, 1.0, 0.0),
)


def build_system(**overrides):
    """Return (transition, potential, cloud, model) for the default configuration."""
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    transition = CyclingTransition.at_field(cfg["B_gauss"])
    potential = PairPotential(transition)
    cloud = BECCloud(N=cfg["N"], trap_frequencies_Hz=cfg["trap_frequencies_Hz"],
                     a_s_bohr=cfg["a_s_bohr"])
    model = QuasiStaticLZModel(transition, potential, cloud,
                               k_hat=cfg["k_hat"], eps=cfg["eps"])
    return transition, potential, cloud, model


def main():
    transition, potential, cloud, model = build_system()
    G = transition.linewidth_Hz
    print(transition.summary())
    print()
    print(cloud.summary())
    print()
    print("driven fraction of the total intensity: %.3f (beam perpendicular to B)"
          % model.drive_fraction)
    print()

    detunings = np.array([-10, -3, -1, -0.3, 0.3, 1, 3, 10], dtype=float)
    intensities = transition.I_sat * np.array([0.01, 0.1, 1.0])
    depths = np.array([10e-6, 50e-6, 500e-6])

    print("beta (cm^3/s), rows are detuning in linewidths, U = 50 uK")
    print("%10s" % "Delta/G" + "".join("%14s" % ("I=%.2g Isat" % (I / transition.I_sat))
                                       for I in intensities))
    for d in detunings:
        row = "".join("%14.3e" % model.beta(d * G, I, 50e-6) for I in intensities)
        print("%10.1f" % d + row)
    print()

    print("trap-depth dependence at I = 0.1 I_sat")
    print("%10s" % "Delta/G" + "".join("%14s" % ("U=%.0f uK" % (U * 1e6))
                                       for U in depths) + "%10s" % "red/blue")
    for d in (1.0, 3.0, 10.0):
        vals = [model.beta(-d * G, 0.1 * transition.I_sat, U) for U in depths]
        blue = model.beta(+d * G, 0.1 * transition.I_sat, 0.1 * 500e-6)
        row = "".join("%14.3e" % v for v in vals)
        ratio = vals[1] / model.beta(+d * G, 0.1 * transition.I_sat, 50e-6)
        print("%10.1f" % d + row + "%10.2f" % ratio)
    print()

    print("Condon lobes and pair-picture validity")
    print("%10s %14s %14s %16s" % ("Delta/G", "R_C(0 deg) nm", "R_C(90 deg) nm",
                                   "neighbours in R_C"))
    for d in (-10.0, -3.0, -1.0, -0.3):
        th, rc_a = model.condon_surface(d * G, ANTISYMMETRIC)
        _, rc_s = model.condon_surface(d * G, SYMMETRIC)
        r0 = np.nanmax([rc_a[0], rc_s[0]]) * 1e9
        mid = len(th) // 2
        r90 = np.nanmax([rc_a[mid], rc_s[mid]]) * 1e9
        n = max(model.pair_fraction_inside_condon(d * G, branch=SYMMETRIC),
                model.pair_fraction_inside_condon(d * G, branch=ANTISYMMETRIC))
        print("%10.1f %14.1f %14.1f %16.3f" % (d, r0, r90, n))
    print()

    cd = CoupledDipole.from_cloud(transition, cloud, n_atoms=800)
    print(cd.summary(-1.0 * G))
    print()

    I = 0.1 * transition.I_sat
    beta = model.beta(-3 * G, I, 50e-6)
    tau = initial_loss_rate(cloud, beta)
    print("at -3 linewidths, I = 0.1 I_sat, U = 50 uK:")
    print("  beta          = %.3e cm^3/s" % beta)
    print("  initial 1/e   = %.1f us" % (tau * 1e6))
    t, N = integrate_loss(cloud, model, -3 * G, I, 5.0 * tau, 50e-6,
                          beta_cm3_s=beta)
    for frac in (0.25, 0.5, 1.0):
        idx = int(frac * (len(t) - 1))
        print("  N(%6.1f us)  = %.3e" % (t[idx] * 1e6, N[idx]))


if __name__ == "__main__":
    main()
