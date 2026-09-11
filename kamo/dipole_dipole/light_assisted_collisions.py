"""Two-body loss from light-assisted collisions on a closed sigma transition.

Physical picture
----------------
A pair of ground-state atoms at separation R is shifted out of resonance by the
resonant dipole-dipole interaction.  Where the shift matches the laser detuning the
pair is promoted onto a molecular potential curve, accelerates, and converts
potential energy into kinetic energy.  If it gains more than twice the trap depth
before spontaneously decaying, both atoms are lost.

Two regimes
-----------
* **quasistatic** -- the pair barely moves while the excitation happens, so the
  excitation rate is a saturated Lorentzian evaluated at the shifted resonance.
  This is the correct picture for a condensate, where relative velocities are of
  order 1 cm/s.
* **flux** -- the Gallagher-Pritchard / Julienne-Vigue picture: pairs stream through
  the Condon surface at velocity v and are excited with a Landau-Zener probability
  1 - exp(-pi Omega^2 / (2 alpha)), alpha = |dV/dR| v / hbar.  Appropriate for a hot,
  dilute sample.

The default regime is chosen from the adiabaticity parameter
``Gamma_b^2 hbar / (|dV/dR| v)``: large means the pair dwells on resonance for many
lifetimes, so quasistatic applies.

Saturation enters non-perturbatively in both regimes -- through the ``Omega^2/2``
term of the saturated Lorentzian, and through the exponential of the Landau-Zener
probability, which tends to 1 at large Rabi frequency.

What sets the red/blue asymmetry
--------------------------------
Naively red detuning (attractive branch) should dominate and blue should be shielded.
That is only true for deep traps.  Loss requires releasing 2 U, and when
2 U << hbar |Delta| the pair only has to move a few percent of the Condon radius,
which takes the same time inward or outward.  Two effects then actually favour blue:

* the near-field angular factor |1 - 1.5 sin^2 theta| is largest along the field
  axis, so the polar Condon lobes reached by blue detuning are bigger than the
  equatorial band reached by red;
* at kR < 1 the drive couples mainly to the symmetric pair state, and for a sigma
  transition the symmetric state is the *repulsive* one along the field axis.

At 520.6 G with a 1e5-atom K-39 condensate this model gives red/blue of about 0.25
at 1 linewidth and 10 uK depth, rising to about 4 at 500 uK.  Shielding is a
deep-trap phenomenon here, not a generic one.

Open loss channels at high field
--------------------------------
Radiative escape (RE) and fine-structure-changing collisions (FCC, releasing the
4P3/2 - 4P1/2 splitting) are the only open channels for the k-team geometry.
Zeeman-changing exits are endothermic: mJ = -1/2 is the lowest ground Zeeman state
and mJ = -3/2 the lowest excited one, so those channels cost ~730 MHz and ~970 MHz
respectively.  Nuclear-spin exchange releases only ~100 kHz (about 5 uK).

Drive geometry
--------------
The pair states are the symmetric and antisymmetric combinations, and the drive
couples to them with amplitudes sqrt(2) Omega cos(phi_k) and sqrt(2) Omega
sin(phi_k), where phi_k = k . (r1 - r2) / 2.  At the Condon radius kR is 0.4 to 1.4,
so this phase is *not* small and both branches are driven.  The beam direction
therefore enters through a third angle, and the loss integral runs over
(R, theta, phi_azimuthal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from kamo import constants as c
from .pair import ANTISYMMETRIC, SYMMETRIC


@dataclass
class LossChannels:
    """Breakdown of a beta evaluation.  Rate coefficients are in cm^3/s."""

    beta: float
    beta_attractive: float
    beta_repulsive: float
    beta_fine_structure: float
    beta_symmetric: float
    beta_antisymmetric: float
    regime: str
    adiabaticity: float
    detuning_Hz: float
    intensity: float
    trap_depth_K: float

    def __str__(self) -> str:
        return "\n".join([
            "beta = %.4e cm^3/s   (detuning %.3f MHz, I = %.4g W/m^2, U = %.1f uK)"
            % (self.beta, self.detuning_Hz / 1e6, self.intensity,
               self.trap_depth_K * 1e6),
            "  regime            : %s (adiabaticity %.3g)" % (self.regime,
                                                              self.adiabaticity),
            "  attractive branch : %.4e cm^3/s" % self.beta_attractive,
            "  repulsive branch  : %.4e cm^3/s" % self.beta_repulsive,
            "  via fine structure: %.4e cm^3/s" % self.beta_fine_structure,
            "  symmetric branch  : %.4e cm^3/s" % self.beta_symmetric,
            "  antisym.  branch  : %.4e cm^3/s" % self.beta_antisymmetric,
        ])


class QuasiStaticLZModel:
    """Two-body loss rate coefficient beta(detuning, intensity; trap depth).

    Parameters
    ----------
    transition : CyclingTransition
    potential : PairPotential
    cloud : BECCloud
        Supplies the pair correlation g2 and the relative-velocity scale.
    k_hat, eps : 3-sequences
        Beam propagation direction and polarisation, in the frame where z is the
        bias field.  The default is a beam perpendicular to B, polarised along
        (k x B) -- which puts half the total intensity on the driven sigma channel.
    R_min, R_max : float
        Radial integration limits (m).  R_min should sit outside the range where
        the asymptotic dipole-dipole form breaks down; the integrand there is
        suppressed as R^9, so the result is insensitive to it.
    R_short : float
        Separation at which a fine-structure-changing collision occurs.
    n_R, n_theta, n_phi : int
        Quadrature resolution.  n_R must resolve the resonance width in R, which is
        of order R Gamma / (3 |Delta|).
    """

    def __init__(self, transition, potential, cloud,
                 k_hat: Sequence[float] = (1.0, 0.0, 0.0),
                 eps: Sequence[float] = (0.0, 1.0, 0.0),
                 R_min: float = 5e-9, R_max: float = 1.0e-6,
                 R_short: float = 2e-9,
                 n_R: int = 700, n_theta: int = 48, n_phi: int = 24):
        self.transition = transition
        self.potential = potential
        self.cloud = cloud
        self.k_hat = np.asarray(k_hat, dtype=float) / np.linalg.norm(k_hat)
        self.eps = np.asarray(eps, dtype=complex)
        self.drive_fraction = transition.drive_projection(k_hat, eps)[transition.q]
        self.R_min = float(R_min)
        self.R_max = float(R_max)
        self.R_short = float(R_short)
        self.n_R = int(n_R)
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self._grid_cache = None

    # ------------------------------------------------------------------ grid
    def _grid(self):
        """Quadrature grid and the R-, theta- and phi-dependent quantities.

        Symmetry lets us halve both angular ranges: the pair potential depends on
        theta only through sin^2(theta), and the drive phase only through
        |cos(phi)|.  Hence x = cos(theta) runs over [0, 1], phi over [0, pi], and
        the measure carries a factor of 4.
        """
        if self._grid_cache is not None:
            return self._grid_cache

        lnR = np.linspace(np.log(self.R_min), np.log(self.R_max), self.n_R)
        R = np.exp(lnR)
        # R^2 dR = R^3 d(lnR), trapezoid weights in lnR
        wR = np.gradient(lnR) * R ** 3

        x, wx = np.polynomial.legendre.leggauss(self.n_theta)
        x = 0.5 * (x + 1.0)                 # cos(theta) in [0, 1]
        wx = 0.5 * wx
        theta = np.arccos(np.clip(x, -1.0, 1.0))

        phi, wp = np.polynomial.legendre.leggauss(self.n_phi)
        phi = 0.5 * np.pi * (phi + 1.0)     # [0, pi]
        wp = 0.5 * np.pi * wp

        # broadcast shapes: (n_R, n_theta, n_phi)
        Rg = R[:, None, None]
        thg = theta[None, :, None]
        phg = phi[None, None, :]

        weight = 4.0 * wR[:, None, None] * wx[None, :, None] * wp[None, None, :]

        # k . rhat for a beam along k_hat with the bias field along z
        ones = np.ones(np.broadcast_shapes(Rg.shape, thg.shape, phg.shape))
        r_hat = np.stack([np.sin(thg) * np.cos(phg) * ones,
                          np.sin(thg) * np.sin(phg) * ones,
                          np.cos(thg) * ones], axis=-1)
        k_dot_r = np.tensordot(r_hat, self.k_hat, axes=([-1], [0]))
        phase = 0.5 * self.transition.k * Rg * k_dot_r

        self._grid_cache = dict(R=Rg, theta=thg, phi=phg, weight=weight,
                                phase=phase, g2=self.cloud.g2(Rg))
        return self._grid_cache

    # -------------------------------------------------------------- physics
    def rabi_total(self, intensity):
        """Single-atom Rabi frequency (rad/s) from the *total* beam intensity.

        The beam geometry projection onto the driven spherical component is applied
        here, so callers pass the intensity they actually measure.
        """
        return self.transition.rabi(np.asarray(intensity, dtype=float)
                                    * self.drive_fraction)

    def _branch_terms(self, branch, detuning_Hz, Omega):
        """Excitation rate and loss probability on one branch, on the grid.

        Returns
        -------
        (rate, Gamma_branch, V_branch) : each shaped like the grid.  ``rate`` is the
        saturated scattering rate of the pair on this branch, in 1/s.
        """
        g = self._grid()
        pot = self.potential
        R, theta, phase = g["R"], g["theta"], g["phase"]

        V = pot.V(R, theta, branch)                       # rad/s
        Gam_b = pot.Gamma(R, theta, branch)               # rad/s
        delta = 2.0 * np.pi * float(detuning_Hz) - V      # rad/s

        # sqrt(2) Omega cos(phase) for the symmetric branch, sin for antisymmetric
        proj = np.cos(phase) if branch > 0 else np.sin(phase)
        Om_b = np.sqrt(2.0) * Omega * proj

        rate = Gam_b * (Om_b ** 2 / 4.0) / (delta ** 2 + Gam_b ** 2 / 4.0
                                            + Om_b ** 2 / 2.0)
        return rate, Gam_b, V

    def _survival(self, branch, Gam_b, trap_depth_K):
        """Probability of surviving long enough to be lost, and the FCC share.

        Loss requires converting 2 U of potential energy into kinetic energy before
        spontaneous decay -- U for each atom, since the released energy is shared in
        the centre-of-mass frame.  A pair that instead reaches R_short exits through
        the fine-structure channel and is lost regardless of trap depth.
        """
        g = self._grid()
        R, theta = g["R"], g["theta"]
        pot = self.potential

        W = 2.0 * c.kB * float(trap_depth_K)
        t_U = pot.flight_time(R, theta, branch, W)

        # time to reach R_short (inward only); infinite on a repulsive branch
        A = pot.C3(theta, branch)
        E0 = np.abs(A) / R ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            W_short = E0 * ((R / self.R_short) ** 3 - 1.0)
        W_short = np.where(R > self.R_short, W_short, 0.0)
        t_fcc = pot.flight_time(R, theta, branch, W_short)
        t_fcc = np.where(A < 0, t_fcc, np.inf)

        t_eff = np.minimum(t_U, t_fcc)
        p_loss = np.where(np.isfinite(t_eff), np.exp(-Gam_b * t_eff), 0.0)
        p_fcc = np.where(np.isfinite(t_fcc), np.exp(-Gam_b * t_fcc), 0.0)
        return p_loss, np.minimum(p_fcc, p_loss), (A < 0)

    def adiabaticity(self, detuning_Hz, branch=SYMMETRIC, theta=0.0):
        """Gamma^2 hbar / (|dV/dR| v_rel), evaluated at the Condon radius.

        Much greater than 1 means the pair dwells on resonance for many excited-state
        lifetimes, so the quasistatic treatment applies; much less than 1 means the
        pair sweeps through and the Landau-Zener (flux) treatment applies.
        """
        pot = self.potential
        R_C = np.asarray(pot.condon_radius(detuning_Hz, theta, branch))
        if not np.all(np.isfinite(R_C)):
            # fall back to the other branch, which always has the right sign
            R_C = np.asarray(pot.condon_radius(detuning_Hz, theta, -branch))
        R_C = float(np.nanmax(R_C))
        if not np.isfinite(R_C):
            return np.inf
        delta = 2.0 * np.pi * abs(float(detuning_Hz))
        dVdR = 3.0 * delta / R_C                          # rad/s per m, near field
        v = self.cloud.relative_velocity()
        return float(self.transition.Gamma ** 2 / (dVdR * v))

    # ----------------------------------------------------------------- beta
    def beta_quasistatic(self, detuning_Hz, intensity, trap_depth_K=50e-6,
                         include_survival=True):
        """Quasistatic two-body loss rate coefficient, with a channel breakdown.

        Evaluates

            beta = integral over d^3r of  g2(r) * sum_b R_exc,b(r) * P_loss,b(r)

        where R_exc is a saturated Lorentzian at the dipole-shifted resonance and
        P_loss is the probability of surviving long enough to release 2 U (or to
        reach the fine-structure crossing).  The factor convention is such that
        dn/dt = -beta n^2 removes two atoms per event.

        Parameters
        ----------
        detuning_Hz : float
            Laser detuning from the bare single-atom resonance.
        intensity : float
            *Total* beam intensity in W/m^2 (the polarisation projection is applied
            internally).
        trap_depth_K : float
            Trap depth per atom, in kelvin.
        include_survival : bool
            Set False to drop the survival factor, which turns the result into the
            integrated pair excitation rate rather than a loss rate.  Diagnostic
            only: it depends on R_max, and in the weak-drive limit it recovers the
            analytic quasistatic wing scaling of Delta^-2.

        Returns
        -------
        LossChannels
        """
        g = self._grid()
        Omega = self.rabi_total(intensity)
        measure = g["weight"] * g["g2"]

        parts = {}
        b_att = b_rep = b_fcc = 0.0
        for branch, name in ((SYMMETRIC, "sym"), (ANTISYMMETRIC, "anti")):
            rate, Gam_b, _ = self._branch_terms(branch, detuning_Hz, Omega)
            p_loss, p_fcc, attractive = self._survival(branch, Gam_b, trap_depth_K)
            if not include_survival:
                p_loss = np.ones_like(p_loss)
            integrand = measure * rate * p_loss
            parts[name] = float(np.sum(integrand))
            b_att += float(np.sum(np.where(attractive, integrand, 0.0)))
            b_rep += float(np.sum(np.where(attractive, 0.0, integrand)))
            b_fcc += float(np.sum(measure * rate * p_fcc))

        to_cm3 = 1e6
        total = (parts["sym"] + parts["anti"]) * to_cm3
        return LossChannels(
            beta=total,
            beta_attractive=b_att * to_cm3,
            beta_repulsive=b_rep * to_cm3,
            beta_fine_structure=b_fcc * to_cm3,
            beta_symmetric=parts["sym"] * to_cm3,
            beta_antisymmetric=parts["anti"] * to_cm3,
            regime="quasistatic",
            adiabaticity=self.adiabaticity(detuning_Hz),
            detuning_Hz=float(detuning_Hz),
            intensity=float(intensity),
            trap_depth_K=float(trap_depth_K),
        )

    def beta_flux(self, detuning_Hz, intensity, trap_depth_K=50e-6):
        """Gallagher-Pritchard / Julienne-Vigue flux form of the rate coefficient.

        Pairs stream through the Condon surface at the relative velocity of the
        sample and are promoted with a Landau-Zener probability
        1 - exp(-pi Omega_b^2 / (2 alpha)), alpha = |dV/dR| v_rel.  The result is

            beta = integral over solid angle of  R_C^2 v_rel P_LZ P_loss g2(R_C)

        which is accurate up to a geometric factor of order unity in the incoming
        flux.  Included for comparison with the quasistatic result; for a condensate
        the adiabaticity parameter is far above 1 and the quasistatic form is the
        correct one.
        """
        pot = self.potential
        v = self.cloud.relative_velocity()
        Omega = self.rabi_total(intensity)
        W = 2.0 * c.kB * float(trap_depth_K)

        x, wx = np.polynomial.legendre.leggauss(self.n_theta)
        x = 0.5 * (x + 1.0)
        wx = 0.5 * wx
        theta = np.arccos(np.clip(x, -1.0, 1.0))
        phi, wp = np.polynomial.legendre.leggauss(self.n_phi)
        phi = 0.5 * np.pi * (phi + 1.0)
        wp = 0.5 * np.pi * wp

        b_att = b_rep = b_fcc = 0.0
        parts = {}
        for branch, name in ((SYMMETRIC, "sym"), (ANTISYMMETRIC, "anti")):
            R_C = pot.condon_radius(detuning_Hz, theta, branch)      # (n_theta,)
            ok = np.isfinite(R_C)
            R_Cs = np.where(ok, R_C, 1.0)[:, None]
            th = theta[:, None]

            r_hat = np.stack([np.sin(th) * np.cos(phi[None, :]),
                              np.sin(th) * np.sin(phi[None, :]),
                              np.cos(th) * np.ones_like(phi[None, :])], axis=-1)
            phase = 0.5 * self.transition.k * R_Cs * np.tensordot(
                r_hat, self.k_hat, axes=([-1], [0]))
            proj = np.cos(phase) if branch > 0 else np.sin(phase)
            Om_b = np.sqrt(2.0) * Omega * proj

            delta = 2.0 * np.pi * abs(float(detuning_Hz))
            dVdR = 3.0 * delta / R_Cs                                # near field
            alpha = np.maximum(dVdR * v, 1e-300)
            p_lz = 1.0 - np.exp(-np.pi * Om_b ** 2 / (2.0 * alpha))

            Gam_b = pot.Gamma(R_Cs, th, branch)
            t_U = pot.flight_time(R_Cs, th, branch, W)
            A = pot.C3(th, branch)
            E0 = np.abs(A) / R_Cs ** 3
            with np.errstate(divide="ignore", invalid="ignore"):
                W_short = np.maximum(E0 * ((R_Cs / self.R_short) ** 3 - 1.0), 0.0)
            t_fcc = np.where(A < 0, pot.flight_time(R_Cs, th, branch, W_short), np.inf)
            t_eff = np.minimum(t_U, t_fcc)
            p_loss = np.where(np.isfinite(t_eff), np.exp(-Gam_b * t_eff), 0.0)
            p_fcc = np.where(np.isfinite(t_fcc), np.exp(-Gam_b * t_fcc), 0.0)

            w2 = 4.0 * wx[:, None] * wp[None, :] * np.where(ok, 1.0, 0.0)[:, None]
            integrand = w2 * R_Cs ** 2 * v * p_lz * p_loss * self.cloud.g2(R_Cs)
            parts[name] = float(np.sum(integrand))
            attractive = np.broadcast_to(A < 0, integrand.shape)
            b_att += float(np.sum(np.where(attractive, integrand, 0.0)))
            b_rep += float(np.sum(np.where(attractive, 0.0, integrand)))
            b_fcc += float(np.sum(w2 * R_Cs ** 2 * v * p_lz * p_fcc
                                  * self.cloud.g2(R_Cs)))

        to_cm3 = 1e6
        return LossChannels(
            beta=(parts["sym"] + parts["anti"]) * to_cm3,
            beta_attractive=b_att * to_cm3,
            beta_repulsive=b_rep * to_cm3,
            beta_fine_structure=b_fcc * to_cm3,
            beta_symmetric=parts["sym"] * to_cm3,
            beta_antisymmetric=parts["anti"] * to_cm3,
            regime="flux",
            adiabaticity=self.adiabaticity(detuning_Hz),
            detuning_Hz=float(detuning_Hz),
            intensity=float(intensity),
            trap_depth_K=float(trap_depth_K),
        )

    def beta(self, detuning_Hz, intensity, trap_depth_K=50e-6, regime="auto",
             full=False):
        """Two-body loss rate coefficient in cm^3/s.

        Parameters
        ----------
        detuning_Hz : float
            Laser detuning from the bare single-atom resonance.  Negative is red.
        intensity : float
            Total beam intensity, W/m^2.
        trap_depth_K : float
            Trap depth per atom (K).
        regime : {"auto", "quasistatic", "flux"}
            "auto" picks quasistatic when the adiabaticity parameter exceeds 1.
        full : bool
            Return the LossChannels breakdown instead of a bare float.
        """
        if regime == "auto":
            regime = "quasistatic" if self.adiabaticity(detuning_Hz) > 1.0 else "flux"
        if regime == "quasistatic":
            res = self.beta_quasistatic(detuning_Hz, intensity, trap_depth_K)
        elif regime == "flux":
            res = self.beta_flux(detuning_Hz, intensity, trap_depth_K)
        else:
            raise ValueError("regime must be auto, quasistatic or flux.")
        return res if full else res.beta

    def beta_grid(self, detunings_Hz, intensities, trap_depths_K=(50e-6,),
                  regime="auto"):
        """beta on a (detuning, intensity, trap depth) grid, in cm^3/s.

        Returns
        -------
        (n_detuning, n_intensity, n_depth) ndarray.

        Notes
        -----
        The trap-depth axis is usually nearly flat: hbar |Delta| / kB spans roughly
        90 uK to 3 mK over a +/-10 linewidth window on the K-39 D2 line, far above
        any optical-dipole-trap depth, so essentially every excited pair that
        survives long enough is lost.  The depth only matters within a fraction of a
        linewidth of resonance.
        """
        det = np.atleast_1d(np.asarray(detunings_Hz, dtype=float))
        ints = np.atleast_1d(np.asarray(intensities, dtype=float))
        depths = np.atleast_1d(np.asarray(trap_depths_K, dtype=float))
        out = np.empty((det.size, ints.size, depths.size))
        for i, d in enumerate(det):
            for j, I in enumerate(ints):
                for k, U in enumerate(depths):
                    out[i, j, k] = self.beta(d, I, U, regime=regime)
        return out

    def condon_surface(self, detuning_Hz, branch=SYMMETRIC, n_theta=181):
        """Condon radius versus polar angle, for plotting the Condon lobes.

        Returns
        -------
        (theta, R_C) with R_C in metres and NaN where the branch does not cross.
        """
        theta = np.linspace(0.0, np.pi, n_theta)
        return theta, self.potential.condon_radius(detuning_Hz, theta, branch)

    def pair_fraction_inside_condon(self, detuning_Hz, density=None,
                                    branch=SYMMETRIC, n_theta=181):
        """Mean number of neighbours inside the Condon surface.

        A value approaching or exceeding 1 means the two-body picture is breaking
        down and the N-atom coupled-dipole model
        (kamo.dipole_dipole.coupled_dipole) should be consulted.
        """
        if density is None:
            density = self.cloud.peak_density
        theta, R_C = self.condon_surface(detuning_Hz, branch, n_theta)
        R_C = np.nan_to_num(R_C, nan=0.0)
        # (1/3) integral of R_C^3 dOmega
        volume = float(np.trapezoid(R_C ** 3 * np.sin(theta), theta)) * 2.0 * np.pi / 3.0
        return float(volume * density)
