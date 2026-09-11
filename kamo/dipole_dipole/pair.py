"""Two-atom molecular potentials, Condon surfaces and post-excitation flight.

For a closed sigma transition every atom shares one (complex, circular) dipole
orientation, so the pair problem depends only on the separation R and the angle
theta between the interatomic axis and the bias field.  The symmetric and
antisymmetric pair states carry shifts +J and -J and decay rates Gamma +/- Gamma_12
(see kamo.dipole_dipole.green_tensor).

Sign convention and the red/blue asymmetry
------------------------------------------
At every angle one branch is attractive and the other repulsive, so a resonance
exists for *either* sign of the laser detuning.  But the sign of the potential at
resonance always equals the sign of the detuning:

* Red detuning lands on the attractive branch.  The pair accelerates inward and
  can convert an unbounded amount of potential energy into kinetic energy.
* Blue detuning lands on the repulsive branch.  The pair flies apart and the total
  energy it can gain is capped at |A_b| / R0^3, which at resonance is hbar*|Delta|.

Optical shielding is therefore *not* automatic.  Loss only needs the pair to release
2 U, and when 2 U is much smaller than hbar*|Delta| the required excursion is a few
percent of R0 in either direction, so the two branches take almost the same time and
lose at almost the same rate.  Blue detuning is suppressed only once 2 U approaches
hbar*|Delta| -- deep traps, or detunings within about a linewidth.  See
kamo.dipole_dipole.light_assisted_collisions for the numbers.

The magic angle (arcsin(sqrt(2/3)) = 54.7 deg) is where the near-field coupling
vanishes, so the Condon radius diverges there and the Condon surfaces are lobes,
not spheres.

Retardation: potential curves use the full Green function.  The near-field 1/R^3
form is kept only for the analytic flight-time integral -- valid because the flight
happens at R < R0, where kR is smaller and the near field is *more* accurate -- and
for cross-checks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import beta as _beta_fn
from scipy.special import betainc as _betainc

from kamo import constants as c
from .green_tensor import MAGIC_ANGLE_RAD, dipole_projection, pair_branches

# Gauss-Legendre nodes for the numeric flight-time fallback.
_GL_N = 48
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_N)
_GL_X = 0.5 * (_GL_X + 1.0)     # map [-1, 1] -> [0, 1]
_GL_W = 0.5 * _GL_W

# (1/3) * B(5/6, 1/2), the full inward fall time in units of tau0.
_FALL_PREFACTOR = _beta_fn(5.0 / 6.0, 0.5) / 3.0

SYMMETRIC = +1
ANTISYMMETRIC = -1


class PairPotential:
    """Molecular potential curves for two atoms on a closed sigma transition.

    Parameters
    ----------
    transition : CyclingTransition
    reduced_mass : float, optional
        Defaults to m(K39) / 2.

    All potentials are returned as *angular frequencies* (rad/s) relative to the
    bare single-atom resonance, matching the detuning convention used everywhere in
    this package.  Multiply by hbar for an energy.
    """

    def __init__(self, transition, reduced_mass: Optional[float] = None):
        self.transition = transition
        if reduced_mass is None:
            reduced_mass = c.m_K / 2.0
        self.reduced_mass = float(reduced_mass)
        self._d_hat = transition.d_hat
        self.magic_angle_rad = MAGIC_ANGLE_RAD

    # ------------------------------------------------------------- geometry
    def p_of_theta(self, theta):
        """|dhat . rhat|^2 as a function of the angle to the bias field.

        Equals sin^2(theta)/2 for a sigma transition and cos^2(theta) for pi.
        Computed from the actual dipole unit vector rather than hard-coded.
        """
        theta = np.asarray(theta, dtype=float)
        r_hat = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)
        return dipole_projection(self._d_hat, r_hat)

    def angular_factor(self, theta):
        """1 - 3p, the near-field angular factor.

        Equals 1 - 1.5 sin^2(theta) for a sigma transition, and changes sign at the
        magic angle.
        """
        return 1.0 - 3.0 * self.p_of_theta(theta)

    # ----------------------------------------------------------- potentials
    def branches(self, R, theta):
        """Full (retarded) pair-state shifts and decay rates.

        Returns
        -------
        dict with V_sym, V_anti (rad/s) and Gamma_sym, Gamma_anti (rad/s),
        broadcast over R and theta.
        """
        xi = self.transition.k * np.asarray(R, dtype=float)
        p = self.p_of_theta(theta)
        return pair_branches(xi, p, self.transition.Gamma)

    def V(self, R, theta, branch=SYMMETRIC):
        """Pair-state shift (rad/s) of the symmetric (+1) or antisymmetric (-1) branch."""
        br = self.branches(R, theta)
        return br["V_sym"] if branch > 0 else br["V_anti"]

    def Gamma(self, R, theta, branch=SYMMETRIC):
        """Collective decay rate (rad/s) of the given branch."""
        br = self.branches(R, theta)
        return br["Gamma_sym"] if branch > 0 else br["Gamma_anti"]

    def C3(self, theta, branch=SYMMETRIC):
        """Signed near-field coefficient A_b in J*m^3, so that E_b(R) = A_b / R^3.

        A_b < 0 is attractive, A_b > 0 repulsive.
        """
        t = self.transition
        A_sym = c.hbar * 0.75 * t.Gamma * self.angular_factor(theta) / t.k ** 3
        return A_sym if branch > 0 else -A_sym

    # ------------------------------------------------------- Condon surfaces
    def condon_radius_near_field(self, detuning_Hz, theta, branch=SYMMETRIC):
        """Analytic near-field Condon radius, NaN where the branch has the wrong sign.

        Solves A_b / R^3 = hbar * 2 pi * detuning.  Use condon_radius for the
        retarded (correct) value.
        """
        delta = 2.0 * np.pi * np.asarray(detuning_Hz, dtype=float)
        A = self.C3(theta, branch) / c.hbar          # rad/s * m^3
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = A / delta
            R = np.where(ratio > 0, np.abs(ratio) ** (1.0 / 3.0), np.nan)
        return R

    def condon_radius(self, detuning_Hz, theta, branch=SYMMETRIC,
                      R_min=1e-9, R_max=5e-6, n_scan=4000):
        """Smallest R where the retarded branch potential equals the laser detuning.

        Scans a log grid from R_min to R_max and refines the first sign change of
        V_branch(R) - 2 pi detuning by linear interpolation in log R.  Returns NaN
        where no crossing exists (wrong branch sign, or near the magic angle where
        the coupling vanishes).

        Parameters
        ----------
        detuning_Hz : float
            Laser detuning from the bare atomic resonance.
        theta : array_like
            Angle(s) between the interatomic axis and the bias field.
        branch : +1 or -1
        """
        delta = 2.0 * np.pi * float(detuning_Hz)
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        R_grid = np.logspace(np.log10(R_min), np.log10(R_max), n_scan)
        Vg = self.V(R_grid[None, :], theta[:, None], branch) - delta     # (n_theta, n_R)
        out = np.full(theta.shape, np.nan)
        sign_change = np.sign(Vg[:, :-1]) * np.sign(Vg[:, 1:]) < 0
        lg = np.log(R_grid)
        for i in range(theta.size):
            idx = np.flatnonzero(sign_change[i])
            if idx.size == 0:
                continue
            j = idx[0]
            v0, v1 = Vg[i, j], Vg[i, j + 1]
            out[i] = float(np.exp(lg[j] + (lg[j + 1] - lg[j]) * (-v0) / (v1 - v0)))
        return out

    # ------------------------------------------------ flight after excitation
    def flight_time(self, R0, theta, branch, energy_gain_J, E_kin0_J=0.0,
                    exact=False):
        """Time to convert energy_gain_J of potential energy into kinetic energy.

        The pair is created at separation R0 on the given branch (at rest unless
        E_kin0_J is supplied) and moves on the near-field A_b / R^3 curve: inward
        if the branch is attractive, outward if repulsive.

        Returns
        -------
        ndarray : time in seconds, or inf where the requested energy gain is
        unreachable (a repulsive branch can release at most |A_b| / R0^3).

        Notes
        -----
        The inward, zero-initial-velocity case has the closed form

            t = tau0 * (1/3) B(5/6, 1/2) [1 - I_{u^3}(5/6, 1/2)]
            tau0 = R0 sqrt(mu / (2 E0)),   u = R_target / R0

        which is the default because the loss integral evaluates it on a large
        (R, theta, phi) grid.  Setting exact=True, or a non-zero E_kin0_J, switches
        to a 48-node Gauss-Legendre quadrature of the same integral.
        """
        R0 = np.asarray(R0, dtype=float)
        A = self.C3(theta, branch)                       # J*m^3, signed
        A, R0 = np.broadcast_arrays(A, R0)
        E0 = np.abs(A) / R0 ** 3                         # |potential energy| at R0
        W = np.asarray(energy_gain_J, dtype=float)
        Ek0 = np.asarray(E_kin0_J, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(E0 > 0, W / E0, np.inf)

        attractive = A < 0
        u_in = (1.0 + w) ** (-1.0 / 3.0)                 # < 1, always defined
        with np.errstate(invalid="ignore"):
            u_out = np.where(w < 1.0, (1.0 - w) ** (-1.0 / 3.0), np.inf)

        tau0 = R0 * np.sqrt(self.reduced_mass / (2.0 * np.maximum(E0, 1e-300)))

        if not exact and np.all(Ek0 == 0.0):
            t_in = tau0 * _FALL_PREFACTOR * (
                1.0 - _betainc(5.0 / 6.0, 0.5, np.clip(u_in ** 3, 0.0, 1.0)))
            t_out = self._outward_time(tau0, u_out)
            return np.where(attractive, t_in, t_out)

        return self._numeric_time(tau0, E0, u_in, u_out, attractive, Ek0)

    def _outward_time(self, tau0, u_end):
        """tau0 * integral from 1 to u_end of u^{3/2} (u^3 - 1)^{-1/2} du.

        Substitutes u = 1 + (u_end - 1) s^2 to remove the inverse-square-root
        singularity at u = 1.
        """
        u_end = np.asarray(u_end, dtype=float)
        finite = np.isfinite(u_end) & (u_end > 1.0)
        span = np.where(finite, u_end - 1.0, 0.0)
        shape = (-1,) + (1,) * np.ndim(u_end)
        s = _GL_X.reshape(shape)
        w = _GL_W.reshape(shape)
        u = 1.0 + span * s ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = u ** 1.5 / np.sqrt(np.maximum(u ** 3 - 1.0, 1e-300))
            jac = 2.0 * span * s
            val = np.sum(w * integrand * jac, axis=0)
        return np.where(finite, tau0 * val, np.inf)

    def _numeric_time(self, tau0, E0, u_in, u_out, attractive, Ek0):
        """Quadrature that also handles a non-zero initial relative kinetic energy."""
        u_end = np.where(attractive, u_in, u_out)
        finite = np.isfinite(u_end)
        u_end_safe = np.where(finite, u_end, 1.0)
        span = u_end_safe - 1.0
        shape = (-1,) + (1,) * np.ndim(u_end)
        s = _GL_X.reshape(shape)
        w = _GL_W.reshape(shape)
        u = 1.0 + span * s ** 2
        jac = np.abs(2.0 * span * s)
        with np.errstate(divide="ignore", invalid="ignore"):
            dE = E0 * np.abs(1.0 - u ** -3.0) + Ek0
            v = np.sqrt(2.0 * np.maximum(dE, 1e-300) / self.reduced_mass)
            val = np.sum(w * jac / v, axis=0)
        R0 = tau0 * np.sqrt(2.0 * np.maximum(E0, 1e-300) / self.reduced_mass)
        return np.where(finite, R0 * val, np.inf)
