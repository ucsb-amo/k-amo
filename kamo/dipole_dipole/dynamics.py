"""Integrate two-body loss over a Thomas-Fermi cloud and a pulse train.

Thin wrapper over the rate coefficient: given beta, integrate

    dN/dt = -beta * integral of n^2 dV

keeping the Thomas-Fermi shape and letting the peak density follow N, since
n0 scales as N^(2/5).  The result is directly comparable to the atom-number and
APD pulse-decay traces recorded by the experiment.

One-body scattering (recoil heating, off-resonant depumping) is not included: this
module answers only what the dipole-dipole loss channel contributes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp


def _n2_integral_for_N(cloud, N):
    """Integral of n^2 dV for a Thomas-Fermi cloud rescaled to atom number N."""
    if N <= 0:
        return 0.0
    scaled = type(cloud)(
        N=N,
        trap_frequencies_Hz=cloud.trap_frequencies_Hz,
        a_s_bohr=cloud.a_s_bohr,
        mass=cloud.mass,
        temperature_K=cloud.temperature_K,
    )
    return scaled.density_squared_integral


def integrate_loss(cloud, model, detuning_Hz, intensity, duration,
                   trap_depth_K=50e-6, pulses: Optional[Sequence] = None,
                   n_eval=200, regime="auto", beta_cm3_s=None):
    """Atom number versus time under two-body light-assisted loss.

    Parameters
    ----------
    cloud : BECCloud
    model : QuasiStaticLZModel
        Used to evaluate beta once, at the given detuning and intensity.
    detuning_Hz, intensity : float
        Laser detuning (Hz) and total intensity (W/m^2).
    duration : float
        Total time to integrate over, in seconds.
    trap_depth_K : float
    pulses : sequence of (t_start, t_stop), optional
        Windows during which the light is on.  If omitted the light is on for the
        whole duration.
    n_eval : int
        Number of output samples.
    beta_cm3_s : float, optional
        Supply a precomputed beta (cm^3/s) to skip the model evaluation.

    Returns
    -------
    (t, N) : ndarrays of shape (n_eval,).  Times in seconds.
    """
    if beta_cm3_s is None:
        beta_cm3_s = model.beta(detuning_Hz, intensity, trap_depth_K, regime=regime)
    beta_si = float(beta_cm3_s) * 1e-6            # cm^3/s -> m^3/s

    windows = list(pulses) if pulses is not None else [(0.0, float(duration))]

    def light_on(t):
        return any(t0 <= t <= t1 for (t0, t1) in windows)

    def rhs(t, y):
        N = max(float(y[0]), 0.0)
        if N <= 0.0 or not light_on(t):
            return [0.0]
        return [-beta_si * _n2_integral_for_N(cloud, N)]

    t_eval = np.linspace(0.0, float(duration), int(n_eval))
    sol = solve_ivp(rhs, (0.0, float(duration)), [cloud.N], t_eval=t_eval,
                    rtol=1e-8, atol=1.0, max_step=float(duration) / 200.0)
    return sol.t, sol.y[0]


def initial_loss_rate(cloud, beta_cm3_s):
    """Initial 1/e time of the atom number, in seconds, for a given beta."""
    beta_si = float(beta_cm3_s) * 1e-6
    rate = beta_si * cloud.density_squared_integral / cloud.N
    return 1.0 / rate if rate > 0 else np.inf
