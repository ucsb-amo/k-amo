"""Calibrate the coupled-channels potentials against measured resonance positions.

The Falke/Tiemann K2 curves fix everything except a small uncertainty in the
short-range phase of each spin state, i.e. in the singlet and triplet
scattering lengths ``a_S``, ``a_T``.  This module fits those two numbers
(realised as the inner-wall corrections ``delta_S``, ``delta_T`` of
:class:`~kamo.scattering.coupled_channels.CoupledChannels`) so that the
coupled-channels pole positions reproduce the measured Feshbach resonances of
:mod:`kamo.scattering.data.k39_feshbach`::

    chi^2 = sum_j [ (B0_cc,j(a_S, a_T) - B0_exp,j) / sigma_j ]^2 ,
    sigma_j^2 = sigma_exp,j^2 + sigma_model^2

``sigma_model`` (default 20 mG) stands in for what the s-wave model leaves
out: magnetic dipole-dipole / second-order spin-orbit couplings and higher
partial waves, which shift s-wave poles by up to a few 10 mG.

Only pole positions are fitted.  Zero crossings are NOT: with (a_S, a_T) as
the only freedom, the pole-calibrated model puts the |1,1> 350.4 G and |1,0>
393.2 G zeros ~0.5 G high while matching 490.1 and 504.9 G.  (High-field poles
depend almost only on a_S, zeros also on a_T; no (a_S, a_T) fixes them all
without spoiling the 33.582 G pole.)  Etrych's MOLSCAT misses in the same
direction.  In ``a`` this is <~0.4 a0; the empirical backend is exact at the
measured zeros.

>>> from kamo.scattering.calibration import calibrate
>>> res = calibrate()                 # ~ a few minutes
>>> res.a_S, res.a_T, res.chi2_red
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .resonances import locate_pole, refine_pole
from .data import k39_feshbach as kf


@dataclass
class CalibrationResult:
    a_S: float
    a_T: float
    a_S_unc: float
    a_T_unc: float
    corr: float                  # correlation coefficient of (a_S, a_T)
    delta_S: float
    delta_T: float
    chi2: float
    dof: int
    table: List[dict] = field(default_factory=list)

    @property
    def chi2_red(self) -> float:
        return self.chi2 / max(self.dof, 1)

    def summary(self) -> str:
        lines = [f"a_S = {self.a_S:.3f} +- {self.a_S_unc:.3f} a0   "
                 f"a_T = {self.a_T:.3f} +- {self.a_T_unc:.3f} a0   (corr {self.corr:+.2f})",
                 f"delta_S = {self.delta_S:.5e}, delta_T = {self.delta_T:.5e} Hartree/a0^2",
                 f"chi2 = {self.chi2:.2f} for {self.dof} dof (chi2/dof = {self.chi2_red:.2f})",
                 f"{'channel':16s} {'B0_exp':>10s} {'unc':>6s} {'B0_cc':>10s} {'cc-exp':>8s} source"]
        for row in self.table:
            lines.append(f"{row['label']:16s} {row['B0_exp']:10.4f} {row['unc']:6.4f} "
                         f"{row['B0_cc']:10.4f} {row['B0_cc'] - row['B0_exp']:+8.4f} {row['source']}")
        return "\n".join(lines)


def _channel_fn(cc, r):
    return lambda B: cc.scattering_length(r.state_a, r.state_b, B).real


def resonance_positions(cc, resonances: Sequence[kf.MeasuredResonance],
                        guesses: Optional[Sequence[float]] = None) -> np.ndarray:
    """Coupled-channels pole positions (G) nearest each resonance's ``B0``.

    With ``guesses`` (previous positions) only a cheap local refinement is done.
    """
    out = []
    for k, r in enumerate(resonances):
        fn = _channel_fn(cc, r)
        if guesses is not None:
            out.append(refine_pole(fn, guesses[k]))
        else:
            w = abs(r.width_theory) if r.width_theory else 1.0
            out.append(locate_pole(fn, r.B0, half_window=1.5, dB=min(0.05, w / 5)))
    return np.array(out)


def calibrate(resonances: Optional[Sequence[kf.MeasuredResonance]] = None, cc=None,
              sigma_model: float = 0.02, x0: Optional[Sequence[float]] = None,
              verbose: bool = True) -> CalibrationResult:
    """Fit ``(a_S, a_T)`` of the Tiemann coupled-channels model to measured ``B0``.

    Parameters
    ----------
    resonances : measured resonances (default :func:`k39_feshbach.fit_set`).
    cc : a :class:`CoupledChannels` instance (default: a new one, Tiemann).
    sigma_model : model uncertainty added in quadrature to each B0 error (G).
    x0 : starting ``(a_S, a_T)``; default is the current potentials' values.

    Returns a :class:`CalibrationResult`; ``cc`` is left set to the best fit.
    Uncertainties are from the fit covariance, scaled by sqrt(chi2/dof) when
    that exceeds 1.
    """
    from scipy.optimize import least_squares
    from .coupled_channels import CoupledChannels

    res_list = list(resonances) if resonances is not None else kf.fit_set()
    if cc is None:
        cc = CoupledChannels()
    B_exp = np.array([r.B0 for r in res_list])
    sig = np.sqrt(np.array([r.B0_unc for r in res_list]) ** 2 + sigma_model ** 2)

    if x0 is None:
        x0 = cc.singlet_triplet_a()
    cc.set_scattering_lengths(*x0, tol=1e-9)
    state = {"guess": resonance_positions(cc, res_list)}

    def resid(x):
        cc.set_scattering_lengths(x[0], x[1], tol=1e-9)
        pos = resonance_positions(cc, res_list, guesses=state["guess"])
        state["guess"] = pos
        r = (pos - B_exp) / sig
        if verbose:
            print(f"  a_S={x[0]:.4f} a_T={x[1]:.4f}  chi2={np.sum(r**2):.3f}", flush=True)
        return r

    sol = least_squares(resid, np.asarray(x0, float), x_scale=[0.05, 0.02],
                        diff_step=[2e-5, 2e-5], method="lm")
    resid(sol.x)                       # leave cc and the guesses at the optimum
    chi2 = float(np.sum(sol.fun ** 2))
    dof = len(res_list) - 2
    J = sol.jac
    cov = np.linalg.inv(J.T @ J)
    scale = max(1.0, chi2 / max(dof, 1))
    cov = cov * scale
    unc = np.sqrt(np.diag(cov))
    table = [dict(label=r.label, B0_exp=r.B0, unc=r.B0_unc, B0_cc=float(b), source=r.source)
             for r, b in zip(res_list, state["guess"])]
    return CalibrationResult(a_S=float(sol.x[0]), a_T=float(sol.x[1]),
                             a_S_unc=float(unc[0]), a_T_unc=float(unc[1]),
                             corr=float(cov[0, 1] / (unc[0] * unc[1])),
                             delta_S=cc.delta_S, delta_T=cc.delta_T,
                             chi2=chi2, dof=dof, table=table)
