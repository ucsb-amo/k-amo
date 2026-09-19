"""Statistics for heavy-tailed observables.

The near-field shift distribution goes as ``P(omega) ~ 1/omega^2`` (Levy-type,
divergent variance), so sample means of anything tail-weighted converge slowly
and are dominated by rare close pairs.  Rules built in here:

* always report the **median and the trimmed mean** next to the mean.  The
  median is the typical shot; the mean is what the photon budget pays.  Neither
  alone is "the" answer;
* report the number of configurations and the spread with every number;
* 10-20 configurations per angle suffice for medians and trimmed means; mean
  excesses need >~ 50 for ~15%.  :func:`check_precision` warns when the sample
  does not support the precision asked for;
* **never fit one global line through an S_z scan spanning several CSS angles**
  -- the response is not odd in ``S_z`` once propagation is included and a
  global fit manufactures a slope and an intercept.  :func:`fit_within_groups`
  fits within each angle group against the binomial ``S_z`` scatter;
* express readout quantities in atom-equivalent units (:mod:`.detect`);
* the near-field share of configuration noise is
  ``sqrt(var_full - var_far)`` within an angle group (:func:`near_field_noise`);
* fix and record seeds so every figure is reproducible from saved positions and spins.
"""

from __future__ import annotations

import warnings
from typing import Dict, Sequence

import numpy as np
from scipy import stats as sps


class PrecisionWarning(UserWarning):
    pass


def trimmed_mean(x, trim: float = 0.1) -> float:
    """Mean after dropping the ``trim`` fraction at each end."""
    return float(sps.trim_mean(np.asarray(x, dtype=float), trim))


def robust_summary(x, trim: float = 0.1) -> Dict[str, float]:
    """mean, sem, std, median, trimmed mean, quartiles, n -- report all of them."""
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    out = dict(n=n, mean=float(a.mean()), std=float(a.std(ddof=1)) if n > 1 else float("nan"),
               median=float(np.median(a)), trimmed_mean=trimmed_mean(a, trim) if n > 2 else float(a.mean()),
               p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
               min=float(a.min()), max=float(a.max()))
    out["sem"] = out["std"] / np.sqrt(n) if n > 1 else float("nan")
    out["median_err"] = 1.2533 * out["sem"] if n > 1 else float("nan")   # ~ for a normal core
    return out


def format_summary(s: Dict[str, float], name: str = "", fmt: str = ".3f") -> str:
    return (f"{name:>24s}  mean {s['mean']:{fmt}} +- {s['sem']:{fmt}}   median {s['median']:{fmt}}"
            f"   trimmed {s['trimmed_mean']:{fmt}}   std {s['std']:{fmt}}   n = {s['n']}")


def check_precision(x, target_relative: float, name: str = "quantity") -> bool:
    """Warn when the standard error of the mean exceeds ``target_relative * |mean|``,
    and say how many configurations would be needed."""
    a = np.asarray(x, dtype=float).ravel()
    if a.size < 2:
        warnings.warn(f"{name}: one sample, no error estimate", PrecisionWarning, stacklevel=2)
        return False
    sem = a.std(ddof=1) / np.sqrt(a.size)
    rel = sem / max(abs(a.mean()), 1e-300)
    if rel > target_relative:
        need = int(np.ceil(a.size * (rel / target_relative) ** 2))
        warnings.warn(f"{name}: relative standard error {rel:.1%} exceeds the requested "
                      f"{target_relative:.1%} with n = {a.size}; ~{need} configurations needed "
                      "(heavy tails may make even that optimistic)", PrecisionWarning, stacklevel=2)
        return False
    return True


def fit_within_groups(group, S_z, y) -> Dict[float, dict]:
    """Linear fit ``y = a + b S_z`` INSIDE each group (e.g. each CSS angle).

    Returns ``{group: dict(slope, slope_err, intercept, intercept_err, n, S_z_std)}``.
    Groups with fewer than three distinct ``S_z`` values get NaN slopes rather
    than a fit through two points.  Refuses to be used as a global fit: pass
    one group per angle.
    """
    group = np.asarray(group)
    S_z = np.asarray(S_z, dtype=float)
    y = np.asarray(y, dtype=float)
    out = {}
    for g in np.unique(group):
        m = group == g
        s, v = S_z[m], y[m]
        rec = dict(n=int(m.sum()), S_z_std=float(s.std()), S_z_mean=float(s.mean()),
                   y_mean=float(v.mean()), slope=float("nan"), slope_err=float("nan"),
                   intercept=float("nan"), intercept_err=float("nan"))
        if np.unique(s).size >= 3:
            (b, a), cov = np.polyfit(s, v, 1, cov=True)
            rec.update(slope=float(b), intercept=float(a), slope_err=float(np.sqrt(cov[0, 0])),
                       intercept_err=float(np.sqrt(cov[1, 1])))
        out[float(g)] = rec
    return out


def near_field_noise(y_full, y_far) -> float:
    """``sqrt(max(var_full - var_far, 0))``: the near-field share of the scatter
    within one angle group (both arrays over the same configurations)."""
    vf = np.var(np.asarray(y_full, dtype=float), ddof=1)
    va = np.var(np.asarray(y_far, dtype=float), ddof=1)
    return float(np.sqrt(max(vf - va, 0.0)))


def binomial_S_z_std(N: int, theta: float) -> float:
    """Binomial scatter of ``S_z`` for a CSS: ``sqrt(N p (1 - p))`` with ``p = cos^2(theta/2)``."""
    p = np.cos(0.5 * theta) ** 2
    return float(np.sqrt(N * p * (1 - p)))
