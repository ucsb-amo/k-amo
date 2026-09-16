"""Many configurations, in parallel, and the CSS-angle scan of the excess law.

Configurations are embarrassingly parallel and heavy-tailed statistics need
many of them, so this is the first optimisation: :func:`run_ensemble` farms the
solves out with joblib (loky processes, one BLAS thread each) before anything
else is made faster.

Everything a worker returns is a lightweight :class:`SolveResult` (no O(N^2)
matrices) carrying positions, spins and seeds, so any observable -- fields,
detected signals, per-atom diagnostics -- can be recomputed afterwards without
re-solving.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cloud import GaussianProfile, like_pair_fraction, sample_configuration
from .solver import SolveResult, solve
from .stats import robust_summary
from .system import IncidentField, OperatingPoint

XI_CIRC = 1.0 / (12 * np.pi * np.sqrt(3.0))    #: 0.0153140, the sigma- tail coefficient
XI_LIN = 1.0 / (6 * np.pi * np.sqrt(3.0))      #: the published linear-dipole value


def excess_law(eta_eff: float, theta, xi: float = XI_CIRC) -> np.ndarray:
    """``1 + xi eta_eff (1 - sin^2(theta) / 2)``."""
    return 1.0 + xi * eta_eff * like_pair_fraction(theta)


def _solve_one(profile, op, theta, seed, variants, incident, solve_kw, N):
    cfg = sample_configuration(profile, theta=theta, seed=seed, N=N)
    out = {}
    for v in variants:
        r = solve(cfg, op, v, incident=incident, keep_matrices=False, warn=False, **solve_kw)
        out[v] = r
    return out


@dataclass
class EnsembleResult:
    profile: GaussianProfile
    op: OperatingPoint
    theta: float
    seeds: np.ndarray
    variants: tuple
    results: Dict[str, List[SolveResult]]
    seconds: float = float("nan")

    @property
    def n_config(self) -> int:
        return int(self.seeds.size)

    def excitation(self, variant: str) -> np.ndarray:
        return np.array([r.excitation for r in self.results[variant]])

    def radiated(self, variant: str) -> np.ndarray:
        return np.array([r.radiated_power for r in self.results[variant]])

    def forward(self, variant: str) -> np.ndarray:
        return np.array([r.forward_amplitude() for r in self.results[variant]])

    @property
    def S_z(self) -> np.ndarray:
        v = self.variants[0]
        return np.array([r.config.S_z for r in self.results[v]])

    def ratio(self, num: str, den: str) -> np.ndarray:
        """Per-configuration ``excitation(num) / excitation(den)``."""
        return self.excitation(num) / self.excitation(den)

    def checks_passed(self) -> bool:
        return all(r.checks.passed for v in self.variants for r in self.results[v])

    def summary(self) -> str:
        lines = [f"Ensemble  theta = {self.theta:.3f}  N = {self.profile.N:.0f}  "
                 f"n_config = {self.n_config}  ({self.seconds:.1f} s)  "
                 f"eta_eff = {self.profile.eta_eff(self.op.wavelength):.2f}"]
        for v in self.variants:
            s = robust_summary(self.excitation(v))
            lines.append(f"  {v:>12s} excitation: mean {s['mean']:.4f} +- {s['sem']:.4f}  "
                         f"median {s['median']:.4f}  trimmed {s['trimmed_mean']:.4f}")
        return "\n".join(lines)


def run_ensemble(profile: GaussianProfile, op: OperatingPoint, theta: float, n_config: int,
                 seed0: int = 0, variants: Sequence[str] = ("full", "far", "independent"),
                 n_jobs: int = 1, incident: Optional[IncidentField] = None,
                 N: Optional[int] = None, **solve_kw) -> EnsembleResult:
    """Sample and solve ``n_config`` configurations (seeds ``seed0 .. seed0 + n_config - 1``)."""
    seeds = np.arange(seed0, seed0 + n_config)
    variants = tuple(variants)
    t0 = time.perf_counter()
    if n_jobs == 1:
        outs = [_solve_one(profile, op, theta, int(s), variants, incident, solve_kw, N) for s in seeds]
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            outs = Parallel(n_jobs=n_jobs)(
                delayed(_solve_one)(profile, op, theta, int(s), variants, incident, solve_kw, N)
                for s in seeds)
    results = {v: [o[v] for o in outs] for v in variants}
    return EnsembleResult(profile, op, float(theta), seeds, variants, results,
                          time.perf_counter() - t0)


# ------------------------------------------------------------- angle scan


@dataclass
class AngleScan:
    thetas: np.ndarray
    ensembles: List[EnsembleResult]
    eta_eff: float

    def table(self, num: str = "full", den: str = "far", trim: float = 0.1) -> List[dict]:
        rows = []
        for th, e in zip(self.thetas, self.ensembles):
            r = e.ratio(num, den)
            s = robust_summary(r, trim)
            rows.append(dict(theta=float(th), n=s["n"], mean=s["mean"], sem=s["sem"],
                             median=s["median"], trimmed=s["trimmed_mean"], std=s["std"],
                             predicted=float(excess_law(self.eta_eff, th)),
                             like_fraction=float(like_pair_fraction(th))))
        return rows

    def format(self, num: str = "full", den: str = "far") -> str:
        lines = [f"excess = excitation({num}) / excitation({den}),  eta_eff = {self.eta_eff:.2f}",
                 f"{'theta':>7s} {'n':>4s} {'mean':>8s} {'sem':>7s} {'median':>8s} {'trimmed':>8s} "
                 f"{'std':>7s} {'law':>7s}"]
        for r in self.table(num, den):
            lines.append(f"{r['theta']:7.3f} {r['n']:4d} {r['mean']:8.3f} {r['sem']:7.3f} "
                         f"{r['median']:8.3f} {r['trimmed']:8.3f} {r['std']:7.3f} {r['predicted']:7.3f}")
        return "\n".join(lines)


def css_angle_scan(profile: GaussianProfile, op: OperatingPoint, thetas, n_config: int,
                   seed0: int = 0, variants=("full", "far", "independent", "rg"),
                   n_jobs: int = 1, incident=None, N=None, **solve_kw) -> AngleScan:
    """Ensembles at each CSS angle; the same seeds are reused at every angle so
    position disorder is identical across angles and only the spins change."""
    ens = [run_ensemble(profile, op, float(th), n_config, seed0, variants, n_jobs, incident, N,
                        **solve_kw) for th in np.atleast_1d(thetas)]
    return AngleScan(np.atleast_1d(np.asarray(thetas, dtype=float)), ens,
                     profile.eta_eff(op.wavelength))
