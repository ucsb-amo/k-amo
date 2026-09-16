"""Timings: kernel build, LU (double / mixed), GMRES, GPU, field evaluation.

Run ``python -m kamo.dd_solver.benchmark`` (add ``--gpu`` if torch sees a CUDA
device, ``--N 500 1000 2000 3500``).  Every timing is a wall-clock median of
``repeat`` runs after a warm-up, with nothing else running.  The LU / GMRES
crossover is measured rather than assumed: GMRES iteration counts grow
superlinearly with density, so the crossover moves with the profile.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import numpy as np

from .cloud import GaussianProfile, sample_configuration
from .fields import scattered_field
from .solver import build_matrix, solve_linear
from .system import OperatingPoint, default_incident


def _timeit(fn, repeat: int = 3):
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_one(N: int, op: OperatingPoint, profile: GaussianProfile, gpu: bool = False,
              repeat: int = 3, cond: bool = True, grid_points: int = 200 * 200, seed: int = 0) -> dict:
    cfg = sample_configuration(profile.with_atom_number(N), theta=np.pi, seed=seed)
    det = op.detunings(cfg.spins)
    inc = default_incident(op)
    Omega = inc.drive(cfg.positions, op.e_hat)
    out = dict(N=N)
    out["build"] = _timeit(lambda: build_matrix(cfg.positions, det, op, "full"), repeat)
    M, J, G = build_matrix(cfg.positions, det, op, "full")
    b = -0.5 * Omega
    out["matrix_MB"] = M.nbytes / 1e6
    out["lu_double"] = _timeit(lambda: solve_linear(M, b, "lu", "double"), repeat)
    out["lu_mixed"] = _timeit(lambda: solve_linear(M, b, "lu", "mixed"), repeat)
    x_ref, _ = solve_linear(M, b, "lu", "double")
    x_mix, _ = solve_linear(M, b, "lu", "mixed")
    out["mixed_rel_err"] = float(np.max(np.abs(x_mix - x_ref)) / np.max(np.abs(x_ref)))
    xg, info = solve_linear(M, b, "gmres")
    out["gmres"] = _timeit(lambda: solve_linear(M, b, "gmres"), max(repeat - 1, 1))
    out["gmres_iterations"] = info.get("iterations")
    out["gmres_rel_err"] = float(np.max(np.abs(xg - x_ref)) / np.max(np.abs(x_ref)))
    if cond and N <= 2500:
        out["cond"] = float(np.linalg.cond(M))
    if gpu:
        out["gpu_double"] = _timeit(lambda: solve_linear(M, b, "lu", "double", "gpu"), repeat)
        out["gpu_mixed"] = _timeit(lambda: solve_linear(M, b, "lu", "mixed", "gpu"), repeat)
    # field on a grid of `grid_points` points, chunked
    pts = np.random.default_rng(1).normal(size=(grid_points, 3)) * 2e-6
    out["field_grid_cpu"] = _timeit(lambda: scattered_field(pts, cfg.positions, x_ref, op.k, op.e_hat),
                                    max(repeat - 1, 1))
    if gpu:
        out["field_grid_gpu"] = _timeit(
            lambda: scattered_field(pts, cfg.positions, x_ref, op.k, op.e_hat, backend="gpu"),
            max(repeat - 1, 1))
    return out


def run(Ns=(500, 1000, 2000, 3500), gpu: bool = False, repeat: int = 3,
        profile: Optional[GaussianProfile] = None, op: Optional[OperatingPoint] = None) -> List[dict]:
    op = OperatingPoint.nominal() if op is None else op
    profile = GaussianProfile.operating_point(500) if profile is None else profile
    rows = []
    for N in Ns:
        rows.append(bench_one(int(N), op, profile, gpu, repeat))
        print(format_row(rows[-1]))
    return rows


def format_row(r: dict) -> str:
    s = (f"N={r['N']:5d}  build {r['build']:.3f}s  LU {r['lu_double']:.3f}s  mixed {r['lu_mixed']:.3f}s "
         f"(err {r['mixed_rel_err']:.1e})  GMRES {r['gmres']:.3f}s/{r['gmres_iterations']} it "
         f"(err {r['gmres_rel_err']:.1e})  {r['matrix_MB']:.0f} MB")
    if "cond" in r:
        s += f"  cond {r['cond']:.0f}"
    if "gpu_double" in r:
        s += f"  GPU {r['gpu_double']:.3f}s / mixed {r['gpu_mixed']:.3f}s"
    s += f"  field(40k pts) {r['field_grid_cpu']:.2f}s"
    if "field_grid_gpu" in r:
        s += f" / gpu {r['field_grid_gpu']:.2f}s"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, nargs="+", default=[500, 1000, 2000, 3500])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args(argv)
    run(a.N, a.gpu, a.repeat)


if __name__ == "__main__":
    main()
