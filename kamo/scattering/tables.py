"""Precomputed coupled-channels a(B) tables for every pair of K39 ground states.

The calibrated coupled-channels model (:class:`~kamo.scattering.CoupledChannels`)
costs ~40-90 ms per field; these tables make it instant.  They ship with kamo as
``data/k39_cc_tables.npz`` and back the default ``method="table"`` of
:func:`kamo.scattering.lookup.scattering_length`.

Representation
--------------
``a(B)`` is meromorphic in B: smooth apart from Feshbach poles.  Each pair stores
``a`` on a uniform grid plus its poles ``z_i`` (complex: ``B0_i + i gamma_i/2``,
with ``gamma_i > 0`` only where the pole is smoothed by inelastic loss).  On load,
the pole-free numerator

    N(B) = a(B) * prod_i (B - z_i) / L

is cubic-spline interpolated and ``a = N / prod_i (B - z_i) * L`` is recovered.
Because N is smooth, a coarse grid stays accurate right up to each pole, and at
zero crossings.

Poles are found from sign changes of Re a (Brent), plus a second-difference test
on the pole-subtracted residual that catches features with no sign change: a
narrow pole with its zero crossing in the same cell, or a lossy resonance.  Each
candidate's complex position is refined by an iterated three-point fit of
``a = b - s/(B - z)``.

Grid and accuracy
-----------------
0.05 G steps on 0-2 G (the F=1 threshold cusp near B -> 0), then 0.5 G to 1000 G;
lookups are served on 1-1000 G.  Validated against direct coupled-channels
evaluations at 400 random fields per pair plus points 0.003-0.3 G from every pole
(2026-09-11): median error ~1e-6 a0, worst relative error 5e-3 (in lossy F=1+F=2
mixtures; typical worst <2e-4).  A 1 G grid failed that bar (1.3e-2), hence 0.5 G.

Storage: float32 values, XOR-delta + byte-shuffle + lzma (:func:`_pack`), ~225 kB
for all 36 pairs, lossless on the float32 values (plain npz compression: 410 kB).

Regenerate after changing the calibration (~15 min on 18 worker processes)::

    python -m kamo.scattering.tables --build --checkpoints <scratch dir>
"""

from __future__ import annotations

import datetime
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

TABLE_PATH = Path(__file__).parent / "data" / "k39_cc_tables.npz"
B_MIN, B_MAX = 0.0, 1000.0      # tabulated
B_VALID = (1.0, 1000.0)         # served.  Below ~1 G the F=1 pair thresholds become
                                # degenerate (splitting ~ B^2), a(B) has a threshold cusp
                                # near 0.01-0.1 G, and a zero-energy a is ill-defined.
LOW_FIELD = (2.0, 0.05)         # fine segment [0, 2) G at 0.05 G contains that cusp
_L = 100.0                      # G, scale of the pole factors (keeps N O(1))


def make_grid(h: float = 0.5) -> np.ndarray:
    """Field grid: 0.05 G steps below 2 G, then ``h`` up to ``B_MAX``."""
    b_lo, h_lo = LOW_FIELD
    return np.round(np.concatenate([np.arange(B_MIN, b_lo, h_lo),
                                    np.arange(b_lo, B_MAX + h / 2, h)]), 9)


def all_pairs():
    """The 36 unordered pairs of K39 ground states, canonical (sorted) order."""
    states = [(F, m) for F in (1, 2) for m in range(-F, F + 1)]
    return [(a, b) for i, a in enumerate(states) for b in states[i:]]


def pair_key(a, b) -> str:
    (Fa, ma), (Fb, mb) = sorted((tuple(a), tuple(b)))
    return f"{Fa}{ma:+d}_{Fb}{mb:+d}"


# ---------------------------------------------------------------- building
def _local_step(B, x):
    k = int(np.clip(np.searchsorted(B, x), 1, len(B) - 1))
    return B[k] - B[k - 1]


def _pole_fit(fn, B0, d=0.05):
    """Local single-pole fit ``a = b - s/(B - z)`` from three samples: ``(z, s)``."""
    Bs = np.array([B0 - d, B0 + d / 3, B0 + d])
    a = np.array([complex(fn(x)) for x in Bs])
    rho = (a[0] - a[1]) / (a[1] - a[2]) * (Bs[2] - Bs[1]) / (Bs[1] - Bs[0])
    z = (Bs[2] - rho * Bs[0]) / (1 - rho)
    s = -(a[0] - a[1]) * (Bs[0] - z) * (Bs[1] - z) / (Bs[1] - Bs[0])
    return z, s


def _refine_pole(fn, z, d, n_iter=12):
    """Iterate the three-point pole fit, shrinking the sample spacing onto ``z``.

    For an isolated feature ``a = b - s/(B - z)`` (constant ``b``) one fit is exact;
    iterating removes the background slope and neighbouring tails.  Converges on
    narrow poles whose zero shares a grid cell, and on lossy ones (complex ``z``).
    """
    s = np.nan
    for _ in range(n_iter):
        z_new, s = _pole_fit(fn, z.real, d)
        if not (np.isfinite(z_new.real) and np.isfinite(z_new.imag)):
            return None, np.nan
        dz = abs(z_new - z)
        z = z_new
        if dz < 1e-9:
            break
        d = float(np.clip(4 * abs(z.imag) + 10 * dz, 1e-5, d))
    return z, s


def _poles_on_grid(fn, B, a, d2_thresh=0.5):
    """Complex poles ``B0 + i gamma/2`` of ``a`` (sampled on grid ``B``).

    1. Sign changes of Re a that are poles (Brent): exact for elastic poles; for
       lossy ones the crossing is biased by ``Re b * gamma^2 / 4s``, so their
       complex position is refined by the iterated pole fit.
    2. Features without a sign change (a narrow pole and its zero in one cell, or a
       lossy resonance whose Re a never crosses zero): grid points where the
       second difference of the pole-subtracted residual is anomalous seed the
       iterated pole fit.
    """
    from .resonances import _resolve_sign_change
    zs = []
    re = a.real
    for k in range(len(B) - 1):
        if re[k] * re[k + 1] < 0:
            B0, kind = _resolve_sign_change(fn, B[k], B[k + 1])
            if kind != 'pole':
                continue
            d = min(0.05, _local_step(B, B0) / 10)
            z, s = _pole_fit(fn, B0, d)
            if abs(z.imag) > 1e-6:
                z2, s2 = _refine_pole(fn, z, d)
                if z2 is not None and abs(z2.real - B0) < 2 * _local_step(B, B0):
                    z, s = z2, s2
            else:
                z = complex(B0, 0.0)
            zs.append((z, s))

    tried = []
    for _ in range(4):
        r = a.copy()
        for z, s in zs:
            r = r + s / (B - z)
        hl, hr = B[1:-1] - B[:-2], B[2:] - B[1:-1]
        d2 = np.abs((r[2:] - r[1:-1]) / hr - (r[1:-1] - r[:-2]) / hl) * 0.5 * (hl + hr)
        flag = np.where(d2 > np.maximum(d2_thresh, 1e-3 * np.abs(r[1:-1])))[0] + 1
        known = np.array([z.real for z, _ in zs])
        new = []
        for k in flag[np.argsort(-d2[flag - 1])]:          # strongest first
            h = _local_step(B, B[k])
            if known.size and np.min(np.abs(known - B[k])) < 2.5 * h:
                continue
            if any(abs(B[k] - x) < 2.5 * h for x in tried + [z.real for z, _ in new]):
                continue
            tried.append(B[k])
            z, s = _refine_pole(fn, complex(B[k], 0.0), h)
            if (z is None or not np.isfinite(s) or abs(s) < 1e-4
                    or abs(z.real - B[k]) > 2 * h or abs(z.imag) > 5 * h):
                continue
            if known.size and np.min(np.abs(known - z.real)) < 1e-5:
                continue
            new.append((z, s))
        if not new:
            break
        zs += new
    out = [complex(z.real, z.imag if abs(z.imag) > 1e-7 else 0.0) for z, _ in zs]
    return sorted(out, key=lambda z: z.real)


def refresh_poles(checkpoint_dir, workers=None):
    """Re-run the pole search on existing checkpoints (grid values are reused)."""
    from concurrent.futures import ProcessPoolExecutor
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for pair in ex.map(_repole_worker, [(p, str(checkpoint_dir)) for p in all_pairs()]):
            print(f"re-poled {pair_key(*pair)}", flush=True)


def _repole_worker(args):
    pair, ckpt = args
    path = Path(ckpt) / f"{pair_key(*pair)}.npz"
    with np.load(path) as d:
        B, a = d["B"], d["a"]
    cc = worker_cc()
    fn = lambda x: cc.scattering_length(*pair, float(np.clip(x, B_MIN, B_MAX)))
    np.savez(path, B=B, a=a, poles=np.array(_poles_on_grid(fn, B, a), dtype=complex))
    return pair


def build_pair(pair, h=0.5, cc=None):
    """Tabulate one pair: ``(B, a, poles)`` on :func:`make_grid` ``(h)``."""
    from .coupled_channels import CoupledChannels
    cc = cc or CoupledChannels(B_max=B_MAX)
    a_st, b_st = pair
    fn = lambda B: cc.scattering_length(a_st, b_st, float(np.clip(B, B_MIN, B_MAX)))
    B = make_grid(h)
    a = np.array([fn(x) for x in B])
    poles = _poles_on_grid(fn, B, a)
    return B, a, np.array(poles, dtype=complex)


def worker_cc(tries=20):
    """A CoupledChannels for a worker process.  ARC initialises a shared SQLite
    database on first use; parallel first uses race ("index already exists"),
    so retry with jitter."""
    import random, sqlite3, time
    for k in range(tries):
        try:
            # the import itself can trigger ARC's database set-up (kamo.constants)
            from .coupled_channels import CoupledChannels
            return CoupledChannels(B_max=B_MAX)
        except sqlite3.OperationalError:
            if k == tries - 1:
                raise
            time.sleep(random.uniform(0.2, 2.0))


def _build_worker(args):
    pair, h, ckpt = args
    out = Path(ckpt) / f"{pair_key(*pair)}.npz"
    if not out.exists():
        B, a, poles = build_pair(pair, h, cc=worker_cc())
        np.savez(out, B=B, a=a, poles=poles)
    return pair


def build_all(checkpoint_dir, h=0.5, workers=None, pairs=None):
    """Compute every pair in parallel into ``checkpoint_dir`` (full precision, resumable)."""
    from concurrent.futures import ProcessPoolExecutor
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    pairs = pairs or all_pairs()
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    # one BLAS/OpenMP thread per worker: the matrices are 1x1-8x8, and threaded
    # BLAS in every process thrashes (observed ~6x slowdown with 18 workers)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for pair in ex.map(_build_worker, [(p, h, str(checkpoint_dir)) for p in pairs]):
            print(f"done {pair_key(*pair)}", flush=True)


def _pack(arrays):
    """float32 arrays -> one lzma blob: XOR successive bit patterns, byte-shuffle, lzma.

    Lossless on the float32 values; ~2x smaller than zlib on the raw floats because
    smooth data leaves the high bytes of the XOR-delta nearly constant.
    """
    import lzma
    parts = []
    for v in arrays:
        u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
        x = u.copy()
        x[1:] ^= u[:-1]
        parts.append(x.view(np.uint8).reshape(-1, 4).T.tobytes())
    return np.frombuffer(lzma.compress(b"".join(parts), preset=9 | lzma.PRESET_EXTREME),
                         dtype=np.uint8)


def _unpack(blob, n_arrays, n):
    import lzma
    raw = np.frombuffer(lzma.decompress(blob.tobytes()), dtype=np.uint8)
    out = []
    for i in range(n_arrays):
        x = raw[i * 4 * n:(i + 1) * 4 * n].reshape(4, n).T.copy().view(np.uint32).ravel()
        out.append(np.bitwise_xor.accumulate(x).view(np.float32))
    return out


def write_table(checkpoint_dir, path=TABLE_PATH, stride=1):
    """Assemble the shipped table from checkpoints, keeping every ``stride``-th point
    of the uniform part of the grid (the low-field segment is kept whole).

    File layout (``np.load``-able): ``h``, ``meta``, ``poles_<pair>`` (complex128), and
    ``keys`` + ``blob``: the float32 ``re_<pair>`` / ``im_<pair>`` columns packed by
    :func:`_pack` (``im`` only for lossy pairs).
    """
    from .data import k39_calibration as kc
    keys, cols, poles_d, h = [], [], {}, None
    for pair in all_pairs():
        k = pair_key(*pair)
        with np.load(Path(checkpoint_dir) / f"{k}.npz") as d:
            B, a, poles = d["B"], d["a"], d["poles"]
        keep = np.concatenate([np.where(B < LOW_FIELD[0])[0],
                               np.where(B >= LOW_FIELD[0])[0][::stride]])
        a, h = a[keep], (B[-1] - B[-2]) * stride
        keys.append(f"re_{k}"); cols.append(a.real)
        if np.any(np.abs(a.imag) > 0):
            keys.append(f"im_{k}"); cols.append(a.imag)
        poles_d[f"poles_{k}"] = poles
    meta = dict(format=1, B_valid=B_VALID, h=float(h), low_field=LOW_FIELD, dtype="float32",
                a_S=kc.A_SINGLET, a_T=kc.A_TRIPLET, delta_S=kc.DELTA_S, delta_T=kc.DELTA_T,
                created=datetime.date.today().isoformat(),
                model="kamo.scattering.CoupledChannels on the Falke 2008 curves, inner walls "
                      "calibrated to measured resonance positions (k39_calibration)")
    import json
    np.savez(path, h=np.array(h), meta=np.array(json.dumps(meta)), keys=np.array(keys),
             blob=_pack(cols), **poles_d)
    _load.cache_clear(); _spline.cache_clear()
    return path


# ---------------------------------------------------------------- lookup
@lru_cache(maxsize=4)
def _load(path=TABLE_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing; build it with "
                                f"`python -m kamo.scattering.tables --build`")
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    keys = [str(k) for k in d.pop("keys")]
    n = len(make_grid(float(d["h"])))
    d.update(zip(keys, _unpack(d.pop("blob"), len(keys), n)))
    return d


@lru_cache(maxsize=None)
def _spline(key, path=TABLE_PATH):
    from scipy.interpolate import CubicSpline
    d = _load(path)
    B = make_grid(float(d["h"]))
    a = d[f"re_{key}"].astype(float) + 1j * (d[f"im_{key}"].astype(float)
                                             if f"im_{key}" in d else 0.0)
    poles = d[f"poles_{key}"]
    return CubicSpline(B, a * _pole_product(B, poles)), poles


def _pole_product(B, poles):
    P = np.ones_like(B, dtype=complex)
    for z in poles:
        P = P * (B - z) / _L
    return P


def table_scattering_length(state_a, state_b, B, path=TABLE_PATH) -> np.ndarray:
    """Complex a (a0) of the pair from the precomputed table, at fields ``B`` (array)."""
    spl, poles = _spline(pair_key(state_a, state_b), path)
    B = np.asarray(B, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return spl(B) / _pole_product(B, poles)


def table_poles(state_a, state_b, max_width=1.0, path=TABLE_PATH) -> np.ndarray:
    """Resonance poles ``B0 + i gamma/2`` (G) of the pair inside the valid field range.

    The table also factors out broad structures (e.g. exit-channel resonances seen
    in a lossy channel) as wide complex poles; ``max_width`` (G) drops those with
    ``gamma > max_width``.
    """
    _, poles = _spline(pair_key(state_a, state_b), path)
    keep = ((poles.real >= B_VALID[0]) & (poles.real <= B_VALID[1])
            & (2 * np.abs(poles.imag) <= max_width))
    return poles[keep]


def table_meta(path=TABLE_PATH) -> dict:
    """Provenance of the table: model, calibration (a_S, a_T, deltas), grid, date."""
    import json
    return json.loads(str(_load(path)["meta"]))


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--h", type=float, default=0.5, help="computed grid step (G)")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th point when writing")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--checkpoints", default="k39_cc_table_checkpoints")
    ap.add_argument("--out", default=str(TABLE_PATH))
    args = ap.parse_args()
    if args.build:
        t = time.time()
        build_all(args.checkpoints, h=args.h, workers=args.workers)
        write_table(args.checkpoints, args.out, stride=args.stride)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB) in {time.time() - t:.0f} s")
