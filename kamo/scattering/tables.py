"""Precomputed coupled-channels a(B) tables for every pair of K39 ground states.

The calibrated coupled-channels model (:class:`~kamo.scattering.CoupledChannels`)
costs ~40-90 ms per field; these tables make it instant.  They ship with kamo as
``data/k39_cc_tables.npz`` and back the default ``method="table"`` of
:func:`kamo.scattering.lookup.scattering_length`.

Representation
--------------
``a(B)`` is meromorphic in B apart from Feshbach poles and, below ~0.2 G, a handful of
step discontinuities where an inelastic channel opens (the Zeeman splitting closes, a
threshold crosses the entrance channel, and ``Im a`` jumps).  Each pair stores ``a`` on
a uniform grid, its poles ``z_i`` (complex: ``B0_i + i gamma_i/2``, with ``gamma_i > 0``
only where the pole is smoothed by inelastic loss), and its break fields.  The grid is
split at the breaks and each smooth piece splined separately -- interpolating through a
jump corrupts several cells either side.  On load,
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
Three segments: 0.001 G on 0-0.3 G (poles sit as close as 0.003 G to zero and the
F=1 thresholds collapse as B -> 0), 0.05 G to 2 G, then 0.5 G to 1000 G.  Lookups are
served on the whole 0-1000 G.  Validated against direct coupled-channels
evaluations at 40 random fields per band per pair plus points 0.003-0.3 G from every
pole (2026-09-11).  Worst relative error over all 36 pairs, by band:

    0.01-0.3 G   1.4e-3        0.3-2 G   3.3e-5        2-1000 G   5.2e-6

A 1 G step above 2 G failed an earlier 5e-3 bar (1.3e-2), hence 0.5 G.  ``B = 0`` is a
grid node and reproduces the model exactly (worst 1.6e-5 a0 across the 36 pairs).

The exception is ``0 < B < 0.01 G``, where the table is indicative only (errors reach
~100%).  More channel openings live down there -- e.g. |1,-1>+|2,0> jumps from 245 to
87 a0 between 1e-5 and 1e-4 G -- below the 0.001 G step, so they are neither resolved
nor split out as breaks.  Resolving them needs a near-zero sub-grid (the pole finder
otherwise fits a spurious "pole" that just tracks the step size: 0.0020 G at h=0.002,
0.0012 at h=0.001, 0.0005 at h=0.0005).  Use ``method="cc"`` in that window.

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
B_VALID = (0.0, 1000.0)         # served.  See the accuracy note above for 0 < B < 0.01 G,
                                # where unresolved channel openings make the table
                                # indicative only (B = 0 itself is exact).
LOW_FIELD = (2.0, 0.05)         # mid segment [ULTRA_LOW[0], 2) G at 0.05 G
ULTRA_LOW = (0.3, 0.001)        # finest segment [0, 0.3) G: resolves the near-zero poles
                                # (0.003-0.2 G) and the collapsing F=1 thresholds
_L = 100.0                      # G, scale of the pole factors (keeps N O(1))


def make_grid(h: float = 0.5) -> np.ndarray:
    """Field grid: 0.001 G below 0.3 G, 0.05 G to 2 G, then ``h`` up to ``B_MAX``."""
    b_ul, h_ul = ULTRA_LOW
    b_lo, h_lo = LOW_FIELD
    return np.round(np.concatenate([np.arange(B_MIN, b_ul, h_ul),
                                    np.arange(b_ul, b_lo, h_lo),
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


def _find_jumps(B, a, poles=(), min_jump=5.0, ratio=50.0):
    """Indices ``k`` where ``a`` steps discontinuously between ``B[k]`` and ``B[k+1]``.

    Sub-gauss channel openings (an inelastic threshold crossing the entrance channel as
    the Zeeman splitting closes) make ``a(B)`` genuinely discontinuous: one cell carries a
    large change and both neighbours stay smooth.  A narrow pole instead spikes and comes
    back, so it moves *two* adjacent cells -- hence the two-sided neighbour test.
    """
    da = np.abs(np.diff(a))
    if da.size < 3:
        return []
    nb = np.maximum(np.r_[da[1], da[:-1]], np.r_[da[1:], da[-2]])
    idx = np.where((da > min_jump) & (da > ratio * np.maximum(nb, 1e-9)))[0]
    if len(poles):
        pr = np.asarray(poles).real
        idx = np.array([k for k in idx if np.min(np.abs(pr - B[k])) > 3e-3], dtype=int)
    return list(idx)


def _refine_jump(fn, lo, hi, tol=1e-7):
    """Bisect the discontinuity inside the cell ``[lo, hi]`` onto the threshold field.

    Both sides are smooth and far apart in value, so a sample belongs to whichever
    endpoint it is closer to.
    """
    a_lo, a_hi = complex(fn(lo)), complex(fn(hi))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        a_mid = complex(fn(mid))
        if abs(a_mid - a_lo) < abs(a_mid - a_hi):
            lo, a_lo = mid, a_mid
        else:
            hi, a_hi = mid, a_mid
    return 0.5 * (lo + hi)


def find_breaks(fn, B, a):
    """Threshold fields (G) where ``a(B)`` jumps, refined to ~1e-7 G."""
    return np.array([_refine_jump(fn, B[k], B[k + 1]) for k in _find_jumps(B, a)])


def _poles_on_grid(fn, B, a, d2_thresh=0.5, breaks=()):
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
    breaks = np.asarray(breaks, dtype=float)
    def _at_break(x, w):
        return breaks.size and np.min(np.abs(breaks - x)) < w
    for k in range(len(B) - 1):
        if breaks.size and np.any((breaks > B[k]) & (breaks < B[k + 1])):
            continue                      # a discontinuity, not a pole
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
            if _at_break(B[k], 3 * h):
                continue
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
    """Re-run the pole and jump search on existing checkpoints (grid values are reused)."""
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
    poles, breaks = analyse_pair(fn, B, a)
    np.savez(path, B=B, a=a, poles=poles, breaks=breaks)
    return pair


def build_pair(pair, h=0.5, cc=None):
    """Tabulate one pair: ``(B, a, poles, breaks)`` on :func:`make_grid` ``(h)``."""
    from .coupled_channels import CoupledChannels
    cc = cc or CoupledChannels(B_max=B_MAX)
    a_st, b_st = pair
    fn = lambda B: cc.scattering_length(a_st, b_st, float(np.clip(B, B_MIN, B_MAX)))
    B = make_grid(h)
    a = np.array([fn(x) for x in B])
    return (B, a) + analyse_pair(fn, B, a)


def analyse_pair(fn, B, a):
    """``(poles, breaks)`` of a tabulated pair -- the part that needs no new grid values."""
    breaks = find_breaks(fn, B, a)
    poles = np.array(_poles_on_grid(fn, B, a, breaks=breaks), dtype=complex)
    if breaks.size and poles.size:       # a break that landed on a real pole is not a jump
        breaks = breaks[np.min(np.abs(breaks[:, None] - poles.real[None, :]), axis=1) > 3e-3]
    return poles, breaks


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
        B, a, poles, breaks = build_pair(pair, h, cc=worker_cc())
        np.savez(out, B=B, a=a, poles=poles, breaks=breaks)
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
            poles_d[f"breaks_{k}"] = d["breaks"] if "breaks" in d else np.empty(0)
        keep = np.concatenate([np.where(B < LOW_FIELD[0])[0],
                               np.where(B >= LOW_FIELD[0])[0][::stride]])
        a, h = a[keep], (B[-1] - B[-2]) * stride
        keys.append(f"re_{k}"); cols.append(a.real)
        if np.any(np.abs(a.imag) > 0):
            keys.append(f"im_{k}"); cols.append(a.imag)
        poles_d[f"poles_{k}"] = poles
    meta = dict(format=3, B_valid=B_VALID, h=float(h), low_field=LOW_FIELD,
                ultra_low=ULTRA_LOW, dtype="float32",
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
    import json
    fmt = json.loads(str(d["meta"])).get("format")
    if fmt != 3:
        raise ValueError(f"{path} is format {fmt}, expected 3 (the grid gained a 0.001 G "
                         f"segment below 0.3 G, and a(B) is now split at its sub-gauss "
                         f"threshold jumps); rebuild with "
                         f"`python -m kamo.scattering.tables --build`")
    n = len(make_grid(float(d["h"])))
    d.update(zip(keys, _unpack(d.pop("blob"), len(keys), n)))
    return d


@lru_cache(maxsize=None)
def _spline(key, path=TABLE_PATH):
    """``(segment splines, break fields, poles)`` for one pair.

    ``a(B)`` is split at its threshold jumps (:func:`find_breaks`) and splined on each
    smooth piece; interpolating through a jump corrupts several cells either side.
    """
    from scipy.interpolate import CubicSpline
    d = _load(path)
    B = make_grid(float(d["h"]))
    a = d[f"re_{key}"].astype(float) + 1j * (d[f"im_{key}"].astype(float)
                                             if f"im_{key}" in d else 0.0)
    poles = d[f"poles_{key}"]
    breaks = np.asarray(d.get(f"breaks_{key}", np.empty(0)), dtype=float)
    aP = a * _pole_product(B, poles)
    edges = np.searchsorted(B, breaks)
    segs = tuple(CubicSpline(B[lo:hi], aP[lo:hi]) for lo, hi in
                 zip(np.r_[0, edges].astype(int), np.r_[edges, len(B)].astype(int)))
    return segs, breaks, poles


def _pole_product(B, poles):
    P = np.ones_like(B, dtype=complex)
    for z in poles:
        P = P * (B - z) / _L
    return P


def table_scattering_length(state_a, state_b, B, path=TABLE_PATH) -> np.ndarray:
    """Complex a (a0) of the pair from the precomputed table, at fields ``B`` (array)."""
    segs, breaks, poles = _spline(pair_key(state_a, state_b), path)
    B = np.asarray(B, dtype=float)
    flat = np.atleast_1d(B).ravel()
    num = np.empty(flat.shape, dtype=complex)
    which = (np.searchsorted(breaks, flat, side="right") if breaks.size
             else np.zeros(flat.shape, dtype=int))
    for j, spl in enumerate(segs):
        m = which == j
        if m.any():
            num[m] = spl(flat[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (num.reshape(B.shape) if B.ndim else num[0]) / _pole_product(B, poles)


def table_poles(state_a, state_b, max_width=1.0, path=TABLE_PATH) -> np.ndarray:
    """Resonance poles ``B0 + i gamma/2`` (G) of the pair inside the valid field range.

    The table also factors out broad structures (e.g. exit-channel resonances seen
    in a lossy channel) as wide complex poles; ``max_width`` (G) drops those with
    ``gamma > max_width``.
    """
    poles = _spline(pair_key(state_a, state_b), path)[2]
    keep = ((poles.real >= B_VALID[0]) & (poles.real <= B_VALID[1])
            & (2 * np.abs(poles.imag) <= max_width))
    return poles[keep]


def table_breaks(state_a, state_b, path=TABLE_PATH) -> np.ndarray:
    """Fields (G) where ``a(B)`` of the pair jumps discontinuously.

    Sub-gauss inelastic-channel openings; the table splines each side separately, so
    ``a`` is correct up to either edge but undefined exactly at the break.
    """
    return _spline(pair_key(state_a, state_b), path)[1]


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
