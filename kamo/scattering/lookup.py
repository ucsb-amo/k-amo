"""One-call scattering-length lookup for a pair of K39 ground states.

This is the engine behind ``Potassium39.get_scattering_length``.

>>> from kamo.scattering.lookup import scattering_length
>>> scattering_length((1, -1), None, 520.58)           # two atoms in |1,-1>
>>> scattering_length((1, -1), (1, 0), [56.83, 113.0])  # one atom in each state

Methods
-------
``"table"`` (default)
    The calibrated coupled channels, precomputed for all 36 pairs of ground states and
    shipped with kamo (:mod:`kamo.scattering.tables`).  Instant; complex ``a`` in lossy
    channels.  Served on 0-1000 G: B = 0 is exact, and above 0.01 G the table tracks the
    model to 1.4e-3 relative or better.  In 0 < B < 0.01 G unresolved channel openings
    make it indicative only -- use ``method="cc"`` there.
``"cc"``
    The same model evaluated directly (:class:`~kamo.scattering.CoupledChannels`), for
    checks or fields the table does not cover.  The first call takes ~1.5 s of setup, then
    each field ~40-90 ms.  Results are memoised.
``"empirical"``
    Measured-resonance model.  Instant, but covers only the channels with measured
    resonances: |1,+-1>, |1,0> intra and the |1,1>+|1,0>, |1,1>+|1,-1>, |1,0>+|1,-1> mixtures.
    Its background can be off by a few a0 far from resonances.
``"kokkelmans"``
    S. Kokkelmans' (TU Eindhoven) coupled-channels tables, provided to the lab (unpublished).
    Same-state pairs only, 1-1000 G on a 0.5 G grid.  Read from the Tweezers shared drive
    (see ``KOKKELMANS_DIRS``); the files are not redistributed with kamo.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

METHODS = ("table", "cc", "empirical", "kokkelmans")

KOKKELMANS_CITATION = (
    "S. J. J. M. F. Kokkelmans (Eindhoven University of Technology), coupled-channels "
    "calculations of 39K ground-state s-wave scattering lengths, |F,mF>+|F,mF> for F=1,2, "
    "1-1000 G (files dated 2023-08-18), private communication to the UCSB Weld group; "
    "unpublished.  Cite as a private communication.")

# Kokkelmans tables: first existing directory wins.  $KAMO_KOKKELMANS_DIR overrides.
KOKKELMANS_DIRS = (
    "G:/Shared drives/Tweezers/Calculations/Kokkelmans Feshbach data/K39_WideScans",
    "B:/_K/Resources/scattering_lengths/Kokkelmans_data_2",          # older lab copy
)
_KOK_FILES = {(1, 1): "aa", (1, 0): "bb", (1, -1): "cc", (2, -2): "dd",
              (2, -1): "ee", (2, 0): "ff", (2, 1): "gg", (2, 2): "hh"}
_CC_MEMO: dict = {}


def kokkelmans_dir():
    """Directory holding the Kokkelmans ``xx_1G_1000G`` folders, or None if none is reachable."""
    for d in ([os.environ["KAMO_KOKKELMANS_DIR"]] if "KAMO_KOKKELMANS_DIR" in os.environ
              else []) + list(KOKKELMANS_DIRS):
        if os.path.isdir(d):
            return d
    return None


def _state(s, name):
    try:
        F, mF = (int(s[0]), int(s[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{name} must be an (F, mF) pair, got {s!r}") from None
    if (F, mF) != (s[0], s[1]) or F not in (1, 2) or abs(mF) > F:
        raise ValueError(f"{name} = {tuple(s)} is not a K39 ground state |F, mF> "
                         f"(F in {{1, 2}}, |mF| <= F)")
    return F, mF


def available_pairs(method: str = "table"):
    """Pairs ``(state_a, state_b)`` with data for ``method`` ("table"/"cc": all 36 pairs)."""
    from .tables import all_pairs
    if method == "empirical":
        from .data import k39_params as kp
        return sorted(tuple(sorted(k)) if len(k) == 2 else (tuple(k)[0],) * 2
                      for k in kp.RESONANCES)
    if method == "kokkelmans":
        return [(s, s) for s in _KOK_FILES]
    if method in ("cc", "table"):
        return all_pairs()
    raise ValueError(f"method must be one of {METHODS}, not {method!r}")


@lru_cache(maxsize=1)
def _coupled_channels():
    from .coupled_channels import CoupledChannels
    return CoupledChannels(B_max=1000.0)


@lru_cache(maxsize=None)
def kokkelmans_table(state):
    """``(B, a_re, a_im, sigma_el, sigma_inel)`` arrays of the Kokkelmans table for ``state``.

    Columns as provided: a in a0 (``a = a_re - i a_im``); the two cross sections are as
    written in the files (the elastic one is 8 pi a^2 in a0^2).
    """
    root = kokkelmans_dir()
    if root is None:
        raise FileNotFoundError(
            "Kokkelmans tables not found in any of " + ", ".join(KOKKELMANS_DIRS)
            + " (set KAMO_KOKKELMANS_DIR).  method='table' or 'cc' need no external files.")
    d = f"{root}/{_KOK_FILES[tuple(state)]}_1G_1000G"
    B = np.loadtxt(f"{d}/Bval.txt")
    cols = np.loadtxt(f"{d}/data.txt")
    return (B,) + tuple(cols.T)


def _kokkelmans_complex(state):
    B, ar, ai, _, _ = kokkelmans_table(state)
    return B, ar - 1j * ai


def scattering_length(state_a, state_b=None, B_gauss=0.0, method: str = "table",
                      interp: bool = False, return_complex: bool = False):
    """s-wave scattering length (a0) of the pair ``state_a + state_b`` at ``B_gauss``.

    Parameters
    ----------
    state_a, state_b : (F, mF)
        Hyperfine states of the two atoms.  ``state_b=None`` means both atoms are in
        ``state_a``.
    B_gauss : float or array-like
        Magnetic field(s), G.  ``"table"`` covers 0-1000 G; see the module docstring for
        the 0 < B < 0.01 G caveat.
    method : {"table", "cc", "empirical", "kokkelmans"}
        See the module docstring.
    interp : bool
        ``"kokkelmans"`` only: interpolate the 0.5 G table linearly instead of taking the
        nearest point.  The other methods are continuous in B.
    return_complex : bool
        Return ``a_re - i a_im`` (``a_im > 0`` = two-body loss) instead of ``Re a``.

    Returns
    -------
    float (or complex) for scalar ``B_gauss``, else an ndarray of the same shape.

    Raises
    ------
    ValueError
        Invalid state or method, a pair with no data for ``method``, or a field outside
        the method's range.
    """
    a = _state(state_a, "state_a")
    b = a if state_b is None else _state(state_b, "state_b")
    pair = tuple(sorted((a, b)))
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if pair not in available_pairs(method):
        hint = ("" if method in ("cc", "table")
                else "  method='table' covers every pair of ground states.")
        raise ValueError(f"no {method!r} scattering-length data for |{a[0]},{a[1]:+d}>+"
                         f"|{b[0]},{b[1]:+d}>; available: {available_pairs(method)}.{hint}")

    from .tables import B_VALID
    B = np.asarray(B_gauss, dtype=float)
    Bf = np.atleast_1d(B).ravel()
    lo, hi = {"table": B_VALID, "cc": (0.0, 1000.0), "empirical": (0.0, np.inf),
              "kokkelmans": (1.0, 999.998)}[method]
    bad = (Bf < lo) | (Bf > hi)
    if np.any(bad):
        raise ValueError(f"B outside the {method!r} range [{lo}, {hi}] G "
                         f"(got {Bf[bad].min():g} .. {Bf[bad].max():g})")

    if method == "table":
        from .tables import table_scattering_length
        out = np.asarray(table_scattering_length(a, b, Bf), dtype=complex)
    elif method == "empirical":
        from .backends.empirical import EmpiricalBackend
        out = np.asarray(EmpiricalBackend().scattering_length(a, b, Bf), dtype=complex)
    elif method == "kokkelmans":
        Bt, at = _kokkelmans_complex(a)
        if interp:
            out = np.interp(Bf, Bt, at.real) + 1j * np.interp(Bf, Bt, at.imag)
        else:
            out = at[np.abs(Bf[:, None] - Bt[None, :]).argmin(axis=1)]
    else:
        cc = _coupled_channels()
        out = np.empty(Bf.shape, dtype=complex)
        for i, x in enumerate(Bf):
            key = (pair, x)
            if key not in _CC_MEMO:
                if len(_CC_MEMO) > 200_000:
                    _CC_MEMO.clear()
                _CC_MEMO[key] = cc.scattering_length(a, b, x)
            out[i] = _CC_MEMO[key]

    out = out if return_complex else out.real
    if B.ndim == 0:
        return out[0].item()
    return out.reshape(B.shape)
