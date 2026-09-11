"""K39 scattering parameters for the empirical backend (literature-verified).

Resonance table
---------------
Built from the measured-resonance database :mod:`.k39_feshbach` (deep-research
pass 2026-09-10): experimental positions ``B0`` (Etrych 2023, Chapurin 2019,
D'Errico 2007, Tanzi 2018), with widths / local backgrounds from Etrych's
coupled-channels characterisation (experiments measure positions and at best
the pole strength ``a_bg*Delta``, not the split).  Six channels: the three
F=1 intra channels plus the |1,1>+|1,0>, |1,1>+|1,-1> and |1,0>+|1,-1>
mixtures.

Model
-----
Sum-of-poles form (as Chapurin et al. 2019, Suppl. Eq. S4)::

    a(B) = b(B) - sum_i s_i / (B - B0_i)       s_i = a_bg,i * Delta_i  (pole strength)

``b(B)`` is a slowly varying background, piecewise linear through anchors
placed at every pole (``b = a_bg,i`` minus the other poles' tails there), at
measured zero crossings not within 5 G of a pole (there ``a = 0`` exactly),
and at published coupled-channels values more than 10 G from any pole
(``k39_feshbach.THEORY_POINTS``).  Unlike the product form
``a_bg * prod(1 - Delta_i/(B - B0_i))`` this does not force spurious zeros
between overlapping resonances (|1,-1>: 33.6 G and 162.4 G).  For lossy
channels ``B - B0_i`` becomes ``B - B0_i - i gamma_i/2`` (``gamma_i > 0``),
giving ``a = a_re - i a_im`` with ``a_im >= 0``.

Singlet/triplet constants
-------------------------
``a_S``/``a_T`` below are the Falke 2008 values; the coupled-channels fit to the
resonance database is in :mod:`.k39_calibration`, and other literature values
in ``k39_feshbach.SINGLET_TRIPLET_LITERATURE``.  Units: a0, a.u., Gauss.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import k39_feshbach as kf

PARAMS_VERIFIED = True

# --------------------------------------------------------------------------
# Singlet / triplet background scattering lengths and van der Waals C6.
# --------------------------------------------------------------------------
A_SINGLET_A0 = 138.49       # X^1 Sigma_g+  a_S [a0]  Falke 2008 (pot. B, BO-corrected)
A_SINGLET_UNC_A0 = 0.12
A_TRIPLET_A0 = -33.48       # a^3 Sigma_u+  a_T [a0]  Falke 2008
A_TRIPLET_UNC_A0 = 0.18
# Best central estimates (Chapurin 2019, no individual error bars):
A_SINGLET_CHAPURIN_A0 = 138.85
A_TRIPLET_CHAPURIN_A0 = -33.40

C6_AU = 3921.0              # K2 dispersion coeff [a.u.]  D'Errico 2007 fit
C6_UNC_AU = 8.0
# Alternatives (all give R_vdW = 64.5-64.6 a0): Derevianko 3897(15); Falke ~3925.9.
C8_AU = None
_ST_SOURCE = "Falke 2008 (a_S,a_T); D'Errico 2007 (C6=3921(8)); Chapurin 2019 central"


@dataclass(frozen=True)
class FeshbachResonance:
    """One s-wave Feshbach resonance of a collision channel (empirical model).

    Attributes
    ----------
    B0_gauss : resonance position (G), experimental where measured.
    width_gauss : magnetic width Delta (G), ``B_zero = B0 + Delta`` for an isolated pole.
    a_bg_a0 : *local* background near this pole (a0), Etrych CC.
    zero_crossing_gauss : measured ``a = 0`` field tied to this pole, or None.
    decay_gauss : inelastic width gamma (G, > 0) for lossy channels, else None.
    source : provenance.
    verified : True for a measured position; False for theory-only.
    pole_strength_a0G : ``a_bg*Delta`` (a0 G); measured where reported, else
        ``a_bg_a0 * width_gauss``.
    """

    B0_gauss: float
    width_gauss: float
    a_bg_a0: float
    zero_crossing_gauss: Optional[float] = None
    decay_gauss: Optional[float] = None
    source: str = ""
    verified: bool = True
    pole_strength_a0G: Optional[float] = None

    @property
    def pole_strength(self) -> float:
        if self.pole_strength_a0G is not None:
            return self.pole_strength_a0G
        return self.a_bg_a0 * self.width_gauss


def channel_key(state_a: Tuple[int, int], state_b: Tuple[int, int]):
    """Order-independent key for a pair channel (intra or inter)."""
    return frozenset((tuple(state_a), tuple(state_b)))


def _build_tables():
    res, zcs = {}, {}
    for z in kf.ZERO_CROSSINGS:
        zcs.setdefault(channel_key(z.state_a, z.state_b), []).append(z.B_zero)
    for r in kf.RESONANCES + kf.THEORY_ONLY:
        if r.partial_wave != 's' or r.width_theory is None or r.a_bg_theory is None:
            continue
        key = channel_key(r.state_a, r.state_b)
        zc = [z for z in zcs.get(key, []) if abs(z - (r.B0 + r.width_theory)) < 2.0]
        decay = -r.gamma_inel_theory if (r.lossy and r.gamma_inel_theory) else None
        res.setdefault(key, []).append(FeshbachResonance(
            r.B0, r.width_theory, r.a_bg_theory, zc[0] if zc else None, decay,
            r.source, r.B0_unc > 0, r.pole_strength))
    for k in res:
        res[k].sort(key=lambda x: x.B0_gauss)
    for k in zcs:
        zcs[k].sort()
    return res, zcs


RESONANCES, ZERO_CROSSINGS_GAUSS = _build_tables()


def _pole_sum(res: List[FeshbachResonance], B, skip: Optional[int] = None):
    return sum(r.pole_strength / (B - r.B0_gauss)
               for i, r in enumerate(res) if i != skip)


@lru_cache(maxsize=None)
def _background_anchors(key) -> Tuple[Tuple[float, float], ...]:
    res = RESONANCES.get(key, [])
    poles = [(r.B0_gauss, r.a_bg_a0 + _pole_sum(res, r.B0_gauss, skip=i))
             for i, r in enumerate(res)]
    zeros = [(Bz, _pole_sum(res, Bz)) for Bz in ZERO_CROSSINGS_GAUSS.get(key, [])
             if all(abs(Bz - r.B0_gauss) > 5.0 for r in res)]   # a(Bz) = 0 exactly
    # The local a_bg of strongly overlapping poles absorbs part of the
    # neighbour's tail, so the implied background can be inconsistent (|1,0>
    # at 58.97 G gives -56 a0 against ~-24 elsewhere): reject pole anchors
    # more than 15 a0 from the channel median.  Measured zeros are kept.
    med = np.median([p[1] for p in poles + zeros])
    pts = [p for p in poles if abs(p[1] - med) < 15.0] + zeros
    # published coupled-channels values far (> 10 G) from any pole pin the
    # background where no resonance data exist (|1,-1>+|1,0> near 56 G)
    for a_, b_, B, a_pub, _model, _src in kf.THEORY_POINTS:
        if channel_key(a_, b_) == key and all(abs(B - r.B0_gauss) > 10.0 for r in res):
            pts.append((B, a_pub + _pole_sum(res, B)))
    pts = sorted(pts or poles)
    merged = [list(pts[0])]
    for B, b in pts[1:]:                     # average anchors closer than 1 G
        if B - merged[-1][0] < 1.0:
            n = merged[-1][2] if len(merged[-1]) > 2 else 1
            merged[-1] = [(merged[-1][0] * n + B) / (n + 1), (merged[-1][1] * n + b) / (n + 1), n + 1]
        else:
            merged.append([B, b])
    return tuple((m[0], m[1]) for m in merged)


def background_function(state_a, state_b) -> Callable:
    """Smooth background ``b(B)`` (a0) of the sum-of-poles model for the channel."""
    anchors = _background_anchors(channel_key(state_a, state_b))
    if not anchors:
        raise KeyError(f"no resonances tabulated for {state_a}+{state_b}")
    xs = np.array([p[0] for p in anchors])
    ys = np.array([p[1] for p in anchors])
    return lambda B: np.interp(B, xs, ys)


def singlet_triplet():
    """Return ``(a_S, a_T, C6)`` = (138.49, -33.48, 3921.0) [a0, a0, a.u.]."""
    return A_SINGLET_A0, A_TRIPLET_A0, C6_AU


def background_channel(state_a, state_b) -> Optional[float]:
    """Mean background (a0) of the channel's empirical model, or None."""
    anchors = _background_anchors(channel_key(state_a, state_b))
    return float(np.mean([p[1] for p in anchors])) if anchors else None


def resonances_for(state_a, state_b) -> List[FeshbachResonance]:
    """Tabulated s-wave resonances for the {state_a, state_b} channel ([] if none)."""
    return list(RESONANCES.get(channel_key(state_a, state_b), []))


def zero_crossings_for(state_a, state_b) -> List[float]:
    """Measured a=0 fields (Gauss) of the channel."""
    return list(ZERO_CROSSINGS_GAUSS.get(channel_key(state_a, state_b), []))
