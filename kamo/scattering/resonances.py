"""Locate and characterise Feshbach resonances in a computed a(B).

Works on any callable ``a_of_B(B) -> a`` (scalar field in Gauss, scattering
length in a0), e.g. a coupled-channels channel or the empirical backend.
Poles and zeros are features of ``Re a``.

On a grid, a sign change of ``a`` is either a pole or a zero crossing.  Brent's
method on ``a`` converges to whichever it is (it only needs the bracket), and
``|a|`` at the converged point then tells them apart: ~0 at a zero, huge at a
pole.  A resonance narrower than the grid step hides its pole and zero in one
cell (no sign change), so scan with ``dB`` below the smallest width of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq


@dataclass
class ResonanceFit:
    """Computed resonance parameters, ``a(B) ~ a_bg (1 - Delta/(B - B0))``.

    ``pole_strength = a_bg * Delta`` (a0 G) is the model-independent residue
    (``a ~ -a_bg*Delta/(B - B0)`` at the pole).  ``zero_crossing`` is the
    ``a = 0`` field on the resonance's zero side, if found; ``width``/``a_bg``
    are then ``B_zero - B0`` and ``pole_strength / width`` (the convention of
    Etrych et al. 2023).
    """

    B0: float
    pole_strength: float
    zero_crossing: Optional[float] = None
    width: Optional[float] = None
    a_bg: Optional[float] = None


def _real(fn):
    return lambda B: float(np.real(fn(float(B))))


def _resolve_sign_change(fn, lo, hi, xtol=1e-7):
    """Brent on ``Re a`` across a sign change -> ``(B*, 'pole' | 'zero')``.

    Classified on ``|a|`` (complex modulus): a zero ends below both endpoint
    values, a pole above at least the far one (a grid point can land
    arbitrarily close to the pole).  The modulus matters for lossy poles,
    where ``Re a`` passes smoothly through the background at ``B0`` while
    ``|a| ~ 2 a_bg Delta / Gamma`` peaks there.
    """
    f = _real(fn)
    m = lambda B: abs(complex(fn(float(B))))
    Bs = brentq(f, lo, hi, xtol=xtol)
    kind = 'pole' if m(Bs) > min(m(lo), m(hi)) else 'zero'
    return Bs, kind


def _grid_features(fn, B, xtol):
    a = np.array([np.real(fn(float(b))) for b in B])
    out = []
    for i in range(len(B) - 1):
        if np.isfinite(a[i]) and np.isfinite(a[i + 1]) and a[i] * a[i + 1] < 0:
            out.append(_resolve_sign_change(fn, B[i], B[i + 1], xtol))
    return out


def find_features(a_of_B: Callable, B_lo: float, B_hi: float, dB: float = 0.1,
                  xtol: float = 1e-7):
    """All poles and zero crossings of ``Re a(B)`` in ``[B_lo, B_hi]``.

    Returns ``(poles, zeros)``, sorted lists of fields (Gauss).
    """
    feats = _grid_features(a_of_B, np.arange(B_lo, B_hi + 0.5 * dB, dB), xtol)
    return (sorted(b for b, k in feats if k == 'pole'),
            sorted(b for b, k in feats if k == 'zero'))


def locate_pole(a_of_B: Callable, B_guess: float, half_window: float = 1.0,
                dB: float = 0.05, xtol: float = 1e-7, grow: int = 4) -> float:
    """Pole of ``Re a(B)`` nearest ``B_guess``.

    Scans ``B_guess +- half_window`` with step ``dB`` (doubling the window up
    to ``grow`` times if nothing is found).
    """
    hw = half_window
    for _ in range(grow + 1):
        feats = _grid_features(a_of_B, np.arange(B_guess - hw, B_guess + hw + 0.5 * dB, dB), xtol)
        poles = [b for b, k in feats if k == 'pole']
        if poles:
            return min(poles, key=lambda b: abs(b - B_guess))
        hw *= 2
    raise RuntimeError(f"no pole found within +-{hw / 2:.3g} G of {B_guess} G")


def refine_pole(a_of_B: Callable, B_prev: float, span: float = 0.05,
                xtol: float = 1e-7) -> float:
    """Re-locate a pole known to lie near ``B_prev`` (cheap, for fitting loops).

    Expands ``[B_prev - span, B_prev + span]`` until ``1/a`` changes sign
    continuously across it, then applies Brent's method; falls back to
    :func:`locate_pole` if a zero crossing gets in the way.
    """
    f = _real(a_of_B)
    g = lambda x: 1.0 / f(x)
    lo, hi = B_prev - span, B_prev + span
    for _ in range(6):
        glo, ghi = g(lo), g(hi)
        if glo * ghi < 0 and abs(f(lo)) > 50 and abs(f(hi)) > 50:
            return brentq(g, lo, hi, xtol=xtol)
        lo, hi = B_prev - 2 * (B_prev - lo), B_prev + 2 * (hi - B_prev)
    return locate_pole(a_of_B, B_prev, half_window=max(span * 8, 0.5))


def characterize(a_of_B: Callable, B0: float, zero_search: float = 100.0,
                 dB_max: float = 2.0, B_min: float = 0.0, B_max: float = np.inf,
                 side: Optional[float] = None) -> ResonanceFit:
    """Pole strength and (if present) zero crossing / width / a_bg at pole ``B0``.

    ``pole_strength`` is the symmetric residue ``-(B-B0) Re a(B)`` at
    ``B0 +- eps``, with ``eps`` widened past the inelastic width when ``a`` is
    complex (a lossy pole is smoothed over ``Gamma_inel``).

    The zero crossing (``B_zero = B0 + Delta``) is searched outward with
    geometrically growing steps (capped at ``dB_max``) up to ``zero_search`` G,
    stopping at the next pole.  ``side`` (+1 / -1, i.e. the sign of ``Delta``)
    restricts the search to one side -- needed when a neighbouring resonance's
    zero lies closer than this one's; otherwise the nearer zero is kept.
    """
    fc = lambda B: complex(a_of_B(float(B)))
    f = _real(a_of_B)
    eps = 1e-4
    while True:
        ap, am = fc(B0 + eps), fc(B0 - eps)
        if (max(abs(ap.imag), abs(am.imag)) < 0.02 * min(abs(ap.real), abs(am.real))
                or eps >= 0.5):
            break
        eps *= 4
    strength = -0.5 * eps * (ap.real - am.real)
    fit = ResonanceFit(B0=B0, pole_strength=strength)

    offs = [10 * eps]
    while offs[-1] < zero_search:
        offs.append(min(offs[-1] * 2, offs[-1] + dB_max))
    for sgn in ((side,) if side else (+1.0, -1.0)):
        B_prev, a_prev = B0 + sgn * offs[0], f(B0 + sgn * offs[0])
        for off in offs[1:]:
            B_k = B0 + sgn * off
            if not (B_min <= B_k <= B_max):
                break
            a_k = f(B_k)
            if a_prev * a_k < 0:
                Bs, kind = _resolve_sign_change(a_of_B, min(B_prev, B_k), max(B_prev, B_k))
                if kind == 'zero' and (fit.zero_crossing is None
                                       or abs(Bs - B0) < abs(fit.zero_crossing - B0)):
                    fit.zero_crossing = Bs
                break
            B_prev, a_prev = B_k, a_k
    if fit.zero_crossing is not None:
        fit.width = fit.zero_crossing - B0
        fit.a_bg = strength / fit.width
    return fit
