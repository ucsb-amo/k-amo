"""Empirical (tabulated-resonance) scattering-length backend.

Sum-of-poles form for the s-wave scattering length of a K39 collision channel::

    a(B) = b(B) - sum_i s_i / (B - B0_i)          s_i = a_bg,i * Delta_i

with measured positions ``B0_i``, pole strengths ``s_i`` and a smooth background
``b(B)`` anchored at the poles and at measured zero crossings (see
:mod:`kamo.scattering.data.k39_params`).  In lossy channels the pole
denominators become ``(B - B0_i) - i*gamma_i/2`` (``gamma_i > 0``), so
``a = a_re - i a_im`` with ``a_im >= 0``.

Six channels are tabulated: |1,1>, |1,0>, |1,-1> intra, and the |1,1>+|1,0>,
|1,1>+|1,-1>, |1,0>+|1,-1> mixtures.  Use the coupled-channels backend for
anything else.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..data import k39_params as kp


class EmpiricalBackend:
    """Tabulated-resonance a(B) engine."""

    name = "empirical"

    def has_channel(self, state_a: Tuple[int, int], state_b: Tuple[int, int]) -> bool:
        return bool(kp.resonances_for(state_a, state_b))

    def scattering_length(self, state_a: Tuple[int, int],
                          state_b: Tuple[int, int],
                          B_gauss) -> "np.ndarray | complex | float":
        """Scattering length (a0) of channel ``{a, b}`` at ``B_gauss``.

        Real unless a resonance in the channel carries an inelastic width, then
        complex (``a_re - i a_im``, ``a_im >= 0``).

        Raises
        ------
        KeyError
            If the channel has no tabulated resonances.
        """
        res = kp.resonances_for(state_a, state_b)
        if not res:
            raise KeyError(
                f"No tabulated resonances for channel {tuple(state_a)}+"
                f"{tuple(state_b)}. Add them to kamo.scattering.data.k39_feshbach "
                f"or use the coupled-channels backend.")

        B = np.asarray(B_gauss, dtype=float)
        scalar_in = (B.ndim == 0)
        B = np.atleast_1d(B)

        lossy = any(r.decay_gauss for r in res)
        a = kp.background_function(state_a, state_b)(B).astype(complex if lossy else float)
        for r in res:
            denom = B - r.B0_gauss
            if r.decay_gauss:
                denom = denom - 1j * (r.decay_gauss / 2.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                a = a - r.pole_strength / denom
        if scalar_in:
            return a[0]
        return a
