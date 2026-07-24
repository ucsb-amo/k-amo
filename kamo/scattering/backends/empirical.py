"""Empirical (tabulated-resonance) scattering-length backend.

Multi-resonance product form for the s-wave scattering length of a K39
collision channel::

    a(B) = a_bg_channel * prod_i ( 1 - Delta_i / (B - B0_i) )

with an optional inelastic (two-body-loss) regularisation that makes ``a``
complex near lossy resonances::

    a(B) = a_bg_channel * prod_i ( 1 - Delta_i / ( (B - B0_i) - i*gamma_i/2 ) )

A single per-channel background ``a_bg_channel`` (kp.A_BG_CHANNEL) is used; the
product reconstructs each pole's *local* background via the other poles' tails.
Resonance parameters live in :mod:`kamo.scattering.data.k39_params` and are
literature-verified (Etrych 2023 / Chapurin 2019 / Falke 2008 / D'Errico 2007);
a few secondary widths are estimated (see per-resonance ``.verified``).
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

        Real array/scalar when no resonance in the channel carries an inelastic
        width, else complex (``a_re - i a_im``, ``a_im >= 0`` = loss).

        Raises
        ------
        KeyError
            If the channel has no tabulated resonances.
        """
        res = kp.resonances_for(state_a, state_b)
        if not res:
            raise KeyError(
                f"No tabulated resonances for channel {tuple(state_a)}+"
                f"{tuple(state_b)}. Add them to kamo.scattering.data.k39_params "
                f"or use the MQDT backend.")

        B = np.asarray(B_gauss, dtype=float)
        scalar_in = (B.ndim == 0)
        B = np.atleast_1d(B)

        a_bg = kp.background_channel(state_a, state_b)
        if a_bg is None:
            a_bg = res[0].a_bg_a0
        any_decay = any(r.decay_gauss for r in res)
        dtype = complex if any_decay else float
        prod = np.ones_like(B, dtype=dtype)
        for r in res:
            denom = (B - r.B0_gauss)
            if r.decay_gauss:
                # dissipative convention: a = a_re - i a_im (a_im >= 0 = loss).
                # Strict dissipativity also depends on the (a_bg*width) residue
                # sign; verify per-channel when populating decay_gauss.
                denom = denom - 1j * (r.decay_gauss / 2.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = 1.0 - r.width_gauss / denom
            prod = prod * term

        a = a_bg * prod
        if not any_decay:
            a = a.real
        if scalar_in:
            return a[0]
        return a
