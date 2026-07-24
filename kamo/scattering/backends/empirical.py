"""Empirical (tabulated-resonance) scattering-length backend.

Closed-form model for the s-wave scattering length of a K39 collision channel::

    a(B) = a_bg * prod_i ( 1 - Delta_i / (B - B0_i) )

with an optional inelastic (two-body-loss) regularisation that makes ``a``
complex near lossy resonances::

    a(B) = a_bg * prod_i ( 1 - Delta_i / ( (B - B0_i) + i * gamma_i/2 ) )

The resonance parameters live in
:mod:`kamo.scattering.data.k39_params` and are **provisional / unverified**
(see that module's warning).  This backend is exact *given* the table.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..data import k39_params as kp


class EmpiricalBackend:
    """Tabulated-resonance a(B) engine.

    Parameters
    ----------
    warn_unverified : bool
        Emit a one-time warning that the underlying parameters are provisional
        (default True).
    """

    name = "empirical"

    def __init__(self, warn_unverified: bool = True):
        self._warned = False
        self.warn_unverified = warn_unverified

    def _maybe_warn(self):
        if self.warn_unverified and not self._warned:
            import warnings
            warnings.warn(
                "EmpiricalBackend: K39 resonance parameters are PROVISIONAL and "
                "not literature-verified (kamo.scattering.data.k39_params). "
                "Positions/widths/backgrounds are placeholders — verify before "
                "trusting a(B).", stacklevel=3)
            self._warned = True

    def has_channel(self, state_a: Tuple[int, int], state_b: Tuple[int, int]) -> bool:
        return bool(kp.resonances_for(state_a, state_b))

    def scattering_length(self, state_a: Tuple[int, int],
                          state_b: Tuple[int, int],
                          B_gauss) -> "np.ndarray | complex | float":
        """Scattering length (a0) of channel ``{a, b}`` at ``B_gauss``.

        Returns a real array/scalar when no resonance in the channel carries an
        inelastic width, otherwise a complex array/scalar (``a_re - i a_im``,
        ``a_im >= 0`` = loss).

        Raises
        ------
        KeyError
            If the channel has no tabulated resonances.
        """
        self._maybe_warn()
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
                # NB: strict dissipativity of the product form also depends on
                # the (a_bg * width) residue sign; verify per-channel when
                # populating decay_gauss from a proper (Hutson-style) model.
                denom = denom - 1j * (r.decay_gauss / 2.0)
            # avoid division warnings exactly at an undamped pole
            with np.errstate(divide="ignore", invalid="ignore"):
                term = 1.0 - r.width_gauss / denom
            prod = prod * term

        a = a_bg * prod
        if not any_decay:
            a = a.real
        # convention: a_im >= 0 represents loss -> return a_re - i a_im
        if scalar_in:
            return a[0]
        return a
