"""Coupled-channels backend: first-principles a(B) for ANY channel.

Wraps :class:`kamo.scattering.coupled_channels.CoupledChannels`, which solves the
multichannel radial Schrodinger equation with model singlet/triplet potentials
tuned to the literature ``a_S, a_T, C6``.  Unlike the empirical backend this
needs no per-channel resonance table, so it computes intra AND inter channels
(e.g. |1,-1>+|1,0>) and complex a for inelastic channels.

.. note:: **Accuracy.** The model potentials reproduce the correct van der Waals
   tail, background scattering lengths, high-field (pure singlet/triplet) limits,
   and the qualitative multichannel resonance structure.  They are NOT deep
   enough to be in the full van der Waals *universal* regime, so quantitative
   Feshbach resonance *positions* are only approximate; pinning them requires the
   real Tiemann/Falke Born-Oppenheimer potentials.  For accurate a(B) near known
   resonances use ``backend='empirical'``.
"""

from __future__ import annotations

import numpy as np


class CoupledChannelsBackend:
    """First-principles coupled-channels a(B) engine (any channel)."""

    name = "coupled_channels"

    def __init__(self, B_max: float = 1000.0, dB: float = 0.05,
                 r_in: float = 9.0, r_out: float = 1800.0, h: float = 0.015):
        from ..coupled_channels import CoupledChannels
        self.engine = CoupledChannels(B_max=B_max, dB=dB, r_in=r_in,
                                      r_out=r_out, h=h)

    def scattering_length(self, state_a, state_b, B_gauss):
        """Complex scattering length (a0) for channel {a,b} at B (scalar/array)."""
        B = np.asarray(B_gauss, dtype=float)
        if B.ndim == 0:
            return self.engine.scattering_length(state_a, state_b, float(B))
        return np.array([self.engine.scattering_length(state_a, state_b, float(x))
                         for x in B])
