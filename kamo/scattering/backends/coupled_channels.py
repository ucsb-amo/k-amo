"""Coupled-channels backend: first-principles a(B) for ANY channel.

Wraps :class:`kamo.scattering.coupled_channels.CoupledChannels` with the
Falke/Tiemann K2 potentials, inner walls calibrated to the measured Feshbach
resonance positions (:mod:`kamo.scattering.data.k39_calibration`).  Needs no
per-channel table, so it computes intra AND inter channels (e.g.
|1,-1>+|1,0>), F=2 channels, and complex ``a`` where inelastic channels are
open.

Accuracy: the 15 measured elastic s-wave resonance positions of the database
within 0.07 G, except 58.97 G (-0.14 G; chi2/dof 0.58 with a 20 mG model
floor), and the two held-out lossy |1,1>+|1,-1> resonances within 0.4 sigma.
Median |a - a_Kokkelmans| is 0.003-0.016 a0 away from poles in all 8
intra-state channels (Im a of the F=2 channels to ~0.5%).  Poles are fitted,
zeros are not: the |1,1> 350.4 G and |1,0> 393.2 G zero crossings come out
~0.5 G high (a off by <~0.3 a0 there).  Not included: magnetic
dipole-dipole / second-order spin-orbit couplings and l > 0 partial waves
(no d-wave features, e.g. |1,0> at 60.1 G or |2,-2> at 126 G).
"""

from __future__ import annotations

import numpy as np


class CoupledChannelsBackend:
    """First-principles coupled-channels a(B) engine (any channel)."""

    name = "coupled_channels"

    def __init__(self, B_max: float = 1000.0, dB: float = 0.05, **cc_kwargs):
        from ..coupled_channels import CoupledChannels
        self.engine = CoupledChannels(B_max=B_max, dB=dB, **cc_kwargs)

    def has_channel(self, state_a, state_b) -> bool:
        return True

    def scattering_length(self, state_a, state_b, B_gauss):
        """Complex scattering length (a0) for channel {a,b} at B (scalar/array)."""
        B = np.asarray(B_gauss, dtype=float)
        if B.ndim == 0:
            return self.engine.scattering_length(state_a, state_b, float(B))
        return np.array([self.engine.scattering_length(state_a, state_b, float(x))
                         for x in B.ravel()]).reshape(B.shape)
