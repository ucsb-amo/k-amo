"""Front-end: complex s-wave scattering length a(B) for K39 collision channels.

Ties together Layer-0 thresholds (:mod:`.thresholds`), Layer-1 channel algebra
(:mod:`.channels`) and a pluggable backend (:mod:`.backends`).

>>> from kamo.scattering import ScatteringModel
>>> m = ScatteringModel(B_max=600.0)
>>> m.intra((1, -1), 33.6)            # a(B) for |1,-1>+|1,-1>  [a0]
>>> m.inter((1, -1), (1, 0), 60.0)    # a(B) for |1,-1>+|1,0>
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import channels as _ch
from .data import k39_params as kp
from .thresholds import K39Thresholds


class ScatteringModel:
    """K39 ground-state s-wave scattering length vs magnetic field.

    Parameters
    ----------
    B_max : float
        Upper field for the internal threshold sweep (Gauss).
    backend : {"empirical", "mqdt"} or backend instance
        Engine used to compute a(B).  "mqdt" is currently UNVALIDATED (raises
        on a(B) but exposes the verified frame-transform machinery).
    dB : float
        Threshold sweep step (Gauss).
    """

    def __init__(self, B_max: float = 1000.0, backend="empirical", dB: float = 0.05):
        self.thresholds = K39Thresholds(B_max_gauss=B_max, dB_gauss=dB)
        self.backend = self._make_backend(backend)

    @staticmethod
    def _make_backend(backend):
        if isinstance(backend, str):
            if backend == "empirical":
                from .backends.empirical import EmpiricalBackend
                return EmpiricalBackend()
            if backend == "mqdt":
                from .backends.mqdt import MQDTBackend
                return MQDTBackend()
            raise ValueError(f"Unknown backend {backend!r}; use 'empirical' or 'mqdt'.")
        return backend

    # -- scattering length --------------------------------------------------
    def scattering_length(self, state_a: Tuple[int, int],
                          state_b: Tuple[int, int], B_gauss):
        """Complex/real scattering length (a0) of channel ``{a, b}`` at B."""
        return self.backend.scattering_length(state_a, state_b, B_gauss)

    def intra(self, state: Tuple[int, int], B_gauss):
        """Scattering length (a0) for two atoms both in ``state``."""
        return self.scattering_length(state, state, B_gauss)

    def inter(self, state_a: Tuple[int, int], state_b: Tuple[int, int], B_gauss):
        """Scattering length (a0) for one atom in each of ``a`` and ``b``."""
        return self.scattering_length(state_a, state_b, B_gauss)

    # -- channel structure --------------------------------------------------
    def singlet_triplet_fractions(self, state_a, state_b) -> Tuple[float, float]:
        """``(f_S, f_T)`` singlet/triplet content of the channel (self-checking)."""
        return _ch.singlet_triplet_fractions(_ch.PairChannel(*sorted((tuple(state_a),
                                                                      tuple(state_b)))))

    def background_estimate(self, state_a, state_b) -> float:
        """First-order background a (a0) = f_S*a_S + f_T*a_T (rough; see channels)."""
        a_S, a_T, _ = kp.singlet_triplet()
        ch = _ch.PairChannel(*sorted((tuple(state_a), tuple(state_b))))
        return _ch.background_scattering_length_a0(ch, a_S, a_T)

    def resonances(self, state_a, state_b) -> List[kp.FeshbachResonance]:
        """Provisional tabulated resonances for the channel."""
        return kp.resonances_for(state_a, state_b)

    def zero_crossings(self, state_a, state_b) -> List[float]:
        """Provisional a=0 zero-crossing fields (Gauss) for the channel."""
        return kp.zero_crossings_for(state_a, state_b)

    # -- inelastic / open-channel bookkeeping (M_F conserving) --------------
    def open_loss_channels(self, entrance_a, entrance_b, B_gauss: float) -> list:
        """M_F-conserving pair channels lying strictly below the entrance.

        A collision in ``{entrance_a, entrance_b}`` at ~zero energy can decay
        (two-body inelastic loss) only into channels that (a) conserve total
        ``M_F`` and (b) have a lower pair threshold.  If this list is empty the
        entrance channel is elastic (real a); otherwise it is lossy (a should
        be complex).

        This is the physical reason ``|1,-1>+|1,-1>`` is stable: at M_F = -2 no
        other pair channel sits below it.
        """
        M_F = entrance_a[1] + entrance_b[1]
        entrance = frozenset((tuple(entrance_a), tuple(entrance_b)))
        E_in = float(self.thresholds.pair_threshold(entrance_a, entrance_b, B_gauss))
        out = []
        for ch in _ch.enumerate_channels(M_F):
            if frozenset((ch.a, ch.b)) == entrance:
                continue
            E_c = float(self.thresholds.pair_threshold(ch.a, ch.b, B_gauss))
            if E_c < E_in - 1e-3:   # strictly below (loss is exothermic)
                out.append(ch)
        return out

    def is_lossy(self, entrance_a, entrance_b, B_gauss: float) -> bool:
        """True if the entrance channel has an open M_F-conserving loss channel."""
        return bool(self.open_loss_channels(entrance_a, entrance_b, B_gauss))
