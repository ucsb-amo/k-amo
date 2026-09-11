"""van der Waals MQDT backend for K39 — frame transform complete, a(B) GATED.

What is implemented and self-validating
---------------------------------------
The **frame transformation** from the symmetrized atomic pair-channel basis to
the short-range total-electronic-spin basis ``|S, M_S; m_ia, m_ib>`` — pure
recoupling of the two electron spins.  It is unitary and diagonalises the
short-range (singlet/triplet) interaction; unit tests check its norm and that
it reproduces the singlet/triplet fractions of :mod:`kamo.scattering.channels`.

What is deliberately NOT shipped
--------------------------------
The multichannel ``a(B)`` — combining the singlet/triplet quantum defects
(a_S, a_T, C6) with the Zeeman-shifted thresholds through the QDT reference
functions.  Building that *blind* (without literature-verified K39 parameters
and a validation target) risks shipping plausible-but-wrong physics.  It is
therefore gated: :meth:`scattering_length` raises until parameters are pinned
and the result is validated against the empirical backend / published a(B).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from .. import channels as _ch

# short-range basis: (S, M_S, m_ia, m_ib)
_S_VALUES = (0, 1)


def _electron_recouple_cg(msa, msb, S, MS) -> float:
    """<1/2 msa; 1/2 msb | S MS> for two electron spins."""
    if abs((msa + msb) - MS) > 1e-9:
        return 0.0
    from sympy.physics.wigner import clebsch_gordan
    return float(clebsch_gordan(0.5, 0.5, S, msa, msb, MS))


@lru_cache(maxsize=None)
def _short_range_basis() -> Tuple[Tuple[int, float, float, float], ...]:
    """Enumerate |S, M_S, m_ia, m_ib> short-range basis states."""
    basis = []
    for S in _S_VALUES:
        for MS in [m for m in np.arange(-S, S + 1)]:
            for mia in _ch._MI:
                for mib in _ch._MI:
                    basis.append((int(S), float(MS), float(mia), float(mib)))
    return tuple(basis)


class MQDTBackend:
    """van der Waals MQDT engine (frame transform ready; a(B) gated)."""

    name = "mqdt"

    def frame_transform_vector(self, channel: _ch.PairChannel) -> np.ndarray:
        """Amplitudes of a pair channel over the ``|S,M_S,m_ia,m_ib>`` basis.

        Recouples the two electron spins of ``channel.spin_state()`` (a 64-vector
        over ``(m_sa,m_ia,m_sb,m_ib)``) into total electronic spin.  The result
        is a unit vector (for s-wave-allowed channels).
        """
        psi = channel.spin_state()               # over (i_a, i_b), len 64
        sr = _short_range_basis()
        out = np.zeros(len(sr))
        # psi index = 8*ia + ib ; _ATOM_BASIS[i] = (ms, mi)
        for k, (S, MS, mia, mib) in enumerate(sr):
            amp = 0.0
            for ia, (msa, mia_) in enumerate(_ch._ATOM_BASIS):
                if abs(mia_ - mia) > 1e-9:
                    continue
                for ib, (msb, mib_) in enumerate(_ch._ATOM_BASIS):
                    if abs(mib_ - mib) > 1e-9:
                        continue
                    coeff = _electron_recouple_cg(msa, msb, S, MS)
                    if coeff == 0.0:
                        continue
                    amp += coeff * psi[8 * ia + ib]
            out[k] = amp
        return out

    def frame_transform_matrix(self, M_F: int) -> Tuple[np.ndarray, List[_ch.PairChannel], tuple]:
        """``(U, channels, sr_basis)`` frame transform for all channels at M_F.

        ``U[i]`` is the short-range amplitude vector of ``channels[i]``.
        Rows are orthonormal (unitary onto the spanned subspace).
        """
        chans = _ch.enumerate_channels(M_F)
        U = np.array([self.frame_transform_vector(ch) for ch in chans])
        return U, chans, _short_range_basis()

    def singlet_fraction_via_frame(self, channel: _ch.PairChannel) -> float:
        """Singlet weight from the frame transform (cross-check on channels.py)."""
        v = self.frame_transform_vector(channel)
        sr = _short_range_basis()
        return float(sum(v[k] ** 2 for k, (S, *_1) in enumerate(sr) if S == 0))

    def scattering_length(self, state_a, state_b, B_gauss):
        raise NotImplementedError(
            "MQDTBackend.scattering_length is GATED: the multichannel a(B) is "
            "not yet validated and the K39 QDT parameters (a_S, a_T, C6, and "
            "resonance validation targets) are unverified (deep-research pass "
            "was rate-limited 2026-07-24). Use backend='empirical' for a(B); "
            "the frame-transform methods here (frame_transform_matrix, "
            "singlet_fraction_via_frame) are validated and usable.")
