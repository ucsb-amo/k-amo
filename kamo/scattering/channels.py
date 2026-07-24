"""Layer 1 — two-body s-wave channel space for K39 ground-state collisions.

Builds the symmetrized pair-channel basis for identical bosonic K39 atoms and
the frame transformation from the asymptotic atomic hyperfine basis to the
short-range total-electronic-spin (singlet / triplet) basis.

Physics encoded here
--------------------
* Single-atom ground state: electron spin ``s = 1/2``, nuclear spin
  ``I = 3/2``; hyperfine ``F in {1, 2}``.  ``|F mF> = sum CG |m_s, m_i>``.
* s-wave (``l = 0``) is spatially symmetric, so for identical bosons the
  internal (spin) state must be **symmetric** under atom exchange:
  ``|a,a>`` for a == b, else ``(|a,b> + |b,a>)/sqrt(2)``.
* Total ``M_F = mF_a + mF_b`` is conserved; channels are grouped by it.
* Short-range interaction is diagonal in total electronic spin
  ``S = s_a + s_b in {0, 1}`` (singlet/triplet) and independent of nuclear
  spin, so a channel's singlet/triplet content sets its background scattering.

Everything here is pure angular-momentum algebra and is self-validating
(orthonormality, ``f_S + f_T = 1``, S^2 eigenvalues); it needs no external
fitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

S_ELEC = 0.5
I_NUC = 1.5

# ordered single-atom uncoupled basis |m_s, m_i>
_MS = [-0.5, 0.5]
_MI = [-1.5, -0.5, 0.5, 1.5]
_ATOM_BASIS = [(ms, mi) for ms in _MS for mi in _MI]   # len 8
_ATOM_INDEX = {sp: k for k, sp in enumerate(_ATOM_BASIS)}


def _cg(j1, m1, j2, m2, j3, m3) -> float:
    from sympy.physics.wigner import clebsch_gordan
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))


@lru_cache(maxsize=None)
def hf_state_vector(F: int, mF: int) -> Tuple[float, ...]:
    """Return ``|F, mF>`` as an 8-vector over the ``|m_s, m_i>`` atom basis.

    ``|F mF> = sum_{m_s + m_i = mF} <s m_s I m_i | F mF> |m_s, m_i>``.
    """
    v = np.zeros(len(_ATOM_BASIS))
    for k, (ms, mi) in enumerate(_ATOM_BASIS):
        if abs((ms + mi) - mF) > 1e-9:
            continue
        v[k] = _cg(S_ELEC, ms, I_NUC, mi, F, mF)
    return tuple(v)


# -- two-spin electronic singlet/triplet projectors (on m_sa, m_sb) ---------
def _electron_singlet_vector() -> np.ndarray:
    """Singlet ``(|up,down> - |down,up>)/sqrt2`` over (m_sa, m_sb) in _MS order."""
    v = np.zeros((2, 2))
    # |up> index 1, |down> index 0 in _MS = [-0.5, +0.5]
    up, down = 1, 0
    v[up, down] = 1 / np.sqrt(2)
    v[down, up] = -1 / np.sqrt(2)
    return v.reshape(-1)


@dataclass(frozen=True)
class PairChannel:
    """A symmetrized s-wave pair channel of two K39 ground-state atoms.

    Attributes
    ----------
    a, b : (F, mF) single-atom labels (unordered; stored canonically a <= b).
    """

    a: Tuple[int, int]
    b: Tuple[int, int]

    @property
    def M_F(self) -> int:
        return self.a[1] + self.b[1]

    @property
    def identical(self) -> bool:
        return self.a == self.b

    def __repr__(self) -> str:
        fa, ma = self.a
        fb, mb = self.b
        return f"|{fa},{ma:+d}>+|{fb},{mb:+d}>"

    # -- symmetrized 64-vector over (m_sa,m_ia, m_sb,m_ib) -----------------
    def spin_state(self) -> np.ndarray:
        """Normalised symmetric two-atom spin state as a length-64 vector.

        Ordering: index = 8*i_a + i_b where i_x indexes ``_ATOM_BASIS``.
        """
        va = np.asarray(hf_state_vector(*self.a))
        vb = np.asarray(hf_state_vector(*self.b))
        psi = np.kron(va, vb)
        if not self.identical:
            psi = psi + np.kron(vb, va)
        nrm = np.linalg.norm(psi)
        if nrm < 1e-14:
            # Symmetric combination vanishes -> this pair supports no s-wave
            # channel (it is a purely antisymmetric / p-wave state).
            return psi
        return psi / nrm

    def is_swave_allowed(self) -> bool:
        """False when the symmetric spin combination vanishes (p-wave only)."""
        return np.linalg.norm(self.spin_state()) > 1e-12


def _singlet_projector_64() -> np.ndarray:
    """64x64 projector onto total electronic **singlet** (S=0).

    Acts on electron spins (m_sa, m_sb); identity on nuclear spins.
    Basis index = 8*i_a + i_b, i_x = 4*ms_slot + mi_slot within _ATOM_BASIS.
    """
    sing = _electron_singlet_vector()          # over (m_sa_slot, m_sb_slot), len 4
    P = np.zeros((64, 64))
    for ia, (msa, mia) in enumerate(_ATOM_BASIS):
        for ib, (msb, mib) in enumerate(_ATOM_BASIS):
            row = 8 * ia + ib
            slot_ab = 2 * _MS.index(msa) + _MS.index(msb)
            for ja, (msa2, mia2) in enumerate(_ATOM_BASIS):
                if mia2 != mia:
                    continue
                for jb, (msb2, mib2) in enumerate(_ATOM_BASIS):
                    if mib2 != mib:
                        continue
                    col = 8 * ja + jb
                    slot_ab2 = 2 * _MS.index(msa2) + _MS.index(msb2)
                    P[row, col] = sing[slot_ab] * sing[slot_ab2]
    return P


_P_SINGLET = None


def singlet_projector() -> np.ndarray:
    global _P_SINGLET
    if _P_SINGLET is None:
        _P_SINGLET = _singlet_projector_64()
    return _P_SINGLET


def singlet_triplet_fractions(channel: PairChannel) -> Tuple[float, float]:
    """Return ``(f_S, f_T)`` — singlet and triplet weight of the channel.

    ``f_S + f_T = 1`` for any s-wave-allowed channel.  These weights set the
    channel's *background* scattering length in the first-order (degenerate
    internal states) approximation; the full resonance structure requires the
    coupled multichannel MQDT treatment.
    """
    psi = channel.spin_state()
    nrm = np.linalg.norm(psi)
    if nrm < 1e-12:
        return (0.0, 0.0)
    psi = psi / nrm
    P = singlet_projector()
    f_S = float(psi @ (P @ psi))
    return (f_S, 1.0 - f_S)


def background_scattering_length_a0(channel: PairChannel,
                                    a_S: float, a_T: float) -> float:
    """First-order background scattering length ``f_S*a_S + f_T*a_T`` (a0).

    .. note::
       This is the *degenerate-internal-states* estimate: correct far from any
       Feshbach resonance, but it does NOT reproduce resonance poles.  Use the
       empirical or MQDT backend for a(B) near resonances.
    """
    f_S, f_T = singlet_triplet_fractions(channel)
    return f_S * a_S + f_T * a_T


def enumerate_channels(M_F: int, F_values=(1, 2)) -> List[PairChannel]:
    """All symmetric s-wave pair channels with total ``M_F`` (canonical order).

    Parameters
    ----------
    M_F : total magnetic quantum number mF_a + mF_b.
    F_values : which single-atom F manifolds to include (default both, 1 and 2).

    Returns
    -------
    list of :class:`PairChannel`, each s-wave-allowed, with ``a <= b``.
    """
    singles = []
    for F in F_values:
        for mF in range(-F, F + 1):
            singles.append((F, mF))
    seen = set()
    out = []
    for i, sa in enumerate(singles):
        for sb in singles[i:]:
            if sa[1] + sb[1] != M_F:
                continue
            a, b = sorted((sa, sb))
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            ch = PairChannel(a, b)
            if ch.is_swave_allowed():
                out.append(ch)
    return out
