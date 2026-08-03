"""Full coupled-channels s-wave scattering-length engine for K39 (l=0).

Solves the multichannel radial Schrodinger equation for two K39 ground-state
atoms in a given total M_F block:

    u''(r) = 2 mu [ W(r) + E_thr - E ] u(r) ,   u = r*psi ,  l = 0

with interaction ``W(r) = P_S V_S(r) + P_T V_T(r)`` built from the
singlet/triplet projectors between symmetrized pair channels and the tuned
model potentials. Channel thresholds E_thr(B) come from kamo.hamiltonian.

Propagated with the renormalized Numerov method; the scattering length is
extracted by matching to asymptotic sin/cos (open), linear (at-threshold) and
exponential (closed) forms and imposing decay of the closed channels.
All internal calculation is in atomic units.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import channels as _ch
from . import potentials as _pot
from .units import HARTREE_HZ
from .thresholds import K39Thresholds

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not a else a[0]


@njit(cache=True)
def _propagate_numba(VS, VT, AS, AT, dvec, h2_12, n, nch):
    """Renormalized-Numerov propagation; returns (u1, u2) at the last two r."""
    I = np.eye(nch)
    Rinv_prev = np.zeros((nch, nch))
    A_prev = np.zeros((nch, nch)); R_prev = np.zeros((nch, nch))
    A_cur = np.zeros((nch, nch))
    for k in range(n):
        Q = VS[k] * AS + VT[k] * AT
        for i in range(nch):
            Q[i, i] += dvec[i]
        A = I - h2_12 * Q
        U = 12.0 * np.linalg.inv(A) - 10.0 * I
        R = U - Rinv_prev
        Rinv_prev = np.linalg.inv(R)
        if k == n - 2:
            A_prev = A.copy(); R_prev = R.copy()
        if k == n - 1:
            A_cur = A.copy()
    u1 = np.linalg.solve(A_prev, np.eye(nch))
    u2 = np.linalg.solve(A_cur, R_prev)
    return u1, u2


def _spin_projection_matrices(channels: List[_ch.PairChannel]):
    """Return (P_S, P_T, Gram) matrices between symmetrized channels."""
    psis = np.array([c.spin_state() for c in channels])   # (nch, 64)
    Psing = _ch.singlet_projector()
    Gram = psis @ psis.T
    P_S = psis @ Psing @ psis.T
    P_T = Gram - P_S
    return P_S, P_T, Gram


class CoupledChannels:
    """Coupled-channels a(B) for K39 ground-state s-wave collisions."""

    def __init__(self, B_max: float = 1000.0, dB: float = 0.05,
                 r_in: float = 9.0, r_out: float = 2000.0, h: float = 0.015,
                 C12_S=None, C12_T=None):
        self.th = K39Thresholds(B_max_gauss=B_max, dB_gauss=dB)
        self.mu = _pot.MU_AU
        self.C6 = _pot.C6_AU
        self.r_in, self.r_out, self.h = r_in, r_out, h
        from .data import k39_params as kp
        a_S, a_T, _ = kp.singlet_triplet()
        self.C12_S = C12_S if C12_S is not None else _pot.tune_C12(a_S)[0]
        self.C12_T = C12_T if C12_T is not None else _pot.tune_C12(a_T)[0]
        n = int(round((r_out - r_in) / h)) + 1
        self.r = r_in + h * np.arange(n)
        self.VS = _pot.v_lj(self.r, self.C6, self.C12_S)
        self.VT = _pot.v_lj(self.r, self.C6, self.C12_T)

    def _spin_matrices_at_B(self, channels, B_gauss):
        """Field-dependent (P_S, P_T) between symmetrized channels at B.

        Uses kamo's field-dependent single-atom eigenvectors so the
        singlet/triplet content follows the Breit-Rabi mixing (correct at all
        fields; -> pure triplet/singlet in the Paschen-Back limit).
        """
        Psing = _ch.singlet_projector()
        psis = np.empty((len(channels), 64))
        for n, c in enumerate(channels):
            ca = self.th.state_composition(*c.a, B_gauss)
            cb = self.th.state_composition(*c.b, B_gauss)
            psi = np.kron(ca, cb)
            if c.a != c.b:
                psi = psi + np.kron(cb, ca)
            nrm = np.linalg.norm(psi)
            psis[n] = psi / nrm if nrm > 1e-14 else psi
        P_S = psis @ Psing @ psis.T
        Gram = psis @ psis.T
        return P_S, Gram - P_S

    def _thresholds_hartree(self, channels, B_gauss) -> np.ndarray:
        E = np.array([float(self.th.pair_threshold(c.a, c.b, B_gauss))
                      for c in channels])          # Hz
        return E / HARTREE_HZ                        # Hartree

    def scattering_length(self, entrance_a, entrance_b, B_gauss: float,
                          E_coll_hartree: float = 0.0) -> complex:
        """Complex s-wave scattering length (a0) of the entrance channel at B."""
        entrance = _ch.PairChannel(*sorted((tuple(entrance_a), tuple(entrance_b))))
        channels = _ch.enumerate_channels(entrance.M_F)
        ei = next(i for i, c in enumerate(channels)
                  if (c.a, c.b) == (entrance.a, entrance.b))
        channels = [channels[ei]] + channels[:ei] + channels[ei + 1:]

        P_S, P_T = self._spin_matrices_at_B(channels, B_gauss)
        nch = len(channels)
        Ethr = self._thresholds_hartree(channels, B_gauss)
        E = Ethr[0] + E_coll_hartree

        mu, h = self.mu, self.h
        h2_12 = h * h / 12.0
        AS = (2.0 * mu) * P_S
        AT = (2.0 * mu) * P_T
        dvec = 2.0 * mu * (Ethr - E)
        n = len(self.r)
        u1, u2 = _propagate_numba(self.VS, self.VT, AS, AT, dvec, h2_12, n, nch)
        return self._match(channels, Ethr, E, u1, u2, self.r[-2], self.r[-1])

    def _match(self, channels, Ethr, E, u1, u2, r1, r2) -> complex:
        """Match propagated u at (r1,r2) to asymptotic forms; return complex a."""
        mu = self.mu
        nch = len(channels)
        tol = 1e-14
        kind = []
        kk = np.zeros(nch)
        for i in range(nch):
            d = E - Ethr[i]
            if d > tol:
                kind.append('open'); kk[i] = np.sqrt(2 * mu * d)
            elif d < -tol:
                kind.append('closed'); kk[i] = np.sqrt(-2 * mu * d)
            else:
                kind.append('thr'); kk[i] = 0.0

        def basis(i, r):
            if kind[i] == 'open':
                return np.sin(kk[i] * r), np.cos(kk[i] * r)
            if kind[i] == 'thr':
                return r, 1.0
            return np.exp(-kk[i] * (r - r2)), np.exp(kk[i] * (r - r2))

        X = np.zeros((nch, nch)); Y = np.zeros((nch, nch))
        for i in range(nch):
            P1, Q1 = basis(i, r1); P2, Q2 = basis(i, r2)
            Mi = np.array([[P1, Q1], [P2, Q2]])
            rhs = np.vstack([u1[i, :], u2[i, :]])
            sol = np.linalg.solve(Mi, rhs)
            X[i, :] = sol[0, :]; Y[i, :] = sol[1, :]

        open_idx = [i for i in range(nch) if kind[i] in ('open', 'thr')]
        closed_idx = [i for i in range(nch) if kind[i] == 'closed']

        if closed_idx:
            from scipy.linalg import null_space
            Vnull = null_space(Y[closed_idx, :])
        else:
            Vnull = np.eye(nch)
        Xo = X[np.ix_(open_idx, range(nch))] @ Vnull
        Yo = Y[np.ix_(open_idx, range(nch))] @ Vnull

        entrance_pos = open_idx.index(0)
        if len(open_idx) == 1:
            xe = Xo[0, 0]; ye = Yo[0, 0]
            if kind[0] == 'thr':
                return complex(-ye / xe)
            return complex(-(ye / xe) / kk[0])
        K = Yo @ np.linalg.inv(Xo)
        Io = np.eye(len(open_idx))
        S = (Io + 1j * K) @ np.linalg.inv(Io - 1j * K)
        See = S[entrance_pos, entrance_pos]
        return complex((1.0 / (1j * kk[0])) * (1 - See) / (1 + See))
