"""Full coupled-channels s-wave scattering-length engine for K39 (l=0).

Solves the multichannel radial Schrodinger equation for two K39 ground-state
atoms in a given total M_F block:

    u''(r) = 2 mu [ W(r) + E_thr - E ] u(r) ,   u = r*psi ,  l = 0

with interaction ``W(r) = P_S V_S(r) + P_T V_T(r)`` built from the
singlet/triplet projectors between symmetrized pair channels.  Channel
thresholds E_thr(B) and the field-dependent single-atom spin states come from
kamo.hamiltonian.

Potentials
----------
``potentials='tiemann'`` (default) uses the Falke/Tiemann K2 Born-Oppenheimer
curves (:mod:`.tiemann`).  A smooth short-range correction
``delta_X * s_X(r)`` (a step on the inner wall, see :func:`short_range_switch`)
can be added to each curve; it leaves the van der Waals tail untouched and is
the knob used to calibrate the singlet/triplet scattering lengths against
measured Feshbach resonance positions (:mod:`.calibration`).  The calibrated
default comes from :mod:`.data.k39_calibration`.  ``potentials='lj'`` selects
the legacy Lennard-Jones model potentials.

Numerics
--------
Renormalized Numerov on a segmented grid: a fine step through the wells and a
step that doubles outward as the local wavelength grows (``h <= h_max``, set
by the most-closed channel).  That is ~10x fewer steps than a uniform grid at
the same accuracy.  The scattering length is extracted by matching to
asymptotic sin/cos (open), linear (at threshold) and exponential (closed)
forms, imposing decay in closed channels.  When other channels are open
below the entrance (inelastic loss), outgoing-wave conditions give a complex
``a = a_re - i a_im`` with ``a_im >= 0``.  All internal calculation is in
atomic units.
"""

from __future__ import annotations

from typing import List, Optional

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
def _propagate_numba(VS, VT, AS, AT, dvec, seg_i0, seg_h, nch):
    """Renormalized Numerov on a segmented grid; returns (u1, u2) at the last two r.

    Segment ``s`` starts at grid index ``seg_i0[s]`` with step ``seg_h[s]``;
    consecutive segments share their boundary point and each step is exactly
    twice the previous one.  At a boundary ``m`` the ratio matrix is rebuilt
    for the doubled step from ``psi_m psi_{m-2}^-1``, which the old segment
    already holds as ``T_m^-1 R_{m-1} R_{m-2} T_{m-2}``.
    """
    n = VS.shape[0]
    nseg = seg_h.shape[0]
    I = np.eye(nch)
    Rinv = np.zeros((nch, nch))
    R_m1 = np.zeros((nch, nch))
    R_m2 = np.zeros((nch, nch))
    for s in range(nseg):
        h2_12 = seg_h[s] * seg_h[s] / 12.0
        k0 = seg_i0[s]
        k1 = seg_i0[s + 1] if s + 1 < nseg else n - 1
        if s > 0:
            ho = seg_h[s - 1] * seg_h[s - 1] / 12.0
            Wm = VS[k0] * AS + VT[k0] * AT
            Wm2 = VS[k0 - 2] * AS + VT[k0 - 2] * AT
            for i in range(nch):
                Wm[i, i] += dvec[i]
                Wm2[i, i] += dvec[i]
            psi_ratio = np.linalg.solve(I - ho * Wm, R_m1 @ R_m2 @ (I - ho * Wm2))
            Rp = (I - h2_12 * Wm) @ psi_ratio @ np.linalg.inv(I - h2_12 * Wm2)
            Rinv = np.linalg.inv(Rp)
        for k in range(k0, k1):
            Q = VS[k] * AS + VT[k] * AT
            for i in range(nch):
                Q[i, i] += dvec[i]
            U = 12.0 * np.linalg.inv(I - h2_12 * Q) - 10.0 * I
            R = U - Rinv
            Rinv = np.linalg.inv(R)
            R_m2 = R_m1
            R_m1 = R
    # last two points: F_{n-2} = I, F_{n-1} = R_{n-2}
    h2_12 = seg_h[nseg - 1] * seg_h[nseg - 1] / 12.0
    Qa = VS[n - 2] * AS + VT[n - 2] * AT
    Qb = VS[n - 1] * AS + VT[n - 1] * AT
    for i in range(nch):
        Qa[i, i] += dvec[i]
        Qb[i, i] += dvec[i]
    u1 = np.linalg.inv(I - h2_12 * Qa)
    u2 = np.linalg.solve(I - h2_12 * Qb, R_m1)
    return u1, u2


def segmented_grid(r_in: float, r_out: float, h: float, r_fine: float = 30.0,
                   h_max: float = 0.4):
    """Radial grid with step ``h`` out to ``r_fine``, then doubling to ``h_max``.

    Beyond ``r_fine`` the local wavenumber falls as ``r^-3`` (C6 tail), so the
    step doubles every time r grows by ``2^(1/3)``, capped at ``h_max``.
    Returns ``(r, seg_i0, seg_h)``.
    """
    pts = [r_in + h * np.arange(int(round((r_fine - r_in) / h)) + 1)]
    seg_i0, seg_h = [0], [h]
    n_tot = len(pts[0])
    hh, r0 = h, pts[0][-1]
    while r0 < r_out - 1e-9:
        if 2 * hh > h_max:
            # step is capped: extend the current segment to r_out
            nstep = int(np.ceil((r_out - r0) / hh))
            pts.append(r0 + hh * np.arange(1, nstep + 1))
            break
        hh *= 2
        r_next = min(r0 * 2 ** (1 / 3), r_out)
        nstep = max(int(np.ceil((r_next - r0) / hh)), 2)
        seg = r0 + hh * np.arange(1, nstep + 1)
        seg_i0.append(n_tot - 1)
        seg_h.append(hh)
        pts.append(seg)
        n_tot += nstep
        r0 = seg[-1]
    return (np.concatenate(pts), np.array(seg_i0, dtype=np.int64),
            np.array(seg_h, dtype=float))


def inner_wall_shape(r_bohr, r_eq: float):
    """Inner-wall correction shape ``(r - r_eq)^2`` for ``r < r_eq``, else 0 (a0^2).

    The potential becomes ``V(r) + delta * inner_wall_shape(r, r_eq)``, the form
    used by Chapurin et al., PRL 123, 233402 (2019): only the repulsive wall
    moves, the well depth and the long-range tail are untouched, and the
    correction is continuous with continuous slope at ``r_eq``.  ``delta`` is
    in Hartree/a0^2.
    """
    x = np.asarray(r_bohr, dtype=float) - r_eq
    return np.where(x < 0.0, x * x, 0.0)


def _spin_projection_matrices(channels: List[_ch.PairChannel]):
    """Return (P_S, P_T, Gram) matrices between zero-field symmetrized channels."""
    psis = np.array([c.spin_state() for c in channels])   # (nch, 64)
    Psing = _ch.singlet_projector()
    Gram = psis @ psis.T
    P_S = psis @ Psing @ psis.T
    P_T = Gram - P_S
    return P_S, P_T, Gram


class CoupledChannels:
    """Coupled-channels a(B) for K39 ground-state s-wave collisions.

    Parameters
    ----------
    B_max, dB : threshold sweep range/step (Gauss); query only B <= B_max.
    r_in, r_out : radial range (a0).
    h : fine radial step (a0) through the wells (to ``r_fine``).  The
        calibration was done at the default; changing it moves ``a_S`` by up
        to ~0.02 a0 (h=0.004), i.e. high-field poles by ~0.05 G, so re-run
        :func:`kamo.scattering.calibration.calibrate` if you change the grid.
    potentials : ``'tiemann'`` (default) or ``'lj'`` (legacy model potentials).
    delta_S, delta_T : inner-wall corrections (Hartree/a0^2) added to the
        singlet / triplet curves as ``delta * inner_wall_shape(r)``.  ``None``
        uses the calibrated defaults of :mod:`.data.k39_calibration`; pass 0.0
        for the bare Falke 2008 curves.  :meth:`set_scattering_lengths` targets
        ``a_S``/``a_T`` directly.
    C12_S, C12_T : Lennard-Jones walls (``potentials='lj'`` only).
    thresholds : share an existing :class:`K39Thresholds` (skips the sweep).
    """

    def __init__(self, B_max: float = 1000.0, dB: float = 0.05,
                 r_in: float = 4.5, r_out: float = 2500.0, h: float = 0.003,
                 r_fine: float = 30.0, h_max: float = 0.4,
                 potentials: str = 'tiemann',
                 delta_S: Optional[float] = None, delta_T: Optional[float] = None,
                 C12_S=None, C12_T=None, thresholds: Optional[K39Thresholds] = None,
                 atom=None):
        self.th = thresholds if thresholds is not None else K39Thresholds(
            B_max_gauss=B_max, dB_gauss=dB, atom=atom)
        self.mu = _pot.MU_AU
        self.C6 = _pot.C6_AU
        self.r_in, self.r_out, self.h = r_in, r_out, h
        self.r, self._seg_i0, self._seg_h = segmented_grid(r_in, r_out, h, r_fine, h_max)
        self.potentials = potentials
        if potentials == 'tiemann':
            from . import tiemann as _tm
            from .data import k39_calibration as kc
            self._VS0 = _tm.potential_hartree(self.r, _tm.SINGLET)
            self._VT0 = _tm.potential_hartree(self.r, _tm.TRIPLET)
            self._sS = inner_wall_shape(self.r, kc.R_EQ_S_A0)
            self._sT = inner_wall_shape(self.r, kc.R_EQ_T_A0)
            delta_S = kc.DELTA_S if delta_S is None else delta_S
            delta_T = kc.DELTA_T if delta_T is None else delta_T
            self.set_short_range(delta_S, delta_T)
        elif potentials == 'lj':
            from .data import k39_params as kp
            a_S, a_T, _ = kp.singlet_triplet()
            self.C12_S = C12_S if C12_S is not None else _pot.tune_C12(a_S)[0]
            self.C12_T = C12_T if C12_T is not None else _pot.tune_C12(a_T)[0]
            self.VS = _pot.v_lj(self.r, self.C6, self.C12_S)
            self.VT = _pot.v_lj(self.r, self.C6, self.C12_T)
            self.delta_S = self.delta_T = 0.0
        else:
            raise ValueError(f"potentials must be 'tiemann' or 'lj', not {potentials!r}")

    # -- short-range calibration knobs --------------------------------------
    def set_short_range(self, delta_S: float, delta_T: float) -> None:
        """Set the singlet/triplet short-range corrections (Hartree)."""
        if self.potentials != 'tiemann':
            raise ValueError("short-range corrections apply to potentials='tiemann'")
        self.delta_S, self.delta_T = float(delta_S), float(delta_T)
        self.VS = self._VS0 + self.delta_S * self._sS
        self.VT = self._VT0 + self.delta_T * self._sT

    def single_channel_a(self, spin: str) -> float:
        """Zero-energy scattering length (a0) of the bare singlet ('S') or triplet ('T') curve."""
        one, zero = np.ones((1, 1)), np.zeros((1, 1))
        AS, AT = (one, zero) if spin == 'S' else (zero, one)
        u1, u2 = _propagate_numba(self.VS, self.VT, 2 * self.mu * AS, 2 * self.mu * AT,
                                  np.zeros(1), self._seg_i0, self._seg_h, 1)
        r1, r2 = self.r[-2], self.r[-1]
        # u = c r + d  ->  a = -d/c
        c = (u2[0, 0] - u1[0, 0]) / (r2 - r1)
        return float(-(u1[0, 0] - c * r1) / c)

    def singlet_triplet_a(self):
        """``(a_S, a_T)`` in a0 for the current potentials."""
        return self.single_channel_a('S'), self.single_channel_a('T')

    def set_scattering_lengths(self, a_S: Optional[float] = None,
                               a_T: Optional[float] = None, tol: float = 1e-4):
        """Tune ``delta_S``/``delta_T`` so the bare curves give ``a_S``/``a_T`` (a0).

        Searches the pole-free branch containing the current value (secant +
        bisection on the short-range shift).  Returns the new ``(delta_S, delta_T)``.
        """
        for spin, target in (('S', a_S), ('T', a_T)):
            if target is None:
                continue
            self._tune_delta(spin, float(target), tol)
        return self.delta_S, self.delta_T

    def _tune_delta(self, spin, target, tol):
        from scipy.optimize import brentq

        def f(d):
            if spin == 'S':
                self.set_short_range(d, self.delta_T)
            else:
                self.set_short_range(self.delta_S, d)
            return self.single_channel_a(spin) - target

        d0 = self.delta_S if spin == 'S' else self.delta_T
        f0 = f(d0)
        # grow a bracket on the side that moves a toward the target, rejecting
        # brackets that straddle a pole (a jumps through infinity)
        step = 2e-7 if spin == 'S' else 5e-8
        for direction in (1.0, -1.0):
            lo, flo = d0, f0
            for k in range(40):
                hi = d0 + direction * step * (1.3 ** k)
                fhi = f(hi)
                if abs(fhi - flo) > 500.0:          # crossed a pole
                    break
                if np.sign(fhi) != np.sign(flo):
                    d = brentq(f, min(lo, hi), max(lo, hi), xtol=1e-14, rtol=1e-12)
                    f(d)
                    return d
                lo, flo = hi, fhi
        f(d0)
        raise RuntimeError(f"could not bracket a_{spin} = {target} without crossing a pole")

    # -- channel construction -------------------------------------------------
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

    def channels_for(self, entrance_a, entrance_b):
        """Channel list for the entrance's M_F block, entrance first."""
        entrance = _ch.PairChannel(*sorted((tuple(entrance_a), tuple(entrance_b))))
        channels = _ch.enumerate_channels(entrance.M_F)
        ei = next(i for i, c in enumerate(channels)
                  if (c.a, c.b) == (entrance.a, entrance.b))
        return [channels[ei]] + channels[:ei] + channels[ei + 1:]

    # -- scattering length ----------------------------------------------------
    def scattering_length(self, entrance_a, entrance_b, B_gauss: float,
                          E_coll_hartree: float = 0.0) -> complex:
        """Complex s-wave scattering length (a0) of the entrance channel at B.

        ``a = a_re - i a_im``; ``a_im > 0`` only when a channel below the
        entrance is open (inelastic loss).
        """
        if B_gauss > self.th.B_max + 1e-9 or B_gauss < 0:
            raise ValueError(f"B = {B_gauss} G outside the threshold sweep [0, {self.th.B_max}]")
        channels = self.channels_for(entrance_a, entrance_b)
        P_S, P_T = self._spin_matrices_at_B(channels, B_gauss)
        nch = len(channels)
        Ethr = self._thresholds_hartree(channels, B_gauss)
        E = Ethr[0] + E_coll_hartree

        mu = self.mu
        AS = (2.0 * mu) * P_S
        AT = (2.0 * mu) * P_T
        dvec = 2.0 * mu * (Ethr - E)
        u1, u2 = _propagate_numba(self.VS, self.VT, AS, AT, dvec,
                                  self._seg_i0, self._seg_h, nch)
        return self._match(Ethr, E, u1, u2, self.r[-2], self.r[-1])

    def _match(self, Ethr, E, u1, u2, r1, r2) -> complex:
        """Match propagated u at (r1,r2) to asymptotic forms; return complex a."""
        mu = self.mu
        nch = len(Ethr)
        tol = 1e-14
        kind = []
        kk = np.zeros(nch)
        for i in range(nch):
            d = E - Ethr[i]
            if i == 0 and abs(d) <= tol:
                kind.append('thr')
            elif d > tol:
                kind.append('open'); kk[i] = np.sqrt(2 * mu * d)
            elif d < -tol:
                kind.append('closed'); kk[i] = np.sqrt(-2 * mu * d)
            else:
                kind.append('thr')

        def basis(i, r):
            if kind[i] == 'open':
                return np.sin(kk[i] * r), np.cos(kk[i] * r)
            if kind[i] == 'thr':
                return r, 1.0
            return np.exp(-kk[i] * (r - r2)), np.exp(kk[i] * (r - r2))

        # u = J X + N Y  (rows = channels, columns = independent solutions)
        X = np.zeros((nch, nch)); Y = np.zeros((nch, nch))
        for i in range(nch):
            P1, Q1 = basis(i, r1); P2, Q2 = basis(i, r2)
            Mi = np.array([[P1, Q1], [P2, Q2]])
            sol = np.linalg.solve(Mi, np.vstack([u1[i, :], u2[i, :]]))
            X[i, :] = sol[0, :]; Y[i, :] = sol[1, :]

        open_idx = [i for i in range(nch) if kind[i] in ('open', 'thr')]
        closed_idx = [i for i in range(nch) if kind[i] == 'closed']
        if closed_idx:
            from scipy.linalg import null_space
            Vnull = null_space(Y[closed_idx, :])     # no growing closed components
        else:
            Vnull = np.eye(nch)
        Xo = X[open_idx, :] @ Vnull
        Yo = Y[open_idx, :] @ Vnull
        K = Yo @ np.linalg.inv(Xo)                   # open-channel K matrix, entrance first

        # eliminate the other open channels with outgoing waves:
        # K_eff = K_ee + i K_eo (1 - i K_oo)^-1 K_oe
        K_eff = complex(K[0, 0])
        if len(open_idx) > 1:
            Koo = K[1:, 1:]
            K_eff = K[0, 0] + 1j * K[0, 1:] @ np.linalg.solve(
                np.eye(len(open_idx) - 1) - 1j * Koo, K[1:, 0])
        if kind[0] == 'thr':
            return complex(-K_eff)                   # u_e ~ r - a
        return complex(-K_eff / kk[0])               # u_e ~ sin(kr) + tan(delta) cos(kr)
