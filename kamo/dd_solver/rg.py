"""Strong-disorder renormalization group for the near-field resonance shifts.

Following Grava, He, Wu and Chang, New J. Phys. 24, 013031 (2022) and Andreoli
et al., PRX 11, 011026 (2021).  Work in the near-field Hamiltonian alone,
``H_ij = (3/4) B_ij / (k r_ij)^3``.  Repeatedly find the active pair maximizing

    K = |H_ij| / (|d_omega| + 1),      d_omega = (omega_i - omega_j) / 2,

diagonalize its 2x2 block, ``omega_pm = <omega> +- sqrt(d_omega^2 + H^2)``, and
deactivate the pair.  Atoms may be renormalized again with new partners.  The
loop ends when no active pair is left.

``omega`` is initialised with the bare transition frequencies RELATIVE TO THE
LASER (``omega_j = -delta_j``), so the two spin species are handled natively:
an unlike-spin pair starts ``18.3 Gamma`` apart and its coupled levels sit at
``midpoint +- sqrt(Delta^2/4 + H^2)``, always FARTHER from the probe than the
bare lines.  The midpoint probe buys that protection for free (test T7).

Implementation
--------------
Neighbour list on ``|H| >= cutoff`` (default ``1e-2 Gamma``; test T11 lowers
it 10x and checks the result is stable) and a lazy max-heap keyed on an upper
bound of ``K``: initially ``|H|`` (since ``K <= |H|``), and on pop the true
``K`` is recomputed and the pair re-pushed if it is smaller.  ``O(P log P)`` for
``P`` pairs instead of ``O(P N)``.  This is exact provided ``K`` never
INCREASES for a pair, which holds when ``|d_omega|`` grows monotonically;
occasional violations only permute weakly interacting pairs.

Naive versus brightness-tracked
-------------------------------
The default ("naive", Grava's principle (c)) gives every renormalized atom unit
oscillator strength and the bare decay ``Gamma_0``.  It reproduces the full
solve to ~10% for ``theta <= pi/2`` (the build specification's measurement).

``tracked=True`` carries the drive amplitude and the decay rate through each
pair rotation, restoring the bright/dark ``2 Gamma / 0`` structure.  It looks
like an obvious improvement and is **empirically rejected**: it overshoots the
full solve by 30-50% at every CSS angle tested.  It is kept behind the flag so
that result stays reproducible, with its sum rules ``<d^2> = <gamma> = 1``
tested (T10).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kernel import near_field_pairs

DEFAULT_CUTOFF = 1e-2      #: |H^near| neighbour-list cutoff, Gamma units (T11 checks it)


@dataclass
class RGResult:
    """Renormalized resonances and the bookkeeping that produced them."""

    omega: np.ndarray              #: renormalized resonance, laser frame, Gamma units
    omega0: np.ndarray             #: bare (``-delta_j``)
    n_pairs: int                   #: pairs in the neighbour list
    n_steps: int                   #: pair decimations performed
    n_renormalized: np.ndarray     #: how many times each atom was rotated
    cutoff: float
    tracked: bool
    drive: Optional[np.ndarray] = None     #: tracked variant: rotated drive amplitudes
    gamma: Optional[np.ndarray] = None     #: tracked variant: rotated decay rates

    @property
    def shifts(self) -> np.ndarray:
        """``omega - omega0``."""
        return self.omega - self.omega0

    @property
    def detunings(self) -> np.ndarray:
        """Effective laser detuning of each renormalized atom, ``-omega``."""
        return -self.omega

    def tail_fraction(self, W: float) -> float:
        """Fraction of atoms with ``|omega - omega0| > W``."""
        return float(np.mean(np.abs(self.shifts) > W))


def renormalize(positions, omega0, k: float, e_hat, cutoff: float = DEFAULT_CUTOFF,
                tracked: bool = False, drive=None) -> RGResult:
    """Run the strong-disorder RG.

    Parameters
    ----------
    positions : (N, 3)
    omega0 : (N,)
        Bare resonances relative to the laser, Gamma units (``-delta_j``).
    k, e_hat
        Wavenumber and driven dipole unit vector (sets the angular factor B).
    cutoff : float
        Neighbour-list threshold on ``|H^near|``.
    tracked : bool
        Brightness-tracked variant (rejected; see module docstring).
    drive : (N,) complex, optional
        Bare drives ``Omega_j`` for the tracked variant (default all ones).
    """
    pos = np.asarray(positions, dtype=float)
    N = pos.shape[0]
    omega = np.array(omega0, dtype=float).copy()
    i_arr, j_arr, H_arr, G_arr, _ = near_field_pairs(pos, k, e_hat, cutoff)
    P = H_arr.size
    n_ren = np.zeros(N, dtype=int)
    d = None
    gam = None
    if tracked:
        d = (np.ones(N, dtype=complex) if drive is None
             else np.array(drive, dtype=complex).copy())
        gam = np.ones(N)

    active = np.ones(P, dtype=bool)
    heap = [(-abs(H_arr[p]), p) for p in range(P)]
    heapq.heapify(heap)
    steps = 0
    while heap:
        neg_bound, p = heapq.heappop(heap)
        if not active[p]:
            continue
        i, j, H = i_arr[p], j_arr[p], H_arr[p]
        dw = 0.5 * (omega[i] - omega[j])
        K = abs(H) / (abs(dw) + 1.0)
        if K < -neg_bound * (1.0 - 1e-12):
            heapq.heappush(heap, (-K, p))          # stale bound: re-key and retry
            continue
        mean = 0.5 * (omega[i] + omega[j])
        s = np.sqrt(dw * dw + H * H)
        hi, lo = (i, j) if omega[i] >= omega[j] else (j, i)
        if tracked:
            # 2x2 block [[w_i, H], [H, w_j]]: eigenvector of w_+ is (cos, sin)
            # with tan(2 phi) = 2H / (w_i - w_j), taken in the (hi, lo) ordering.
            dw_hl = 0.5 * (omega[hi] - omega[lo])
            phi = 0.5 * np.arctan2(H, dw_hl) if (dw_hl != 0 or H != 0) else 0.0
            c, sn = np.cos(phi), np.sin(phi)
            d_hi, d_lo = d[hi], d[lo]
            d[hi] = c * d_hi + sn * d_lo
            d[lo] = -sn * d_hi + c * d_lo
            g_hi, g_lo, Gij = gam[hi], gam[lo], G_arr[p]
            gam[hi] = c * c * g_hi + sn * sn * g_lo + 2 * c * sn * Gij
            gam[lo] = sn * sn * g_hi + c * c * g_lo - 2 * c * sn * Gij
        omega[hi] = mean + s
        omega[lo] = mean - s
        n_ren[i] += 1
        n_ren[j] += 1
        active[p] = False
        steps += 1
    return RGResult(omega, np.asarray(omega0, dtype=float), int(P), steps, n_ren,
                    float(cutoff), bool(tracked), d, gam)


def renormalize_configuration(config, op, cutoff: float = DEFAULT_CUTOFF,
                              tracked: bool = False, incident=None) -> RGResult:
    """:func:`renormalize` for a :class:`Configuration` at an :class:`OperatingPoint`."""
    omega0 = -op.detunings(config.spins)
    drive = None
    if tracked:
        from .system import default_incident
        inc = default_incident(op) if incident is None else incident
        drive = inc.drive(config.positions, op.e_hat)
    return renormalize(config.positions, omega0, op.k, op.e_hat, cutoff, tracked, drive)


def pair_eigenvalues(omega_i: float, omega_j: float, H: float):
    """Exact 2x2 eigenvalues ``<omega> +- sqrt(d_omega^2 + H^2)`` (for tests)."""
    m = 0.5 * (omega_i + omega_j)
    s = np.sqrt(0.25 * (omega_i - omega_j) ** 2 + H * H)
    return m + s, m - s
