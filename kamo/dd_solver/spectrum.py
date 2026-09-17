"""Collective modes and the line shape: what the near field does to the medium.

The solver answers "what is the field for this cloud at this detuning". This
module answers the question behind it: **what does the near field do to the
medium's spectrum**, which is where every near-field number in this package
comes from.

Two views, and they say the same thing.

:func:`collective_modes` diagonalizes the coupled-dipole matrix at a reference
detuning.  Each eigenvalue gives a collective shift and a collective decay rate,
and each eigenvector says how many atoms the mode lives on.  At the operating
density the full kernel spreads the shifts over ``+-26 Gamma`` while the
near-field-free kernel spans only ``+-2.3 Gamma``; the far-shifted modes are the
two-atom ones.  That is the whole mechanism in one picture: the near field does
not add scattering strength, it **inhomogeneously broadens** the medium, and it
does so by making a small number of tightly bound pairs.

:func:`detuning_scan` sweeps the laser and records the line shape.  The
interacting medium's peak extinction is several times lower than the
independent-atom one and the line is correspondingly broader, with the missing
strength pushed into tails far from resonance.

The sum rule that ties them together
------------------------------------
For a uniform detuning the matrix is ``M(delta) = delta I + A``, and

    Int d(delta) Tr Im (delta I + A)^-1  =  -pi Sum_mu sign(Im lambda_mu) ,

which equals ``-pi N`` for any passive medium, whatever the interactions do to
the individual modes.  Interactions move oscillator strength around; they cannot
create or destroy it.  :meth:`ModeSpectrum.sum_rule` returns that sum, and it is
sharp in exactly the way this package needed: a kernel with gain modes (as the
pre-2026-09-17 ``'far'`` variant had) returns ``N - 2 n_gain`` instead of ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .cloud import Configuration
from .solver import build_matrix, solve
from .system import IncidentField, OperatingPoint, default_incident

__all__ = ["ModeSpectrum", "collective_modes", "detuning_scan", "DetuningScan"]


@dataclass
class ModeSpectrum:
    """Eigenmodes of the coupled-dipole matrix at one reference detuning."""

    eigenvalues: np.ndarray        #: (N,) complex, of ``M - delta I``
    shift: np.ndarray              #: (N,) collective shift, Gamma units
    gamma: np.ndarray              #: (N,) collective decay rate, Gamma units
    participation: np.ndarray      #: (N,) atoms the mode lives on (inverse participation ratio)
    overlap: Optional[np.ndarray]  #: (N,) |<mode|Omega>|^2 / |Omega|^2, if a drive was given
    variant: str
    N: int

    # ------------------------------------------------------------- summaries
    def n_subradiant(self, threshold: float = 0.1) -> int:
        """Modes decaying slower than ``threshold * Gamma``."""
        return int(np.sum(self.gamma < threshold))

    def n_superradiant(self, threshold: float = 2.0) -> int:
        return int(np.sum(self.gamma > threshold))

    def n_pair_modes(self, max_atoms: float = 3.0) -> int:
        """Modes localised on fewer than ``max_atoms`` atoms -- the near-field pairs."""
        return int(np.sum(self.participation < max_atoms))

    def n_gain(self) -> int:
        """Modes with a NEGATIVE decay rate.  Must be zero for a passive medium."""
        return int(np.sum(self.gamma < 0))

    def sum_rule(self) -> float:
        """``Sum_mu sign(gamma_mu)``, which must equal ``N``.

        This is ``-1/pi`` times ``Int d(delta) Tr Im (delta I + A)^-1`` (see the
        module docstring): the total oscillator strength, conserved whatever the
        interactions do.  ``N - 2 n_gain`` for a kernel that is not passive.
        """
        return float(np.sum(np.sign(self.gamma)))

    def shift_spread(self, quantile: float = 0.99) -> float:
        """Central ``quantile`` range of the collective shifts (Gamma units).

        The near field's signature: 26 Gamma for the full kernel at the
        operating density against 2 Gamma with the static term removed.
        """
        lo, hi = np.quantile(self.shift, [(1 - quantile) / 2, (1 + quantile) / 2])
        return float(hi - lo)

    def summary(self) -> str:
        s = [f"ModeSpectrum({self.variant!r}, N = {self.N})",
             f"  collective shift: 99% range {self.shift_spread():.2f} Gamma, "
             f"full range [{self.shift.min():+.1f}, {self.shift.max():+.1f}]",
             f"  decay: {self.n_subradiant()} modes below 0.1 Gamma, "
             f"{self.n_superradiant()} above 2 Gamma, min {self.gamma.min():.2e}",
             f"  localisation: {self.n_pair_modes()} modes on fewer than 3 atoms",
             f"  sum rule sum sign(gamma) = {self.sum_rule():.0f} (must be {self.N}); "
             f"gain modes {self.n_gain()}"]
        return "\n".join(s)


def collective_modes(config: Configuration, op: OperatingPoint, variant: str = "full",
                     delta: float = 0.0, incident: Optional[IncidentField] = None,
                     drive: bool = True) -> ModeSpectrum:
    """Diagonalize the coupled-dipole matrix for one configuration.

    Parameters
    ----------
    config, op, variant
        As for :func:`kamo.dd_solver.solver.solve`.
    delta : float
        Reference detuning for every atom.  The eigenvalues are returned
        relative to it, so the shifts do not depend on this choice; it matters
        only for a MIXED-spin configuration, where the two species' detunings
        enter the matrix and ``delta`` is ignored.
    incident, drive
        With ``drive=True`` the overlap of each mode with the actual drive is
        computed, which says which modes the probe can even see.

    Notes
    -----
    ``M`` is complex SYMMETRIC, not Hermitian, so its eigenvectors are not
    orthogonal and the participation ratio below is the usual heuristic rather
    than a probability.  The eigenVALUES are exact.
    """
    N = config.N
    spins = np.asarray(config.spins)
    uniform = bool(np.all(spins == spins[0])) if N else True
    if uniform:
        det = np.full(N, float(delta))
        ref = float(delta)
    else:
        det = op.detunings(spins)
        ref = 0.0
    M, _, _ = build_matrix(config.positions, det, op, variant,
                           strengths=op.strengths(spins))
    lam, V = np.linalg.eig(M - ref * np.eye(N))
    shift = -lam.real
    gamma = 2.0 * lam.imag
    w = np.abs(V) ** 2
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-300)
    participation = 1.0 / np.sum(w ** 2, axis=0)
    ov = None
    if drive:
        inc = default_incident(op) if incident is None else incident
        Om = inc.drive(config.positions, op.e_hat)
        # left eigenvectors of a complex symmetric matrix are the transposes of
        # the right ones, so the modal amplitude is (V^T Omega) / (V^T V)
        num = V.T @ Om
        den = np.einsum("ij,ij->j", V, V)
        amp = num / np.where(np.abs(den) > 0, den, 1.0)
        ov = np.abs(amp) ** 2 / max(float(np.sum(np.abs(Om) ** 2)), 1e-300)
    return ModeSpectrum(lam, shift, gamma, participation, ov, variant, N)


@dataclass
class DetuningScan:
    """Line shape of one configuration against the laser detuning."""

    deltas: np.ndarray
    extinction: dict            #: variant -> (n_delta,) extinction
    excitation: dict            #: variant -> (n_delta,) sum |beta|^2
    radiated: dict              #: variant -> (n_delta,) beta^dag Gamma beta
    N: int

    def peak(self, variant: str) -> tuple:
        """``(delta_peak, height)`` of the extinction."""
        e = self.extinction[variant]
        i = int(np.argmax(e))
        return float(self.deltas[i]), float(e[i])

    def integral(self, variant: str) -> float:
        """``Int d(delta) extinction`` over the scanned window.

        The independent-atom value is ``pi N / 2 * |Omega|^2 / |E0|^2`` for a
        window wide enough to contain the line.  Interactions conserve this
        (the sum rule) but push strength into far tails, so a FINITE window sees
        the interacting medium lose a few per cent -- which is itself the
        measurement of how far the near field throws the strength.
        """
        return float(np.trapezoid(self.extinction[variant], self.deltas))

    def summary(self) -> str:
        lines = [f"DetuningScan  N = {self.N}  "
                 f"delta in [{self.deltas.min():+.1f}, {self.deltas.max():+.1f}]"]
        for v in self.extinction:
            d, h = self.peak(v)
            lines.append(f"  {v:>12s}: peak {h:8.2f} at delta {d:+6.2f}   "
                         f"integral over the window {self.integral(v):9.2f}")
        return "\n".join(lines)


def detuning_scan(config: Configuration, op: OperatingPoint, deltas,
                  variants: Sequence[str] = ("full", "nonear", "independent"),
                  incident: Optional[IncidentField] = None) -> DetuningScan:
    """Sweep the laser across the line for a FIXED configuration.

    Every atom is put at the same detuning, so this is a single-species cloud;
    the point is the collective line shape, not the two-species physics.
    """
    inc = default_incident(op) if incident is None else incident
    deltas = np.asarray(deltas, dtype=float)
    ext = {v: np.empty(deltas.size) for v in variants}
    exc = {v: np.empty(deltas.size) for v in variants}
    rad = {v: np.empty(deltas.size) for v in variants}
    spins = np.full(config.N, 1, dtype=np.int8)
    cfg = Configuration(config.positions, spins, config.theta)
    for i, d in enumerate(deltas):
        o = OperatingPoint(op.linewidth_Hz, op.wavelength, delta_up=float(d), delta_dn=float(d),
                           B_gauss=op.B_gauss, q=op.q, strength_up=op.strength_up,
                           strength_dn=op.strength_dn)
        for v in variants:
            r = solve(cfg, o, v, incident=inc, keep_matrices=(v != "independent"),
                      checks=False, warn=False)
            ext[v][i] = r.extinction
            exc[v][i] = r.excitation
            rad[v][i] = r.radiated_power
    return DetuningScan(deltas, ext, exc, rad, config.N)
