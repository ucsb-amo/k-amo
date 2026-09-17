"""The 3N x 3N vector coupled-dipole solve, and why it is only needed for extra channels.

For a strictly two-level sigma- atom the polarizability is rank one,
``alpha e_hat conj(e_hat)``, so the induced dipole is ALWAYS along ``e_hat`` and
the 3N problem reduces exactly to the N scalar one of :mod:`kamo.dd_solver.solver`
(test T8 checks this to 1e-14).  The scalar solver is therefore not an
approximation.

The vector solve earns its cost only when the OTHER transitions from the same
ground state are included.  At 520.6 G the 4P3/2 branches are 161 Gamma apart per
unit ``m_J``, so from ``m_J = -1/2`` the pi line sits ~161 Gamma and the sigma+ line
~322 Gamma beyond the driven sigma- line; with the dressed strengths
``1 : 0.668 : 0.334`` the amplitude ratios are ``alpha_pi / alpha_- ~ 0.04`` and
``alpha_+ / alpha_- ~ 0.01``.  Small -- but the y-polarized probe drives only
sigma+-, whereas a NEIGHBOUR's near field has a large z component that drives pi,
and that channel lives exclusively on close pairs, exactly the population that
dominates the excess.

Polarizability tensor per atom, in the scalar units of :mod:`.system`:

    alpha_j = sum_q A_q |e_q><e_q|,    A_q = -(1/2) s_q / (delta_q + i/2)

with ``e_{-1} = (x - iy)/sqrt2``, ``e_0 = z``, ``e_{+1} = -(x + iy)/sqrt2`` and
``s_q`` the relative strengths.  The system is written in the DDA form that
tolerates a singular ``alpha``:

    p_j - alpha_j . sum_{l != j} Gt(r_j - r_l) . p_l = alpha_j . E_inc(r_j).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg as sla

from .cloud import Configuration
from .kernel import dyadic_green_tensor, spherical_unit_vector
from .solver import solve as scalar_solve
from .system import IncidentField, OperatingPoint, default_incident

E_Q = {q: spherical_unit_vector(q) for q in (-1, 0, 1)}


def polarizability_tensors(config: Configuration, op: OperatingPoint,
                           channels: str = "all") -> np.ndarray:
    """``alpha_j`` (N, 3, 3), dimensionless.

    ``channels='driven'`` keeps only the sigma- line (rank 1, for T8);
    ``'all'`` adds every channel listed on the operating point.
    """
    N = config.N
    alpha = np.zeros((N, 3, 3), dtype=complex)
    e = E_Q[op.q]
    proj_driven = np.outer(e, np.conj(e))
    for spin in (1, -1):
        sel = config.spins == spin
        if not np.any(sel):
            continue
        d0 = op.delta_up if spin > 0 else op.delta_dn
        a = -0.5 / (d0 + 0.5j) * proj_driven
        if channels == "all":
            for ch in op.channels(spin):
                eq = E_Q[ch.q]
                a = a + (-0.5 * ch.strength / (ch.detuning + 0.5j)) * np.outer(eq, np.conj(eq))
        alpha[sel] = a
    return alpha


def build_vector_matrix(positions, alpha, k: float, block: int = 128) -> np.ndarray:
    """``M`` (3N, 3N): identity minus ``alpha_j Gt_jl`` off the block diagonal."""
    pos = np.asarray(positions, dtype=float)
    N = pos.shape[0]
    M = np.zeros((3 * N, 3 * N), dtype=complex)
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        d = pos[i0:i1, None, :] - pos[None, :, :]
        rows = np.arange(i0, i1)
        d[rows - i0, rows, :] = 1.0                    # dummy: avoid r = 0 in the dyadic
        Gt = dyadic_green_tensor(k, d)                 # (b, N, 3, 3)
        Gt[rows - i0, rows] = 0.0
        blk = -np.einsum("bij,bljk->blik", alpha[i0:i1], Gt)   # (b, N, 3, 3)
        M[3 * i0:3 * i1] = blk.transpose(0, 2, 1, 3).reshape(3 * (i1 - i0), 3 * N)
    idx = np.arange(3 * N)
    M[idx, idx] += 1.0
    return M


@dataclass
class VectorResult:
    p: np.ndarray                    #: induced dipoles (N, 3), scalar-solver units
    config: Configuration
    op: OperatingPoint
    incident: IncidentField
    alpha: np.ndarray
    channels: str
    timings: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return int(self.p.shape[0])

    def amplitude(self, q: int) -> np.ndarray:
        """``conj(e_q) . p_j`` -- the amplitude on spherical component ``q``."""
        return self.p @ np.conj(E_Q[q])

    @property
    def beta(self) -> np.ndarray:
        """The sigma- amplitude, comparable with the scalar solver's ``beta``."""
        return self.amplitude(self.op.q)

    def out_of_plane(self) -> np.ndarray:
        """``|p_j - e_hat (conj(e_hat) . p_j)|``: dipole amplitude off the driven axis."""
        e = E_Q[self.op.q]
        return np.linalg.norm(self.p - np.outer(self.beta, e), axis=1)

    def channel_power(self) -> dict:
        """``sum_j |conj(e_q) . p_j|^2`` per channel and the fraction out of the e_hat plane."""
        pw = {q: float(np.sum(np.abs(self.amplitude(q)) ** 2)) for q in (-1, 0, 1)}
        tot = float(np.sum(np.abs(self.p) ** 2))
        pw["total"] = tot
        pw["fraction_off_driven"] = 1.0 - pw[self.op.q] / tot
        return pw

    def far_field_amplitude(self, nhat) -> np.ndarray:
        """``F(n) = sum_j e^{-ik n.r_j} (1 - n n) . p_j`` (M, 3)."""
        n = np.asarray(nhat, dtype=float).reshape(-1, 3)
        phase = np.exp(-1j * self.op.k * (n @ self.config.positions.T))     # (M, N)
        S = phase @ self.p                                                   # (M, 3)
        return S - n * np.sum(n * S, axis=1)[:, None]

    def forward_amplitude(self) -> complex:
        """Scalar-equivalent forward amplitude: ``i sqrt2 F_y(khat)``, which equals
        ``sum_j beta_j e^{-ikx_j}`` when only the sigma- channel is present."""
        kh = self.incident.khat
        F = self.far_field_amplitude(kh[None, :])[0]
        pol = self.incident.polarization
        return complex(1j * np.sqrt(2.0) * (F @ np.conj(pol)))

    def single_atom_forward_amplitude(self) -> float:
        """``|beta|`` of one independent atom -- the 'atom' unit for forward amplitudes."""
        d = np.where(self.config.spins > 0, self.op.delta_up, self.op.delta_dn)
        Om = self.incident.drive(self.config.positions, self.op.e_hat)
        return float(np.mean(np.abs(-0.5 * Om / (d + 0.5j))))


def solve_vector(config: Configuration, op: OperatingPoint, channels: str = "all",
                 incident: Optional[IncidentField] = None, block: int = 128) -> VectorResult:
    """Solve the 3N x 3N system (dense LU, complex128)."""
    inc = default_incident(op) if incident is None else incident
    t0 = time.perf_counter()
    alpha = polarizability_tensors(config, op, channels)
    M = build_vector_matrix(config.positions, alpha, op.k, block)
    t_build = time.perf_counter() - t0
    rhs = np.einsum("jik,jk->ji", alpha, inc.field(config.positions)).ravel()
    t0 = time.perf_counter()
    lu, piv = sla.lu_factor(M, overwrite_a=True, check_finite=False)
    p = sla.lu_solve((lu, piv), rhs, check_finite=False).reshape(-1, 3)
    return VectorResult(p, config, op, inc, alpha, channels,
                        dict(build=t_build, solve=time.perf_counter() - t0))


def reduction_check(config: Configuration, op: OperatingPoint, incident=None) -> dict:
    """T8: the rank-1 vector solve against the scalar one.

    Returns ``max |beta_vec - beta_scalar| / max |beta|`` and the largest
    out-of-e_hat amplitude relative to ``max |beta|``; both must be ~1e-14.
    """
    vr = solve_vector(config, op, "driven", incident)
    sr = scalar_solve(config, op, "full", incident, keep_matrices=False)
    scale = float(np.max(np.abs(sr.beta)))
    return dict(beta_difference=float(np.max(np.abs(vr.beta - sr.beta)) / scale),
                out_of_plane=float(np.max(vr.out_of_plane()) / scale))


def channel_effect(config: Configuration, op: OperatingPoint, incident=None) -> dict:
    """What the pi and sigma+ channels change, in atom units (one configuration).

    Two distinct effects, reported separately:

    ``sigma_minus_shift_atoms``
        ``|sum_j (beta_j^all - beta_j^driven) e^{-ikx_j}|`` over the forward
        amplitude of one independent atom: the FEEDBACK of the pi and sigma+
        channels on the sigma- amplitudes.  The pi channel is driven only by the
        z component of a neighbour's near field, so this lives on close pairs and
        has a large configuration variance.
    ``sigma_plus_background_atoms``
        The direct, coherent drive of the far-detuned sigma+ line by the
        y-polarized probe (``|e_+ . y|^2 = 1/2``) on the y-polarized forward
        amplitude, i.e. the total forward change minus the sigma- feedback.  It
        is spin-independent and proportional to N (about 1% of the sigma-
        amplitude per atom at 520.6 G): what a single-species scalar propagation
        leaves out, and what a susceptibility term would capture.
    ``fraction_off_driven``
        Share of dipole power outside the ``e_hat`` plane.
    """
    va = solve_vector(config, op, "all", incident)
    vd = solve_vector(config, op, "driven", incident)
    unit = va.single_atom_forward_amplitude()
    kh = va.incident.khat
    phase = np.exp(-1j * op.k * (config.positions @ kh))
    d_minus = np.sum((va.beta - vd.beta) * phase)
    total = va.forward_amplitude() - vd.forward_amplitude()
    # the sigma- feedback's own contribution to the y forward amplitude
    d_minus_y = 1j * np.sqrt(2.0) * d_minus * (E_Q[op.q] @ np.conj(va.incident.polarization))
    return dict(sigma_minus_shift_atoms=abs(d_minus) / unit,
                sigma_minus_shift_relative=abs(d_minus) / abs(np.sum(vd.beta * phase)),
                total_forward_shift_atoms=abs(total) / unit,
                sigma_plus_background_atoms=abs(total - d_minus_y) / unit,
                excitation_ratio=float(np.sum(np.abs(va.beta) ** 2) / np.sum(np.abs(vd.beta) ** 2)),
                fraction_off_driven=va.channel_power()["fraction_off_driven"],
                timings=va.timings)
