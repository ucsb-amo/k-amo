"""The driven two-level cycling transition, parameterised from the kamo Hamiltonian.

At high field (the k-team runs ~520 G) K-39 is deep in the Paschen-Back regime, so
the uncoupled states are good eigenstates and the nuclear spin is a spectator.  The
sigma-minus transition

    |4S1/2; mJ = -1/2, mI = -1/2>  ->  |4P3/2; mJ = -3/2, mI = -1/2>

is closed: the excited state can only decay back to mJ = -1/2.  CyclingTransition
pulls every number the dipole-dipole model needs -- transition frequency, dipole
matrix element, linewidth, saturation intensity, and the driven spherical component
-- out of kamo.hamiltonian.AtomicStructure at a given field, and keeps a diagnostic
table of the competing channels so the two-level assumption stays auditable.

At 520.6 G the nearest competing channel is pi at +970 MHz (161 linewidths), so the
closed two-level treatment is excellent.  The ground state is only about 97.5% pure
(residual hyperfine flip-flop admixture of mJ = +1/2, mI = -3/2), which opens a weak
pi channel near -9 MHz -- inside a +/-10 linewidth window -- but at a relative
strength of about 8e-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from kamo import constants as c
from .green_tensor import spherical_unit_vector

# Default states for the k-team high-field cycling transition.
GROUND_STATE = (4, 0, 0.5, -0.5, -0.5)
EXCITED_STATE = (4, 1, 1.5, -1.5, -0.5)
DEFAULT_MANIFOLDS = [(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)]

# 4P3/2 - 4P1/2 splitting, released by a fine-structure-changing collision.
# Computed from ARC at construction; this is only the fallback.
_K39_FINE_STRUCTURE_HZ = 1.7301e12


@dataclass
class CompetingChannel:
    """One dipole-allowed transition out of the driven ground state."""

    polarization: str
    q: int
    m_j: float
    m_i: float
    detuning_Hz: float
    dipole_ea0: float
    relative_strength: float = 0.0


class CyclingTransition:
    """Two-level parameters for a closed sigma transition at a static field.

    Build with :meth:`at_field`.  All frequencies are stored in Hz except
    :attr:`Gamma`, which is an angular rate (rad/s) as used throughout the
    dipole-dipole module.

    Attributes
    ----------
    B_gauss : float
    ground, excited : 5-tuples (n, l, j, m_j, m_i)
    q : int
        Spherical component of the driven transition (-1 for sigma-minus).
    f0_Hz : float
        Transition frequency at this field.
    d_Cm : float
        Two-level dipole matrix element.
    Gamma : float
        Excited-state decay rate in rad/s (1/tau).
    wavelength_m : float
    ground_purity, excited_purity : float
        Overlap of the dressed eigenstate with the uncoupled ket -- a check on the
        Paschen-Back assumption.
    channels : list of CompetingChannel
    fine_structure_Hz : float
        4P3/2 - 4P1/2 splitting, the energy released by an FCC event.
    """

    def __init__(self, B_gauss, ground, excited, q, f0_Hz, d_Cm, Gamma,
                 wavelength_m, ground_purity, excited_purity, channels,
                 fine_structure_Hz):
        self.B_gauss = float(B_gauss)
        self.ground = tuple(ground)
        self.excited = tuple(excited)
        self.q = int(q)
        self.f0_Hz = float(f0_Hz)
        self.d_Cm = float(d_Cm)
        self.Gamma = float(Gamma)
        self.wavelength_m = float(wavelength_m)
        self.ground_purity = float(ground_purity)
        self.excited_purity = float(excited_purity)
        self.channels = list(channels)
        self.fine_structure_Hz = float(fine_structure_Hz)

    # ------------------------------------------------------------------ build
    @classmethod
    def at_field(cls, B_gauss: float = 520.6,
                 ground: tuple = GROUND_STATE,
                 excited: tuple = EXCITED_STATE,
                 q: int = -1,
                 manifolds: Optional[list] = None,
                 model=None) -> "CyclingTransition":
        """Diagonalise H0 + Zeeman at B_gauss and extract the two-level parameters.

        Parameters
        ----------
        B_gauss : float
            Static field in Gauss.
        ground, excited : 5-tuple
            (n, l, j, m_j, m_i) of the driven states.
        q : int
            Spherical component driven (-1 = sigma-minus).
        manifolds : list of (n, l, j), optional
            Basis for the diagonalisation.  Defaults to 4S1/2 + 4P1/2 + 4P3/2.
        model : AtomicStructure, optional
            Reuse an existing structure (avoids re-opening the ARC database).
        """
        from kamo.hamiltonian import AtomicStructure

        if model is None:
            model = AtomicStructure(manifolds or DEFAULT_MANIFOLDS)
        basis = model.basis
        atom = model.builder.atom

        E, V = model.solve(B_gauss=B_gauss)

        def dressed(state):
            b = basis.index_of(*state)
            w = np.abs(V[b, :]) ** 2
            idx = int(np.argmax(w))
            return idx, float(w[idx])

        g_idx, g_pur = dressed(ground)
        e_idx, e_pur = dressed(excited)
        f0 = float(E[e_idx] - E[g_idx])

        n_e, l_e, j_e = excited[0], excited[1], excited[2]

        def dressed_dipole(target_idx, qq):
            """Dipole matrix element (e*a0) between the dressed ground state and
            the dressed excited eigenstate target_idx, for spherical component qq."""
            tot = 0.0 + 0.0j
            for a in basis:
                if a.l != 0:
                    continue
                ca = V[a.index, g_idx]
                if ca == 0.0:
                    continue
                for b in basis:
                    if b.n != n_e or b.l != l_e or abs(b.j - j_e) > 1e-9:
                        continue
                    if abs(a.m_i - b.m_i) > 1e-9:      # nuclear spin is a spectator
                        continue
                    if abs((a.m_j + qq) - b.m_j) > 1e-9:
                        continue
                    cb = V[b.index, target_idx]
                    if cb == 0.0:
                        continue
                    tot += np.conj(ca) * cb * atom.getDipoleMatrixElement(
                        a.n, a.l, a.j, a.m_j, b.n, b.l, b.j, b.m_j, qq)
            return tot

        d_ea0 = abs(dressed_dipole(e_idx, q))
        d_Cm = d_ea0 * c.a0 * c.e

        channels: List[CompetingChannel] = []
        for qq, name in ((-1, "sigma-"), (0, "pi"), (1, "sigma+")):
            for idx in range(len(E)):
                s = basis[int(np.argmax(np.abs(V[:, idx]) ** 2))]
                if s.n != n_e or s.l != l_e or abs(s.j - j_e) > 1e-9:
                    continue
                amp = abs(dressed_dipole(idx, qq))
                if amp < 1e-6:
                    continue
                channels.append(CompetingChannel(
                    polarization=name, q=qq, m_j=s.m_j, m_i=s.m_i,
                    detuning_Hz=float(E[idx] - E[g_idx] - f0),
                    dipole_ea0=float(amp),
                    relative_strength=float((amp / d_ea0) ** 2) if d_ea0 > 0 else np.nan))
        channels.sort(key=lambda ch: -ch.dipole_ea0)

        Gamma = 1.0 / atom.getStateLifetime(n_e, l_e, j_e)
        lam = atom.getTransitionWavelength(ground[0], ground[1], ground[2],
                                           n_e, l_e, j_e)
        try:
            fs = abs(atom.getTransitionFrequency(n_e, l_e, 0.5, n_e, l_e, 1.5))
        except Exception:
            fs = _K39_FINE_STRUCTURE_HZ

        return cls(B_gauss, ground, excited, q, f0, d_Cm, Gamma, abs(lam),
                   g_pur, e_pur, channels, fs)

    # -------------------------------------------------------------- properties
    @property
    def k(self) -> float:
        """Wavenumber 2 pi / lambda (1/m)."""
        return 2.0 * np.pi / self.wavelength_m

    @property
    def d_ea0(self) -> float:
        """Dipole matrix element in units of e*a0."""
        return self.d_Cm / (c.a0 * c.e)

    @property
    def linewidth_Hz(self) -> float:
        """Gamma / 2 pi."""
        return self.Gamma / (2.0 * np.pi)

    @property
    def I_sat(self) -> float:
        """Saturation intensity of this transition, W/m^2.

        Defined so that Omega^2 = Gamma^2 I / (2 I_sat), i.e.
        I_sat = c eps0 hbar^2 Gamma^2 / (4 d^2).
        """
        return c.c * c.epsilon0 * c.hbar ** 2 * self.Gamma ** 2 / (4.0 * self.d_Cm ** 2)

    @property
    def d_hat(self) -> np.ndarray:
        """Complex transition-dipole unit vector for the driven component."""
        return spherical_unit_vector(self.q)

    @property
    def fine_structure_energy_J(self) -> float:
        """Energy released per pair by a fine-structure-changing collision."""
        return c.h * self.fine_structure_Hz

    # ------------------------------------------------------------------ drive
    def drive_projection(self, k_hat=(1.0, 0.0, 0.0), eps=(0.0, 1.0, 0.0)) -> Dict[int, float]:
        """Fraction of the total intensity landing on each spherical component.

        The quantisation axis is z (the bias field).  eps is the (possibly complex)
        polarisation unit vector, which must be transverse to k_hat.

        Returns
        -------
        dict {q: |c_q|^2} for q in (-1, 0, +1), summing to 1.

        Notes
        -----
        For a beam along x (perpendicular to B) polarised along y the result is
        50% sigma-minus, 50% sigma-plus and no pi, so the driven channel sees
        *half* the total intensity.  The same beam polarised along B gives pure pi
        and zero sigma.
        """
        k_hat = np.asarray(k_hat, dtype=float)
        k_hat = k_hat / np.linalg.norm(k_hat)
        eps = np.asarray(eps, dtype=complex)
        norm = np.linalg.norm(eps)
        if norm == 0:
            raise ValueError("eps must be non-zero.")
        eps = eps / norm
        if abs(np.dot(np.conj(eps), k_hat)) > 1e-9:
            raise ValueError("eps must be transverse to k_hat.")
        out = {}
        for qq in (-1, 0, 1):
            e_q = spherical_unit_vector(qq)
            out[qq] = float(abs(np.dot(np.conj(e_q), eps)) ** 2)
        return out

    def driven_intensity(self, intensity, k_hat=(1.0, 0.0, 0.0), eps=(0.0, 1.0, 0.0)):
        """Intensity (W/m^2) actually coupling to the driven channel."""
        frac = self.drive_projection(k_hat, eps)[self.q]
        return np.asarray(intensity, dtype=float) * frac

    def rabi(self, intensity):
        """Single-atom Rabi frequency (rad/s) for an intensity *already projected*
        onto the driven channel.  Omega = d E0 / hbar, E0 = sqrt(2 I / (c eps0))."""
        I = np.asarray(intensity, dtype=float)
        E0 = np.sqrt(2.0 * I / (c.c * c.epsilon0))
        return self.d_Cm * E0 / c.hbar

    def saturation_parameter(self, intensity, detuning_Hz=0.0):
        """s = (I/I_sat) / (1 + (2 Delta/Gamma)^2)."""
        I = np.asarray(intensity, dtype=float)
        delta = 2.0 * np.pi * np.asarray(detuning_Hz, dtype=float)
        return (I / self.I_sat) / (1.0 + (2.0 * delta / self.Gamma) ** 2)

    # ------------------------------------------------------------- diagnostics
    def competing_channels(self, max_detuning_Hz=None) -> List[CompetingChannel]:
        """Dipole-allowed channels out of the driven ground state, strongest first."""
        if max_detuning_Hz is None:
            return list(self.channels)
        return [ch for ch in self.channels if abs(ch.detuning_Hz) <= max_detuning_Hz]

    def summary(self) -> str:
        """Parameter dump, including the competing-channel table."""
        head = "CyclingTransition at B = %.4f G" % self.B_gauss
        lines = [head]
        lines.append("  ground  (n,l,j,mJ,mI) = %s   purity %.6f"
                     % (str(self.ground), self.ground_purity))
        lines.append("  excited (n,l,j,mJ,mI) = %s   purity %.6f"
                     % (str(self.excited), self.excited_purity))
        lines.append("  driven component q    = %+d" % self.q)
        lines.append("  d           = %.4f e*a0  (%.4e C*m)" % (self.d_ea0, self.d_Cm))
        lines.append("  Gamma/2pi   = %.4f MHz" % (self.linewidth_Hz / 1e6))
        lines.append("  lambda      = %.4f nm  (k = %.4e 1/m)"
                     % (self.wavelength_m * 1e9, self.k))
        lines.append("  I_sat       = %.4f mW/cm^2" % (self.I_sat / 10.0))
        lines.append("  4P3/2-4P1/2 = %.4f THz" % (self.fine_structure_Hz / 1e12))
        lines.append("  competing channels:")
        lines.append("    %-8s %-16s %16s %12s %13s"
                     % ("pol", "(mJ, mI)", "detuning (MHz)", "|d| (e*a0)", "|d|^2 ratio"))
        for ch in self.channels:
            lines.append("    %-8s %-16s %16.2f %12.4f %13.3e"
                         % (ch.polarization,
                            "(%+.1f, %+.1f)" % (ch.m_j, ch.m_i),
                            ch.detuning_Hz / 1e6, ch.dipole_ea0, ch.relative_strength))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return ("CyclingTransition(B=%.4f G, d=%.4f e*a0, Gamma/2pi=%.4f MHz, "
                "lambda=%.3f nm)"
                % (self.B_gauss, self.d_ea0, self.linewidth_Hz / 1e6,
                   self.wavelength_m * 1e9))
