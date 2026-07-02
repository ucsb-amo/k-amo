"""Hamiltonian construction for multi-manifold K39 structure calculations.

All operators are returned as dense Hermitian ``numpy`` arrays in the uncoupled
|n l j; m_j m_i> basis (see :mod:`kamo.hamiltonian.basis`).  **Energies are in
Hz** (i.e. every energy has been divided by Planck's constant ``h``).

Terms
-----
* ``h0``                  : fine structure + hyperfine ``A * (I.J)`` (optional
                            electric-quadrupole ``B`` term).
* ``zeeman_operator``     : paramagnetic Zeeman, returned *per Gauss*.
* ``diamagnetic_operator``: diamagnetic term, returned *per Gauss^2*.
* ``laser_rwa_operator``  : rotating-wave dipole coupling to a Gaussian beam.
* ``laser_stark_operator``: effective AC-Stark shift (scalar + tensor), *per W/m^2*.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from kamo import constants as c
from .basis import Basis

# ---------------------------------------------------------------------------
# unit conversions
# ---------------------------------------------------------------------------
_EV_TO_HZ = c.e / c.h                       # eV -> Hz
_HARTREE_EV = 27.211386245988               # Hartree -> eV
_GAUSS_TO_TESLA = 1.0e-4

# K39 hyperfine electric-quadrupole B constants (Hz).  Only a few are known;
# extend as needed.  Missing entries default to 0 (pure magnetic-dipole).
_HYPERFINE_B_HZ: Dict[Tuple[int, int, float], float] = {
    (4, 1, 1.5): 2.786e6,   # 4P_3/2
}

# polarization helpers: spherical amplitudes {q: amplitude}
_POLARIZATIONS = {
    "pi": {0: 1.0},
    "sigma+": {+1: 1.0},
    "sigma-": {-1: 1.0},
}


def _clebsch(j1, m1, j2, m2, j3, m3) -> float:
    from sympy.physics.wigner import clebsch_gordan
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))


@lru_cache(maxsize=None)
def _wig3j(j1, j2, j3, m1, m2, m3) -> float:
    from sympy.physics.wigner import wigner_3j
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


class HamiltonianBuilder:
    """Builds Hamiltonian-term matrices for a given :class:`Basis`.

    Parameters
    ----------
    basis : Basis
    atom : kamo.Potassium39, optional
        Reuse an existing ARC atom object (avoids re-opening the ARC database).
    energy_reference_nlj : (n, l, j), optional
        Fine-structure energies are reported relative to this manifold so that
        matrix entries stay small.  Defaults to the first manifold in the basis.
    """

    def __init__(self, basis: Basis, atom=None, energy_reference_nlj=None):
        self.basis = basis
        if atom is None:
            from kamo import Potassium39
            atom = Potassium39()
        self.atom = atom
        self.I = basis.manifolds[0].i_nuclear

        if energy_reference_nlj is None:
            m0 = basis.manifolds[0]
            energy_reference_nlj = (m0.n, m0.l, m0.j)
        self.energy_reference_nlj = energy_reference_nlj
        n, l, j = energy_reference_nlj
        self._e_ref_hz = self.atom.getEnergy(n, l, j) * _EV_TO_HZ

    # ------------------------------------------------------------------ H0
    def h0(self, include_quadrupole: bool = True) -> np.ndarray:
        """Field-free Hamiltonian (Hz): fine structure + hyperfine ``A (I.J)``.

        Parameters
        ----------
        include_quadrupole : bool
            Add the electric-quadrupole term for manifolds with a known B
            constant (currently 4P_3/2).
        """
        dim = self.basis.dim
        H = np.zeros((dim, dim), dtype=float)

        for man, sl in self.basis.manifold_slices():
            e_fine = self.atom.getEnergy(man.n, man.l, man.j) * _EV_TO_HZ - self._e_ref_hz
            A_hz = c.get_hyperfine_constant(man.l, man.j) / c.h  # A already carries h
            IJ = self._ij_operator(man.j, self.I)
            block = e_fine * np.eye(man.dim) + A_hz * IJ

            if include_quadrupole:
                B_hz = _HYPERFINE_B_HZ.get((man.n, man.l, man.j), 0.0)
                if B_hz != 0.0 and man.j > 0.5:
                    jj = man.j * (man.j + 1)
                    ii = self.I * (self.I + 1)
                    denom = 2 * self.I * (2 * self.I - 1) * man.j * (2 * man.j - 1)
                    quad = (3 * (IJ @ IJ) + 1.5 * IJ - ii * jj * np.eye(man.dim))
                    block = block + B_hz * quad / denom

            H[sl, sl] = block
        return 0.5 * (H + H.T)

    def _ij_operator(self, j: float, I: float) -> np.ndarray:
        """(I.J) operator in the |m_j, m_i> basis of a single manifold."""
        mjs = [(-j + k) for k in range(int(round(2 * j)) + 1)]
        mis = [(-I + k) for k in range(int(round(2 * I)) + 1)]
        states = [(mj, mi) for mj in mjs for mi in mis]
        d = len(states)
        M = np.zeros((d, d), dtype=float)

        def jp(m):  # J+ coefficient
            return np.sqrt(j * (j + 1) - m * (m + 1))

        def jm(m):  # J- coefficient
            return np.sqrt(j * (j + 1) - m * (m - 1))

        for a, (mj, mi) in enumerate(states):
            for b, (mjp, mip) in enumerate(states):
                if abs(mj - mjp) < 1e-9 and abs(mi - mip) < 1e-9:
                    M[a, b] += mj * mi
                # J+ I- : mj->mj+1, mi->mi-1
                if abs(mjp - (mj + 1)) < 1e-9 and abs(mip - (mi - 1)) < 1e-9:
                    M[a, b] += 0.5 * jp(mj) * np.sqrt(I * (I + 1) - mi * (mi - 1))
                # J- I+ : mj->mj-1, mi->mi+1
                if abs(mjp - (mj - 1)) < 1e-9 and abs(mip - (mi + 1)) < 1e-9:
                    M[a, b] += 0.5 * jm(mj) * np.sqrt(I * (I + 1) - mi * (mi + 1))
        return M

    # -------------------------------------------------------------- Zeeman
    def zeeman_operator(self) -> np.ndarray:
        """Paramagnetic Zeeman operator *per Gauss* (Hz/G).

        ``H_zeeman(B_gauss) = zeeman_operator() * B_gauss``.

        H = mu_B B (g_J m_j + g_I m_i)/h, diagonal in the uncoupled basis.
        """
        dim = self.basis.dim
        diag = np.zeros(dim, dtype=float)
        for s in self.basis:
            g_j = c.get_total_electronic_g_factor(s.l, s.j)
            val = c.mu_b * (g_j * s.m_j + c.g_I * s.m_i) / c.h  # Hz per Tesla
            diag[s.index] = val * _GAUSS_TO_TESLA               # Hz per Gauss
        return np.diag(diag)

    # ---------------------------------------------------------- diamagnetic
    def diamagnetic_operator(self) -> np.ndarray:
        """Diamagnetic operator *per Gauss^2* (Hz/G^2).

        ``H_dia(B_gauss) = diamagnetic_operator() * B_gauss**2``.

        H_dia = (e^2 B^2 / 8 m_e) (x^2 + y^2), with x^2+y^2 = r^2 sin^2(theta).
        Diagonal in (m_j, m_i); couples manifolds with l' = l or l +/- 2.
        Radial integrals <r^2> come from ARC numerical wavefunctions; the
        angular part uses analytic 3j symbols with a Clebsch-Gordan expansion
        into |m_l, m_s>.
        """
        dim = self.basis.dim
        M = np.zeros((dim, dim), dtype=float)

        prefactor = (c.e ** 2 / (8 * c.m_e)) * c.a0 ** 2 / c.h  # Hz per Tesla^2 (times matrix in a0^2)
        prefactor *= _GAUSS_TO_TESLA ** 2                       # -> Hz per Gauss^2

        for a in self.basis:
            for b in self.basis:
                if b.index < a.index:
                    continue
                if abs(a.m_j - b.m_j) > 1e-9 or abs(a.m_i - b.m_i) > 1e-9:
                    continue
                if abs(a.l - b.l) not in (0, 2):
                    continue
                ang = self._sin2_reduced(a.l, a.j, a.m_j, b.l, b.j, b.m_j)
                if abs(ang) < 1e-14:
                    continue
                r2 = self._radial_rp(a.n, a.l, a.j, b.n, b.l, b.j, power=2)
                val = prefactor * r2 * ang
                M[a.index, b.index] = val
                M[b.index, a.index] = val
        return M

    def _sin2_reduced(self, l1, j1, mj1, l2, j2, mj2) -> float:
        """<l1 j1 mj1| sin^2(theta) |l2 j2 mj2>, radial part factored out.

        sin^2 = (2/3)(1 - P2).  Uses CG expansion into |m_l, m_s>; the operator
        is diagonal in m_s so mj1 must equal mj2.
        """
        if abs(mj1 - mj2) > 1e-9:
            return 0.0
        s = 0.5
        total = 0.0
        for m_s in (-0.5, 0.5):
            m_l1 = mj1 - m_s
            m_l2 = mj2 - m_s
            if abs(m_l1) > l1 + 1e-9 or abs(m_l2) > l2 + 1e-9:
                continue
            cg1 = _clebsch(l1, m_l1, s, m_s, j1, mj1)
            cg2 = _clebsch(l2, m_l2, s, m_s, j2, mj2)
            if cg1 == 0.0 or cg2 == 0.0:
                continue
            # <l1 m_l1| sin^2 |l2 m_l2>, requires m_l1 == m_l2
            if abs(m_l1 - m_l2) > 1e-9:
                continue
            delta = 1.0 if (l1 == l2) else 0.0
            p2 = self._p2_angular(l1, m_l1, l2, m_l2)
            ang = (2.0 / 3.0) * (delta - p2)
            total += cg1 * cg2 * ang
        return total

    @staticmethod
    def _p2_angular(l1, m1, l2, m2) -> float:
        """<l1 m1| P2(cos theta) |l2 m2> for spherical harmonics (m1 == m2)."""
        if abs(m1 - m2) > 1e-9:
            return 0.0
        m = int(round(m1))
        pref = ((-1) ** m) * np.sqrt((2 * l1 + 1) * (2 * l2 + 1))
        return pref * _wig3j(l1, 2, l2, 0, 0, 0) * _wig3j(l1, 2, l2, -m, 0, m)

    @lru_cache(maxsize=None)
    def _radial_rp(self, n1, l1, j1, n2, l2, j2, power=2) -> float:
        """<n1 l1 | r^power | n2 l2> in units of a0^power (ARC wavefunctions)."""
        e1 = self.atom.getEnergy(n1, l1, j1) / _HARTREE_EV
        e2 = self.atom.getEnergy(n2, l2, j2) / _HARTREE_EV
        inner = self.atom.alphaC ** (1.0 / 3.0)
        nmax = max(n1, n2)
        outer = 2.0 * nmax * (nmax + 15.0)
        step = 1e-3
        r1, u1 = self.atom.radialWavefunction(l1, 0.5, j1, e1, inner, outer, step)
        r2, u2 = self.atom.radialWavefunction(l2, 0.5, j2, e2, inner, outer, step)
        r1 = np.asarray(r1); u1 = np.asarray(u1)
        r2 = np.asarray(r2); u2 = np.asarray(u2)
        # interpolate u2 onto r1 grid (grids match when params match, but be safe)
        u2i = np.interp(r1, r2, u2, left=0.0, right=0.0)
        integrand = u1 * u2i * r1 ** power
        return float(np.trapezoid(integrand, r1))

    # --------------------------------------------------------------- laser
    def _resolve_polarization(self, polarization) -> Dict[int, complex]:
        if isinstance(polarization, str):
            if polarization not in _POLARIZATIONS:
                raise ValueError(
                    f"Unknown polarization '{polarization}'. "
                    f"Use one of {list(_POLARIZATIONS)} or a {{q: amplitude}} dict.")
            return dict(_POLARIZATIONS[polarization])
        if isinstance(polarization, dict):
            return polarization
        raise TypeError("polarization must be a str or {q: amplitude} dict.")

    def _photon_index(self) -> Dict[Tuple[int, int, float], int]:
        """Assign a rotating-frame photon index to each manifold via BFS over
        dipole-allowed (|dl|=1) connections.  Lowest-energy manifold -> 0."""
        mans = self.basis.manifolds
        energies = {m.nlj: self.atom.getEnergy(*m.nlj) for m in mans}
        order = sorted(mans, key=lambda m: energies[m.nlj])
        idx: Dict[Tuple[int, int, float], int] = {order[0].nlj: 0}
        changed = True
        while changed:
            changed = False
            for ma in mans:
                if ma.nlj not in idx:
                    continue
                for mb in mans:
                    if abs(ma.l - mb.l) != 1:
                        continue
                    step = 1 if energies[mb.nlj] > energies[ma.nlj] else -1
                    want = idx[ma.nlj] + step
                    if mb.nlj not in idx:
                        idx[mb.nlj] = want
                        changed = True
                    elif idx[mb.nlj] != want:
                        # inconsistent loop -> keep first assignment but warn
                        import warnings
                        warnings.warn(
                            f"Inconsistent RWA photon index for {mb.nlj}; "
                            "manifolds form a loop the single-frequency RWA "
                            "cannot represent consistently.")
        # manifolds with no dipole connection default to 0
        for m in mans:
            idx.setdefault(m.nlj, 0)
        return idx

    def laser_rwa_operator(self, beam, polarization="pi"):
        """Rotating-wave dipole coupling to a Gaussian ``beam``.

        Returns
        -------
        dict with keys
            ``coupling`` : ndarray, Hz per (V/m) of field amplitude E0.
                ``H_couple(E0) = coupling * E0``.
            ``frame_shift`` : ndarray (diagonal), Hz, intensity-independent
                rotating-frame energy offsets (``-p * f_laser`` per manifold).
            ``f_laser`` : laser frequency (Hz).

        The full RWA Hamiltonian at intensity ``I`` is
            ``h0() + diag(frame_shift) + coupling * sqrt(2 I / (c eps0))``.
        Use :func:`kamo.hamiltonian.sweep_intensity` to build this over a sweep.
        """
        pol = self._resolve_polarization(polarization)
        f_laser = beam.frequency()
        pidx = self._photon_index()

        dim = self.basis.dim
        coupling = np.zeros((dim, dim), dtype=complex)

        for a in self.basis:
            for b in self.basis:
                if b.index <= a.index:
                    continue
                if abs(a.l - b.l) != 1:
                    continue
                if abs(a.m_i - b.m_i) > 1e-9:      # nuclear spin is a spectator
                    continue
                d_tot = 0.0
                for q, amp in pol.items():
                    if abs((a.m_j + q) - b.m_j) > 1e-9:
                        continue
                    d = self.atom.getDipoleMatrixElement(
                        a.n, a.l, a.j, a.m_j, b.n, b.l, b.j, b.m_j, q)
                    d_tot += amp * d
                if d_tot == 0.0:
                    continue
                d_si = d_tot * c.a0 * c.e            # C*m
                # RWA co-rotating coupling: H = (d E0)/2 ; convert to Hz
                val = 0.5 * d_si / c.h              # Hz per (V/m)
                coupling[a.index, b.index] = val
                coupling[b.index, a.index] = np.conj(val)

        frame = np.zeros(dim, dtype=float)
        for man, sl in self.basis.manifold_slices():
            frame[sl] = -pidx[man.nlj] * f_laser

        return {"coupling": coupling, "frame_shift": frame, "f_laser": f_laser}

    def laser_stark_operator(self, beam, polarizabilities=None,
                             include_tensor: bool = True) -> np.ndarray:
        """Effective AC-Stark operator *per W/m^2* (Hz per W/m^2).

        ``H_stark(I) = laser_stark_operator(beam) * I``.

        Uses scalar (and optionally tensor) polarizabilities from
        :class:`kamo.light_shift.ComputePolarizabilities` evaluated at the beam
        wavelength.  Diagonal in the uncoupled basis.
        """
        if polarizabilities is None:
            from kamo import ComputePolarizabilities
            # force_arc=True uses ARC dipole elements directly and avoids the
            # portal-data (pandas) code path.
            polarizabilities = ComputePolarizabilities(force_arc=True)

        dim = self.basis.dim
        diag = np.zeros(dim, dtype=float)
        # U = -1/(2 eps0 c) * alpha_SI * I ; convert alpha au->SI, energy->Hz
        pre = -1.0 / (2 * c.epsilon0 * c.c) * c.convert_polarizability_au_to_SI / c.h

        for man, sl in self.basis.manifold_slices():
            a_s, a_v, a_t = polarizabilities.compute_fine_structure_polarizability(
                man.n, man.l, man.j, beam.wavelength)
            a_s = float(np.atleast_1d(a_s)[0])
            a_t = float(np.atleast_1d(a_t)[0]) if a_t is not None else 0.0
            for s in self.basis.states[sl]:
                shift = pre * a_s
                if include_tensor and man.j > 0.5 and a_t != 0.0:
                    j = man.j
                    tens = (3 * s.m_j ** 2 - j * (j + 1)) / (j * (2 * j - 1))
                    shift += pre * a_t * tens
                diag[s.index] = shift
        return np.diag(diag)
