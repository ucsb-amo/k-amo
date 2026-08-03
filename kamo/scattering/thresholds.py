"""Layer 0 — single-atom Zeeman thresholds for K39 4S1/2 (REUSES kamo.hamiltonian).

A collision channel's *threshold* is the sum of the two colliding atoms'
internal (hyperfine + Zeeman) energies at field ``B``.  These energies come
directly from :class:`kamo.hamiltonian.AtomicStructure` via **one**
eigenshuffle-tracked magnetic sweep; states are addressed by their low-field
``(F, mF)`` labels, which the sweep follows correctly through the Breit-Rabi
avoided crossings.

.. note::

   Do not diagonalise at ``B=0`` and identify states by high-field ``(m_j, m_i)``
   representatives — in K39 (inverted-order labelling) that mislabels F=1 vs F=2
   across the 461.7 MHz gap.  The tracked sweep used here is the correct route.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# K39 electronic ground manifold 4S1/2
GROUND = (4, 0, 0.5)


class K39Thresholds:
    """Zeeman-state energies E(F, mF; B) for the K39 ground manifold.

    Parameters
    ----------
    B_max_gauss : float
        Upper end of the tracked sweep (Gauss).  Queries must lie in
        ``[0, B_max_gauss]``.
    dB_gauss : float
        Sweep step (Gauss); default 0.05 for smooth interpolation.
    """

    def __init__(self, B_max_gauss: float = 1000.0, dB_gauss: float = 0.05):
        from kamo.hamiltonian import AtomicStructure
        self.B_max = float(B_max_gauss)
        self.dB = float(dB_gauss)
        self.model = AtomicStructure([GROUND])
        self._sweep = self.model.magnetic_sweep(B_max=self.B_max, dB=self.dB)

    # -- single-atom energies ----------------------------------------------
    def energy(self, F: int, mF: int, B_gauss) -> "float | np.ndarray":
        """Energy (Hz) of ``|F, mF>`` at field ``B_gauss`` (scalar or array)."""
        n, l, j = GROUND
        return self._sweep.get_energy(n, l, j, int(F), int(mF), at=B_gauss)

    def pair_threshold(self, state_a: Tuple[int, int],
                       state_b: Tuple[int, int], B_gauss) -> "float | np.ndarray":
        """Threshold energy (Hz) of the pair channel ``{a, b}`` at ``B_gauss``.

        ``E_thresh = E(state_a; B) + E(state_b; B)`` — the internal energy of
        two well-separated atoms.  Collision energy is measured relative to
        this; for the scattering length we take the E -> 0 (threshold) limit.
        """
        Fa, mFa = state_a
        Fb, mFb = state_b
        return self.energy(Fa, mFa, B_gauss) + self.energy(Fb, mFb, B_gauss)

    def state_composition(self, F: int, mF: int, B_gauss: float) -> np.ndarray:
        """Field-dependent composition of |F,mF> over the |m_s=m_j, m_i> basis.

        Returns the 8-vector of amplitudes of the Zeeman eigenstate that
        adiabatically connects to |F,mF>, at field ``B_gauss``, in the kamo
        (m_j, m_i) basis of the 4S1/2 manifold (m_j = m_s since l=0).  This is
        the field-dependent single-atom spin state needed to build the
        singlet/triplet frame transformation at finite field.
        """
        n, l, j = GROUND
        idx = self._sweep._tracked_index_F_mF(n, l, j, int(F), int(mF), step=0)
        step = self._sweep.nearest_step(float(B_gauss))
        v = self._sweep.vectors[step][:, idx]
        return np.real_if_close(v, tol=1000).astype(float)

    def hyperfine_splitting_hz(self) -> float:
        """Zero-field F=2 <-> F=1 splitting (Hz) — a self-consistency probe.

        For K39 this should be ~461.7e6 Hz.
        """
        return float(self.energy(2, 2, 0.0) - self.energy(1, 1, 0.0))
