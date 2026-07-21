"""High-level, easy-to-use entry point for K39 multi-manifold structure.

Example
-------
>>> from kamo.hamiltonian import AtomicStructure
>>> from kamo import GaussianBeam
>>> model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)])
>>>
>>> # Magnetic field sweep 0 -> 600 G in 0.1 G steps (default), states tracked
>>> res = model.magnetic_sweep(B_max=600.0)              # dB=0.1 G default
>>> res.energies.shape                                   # (n_steps, n_states)
>>>
>>> # Diamagnetic option
>>> res_d = model.magnetic_sweep(B_max=600.0, diamagnetic=True)
>>>
>>> # Laser intensity sweep (RWA dipole coupling)
>>> beam = GaussianBeam(waist=50e-6, wavelength=767e-9, power=1e-3)
>>> res_l = model.laser_sweep(beam, I_max=beam.I0, model="rwa",
...                           polarization="sigma+")
>>>
>>> # One-off diagonalization at a single field
>>> E, V = model.solve(B_gauss=10.0)
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .basis import Basis
from .builder import HamiltonianBuilder
from .diagonalize import (MagneticSweepResult, LaserSweepResult,
                          SweepResult, diagonalize, sweep_field,
                          sweep_intensity)
from .state_labels import StateLabelMixin


def make_nlj_basis(
    n: int,
    l: int,
    n_range: int = 0,
    l_range: int = 1,
) -> List[Tuple[int, int, float]]:
    """Build a list of ``(n, l, j)`` manifold tuples centred on ``(n, l)``.

    The basis spans:

    * principal quantum number  ``n' ∈ [n - n_range, n + n_range]``
    * orbital angular momentum  ``l' ∈ [max(0, l - l_range), l + l_range]``
      (capped to ``l' ≤ n' - 1``);
    * all physically valid ``j'`` for each ``(n', l')``
      (i.e. ``j' = l' + 1/2`` and, for ``l' > 0``, also ``j' = l' - 1/2``).

    Parameters
    ----------
    n : int
        Centre principal quantum number.
    l : int
        Centre orbital angular-momentum quantum number.
    n_range : int, optional
        Half-width of the n window (default 0 → only ``n``).
    l_range : int, optional
        Half-width of the l window (default 1 → ``l ± 1``).

    Returns
    -------
    list of (n, l, j) tuples

    Examples
    --------
    >>> make_nlj_basis(4, 0)
    # n_range=0, l_range=1 → l ∈ {0, 1}
    [(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)]

    >>> make_nlj_basis(4, 0, n_range=1)
    # n ∈ {3, 4, 5}, l ∈ {0, 1}
    [(3, 0, 0.5), (3, 1, 0.5), (3, 1, 1.5),
     (4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5),
     (5, 0, 0.5), (5, 1, 0.5), (5, 1, 1.5)]

    >>> make_nlj_basis(59, 0, n_range=3, l_range=2)
    # matches pairinteraction default basis for Rydberg S states
    """
    manifolds = []
    l_lo = max(0, l - l_range)
    l_hi = l + l_range
    for n_prime in range(n - n_range, n + n_range + 1):
        if n_prime < 1:
            continue
        for l_prime in range(l_lo, min(l_hi, n_prime - 1) + 1):
            # j = l - 1/2 (only valid when l > 0)
            if l_prime > 0:
                manifolds.append((n_prime, l_prime, l_prime - 0.5))
            # j = l + 1/2 (always valid)
            manifolds.append((n_prime, l_prime, l_prime + 0.5))
    return manifolds


class AtomicStructure(StateLabelMixin):
    """Build a basis from (n, l, j) manifolds and diagonalize its Hamiltonian.

    Parameters
    ----------
    manifolds : iterable of (n, l, j) tuples (or Manifold objects).
    atom : kamo.Potassium39, optional
        Reuse an existing ARC atom object.
    energy_reference_nlj : (n, l, j), optional
        Fine-structure energy reference (defaults to first manifold).
    """

    def __init__(self, manifolds: Iterable, atom=None,
                 energy_reference_nlj=None):
        self.basis = Basis(manifolds)
        self.builder = HamiltonianBuilder(
            self.basis, atom=atom, energy_reference_nlj=energy_reference_nlj)

    @classmethod
    def around(cls, n: int, l: int, n_range: int = 0, l_range: int = 1,
               atom=None, energy_reference_nlj=None) -> "AtomicStructure":
        """Construct an :class:`AtomicStructure` centred on ``(n, l)``.

        Shorthand for ``AtomicStructure(make_nlj_basis(n, l, n_range, l_range))``.

        Parameters
        ----------
        n : int
            Centre principal quantum number.
        l : int
            Centre orbital angular-momentum quantum number.
        n_range : int, optional
            Half-width of the n window (default 0 → only shell ``n``).
        l_range : int, optional
            Half-width of the l window (default 1 → ``l ± 1``).
        atom, energy_reference_nlj :
            Forwarded to :class:`AtomicStructure` (see its docstring).

        Examples
        --------
        # 4S₁/₂ + 4P₁/₂ + 4P₃/₂  (ground state + first excited manifolds)
        >>> model = AtomicStructure.around(4, 0)

        # n=59 Rydberg S-state with ±3 n shells and l up to 2
        >>> model = AtomicStructure.around(59, 0, n_range=3, l_range=2)
        """
        manifolds = make_nlj_basis(n, l, n_range=n_range, l_range=l_range)
        return cls(manifolds, atom=atom, energy_reference_nlj=energy_reference_nlj)

    def __getitem__(self, key):
        """Access a manifold in the atomic structure.

        Parameters
        ----------
        key : int or tuple
            - int: index of the manifold by order of appearance (0-based).
            - 3-tuple (n, l, j): manifold with those quantum numbers.

        Returns
        -------
        Manifold

        Examples
        --------
        >>> model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)])
        >>> model[0]                    # first manifold: (4, 0, 0.5)
        >>> model[(4, 1, 0.5)]          # second manifold by quantum numbers
        """
        if isinstance(key, int):
            return self.basis.manifold_by_index(key)
        elif isinstance(key, tuple) and len(key) == 3:
            return self.basis[key]
        else:
            raise TypeError(
                "AtomicStructure indices must be int (manifold order) "
                "or 3-tuple (n, l, j) (manifold quantum numbers)."
            )

    def states(self, n=None, l=None, j=None, F=None, mF=None, mJ=None, mI=None):
        """Return states in the basis, with optional filtering.

        Delegates to :meth:`~.basis.Basis.states`; all keyword arguments
        are forwarded unchanged.

        Parameters
        ----------
        n, l, j : optional
            Restrict to manifolds matching these quantum numbers.
        F, mF, mJ, mI : optional
            Sub-manifold quantum-number filters (see :meth:`~.basis.Manifold.states`).

        Returns
        -------
        list of (n, l, j, m_j, m_i) 5-tuples

        Examples
        --------
        >>> model.states()                  # all states
        >>> model.states(n=4, l=0)          # ground manifold
        >>> model.states(n=4, l=0, mJ=0.5)  # ground manifold, mJ=+1/2
        >>> model.states(F=1, mF=-1)        # |F=1, mF=-1> across all manifolds
        >>> res.plot(states=model.states(n=4, l=0, F=2))  # pass directly to plot
        """
        return self.basis.states(n=n, l=l, j=j, F=F, mF=mF, mJ=mJ, mI=mI)


    # ------------------------------------------------------------------ H0
    def h0(self, include_quadrupole: bool = True) -> np.ndarray:
        """Field-free Hamiltonian (Hz)."""
        return self.builder.h0(include_quadrupole=include_quadrupole)

    def solve(self, B_gauss: float = 0.0, include_diamagnetic: bool = False,
              include_quadrupole: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize H at a single static magnetic field.

        Returns ``(energies, V)`` where columns of ``V`` are the new eigenstates
        expressed in the old (uncoupled) basis.
        """
        H = self.builder.h0(include_quadrupole=include_quadrupole)
        if B_gauss:
            H = H + self.builder.zeeman_operator() * B_gauss
            if include_diamagnetic:
                H = H + self.builder.diamagnetic_operator() * (B_gauss ** 2)
        return diagonalize(H)

    def get_energy(self, n: int, l: int, j: float,
                   m_j: float, m_i: float,
                   B_gauss: float = 0.0,
                   include_diamagnetic: bool = False,
                   include_quadrupole: bool = True) -> float:
        """Return the energy (Hz) of ``|n l j; m_j m_i>`` at *B_gauss*.

        Uses a magnetic sweep from 0 to *B_gauss* with eigenshuffle tracking
        so that state labels are correct through avoided crossings.  At
        ``B_gauss=0`` the field-free Hamiltonian is diagonalized directly.

        Parameters
        ----------
        n, l, j : fine-structure quantum numbers.
        m_j, m_i : magnetic quantum numbers of J and I.
        B_gauss : float
            Static magnetic field in Gauss (default 0).
        include_diamagnetic : bool
            Add the diamagnetic term (default False).
        include_quadrupole : bool
            Include the electric-quadrupole hyperfine term (default True).

        Returns
        -------
        float : energy in Hz.
        """
        if B_gauss == 0.0:
            E, V = self.solve(B_gauss=0.0,
                              include_diamagnetic=include_diamagnetic,
                              include_quadrupole=include_quadrupole)
            try:
                basis_idx = self.basis.index_of(n, l, j, m_j, m_i)
            except KeyError:
                raise KeyError(
                    f"|{n},{l},{j}; m_j={m_j}, m_i={m_i}> is not in the basis.")
            weights = np.abs(V[basis_idx, :]) ** 2
            return float(E[np.argmax(weights)])
        dB = 0.1
        res = self.magnetic_sweep(
            B_max=abs(B_gauss) + dB, dB=dB,
            diamagnetic=include_diamagnetic,
            include_quadrupole=include_quadrupole,
        )
        return res.get_energy(n, l, j, m_j, m_i, at=B_gauss)

    # -------------------------------------------------------------- sweeps
    def magnetic_sweep(self, B_max: float, dB: float = 0.1,
                       diamagnetic: bool = False,
                       include_quadrupole: bool = True) -> MagneticSweepResult:
        """Sweep B from 0 to ``B_max`` (step ``dB``, default 0.1 G) with
        eigenshuffle state tracking.  See :func:`sweep_field`."""
        return sweep_field(self.builder, B_max, dB=dB,
                           include_diamagnetic=diamagnetic,
                           include_quadrupole=include_quadrupole)

    def laser_sweep(self, beam, I_max: float = None, n_points: int = 200,
                    model: str = "rwa", polarization="pi",
                    B_gauss: float = 0.0,
                    include_quadrupole: bool = True,
                    polarizabilities=None) -> LaserSweepResult:
        """Sweep laser intensity from 0 to ``I_max`` with eigenshuffle tracking.

        ``model="rwa"`` uses rotating-wave dipole coupling built from ``beam``;
        ``model="stark"`` uses the effective AC-Stark operator.  See
        :func:`sweep_intensity`.

        To run a laser sweep *at a field taken from a magnetic sweep*, pass that
        field via ``B_gauss`` (a static Zeeman term is added to H0, so at
        intensity 0 the eigenstates match the magnetic-sweep states there)::

            resB = model.magnetic_sweep(B_max=600.0)
            resL = model.laser_sweep(beam, B_gauss=resB.field_at(200.0))
        """
        if I_max is None:
            I_max = beam.I0
        return sweep_intensity(self.builder, beam, I_max, n_points=n_points,
                               model=model, polarization=polarization,
                               B_gauss=B_gauss,
                               include_quadrupole=include_quadrupole,
                               polarizabilities=polarizabilities)
