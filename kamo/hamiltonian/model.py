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

from typing import Iterable, Optional, Tuple

import numpy as np

from .basis import Basis
from .builder import HamiltonianBuilder
from .diagonalize import (SweepResult, diagonalize, sweep_field,
                          sweep_intensity)


class AtomicStructure:
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
        """Return the energy (Hz) of the state ``|n l j; m_j m_i>`` at ``B_gauss``.

        The Hamiltonian is diagonalized at ``B_gauss`` and the eigenstate with
        the largest overlap on the specified uncoupled basis state is returned.
        At low fields this is exact; at high fields it reflects the dominant
        uncoupled character of the dressed eigenstate.

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
        try:
            basis_idx = self.basis.index_of(n, l, j, m_j, m_i)
        except KeyError:
            raise KeyError(
                f"|{n},{l},{j}; m_j={m_j}, m_i={m_i}> is not in the basis.")
        E, V = self.solve(B_gauss=B_gauss,
                          include_diamagnetic=include_diamagnetic,
                          include_quadrupole=include_quadrupole)
        weights = np.abs(V[basis_idx, :]) ** 2
        return float(E[np.argmax(weights)])

    # -------------------------------------------------------------- sweeps
    def magnetic_sweep(self, B_max: float, dB: float = 0.1,
                       diamagnetic: bool = False,
                       include_quadrupole: bool = True) -> SweepResult:
        """Sweep B from 0 to ``B_max`` (step ``dB``, default 0.1 G) with
        eigenshuffle state tracking.  See :func:`sweep_field`."""
        return sweep_field(self.builder, B_max, dB=dB,
                           include_diamagnetic=diamagnetic,
                           include_quadrupole=include_quadrupole)

    def laser_sweep(self, beam, I_max: float = None, n_points: int = 200,
                    model: str = "rwa", polarization="pi",
                    B_gauss: float = 0.0,
                    include_quadrupole: bool = True,
                    polarizabilities=None) -> SweepResult:
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
