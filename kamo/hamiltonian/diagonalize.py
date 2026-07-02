"""Direct diagonalization, eigenshuffle state tracking, and field/intensity sweeps.

The central routines return both the **energies** of the new eigenstates and the
**transformation** ``V`` from the old (uncoupled) basis to the new eigenbasis:
columns of ``V`` are the eigenvectors expressed in the old basis, so that

    H = V @ diag(E) @ V.conj().T          and      c_new = V.conj().T @ c_old .

For sweeps, the eigenpairs at each step are re-ordered by :func:`eigenshuffle`
so a given output index follows one physical state continuously through avoided
crossings, starting from the zero-field / zero-intensity ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .basis import Basis
from .builder import _clebsch


def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize a Hermitian Hamiltonian.

    Returns
    -------
    energies : ndarray, shape (n,)
        Eigenvalues in ascending order (same units as ``H``, i.e. Hz here).
    V : ndarray, shape (n, n)
        Transformation old->new: columns are eigenvectors in the old basis.
    """
    energies, V = np.linalg.eigh(H)
    return energies, V


def eigenshuffle(matrices) -> Tuple[np.ndarray, np.ndarray]:
    """Track eigenpairs continuously across a sequence of Hermitian matrices.

    Parameters
    ----------
    matrices : sequence of (n, n) Hermitian arrays, ordered along a sweep
        parameter (e.g. increasing B or intensity).  The first matrix defines
        the reference ordering (ascending eigenvalue).

    Returns
    -------
    energies : ndarray, shape (n_steps, n)
        ``energies[k, i]`` is the energy of tracked state ``i`` at step ``k``.
    vectors : ndarray, shape (n_steps, n, n)
        ``vectors[k][:, i]`` is the eigenvector of tracked state ``i`` at step
        ``k`` (in the old basis), with sign chosen for continuity.

    Notes
    -----
    Consecutive slices are matched by maximizing total squared eigenvector
    overlap using the Hungarian assignment algorithm; this correctly follows
    states through avoided crossings where a naive energy sort would swap them.
    """
    from scipy.optimize import linear_sum_assignment

    mats = list(matrices)
    n_steps = len(mats)
    if n_steps == 0:
        raise ValueError("eigenshuffle received an empty sequence.")

    n = mats[0].shape[0]
    energies = np.empty((n_steps, n), dtype=float)
    vectors = np.empty((n_steps, n, n), dtype=complex)

    # reference: ascending eigenvalue order
    e0, v0 = np.linalg.eigh(mats[0])
    energies[0] = e0
    vectors[0] = v0
    prev_v = v0

    for k in range(1, n_steps):
        e, v = np.linalg.eigh(mats[k])
        # overlap[i, j] = |<prev_i | v_j>|^2
        overlap = np.abs(prev_v.conj().T @ v) ** 2
        # maximize overlap -> minimize (1 - overlap)
        row, col = linear_sum_assignment(1.0 - overlap)
        order = np.empty(n, dtype=int)
        order[row] = col
        e = e[order]
        v = v[:, order]
        # fix sign/phase for continuity
        for i in range(n):
            ov = np.vdot(prev_v[:, i], v[:, i])
            if abs(ov) > 0:
                v[:, i] *= np.conj(ov) / abs(ov)
        energies[k] = e
        vectors[k] = v
        prev_v = v

    return energies, vectors


@dataclass
class SweepResult:
    """Result of a magnetic-field or laser-intensity sweep.

    Attributes
    ----------
    param : ndarray, shape (n_steps,)
        Swept parameter values (Gauss for B-sweeps, W/m^2 for intensity sweeps).
    param_name : str
    energies : ndarray, shape (n_steps, n_states)
        Tracked eigen-energies in Hz.  ``energies[:, i]`` follows one physical
        state across the whole sweep.
    vectors : ndarray, shape (n_steps, n, n)
        Transformation old->new at each step (columns are eigenvectors).
    basis : Basis
    """

    param: np.ndarray
    param_name: str
    energies: np.ndarray
    vectors: np.ndarray
    basis: Basis

    @property
    def energies_MHz(self) -> np.ndarray:
        return self.energies / 1e6

    def transform(self, step: int) -> np.ndarray:
        """Transformation matrix V (old->new) at the given sweep step."""
        return self.vectors[step]

    def dominant_state(self, i: int, step: int = 0):
        """Return the uncoupled BasisState with the largest weight in tracked
        state ``i`` at ``step`` (useful for labelling)."""
        weights = np.abs(self.vectors[step][:, i]) ** 2
        return self.basis[int(np.argmax(weights))]

    def label(self, i: int, step: int = 0) -> str:
        s = self.dominant_state(i, step)
        return (f"|{s.n},{s.l},{s.j}; m_j={s.m_j:+.1f}, m_i={s.m_i:+.1f}> "
                f"(m_F={s.m_f:+.1f})")

    def convert_label(self, state: tuple) -> tuple:
        """Convert a state 5-tuple between uncoupled and coupled-basis labels.

        Direction is detected automatically from the type of the last two
        elements:

        * ``(n, l, j, m_j: float, m_i: float)`` → ``(n, l, j, F: int, mF: int)``
          — finds the F with the largest CG weight for the given ``(m_j, m_i)``.
        * ``(n, l, j, F: int, mF: int)`` → ``(n, l, j, m_j: float, m_i: float)``
          — finds the ``(m_j, m_i)`` pair with the largest CG weight for the
          given ``(F, mF)``.

        The conversion uses Clebsch-Gordan coefficients and is purely algebraic
        (no sweep data required).  It is most meaningful at low field where the
        coupled and uncoupled labels are nearly exact.

        Parameters
        ----------
        state : 5-tuple

        Returns
        -------
        5-tuple with the other label convention.

        Examples
        --------
        >>> res.convert_label((4, 0, 0.5, -0.5, -1.5))   # m_j, m_i -> F, mF
        (4, 0, 0.5, 1, -2)
        >>> res.convert_label((4, 0, 0.5, 1, -2))         # F, mF -> m_j, m_i
        (4, 0, 0.5, -0.5, -1.5)
        """
        if len(state) != 5:
            raise ValueError("state must be a 5-tuple (n, l, j, a, b).")
        n, l, j, a, b = state

        # find i_nuclear for this manifold
        i_nuc = None
        for man in self.basis.manifolds:
            if man.n == n and man.l == l and abs(man.j - j) < 1e-9:
                i_nuc = man.i_nuclear
                break
        if i_nuc is None:
            raise KeyError(f"Manifold ({n}, {l}, {j}) not in basis.")

        allowed_F = [abs(j - i_nuc) + k
                     for k in range(int(round(2 * min(j, i_nuc))) + 1)]

        if isinstance(a, int) and isinstance(b, int):
            # F, mF -> dominant (m_j, m_i)
            F, mF = a, b
            best_mj, best_mi, best_cg = None, None, -1.0
            mj_vals = [-j + k for k in range(int(round(2 * j)) + 1)]
            mi_vals = [-i_nuc + k for k in range(int(round(2 * i_nuc)) + 1)]
            for m_j in mj_vals:
                m_i = mF - m_j
                if not any(abs(m_i - v) < 1e-9 for v in mi_vals):
                    continue
                cg = abs(_clebsch(j, m_j, i_nuc, m_i, F, mF))
                if cg > best_cg:
                    best_cg, best_mj, best_mi = cg, m_j, m_i
            if best_mj is None:
                raise ValueError(
                    f"|{n},{l},{j}; F={F}, mF={mF}> has no valid (m_j, m_i).")
            return (n, l, j, best_mj, best_mi)

        else:
            # m_j, m_i -> dominant F  (mF = m_j + m_i is fixed)
            m_j, m_i = float(a), float(b)
            mF = m_j + m_i
            best_F, best_cg = None, -1.0
            for F in allowed_F:
                if abs(mF) > F + 1e-9:
                    continue
                cg = abs(_clebsch(j, m_j, i_nuc, m_i, F, mF))
                if cg > best_cg:
                    best_cg, best_F = cg, F
            if best_F is None:
                raise ValueError(
                    f"|{n},{l},{j}; m_j={m_j}, m_i={m_i}> has no valid F.")
            return (n, l, j, int(round(best_F)), int(round(mF)))

    # -- state lookup -------------------------------------------------------
    def indices_for(self, n: int, l: int, j: float,
                    m_j: float = None, m_i: float = None,
                    step: int = 0) -> List[int]:
        """Return tracked-state indices whose dominant uncoupled component
        belongs to manifold ``(n, l, j)``, or to the single state
        ``|n l j; m_j m_i>`` when both magnetic quantum numbers are given.

        Parameters
        ----------
        n, l, j : fine-structure quantum numbers.
        m_j, m_i : if both supplied, restrict to the one matching state;
            if omitted, return all states in the ``(n, l, j)`` manifold.
            If both are Python ``int``, they are interpreted as **F, mF**
            (low-field coupled-basis labels) via a CG overlap search.
        step : sweep step at which dominant-basis weights are evaluated.

        Returns
        -------
        list of int
        """
        if m_j is not None and m_i is not None:
            if isinstance(m_j, int) and isinstance(m_i, int):
                # integer args -> interpret as F, mF (low-field coupled labels)
                return [self._tracked_index_F_mF(n, l, j, F=m_j, mF=m_i, step=step)]
            return [self._tracked_index(n, l, j, m_j, m_i, step)]
        out = []
        for i in range(self.energies.shape[1]):
            s = self.dominant_state(i, step)
            if s.n == n and s.l == l and abs(s.j - j) < 1e-9:
                out.append(i)
        return out

    def _resolve_states(self, states, step: int = 0) -> List[int]:
        """Convert a *states* specifier to a list of tracked-state indices.

        Accepted forms
        --------------
        ``None``
            All states.
        ``(n, l, j)`` — 3-tuple of numbers
            All tracked states in the ``(n, l, j)`` manifold.
        ``(n, l, j, m_j, m_i)`` — 5-tuple of numbers
            The single tracked state whose dominant component is
            ``|n l j; m_j m_i>``.
        list/sequence of tuples
            Each element is a ``(n, l, j)`` or ``(n, l, j, m_j, m_i)`` tuple;
            results are concatenated (duplicates preserved).
        sequence of int
            Explicit index list (existing behaviour).
        """
        if states is None:
            return list(range(self.energies.shape[1]))

        def _is_qn_tuple(t):
            return (isinstance(t, tuple)
                    and len(t) in (3, 5)
                    and all(isinstance(s, (int, float)) for s in t))

        def _resolve_one(t):
            if len(t) == 3:
                n, l, j = t
                return self.indices_for(n, l, j, step=step)
            else:
                n, l, j, m_j, m_i = t
                return self.indices_for(n, l, j, m_j, m_i, step=step)

        if _is_qn_tuple(states):
            return _resolve_one(states)

        # list/sequence — check whether elements are qn-tuples or plain ints
        items = list(states)
        if items and _is_qn_tuple(items[0]):
            out = []
            for t in items:
                out.extend(_resolve_one(t))
            return out

        return items

    def _tracked_index(self, n: int, l: int, j: float,
                       m_j: float, m_i: float, step: int) -> int:
        """Return the tracked-state column index whose uncoupled component
        ``|n l j; m_j m_i>`` has the largest weight at ``step``."""
        try:
            basis_idx = self.basis.index_of(n, l, j, m_j, m_i)
        except KeyError:
            raise KeyError(
                f"|{n},{l},{j}; m_j={m_j}, m_i={m_i}> is not in the basis.")
        weights = np.abs(self.vectors[step][basis_idx, :]) ** 2
        return int(np.argmax(weights))

    def _tracked_index_F_mF(self, n: int, l: int, j: float,
                            F: int, mF: int, step: int) -> int:
        """Return the tracked-state column index for the low-field state
        ``|n l j; F, mF>`` (coupled basis).

        The coupled state is CG-expanded into the uncoupled ``|m_j, m_i>``
        basis; the tracked state at ``step`` with the largest overlap is
        returned.  This is most meaningful at low field (step 0) where the
        hyperfine eigenstates are nearly pure ``|F, mF>``.
        """
        # find i_nuclear from the basis manifold
        i_nuc = None
        for man in self.basis.manifolds:
            if man.n == n and man.l == l and abs(man.j - j) < 1e-9:
                i_nuc = man.i_nuclear
                break
        if i_nuc is None:
            raise KeyError(f"Manifold ({n}, {l}, {j}) not in basis.")

        # build the |F, mF> state vector in the full basis
        psi = np.zeros(self.basis.dim, dtype=complex)
        for s in self.basis.states:
            if s.n == n and s.l == l and abs(s.j - j) < 1e-9:
                cg = _clebsch(j, s.m_j, i_nuc, s.m_i, F, mF)
                if cg:
                    psi[s.index] = cg

        # overlap of each tracked eigenvector with |F, mF>
        # vectors[step] has shape (n_basis, n_tracked); columns are eigenvectors
        overlaps = np.abs(psi @ self.vectors[step]) ** 2
        return int(np.argmax(overlaps))

    def get_energy(self, n: int, l: int, j: float,
                   m_j: float, m_i: float,
                   identify_at_step: int = 0) -> np.ndarray:
        """Return the energy track (Hz) for the state whose dominant uncoupled
        component is ``|n l j; m_j m_i>``.

        The state is identified at ``identify_at_step`` (default 0, i.e. the
        start of the sweep where the uncoupled labels are most meaningful) and
        then followed continuously across the entire sweep.

        Parameters
        ----------
        n, l, j : fine-structure quantum numbers.
        m_j, m_i : magnetic quantum numbers of J and I.
        identify_at_step : int
            Step at which to choose the tracked state by maximum uncoupled-
            basis weight (default 0).

        Returns
        -------
        energies : ndarray, shape (n_steps,)
            Energy in Hz of the tracked state at every sweep point.
        """
        i = self._tracked_index(n, l, j, m_j, m_i, identify_at_step)
        return self.energies[:, i]

    def get_transition_frequency(self, state_a, state_b,
                                 identify_at_step: int = 0) -> np.ndarray:
        """Return the transition frequency ``E_b - E_a`` (Hz) across the sweep.

        Each state is specified as a ``(n, l, j, m_j, m_i)`` 5-tuple; the
        dominant uncoupled-basis component is used to identify and track the
        state throughout the sweep.

        Parameters
        ----------
        state_a, state_b : tuple ``(n, l, j, m_j, m_i)``
            Quantum numbers identifying each state.
        identify_at_step : int
            Sweep step at which quantum-number labels are most meaningful
            (default 0, i.e. the start of the sweep).

        Returns
        -------
        freq : ndarray, shape (n_steps,)
            ``E_b - E_a`` in Hz at every sweep point.  Positive when state b
            lies above state a.
        """
        if len(state_a) != 5 or len(state_b) != 5:
            raise ValueError(
                "Each state must be a (n, l, j, m_j, m_i) 5-tuple.")
        ea = self.get_energy(*state_a, identify_at_step=identify_at_step)
        eb = self.get_energy(*state_b, identify_at_step=identify_at_step)
        return eb - ea

    # -- convenience --------------------------------------------------------
    def nearest_step(self, value: float) -> int:
        """Index of the sweep step whose parameter is closest to ``value``.

        For a magnetic sweep, ``value`` is a field in Gauss; use with
        :meth:`AtomicStructure.laser_sweep` to start a laser sweep at that
        field (see ``field_at``)."""
        return int(np.argmin(np.abs(self.param - value)))

    def field_at(self, value: float) -> float:
        """The magnetic-field value (Gauss) at the sweep step nearest ``value``.

        Convenience for chaining a laser sweep onto a magnetic sweep::

            resB = model.magnetic_sweep(B_max=600.0)
            resL = model.laser_sweep(beam, I_max=beam.I0,
                                     B_gauss=resB.field_at(200.0))
        """
        return float(self.param[self.nearest_step(value)])

    def x_axis(self, unit: Optional[str] = None):
        """Return ``(x_values, x_label)`` for plotting.

        ``unit`` overrides the default axis scaling:
        for a field sweep ``"G"`` (default); for an intensity sweep
        ``"W/m^2"`` (default), ``"mW/cm^2"``, or ``"kW/cm^2"``.
        """
        name = self.param_name.lower()
        x = self.param
        if "gauss" in name or "b (" in name:
            return x, (self.param_name if unit is None else f"B ({unit})")
        # intensity sweep
        if unit in (None, "W/m^2"):
            return x, "Intensity (W/m$^2$)"
        if unit == "mW/cm^2":
            return x * 0.1, "Intensity (mW/cm$^2$)"
        if unit == "kW/cm^2":
            return x * 1e-7, "Intensity (kW/cm$^2$)"
        return x, self.param_name

    def plot(self, ax=None, energy_unit: str = "MHz", x_unit: Optional[str] = None,
             states=None, label_states: bool = False, label_step: int = 0,
             energy_offset: float = 0.0, legend: bool = False, **plot_kwargs):
        """Plot tracked eigen-energies versus the swept parameter.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on; a new figure/axes is created if omitted.
        energy_unit : {"Hz", "kHz", "MHz", "GHz", "THz"}
            Units for the energy axis (default MHz).
        x_unit : str, optional
            Passed to :meth:`x_axis` (e.g. ``"mW/cm^2"`` for intensity sweeps).
        states : sequence of int, or (n, l, j) tuple, or (n, l, j, m_j, m_i) tuple, or list of such tuples, optional
            Which states to plot.  Accepts:
            - ``None`` (default): all states.
            - ``(n, l, j)`` 3-tuple: all states in that manifold.
            - ``(n, l, j, m_j, m_i)`` 5-tuple: a single state.
            - list of 3- or 5-tuples: union of the above, concatenated.
            - sequence of int: explicit tracked-state indices.
        label_states : bool
            Annotate each line with its dominant-basis-state label.
        label_step : int
            Sweep step at which dominant-state labels are evaluated (default 0,
            i.e. the start of the sweep).
        energy_offset : float
            Value (in Hz) subtracted from every energy before scaling; handy for
            referencing to a particular state or manifold.
        legend : bool
            Show a legend built from the state labels.
        **plot_kwargs : forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib Axes.
        """
        import matplotlib.pyplot as plt

        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9,
                 "THz": 1e12}[energy_unit]
        if ax is None:
            _, ax = plt.subplots()

        x, xlabel = self.x_axis(x_unit)
        idxs = self._resolve_states(states, label_step)
        y = (self.energies - energy_offset) / scale

        for i in idxs:
            lbl = self.label(i, label_step) if (label_states or legend) else None
            (line,) = ax.plot(x, y[:, i], label=lbl, **plot_kwargs)
            if label_states:
                ax.annotate(lbl, (x[-1], y[-1, i]), fontsize=6,
                            va="center", ha="left",
                            xytext=(3, 0), textcoords="offset points")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Energy ({energy_unit})")
        if legend:
            ax.legend(fontsize=6, loc="best")
        return ax


def sweep_field(builder, B_max: float, dB: float = 0.1,
                include_diamagnetic: bool = False,
                include_quadrupole: bool = True) -> SweepResult:
    """Diagonalize H(B) = H0 + B*Zeeman [+ B^2*diamagnetic] over a field sweep.

    The field is swept from 0 to ``B_max`` in steps of ``dB`` (default 0.1 G),
    and eigenshuffle tracks each state from the zero-field ordering through
    avoided crossings.

    Parameters
    ----------
    builder : HamiltonianBuilder
    B_max : float
        Maximum field in Gauss.
    dB : float
        Field step in Gauss (default 0.1).
    include_diamagnetic : bool
        Include the (optional) B^2 diamagnetic term.
    include_quadrupole : bool
        Pass-through to ``h0`` for the electric-quadrupole hyperfine term.
    """
    H0 = builder.h0(include_quadrupole=include_quadrupole)
    Zop = builder.zeeman_operator()
    Dop = builder.diamagnetic_operator() if include_diamagnetic else None

    n_steps = int(np.floor(B_max / dB + 1e-9)) + 1
    B = np.arange(n_steps) * dB

    mats = []
    for b in B:
        H = H0 + Zop * b
        if Dop is not None:
            H = H + Dop * (b ** 2)
        mats.append(H)

    energies, vectors = eigenshuffle(mats)
    return SweepResult(B, "B (Gauss)", energies, vectors, builder.basis)


def sweep_intensity(builder, beam, I_max: float, n_points: int = 200,
                    model: str = "rwa", polarization="pi",
                    B_gauss: float = 0.0, include_quadrupole: bool = True,
                    polarizabilities=None) -> SweepResult:
    """Diagonalize H(I) over a laser-intensity sweep from 0 to ``I_max``.

    Parameters
    ----------
    builder : HamiltonianBuilder
    beam : kamo.GaussianBeam
        Supplies the laser frequency (RWA) / wavelength (Stark).
    I_max : float
        Maximum intensity in W/m^2.
    n_points : int
        Number of intensity steps (linearly spaced from 0).
    model : {"rwa", "stark"}
        "rwa"   -> rotating-wave dipole coupling (couples manifolds; primary).
        "stark" -> effective AC-Stark shift operator (diagonal).
    polarization : str or {q: amplitude}
        Laser polarization for the RWA model.
    B_gauss : float
        Optional static magnetic field added via the Zeeman operator.
    """
    H0 = builder.h0(include_quadrupole=include_quadrupole)
    if B_gauss:
        H0 = H0 + builder.zeeman_operator() * B_gauss

    I = np.linspace(0.0, I_max, n_points)

    if model == "rwa":
        rwa = builder.laser_rwa_operator(beam, polarization=polarization)
        base = H0 + np.diag(rwa["frame_shift"])
        coupling = rwa["coupling"]
        # field amplitude E0 = sqrt(2 I / (c eps0))
        from kamo import constants as c
        E0 = np.sqrt(2.0 * I / (c.c * c.epsilon0))
        mats = [base + coupling * e0 for e0 in E0]
    elif model == "stark":
        Sop = builder.laser_stark_operator(beam, polarizabilities=polarizabilities)
        mats = [H0 + Sop * inten for inten in I]
    else:
        raise ValueError("model must be 'rwa' or 'stark'.")

    energies, vectors = eigenshuffle(mats)
    return SweepResult(I, "Intensity (W/m^2)", energies, vectors, builder.basis)
