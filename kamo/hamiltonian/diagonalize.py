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

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Interpolation helpers shared by SweepResult methods and subclasses
# ---------------------------------------------------------------------------

def _interp_at(
    param: np.ndarray, track: np.ndarray, at: "float | np.ndarray | None"
) -> "float | np.ndarray":
    """Interpolate *track* (shape ``(n_steps,)``) at *at* parameter value(s).

    Parameters
    ----------
    at : None, scalar, or array-like
        * ``None`` → return *track* unchanged.
        * scalar float → return a single ``float``.
        * array-like → return an ``ndarray`` of the same shape.
    """
    if at is None:
        return track
    scalar_in = np.ndim(at) == 0
    at_arr = np.atleast_1d(np.asarray(at, dtype=float))
    out = np.interp(at_arr, param, track)
    return float(out[0]) if scalar_in else out


def _interpolate_crossing(
    x: np.ndarray, diff: np.ndarray, branch: int
) -> float:
    """Linear interpolation of the *branch*-th zero-crossing of *diff*."""
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        raise ValueError("No crossing found in the sweep range.")
    if branch >= len(sign_changes):
        raise ValueError(
            f"Only {len(sign_changes)} crossing(s) found; "
            f"branch={branch} is out of range."
        )
    k = sign_changes[branch]
    x0, x1 = x[k], x[k + 1]
    y0, y1 = diff[k], diff[k + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _zeeman_eigensystem(builder, B_gauss: float):
    """Lab-frame Zeeman-dressed eigensystem at *B_gauss*.

    Returns ``(E_lab, V_lab)`` — eigenvalues in Hz and eigenvector matrix
    (columns = eigenvectors in the uncoupled basis).  Unlike the vectors
    stored in a ``LaserSweepResult`` from an RWA sweep, these are NOT in the
    rotating frame and can be used directly for optical detuning calculations.
    """
    H = builder.h0()
    if B_gauss:
        H = H + builder.zeeman_operator() * B_gauss
    return diagonalize(H)


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
        """Return a human-readable label for tracked state ``i`` at ``step``.

        The label uses the **Paschen-Back (adiabatic) convention**:

        * ``(m_j, m_i)`` are the high-field limiting quantum numbers of the
          dominant uncoupled component of the eigenstate.
        * ``(F, m_F)`` are the zero-field hyperfine quantum numbers that
          adiabatically connect to those high-field numbers.

        The label is looked up from the manifold's cached
        :attr:`~.basis.Manifold.label_map`, so this method is O(1) after the
        first call and is correct for both magnetic-field and laser-intensity
        sweeps (the dressed states of a laser sweep are labelled by their
        dominant bare-state character).
        """
        s = self.dominant_state(i, step)
        man = next((m for m in self.basis.manifolds
                    if m.n == s.n and m.l == s.l and abs(m.j - s.j) < 1e-9), None)
        if man is not None:
            F, mF = man.label_for(s.m_j, s.m_i)
            return (f"|{s.n},{s.l},{s.j}; m_j={s.m_j:+.1f}, m_i={s.m_i:+.1f}> "
                    f"(F={F}, mF={mF:+d})")
        return f"|{s.n},{s.l},{s.j}; m_j={s.m_j:+.1f}, m_i={s.m_i:+.1f}>"

    def convert_label(self, state: tuple) -> tuple:
        """Convert a state 5-tuple between uncoupled and coupled-basis labels.

        Direction is detected automatically from the type of the last two
        elements:

        * ``(n, l, j, m_j: float, m_i: float)`` → ``(n, l, j, F: int, mF: int)``
        * ``(n, l, j, F: int, mF: int)`` → ``(n, l, j, m_j: float, m_i: float)``

        The mapping is **bijective**: F values are processed in ascending order
        and each claims the available ``(m_j, m_i)`` component with the largest
        CG magnitude (ties broken by most-negative ``m_j``).  This ensures that
        equal-weight cases such as ``|F=1, mF=0>`` and ``|F=2, mF=0>`` (both
        50/50 superpositions of the same two uncoupled states) get distinct
        representatives:

        * ``(F=1, mF=0)`` ↔ ``(m_j=-1/2, m_i=+1/2)``   [lower mJ → lower F]
        * ``(F=2, mF=0)`` ↔ ``(m_j=+1/2, m_i=-1/2)``   [upper mJ → upper F]

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

        # build the bijective {F: (mj, mi)} map for the relevant mF
        mF_query = (a + b) if not (isinstance(a, int) and isinstance(b, int)) else b
        rep_map = self._bijective_F_representative(n, l, j, mF_query, i_nuc)

        if isinstance(a, int) and isinstance(b, int):
            F, mF = a, b
            key = round(float(F), 9)
            if key not in rep_map:
                raise ValueError(f"|{n},{l},{j}; F={F}, mF={mF}> has no valid (m_j, m_i).")
            return (n, l, j, rep_map[key][0], rep_map[key][1])
        else:
            m_j, m_i = float(a), float(b)
            for F_key, (mj, mi) in rep_map.items():
                if abs(mj - m_j) < 1e-9 and abs(mi - m_i) < 1e-9:
                    return (n, l, j, int(round(F_key)), int(round(m_j + m_i)))
            raise ValueError(f"|{n},{l},{j}; m_j={m_j}, m_i={m_i}> is not a representative state.")

    def _bijective_F_representative(self, n, l, j, mF, i_nuc) -> dict:
        """Return ``{F: (mj, mi)}`` giving each F's unique representative
        uncoupled component for the given ``mF``.

        Uses the **Paschen-Back adiabatic-connection** convention: for a given
        mF, the valid ``(mJ, mI)`` pairs are sorted by mJ ascending and the
        valid F values are sorted ascending, then paired bijectively.

        This correctly handles states that swap their dominant uncoupled
        character across avoided crossings in the Breit-Rabi diagram — e.g.
        in K-39 (inverted hyperfine, A < 0) the F=1,mF=−1 state is
        predominantly ``(mJ=+½, mI=−3/2)`` at zero field but adiabatically
        connects to ``(mJ=−½, mI=−½)`` at high field.  The CG-dominant
        zero-field label is therefore misleading; the high-field label is used
        here.
        """
        mj_vals = sorted(-j + k for k in range(int(round(2 * j)) + 1))
        mi_vals_set = {round(-i_nuc + k, 9)
                       for k in range(int(round(2 * i_nuc)) + 1)}

        # valid (mJ, mI) pairs for this mF, ordered by mJ ascending
        valid_pairs = [(mj, mF - mj) for mj in mj_vals
                       if round(mF - mj, 9) in mi_vals_set]

        # valid F values for this mF, sorted ascending
        all_F = sorted(abs(j - i_nuc) + k
                       for k in range(int(round(2 * min(j, i_nuc))) + 1))
        valid_F = [F for F in all_F if abs(mF) <= F + 1e-9]

        # k-th F (ascending) ↔ k-th (mJ, mI) pair (mJ ascending)
        return {round(F, 9): pair for F, pair in zip(valid_F, valid_pairs)}



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
        ``Manifold``
            All tracked states in that manifold.
        ``(n, l, j)`` — 3-tuple of numbers
            All tracked states in the ``(n, l, j)`` manifold.
        ``(n, l, j, m_j, m_i)`` — 5-tuple of numbers
            The single tracked state whose dominant component is
            ``|n l j; m_j m_i>``.
        list/sequence of tuples or Manifold objects
            Each element is a ``Manifold``, ``(n, l, j)`` tuple,
            or ``(n, l, j, m_j, m_i)`` tuple; results are concatenated
            (duplicates preserved).
        sequence of int
            Explicit index list (existing behaviour).
        """
        from .basis import Manifold

        if states is None:
            return list(range(self.energies.shape[1]))

        def _is_qn_tuple(t):
            return (isinstance(t, tuple)
                    and len(t) in (3, 5)
                    and all(isinstance(s, (int, float)) for s in t))

        def _resolve_one(t):
            if isinstance(t, Manifold):
                return self.indices_for(t.n, t.l, t.j, step=step)
            elif len(t) == 3:
                n, l, j = t
                return self.indices_for(n, l, j, step=step)
            else:
                n, l, j, m_j, m_i = t
                return self.indices_for(n, l, j, m_j, m_i, step=step)

        # Handle single Manifold
        if isinstance(states, Manifold):
            return _resolve_one(states)

        # Handle single qn-tuple
        if _is_qn_tuple(states):
            return _resolve_one(states)

        # list/sequence — check whether elements are qn-tuples, Manifolds, or plain ints
        items = list(states)
        if items:
            if isinstance(items[0], Manifold) or _is_qn_tuple(items[0]):
                out = []
                for t in items:
                    out.extend(_resolve_one(t))
                return out

        return [int(i) for i in items]

    def _tracked_index(self, n: int, l: int, j: float,
                       m_j: float, m_i: float, step: int) -> int:
        """Return the tracked-state column index for the state whose
        **adiabatic (Paschen-Back) label** is ``(m_j, m_i)``.

        ``(m_j, m_i)`` is interpreted as the high-field limiting quantum
        numbers (same convention as :meth:`convert_label`): the bijective
        representative map is inverted to find ``(F, mF)`` and then the
        tracked state is located via CG overlap with ``|F, mF>``.
        """
        mF = m_j + m_i
        i_nuc = None
        for man in self.basis.manifolds:
            if man.n == n and man.l == l and abs(man.j - j) < 1e-9:
                i_nuc = man.i_nuclear
                break
        if i_nuc is None:
            raise KeyError(f"Manifold ({n}, {l}, {j}) not in basis.")
        rep_map = self._bijective_F_representative(n, l, j, mF, i_nuc)
        for F_key, (mj_rep, mi_rep) in rep_map.items():
            if abs(mj_rep - m_j) < 1e-9 and abs(mi_rep - m_i) < 1e-9:
                return self._tracked_index_F_mF(
                    n, l, j, int(round(F_key)), int(round(mF)), step)
        raise KeyError(
            f"|{n},{l},{j}; m_j={m_j}, m_i={m_i}> is not a valid "
            f"adiabatic-label representative.")

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
        for s in self.basis.state_list:
            if s.n == n and s.l == l and abs(s.j - j) < 1e-9:
                cg = _clebsch(j, s.m_j, i_nuc, s.m_i, F, mF)
                if cg:
                    psi[s.index] = cg

        # overlap of each tracked eigenvector with |F, mF>
        # vectors[step] has shape (n_basis, n_tracked); columns are eigenvectors
        overlaps = np.abs(psi @ self.vectors[step]) ** 2
        return int(np.argmax(overlaps))

    def get_energy(
        self,
        n: int, l: int, j: float, m_j: float, m_i: float,
        identify_at_step: int = 0,
        at: "float | np.ndarray | None" = None,
    ) -> "np.ndarray | float":
        """Return the energy (Hz) of the state ``|n l j; m_j m_i>``.

        The state is identified at ``identify_at_step`` (default 0, i.e. the
        start of the sweep) and then followed via eigenshuffle.

        Parameters
        ----------
        n, l, j : fine-structure quantum numbers.
        m_j, m_i : magnetic quantum numbers of J and I.
        identify_at_step : int
            Step at which to choose the tracked state by dominant uncoupled
            weight (default 0).
        at : None, float, or array-like
            * ``None`` (default) — return the full energy track, shape
              ``(n_steps,)``.
            * scalar — return a single ``float`` via linear interpolation.
            * array-like — return an ``ndarray`` interpolated at each value.

        Returns
        -------
        ndarray or float
        """
        i = self._tracked_index(n, l, j, m_j, m_i, identify_at_step)
        return _interp_at(self.param, self.energies[:, i], at)

    def get_transition_frequency(
        self,
        state_a, state_b,
        identify_at_step: int = 0,
        at: "float | np.ndarray | None" = None,
    ) -> "np.ndarray | float":
        """Return the transition frequency ``E_b − E_a`` (Hz).

        Parameters
        ----------
        state_a, state_b : tuple ``(n, l, j, m_j, m_i)``
            Quantum numbers identifying each state.
        identify_at_step : int
            Sweep step for dominant-basis identification (default 0).
        at : None, float, or array-like
            * ``None`` — return full track, shape ``(n_steps,)``.
            * scalar — single ``float`` via interpolation.
            * array-like — ``ndarray`` interpolated at each value.

        Returns
        -------
        ndarray or float
        """
        if len(state_a) != 5 or len(state_b) != 5:
            raise ValueError(
                "Each state must be a (n, l, j, m_j, m_i) 5-tuple.")
        ea = self.get_energy(*state_a, identify_at_step=identify_at_step)
        eb = self.get_energy(*state_b, identify_at_step=identify_at_step)
        track = eb - ea
        return _interp_at(self.param, track, at)

    def transition_frequency_shift(
        self,
        state_a, state_b,
        identify_at_step: int = 0,
        at: "float | np.ndarray | None" = None,
    ) -> "np.ndarray | float":
        """Net shift of the ``E_b − E_a`` transition frequency relative to the
        first sweep step (zero-field / zero-intensity value).

        Parameters
        ----------
        state_a, state_b : 5-tuples ``(n, l, j, m_j, m_i)``
        identify_at_step : int
            Step for dominant-basis identification (default 0).
        at : None, float, or array-like
            * ``None`` — return full shift track, shape ``(n_steps,)``.
            * scalar — single ``float`` via interpolation.
            * array-like — ``ndarray`` interpolated at each value.

        Returns
        -------
        ndarray or float
            Shift in Hz: ``f_transition(param) − f_transition(param[0])``.
        """
        freq = self.get_transition_frequency(
            state_a, state_b, identify_at_step=identify_at_step
        )
        shift = freq - freq[0]
        return _interp_at(self.param, shift, at)

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
             energy_offset: float = 0.0, legend: bool = False,
             plot_differential=False, **plot_kwargs):
        """Plot tracked eigen-energies versus the swept parameter.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on; a new figure/axes is created if omitted.
        energy_unit : {"Hz", "kHz", "MHz", "GHz", "THz"}
            Units for the energy axis (default MHz).
        x_unit : str, optional
            Passed to :meth:`x_axis` (e.g. ``"mW/cm^2"`` for intensity sweeps).
        states : Manifold, sequence of int, (n, l, j) tuple, (n, l, j, m_j, m_i) tuple, or list of such, optional
            Which states to plot.  Accepts:
            - ``None`` (default): all states.
            - ``Manifold`` object: all states in that manifold.
            - ``(n, l, j)`` 3-tuple: all states in that manifold.
            - ``(n, l, j, m_j, m_i)`` 5-tuple: a single state.
            - list of Manifolds and/or 3- or 5-tuples: union, concatenated.
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
        plot_differential : bool, (n, l, j) 3-tuple, or (n, l, j, m_j, m_i) 5-tuple, default False
            Subtract a reference energy from every plotted state:

            * ``False`` (default) — no subtraction.
            * ``True`` — subtract each state's own value at the *first* sweep
              step (step 0), so every line starts at zero.  Useful for seeing
              how states shift relative to their zero-field / zero-intensity
              starting energy.
            * ``(n, l, j)`` 3-tuple — subtract the **fine-structure
              centre-of-gravity** energy of that manifold at zero field
              (step 0), computed as the mean energy over all states in the
              manifold.  This removes the absolute fine-structure energy so
              that the plot shows only hyperfine and Zeeman shifts.
            * ``(n, l, j, m_j, m_i)`` 5-tuple — subtract the zero-field
              energy of the specified state from every plotted state.  Useful
              for computing differential shifts relative to a reference state
              (e.g. a clock state) across the whole sweep.
        **plot_kwargs : forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib Axes.

        Examples
        --------
        Plot all ground-manifold states in a magnetic sweep:

        >>> model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)])
        >>> res = model.magnetic_sweep(B_max=600.0)
        >>> res.plot(states=model[0])  # first manifold (4, 0, 0.5)
        >>> res.plot(states=(4, 0, 0.5))  # equivalent: by quantum numbers

        Differential plots:

        >>> res.plot(states=model[0], plot_differential=True)
        >>> res.plot(states=model[0], plot_differential=(4, 0, 0.5, -0.5, -1.5))
        """
        import matplotlib.pyplot as plt

        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9,
                 "THz": 1e12}[energy_unit]
        if ax is None:
            fig, ax = plt.subplots()

        x, xlabel = self.x_axis(x_unit)
        idxs = self._resolve_states(states, label_step)

        # --- build reference track (shape: (n_steps,) or None) ---
        ref_track: np.ndarray | None = None
        if plot_differential is True:
            pass  # handled per-state below
        elif plot_differential is not False:
            # 3-tuple (n, l, j) → fine-structure CoG at zero field (constant)
            if (isinstance(plot_differential, tuple)
                    and len(plot_differential) == 3
                    and all(isinstance(x, (int, float))
                            for x in plot_differential)):
                n_ref, l_ref, j_ref = plot_differential
                man_idxs = self.indices_for(n_ref, l_ref, float(j_ref), step=0)
                if not man_idxs:
                    raise ValueError(
                        f"No states found in manifold "
                        f"({n_ref}, {l_ref}, {j_ref}).")
                cog_hz = float(np.mean(self.energies[0, man_idxs]))
                ref_track = np.full(len(self.param), cog_hz)
            else:
                # treat as a single state specifier — resolve to one index
                ref_idxs = self._resolve_states(plot_differential, label_step)
                if len(ref_idxs) != 1:
                    raise ValueError(
                        "plot_differential state specifier must resolve to "
                        f"exactly one tracked state; got {len(ref_idxs)}.")
                ref_track = self.energies[:, ref_idxs[0]]

        y_all = (self.energies - energy_offset) / scale

        for i in idxs:
            yi = y_all[:, i].copy()
            if plot_differential is True:
                yi = yi - yi[0]
            elif ref_track is not None:
                yi = yi - ref_track[0] / scale

            lbl = self.label(i, label_step) if (label_states or legend) else None
            (line,) = ax.plot(x, yi, label=lbl, **plot_kwargs)
            if label_states:
                ax.annotate(lbl, (x[-1], yi[-1]), fontsize=6,
                            va="center", ha="left",
                            xytext=(3, 0), textcoords="offset points")

        ax.set_xlabel(xlabel)
        diff_suffix = " (differential)" if plot_differential is not False else ""
        ax.set_ylabel(f"Energy ({energy_unit}){diff_suffix}")
        if legend:
            ax.legend(fontsize=6, loc="best")
        return fig, ax


# ---------------------------------------------------------------------------
# Typed sweep-result subclasses
# ---------------------------------------------------------------------------

@dataclass
class MagneticSweepResult(SweepResult):
    """Result of a magnetic-field sweep, with B-specific inversion methods.

    Returned by :func:`sweep_field` and
    :meth:`~.model.AtomicStructure.magnetic_sweep`.
    """

    _builder: Any = field(default=None, repr=False)

    def field_from_splitting(
        self,
        state_a: tuple, state_b: tuple,
        f_measured_hz: float,
        branch: int = 0,
    ) -> float:
        """Return the field (G) where |E_b − E_a| equals *f_measured_hz*.

        Parameters
        ----------
        state_a, state_b : 5-tuples ``(n, l, j, m_j, m_i)``
        f_measured_hz : float
            Target splitting in Hz.
        branch : int
            Which zero-crossing to return (0 = first from zero field).

        Raises
        ------
        ValueError
            If no crossing is found.  Re-run
            :meth:`~.model.AtomicStructure.magnetic_sweep` with a larger
            ``B_max``.
        """
        freq = self.get_transition_frequency(state_a, state_b)
        diff = np.abs(freq) - f_measured_hz
        try:
            return _interpolate_crossing(self.param, diff, branch)
        except ValueError:
            raise ValueError(
                f"Splitting never equals {f_measured_hz / 1e6:.4f} MHz in "
                f"[0, {self.param[-1]:.1f}] G.  Re-run magnetic_sweep with a "
                f"larger B_max."
            ) from None

    def laser_sweep(
        self,
        beam,
        B_gauss: Optional[float] = None,
        I_max: Optional[float] = None,
        n_points: int = 200,
        model: str = "rwa",
        polarization: str = "pi",
        include_quadrupole: bool = True,
        polarizabilities=None,
    ) -> "LaserSweepResult":
        """Chain: run a laser-intensity sweep at a field from this sweep.

        Parameters
        ----------
        beam : GaussianBeam
        B_gauss : float, optional
            Field offset for the laser sweep (G).  Uses
            :meth:`~.diagonalize.SweepResult.field_at` to snap to the nearest
            swept step.  Defaults to the last step of this sweep.
        I_max : float, optional
            Maximum intensity (W/m²).  Defaults to ``beam.I0``.
        """
        if self._builder is None:
            raise RuntimeError(
                "No builder stored.  Use model.magnetic_sweep() to create "
                "this result."
            )
        b = self.field_at(B_gauss) if B_gauss is not None else float(self.param[-1])
        if I_max is None:
            I_max = beam.I0
        return sweep_intensity(
            self._builder, beam, I_max, n_points=n_points,
            model=model, polarization=polarization,
            B_gauss=b, include_quadrupole=include_quadrupole,
            polarizabilities=polarizabilities,
        )


@dataclass
class LaserSweepResult(SweepResult):
    """Result of a laser-intensity sweep, with spectroscopy methods.

    Returned by :func:`sweep_intensity` and
    :meth:`~.model.AtomicStructure.laser_sweep`.

    Extra attributes
    ----------------
    beam : GaussianBeam
        The laser beam used for this sweep.
    polarization : str or dict
        Beam polarization.
    B_gauss : float
        Static magnetic field offset (G) used in the sweep Hamiltonian.
    """

    beam: Any = field(default=None, repr=False)
    polarization: Any = "pi"
    B_gauss: float = 0.0
    _builder: Any = field(default=None, repr=False)

    def intensity_from_splitting_shift(
        self,
        state_a: tuple, state_b: tuple,
        df_measured_hz: float,
        branch: int = 0,
    ) -> float:
        """Return the intensity (W/m²) where the splitting shift equals *df_measured_hz*.

        The shift is ``|f_ab(I) − f_ab(0)|``.

        Parameters
        ----------
        state_a, state_b : 5-tuples ``(n, l, j, m_j, m_i)``
        df_measured_hz : float
            Target shift magnitude in Hz.
        branch : int
            Which crossing (0 = first from zero intensity).

        Raises
        ------
        ValueError
            If no crossing is found.  Re-run with a larger ``I_max``.
        """
        freq = self.get_transition_frequency(state_a, state_b)
        shift = np.abs(freq - freq[0])
        diff = shift - df_measured_hz
        try:
            return _interpolate_crossing(self.param, diff, branch)
        except ValueError:
            raise ValueError(
                f"Splitting shift never reaches {df_measured_hz / 1e3:.3f} kHz "
                f"within [0, {self.param[-1]:.3e}] W/m².  Re-run laser_sweep "
                f"with a larger I_max."
            ) from None

    def scattering_rate(
        self,
        ground_state: tuple,
        intensity_Wpm2: "float | np.ndarray",
        delta_hz: float = 0.0,
        weak_probe: bool = True,
    ) -> "float | np.ndarray":
        """Total photon scattering rate (s⁻¹) for *ground_state* at *intensity_Wpm2*.

        Sums two-level RWA contributions from every excited eigenstate in the
        basis, using the **Zeeman-dressed** (I=0) basis for transition
        frequencies and coupling elements::

            Γ_scatter = Σ_e  Γ_e · Ω_ge² / (2 · (Δ_ge² + Γ_e²/4))

        where ``Ω_ge = 2 |C_ge| E₀``, ``E₀ = sqrt(2I/(ε₀c))``, and
        ``Δ_ge = f_laser − (E_e − E_g)``.

        Parameters
        ----------
        ground_state : 5-tuple ``(n, l, j, m_j, m_i)``
        intensity_Wpm2 : float or array-like
            Laser intensity in W/m².  Returns a float when scalar, ndarray
            when array-like.
        delta_hz : float
            Additional frequency offset added to the laser frequency
            (positive = blue shift).  Use to scan detuning.
        weak_probe : bool
            If True (default), unsaturated formula.  If False, includes
            saturation: ``Ω²/2`` added to the denominator.

        Returns
        -------
        float or ndarray : scattering rate(s) in s⁻¹.
        """
        if self._builder is None:
            raise RuntimeError(
                "No builder stored.  Use model.laser_sweep() to create this result."
            )
        from kamo import constants as c

        # Re-solve the lab-frame Hamiltonian (H0 + B·Zeeman, no rotating frame).
        # self.vectors[0] from an RWA sweep is in the rotating frame and cannot
        # be used directly for computing optical detunings.
        E_lab, V_lab = _zeeman_eigensystem(self._builder, self.B_gauss)

        # Identify ground dressed state by max overlap with uncoupled ket
        n_g, l_g, j_g, mj_g, mi_g = ground_state
        g_bare = self._builder.basis.index_of(n_g, l_g, j_g, mj_g, mi_g)
        g_idx = int(np.argmax(np.abs(V_lab[g_bare, :]) ** 2))
        E_g = E_lab[g_idx]

        # RWA coupling in lab-frame Zeeman basis
        rwa = self._builder.laser_rwa_operator(
            self.beam, polarization=self.polarization
        )
        C = V_lab.conj().T @ rwa["coupling"] @ V_lab

        # Natural linewidths (per excited manifold)
        gamma_by_nlj: dict = {}
        for man in self._builder.basis.manifolds:
            if man.l > 0:
                tau = self._builder.atom.getStateLifetime(man.n, man.l, man.j)
                gamma_by_nlj[(man.n, man.l, man.j)] = 1.0 / tau

        if not gamma_by_nlj:
            raise ValueError(
                "No excited (l > 0) manifolds in the basis."
            )

        # Precompute per-transition (gamma, |C_ge|, Delta) — independent of I
        transitions: list = []
        f_laser = self.beam.frequency() + delta_hz
        for e_idx in range(len(E_lab)):
            s_dom = self._builder.basis[
                int(np.argmax(np.abs(V_lab[:, e_idx]) ** 2))
            ]
            gamma_e = gamma_by_nlj.get((s_dom.n, s_dom.l, s_dom.j))
            if gamma_e is None:
                continue
            c_abs = abs(C[g_idx, e_idx])
            if c_abs == 0.0:
                continue
            Delta = f_laser - (E_lab[e_idx] - E_g)   # lab-frame detuning
            transitions.append((gamma_e, c_abs, Delta))

        # Vectorised over intensity
        scalar_in = np.ndim(intensity_Wpm2) == 0
        I_arr = np.atleast_1d(np.asarray(intensity_Wpm2, dtype=float))
        E0_arr = np.sqrt(2.0 * I_arr / (c.epsilon0 * c.c))

        rates = np.zeros(len(I_arr))
        for gamma_e, c_abs, Delta in transitions:
            Omega = 2.0 * c_abs * E0_arr          # shape (n_I,)
            if weak_probe:
                denom = 2.0 * (Delta ** 2 + gamma_e ** 2 / 4.0)
                rates += gamma_e * Omega ** 2 / denom
            else:
                denom = 2.0 * (Delta ** 2 + gamma_e ** 2 / 4.0 + Omega ** 2 / 2.0)
                rates += gamma_e * Omega ** 2 / denom

        return float(rates[0]) if scalar_in else rates

    def dominant_couplings(
        self,
        n_top: Optional[int] = None,
    ) -> List[Tuple[tuple, tuple, float]]:
        """Sorted list of (ground_label, excited_label, |C| Hz/(V/m)) pairs.

        Coupling elements are evaluated in the **lab-frame Zeeman-dressed**
        basis so the result is frame-independent.

        Parameters
        ----------
        n_top : int, optional
            Return only the *n_top* strongest pairs (all by default).

        Returns
        -------
        list of ``(ground_5tuple, excited_5tuple, |C| [Hz/(V/m)])``

            Each label is the dominant uncoupled component
            ``(n, l, j, m_j, m_i)`` of the dressed eigenstate.
        """
        if self._builder is None:
            raise RuntimeError(
                "No builder stored.  Use model.laser_sweep() to create this result."
            )
        _, V_lab = _zeeman_eigensystem(self._builder, self.B_gauss)
        rwa = self._builder.laser_rwa_operator(
            self.beam, polarization=self.polarization
        )
        C = V_lab.conj().T @ rwa["coupling"] @ V_lab

        ground_idxs: list = []
        excited_idxs: list = []
        labels: dict = {}
        for k in range(self.energies.shape[1]):
            s = self._builder.basis[int(np.argmax(np.abs(V_lab[:, k]) ** 2))]
            labels[k] = (s.n, s.l, s.j, s.m_j, s.m_i)
            (ground_idxs if s.l == 0 else excited_idxs).append(k)

        pairs: list = []
        for g in ground_idxs:
            for e in excited_idxs:
                strength = abs(C[g, e])
                if strength > 0.0:
                    pairs.append((labels[g], labels[e], strength))

        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:n_top] if n_top is not None else pairs


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
    return MagneticSweepResult(
        B, "B (Gauss)", energies, vectors, builder.basis,
        _builder=builder,
    )


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
    return LaserSweepResult(
        I, "Intensity (W/m^2)", energies, vectors, builder.basis,
        beam=beam, polarization=polarization, B_gauss=B_gauss,
        _builder=builder,
    )
