"""Basis construction for multi-manifold K39 structure calculations.

The basis is the *uncoupled* fine-structure + nuclear-spin basis

    |n, l, j; m_j, m_i>

with the K39 nuclear spin I = 3/2.  This basis is the natural one for building
Zeeman Hamiltonians valid at arbitrary field, and the zero-field eigenstates of
the hyperfine Hamiltonian recover the |F, m_F> states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# K39 nuclear spin
I_NUCLEAR = 1.5


def _half_integer_range(spin: float) -> List[float]:
    """Return [-spin, ..., spin] in integer steps (works for half-integers)."""
    n = int(round(2 * spin)) + 1
    return [(-spin + k) for k in range(n)]


@dataclass(frozen=True)
class BasisState:
    """A single uncoupled basis state |n l j; m_j m_i>.

    Attributes
    ----------
    n, l, j : quantum numbers of the fine-structure manifold.
    m_j, m_i : magnetic quantum numbers of J and I.
    index : position of the state in the owning :class:`Basis`.
    """

    n: int
    l: int
    j: float
    m_j: float
    m_i: float
    index: int = -1

    @property
    def m_f(self) -> float:
        """Total magnetic quantum number m_F = m_j + m_i (a good quantum number)."""
        return self.m_j + self.m_i

    @property
    def nlj(self) -> Tuple[int, int, float]:
        return (self.n, self.l, self.j)

    def __repr__(self) -> str:
        return (f"|{self.n},{self.l},{self.j}; m_j={self.m_j:+.1f}, "
                f"m_i={self.m_i:+.1f}>")


class Manifold:
    """A single (n, l, j) fine-structure manifold of K39.

    Enumerates all uncoupled sublevels |m_j, m_i> with I = 3/2.
    """

    def __init__(self, n: int, l: int, j: float, i_nuclear: float = I_NUCLEAR):
        if abs(j - (l + 0.5)) > 1e-9 and abs(j - abs(l - 0.5)) > 1e-9:
            raise ValueError(f"j={j} is not compatible with l={l} (need j = l +/- 1/2).")
        self.n = int(n)
        self.l = int(l)
        self.j = float(j)
        self.i_nuclear = float(i_nuclear)
        # Lazily built by _build_label_cache()
        self._label_cache: dict | None = None      # (mj, mi) -> (F, mF)
        self._label_cache_rev: dict | None = None  # (F, mF) -> (mj, mi)

    @property
    def nlj(self) -> Tuple[int, int, float]:
        return (self.n, self.l, self.j)

    @property
    def dim(self) -> int:
        return int(round((2 * self.j + 1) * (2 * self.i_nuclear + 1)))

    def substates(self) -> List[Tuple[float, float]]:
        """All (m_j, m_i) pairs, ordered by (m_j, m_i)."""
        pairs = []
        for m_j in _half_integer_range(self.j):
            for m_i in _half_integer_range(self.i_nuclear):
                pairs.append((m_j, m_i))
        return pairs

    def allowed_F(self) -> List[float]:
        """|j - I| .. j + I."""
        f_min = abs(self.j - self.i_nuclear)
        f_max = self.j + self.i_nuclear
        n = int(round(f_max - f_min)) + 1
        return [f_min + k for k in range(n)]

    def states(self, *, F=None, mF=None, mJ=None, mI=None
                   ) -> List[Tuple[int, int, float, float, float]]:
        """Return states in this manifold, optionally filtered by quantum numbers.

        All keyword arguments are optional and combinable.  All supplied filters
        must match simultaneously — a state failing any single filter is excluded.

        Parameters
        ----------
        F : int or float, optional
            Total angular momentum F = j + I.  Must lie in ``allowed_F()``.
        mF : float, optional
            Total magnetic quantum number m_F = m_j + m_i.  Range: [-F, F]
            if F is also given, otherwise [-j-I, j+I].
            Incompatible with simultaneously specifying mJ and mI (since those
            already fix m_F = mJ + mI).
        mJ : float, optional
            Magnetic quantum number of J.  Range: [-j, j].
        mI : float, optional
            Magnetic quantum number of nuclear spin I.  Range: [-I, I].

        Returns
        -------
        list of (n, l, j, m_j, m_i) 5-tuples

        Raises
        ------
        ValueError
            If a supplied value is out of range for this manifold, if mF is
            given together with both mJ and mI (over-constrained), or if F is
            not in ``allowed_F()``.

        Examples
        --------
        >>> gs = model[0]
        >>> gs.states()                        # all 8 states
        >>> gs.states(mJ=-0.5)                 # m_j = -1/2 sector (4 states)
        >>> gs.states(mI=1.5)                  # m_i = +3/2 (2 states)
        >>> gs.states(mF=0.0)                  # m_F = 0 (2 states)
        >>> gs.states(F=2)                     # states with F=2 label (5 states)
        >>> gs.states(F=1, mF=-1)              # single (F=1, mF=-1) state
        >>> gs.states(mJ=0.5, mI=-0.5)         # single uncoupled state
        >>> gs.states(mJ=0.5, mF=0.0)          # mJ + mI = 0 and mJ = 0.5 -> mI = -0.5
        >>> res.plot(states=gs.states(mF=0.0)) # pass directly to plot
        """
        # ---- validate individual values against manifold ranges ----
        valid_mj = set(round(v, 9) for v in _half_integer_range(self.j))
        valid_mi = set(round(v, 9) for v in _half_integer_range(self.i_nuclear))
        valid_F  = set(round(v, 9) for v in self.allowed_F())
        mF_range = self.j + self.i_nuclear  # max |m_F|

        if mJ is not None and round(float(mJ), 9) not in valid_mj:
            raise ValueError(
                f"mJ={mJ} is not valid for j={self.j}; "
                f"allowed: {sorted(valid_mj)}")
        if mI is not None and round(float(mI), 9) not in valid_mi:
            raise ValueError(
                f"mI={mI} is not valid for I={self.i_nuclear}; "
                f"allowed: {sorted(valid_mi)}")
        if F is not None and round(float(F), 9) not in valid_F:
            raise ValueError(
                f"F={F} is not valid for this manifold (j={self.j}, I={self.i_nuclear}); "
                f"allowed: {sorted(valid_F)}")
        if mF is not None and abs(float(mF)) > mF_range + 1e-9:
            raise ValueError(
                f"mF={mF} is out of range for this manifold; "
                f"|mF| must be <= {mF_range}")
        if F is not None and mF is not None and abs(float(mF)) > float(F) + 1e-9:
            raise ValueError(
                f"mF={mF} is out of range for F={F}; |mF| must be <= F")

        # ---- over-constrained check: mJ + mI already fix mF ----
        if mF is not None and mJ is not None and mI is not None:
            implied = round(float(mJ) + float(mI), 9)
            if abs(implied - float(mF)) > 1e-9:
                raise ValueError(
                    f"mJ={mJ} + mI={mI} = {implied} is inconsistent with mF={mF}")
            # redundant but harmless — drop mF and proceed via mJ+mI
            mF = None

        # ---- build the set of (m_j, m_i) that satisfy the F [AND mF] filter ----
        # Use the bijective Paschen-Back representative convention.
        # AND logic: when mF is also specified we only build the representative
        # for that specific mF, not for every mF in F's range.  When mF is not
        # specified, we collect representatives for all mF values of F.
        f_pairs: set | None = None
        if F is not None:
            f_pairs = set()
            F_float = float(F)
            all_F = sorted(self.allowed_F())
            mj_vals = sorted(-self.j + k for k in range(int(round(2 * self.j)) + 1))
            mi_set = set(round(-self.i_nuclear + k, 9)
                         for k in range(int(round(2 * self.i_nuclear)) + 1))
            # AND: if mF is specified, restrict to just that mF value
            mF_values = ([float(mF)] if mF is not None
                         else list(_half_integer_range(F_float)))
            for mF_v in mF_values:
                valid_pairs = [(mj, mF_v - mj) for mj in mj_vals
                               if round(mF_v - mj, 9) in mi_set]
                valid_F_for_mF = [f for f in all_F if abs(mF_v) <= f + 1e-9]
                # k-th F (ascending) <-> k-th (mj, mi) pair (mj ascending)
                rep_map = {round(f, 9): pair
                           for f, pair in zip(valid_F_for_mF, valid_pairs)}
                key = round(F_float, 9)
                if key in rep_map:
                    mj_r, mi_r = rep_map[key]
                    f_pairs.add((round(mj_r, 9), round(mi_r, 9)))

        # ---- filter all substates (all active conditions must pass = AND) ----
        result = []
        for m_j, m_i in self.substates():
            m_j_r = round(m_j, 9)
            m_i_r = round(m_i, 9)
            if mJ is not None and abs(m_j_r - round(float(mJ), 9)) > 1e-9:
                continue
            if mI is not None and abs(m_i_r - round(float(mI), 9)) > 1e-9:
                continue
            if mF is not None and abs(round(m_j + m_i, 9) - round(float(mF), 9)) > 1e-9:
                continue
            if f_pairs is not None and (m_j_r, m_i_r) not in f_pairs:
                continue
            result.append((self.n, self.l, self.j, m_j, m_i))
        return result

    # -- Paschen-Back label map --------------------------------------------
    def _build_label_cache(self) -> None:
        """Build the bijective Paschen-Back (mj, mi) <-> (F, mF) label map.

        For each total magnetic quantum number m_F, the valid (m_j, m_i)
        pairs are sorted by m_j ascending and the valid F values are sorted
        ascending, then paired bijectively.  This gives the adiabatic
        connection between the high-field Paschen-Back states and the
        zero-field hyperfine (F, m_F) states.
        """
        fwd: dict = {}   # (mj, mi) -> (F, mF)
        rev: dict = {}   # (F, mF) -> (mj, mi)

        all_F = sorted(self.allowed_F())
        mj_vals = sorted(-self.j + k for k in range(int(round(2 * self.j)) + 1))
        mi_set = {round(-self.i_nuclear + k, 9)
                  for k in range(int(round(2 * self.i_nuclear)) + 1)}

        # all possible m_F = m_j + m_i
        n_mF = int(round(2 * (self.j + self.i_nuclear))) + 1
        all_mF = [-self.j - self.i_nuclear + k for k in range(n_mF)]

        for mF_v in all_mF:
            # (mj, mi) pairs with the right m_F, ordered by mj ascending
            valid_pairs = []
            for mj in mj_vals:
                mi = round(mF_v - mj, 9)
                if mi in mi_set:
                    valid_pairs.append((mj, mi))
            # F values that can have this m_F, ordered ascending
            valid_F = [f for f in all_F if abs(mF_v) <= f + 1e-9]
            # k-th F (ascending) <-> k-th (mj, mi) pair (mj ascending)
            for F_v, (mj_r, mi_r) in zip(valid_F, valid_pairs):
                key_fwd = (round(mj_r, 9), mi_r)
                key_rev = (int(round(F_v)), int(round(mF_v)))
                fwd[key_fwd] = key_rev
                rev[key_rev] = key_fwd

        self._label_cache = fwd
        self._label_cache_rev = rev

    @property
    def label_map(self) -> dict:
        """Bijective map ``{(mj, mi): (F, mF)}`` (Paschen-Back adiabatic connection).

        Built on first access; subsequent accesses are O(1).
        """
        if self._label_cache is None:
            self._build_label_cache()
        return self._label_cache

    @property
    def reverse_label_map(self) -> dict:
        """Bijective map ``{(F, mF): (mj, mi)}`` (inverse of :attr:`label_map`)."""
        if self._label_cache_rev is None:
            self._build_label_cache()
        return self._label_cache_rev

    def label_for(self, m_j: float, m_i: float) -> tuple:
        """Return ``(F, mF)`` for the state that adiabatically connects to
        the high-field Paschen-Back state ``|m_j, m_i>``.

        Parameters
        ----------
        m_j, m_i : float
            Uncoupled magnetic quantum numbers.

        Returns
        -------
        (F, mF) : (int, int)
        """
        key = (round(float(m_j), 9), round(float(m_i), 9))
        try:
            return self.label_map[key]
        except KeyError:
            raise KeyError(
                f"|{self.n},{self.l},{self.j}; m_j={m_j}, m_i={m_i}> "
                f"not found in label map for this manifold.")

    def state_for(self, F: int, mF: int) -> tuple:
        """Return ``(mj, mi)`` for the high-field Paschen-Back state that
        adiabatically connects to the zero-field state ``|F, mF>``.

        Parameters
        ----------
        F, mF : int
            Coupled-basis quantum numbers.

        Returns
        -------
        (mj, mi) : (float, float)
        """
        key = (int(round(F)), int(round(mF)))
        try:
            return self.reverse_label_map[key]
        except KeyError:
            raise KeyError(
                f"|{self.n},{self.l},{self.j}; F={F}, mF={mF}> "
                f"not found in label map for this manifold.")

    def __repr__(self) -> str:
        return f"Manifold(n={self.n}, l={self.l}, j={self.j}, dim={self.dim})"


class Basis:
    """An ordered collection of (n, l, j) manifolds and their sublevels.

    Parameters
    ----------
    manifolds : iterable of Manifold or (n, l, j) tuples.

    The basis states are ordered manifold-by-manifold in the order provided.
    """

    def __init__(self, manifolds: Iterable):
        self.manifolds: List[Manifold] = []
        for m in manifolds:
            if isinstance(m, Manifold):
                self.manifolds.append(m)
            else:
                n, l, j = m
                self.manifolds.append(Manifold(n, l, j))

        # Reject duplicate manifolds (would make the basis singular).
        seen = set()
        for man in self.manifolds:
            key = (man.n, man.l, man.j)
            if key in seen:
                raise ValueError(f"Duplicate manifold {key} in basis.")
            seen.add(key)

        self._states: List[BasisState] = []
        idx = 0
        for man in self.manifolds:
            for (m_j, m_i) in man.substates():
                self._states.append(BasisState(man.n, man.l, man.j, m_j, m_i, idx))
                idx += 1

    # -- container protocol -------------------------------------------------
    @property
    def state_list(self) -> List[BasisState]:
        """Flat list of all :class:`BasisState` objects in basis order."""
        return self._states

    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self):
        return iter(self._states)

    def __getitem__(self, key):
        """Access a basis state or manifold.

        Parameters
        ----------
        key : int or tuple
            - int: index into the basis states; returns a BasisState.
            - 3-tuple (n, l, j): returns the Manifold with those quantum numbers.

        Returns
        -------
        BasisState or Manifold

        Examples
        --------
        >>> basis[0]                    # first basis state
        >>> basis[(4, 0, 0.5)]          # manifold by (n, l, j)
        """
        if isinstance(key, int):
            return self._states[key]
        elif isinstance(key, tuple) and len(key) == 3:
            n, l, j = key
            for man in self.manifolds:
                if man.n == n and man.l == l and abs(man.j - j) < 1e-9:
                    return man
            raise KeyError(f"Manifold {key} not in basis.")
        else:
            raise TypeError(
                "Basis indices must be int (for states) or "
                "3-tuple (n, l, j) (for manifolds)."
            )

    @property
    def dim(self) -> int:
        return len(self._states)

    # -- lookups ------------------------------------------------------------
    def index_of(self, n: int, l: int, j: float, m_j: float, m_i: float) -> int:
        """Index of the state with the given quantum numbers."""
        for s in self._states:
            if (s.n == n and s.l == l and abs(s.j - j) < 1e-9
                    and abs(s.m_j - m_j) < 1e-9 and abs(s.m_i - m_i) < 1e-9):
                return s.index
        raise KeyError(f"State |{n},{l},{j}; {m_j},{m_i}> not in basis.")

    def manifold_slices(self):
        """Yield (Manifold, slice) giving the block of each manifold."""
        start = 0
        for man in self.manifolds:
            yield man, slice(start, start + man.dim)
            start += man.dim

    def manifold_by_index(self, idx: int) -> Manifold:
        """Return the manifold at the given index (0-based, ordered by appearance).

        Parameters
        ----------
        idx : int
            Index into the manifolds list.

        Returns
        -------
        Manifold
        """
        return self.manifolds[idx]

    def states(self, n=None, l=None, j=None, F=None, mF=None, mJ=None, mI=None
               ) -> List[Tuple[int, int, float, float, float]]:
        """Return states in the basis, with optional filtering.

        Manifold filters ``n``, ``l``, ``j`` restrict which manifolds are
        searched; state-level filters ``F``, ``mF``, ``mJ``, ``mI`` are
        forwarded to each matching manifold's own
        :meth:`~Manifold.states` method.

        Parameters
        ----------
        n : int, optional
            Principal quantum number.
        l : int, optional
            Orbital angular momentum quantum number.
        j : float, optional
            Total angular momentum quantum number.
        F, mF, mJ, mI : optional
            Forwarded to :meth:`Manifold.states` for sub-manifold filtering.
            See that method for full documentation.

        Returns
        -------
        list of (n, l, j, m_j, m_i) 5-tuples

        Examples
        --------
        >>> basis.states()                       # all states in every manifold
        >>> basis.states(n=4, l=0)               # ground manifold states
        >>> basis.states(j=1.5)                  # all j=3/2 manifolds
        >>> basis.states(n=4, l=0, mJ=-0.5)      # ground manifold, mJ=-1/2
        >>> basis.states(F=2, mF=0)              # F=2, mF=0 across all manifolds
        """
        result = []
        for man in self.manifolds:
            if n is not None and man.n != int(n):
                continue
            if l is not None and man.l != int(l):
                continue
            if j is not None and abs(man.j - float(j)) > 1e-9:
                continue
            result.extend(man.states(F=F, mF=mF, mJ=mJ, mI=mI))
        return result

        return np.array([s.m_f for s in self.states])

    def __repr__(self) -> str:
        return f"Basis(dim={self.dim}, manifolds={self.manifolds})"
