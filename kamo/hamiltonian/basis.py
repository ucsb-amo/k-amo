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

        self.states: List[BasisState] = []
        idx = 0
        for man in self.manifolds:
            for (m_j, m_i) in man.substates():
                self.states.append(BasisState(man.n, man.l, man.j, m_j, m_i, idx))
                idx += 1

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def __getitem__(self, i: int) -> BasisState:
        return self.states[i]

    @property
    def dim(self) -> int:
        return len(self.states)

    # -- lookups ------------------------------------------------------------
    def index_of(self, n: int, l: int, j: float, m_j: float, m_i: float) -> int:
        """Index of the state with the given quantum numbers."""
        for s in self.states:
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

    def m_f_values(self) -> np.ndarray:
        return np.array([s.m_f for s in self.states])

    def __repr__(self) -> str:
        return f"Basis(dim={self.dim}, manifolds={self.manifolds})"
