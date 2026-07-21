"""Utilities for formatting and labeling quantum states.

Provides helper functions to generate human-readable labels for K39 states
in both uncoupled (|n,l,j; m_j, m_i>) and coupled (|n,l,j; F, m_F>) bases.
"""

from __future__ import annotations

from typing import Tuple, Union

# Spectroscopic (Russell-Saunders) orbital angular-momentum letters, l=0,1,2,...
_L_LETTERS = "SPDFGHIKLMNOQRTUV"


def _is_integer_valued(x: float) -> bool:
    """True if ``x`` is numerically an integer (e.g. an F quantum number)."""
    return float(x) == round(float(x))


def rs_state_label(*state) -> str:
    r"""Return a Russell-Saunders (LS-coupling) term-symbol label for a state.

    Builds a label of the form ``$n L_J$`` (e.g. ``$4P_{3/2}$``) using
    spectroscopic notation for ``l`` (S, P, D, F, ...), and optionally
    appends a ket for any magnetic/hyperfine quantum numbers supplied.

    Parameters
    ----------
    *state : (n, l, j) or (n, l, j, a) or (n, l, j, a, b)
        Either three positional args ``n, l, j`` (or a single 3/4/5-tuple)
        may be given:

        * ``(n, l, j)`` -- bare term symbol, no ket appended.
        * ``(n, l, j, F)`` -- appends ``|F=F>``.
        * ``(n, l, j, m_J)`` -- appends ``|m_J=m_J>`` (``m_J`` is a
          half-integer float, distinguishing it from the integer ``F``).
        * ``(n, l, j, F, m_F)`` -- appends ``|F=F, m_F=m_F>``.
        * ``(n, l, j, m_J, m_I)`` -- appends ``|m_J=m_J, m_I=m_I>``.

        The coupled- vs. uncoupled-basis ket is auto-detected the same way
        as :func:`format_state`: an integer-valued 4th element means
        ``F``/``m_F``, a non-integer (half-integer) value means ``m_J``/``m_I``.

    Returns
    -------
    str
        LaTeX-formatted label, e.g. ``"$4P_{3/2}|F=2\\rangle$"``.

    Examples
    --------
    >>> rs_state_label(4, 0, 0.5)
    '$4S_{1/2}$'
    >>> rs_state_label(4, 1, 1.5, 2)
    '$4P_{3/2}|F=2\\rangle$'
    >>> rs_state_label(4, 1, 1.5, 0.5)
    '$4P_{3/2}|m_J=+0.5\\rangle$'
    >>> rs_state_label(4, 1, 1.5, 2, -2)
    '$4P_{3/2}|F=2, m_F=-2\\rangle$'
    >>> rs_state_label(4, 1, 1.5, 0.5, -1.5)
    '$4P_{3/2}|m_J=+0.5, m_I=-1.5\\rangle$'
    >>> rs_state_label((4, 0, 0.5, 1, -1))    # tuple form also accepted
    '$4S_{1/2}|F=1, m_F=-1\\rangle$'
    """
    if len(state) == 1 and isinstance(state[0], (tuple, list)):
        state = tuple(state[0])

    if len(state) not in (3, 4, 5):
        raise ValueError(
            "rs_state_label expects (n, l, j) plus 0, 1, or 2 additional "
            f"quantum numbers (3, 4, or 5 total); got {len(state)}."
        )

    n, l, j = int(state[0]), int(state[1]), float(state[2])
    l_sym = _L_LETTERS[l] if l < len(_L_LETTERS) else f"l={l}"
    j2 = int(round(2 * j))
    j_str = f"{j2}/2" if (j2 % 2) else f"{j2 // 2}"
    term = rf'{n}{l_sym}_{{{j_str}}}'

    if len(state) == 3:
        return rf'${term}$'

    if len(state) == 4:
        a = state[3]
        if _is_integer_valued(a):
            ket = rf'|F={int(round(a))}\rangle'
        else:
            ket = rf'|m_J={float(a):+.1f}\rangle'
        return rf'${term}{ket}$'

    # len(state) == 5
    a, b = state[3], state[4]
    if _is_integer_valued(a):
        ket = rf'|F={int(round(a))}, m_F={int(round(b)):+d}\rangle'
    else:
        ket = rf'|m_J={float(a):+.1f}, m_I={float(b):+.1f}\rangle'
    return rf'${term}{ket}$'


def uncoupled_label(n: int, l: int, j: float, m_j: float, m_i: float) -> str:
    r"""Format an uncoupled basis state |n, l, j; m_j, m_i> as a string.

    Parameters
    ----------
    n, l, j : int, int, float
        Principal, orbital, and total angular momentum quantum numbers.
    m_j, m_i : float, float
        Magnetic quantum numbers of J and nuclear spin I (I=3/2 for K39).

    Returns
    -------
    str
        Human-readable label, e.g. ``"|4,0,0.5; m_j=-0.5, m_i=+1.5>"``.
    """
    return f"|{n},{l},{j}; m_j={m_j:+.1f}, m_i={m_i:+.1f}>"


def coupled_label(n: int, l: int, j: float, F: int, m_F: int) -> str:
    r"""Format a coupled basis state |n, l, j; F, m_F> as a string.

    Parameters
    ----------
    n, l, j : int, int, float
        Principal, orbital, and total angular momentum quantum numbers.
    F, m_F : int, int
        Total angular momentum (with nuclear spin) and its magnetic quantum number.

    Returns
    -------
    str
        Human-readable label, e.g. ``"|4,0,0.5; F=1, m_F=-1>"``.
    """
    return f"|{n},{l},{j}; F={F}, m_F={m_F:+d}>"


def both_labels(n: int, l: int, j: float, m_j: float, m_i: float,
                F: int | None = None, m_F: int | None = None) -> str:
    r"""Format a state showing both uncoupled and coupled quantum numbers.

    Useful for displaying the connection between the two bases, especially
    in avoided-crossing diagrams.

    Parameters
    ----------
    n, l, j : int, int, float
        Principal, orbital, and total angular momentum quantum numbers.
    m_j, m_i : float, float
        Uncoupled magnetic quantum numbers.
    F, m_F : int | None, int | None
        Coupled quantum numbers (optional). If None, only uncoupled label shown.

    Returns
    -------
    str
        Label combining both representations, e.g.
        ``"|4,0,0.5; m_j=-0.5, m_i=+1.5> (F=1, m_F=-1)"``.
    """
    uncoup = uncoupled_label(n, l, j, m_j, m_i)
    if F is not None and m_F is not None:
        coup = f"F={F}, m_F={m_F:+d}"
        return f"{uncoup} ({coup})"
    return uncoup


def format_state(
    n: int,
    l: int,
    j: float,
    a: Union[float, int],
    b: Union[float, int],
    basis_type: str = "auto",
) -> str:
    r"""Format a quantum state label, auto-detecting the basis.

    Parameters
    ----------
    n, l, j : int, int, float
        Quantum numbers of the manifold.
    a, b : float or int
        Last two quantum numbers. Type is used to detect the basis:
        * Both float → uncoupled basis (m_j, m_i)
        * Both int → coupled basis (F, m_F)
        * If ambiguous, use ``basis_type`` parameter.
    basis_type : {"auto", "uncoupled", "coupled"}
        Explicit basis selection if ``a, b`` types are ambiguous.

    Returns
    -------
    str
        Formatted label string.

    Raises
    ------
    ValueError
        If basis cannot be determined.

    Examples
    --------
    >>> format_state(4, 0, 0.5, -0.5, 1.5)  # uncoupled (floats)
    '|4,0,0.5; m_j=-0.5, m_i=+1.5>'
    >>> format_state(4, 0, 0.5, 1, -1)      # coupled (ints)
    '|4,0,0.5; F=1, m_F=-1>'
    """
    # Auto-detect basis from types if not explicitly specified
    if basis_type == "auto":
        # If either a or b is a float, treat as uncoupled
        if isinstance(a, float) or isinstance(b, float):
            basis_type = "uncoupled"
        elif isinstance(a, int) and isinstance(b, int):
            basis_type = "coupled"
        else:
            raise ValueError(
                "Cannot auto-detect basis from types. "
                "Use basis_type='uncoupled' or 'coupled' explicitly."
            )

    if basis_type == "uncoupled":
        return uncoupled_label(n, l, j, float(a), float(b))
    elif basis_type == "coupled":
        return coupled_label(n, l, j, int(a), int(b))
    else:
        raise ValueError(
            f"basis_type must be 'auto', 'uncoupled', or 'coupled'; "
            f"got {basis_type!r}"
        )


class StateLabelMixin:
    """Mixin giving any class instance-method access to the label helpers.

    Mix this into classes that represent or hold quantum states (manifolds,
    bases, sweep results, the top-level :class:`~.model.AtomicStructure`, ...)
    so that formatting a label doesn't require a separate module import::

        model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)])
        model.rs_state_label(4, 1, 1.5, 2)      # '$4P_{3/2}|F=2\\rangle$'
        model.format_state(4, 0, 0.5, -0.5, 1.5)

    Each method simply forwards to the corresponding module-level function
    in :mod:`kamo.hamiltonian.state_labels`; ``self`` is unused.
    """

    rs_state_label = staticmethod(rs_state_label)
    uncoupled_label = staticmethod(uncoupled_label)
    coupled_label = staticmethod(coupled_label)
    both_labels = staticmethod(both_labels)
    format_state = staticmethod(format_state)
