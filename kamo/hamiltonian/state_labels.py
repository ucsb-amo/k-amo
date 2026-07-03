"""Utilities for formatting and labeling quantum states.

Provides helper functions to generate human-readable labels for K39 states
in both uncoupled (|n,l,j; m_j, m_i>) and coupled (|n,l,j; F, m_F>) bases.
"""

from __future__ import annotations

from typing import Tuple, Union


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
