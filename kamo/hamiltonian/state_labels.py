"""Utilities for formatting and labeling quantum states.

:func:`state_label` is the one formatter every kamo state label goes through
(sweep legends, :class:`~kamo.atom_properties.alkali.PortalAlkali`, ...).  It
converts between the uncoupled ``(m_J, m_I)`` and coupled ``(F, m_F)`` labels
with the manifold's Paschen-Back label map
(:attr:`~.basis.Manifold.label_map`), then prints signed fractions, e.g.
``$4S_{1/2}|m_J=-1/2, m_I=+3/2\\rangle$``.

Which basis a pair of magnetic quantum numbers is in is read from their
values, given the nuclear spin (:func:`is_coupled`): ``m_J`` is always a
half-integer and ``m_I`` has the parity of ``I``, while ``F`` and ``m_F`` have
the parity of ``I + 1/2``. For half-integer ``I`` (K39, Rb87, ...) that is the
familiar rule "integers are ``(F, m_F)``, half-integers ``(m_J, m_I)``"; for
integer ``I`` (Li6, K40) it is the reverse. Every label function takes
``atom=`` to say which atom's manifold is meant; the default is kamo's
default atom (39K).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

# Spectroscopic (Russell-Saunders) orbital angular-momentum letters, l=0,1,2,...
_L_LETTERS = "SPDFGHIKLMNOQRTUV"

# Manifolds already warned about having no hyperfine structure.
_WARNED_NO_HYPERFINE: set = set()


def _is_integer_valued(x: float) -> bool:
    """True if ``x`` is numerically an integer (e.g. an F quantum number)."""
    return float(x) == round(float(x))


def _qn(x: float):
    """``x`` rounded to the nearest half-integer, as an ``int`` when it is
    integer-valued (so ``(F, m_F)`` keys are ints for half-integer I, as
    they always were) and a ``float`` otherwise."""
    r = round(2 * float(x)) / 2
    return int(r) if r == int(r) else r


def _is_half(x: float) -> bool:
    return int(round(2 * float(x))) % 2 == 1


def is_coupled(a, b, I: float) -> bool:
    """True if ``(a, b)`` are coupled ``(F, m_F)`` labels for nuclear spin ``I``,
    False if they are uncoupled ``(m_J, m_I)``.

    ``m_I`` has the parity of ``I`` and ``m_F`` the parity of ``I + 1/2``, so
    the second number decides. Raises ``ValueError`` if the pair is neither
    (e.g. an integer with a half-integer for half-integer I).
    """
    a, b, I = float(a), float(b), float(I)
    for x in (a, b):
        if abs(2 * x - round(2 * x)) > 1e-9:
            raise ValueError(f"magnetic quantum numbers must be integers or "
                             f"half-integers; got {a!r}, {b!r}.")
    I_half = _is_half(I)
    if _is_half(b) == I_half:                 # b has the parity of I -> m_I
        if not _is_half(a):                   # m_J must be a half-integer
            raise ValueError(
                f"({a!r}, {b!r}) is neither (F, m_F) nor (m_J, m_I) for "
                f"I = {I:g}: m_J must be a half-integer.")
        return False
    if _is_half(a) != _is_half(b):            # F and m_F share a parity
        kind = "integers" if I_half else "half-integers"
        mi = "a half-integer" if I_half else "an integer"
        raise ValueError(
            f"({a!r}, {b!r}) is neither (F, m_F) nor (m_J, m_I) for I = {I:g}: "
            f"F and m_F must both be {kind}, or m_J a half-integer with m_I {mi}.")
    return True


def _atom_of(obj):
    """The atom an object was built for (``obj.atom``, ``obj.builder.atom`` or
    ``obj.basis.atom``), or None."""
    for path in (("atom",), ("builder", "atom"), ("basis", "atom")):
        cur = obj
        for name in path:
            cur = getattr(cur, name, None)
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


def _frac(x: float, signed: bool = True) -> str:
    """``x`` as an integer or half-integer fraction: ``-1/2``, ``+3/2``, ``0``, ``+1``.

    ``signed`` puts an explicit ``+`` on positive values (magnetic quantum
    numbers); leave it off for ``j`` and ``F``.
    """
    k = int(round(2 * float(x)))
    mag = f"{abs(k)}/2" if k % 2 else f"{abs(k) // 2}"
    if k < 0:
        return "-" + mag
    if k > 0 and signed:
        return "+" + mag
    return mag


_MANIFOLDS: dict = {}


def _manifold(n: int, l: int, j: float, atom=None):
    """Shared :class:`~.basis.Manifold` ``(n, l, j)`` of ``atom`` (validates
    ``j`` against ``l``). One cached manifold per species and ``(n, l, j)``;
    ``atom=None`` means kamo's default atom."""
    from .basis import Manifold          # basis imports this module
    from kamo.atom_properties.alkali import default_atom, species_of
    if atom is None:
        atom = default_atom()
    key = (int(n), int(l), round(float(j), 9), species_of(atom))
    if key not in _MANIFOLDS:
        _MANIFOLDS[key] = Manifold(n, l, j, atom=atom)
    return _MANIFOLDS[key]


def clear_manifold_cache():
    """Forget every cached manifold (tests, or after changing an atom's data)."""
    _MANIFOLDS.clear()


def _term(n: int, l: int, j: float, tex: bool) -> str:
    """Russell-Saunders term symbol ``nL_J``."""
    l_sym = _L_LETTERS[l] if l < len(_L_LETTERS) else f"(l={l})"
    j_str = _frac(j, signed=False)
    return rf"{n}{l_sym}_{{{j_str}}}" if tex else f"{n}{l_sym}_{j_str}"


def _ket(names: Tuple[str, ...], values: Tuple[str, ...], tex: bool) -> str:
    inner = ", ".join(f"{k}={v}" for k, v in zip(names, values))
    return rf"|{inner}\rangle" if tex else f"|{inner}⟩"


def state_label(*state, basis: Optional[str] = None, math: bool = True,
                term: bool = True, tex: bool = True, atom=None) -> str:
    r"""Label a state, converting between coupled and uncoupled numbers.

    Parameters
    ----------
    *state : (n, l, j) or (n, l, j, a) or (n, l, j, a, b)
        Positional quantum numbers, or a single 3/4/5-tuple.  The basis of
        ``a, b`` is read from their values with :func:`is_coupled`: for
        half-integer nuclear spin, integers are the coupled ``(F, m_F)`` and
        half-integers the uncoupled ``(m_J, m_I)``.
    atom : optional
        The atom whose manifold (nuclear spin, hyperfine order) the labels
        refer to; default kamo's default atom (39K).

        * ``(n, l, j)`` -- bare term symbol.
        * ``(n, l, j, F)`` or ``(n, l, j, m_J)`` -- one-number ket.
        * ``(n, l, j, F, m_F)`` or ``(n, l, j, m_J, m_I)`` -- full ket.
    basis : {None, "coupled", "uncoupled"}
        Basis of the printed ket (5-number states only).  ``None`` (default)
        keeps the input's basis.  Conversion uses the Paschen-Back adiabatic
        connection of :attr:`~.basis.Manifold.label_map`: ``(m_J, m_I)`` is
        the high-field state that the zero-field ``|F, m_F>`` turns into.
    math : bool
        Wrap the TeX label in ``$...$`` (default True).  Pass False to embed
        it in a larger math string.
    term : bool
        Include the term symbol ``nL_J`` (default True).
    tex : bool
        TeX output (default True); False gives plain text, e.g.
        ``4S_1/2|m_J=-1/2, m_I=+3/2⟩``.

    Coupled labels for a manifold with no hyperfine structure (no A constant:
    l >= 3 or a core orbital) are not physical: F is not a good quantum
    number at any B > 0.  Asking for one returns the uncoupled label instead, with a
    warning once per manifold.

    Raises
    ------
    ValueError
        Mixed integer/half-integer ``a, b``, quantum numbers not in the
        manifold, or ``basis`` given for a 3- or 4-number state.

    Examples
    --------
    >>> state_label(4, 0, 0.5)
    '$4S_{1/2}$'
    >>> state_label(4, 0, 0.5, 1, -1)
    '$4S_{1/2}|F=1, m_F=-1\\rangle$'
    >>> state_label(4, 0, 0.5, 1, -1, basis="uncoupled")
    '$4S_{1/2}|m_J=-1/2, m_I=-1/2\\rangle$'
    >>> state_label(4, 1, 1.5, 0.5, -1.5, basis="coupled")
    '$4P_{3/2}|F=3, m_F=-1\\rangle$'
    >>> state_label(4, 1, 1.5, 2)
    '$4P_{3/2}|F=2\\rangle$'
    >>> state_label((4, 0, 0.5, -0.5, 1.5), term=False, math=False)
    '|m_J=-1/2, m_I=+3/2\\rangle'
    """
    if len(state) == 1 and isinstance(state[0], (tuple, list)):
        state = tuple(state[0])
    if len(state) not in (3, 4, 5):
        raise ValueError(
            "state_label expects (n, l, j) plus 0, 1, or 2 additional "
            f"quantum numbers (3, 4, or 5 total); got {len(state)}.")
    if basis not in (None, "coupled", "uncoupled"):
        raise ValueError(
            f"basis must be None, 'coupled', or 'uncoupled'; got {basis!r}.")

    n, l, j = int(state[0]), int(state[1]), float(state[2])
    ket = ""
    if len(state) == 4:
        if basis is not None:
            raise ValueError("basis conversion needs both magnetic quantum "
                             "numbers (a 5-number state).")
        # One number: F when it is an integer, m_J otherwise. (For integer I
        # both F and m_J are half-integers; a lone half-integer reads as m_J.)
        a = state[3]
        if _is_integer_valued(a):
            ket = _ket(("F",), (_frac(a, signed=False),), tex)
        else:
            ket = _ket(("m_J",), (_frac(a),), tex)
    elif len(state) == 5:
        a, b = state[3], state[4]
        man = _manifold(n, l, j, atom=atom)
        coupled_in = is_coupled(a, b, man.i_nuclear)
        try:
            if coupled_in:
                F, mF = _qn(a), _qn(b)
                m_j, m_i = man.state_for(F, mF)
            else:
                m_j, m_i = float(a), float(b)
                F, mF = man.label_for(m_j, m_i)
        except KeyError as err:
            raise ValueError(err.args[0]) from None

        want_coupled = coupled_in if basis is None else basis == "coupled"
        if want_coupled and not man.hyperfine_resolved:
            if man.nlj not in _WARNED_NO_HYPERFINE:
                _WARNED_NO_HYPERFINE.add(man.nlj)
                warnings.warn(
                    f"{_term(n, l, j, tex=False)} has no hyperfine structure "
                    "in kamo (no A constant), so (F, m_F) is not a good "
                    "quantum number; labelling it by (m_J, m_I) instead.",
                    UserWarning, stacklevel=2)
            want_coupled = False
        if want_coupled:
            ket = _ket(("F", "m_F"), (_frac(F, signed=False), _frac(mF)), tex)
        else:
            ket = _ket(("m_J", "m_I"), (_frac(m_j), _frac(m_i)), tex)

    label = (_term(n, l, j, tex) if term else "") + ket
    return f"${label}$" if (tex and math and label) else label


def rs_state_label(*state) -> str:
    r"""Russell-Saunders term symbol plus ket, in the input's own basis.

    Same as :func:`state_label` with its defaults (no basis conversion, TeX,
    ``$``-wrapped).

    Examples
    --------
    >>> rs_state_label(4, 1, 1.5, 2, -2)
    '$4P_{3/2}|F=2, m_F=-2\\rangle$'
    >>> rs_state_label(4, 1, 1.5, 0.5, -1.5)
    '$4P_{3/2}|m_J=+1/2, m_I=-3/2\\rangle$'
    """
    return state_label(*state)


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
        Human-readable label, e.g. ``"|4,0,1/2; m_j=-1/2, m_i=+3/2>"``.
    """
    return f"|{n},{l},{_frac(j, signed=False)}; m_j={_frac(m_j)}, m_i={_frac(m_i)}>"


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
        Human-readable label, e.g. ``"|4,0,1/2; F=1, m_F=-1>"``.
    """
    return f"|{n},{l},{_frac(j, signed=False)}; F={_frac(F, signed=False)}, m_F={_frac(m_F)}>"


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
        ``"|4,0,1/2; m_j=-1/2, m_i=+3/2> (F=1, m_F=+1)"``.
    """
    uncoup = uncoupled_label(n, l, j, m_j, m_i)
    if F is not None and m_F is not None:
        return f"{uncoup} (F={_frac(F, signed=False)}, m_F={_frac(m_F)})"
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
    '|4,0,1/2; m_j=-1/2, m_i=+3/2>'
    >>> format_state(4, 0, 0.5, 1, -1)      # coupled (ints)
    '|4,0,1/2; F=1, m_F=-1>'
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
        model.state_label(4, 1, 1.5, 2, -2, basis="uncoupled")
        model.format_state(4, 0, 0.5, -0.5, 1.5)

    Each method forwards to the corresponding module-level function in
    :mod:`kamo.hamiltonian.state_labels`. :meth:`state_label` passes the
    object's atom (``self.atom``, ``self.builder.atom`` or ``self.basis.atom``)
    unless ``atom=`` is given.
    """

    def state_label(self, *state, **kwargs) -> str:
        if kwargs.get("atom") is None:
            kwargs["atom"] = _atom_of(self)
        return state_label(*state, **kwargs)
    rs_state_label = staticmethod(rs_state_label)
    uncoupled_label = staticmethod(uncoupled_label)
    coupled_label = staticmethod(coupled_label)
    both_labels = staticmethod(both_labels)
    format_state = staticmethod(format_state)
