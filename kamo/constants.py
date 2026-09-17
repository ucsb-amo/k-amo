import arc
import scipy.constants as scon
import numpy as np

# arc.Potassium39() is NOT instantiated at import time.
# It opens a SQLite database which is not safe to run concurrently across
# multiple processes (e.g. joblib worker pool).  atom_K39 and the derived
# m_K are created lazily on first access via module __getattr__ below.

c = scon.c
kB = scon.k
h = scon.h
a0 = scon.physical_constants['Bohr radius'][0]
hbar = h / 2 / np.pi
e = scon.e

m_e = scon.m_e
epsilon0 = scon.epsilon_0
epsilon_0 = scon.epsilon_0

convert_polarizability_au_to_SI = 4 * np.pi * epsilon0 * a0**3


def ac_stark_shift_J(polarizability_SI, intensity, n_medium=1.0):
    """AC Stark (dipole) potential ``U = -alpha I / (2 c eps0 n)``, in J.

    The one copy of this formula: kamo.light_shift, kamo.hamiltonian,
    GaussianBeam and kamo.trap all delegate here.  ``polarizability_SI`` is in
    C m^2/V (convert atomic units with ``convert_polarizability_au_to_SI``) and
    ``intensity`` in W/m^2.  A positive (red-detuned) polarizability gives an
    attractive, negative U.  ``n_medium`` is the refractive index: with
    ``I = c eps0 n |E|^2 / 2`` and ``U = -alpha |E|^2 / 4`` the shift carries
    ``1/n``.  Plain arithmetic, so numpy arrays and torch tensors both work.
    """
    return -polarizability_SI * intensity / (2.0 * c * epsilon0 * n_medium)

# CODATA.  Was the rounded 1.6022e-19 (14.6 ppm off), which shifted every ARC
# transition energy -- about 0.01 nm on the D lines.
convert_joules_per_electronvolt = e

#bohr magneton in J / T
mu_b = e * hbar / (2 * m_e)

# The 39K atomic facts (nuclear g-factor, g_J, hyperfine constants, ...)
# live on kamo.Potassium39 since 2026-09 (kamo.atom_properties.alkali); the
# module-level names below are kept as deprecated aliases. `m_K` stays a
# silent alias: it is read at import time by other packages, in worker
# processes, so it must not warn and must not touch ARC's database (it reads
# the class attribute).
_arc_atom_K39 = None

def _atom_K39():
    """Return the lazily-instantiated ARC Potassium39 singleton.

    Use this inside module functions: module-level ``__getattr__`` only fires on
    *attribute* access (``constants.atom_K39``), not on bare global-name lookups
    inside functions, so referencing ``atom_K39`` directly there raises
    ``NameError`` until something first accesses it as an attribute.
    """
    global _arc_atom_K39
    if _arc_atom_K39 is None:
        _arc_atom_K39 = arc.Potassium39()
    return _arc_atom_K39

_DEPRECATED_K39 = {
    # name: (attribute of kamo.Potassium39 (class-level), description)
    'g_I': ('gI', "the 39K nuclear g-factor"),
    'g_L': ('gL', "the 39K orbital g-factor 1 - m_e/M"),
    'g_J_4S': ('g_J_ground', "the measured 39K 4S_1/2 g_J"),
}


def __getattr__(name):
    if name == 'm_K':
        val = arc.Potassium39.mass          # class attribute: no database
    elif name == 'atom_K39':
        val = _atom_K39()
    elif name in _DEPRECATED_K39:
        attr, what = _DEPRECATED_K39[name]
        import warnings
        warnings.warn(
            f"kamo.constants.{name} ({what}) is deprecated; use "
            f"kamo.Potassium39.{attr} (or the same attribute of any kamo atom).",
            DeprecationWarning, stacklevel=2)
        from kamo.atom_properties.k39 import Potassium39
        return getattr(Potassium39, attr)
    else:
        raise AttributeError(f"module 'kamo.constants' has no attribute {name!r}")
    globals()[name] = val    # cache so __getattr__ is not called again
    return val


def _warn_deprecated(func, replacement):
    import warnings
    warnings.warn(f"kamo.constants.{func} is deprecated; use {replacement}.",
                  DeprecationWarning, stacklevel=3)

#hyperfine constants
def get_hyperfine_constant(l, j, iso=39, n=None):
    """Return the magnetic-dipole hyperfine A constant in Joules (A * h).

    Deprecated: use ``atom.hyperfine_constants(n, l, j)`` on a kamo atom.
    Thin wrapper over :func:`kamo.atom_properties.hyperfine.hyperfine_constants`,
    which also gives B, uncertainties and the source of each number. ARC's
    hyperfine table (Arimondo 1977) is not used.

    Args:
        l (int): orbital angular momentum quantum number.
        j (float): total angular momentum quantum number.
        iso (int): isotope mass number; 39 (default), 40, or 41.
        n (int, optional): principal quantum number. Defaults to the lowest
            valence state of that l (4s, 4p, 3d).

    Returns:
        float | None: A constant in Joules, or None if there is no value
        (l >= 3, or a core orbital such as 3s).
    """
    _warn_deprecated("get_hyperfine_constant", "atom.hyperfine_constants(n, l, j)")
    from kamo.atom_properties.hyperfine import hyperfine_constants, lowest_valence_n
    hc = hyperfine_constants(lowest_valence_n(l) if n is None else n, l, j, iso=iso)
    return h * hc.A_Hz if hc.has_A else None

# electron spin g-factor (CODATA; magnitude)
g_S = abs(scon.physical_constants['electron g factor'][0])


#total electronic g-factors
def get_total_electronic_g_factor(l, j, s=0.5, n=None):
    """Electronic g-factor g_J of 39K (positive; H_Z = mu_B B (g_J m_j + g_I m_i)).

    Deprecated: use ``atom.g_J(l, j, n=n)`` on a kamo atom.

    4S_1/2 (``l = 0`` with ``n`` 4 or None) returns the measured value. Every
    other state uses the Landé formula with g_S = 2.00231930 and g_L = 1 - m_e/M:

        g_J = g_L [J(J+1) - S(S+1) + L(L+1)] / 2J(J+1)
            + g_S [J(J+1) + S(S+1) - L(L+1)] / 2J(J+1)

    That gives 0.665875 (4P_1/2) and 1.334097 (4P_3/2). Until 2026-09 these
    were the g_S = 2 values 2/3 and 4/3, which put the 4P_3/2 m_J = -3/2 level
    0.84 MHz off at 520 G. Relativistic and QED corrections for the excited
    states are of order 1e-5 to 1e-4 and are left out.
    """
    _warn_deprecated("get_total_electronic_g_factor", "atom.g_J(l, j, n=n)")
    from kamo.atom_properties.k39 import Potassium39
    return Potassium39.g_J(Potassium39, l, j, n=n, s=s)