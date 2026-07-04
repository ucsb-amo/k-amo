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
convert_joules_per_electronvolt = 1.6022e-19

#bohr magneton in J / T
mu_b = e * hbar / (2 * m_e)

# Lazy singleton for the ARC atom object.  Accessing `atom_K39` or `m_K`
# triggers the first (and only) instantiation; subsequent accesses use the
# cached value stored back into the module namespace.
_arc_atom_K39 = None

def __getattr__(name):
    if name in ('atom_K39', 'm_K'):
        global _arc_atom_K39
        if _arc_atom_K39 is None:
            _arc_atom_K39 = arc.Potassium39()
        val = _arc_atom_K39 if name == 'atom_K39' else _arc_atom_K39.mass
        # Cache in module namespace so __getattr__ is not called again
        globals()[name] = val
        return val
    raise AttributeError(f"module 'kamo.constants' has no attribute {name!r}")

#K39 total nuclear g-factor
g_I = -0.00014193489

#hyperfine constants
def get_hyperfine_constant(l, j, iso=39, n=None):
    """Return the magnetic-dipole hyperfine A constant in Joules (A * h).

    Args:
        l (int): orbital angular momentum quantum number.
        j (float): total angular momentum quantum number.
        iso (int): isotope mass number; 39 (default), 40, or 41.
        n (int, optional): principal quantum number.  When supplied and no
            hardcoded value exists for the requested (l, j) pair, the function
            falls back to ``arc.Potassium39().getHFSCoefficients(n, l, j)``
            (iso=39 only).  Pass n=None to suppress the ARC fallback.

    Returns:
        float | None: A constant in Joules, or None if no data is available.
    """
    # ── hardcoded table (n-independent approximation) ─────────────────────
    if iso == 39:
        if l == 0:
            return h * 230.8598601e6
        if l == 1:
            if j == 0.5:
                return h * 27.775e6
            elif j == 1.5:
                return h * 6.093e6
    elif iso == 40:
        if l == 0:
            return h * -285.7308e6
        if l == 1:
            if j == 0.5:
                return h * -34.523e6
            elif j == 1.5:
                return h * -7.585e6
    elif iso == 41:
        if l == 0:
            return h * 127.0069352e6
        if l == 1:
            if j == 0.5:
                return h * 15.245e6
            elif j == 1.5:
                return h * 3.363e6

    # ── ARC fallback (requires n; iso=39 only) ────────────────────────────
    if n is not None and iso == 39:
        try:
            A_hz, _ = atom_K39.getHFSCoefficients(n, l, j)
            return h * A_hz
        except (ValueError, Exception):
            pass

    return None
            
#total electronic g-factors
def get_total_electronic_g_factor(l, j, s=0.5):
    # Hardcoded precise values for the principal K-39 manifolds
    if l == 0:
        return 2.00229421
    if l == 1:
        if j == 0.5:
            return 2/3
        if j == 1.5:
            return 4/3
    # Landé formula fallback for any other (l, j)
    return 1.0 + (j*(j+1) + s*(s+1) - l*(l+1)) / (2*j*(j+1))