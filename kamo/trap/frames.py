"""Vectors, frames and polarization geometry for :mod:`kamo.trap`.

Geometry convention (all of kamo.trap)
--------------------------------------
Lab frame ``(x, y, z)``, matching :mod:`kamo.imaging`: x is the default beam
propagation axis; z is the default quantization axis *and* the default vertical.
A beam's own coordinates are ``(u, v, s)`` -- ``u, v`` transverse, ``s`` the
signed distance from the waist plane along the propagation direction -- and
``(u_hat, v_hat, k_hat)`` is always right-handed.  Neither ``GaussianBeam``'s
z-along-propagation nor kamo.imaging's x-along-propagation is reused for the
beam frame, so the two can never be confused.

Polarization is a lab-frame complex 3-vector and is never resolved into a
quantization frame.  The AC Stark shift needs only two numbers, and both are
rotation invariants that can be computed directly in the lab frame
(:func:`polarization_geometry`)::

    beta  = Im(eps x eps*) . Bhat        circularity (a pseudoscalar)
    gamma = (3 |eps . Bhat|^2 - 1) / 2   alignment

Doing it this way sidesteps a real inconsistency.  :mod:`kamo.light_shift`
takes a polarization vector whose index 0 is the quantization axis, while
:mod:`kamo.imaging` puts the quantization axis along z.  The default
``eps = (0, 1, i)/sqrt(2)`` gives ``(beta, gamma) = (-1, -1/2)`` -- pure sigma+
-- with B along x, but ``(0, +1/4)`` with B along z: no vector shift at all.
Computing the invariants also avoids the handedness error a frame built from
cross products invites, since beta flips sign under a reflection.

Validation
----------
Vectors are copied, cast (polarization to complex128, directions to float64)
and normalized.  Length-2 polarizations, the zero vector, NaN/inf, complex
directions and non-transverse polarizations are rejected with a message that
says what to do instead.
"""

from __future__ import annotations

import warnings

import numpy as np

TRANSVERSE_TOL = 1e-6          #: |eps . khat| allowed after normalization
_ZERO_NORM2 = 1e-30            #: squared norm below which a vector is "zero"
_IMAG_TOL = 1e-12              #: relative imaginary part tolerated in a direction
_PARALLEL_TOL = 1e-6           #: transverse remainder below which a hint is parallel
TRANSVERSALITY_MODES = ("raise", "warn", "project", "ignore")


# ------------------------------------------------------------------ casting

def _as_3vector(v, name: str, *, polarization: bool) -> np.ndarray:
    """Copy ``v`` into a finite numeric array of shape (3,)."""
    try:
        a = np.array(v)                       # a copy: callers are never aliased
    except Exception as err:                  # ragged input, exotic objects
        raise ValueError(f"{name} could not be converted to an array: {err}") from None
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {a.dtype}")
    if a.shape != (3,):
        if polarization and a.shape == (2,):
            raise ValueError(
                f"{name} must be a lab-frame 3-vector; got a length-2 vector.  "
                "Length-2 polarizations are the kamo.light_shift convention "
                "(components relative to the quantization axis), which "
                "kamo.trap does not accept -- see kamo.trap.frames.")
        raise ValueError(f"{name} must be a 3-vector; got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf: {a}")
    return a


def as_unit_complex_vector(v, name: str = "polarization") -> np.ndarray:
    """Cast ``v`` to a normalized complex128 3-vector.

    Accepts lists, tuples and arrays, real or complex.  The global phase is
    kept: it is unobservable for a single beam but not for coherent sums.
    """
    a = _as_3vector(v, name, polarization=True).astype(np.complex128)
    n2 = float(np.sum(np.abs(a) ** 2))
    if n2 < _ZERO_NORM2:
        raise ValueError(f"{name} is the zero vector; it defines no polarization.")
    return a / np.sqrt(n2)


def as_unit_real_vector(v, name: str = "direction") -> np.ndarray:
    """Cast ``v`` to a normalized float64 3-vector.

    A complex input is accepted only if its imaginary part is round-off
    (``< 1e-12`` of its largest component); an evanescent or otherwise complex
    wavevector is not modelled.
    """
    a = _as_3vector(v, name, polarization=False)
    if np.iscomplexobj(a):
        scale = float(np.max(np.abs(a)))
        if float(np.max(np.abs(a.imag))) > _IMAG_TOL * max(scale, 1e-300):
            raise ValueError(
                f"{name} must be real; got {a}.  A complex (evanescent) "
                "direction is not modelled.")
        a = a.real
    a = a.astype(np.float64)
    n2 = float(np.dot(a, a))
    if n2 < _ZERO_NORM2:
        raise ValueError(f"{name} is the zero vector; it defines no direction.")
    return a / np.sqrt(n2)


def check_transverse(polarization, propagation_direction,
                     tol: float = TRANSVERSE_TOL, mode: str = "raise") -> np.ndarray:
    """Normalized polarization, checked to be transverse to the propagation.

    A paraxial beam's field has (to leading order) no component along ``k``.
    ``mode`` decides what happens when ``|eps . khat| > tol``:

    ``"raise"`` (default) raises ``ValueError``; ``"warn"`` warns and returns
    ``eps`` unchanged; ``"project"`` removes the longitudinal part and
    renormalizes (raising if nothing transverse is left); ``"ignore"`` returns
    ``eps`` unchanged silently.
    """
    if mode not in TRANSVERSALITY_MODES:
        raise ValueError(f"transversality must be one of {TRANSVERSALITY_MODES}; "
                         f"got {mode!r}")
    eps = as_unit_complex_vector(polarization, "polarization")
    khat = as_unit_real_vector(propagation_direction, "propagation_direction")
    along = complex(np.dot(khat, eps))
    if abs(along) <= tol or mode == "ignore":
        return eps
    msg = (f"polarization {eps} is not transverse to propagation_direction "
           f"{khat}: |eps . khat| = {abs(along):.3g} > {tol:g}.  Pass "
           "transversality='project' to remove the longitudinal part, or "
           "'warn' / 'ignore'.")
    if mode == "raise":
        raise ValueError(msg)
    if mode == "warn":
        warnings.warn(msg, UserWarning, stacklevel=2)
        return eps
    transverse = eps - khat * along                       # mode == "project"
    n2 = float(np.sum(np.abs(transverse) ** 2))
    if n2 < _ZERO_NORM2:
        raise ValueError("polarization is parallel to propagation_direction; "
                         "nothing transverse remains after projection.")
    return transverse / np.sqrt(n2)


# ------------------------------------------------------------------- frames

def orthonormal_frame(axis, first=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal triad ``(e1, e2, e3)`` with ``e3`` along ``axis``.

    With ``first`` given, ``e1`` is its component perpendicular to ``axis``
    (Gram-Schmidt) -- use this whenever the in-plane orientation matters, e.g.
    a light sheet's thin axis.  Without it, ``e1`` comes from the seed rule of
    :func:`kamo.imaging.farfield.cone_quadrature` (seed ``z`` unless ``axis`` is
    within ~26 degrees of z, then ``x``).  That rule is discontinuous in
    ``axis``: two nearly identical axes either side of the switch give in-plane
    axes rotated by ~90 degrees.  Harmless for quadrature, visible in plots.
    """
    e3 = as_unit_real_vector(axis, "axis")
    if first is None:
        seed = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(seed, e3)
    else:
        t = as_unit_real_vector(first, "first")
        e1 = t - e3 * np.dot(e3, t)
    n = float(np.linalg.norm(e1))
    if n < _PARALLEL_TOL:
        raise ValueError("`first` is parallel to `axis`; it has no transverse "
                         "component to orient the frame.")
    e1 = e1 / n
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def as_rotation_matrix(R, proper: bool = True, atol: float = 1e-10) -> np.ndarray:
    """Validate a 3x3 orthogonal matrix.  ``proper=True`` rejects reflections."""
    M = np.array(R)
    if M.shape != (3, 3) or not np.issubdtype(M.dtype, np.number):
        raise ValueError(f"rotation must be a numeric 3x3 array; got shape {M.shape}")
    if np.iscomplexobj(M):
        raise ValueError("rotation must be real.")
    M = M.astype(np.float64)
    if not np.all(np.isfinite(M)):
        raise ValueError("rotation contains NaN or inf.")
    if not np.allclose(M @ M.T, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthogonal (R R^T != I).")
    if proper and np.linalg.det(M) < 0:
        raise ValueError("rotation has det = -1 (a reflection); pass proper=False "
                         "to allow it.")
    return M


def rotation_about(axis, angle: float) -> np.ndarray:
    """Proper rotation by ``angle`` (rad, right-hand rule) about ``axis``."""
    k = as_unit_real_vector(axis, "axis")
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ------------------------------------------------------ polarization geometry

def polarization_geometry(polarization, quantization_axis) -> tuple[float, float]:
    """``(beta, gamma)`` for a lab-frame polarization about a lab-frame axis.

    ``beta = Im(eps x eps*) . Bhat`` and ``gamma = (3|eps . Bhat|^2 - 1)/2``,
    with ``eps`` normalized.  They enter the hyperfine polarizability as::

        alpha_F = alpha_s - beta mF/(2F) alpha_v
                  + gamma (3 mF^2 - F(F+1))/(F(2F-1)) alpha_t

    and agree with :func:`kamo.hamiltonian.builder._polarization_geometry`
    (spherical basis about B) and with the Cartesian convention of
    :meth:`kamo.light_shift.ComputePolarizabilities.compute_complete_polarizability`
    (index 0 = quantization axis) once those are expressed in a right-handed
    frame whose first axis is ``Bhat``.  Reference values: pi (eps || B) gives
    (0, +1); sigma+ gives (-1, -1/2); sigma- gives (+1, -1/2); any linear
    polarization perpendicular to B gives (0, -1/2).
    """
    eps = as_unit_complex_vector(polarization, "polarization")
    bhat = as_unit_real_vector(quantization_axis, "quantization_axis")
    beta = float(np.dot(np.imag(np.cross(eps, np.conj(eps))), bhat))
    gamma = float(1.5 * abs(np.dot(bhat, eps)) ** 2 - 0.5)
    return beta, gamma
