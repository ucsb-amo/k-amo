"""What a microscope makes of the propagated field.

The propagator stops well downstream of the cloud, where the imprinted phase has
already diffracted away.  An imaging system focused on the atoms sees the field
*at* the cloud, so the scattered part is propagated back before anything is
measured.  This module does that, then forms the two observables: the far-field
angular distribution (how much light a finite-NA lens collects) and the
phase-contrast image (what the camera records).

Quick start
-----------
>>> from kamo.imaging import readout
>>> psi = readout.refocus(result)                    # field at the cloud plane
>>> D, phi = readout.optical_depth(psi), readout.recovered_phase(psi, result.grid)
>>> image = readout.phase_contrast(psi, result.grid, NA=0.42)

The two systematics this exists to expose
-----------------------------------------
**Depth of field.**  A cloud comparable in length to ``lambda / NA^2`` cannot be
brought into focus all at once, so even a perfect, aberration-free system recovers
only a fraction of the projected phase.  Comparing :func:`recovered_phase` against
:meth:`~kamo.imaging.response.TwoLevelResponse.thin_screen` measures that fraction.
It is geometric, and it survives the weak-phase limit.

**Nonlinearity.**  The textbook inversion ``I/I0 = 1 + 2 phi`` holds only for
``|phi| << 1``.  At several radians it overstates the excursion, and the response
can stop being monotonic entirely -- so a signal maps back to more than one column
phase.  :func:`invert_signal` interpolates a real forward curve instead of
trusting the linear formula, and refuses to invert across a turning point.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import scipy.fft as _sfft

from .bpm import PropagationResult, Propagator, UniformMixture
from .grid import TransverseGrid

_FFT_WORKERS = -1


def _fft2(a):
    return _sfft.fft2(a, workers=_FFT_WORKERS)


def _ifft2(a):
    return _sfft.ifft2(a, workers=_FFT_WORKERS)


# ------------------------------------------------------------------- refocus


def refocus(result: PropagationResult, x_focus: float = 0.0) -> np.ndarray:
    """Move the transmission function from the exit plane back to ``x_focus``.

    Writing the exit-plane scattered field as ``E_vac(x_edge) (psi_exit - 1)``,
    propagating it backward by ``d = x_edge - x_focus`` with the angular-spectrum
    kernel, and renormalizing to the reference plane wave at the new plane gives

        psi(x_focus) = 1 + ifft[ fft(psi_exit - 1) exp(-i kx d) ] exp(+i k d)

    The unscattered component is a plane wave and is handled by that closing
    factor; only the scattered part needs the kernel.

    This propagates back through the cloud's own outer wings with a VACUUM kernel,
    which is deliberate: it is exactly what a real microscope does, since it can
    only collect downstream and refocuses linearly.  The depth-of-field loss that
    results is therefore physical, not an artefact -- see the module docstring.
    """
    g = result.grid
    d = result.x_edge - float(x_focus)
    back = _ifft2(_fft2(result.psi_exit - 1.0) * g.back_propagator(d))
    return 1.0 + back * np.exp(1j * g.k * d)


# -------------------------------------------------------------- exit-plane obs


def optical_depth(psi: np.ndarray, at=None) -> float:
    """On-axis optical depth ``D = -2 ln|psi|``.

    Note this is the MEASURED optical depth at the given plane, which for a
    lensing cloud is not the true absorption: refraction moves light onto or off
    the axis faster than the atoms remove it, and D can even go negative.
    """
    c = psi.shape[0] // 2 if at is None else at
    return float(-2 * np.log(np.abs(psi[c, c])))


def recovered_phase(psi: np.ndarray, grid: TransverseGrid,
                    half_width: float = 6.0e-6) -> float:
    """On-axis phase excursion of ``psi``, relative to the edge of a window.

    Unwraps the phase along a transverse cut so several-radian excursions are
    read correctly, then reports centre minus edge.  ``half_width`` must sit well
    outside the cloud so the reference point is genuinely unperturbed.
    """
    win = grid.window(half_width)
    line = np.unwrap(np.angle(psi[win, grid.center]))
    return float(line[len(line) // 2] - line[0])


# ---------------------------------------------------------------- far field


def far_field(result: PropagationResult, grid: Optional[TransverseGrid] = None,
              dipole: bool = True) -> np.ndarray:
    """Per-mode scattered power ``(k_x/k) |E~|^2``, optionally dipole-weighted.

    The ``k_x/k`` obliquity factor converts a transverse-plane amplitude into
    power flow along the propagation axis.  The dipole weight applies the vector
    radiation pattern to a scalar calculation, which is acceptable while the
    coherent light stays in a forward cone -- across NA 0.42 the weight varies by
    only 18%.
    """
    g = result.grid if grid is None else grid
    Es = _fft2(result.E_scat) * g.d**2
    W = np.where(g.propagating, (g.KX / g.k) * np.abs(Es) ** 2, 0.0)
    return W * g.dipole_weight() if dipole else W


def into_NA(W: np.ndarray, grid: TransverseGrid, NA: float) -> float:
    """Fraction of the far-field power inside a collection numerical aperture."""
    mask = grid.na_mask(NA)
    total = float(W.sum())
    return float(W[mask].sum()) / total if total > 0 else float("nan")


def born_far_field(grid: TransverseGrid, widths, N: float,
                   khat_probe=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Analytic first-Born far field of a Gaussian cloud, on the same grid.

    ``|f(q)|^2 = exp(-sum q_i^2 w_i^2 / 2)`` for ``q = k(nhat - khat)``, times the
    dipole weight and the obliquity factor.  This is the reference the propagation
    must reproduce in the weak-phase limit -- and the check that pins down the
    propagator, the dipole weighting and the ``q_x`` handling all at once.
    """
    w = np.asarray(widths, dtype=float)
    kh = np.asarray(khat_probe, dtype=float) * grid.k
    q = np.stack([grid.KX - kh[0], grid.KY - kh[1], grid.KZ - kh[2]], axis=-1)
    ff = np.exp(-((q * w) ** 2).sum(-1) / 2)
    return np.where(grid.propagating,
                    ff * grid.dipole_weight() * (grid.KX / grid.k), 0.0)



class ForwardModeScan(NamedTuple):
    """What :func:`forward_mode_scan` measured at each atom number."""

    N: np.ndarray
    """Atom numbers propagated."""
    forward: np.ndarray
    """Coherent power in the ``k_perp = 0`` far-field mode -- the forward mode."""
    total: np.ndarray
    """Total coherent far-field power, summed over every propagating mode."""
    lobe: np.ndarray
    """Coherent power inside the Born forward lobe of each cloud."""
    phi_thin: np.ndarray
    """Thin-screen peak phase, as the marker for where weak-phase ends."""
    widths: np.ndarray
    """``(n, 3)`` cloud 1/e radii -- these GROW with N, unlike a density rescale."""
    edge: np.ndarray
    """Edge-power fraction, the wraparound check, per point."""


def forward_mode_scan(response, cloud, species, N_values, *, n_grid: int = 512,
                      L_box: float = 36.0e-6, n_slices: int = 180,
                      saturate: bool = False, s0_incident: float = 0.0,
                      progress=None) -> ForwardModeScan:
    """Run the full propagation at each atom number and read the forward mode.

    The analytic ``N + N(N-1)|f(q)|^2`` says the ``q = 0`` mode grows as ``N^2``.
    This measures the same thing through the propagator instead, so the answer
    carries the lensing, the depth of field and the exact angular spectrum rather
    than a first-Born form factor.

    At each ``N`` the cloud is REBUILT at that atom number via
    :meth:`~kamo.BEC_properties.variational.GaussianVariationalCloud.with_atom_number`,
    so the widths relax as they physically would.  That is the difference between
    this and scaling the density at fixed widths: a real cloud at 8x the atom
    number is also longer and fatter, which dilutes the peak density and reopens
    the depth of field, so the two agree only in the weak-phase limit.

    The extracted observable is ``W[0, 0]``, the ``k_perp = 0`` far-field mode.
    That is the forward-scatter mode in the strict sense -- the one component
    that is collinear with, and therefore interferes with, the unscattered probe,
    which is exactly the amplitude a phase-contrast measurement reads.

    Parameters
    ----------
    response : TwoLevelResponse
    cloud : GaussianVariationalCloud
        Template; only its trap and scattering length are used.
    species : tuple
        Weighted ``(fraction, detuning)`` pairs, from ``ProbeBeam.species``.
    N_values : array_like
        Atom numbers to propagate.
    saturate : bool
        Default False, so the scan isolates the LENSING departure from ``N^2``.
        With saturation on, bleaching dominates once the cloud is dense and the
        two effects are no longer separable.
    progress : callable, optional
        Called as ``progress(i, N)`` before each propagation.

    Returns
    -------
    ForwardModeScan
    """
    N_values = np.asarray(N_values, dtype=float)
    n = N_values.size
    out = {k: np.empty(n) for k in
           ("forward", "total", "lobe", "phi_thin", "edge")}
    widths = np.empty((n, 3))

    for i, N in enumerate(N_values):
        if progress is not None:
            progress(i, float(N))
        cl = cloud.with_atom_number(float(N))
        prop = Propagator.for_cloud(response, cl, n_grid=n_grid, L_box=L_box,
                                    n_slices=n_slices)
        res = prop.propagate(UniformMixture(cl, response, species),
                             saturate=saturate, s0_incident=s0_incident)
        W = far_field(res)

        # The Born lobe closes as 1/(k w); integrating out to two 1/e half-widths
        # follows it, so the "lobe" column is a like-for-like coherent power even
        # though the cloud is a different size at every point.
        half = np.sqrt(2.0) / (res.grid.k * float(np.max(cl.widths[1:])))
        lobe = res.grid.na_mask(min(2.0 * half, 0.99))

        out["forward"][i] = float(W[0, 0])
        out["total"][i] = float(W.sum())
        out["lobe"][i] = float(W[lobe].sum())
        out["phi_thin"][i] = float(
            response.thin_screen(cl.peak_column_density, species)[1])
        out["edge"][i] = float(res.edge_power_fraction())
        widths[i] = cl.widths

    return ForwardModeScan(N=N_values, widths=widths, **out)

# -------------------------------------------------------------- phase contrast


def phase_contrast(psi: np.ndarray, grid: TransverseGrid, theta: float = np.pi / 2,
                   NA: Optional[float] = None) -> np.ndarray:
    """Phase-contrast image ``I / I0``.

    Apertures the field at the collection NA, retards the unscattered
    (``k_perp = 0``) component by ``theta``, and transforms back.  In the
    weak-phase limit with ``theta = pi/2`` this is the textbook
    ``I/I0 = 1 + 2 phi``; the point of computing it properly is how far from that
    the answer is.

    Idealizations: a perfect phase plate acting on exactly one Fourier component,
    with no finite dimple size and no aberrations.
    """
    Ek = _fft2(psi)
    if NA is not None:
        Ek = np.where(grid.na_mask(NA), Ek, 0.0)
    Ek[0, 0] *= np.exp(1j * theta)
    return np.abs(_ifft2(Ek)) ** 2


def weak_phase_image(phi_map) -> np.ndarray:
    """Textbook weak-phase phase-contrast signal ``I/I0 = 1 + 2 phi``.

    Kept for side-by-side comparison, not for use: it is unbounded below and goes
    unphysically negative once ``phi < -1/2``.
    """
    return 1.0 + 2.0 * np.asarray(phi_map)


def signal_on_axis(image: np.ndarray) -> float:
    """``I/I0 - 1`` at the centre of an image."""
    c = image.shape[0] // 2
    return float(image[c, c] - 1.0)


# ------------------------------------------------------------------ inversion


def invert_signal(signal, xi_curve: Tuple[np.ndarray, np.ndarray],
                  strict: bool = True):
    """Invert a phase-contrast signal onto spin imbalance via a forward curve.

    Parameters
    ----------
    signal : array_like
        Measured ``I/I0 - 1``.
    xi_curve : (xi, signal) arrays
        A forward calibration computed by propagating a range of imbalances --
        see :func:`imbalance_curve`.
    strict : bool
        Raise if the curve is not monotonic over its range, rather than returning
        a silently wrong branch.  The response IS non-monotonic at large ``|phi|``:
        past the turning point the lens defocuses the probe faster than the phase
        grows, so two imbalances give the same signal.

    Returns
    -------
    ndarray
        Inferred ``xi``; NaN outside the curve's range.
    """
    xi, sig = (np.asarray(a, dtype=float) for a in xi_curve)
    order = np.argsort(xi)
    xi, sig = xi[order], sig[order]

    dif = np.diff(sig)
    if not (np.all(dif > 0) or np.all(dif < 0)):
        # Split at every sign change of the slope and keep the branch containing
        # the balanced point xi = 0 -- that is the operating point, and it is the
        # only branch a near-balanced measurement can be on.
        turns = 1 + np.flatnonzero(np.sign(dif[1:]) != np.sign(dif[:-1]))
        edges = [0, *turns.tolist(), len(xi) - 1]
        branches = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        pick = min(branches, key=lambda b: abs(xi[b[0]:b[1] + 1]).min())
        msg = (f"calibration curve is not monotonic (turns near xi = "
               f"{', '.join(f'{xi[t]:+.2f}' for t in turns)}); signals in the "
               f"folded band map onto more than one imbalance")
        if strict:
            raise ValueError(
                msg + " -- pass strict=False to invert the branch through xi = 0")
        lo, hi = pick
        xi, sig = xi[lo:hi + 1], sig[lo:hi + 1]

    if sig[-1] < sig[0]:
        xi, sig = xi[::-1], sig[::-1]
    return np.interp(np.asarray(signal, dtype=float), sig, xi,
                     left=np.nan, right=np.nan)


def imbalance_curve(propagator, source_factory, xi_values, NA=None,
                    theta: float = np.pi / 2, s0_incident: float = 0.0):
    """Forward map from spin imbalance to on-axis phase-contrast signal.

    Parameters
    ----------
    propagator : Propagator
    source_factory : callable
        ``xi -> SusceptibilitySource``.
    xi_values : array_like
        Imbalances to propagate.
    NA, theta : see :func:`phase_contrast`.

    Returns
    -------
    (xi, signal, phase, depth) : tuple of ndarray
        The calibration curve plus the recovered phase and on-axis optical depth
        at each point, which is what shows *why* the signal saturates.
    """
    xi_values = np.asarray(xi_values, dtype=float)
    sig = np.empty_like(xi_values)
    phase = np.empty_like(xi_values)
    depth = np.empty_like(xi_values)
    for i, xi in enumerate(xi_values):
        res = propagator.propagate(source_factory(float(xi)), s0_incident=s0_incident)
        psi = refocus(res)
        sig[i] = signal_on_axis(phase_contrast(psi, res.grid, theta=theta, NA=NA))
        phase[i] = recovered_phase(psi, res.grid)
        depth[i] = optical_depth(psi)
    return xi_values, sig, phase, depth
