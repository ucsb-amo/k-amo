"""Fringe-effective intensity: what a Ramsey phase reports when atoms see different intensities.

A light pulse of duration ``t`` inside a Ramsey gap gives atom ``j`` the phase
``phi_j = kappa * I_j`` with ``kappa = 2 pi nu_bare t`` and ``I_j`` the intensity it sees
relative to the incident probe (``nu_bare`` is the differential light shift of an atom
in the incident beam).  The fringe summed over atoms is a cosine with

    z = < exp(i kappa I_j) >_atoms,      f_LS = arg(z) / (2 pi t),      C = |z|,

so the shift the fringe reports corresponds to an EFFECTIVE intensity

    I_eff = arg(z) / kappa,

the circular mean of the intensity distribution at the scale set by ``kappa``.  For a
narrow distribution (``kappa * sigma_I << 1``) it is the ordinary mean; atoms far out in
a heavy tail wind through many cycles, drop out of the phase and cost contrast instead.
``I_eff`` is therefore a property of the distribution AND of ``kappa`` (the shift and the
pulse length), not of the cloud alone.

Any model that supplies a distribution of intensities over atoms can be put through the
same estimator: a per-atom list (the coupled-dipole solver's driving intensities), or the
propagator's 3D intensity record weighted by the density (the mean-field lens).  Both are
handled here so the two models are compared with the same statistic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DEFAULT_WINDOW_MIN = -np.pi / 2   # same one-sided window as waxa.analysis.lightshift


def wrap_phase(phase, window_min: float = DEFAULT_WINDOW_MIN):
    """Wrap into ``[window_min, window_min + 2 pi)``."""
    return (np.asarray(phase, dtype=float) - window_min) % (2 * np.pi) + window_min


def fringe_effective(values, weights, kappa, window_min: float = DEFAULT_WINDOW_MIN
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """``(I_eff, contrast)`` of a weighted intensity distribution at phase scale ``kappa``.

    Parameters
    ----------
    values : (n,) array
        Intensities relative to the incident probe (a per-atom list or histogram bin
        centres).
    weights : (n,) array
        Atom weights (ones for a per-atom list, density times volume for a histogram).
    kappa : float or (m,) array
        ``2 pi nu_bare t`` in radians per unit intensity; several at once are allowed.
    window_min : float
        Wrap window for ``arg z``.  ``arg z`` is wrapped before dividing by ``kappa``,
        which matches how the fringe phase is read from the data.

    Returns
    -------
    I_eff, contrast : arrays of the shape of ``kappa``.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.shape != w.shape:
        raise ValueError(f"values {v.shape} and weights {w.shape} must match")
    wsum = w.sum()
    if not wsum > 0:
        raise ValueError("weights must have a positive sum")
    k = np.atleast_1d(np.asarray(kappa, dtype=float))
    z = np.exp(1j * k[:, None] * v[None, :]) @ w / wsum
    phase = wrap_phase(np.angle(z), window_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        I_eff = np.where(k != 0, phase / k, np.nan)
    out_I, out_C = I_eff.reshape(np.shape(kappa)), np.abs(z).reshape(np.shape(kappa))
    if np.ndim(kappa) == 0:
        return float(out_I), float(out_C)
    return out_I, out_C


def intensity_histogram(result, source, bins, window: Optional[slice] = None
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Density-weighted histogram of the intensity a propagation recorded in 3D.

    Parameters
    ----------
    result : PropagationResult
        From ``Propagator.propagate(..., record='3d')``; ``result.intensity_3d`` has
        shape ``(n_slices, nw, nw)`` on the transverse crop ``result.window``.
    source : SusceptibilitySource
        The medium that was propagated; ``source.density(x, Y, Z)`` gives the atom
        density on the full transverse grid at slice ``x``.
    bins : (nb + 1,) array
        Bin edges in units of the incident intensity.  Intensities above the last
        edge are counted in ``overflow``; the caller should check it is negligible.
    window : slice, optional
        Overrides ``result.window``.

    Returns
    -------
    hist : (nb,) array
        Atom weight per bin (density summed over voxels; the voxel volume is
        constant, so this is proportional to atom number).
    centres : (nb,) array
    overflow : float
        Weight above the last edge, in the same units as ``hist``.
    """
    I3 = result.intensity_3d
    if I3 is None:
        raise ValueError("propagate with record='3d' to get a 3D intensity record")
    win = result.window if window is None else window
    g = result.grid
    edges = np.asarray(bins, dtype=float)
    hist = np.zeros(edges.size - 1)
    overflow = 0.0
    for k, x in enumerate(result.x_slices):
        n = source.density(float(x), g.Y, g.Z)
        if n is None:
            continue
        n = np.asarray(n)[win, win].ravel()
        I = np.asarray(I3[k], dtype=float).ravel()
        h, _ = np.histogram(I, bins=edges, weights=n)
        hist += h
        overflow += float(n[I >= edges[-1]].sum())
    centres = 0.5 * (edges[1:] + edges[:-1])
    return hist, centres, overflow
