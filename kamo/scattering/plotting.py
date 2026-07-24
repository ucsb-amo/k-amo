"""Plotting helpers for K39 scattering lengths (matplotlib, waxa conventions)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def plot_scattering_length(model, channels: List[Tuple[tuple, tuple]],
                           B_gauss: np.ndarray, ax=None,
                           show_imag: bool = True, ylim_a0=None):
    """Plot a(B) for one or more channels.

    Parameters
    ----------
    model : ScatteringModel
    channels : list of ((F,mF),(F,mF)) pair channels.
    B_gauss : array of fields (Gauss).
    show_imag : also plot -Im(a) (loss) as a dashed line where nonzero.
    ylim_a0 : optional (lo, hi) clip for the y-axis (a0).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure

    B = np.asarray(B_gauss, dtype=float)
    for (a, b) in channels:
        try:
            avals = np.asarray(model.scattering_length(a, b, B), dtype=complex)
        except KeyError:
            continue
        lbl = f"|{a[0]},{a[1]:+d}>+|{b[0]},{b[1]:+d}>"
        (line,) = ax.plot(B, avals.real, label=lbl)
        if show_imag and np.any(np.abs(avals.imag) > 1e-9):
            ax.plot(B, -avals.imag, ls="--", color=line.get_color(), alpha=0.7,
                    label=f"{lbl}  -Im(a)")

    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("B (Gauss)")
    ax.set_ylabel(r"scattering length ($a_0$)")
    if ylim_a0 is not None:
        ax.set_ylim(*ylim_a0)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("K39 s-wave scattering length (Etrych 2023 / Chapurin 2019)")
    return fig, ax
