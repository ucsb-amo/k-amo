"""Demo: K39 s-wave a(B) for the b-field-decoherence target channels.

Run:  python -m kamo.scattering.examples.demo_target_channels

Shows real (elastic) a(B) for |1,-1>+|1,-1>, |1,0>+|1,0>, |1,-1>+|1,0> and the
differential a between the |1,-1> and |1,0> intra-channels (the quantity that
drives density-dependent differential dephasing).  Parameters are PROVISIONAL.
"""

from __future__ import annotations

import numpy as np


def main(save_path: str | None = None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kamo.scattering import ScatteringModel
    from kamo.scattering.plotting import plot_scattering_length

    m = ScatteringModel(B_max=300.0, dB=0.05)
    B = np.linspace(20, 250, 2000)
    channels = [((1, -1), (1, -1)), ((1, 0), (1, 0)), ((1, -1), (1, 0))]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), layout="constrained",
                                   sharex=True)
    plot_scattering_length(m, channels, B, ax=ax0, ylim_a0=(-200, 200))

    a_m1 = np.real(m.intra((1, -1), B))
    a_0 = np.real(m.intra((1, 0), B))
    ax1.plot(B, a_m1 - a_0, color="crimson")
    ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax1.set_xlabel("B (Gauss)")
    ax1.set_ylabel(r"$a_{|1,-1\rangle} - a_{|1,0\rangle}$  ($a_0$)")
    ax1.set_ylim(-200, 200)
    ax1.set_title("differential elastic scattering length (dephasing driver)")

    # report elastic/lossy status
    for ent in channels:
        lossy = any(m.is_lossy(*ent, b) for b in (30, 60, 120, 200))
        print(f"{ent}: two-body lossy at threshold? {lossy}")

    if save_path:
        fig.savefig(save_path, dpi=130)
        print(f"saved {save_path}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
