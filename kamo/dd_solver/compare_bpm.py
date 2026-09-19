"""A/B validation against kamo.imaging's smooth-susceptibility propagation.

One density profile, one incident beam, two codes:

1. :class:`kamo.imaging.Propagator` (split-step angular spectrum, independent-atom
   susceptibility ``chi = n alpha / eps0``) on the profile;
2. this package's microscopic solver on configurations sampled from the same
   profile, with the coherent (configuration-averaged) field reconstructed on
   the propagator's exit plane.

They are compared on the **transmitted field on the downstream plane** and on
the **far-field amplitude** -- never on the field inside the cloud, where the
microscopic field is granular and only a configuration average means anything
(see :mod:`kamo.dd_solver.fields`).  A density sweep from dilute to the operating
point reports where the two diverge and by how much.

Like for like
-------------
* The BPM is scalar and applies its cross section to the whole probe intensity.
  The lab probe (along x, polarized y, B along z) projects on the sigma- dipole
  with ``|e_- . y|^2 = 1/2``, so only half of the field is scattered and the
  forward-scattered y amplitude per atom is ``alpha / 2``.  The BPM response is
  therefore built with ``sigma0 * incident.projection(op.e_hat)``; the microscopic
  ``psi`` is the y component over the incident y field.  With the FULL sigma0
  the BPM would overestimate the phase by 2x in this geometry -- a point worth
  checking in :mod:`kamo.imaging` itself, whose ``s0_incident`` is documented as
  ``I / I_sat`` without a polarization projection.
* The BPM is also an IDEAL closed two-level line, so the cross section carries
  the absolute oscillator strength of the real line (0.977) as well.
* Both far fields go through the same angular-spectrum transform of the same
  windowed exit-plane field (:func:`kamo.imaging.readout.far_field`), and the
  microscopic forward mode is cross-checked against the direct far-field
  amplitude ``F(x_hat)`` of the dipoles.
* The comparison plane is OUTSIDE the cloud, where ``n ~ 0``.  That is not a
  convenience: inside the cloud the configuration-averaged microscopic field is
  the Lorentz LOCAL field ``E_Maxwell + P/(3 eps0)``, which is not what a
  susceptibility propagation reports (see :mod:`kamo.dd_solver.fields`).
* The cloud is axisymmetric about the probe axis, so a transposed window or a
  swapped ``y``/``z`` axis would be invisible here.  The orientation is checked
  separately on a deliberately anisotropic profile (test T23).

What the residual is, and is not.  An rms difference is only meaningful against
its own configuration-noise floor (:meth:`ABPoint.noise_floor`); a balanced
cloud's residual is dominated by the binomial scatter of ``S_z`` between shots
and is pure noise at 40 configurations.

Expected outcome, itself a check on the comparison: agreement at low density and
a growing difference towards the operating point.  Do not expect the ablation
difference (full minus a near-field-free kernel) to explain that gap -- it does
not.  At ``eta_eff = 27`` the near field moves the COHERENT exit field by only a
few per cent, while it moves the EXCITATION by tens of per cent, because the
excess is carried by subradiant pairs that hold population without radiating.
A Clausius-Mossotti local-field correction is the wrong resummation at this
density (Andreoli et al. 2021: the index saturates rather than diverging), but
the Lorentz term is also not what the exit-plane comparison is sensitive to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .cloud import GaussianProfile
from .ensemble import run_ensemble
from .fields import coherent_plane, coherent_plane_multi
from .system import IncidentField, OperatingPoint, default_incident


@dataclass
class ABPoint:
    """One density in the sweep."""

    scale: float
    N: float
    eta_eff: float
    x_plane: float
    y: np.ndarray                      #: window axes on the BPM grid (m)
    psi_bpm: np.ndarray                #: (ny, nz) E / E_vac on the window
    psi: dict                          #: variant -> (ny, nz) coherent psi
    sem: dict                          #: variant -> median relative sem of <E> over the window
    n_config: int
    forward_bpm: float                 #: BPM forward-mode power W[0,0]
    forward: dict                      #: variant -> forward-mode power from the plane
    forward_direct: dict               #: variant -> |<E_f>|^2 (3 pi / k^2)^2 / 2 (direct dipoles)
    total_bpm: float                   #: BPM total far-field power
    total: dict
    seconds: dict = field(default_factory=dict)
    #: variant -> configuration noise floor IN THE UNITS OF :meth:`rms_difference`
    sem_rms: dict = field(default_factory=dict)

    def on_axis(self, variant: Optional[str] = None) -> complex:
        p = self.psi_bpm if variant is None else self.psi[variant]
        c = (p.shape[0] // 2, p.shape[1] // 2)
        return complex(p[c])

    def rms_difference(self, variant: str) -> float:
        """rms ``|psi_micro - psi_bpm|`` over the window, relative to rms ``|psi_bpm - 1|``.

        Compare with :meth:`noise_floor`, which is this same quantity for pure
        configuration noise: a residual within about 2x of the floor is not a
        physical difference.  For a balanced (theta = pi/2) cloud at 40
        configurations the floor is about 0.46 and the measured residuals are
        0.32-0.44, i.e. entirely noise; the polarized clouds' 0.055-0.187 sit
        2.5-8x above their own floor and are real.
        """
        d = self.psi[variant] - self.psi_bpm
        ref = self.psi_bpm - 1.0
        return float(np.sqrt(np.mean(np.abs(d) ** 2) / np.mean(np.abs(ref) ** 2)))

    def noise_floor(self, variant: str) -> float:
        """Configuration noise of ``psi``, in the units of :meth:`rms_difference`."""
        return float(self.sem_rms.get(variant, float("nan")))

    def is_noise_limited(self, variant: str, factor: float = 2.0) -> bool:
        """True when the residual is within ``factor`` of its own noise floor."""
        fl = self.noise_floor(variant)
        return bool(np.isfinite(fl) and self.rms_difference(variant) < factor * fl)

    def row(self) -> dict:
        out = dict(scale=self.scale, N=self.N, eta_eff=self.eta_eff, n_config=self.n_config,
                   T_bpm=abs(self.on_axis()) ** 2, phi_bpm=float(np.angle(self.on_axis())),
                   forward_bpm=self.forward_bpm)
        for v in self.psi:
            a = self.on_axis(v)
            out[f"T_{v}"] = abs(a) ** 2
            out[f"phi_{v}"] = float(np.angle(a))
            out[f"rms_{v}"] = self.rms_difference(v)
            out[f"floor_{v}"] = self.noise_floor(v)
            out[f"forward_{v}"] = self.forward[v]
            out[f"forward_direct_{v}"] = self.forward_direct[v]
            out[f"sem_{v}"] = self.sem[v]
        return out


def bpm_result(profile: GaussianProfile, op: OperatingPoint, theta: float, incident: IncidentField,
               n_grid: int = 512, L_box: float = 36e-6, n_slices: int = 180, x_span_w: float = 3.0):
    """Run kamo.imaging on the profile; returns ``(PropagationResult, response)``.

    The BPM cross section carries two like-for-like factors: the polarization
    projection ``|conj(e_hat) . pol|^2`` (1/2 in the Voigt geometry) and the
    absolute oscillator strength of the driven line (0.977 at 520.583 G), since
    ``TwoLevelResponse`` is an ideal closed two-level line.  The two species'
    strengths differ by 0.1 %, below anything this comparison resolves, so the
    mean is used.
    """
    from kamo.imaging.bpm import Propagator, UniformMixture
    response = op.response(sigma0_scale=incident.projection(op.e_hat) * op.mean_strength)
    prop = Propagator.for_cloud(response, profile, n_grid=n_grid, L_box=L_box,
                                x_span_w=x_span_w, n_slices=n_slices)
    if hasattr(profile, "bpm_source"):
        # a gridded kamo cloud propagates its REAL shape (kamo.trap's
        # GriddedMixture), not the rms-matched centred Gaussian that
        # UniformMixture would rebuild from N and widths
        src = profile.bpm_source(prop, response, op.species(theta))
    else:
        src = UniformMixture(profile, response, op.species(theta))
    return prop.propagate(src, saturate=False, s0_incident=0.0), response


def _plane_far_field(psi_window, res_bpm, window_slice):
    """Embed a windowed exit field into the BPM's full grid and take its far field."""
    from kamo.imaging.bpm import PropagationResult
    from kamo.imaging.readout import far_field
    g = res_bpm.grid
    psi_full = np.ones((g.n, g.n), dtype=complex)
    psi_full[window_slice, window_slice] = psi_window
    E_vac = np.exp(1j * g.k * 2 * res_bpm.x_edge)
    fake = PropagationResult(grid=g, psi_exit=psi_full, E_scat=(psi_full - 1.0) * E_vac,
                             x_edge=res_bpm.x_edge, x_slices=res_bpm.x_slices)
    return far_field(fake, dipole=False)


def compare_at(profile: GaussianProfile, op: OperatingPoint, theta: float = 0.0,
               n_config: int = 40, R_exc: float = 150e-9, variants=("full", "nonear"),
               window: float = 6e-6, n_grid: int = 512, L_box: float = 36e-6,
               n_slices: int = 180, n_jobs: int = 1, seed0: int = 0,
               incident: Optional[IncidentField] = None, scale: float = 1.0,
               backend: str = "cpu") -> ABPoint:
    """One A/B point: BPM versus microscopic on the exit plane and in the far field."""
    inc = default_incident(op) if incident is None else incident
    t0 = time.perf_counter()
    res_b, _ = bpm_result(profile, op, theta, inc, n_grid, L_box, n_slices)
    t_bpm = time.perf_counter() - t0
    g = res_b.grid
    win = g.window_slice(window)
    y = g.axis[win]
    psi_b = res_b.psi_exit[win, win]

    t0 = time.perf_counter()
    ens = run_ensemble(profile, op, theta, n_config, seed0, variants, n_jobs, inc)
    t_solve = time.perf_counter() - t0
    x_plane = res_b.x_edge
    E_vac = np.exp(1j * op.k * x_plane)
    psi, sem, fwd, fwd_direct, tot, t_field, sem_rms = {}, {}, {}, {}, {}, {}, {}
    from kamo.imaging.readout import far_field
    W_b = far_field(res_b, dipole=False)
    t0 = time.perf_counter()
    planes = coherent_plane_multi({v: ens.results[v] for v in variants}, x_plane, y, y,
                                  R_exc=R_exc, backend=backend)
    t_all = time.perf_counter() - t0
    for v in variants:
        cf, p = planes[v]
        t_field[v] = t_all / len(variants)
        psi[v] = p
        sem[v] = cf.relative_sem()
        sem_rms[v] = cf.sem_versus(psi_b)
        W = _plane_far_field(p, res_b, win)
        fwd[v] = float(W[0, 0])
        tot[v] = float(W.sum())
        Ef = np.mean(ens.forward(v))
        fwd_direct[v] = float(abs(3 * np.pi / op.k ** 2 * Ef) ** 2 * inc.projection(op.e_hat))
    return ABPoint(scale, profile.N, profile.eta_eff(op.wavelength), x_plane, y, psi_b, psi, sem,
                   n_config, float(W_b[0, 0]), fwd, fwd_direct, float(W_b.sum()), tot,
                   dict(bpm=t_bpm, solve=t_solve, **{f"field_{v}": t for v, t in t_field.items()}),
                   sem_rms)


def density_sweep(profile: GaussianProfile, op: OperatingPoint, scales=(6.0, 4.0, 2.5, 1.6, 1.0),
                  theta: float = 0.0, progress=None, **kw) -> List[ABPoint]:
    """Dilute the profile (widths x scale, fixed N) from far below to the operating point."""
    out = []
    for s in scales:
        if progress is not None:
            progress(s)
        out.append(compare_at(profile.scaled(s), op, theta, scale=float(s), **kw))
    return out


def format_sweep(points: Sequence[ABPoint], variants=("full", "nonear")) -> str:
    """Console table of the sweep."""
    head = (f"{'scale':>6s} {'eta_eff':>8s} {'T_bpm':>7s} " + " ".join(f"{'T_' + v:>7s}" for v in variants)
            + f" {'phi_bpm':>8s} " + " ".join(f"{'phi_' + v:>8s}" for v in variants)
            + " " + " ".join(f"{'rms_' + v:>7s}" for v in variants)
            + " " + " ".join(f"{'floor_' + v:>9s}" for v in variants)
            + f" {'fwd_bpm':>9s} " + " ".join(f"{'fwd_' + v:>9s}" for v in variants))
    lines = [head]
    for p in points:
        r = p.row()
        lines.append(f"{r['scale']:6.2f} {r['eta_eff']:8.3f} {r['T_bpm']:7.3f} "
                     + " ".join(f"{r['T_' + v]:7.3f}" for v in variants)
                     + f" {r['phi_bpm']:+8.3f} " + " ".join(f"{r['phi_' + v]:+8.3f}" for v in variants)
                     + " " + " ".join(f"{r['rms_' + v]:7.3f}" + ("*" if p.is_noise_limited(v) else " ")
                                      for v in variants)
                     + " " + " ".join(f"{r['floor_' + v]:9.3f}" for v in variants)
                     + f" {r['forward_bpm']:9.3e} "
                     + " ".join(f"{r['forward_' + v]:9.3e}" for v in variants))
    lines.append("* rms within 2x of its own configuration-noise floor: not a physical difference")
    return "\n".join(lines)
