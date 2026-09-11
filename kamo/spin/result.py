"""What one imaging pulse produced, and what a sequence of them produced."""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class ImagingResult:
    """The readout of a single probe pulse, plus what it did to the spins.

    The ground-truth arrays (:attr:`xi_true`, :attr:`phi_z`) have no experimental
    counterpart -- they are here so a reconstruction can be scored against what
    the simulation actually contained, which is the whole point of a forward model.
    """

    propagation: Any                 #: the raw PropagationResult
    psi: np.ndarray                  #: field at the cloud plane, E/E_vac
    image: np.ndarray                #: phase-contrast image, I/I0
    far_field_power: np.ndarray      #: per-mode scattered power
    into_NA: float                   #: coherent fraction inside the NA
    recovered_phase: float           #: on-axis phase excursion (rad)
    optical_depth: float             #: on-axis D = -2 ln|psi|
    signal: float                    #: on-axis I/I0 - 1
    xi_true: np.ndarray              #: column imbalance BEFORE this pulse
    phi_z: np.ndarray                #: 3D z-rotation this pulse imprinted (rad),
                                     #: signed as a precession about +z (= |up>)
    n_scatter: np.ndarray            #: 3D photons scattered per atom
    t_pulse: float
    window: Optional[slice] = None

    # ------------------------------------------------------------- derived maps

    @property
    def transmission(self) -> np.ndarray:
        """``|psi|^2`` at the cloud plane."""
        return np.abs(self.psi) ** 2

    @property
    def phase(self) -> np.ndarray:
        """Wrapped exit phase (rad)."""
        return np.angle(self.psi)

    @property
    def column_phase(self) -> np.ndarray:
        """Unwrapped phase map, referenced to the box edge.

        Unwrapped along both axes from the corner, which is outside the cloud and
        therefore unperturbed.
        """
        return np.unwrap(np.unwrap(np.angle(self.psi), axis=0), axis=1)

    @property
    def light_shift_Hz(self) -> np.ndarray:
        """Differential light shift ``nu_dn - nu_up`` per voxel, in Hz.

        POSITIVE: between the two transitions the probe pushes ``|dn>`` up and
        ``|up>`` down.  This is the splitting; :attr:`phi_z` is the corresponding
        precession about ``+z`` and carries the opposite sign, because
        :mod:`kamo.spin` puts ``|up>`` at ``+z``.
        """
        return -self.phi_z / (2 * np.pi * self.t_pulse)

    @property
    def mean_light_shift_Hz(self) -> float:
        """Density-weighted mean differential light shift over the cloud (Hz).

        Weighted by ATOM NUMBER, ``sum(n nu)/sum(n)`` -- what an average atom
        experiences, not what the probe delivers to the front face.  The rotation
        itself is applied per voxel; this is a reported summary of it.
        """
        n = self._density()
        return float((n * self.light_shift_Hz).sum() / n.sum())

    @property
    def phi_z_column(self) -> np.ndarray:
        """Density-weighted mean of :attr:`phi_z` down each column (rad)."""
        n = self._density()
        return (n * self.phi_z).sum(axis=0) / n.sum(axis=0)

    @property
    def phi_z_spread(self) -> np.ndarray:
        """Density-weighted rms spread of :attr:`phi_z` within each column (rad).

        This -- not the mean -- is what dephases a Ramsey fringe.  A pulse can
        shift every spin and still preserve contrast; only the SPREAD destroys it.
        """
        n = self._density()
        den = n.sum(axis=0)
        m = (n * self.phi_z).sum(axis=0) / den
        var = (n * (self.phi_z - m[None]) ** 2).sum(axis=0) / den
        return np.sqrt(np.clip(var, 0, None))

    @property
    def dephasing_factor(self) -> np.ndarray:
        """Within-column coherence surviving the phase spread, ``|<exp(i phi)>|``.

        The exact factor, not a Gaussian approximation -- the phase distribution
        across a lensed cloud is far from Gaussian.
        """
        n = self._density()
        return np.abs((n * np.exp(1j * self.phi_z)).sum(axis=0) / n.sum(axis=0))

    @property
    def scatter_factor(self) -> np.ndarray:
        """Coherence surviving spontaneous scattering, ``exp(-N_sc)``, per voxel."""
        return np.exp(-self.n_scatter)

    @property
    def photons_per_atom(self) -> float:
        """Density-weighted mean photons scattered per atom during the pulse."""
        n = self._density()
        return float((n * self.n_scatter).sum() / n.sum())

    def _density(self):
        d = getattr(self, "_dens", None)
        if d is None:
            raise AttributeError(
                "ImagingResult needs its density weights; they are attached by "
                "Sequence.run -- construct results through an ImagePulse")
        return d

    def attach_density(self, density: np.ndarray) -> "ImagingResult":
        """Attach the voxel density weights used by the column reductions."""
        self._dens = np.asarray(density, dtype=float)
        return self

    # ---------------------------------------------------------------- reporting

    def summary(self) -> str:
        lines = [
            f"ImagingResult  t_pulse = {self.t_pulse * 1e6:.2f} us",
            f"  recovered phase = {self.recovered_phase:+.4f} rad,  "
            f"on-axis D = {self.optical_depth:+.4f}",
            f"  PCI signal I/I0 - 1 = {self.signal:+.4f}   "
            f"(image range [{self.image.min():.3f}, {self.image.max():.3f}])",
            f"  coherent into NA = {self.into_NA:.5f}",
            f"  peak local s = {self.propagation.s_peak:.4f}, "
            f"<I>/I0 = {self.propagation.mean_intensity:.4f}",
            f"  imprinted |phi_z|: peak {np.abs(self.phi_z).max():.3f} rad, "
            f"column-mean peak {np.abs(self.phi_z_column).max():.3f} rad",
        ]
        try:
            lines.append(
                f"  differential light shift (dn - up): mean "
                f"{self.mean_light_shift_Hz/1e3:+.3f} kHz, peak "
                f"{self.light_shift_Hz.max()/1e3:+.3f} kHz")
            lines.append(
                f"  within-column phase spread: peak "
                f"{self.phi_z_spread.max():.3f} rad -> min coherence "
                f"{self.dephasing_factor.min():.3f}")
            lines.append(f"  photons scattered per atom = {self.photons_per_atom:.4f}"
                         f"  -> coherence x {np.exp(-self.photons_per_atom):.3f}")
        except AttributeError:
            pass
        return "\n".join(lines)

    def __repr__(self):
        return (f"ImagingResult(phi={self.recovered_phase:+.3f} rad, "
                f"D={self.optical_depth:+.3f}, signal={self.signal:+.4f})")


@dataclass
class SequenceResult:
    """Everything a :class:`~kamo.spin.sequence.Sequence` produced."""

    field: Any                       #: the final SpinField
    images: List[ImagingResult] = _dc_field(default_factory=list)
    records: List[dict] = _dc_field(default_factory=list)
    initial_Sz: float = float("nan")
    initial_contrast: float = float("nan")

    @property
    def final_Sz(self) -> float:
        return self.field.Sz_total

    @property
    def final_contrast(self) -> float:
        return self.field.contrast

    def __getitem__(self, i) -> ImagingResult:
        return self.images[i]

    def __len__(self):
        return len(self.images)

    def summary(self) -> str:
        lines = [f"SequenceResult  {len(self.records)} operations, "
                 f"{len(self.images)} images"]
        for r in self.records:
            if r["kind"] == "rotate":
                lines.append(f"  rotate {r['angle'] / np.pi:+.3f} pi about {r['axis']}")
            elif r["kind"] == "free":
                extra = (f", mean-field peak {r['nu_mf_peak_Hz']:.1f} Hz"
                         if "nu_mf_peak_Hz" in r else "")
                lines.append(f"  free {r['duration'] * 1e3:.3f} ms "
                             f"@ {r['detuning_Hz']:.1f} Hz{extra}")
            else:
                lines.append("  " + r["result"].summary().replace("\n", "\n  "))
        lines.append(f"  Sz_total {self.initial_Sz:+.4f} -> {self.final_Sz:+.4f}, "
                     f"contrast {self.initial_contrast:.4f} -> "
                     f"{self.final_contrast:.4f}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"SequenceResult({len(self.images)} images, "
                f"Sz {self.initial_Sz:+.3f} -> {self.final_Sz:+.3f}, "
                f"contrast {self.initial_contrast:.3f} -> {self.final_contrast:.3f})")
