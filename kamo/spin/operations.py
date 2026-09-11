"""Operations that act on a :class:`~kamo.spin.field.SpinField`.

Each is a callable taking ``(field, context)`` and returning ``(field, record)``,
so they compose into a :class:`~kamo.spin.sequence.Sequence`.

The central physical point, which decides the whole design: **a z-rotation
commutes with S_z.**  The imaging pulse's differential light shift therefore
leaves populations -- and so the linear susceptibility -- exactly invariant.
Three consequences:

1. Running a probe pulse twice in a row reproduces the same image.  The imprint is
   invisible until something converts phase into population.
2. One propagation pass per pulse is not an approximation.  The rotation a pulse
   causes cannot change the index that pulse sees, so there is nothing to iterate
   to self-consistency.  (The one channel that WOULD couple back is optical
   pumping, which moves population rather than phase; it is not modelled.)
3. The interesting protocol is a Ramsey sequence:

       Rotate(pi/2, 'y') -> ImagePulse -> Rotate(pi/2, 'y') -> ImagePulse

   The first pulse imprints ``phi_z(r)``, the second pi/2 turns it into
   ``S_z(r) = (1/2) C(r) cos(phi_z(r) + phi_0)``, and only then does a probe see
   spatial structure.

What an imaging pulse does to the spins
---------------------------------------
**Rotation.**  ``phi_z(r) = 2 pi nu_diff(r) t``, where ``nu_diff`` is the
differential AC Stark shift evaluated at the LOCAL saturation parameter from the
propagated field.  The Bloch precession rate equals the full level splitting over
hbar, which is the same number as the spinor's relative phase -- there is no extra
factor of two.

**Decoherence.**  Each spontaneously scattered photon is which-path information:
the two transitions are split by far more than a linewidth, so an emitted photon
identifies which state emitted it.  Scattering on either branch therefore
decoheres, and the coherence decays as the no-scatter survival probability
``exp(-(R_up + R_dn) t / 2)``.  At the midpoint the two rates are equal (the rate
depends on ``delta^2``), so this is ``exp(-R t)``.  This is not a small correction:
a lensed cloud can reach a half photon per atom in a few microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from kamo.imaging import readout
from .field import SpinField

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _axis_vector(axis):
    if isinstance(axis, str):
        try:
            return _AXES[axis.lower()]
        except KeyError:
            raise ValueError(f"axis must be 'x', 'y', 'z' or a 3-vector, got {axis!r}")
    return tuple(float(c) for c in axis)


class Operation:
    """Base class: ``__call__(field, context) -> (field, record)``."""

    label: str = "operation"

    def __call__(self, field: SpinField, context: Dict[str, Any]):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.label})"


@dataclass
class Rotate(Operation):
    """A resonant RF/microwave pulse: a rigid Bloch rotation.

    Parameters
    ----------
    angle : float
        Rotation angle (rad).  ``pi/2`` is the usual Ramsey pulse.
    axis : {'x', 'y', 'z'} or 3-vector
        Rotation axis.  The pulse phase sets it.
    inhomogeneity : ndarray, optional
        Multiplies ``angle`` voxel by voxel, for a spatially varying drive
        amplitude.  Broadcasts to the field shape.

    Idealized as instantaneous and (by default) uniform, which is good while the
    drive wavelength dwarfs the cloud -- centimetres against microns here.
    """

    angle: float
    axis: Any = "y"
    inhomogeneity: Optional[np.ndarray] = None
    label: str = "rotate"

    def __call__(self, field, context):
        a = self.angle if self.inhomogeneity is None \
            else self.angle * np.asarray(self.inhomogeneity)
        field.rotate(_axis_vector(self.axis), a)
        return field, {"kind": "rotate", "angle": self.angle, "axis": self.axis}


@dataclass
class FreeEvolve(Operation):
    """Hold: uniform detuning precession, optionally plus the mean-field shift.

    Parameters
    ----------
    duration : float
        Hold time (s).
    detuning_Hz : float
        Uniform rotating-frame detuning; precesses every spin about z equally.
    mean_field : bool
        Include the density-dependent differential interaction shift.  Requires
        ``a_upup``, ``a_dndn`` and ``a_updn`` in metres.

    The mean-field ("clock") shift is
    ``h nu_mf(r) = [g_uu n_up(r) + g_ud n_dn(r)] - [g_dd n_dn(r) + g_ud n_up(r)]``
    in the local-density approximation, with ``g_ab = 4 pi hbar^2 a_ab / m``.  It
    scales with DENSITY rather than intensity, so it is negligible over a
    microsecond imaging pulse and dominant over a millisecond hold.  Spin-changing
    collisions and any back-action on the density are not included.
    """

    duration: float
    detuning_Hz: float = 0.0
    mean_field: bool = False
    a_upup: Optional[float] = None
    a_dndn: Optional[float] = None
    a_updn: Optional[float] = None
    label: str = "free"

    def __call__(self, field, context):
        import kamo.constants as kc
        phi = 2 * np.pi * self.detuning_Hz * self.duration
        rec = {"kind": "free", "duration": self.duration,
               "detuning_Hz": self.detuning_Hz}
        if self.mean_field:
            if None in (self.a_upup, self.a_dndn, self.a_updn):
                raise ValueError("mean_field=True needs a_upup, a_dndn, a_updn "
                                 "(in metres) -- see kamo.scattering.ScatteringModel")
            n = field.geometry.density
            xi = field.xi
            n_up, n_dn = 0.5 * (1 + xi) * n, 0.5 * (1 - xi) * n
            pref = 4 * np.pi * kc.hbar**2 / kc.m_K
            e_up = pref * (self.a_upup * n_up + self.a_updn * n_dn)
            e_dn = pref * (self.a_dndn * n_dn + self.a_updn * n_up)
            nu_mf = (e_up - e_dn) / kc.h
            phi = phi + 2 * np.pi * nu_mf * self.duration
            rec["nu_mf_peak_Hz"] = float(np.abs(nu_mf).max())
        field.rotate_z(phi)
        return field, rec


@dataclass
class ImagePulse(Operation):
    """Propagate a probe pulse, record the image, and act back on the spins.

    One propagation pass does three jobs, because they all follow from the same
    local intensity field:

    1. the image (transmission, phase, far field, phase contrast);
    2. the z-rotation ``phi_z(r) = 2 pi nu_diff(r) t_pulse``;
    3. the coherence loss ``exp(-R(r) t_pulse)`` from spontaneous scattering.

    Parameters
    ----------
    propagator : kamo.imaging.Propagator
    t_pulse : float
        Pulse duration (s).
    NA : float, optional
        Collection numerical aperture for the far field and the image aperture.
    theta_plate : float
        Phase-plate retardation (rad); ``pi/2`` is standard phase contrast.
    back_action : bool
        Apply (2) and (3) to the field.  Set False to image without disturbing --
        not physical, but useful for isolating what the readout alone would say.
    """

    propagator: Any
    t_pulse: float
    NA: Optional[float] = 0.42
    theta_plate: float = np.pi / 2
    back_action: bool = True
    label: str = "image"

    def __call__(self, field, context):
        from .result import ImagingResult

        probe = context["probe"]
        response = context["response"]
        s0 = probe.s0_incident

        source = field.susceptibility_source(response, probe)
        geom = field.geometry
        res = self.propagator.propagate(
            source, s0_incident=s0, record="3d", record_window=geom.window)
        if res.intensity_3d.shape != geom.shape:
            raise RuntimeError(
                f"propagation record {res.intensity_3d.shape} does not match the "
                f"spin grid {geom.shape}; build the SpinGeometry with "
                f"SpinGeometry.from_propagator(this propagator, cloud)")

        # What the atoms actually saw -- per voxel, from the intensity that
        # propagated to that point.  No average enters the rotation; the
        # density-weighted means on ImagingResult are diagnostics computed after.
        s_local = s0 * res.intensity_3d.astype(float)
        nu_split = probe.differential_light_shift_Hz(s_local)   # dn - up, positive
        # ... and the precession about +z (= |up>) is minus that splitting.
        phi_z = 2 * np.pi * probe.larmor_rate_Hz(s_local) * self.t_pulse
        r_up, r_dn = probe.scattering_rates(s_local)
        n_scatter = 0.5 * (r_up + r_dn) * self.t_pulse

        xi_true, _ = field.collapse_to_columns()
        if self.back_action:
            field.rotate_z(phi_z)
            field.decohere(np.exp(-n_scatter))

        psi = readout.refocus(res)
        image = readout.phase_contrast(psi, res.grid, theta=self.theta_plate,
                                       NA=self.NA)
        W = readout.far_field(res)
        result = ImagingResult(
            propagation=res,
            psi=psi,
            image=image,
            far_field_power=W,
            into_NA=(readout.into_NA(W, res.grid, self.NA)
                     if self.NA is not None else float("nan")),
            recovered_phase=readout.recovered_phase(psi, res.grid),
            optical_depth=readout.optical_depth(psi),
            signal=readout.signal_on_axis(image),
            xi_true=xi_true,
            phi_z=phi_z,
            n_scatter=n_scatter,
            t_pulse=self.t_pulse,
            window=geom.window,
        )
        return field, {"kind": "image", "result": result}
