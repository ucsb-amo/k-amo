"""The imaging probe: one laser, two ground-state transitions.

A dispersive spin readout drives BOTH ground states with a single beam.  The two
transitions are split by the magnetic field, so their detunings are not
independent -- parking the laser a distance ``x`` from their midpoint gives

    Delta_up = x - S/2,    Delta_dn = x + S/2,    S = splitting,

leaving one knob.  :class:`ProbeBeam` carries that geometry so nothing downstream
has to recompute it or get the sign wrong.

Why the midpoint matters
------------------------
At ``x = 0`` the two reduced detunings are exactly opposite, ``delta_dn =
-delta_up``.  Since ``Im alpha`` is EVEN in delta and ``Re alpha`` is ODD, a
mixture with populations ``p_up, p_dn`` sharing one spatial profile has

    D   = sigma0 ncol / (1 + delta^2)                 (independent of spin)
    phi = -(D/2) delta_up (p_up - p_dn)               (pure spin imbalance)

so absorption reads atom number and refraction reads S_z, cleanly separated.  The
midpoint is also where the differential light shift is stationary in ``x``, i.e.
least sensitive to laser-frequency drift.

Quick start
-----------
>>> from kamo import Potassium39
>>> from kamo.imaging import ProbeBeam
>>> atom = Potassium39()
>>> probe = ProbeBeam.from_midpoint(
...     atom, B_gauss=520.58,
...     ground_up=(4, 0, 1/2, -1/2, +1/2), excited_up=(4, 1, 3/2, -3/2, +1/2),
...     ground_dn=(4, 0, 1/2, -1/2, -1/2), excited_dn=(4, 1, 3/2, -3/2, -1/2),
...     s0_incident=0.335)
>>> probe.delta_up, probe.delta_dn
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .response import TwoLevelResponse


class ProbeBeam:
    """A probe laser and the two transitions it drives.

    Parameters
    ----------
    frequency_Hz : float
        Laser frequency.
    f_up, f_dn : float
        The two ground-state transition frequencies (Hz).
    response : TwoLevelResponse
        The atomic response, carrying the linewidth and cross section.
    s0_incident : float
        Incident ON-RESONANCE saturation parameter ``I / I_sat``.  Use
        :meth:`from_offresonant_saturation` if you measured the off-resonant one.
    """

    def __init__(self, frequency_Hz: float, f_up: float, f_dn: float,
                 response: TwoLevelResponse, s0_incident: float = 0.0):
        self.frequency_Hz = float(frequency_Hz)
        self.f_up = float(f_up)
        self.f_dn = float(f_dn)
        self.response = response
        self.s0_incident = float(s0_incident)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_midpoint(cls, atom, B_gauss, ground_up, excited_up, ground_dn,
                      excited_dn, offset_Hz: float = 0.0,
                      s0_incident: float = 0.0, linewidth_Hz: Optional[float] = None,
                      response: Optional[TwoLevelResponse] = None,
                      local_field: bool = False, sigma0: Optional[float] = None):
        """Build from an atom and the four state tuples, parked near the midpoint.

        ``offset_Hz`` displaces the laser from the midpoint (positive = towards
        ``|up>``).  The response's cross section is taken from the ACTUAL
        field-shifted ``|dn>`` transition, so ``sigma0 = 3 (c/f_dn)^2 / (2 pi)``.

        State tuples follow kamo's convention: ``(n, l, j, F, mF)`` with integer
        ``F, mF``, or ``(n, l, j, m_j, m_i)`` with half-integer floats.
        """
        f_up = atom.get_transition_frequency(ground_up, excited_up, B=B_gauss,
                                             relative_mode='absolute')
        f_dn = atom.get_transition_frequency(ground_dn, excited_dn, B=B_gauss,
                                             relative_mode='absolute')
        if response is None:
            if linewidth_Hz is None:
                linewidth_Hz = atom.get_decay_rate(*ground_dn[:3],
                                                   *excited_dn[:3]) / (2 * np.pi)
            response = TwoLevelResponse.from_transition_frequency(
                f_dn, linewidth_Hz, sigma0=sigma0, local_field=local_field)
        f_mid = 0.5 * (f_up + f_dn)
        return cls(f_mid + float(offset_Hz), f_up, f_dn, response, s0_incident)

    def at_offset(self, offset_Hz: float) -> "ProbeBeam":
        """Same transitions and response, laser moved to a new midpoint offset."""
        return ProbeBeam(self.f_midpoint + float(offset_Hz), self.f_up, self.f_dn,
                         self.response, self.s0_incident)

    def with_saturation(self, s0_incident: float) -> "ProbeBeam":
        """Same geometry, different incident saturation parameter."""
        return ProbeBeam(self.frequency_Hz, self.f_up, self.f_dn,
                         self.response, s0_incident)

    @classmethod
    def from_offresonant_saturation(cls, probe: "ProbeBeam", s_offresonant: float,
                                    which: str = "up") -> "ProbeBeam":
        """Rebuild ``probe`` from a measured OFF-resonant saturation parameter.

        A measured ``s`` already carries the Lorentzian denominator of whichever
        transition it was referred to; undo that to recover ``s0 = I / I_sat``.
        """
        delta = probe.delta_up if which == "up" else probe.delta_dn
        return probe.with_saturation(
            float(probe.response.saturation_from_offresonant(s_offresonant, delta)))

    # --------------------------------------------------------------- geometry

    @property
    def f_midpoint(self) -> float:
        """Midpoint of the two transitions (Hz)."""
        return 0.5 * (self.f_up + self.f_dn)

    @property
    def splitting_Hz(self) -> float:
        """Separation of the two transitions, ``|f_up - f_dn|`` (Hz)."""
        return abs(self.f_up - self.f_dn)

    @property
    def offset_Hz(self) -> float:
        """Laser offset from the midpoint (Hz); positive is towards ``|up>``."""
        return self.frequency_Hz - self.f_midpoint

    @property
    def detuning_up_Hz(self) -> float:
        """Signed detuning from the ``|up>`` transition, ``f_laser - f_up``."""
        return self.frequency_Hz - self.f_up

    @property
    def detuning_dn_Hz(self) -> float:
        """Signed detuning from the ``|dn>`` transition, ``f_laser - f_dn``."""
        return self.frequency_Hz - self.f_dn

    @property
    def delta_up(self) -> float:
        """Reduced detuning ``2 (f_laser - f_up) / Gamma_Hz``."""
        return float(self.response.delta(self.detuning_up_Hz))

    @property
    def delta_dn(self) -> float:
        """Reduced detuning ``2 (f_laser - f_dn) / Gamma_Hz``."""
        return float(self.response.delta(self.detuning_dn_Hz))

    @property
    def at_midpoint(self) -> bool:
        """True when ``delta_dn == -delta_up`` to floating-point tolerance."""
        return abs(self.delta_up + self.delta_dn) < 1e-9 * max(abs(self.delta_up), 1.0)

    # --------------------------------------------------------------- mixtures

    def species(self, xi: float = 1.0) -> Tuple[Tuple[float, float], ...]:
        """``((frac_up, delta_up), (frac_dn, delta_dn))`` for spin imbalance ``xi``.

        ``xi = p_up - p_dn`` in [-1, 1]; ``xi = +1`` is all ``|up>``, ``-1`` all
        ``|dn>``, ``0`` balanced.  Feed the result to
        :meth:`~kamo.imaging.response.TwoLevelResponse.thin_screen` or to the
        propagator's uniform-mixture susceptibility source.
        """
        return ((0.5 * (1 + xi), self.delta_up), (0.5 * (1 - xi), self.delta_dn))

    # ---------------------------------------------------------- light shifts

    def light_shifts_Hz(self, s_local=None):
        """``(shift_up, shift_dn)`` ground-state AC Stark shifts in Hz."""
        s = self.s0_incident if s_local is None else s_local
        r = self.response
        return r.light_shift_Hz(self.delta_up, s), r.light_shift_Hz(self.delta_dn, s)

    def differential_light_shift_Hz(self, s_local=None):
        """``nu_dn - nu_up``, the differential AC Stark shift in Hz.

        POSITIVE throughout the band between the two transitions.  There the probe
        is red-detuned from ``|up>`` (``delta_up < 0``, so ``|up>`` is pushed down)
        and blue-detuned from ``|dn>`` (``delta_dn > 0``, pushed up), so ``|dn>``
        sits above ``|up>`` and this difference is positive at every offset with
        ``|x| < S/2``.  At the midpoint it reduces to
        ``(Gamma_Hz/2) s |delta_up| / (1 + delta_up^2 + s)``.

        It is linear in the local intensity, so a spatially varying ``s_local``
        maps directly onto a spatially varying splitting.

        Under ``x -> -x`` the two detunings swap and change sign, so the shift is
        EXACTLY even in the midpoint offset -- at the incident intensity.  It stops
        being even once ``s_local`` comes from a propagated field, because the
        cloud's own lensing is not symmetric in the offset.

        Note the ordering: this is the SPLITTING (``dn`` above ``up``), reported
        positive.  The sense in which it precesses a Bloch vector is the opposite
        sign, because :mod:`kamo.spin` puts ``|up>`` at ``+z`` -- see
        :meth:`larmor_rate_Hz`.
        """
        up, dn = self.light_shifts_Hz(s_local)
        return dn - up

    def larmor_rate_Hz(self, s_local=None):
        """Signed precession rate about ``+z``, where ``+z`` is ``|up>``.

        The Bloch vector precesses at ``(E_up - E_dn)/h``, which is minus the
        splitting reported by :meth:`differential_light_shift_Hz`.  Keeping the two
        as separate methods stops the reporting convention (splitting, positive)
        from silently changing the rotation sense.
        """
        return -self.differential_light_shift_Hz(s_local)

    def scattering_rates(self, s_local=None):
        """``(R_up, R_dn)`` photon scattering rates in s^-1."""
        s = self.s0_incident if s_local is None else s_local
        r = self.response
        return r.scattering_rate(self.delta_up, s), r.scattering_rate(self.delta_dn, s)

    def summary(self) -> str:
        """Human-readable multi-line report."""
        r = self.response
        return "\n".join([
            f"ProbeBeam  f = {self.frequency_Hz / 1e12:.6f} THz "
            f"({self.offset_Hz / 1e6:+.3f} MHz from midpoint)",
            f"  |up>  f = {self.f_up / 1e12:.6f} THz, Delta = "
            f"{self.detuning_up_Hz / 1e6:+8.3f} MHz -> delta = {self.delta_up:+8.3f}",
            f"  |dn>  f = {self.f_dn / 1e12:.6f} THz, Delta = "
            f"{self.detuning_dn_Hz / 1e6:+8.3f} MHz -> delta = {self.delta_dn:+8.3f}",
            f"  splitting S = {self.splitting_Hz / 1e6:.4f} MHz"
            + ("  (at midpoint: delta_dn = -delta_up)" if self.at_midpoint else ""),
            f"  incident s0 = {self.s0_incident:.4f}, "
            f"differential light shift = "
            f"{self.differential_light_shift_Hz() / 1e3:+.3f} kHz",
            f"  {r!r}",
        ])

    def __repr__(self):
        return (f"ProbeBeam(offset={self.offset_Hz / 1e6:+.3f} MHz, "
                f"delta_up={self.delta_up:+.2f}, delta_dn={self.delta_dn:+.2f}, "
                f"s0={self.s0_incident:.4f})")
