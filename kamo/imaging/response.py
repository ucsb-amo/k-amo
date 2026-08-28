"""Complex optical response of a two-level atom, and the medium it makes.

The single object every other module in :mod:`kamo.imaging` asks for the atomic
response.  It carries one transition (wavelength, linewidth, cross section) and
turns a reduced detuning ``delta = 2 Delta / Gamma`` plus a saturation parameter
into a polarizability, a susceptibility, a refractive index, a cross section, a
scattering rate and a light shift -- all from the same closed form, so the
absorptive and dispersive channels can never drift apart.

Quick start
-----------
>>> from kamo.imaging import TwoLevelResponse
>>> r = TwoLevelResponse.from_transition_frequency(391.016e12, linewidth_Hz=6.035e6)
>>> r.sigma0                       # 3 lambda^2 / 2 pi, m^2
>>> r.polarizability(delta=-18.26) # SI, C m^2 / V

Physics
-------
The rotating-wave two-level polarizability is

    alpha(delta) = -(eps0 sigma0 / k) (delta - i) / (1 + delta^2 + s)

with ``sigma0 = 6 pi / k^2 = 3 lambda^2 / (2 pi)``.  ``Im alpha > 0`` absorbs --
the optical theorem gives ``sigma_ext = k Im alpha / eps0 = sigma0/(1+delta^2+s)``
-- while ``Re alpha`` refracts and is ODD in delta: red detuning (delta < 0) gives
``Re alpha > 0``, hence ``n_ref > 1`` and a CONVERGING cloud.

The ``1 + delta^2 -> 1 + delta^2 + s`` replacement is exact, not an
approximation: the steady-state optical Bloch coherence
``rho_eg ~ (Delta + i Gamma/2) / (Delta^2 + Gamma^2/4 + Omega^2/2)`` puts the same
denominator under both quadratures, so saturation power-broadens the dispersive
and absorptive parts identically.  ``s`` throughout is the ON-RESONANCE
saturation parameter ``I / I_sat``; see :meth:`saturation_from_offresonant`.

A slab of density ``n`` has ``chi = n alpha / eps0`` and refractive index
``n_ref = 1 + chi/2`` in the dilute limit, so a field crossing ``dx`` picks up
``exp(i k (n_ref - 1) dx)``.  Pass ``local_field=True`` to replace that with the
Clausius-Mossotti result instead -- see :attr:`local_field`.

.. warning::
   This is a TWO-LEVEL response.  It is a good approximation for a sigma- D2
   transition from a Paschen-Back ground state (at K-39 fields of several hundred
   gauss the excited mj branches split by ~1 GHz, far exceeding their ~30 MHz
   hyperfine spread), but it neglects the residual ground-state mj admixture,
   which is ~2.5% at 520 G.  For the exact multi-level shift use
   :meth:`kamo.hamiltonian.AtomicStructure.laser_sweep`.

   Only the COHERENT (elastic) response is modelled.  At local ``s`` of order 1
   the elastic fraction of total scattering is ``1/(1+s)``; the inelastic
   remainder is incoherent fluorescence and is not carried by a field propagation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import kamo.constants as kc


class TwoLevelResponse:
    """Complex polarizability and medium response of a two-level transition.

    Parameters
    ----------
    wavelength : float
        Transition wavelength in metres.  Prefer building via
        :meth:`from_transition_frequency` so the wavelength is the ACTUAL
        field-shifted transition rather than the nominal line centre.
    linewidth_Hz : float
        Natural linewidth ``Gamma / 2 pi`` in Hz.
    sigma0 : float, optional
        Resonant cross section (m^2).  Defaults to ``3 lambda^2 / (2 pi)``, which
        is correct for a closed transition with unit Clebsch-Gordan coefficient
        (e.g. sigma- D2 from ``m_j = -1/2`` in the Paschen-Back regime).  Override
        only to reproduce a legacy number.
    local_field : bool
        Apply the Clausius-Mossotti (Lorentz-Lorenz) local-field correction when
        converting susceptibility to refractive index.  Default False, matching
        the independent-atom treatment; see :attr:`local_field`.
    """

    def __init__(self, wavelength: float, linewidth_Hz: float,
                 sigma0: Optional[float] = None, local_field: bool = False):
        self.wavelength = float(wavelength)
        self.linewidth_Hz = float(linewidth_Hz)
        self.local_field = bool(local_field)
        self.k = 2 * np.pi / self.wavelength
        self.Gamma = 2 * np.pi * self.linewidth_Hz            # rad/s
        self.sigma0 = float(3 * self.wavelength**2 / (2 * np.pi)
                            if sigma0 is None else sigma0)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_transition_frequency(cls, frequency_Hz: float, linewidth_Hz: float,
                                  **kwargs) -> "TwoLevelResponse":
        """Build from the transition frequency (Hz), i.e. ``lambda = c / f``.

        This is the right entry point at high field, where the transition is
        Zeeman-shifted away from the nominal D2 line centre and the resonant cross
        section ``3 (c/f)^2 / (2 pi)`` shifts with it.
        """
        return cls(kc.c / float(frequency_Hz), linewidth_Hz, **kwargs)

    @classmethod
    def from_atom(cls, atom, ground, excited, B_gauss=0.0, linewidth_Hz=None,
                  **kwargs) -> "TwoLevelResponse":
        """Build from a :class:`kamo.Potassium39` and a pair of state tuples.

        ``ground`` and ``excited`` are ``(n, l, j, a, b)`` tuples in kamo's usual
        convention (ints -> ``(F, mF)``, half-integer floats -> ``(m_j, m_i)``).
        The transition frequency is evaluated at ``B_gauss``, so the cross section
        follows the field.
        """
        f0 = atom.get_transition_frequency(ground, excited, B=B_gauss,
                                           relative_mode='absolute')
        if linewidth_Hz is None:
            # get_decay_rate takes (ground nlj, excited nlj) and returns Gamma in rad/s
            linewidth_Hz = atom.get_decay_rate(*ground[:3], *excited[:3]) / (2 * np.pi)
        return cls.from_transition_frequency(f0, linewidth_Hz, **kwargs)

    # ------------------------------------------------------------- detunings

    def delta(self, detuning_Hz):
        """Reduced detuning ``2 Delta / Gamma`` from a signed detuning in Hz."""
        return 2 * np.asarray(detuning_Hz, dtype=float) / self.linewidth_Hz

    def detuning_Hz(self, delta):
        """Signed detuning in Hz from a reduced detuning."""
        return 0.5 * np.asarray(delta, dtype=float) * self.linewidth_Hz

    def saturation_from_offresonant(self, s_offresonant, delta):
        """Convert an OFF-resonant saturation parameter to the on-resonance one.

        A measured ``s = s0 / (1 + delta^2)`` already carries the Lorentzian
        denominator; everything in this class expects the bare ``s0 = I / I_sat``,
        which is what stays put when the laser is retuned at fixed intensity.
        """
        return np.asarray(s_offresonant, dtype=float) * (1 + np.asarray(delta) ** 2)

    # ----------------------------------------------------------- polarizability

    def polarizability(self, delta, s=0.0):
        """Complex polarizability alpha (SI, C m^2 / V).

        ``alpha = -(eps0 sigma0 / k) (delta - i) / (1 + delta^2 + s)``.
        Broadcasts over arrays in either argument.
        """
        delta = np.asarray(delta, dtype=float)
        return -(kc.epsilon_0 * self.sigma0 / self.k) * (delta - 1j) / \
            (1 + delta**2 + np.asarray(s))

    def susceptibility(self, density, delta, s=0.0):
        """Linear susceptibility ``chi = n alpha / eps0`` (dimensionless)."""
        return np.asarray(density) * self.polarizability(delta, s) / kc.epsilon_0

    def index_minus_one(self, chi):
        """``n_ref - 1`` from a susceptibility.

        Dilute (default): ``chi / 2``.  With :attr:`local_field`, the
        Clausius-Mossotti result ``n^2 = (1 + 2 chi/3) / (1 - chi/3)`` instead.
        """
        chi = np.asarray(chi, dtype=complex)
        if not self.local_field:
            return 0.5 * chi
        return np.sqrt((1 + 2 * chi / 3) / (1 - chi / 3)) - 1.0

    def slice_operator(self, density, delta, dx, s=0.0):
        """Field multiplier ``exp(i k (n_ref - 1) dx)`` for one propagation slice.

        For a mixture, sum :meth:`susceptibility` over species first and pass the
        total through :meth:`index_minus_one` -- susceptibilities add, indices do
        not (and under :attr:`local_field` the distinction matters).
        """
        chi = self.susceptibility(density, delta, s)
        return np.exp(1j * self.k * self.index_minus_one(chi) * dx)

    # -------------------------------------------------------------- observables

    def cross_section(self, delta, s=0.0):
        """Extinction cross section ``sigma0 / (1 + delta^2 + s)``, m^2."""
        return self.sigma0 / (1 + np.asarray(delta, dtype=float)**2 + np.asarray(s))

    def scattering_rate(self, delta, s):
        """Photon scattering rate ``(Gamma/2) s / (1 + delta^2 + s)``, s^-1."""
        return 0.5 * self.Gamma * np.asarray(s) / \
            (1 + np.asarray(delta, dtype=float)**2 + np.asarray(s))

    def light_shift_Hz(self, delta, s):
        """Ground-state AC Stark shift ``(Gamma_Hz/4) s delta / (1+delta^2+s)``, Hz.

        Same denominator as :meth:`polarizability`, so the shift and the
        refraction stay consistent under saturation.
        """
        delta = np.asarray(delta, dtype=float)
        return 0.25 * self.linewidth_Hz * np.asarray(s) * delta / \
            (1 + delta**2 + np.asarray(s))

    def saturation_intensity(self):
        """``I_sat = pi h c Gamma / (3 lambda^3)`` for this transition, W/m^2."""
        return (np.pi * kc.h * kc.c * self.Gamma) / (3 * self.wavelength**3)

    # ------------------------------------------------------------ thin screen

    def thin_screen(self, column_density, species):
        """(D, phi) for a column of atoms -- the thin-phase-screen reference.

        ``species`` is a sequence of ``(population_fraction, delta)`` pairs sharing
        one spatial profile; their susceptibilities add, which is what a spin
        mixture does to the linear response.  Returns the optical depth
        ``D = -2 Re(exponent)`` and the phase ``phi = Im(exponent)`` of
        ``E_out / E_in = exp(i Phi)``.

        This is the SAME response the propagator applies slice by slice, taken over
        the whole column at once -- one definition of the atomic physics, two
        projections of it.  It is the correct answer only for an optically thin,
        diffraction-free cloud; comparing it against a propagation is how the
        thin-screen approximation gets falsified.
        """
        chi = sum(frac * self.susceptibility(column_density, d) for frac, d in species)
        expo = 1j * self.k * self.index_minus_one(chi)
        return -2 * np.real(expo), np.imag(expo)

    def __repr__(self):
        return (f"TwoLevelResponse(lambda={self.wavelength * 1e9:.4f} nm, "
                f"Gamma/2pi={self.linewidth_Hz / 1e6:.4f} MHz, "
                f"sigma0={self.sigma0:.6e} m^2"
                + (", local_field=True)" if self.local_field else ")"))
