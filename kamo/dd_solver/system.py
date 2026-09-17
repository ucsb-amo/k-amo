"""Operating point, dipole basis vectors and incident fields for the coupled-dipole solver.

Everything downstream works in the dimensionless units fixed here:

* frequencies and detunings in units of the natural linewidth ``Gamma``
  (``delta = (omega_laser - omega_atom) / Gamma``, so BLUE is positive);
* lengths in metres, with the laser wavenumber ``k = 2 pi / lambda``;
* dipole amplitudes ``beta_j`` in units of ``6 pi eps0 E0 / k^3``, where ``E0`` is
  the incident field amplitude, so that the scattered field is
  ``E_scat(r) / E0 = sum_j beta_j Gt(r - r_j) . e_hat`` with the dimensionless
  dyadic ``Gt`` of :mod:`kamo.dipole_dipole.green_tensor`;
* the drive ``Omega_j = conj(e_hat) . E_inc(r_j) / E0``.

In these units a two-level atom has the scalar polarizability
``alpha_tilde(delta) = -(1/2) / (delta + i/2)``, and the total power it scatters
is ``sigma0 I |beta|^2`` with ``sigma0 = 3 lambda^2 / 2 pi`` (sanity check S3 in
:mod:`kamo.dd_solver.solver` verifies this against the physical cross section).

Spin labels follow kamo's convention: ``|up> = |1,-1>`` (``m_I = -1/2``) is the
LOWER-frequency transition, so the midpoint probe is BLUE of it
(``delta_up = +9.14``) and RED of ``|dn> = |1,0>`` (``delta_dn = -9.14``).  The
build specification this package was written from had the opposite sign; see
the package README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

import kamo.constants as kc
from kamo.dipole_dipole.green_tensor import spherical_unit_vector

SPIN_UP = 1     #: |1,-1>, m_I = -1/2, the lower-frequency imaging transition
SPIN_DN = -1    #: |1,0>,  m_I = +1/2

#: Field inferred from the measured qubit splitting f(|1,-1> -> |1,0>) = 119.4639 MHz.
B_GAUSS_DEFAULT = 520.5830387285664

# Ground and excited state tuples (n, l, j, m_j, m_i) of the two closed sigma- lines.
G_UP, E_UP = (4, 0, 1 / 2, -1 / 2, -1 / 2), (4, 1, 3 / 2, -3 / 2, -1 / 2)
G_DN, E_DN = (4, 0, 1 / 2, -1 / 2, +1 / 2), (4, 1, 3 / 2, -3 / 2, +1 / 2)


@dataclass(frozen=True)
class Channel:
    """One dipole-allowed transition out of a ground state, other than the driven one.

    ``detuning`` is ``(omega_laser - omega_transition) / Gamma``; ``strength`` is
    ``|d_q / d_driven|^2``, the relative oscillator strength.
    """

    q: int
    detuning: float
    strength: float
    label: str = ""


@dataclass(frozen=True)
class OperatingPoint:
    """The laser, the two driven transitions and the excited-state sub-structure.

    Parameters
    ----------
    linewidth_Hz : float
        ``Gamma / 2 pi``.
    wavelength : float
        LASER wavelength (m); the Green tensor is evaluated at ``k = 2 pi / lambda``.
    delta_up, delta_dn : float
        Detunings of the laser from the ``|up>`` and ``|dn>`` sigma- lines, in
        units of ``Gamma``, positive = blue.
    channels_up, channels_dn : tuple of Channel
        Additional transitions from each ground state (pi, sigma+ and the weak
        hyperfine-admixture lines), used only by the vector solver.
    strength_up, strength_dn : float
        ABSOLUTE oscillator strength of each driven line,
        ``f = |d_dressed|^2 / |d_stretched|^2 = Gamma_partial / Gamma``, where
        ``Gamma = 1/tau`` is the full excited-state decay rate.  1.0 is the ideal
        closed two-level atom.  At 520.583 G the ground states carry a 2.3-2.5 %
        ``m_J = +1/2`` admixture, so kamo gives ``f_up = 0.9774`` and
        ``f_dn = 0.9786``: the polarizability, and hence every absolute
        amplitude, is 2.3 % smaller than the ideal value (added 2026-09-17).
        This is larger than the 0.98 % sigma+ background that used to be quoted
        as the leading correction to a scalar model.

        The line is then not perfectly closed: a fraction ``1 - f`` of the
        scattering is Raman to a state 1.3-1.5 GHz away that is dark to the
        probe, so the extinction exceeds the elastic scattering by ``1/f``.  That
        is why the optical theorem here reads
        ``beta^dag (diag(1/f) + G) beta = -Im(beta^dag Omega)``.
        At ``s0 = 0.2-0.3`` and a 5 us pulse this costs 0.07-0.11 % of the atoms,
        so depumping during the pulse is still negligible.
    """

    linewidth_Hz: float
    wavelength: float
    delta_up: float
    delta_dn: float
    B_gauss: float = float("nan")
    q: int = -1
    channels_up: Tuple[Channel, ...] = ()
    channels_dn: Tuple[Channel, ...] = ()
    provenance: str = ""
    strength_up: float = 1.0
    strength_dn: float = 1.0

    # ------------------------------------------------------------ derived
    @property
    def k(self) -> float:
        """Laser wavenumber (1/m)."""
        return 2 * np.pi / self.wavelength

    @property
    def Gamma(self) -> float:
        """Natural linewidth in rad/s."""
        return 2 * np.pi * self.linewidth_Hz

    @property
    def sigma0(self) -> float:
        """Resonant cross section ``3 lambda^2 / 2 pi`` (m^2)."""
        return 3 * self.wavelength ** 2 / (2 * np.pi)

    @property
    def e_hat(self) -> np.ndarray:
        """Unit vector of the driven dipole (``(x - i y)/sqrt2`` for sigma-)."""
        return spherical_unit_vector(self.q)

    @property
    def splitting(self) -> float:
        """``|delta_up - delta_dn|`` in units of Gamma."""
        return abs(self.delta_up - self.delta_dn)

    @property
    def condon_radius(self) -> float:
        """``(3 / 4|delta|)^{1/3} / k`` (m): the STATIC near-field Condon radius,
        maximised over angles (``B = 1``, pair along ``B``), without retardation.

        A scale, not a prediction (clarified 2026-09-17).  With the package's own
        retarded ``J`` the shell moves: 51.6 nm rather than 53.0 nm for a polar
        pair, and 44.6 rather than 42.1 nm for an in-plane one.  Which pairs are
        BRIGHT-resonant also depends on the sign of the detuning, because the
        symmetric (superradiant) mode is shifted by ``+J``: blue detuning is
        matched by polar pairs (``B > 0``, up to ``B = 1``) and red detuning by
        in-plane pairs, where ``|B| <= 1/2``, so the red bright shell is smaller
        by ``2^{-1/3}`` and encloses 0.6x the volume.  For a retarded,
        angle- and branch-resolved radius use
        :class:`kamo.dipole_dipole.PairPotential`.
        """
        return (0.75 / abs(self.delta_up)) ** (1 / 3) / self.k

    def detunings(self, spins) -> np.ndarray:
        """Per-atom detuning (Gamma units) from a spin array (+1 up, -1 dn)."""
        spins = np.asarray(spins)
        return np.where(spins > 0, self.delta_up, self.delta_dn).astype(float)

    def strengths(self, spins) -> np.ndarray:
        """Per-atom oscillator strength ``f_j`` from a spin array (+1 up, -1 dn)."""
        spins = np.asarray(spins)
        return np.where(spins > 0, self.strength_up, self.strength_dn).astype(float)

    @property
    def mean_strength(self) -> float:
        """``(f_up + f_dn)/2`` -- what to scale a single-species scalar model by."""
        return 0.5 * (self.strength_up + self.strength_dn)

    def polarizability_scalar(self, delta, strength=1.0):
        """Dimensionless two-level polarizability ``-(f/2) / (delta + i/2)``."""
        return -0.5 * np.asarray(strength, dtype=float) / (np.asarray(delta, dtype=float) + 0.5j)

    def polarizability_SI(self, delta, strength=1.0):
        """The same in SI (C m^2 / V): ``(6 pi eps0 / k^3) alpha_tilde``."""
        return 6 * np.pi * kc.epsilon0 / self.k ** 3 * self.polarizability_scalar(delta, strength)

    def channels(self, spin: int) -> Tuple[Channel, ...]:
        return self.channels_up if spin > 0 else self.channels_dn

    # ---------------------------------------------------- kamo.imaging bridge
    def response(self, sigma0_scale: float = 1.0):
        """A :class:`kamo.imaging.TwoLevelResponse` at this wavelength and linewidth.

        ``sigma0_scale`` rescales the cross section.  Pass
        ``abs(conj(e_hat) . polarization)**2`` (1/2 for a y-polarized Voigt probe)
        to make the scalar propagation see only the sigma- projection of the
        incident polarization, which is what the microscopic model does, and
        multiply by :attr:`mean_strength` as well when this operating point
        carries an oscillator strength below 1, since
        :class:`~kamo.imaging.response.TwoLevelResponse` is an ideal closed
        two-level line.  :func:`kamo.dd_solver.compare_bpm.bpm_result` does both.
        """
        from kamo.imaging.response import TwoLevelResponse
        return TwoLevelResponse(self.wavelength, self.linewidth_Hz,
                                sigma0=self.sigma0 * float(sigma0_scale))

    def species(self, theta: float = 0.0):
        """``((p_up, 2 delta_up), (p_dn, 2 delta_dn))`` for :mod:`kamo.imaging`.

        kamo.imaging's reduced detuning is ``2 Delta / Gamma``, twice this
        package's ``delta``.
        """
        p_up = float(np.cos(0.5 * theta) ** 2)
        return ((p_up, 2 * self.delta_up), (1 - p_up, 2 * self.delta_dn))

    # ------------------------------------------------------------ builders
    @classmethod
    def nominal(cls) -> "OperatingPoint":
        """The K-team operating point with kamo's numbers hard-coded (no ARC access).

        From ``Potassium39`` (portal matrix elements, kamo hyperfine constants)
        at B = 520.583 G on 2026-09-16: Gamma/2pi = 6.0309 MHz, the two sigma-
        lines 110.235 MHz = 18.278 Gamma apart, laser at their midpoint
        (lambda = 766.70226 nm), excited 4P3/2 m_J branches 160.9 Gamma apart.
        Channel strengths are the dressed-state ratios from
        :meth:`kamo.dipole_dipole.CyclingTransition.at_field` (0.668 and 0.334
        rather than the bare 2/3 and 1/3), RELATIVE to the driven line, whose own
        absolute strength is ``strength_up/dn`` (0.977).

        Corrected 2026-09-17: the wavelength used to be 766.70215 nm, which is
        the ``|dn>`` LINE, not the midpoint laser this docstring describes (the
        0.108 pm difference is exactly half the line splitting; it moves ``k`` by
        1.4e-7 and nothing else).  The absolute oscillator strengths are new.
        """
        d = 9.139185702159367
        ch_up = (Channel(0, d - 160.928, 0.6676, "pi (m_J'=-1/2)"),
                 Channel(+1, d - 321.618, 0.3341, "sigma+ (m_J'=+1/2)"))
        ch_dn = (Channel(0, -d - 161.921, 0.6679, "pi (m_J'=-1/2)"),
                 Channel(+1, -d - 323.621, 0.3346, "sigma+ (m_J'=+1/2)"))
        return cls(linewidth_Hz=6.030880753766402e6, wavelength=766.7022612592e-9,
                   delta_up=+d, delta_dn=-d, B_gauss=B_GAUSS_DEFAULT, q=-1,
                   channels_up=ch_up, channels_dn=ch_dn,
                   strength_up=0.9774222, strength_dn=0.9786171,
                   provenance="kamo Potassium39 (portal E1, kamo hyperfine, NIST "
                              "energies) at 520.583 G, hard-coded 2026-09-16, "
                              "strengths and midpoint wavelength 2026-09-17")

    @classmethod
    def from_kamo(cls, atom=None, B_gauss: float = B_GAUSS_DEFAULT,
                  strength_min: float = 1e-3, offset_Hz: float = 0.0) -> "OperatingPoint":
        """Build from :class:`kamo.Potassium39` at a field (opens the ARC database).

        The laser sits at the midpoint of the two sigma- lines plus ``offset_Hz``.
        Extra channels come from :meth:`CyclingTransition.at_field` and are kept
        when their relative strength exceeds ``strength_min``.
        """
        from kamo.imaging import ProbeBeam
        from kamo.dipole_dipole import CyclingTransition
        from kamo.hamiltonian import AtomicStructure

        if atom is None:
            from kamo import Potassium39
            atom = Potassium39()
        probe = ProbeBeam.from_midpoint(atom, B_gauss=B_gauss, ground_up=G_UP,
                                        excited_up=E_UP, ground_dn=G_DN,
                                        excited_dn=E_DN, offset_Hz=offset_Hz)
        lw = probe.response.linewidth_Hz
        d_up = probe.detuning_up_Hz / lw
        d_dn = probe.detuning_dn_Hz / lw

        # Pass the caller's atom through: CyclingTransition.at_field otherwise
        # falls back to kamo's DEFAULT atom for the channel structure while the
        # detunings come from `atom`, a silent inconsistency for any other atom
        # or isotope, and it opens ARC a second time (fixed 2026-09-17).
        model = AtomicStructure([(4, 0, 0.5), (4, 1, 0.5), (4, 1, 1.5)], atom=atom)
        out, strengths = [], []
        for g, e, d0 in ((G_UP, E_UP, d_up), (G_DN, E_DN, d_dn)):
            t = CyclingTransition.at_field(B_gauss, ground=g, excited=e, model=model)
            # Absolute strength of the driven line: |d_dressed|^2 / |d_stretched|^2,
            # with |d_stretched|^2 = 3 pi eps0 hbar Gamma / k^3 from the lifetime.
            d2_stretched = 3 * np.pi * kc.epsilon0 * kc.hbar * (2 * np.pi * t.linewidth_Hz) / t.k ** 3
            strengths.append(float(t.d_Cm ** 2 / d2_stretched))
            chans = []
            for ch in t.channels:
                if abs(ch.detuning_Hz) < 1e-3 and ch.q == -1:
                    continue                          # the driven line itself
                if ch.relative_strength < strength_min:
                    continue
                chans.append(Channel(ch.q, d0 - ch.detuning_Hz / lw, ch.relative_strength,
                                     f"{ch.polarization} (m_J'={ch.m_j:+.1f}, m_I={ch.m_i:+.1f})"))
            out.append(tuple(chans))
        return cls(linewidth_Hz=lw, wavelength=kc.c / probe.frequency_Hz,
                   delta_up=float(d_up), delta_dn=float(d_dn), B_gauss=float(B_gauss),
                   q=-1, channels_up=out[0], channels_dn=out[1],
                   strength_up=strengths[0], strength_dn=strengths[1],
                   provenance=f"kamo Potassium39 at {B_gauss:.4f} G (portal E1 matrix "
                              "elements, kamo hyperfine constants, NIST energies)")

    def summary(self) -> str:
        lines = [f"OperatingPoint  B = {self.B_gauss:.4f} G   ({self.provenance})",
                 f"  Gamma/2pi = {self.linewidth_Hz / 1e6:.4f} MHz   lambda = "
                 f"{self.wavelength * 1e9:.4f} nm   sigma0 = {self.sigma0:.4e} m^2",
                 f"  delta_up = {self.delta_up:+.4f} Gamma   delta_dn = {self.delta_dn:+.4f} Gamma"
                 f"   (splitting {self.splitting:.3f} Gamma = "
                 f"{self.splitting * self.linewidth_Hz / 1e6:.3f} MHz)",
                 f"  Condon radius at |delta| = {abs(self.delta_up):.2f}: "
                 f"{self.condon_radius * 1e9:.1f} nm (static, max over angles)",
                 f"  oscillator strength f_up = {self.strength_up:.5f}, "
                 f"f_dn = {self.strength_dn:.5f} (1 = ideal closed line)"]
        for name, chs in (("up", self.channels_up), ("dn", self.channels_dn)):
            for ch in chs:
                lines.append(f"  |{name}> extra channel q={ch.q:+d}: delta = {ch.detuning:+9.2f} "
                             f"Gamma, strength {ch.strength:.4f}  {ch.label}")
        return "\n".join(lines)


# ---------------------------------------------------------------- incident


class IncidentField:
    """Base class: a monochromatic field ``E_inc(r) / E0`` on arbitrary points.

    Subclasses implement :meth:`field`; :meth:`drive` projects it on the driven
    dipole.  The reference amplitude ``E0`` is the peak (focus / plane-wave)
    amplitude, so ``|field| <= 1``.
    """

    polarization: np.ndarray
    khat: np.ndarray
    k: float

    def field(self, points) -> np.ndarray:
        raise NotImplementedError

    def drive(self, points, e_hat) -> np.ndarray:
        """``Omega_j = conj(e_hat) . E_inc(r_j) / E0``."""
        return self.field(points) @ np.conj(np.asarray(e_hat))

    def projection(self, e_hat) -> float:
        """``|conj(e_hat) . polarization|^2`` for an arbitrary driven dipole.

        1/2 for the lab's y-polarized Voigt probe on a sigma- line, 0 for the
        same probe on a pi line.
        """
        return float(abs(np.conj(np.asarray(e_hat, dtype=complex)) @ self.polarization) ** 2)

    @property
    def sigma_projection(self) -> float:
        """``|conj(e_-1) . polarization|^2``: 1/2 for y light.

        Hard-wired to ``q = -1``.  For an :class:`OperatingPoint` with a
        different ``q`` use ``incident.projection(op.e_hat)`` -- this property
        would return 1/2 where the true projection is 0 (noted 2026-09-17).
        """
        return float(abs(np.conj(spherical_unit_vector(-1)) @ self.polarization) ** 2)


class PlaneWave(IncidentField):
    """Unit-amplitude plane wave ``pol exp(i k khat . r)``.

    Default: propagating along +x, polarized along y (the lab's Voigt geometry,
    ``B || z``).
    """

    def __init__(self, k: float, khat=(1.0, 0.0, 0.0), polarization=(0.0, 1.0, 0.0)):
        self.k = float(k)
        kh = np.asarray(khat, dtype=float)
        self.khat = kh / np.linalg.norm(kh)
        pol = np.asarray(polarization, dtype=complex)
        if abs(np.conj(pol) @ self.khat) > 1e-12:
            raise ValueError("polarization must be transverse to khat")
        self.polarization = pol / np.linalg.norm(pol)

    def field(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.exp(1j * self.k * (pts @ self.khat))[:, None] * self.polarization[None, :]

    def __repr__(self):
        return f"PlaneWave(khat={self.khat.tolist()}, pol={self.polarization.tolist()})"


class GaussianBeam(IncidentField):
    """Paraxial Gaussian beam along +x, focus at ``x0``, ``1/e^2`` intensity waist ``w0``.

    ``E / E0 = pol (w0 / w) exp(-rho^2 / w^2) exp(i [k x + k rho^2 / 2R - psi])``
    with the usual ``w(x)``, ``R(x)`` and Gouy phase ``psi(x)``.  For the lab probe
    (``w0 >> 20 um`` over a cloud of ``0.4 um`` radius) the intensity varies by
    less than 0.1% across the atoms, which is why :class:`PlaneWave` is the
    default; this class is here so that statement can be checked rather than
    assumed.
    """

    def __init__(self, k: float, waist: float, x0: float = 0.0,
                 polarization=(0.0, 1.0, 0.0)):
        self.k = float(k)
        self.w0 = float(waist)
        self.x0 = float(x0)
        self.khat = np.array([1.0, 0.0, 0.0])
        pol = np.asarray(polarization, dtype=complex)
        if abs(pol[0]) > 1e-12:
            raise ValueError("polarization must be transverse to x")
        self.polarization = pol / np.linalg.norm(pol)
        self.x_R = 0.5 * self.k * self.w0 ** 2

    def field(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        x = pts[:, 0] - self.x0
        rho2 = pts[:, 1] ** 2 + pts[:, 2] ** 2
        w = self.w0 * np.sqrt(1 + (x / self.x_R) ** 2)
        with np.errstate(divide="ignore"):
            inv_R = x / (x ** 2 + self.x_R ** 2)
        psi = np.arctan(x / self.x_R)
        amp = (self.w0 / w) * np.exp(-rho2 / w ** 2)
        phase = self.k * pts[:, 0] + 0.5 * self.k * rho2 * inv_R - psi
        return (amp * np.exp(1j * phase))[:, None] * self.polarization[None, :]

    def __repr__(self):
        return f"GaussianBeam(w0={self.w0 * 1e6:.1f} um, x0={self.x0 * 1e6:.1f} um)"


def default_incident(op: OperatingPoint) -> PlaneWave:
    """The lab probe: plane wave along +x, polarized along y."""
    return PlaneWave(op.k)
