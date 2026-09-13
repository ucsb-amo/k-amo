import arc
from arc.wigner import Wigner6j, Wigner3j
import numpy as np
import kamo.constants as c
# import pairinteraction.real as pi

dv = -1000.

# States with n below this threshold use kamo.hamiltonian exact diagonalization,
# which includes nuclear spin (m_j, m_i basis).  States at or above use
# pairinteraction, which has no nuclear spin and works in the m_j basis only.
_HAMILTONIAN_N_THRESHOLD = 10

# ARC's sign is trusted when its magnitude is within this factor of the portal's.
_PORTAL_SIGN_TOLERANCE = 1.25


def _angular_factor(l1, j1, l2, j2, s=0.5):
    """|<j1||er||j2>| / |radial element|, in ARC's convention."""
    return abs(np.sqrt((2*j1 + 1) * (2*j2 + 1) * (2*l1 + 1) * (2*l2 + 1))
               * Wigner6j(j1, 1, j2, l2, s, l1) * Wigner3j(l1, 1, l2, 0, 0, 0))


def _own_manifolds(*states):
    """The ``(n, l, j)`` manifolds of ``states`` plus their fine-structure
    partners (same n and l, other j), in first-seen order."""
    out = []
    for st in states:
        n, l = int(st[0]), int(st[1])
        js = (float(st[2]),) if l == 0 else (l - 0.5, l + 0.5)
        for j in js:
            if (n, l, j) not in out:
                out.append((n, l, j))
    return out


class Potassium39(arc.Potassium39):
    """ARC's Potassium39, with E1 data from the UDel portal by default.

    With ``use_portal=True`` every quantity ARC builds from its radial matrix
    element (``getDipoleMatrixElement``, ``getReducedMatrixElementJ``, Rabi
    couplings, ...) uses the portal's recommended magnitude. The sign still
    comes from ARC, which the portal does not provide. At zero temperature,
    ``getTransitionRate`` and ``getStateLifetime`` return the portal's rates
    and lifetimes (measured values where they exist, e.g. 4P). Transitions or
    states the portal lacks fall back to ARC. ``getHFSCoefficients`` returns
    kamo's hyperfine constants (:mod:`kamo.atom_properties.hyperfine`) in
    place of ARC's 1977 table. ``use_portal=False`` gives ARC's own E1 and
    hyperfine data.

    Energies are ARC's tabulated NIST levels (``preferQuantumDefects=False``).
    ARC's default computes every K level from Rydberg quantum defects instead,
    which puts D2 2.55 GHz and D1 4.71 GHz too high and 3D 67 cm^-1 too low.
    Pass ``preferQuantumDefects=True`` to get those energies back (before
    2026-09 kamo used them).
    """

    # ARC leaves K39's nuclear g-factor at 0, which drops the nuclear Zeeman
    # term from its getLandegfExact and breitRabi. Same sign convention
    # (H = mu_B B (g_J J_z + g_I I_z)).
    gI = c.g_I

    def __init__(self, use_portal=True, portal_species="K1", preferQuantumDefects=False):
        self.use_portal = use_portal
        self.portal_species = portal_species
        self._portal_data = None
        super().__init__(preferQuantumDefects=preferQuantumDefects)
        self.cross_section = self.get_cross_section()

    # ------------------------------------------------------------ portal data

    @staticmethod
    def _key(n, l, j):
        return (round(n), round(l), round(2 * j))

    def _portal(self):
        """Portal radial elements, rates and lifetimes, loaded on first use."""
        if self._portal_data is None:
            from kamo.light_shift import udel_portal
            me = udel_portal.dedupe_pairs(udel_portal.matrix_elements(self.portal_species))
            me = me[me.n1.notna() & me.n2.notna()]
            radial = {
                frozenset((self._key(r.n1, r.l1, r.J1), self._key(r.n2, r.l2, r.J2))):
                    r.d_au / _angular_factor(round(r.l1), r.J1, round(r.l2), r.J2)
                for r in me.itertuples()}
            tr = udel_portal.transition_rates(self.portal_species)
            tr = tr[tr.n1.notna() & tr.n2.notna()]
            rates = {(self._key(r.n1, r.l1, r.J1), self._key(r.n2, r.l2, r.J2)): r.A_s
                     for r in tr.itertuples()}
            lifetimes = {self._key(r.n1, r.l1, r.J1): r.tau_s for r in tr.itertuples()}
            self._portal_data = {"radial": radial, "rates": rates, "lifetimes": lifetimes,
                                 "signed": {}, "sign_uncertain": set()}
        return self._portal_data

    def getRadialMatrixElement(self, n1, l1, j1, n2, l2, j2, s=0.5, useLiterature=True):
        """ARC's radial element, rescaled to the portal's magnitude.

        ``useLiterature=False`` returns ARC's own numerical integral.
        """
        if not (self.use_portal and useLiterature):
            return super().getRadialMatrixElement(n1, l1, j1, n2, l2, j2, s=s,
                                                  useLiterature=useLiterature)
        portal = self._portal()
        pair = frozenset((self._key(n1, l1, j1), self._key(n2, l2, j2)))
        if pair not in portal["signed"]:
            arc_value = super().getRadialMatrixElement(n1, l1, j1, n2, l2, j2, s=s)
            magnitude = portal["radial"].get(pair)
            if magnitude is None or arc_value == 0:
                value = arc_value
            else:
                value = float(np.copysign(magnitude, arc_value))
                ratio = abs(arc_value) / magnitude
                if not 1 / _PORTAL_SIGN_TOLERANCE < ratio < _PORTAL_SIGN_TOLERANCE:
                    portal["sign_uncertain"].add(pair)
            portal["signed"][pair] = value
        return portal["signed"][pair]

    def portal_sign_uncertain(self, n1, l1, j1, n2, l2, j2):
        """True if ARC's sign for this transition should not be trusted.

        That is the case when ARC's magnitude is off from the portal's by more
        than ``_PORTAL_SIGN_TOLERANCE``. For K these are all small elements
        (< 0.25 a.u.): 4S-nP with n >= 8, and 4P-4D. Signs matter only where
        different matrix elements interfere (Raman couplings, D1/D2 in one
        coupling); polarizabilities use |d|^2 alone.
        """
        if not self.use_portal:
            return False
        self.getRadialMatrixElement(n1, l1, j1, n2, l2, j2)
        pair = frozenset((self._key(n1, l1, j1), self._key(n2, l2, j2)))
        return pair in self._portal()["sign_uncertain"]

    def getTransitionRate(self, n1, l1, j1, n2, l2, j2, temperature=0.0, s=0.5):
        """Rate (s^-1) from state 1 to state 2.

        At zero temperature this is the portal's Einstein A when it lists the
        decay channel. At finite temperature it is ARC's formula (portal matrix
        element, ARC transition frequency).
        """
        if self.use_portal and not temperature:
            rate = self._portal()["rates"].get((self._key(n1, l1, j1), self._key(n2, l2, j2)))
            if rate is not None:
                return rate
        return super().getTransitionRate(n1, l1, j1, n2, l2, j2, temperature=temperature, s=s)

    def getStateLifetime(self, n, l, j, temperature=0, includeLevelsUpTo=0, s=0.5):
        """Lifetime (s). At zero temperature this is the portal's recommended
        value when it has one; otherwise ARC's sum over decay rates."""
        if self.use_portal and temperature < 0.1:
            tau = self._portal()["lifetimes"].get(self._key(n, l, j))
            if tau is not None:
                return tau
        return super().getStateLifetime(n, l, j, temperature=temperature,
                                        includeLevelsUpTo=includeLevelsUpTo, s=s)

    def getHFSCoefficients(self, n, l, j, s=None):
        """Hyperfine ``(A, B)`` in Hz from :mod:`kamo.atom_properties.hyperfine`.

        ARC's own table is Arimondo 1977: its 4P_1/2 A is 28.85 MHz, against
        27.793(71) MHz recommended now. ARC's hyperfine helpers
        (``getHFSEnergyShift`` users) call this method, so they pick up the new
        values too. ``use_portal=False`` gives ARC's table back.
        """
        if not self.use_portal:
            return super().getHFSCoefficients(n, l, j, s=s)
        from kamo.atom_properties.hyperfine import hyperfine_constants
        hc = hyperfine_constants(n, l, j)
        if not hc.has_A:
            raise ValueError(f"No hyperfine data for state ({n}, {l}, {j}).")
        return hc.A_Hz, hc.B_Hz

    def breitRabi(self, n, l, j, B):
        """Zeeman + hyperfine energies (Hz) of manifold ``(n, l, j)`` at fields
        ``B`` (tesla), in ARC's return format ``(energies, F, mF)``.

        ARC's version halves the quadrupole term: its B denominator is
        ``2I(2I-1) 2J(2J-1)`` instead of ``2I(2I-1) J(2J-1)``. That puts the
        4P_3/2 F'=0 level 1.8 MHz off. This one is built from kamo.hamiltonian,
        so it uses the same A, B, g_J and g_I as every other kamo Zeeman
        calculation (no diamagnetic term, as in ARC). As in ARC, each row of
        energies is sorted ascending (not tracked). F and mF label the columns
        by their order at a field whose Zeeman energy is 1e-4 of the smallest
        hyperfine gap. ARC always labels at 1e-4 T, where small-A manifolds
        (5P_3/2 and up) are already F-mixed and come out with half-integer F.
        A manifold with no hyperfine A (l >= 3) raises ``ValueError``, as
        ``getHFSCoefficients`` does. ``use_portal=False`` gives ARC's original.
        """
        if not self.use_portal:
            return super().breitRabi(n, l, j, B)
        from kamo.atom_properties.hyperfine import hyperfine_constants, hyperfine_energy
        from kamo.hamiltonian.basis import Basis
        from kamo.hamiltonian.builder import HamiltonianBuilder
        hc = hyperfine_constants(n, l, j)
        if not hc.has_A:
            raise ValueError(f"No hyperfine data for state ({n}, {l}, {j}).")
        builder = HamiltonianBuilder(Basis([(n, l, j)]), atom=self)
        h0 = builder.h0()
        zeeman = builder.zeeman_operator() * 1e4          # Hz/G -> Hz/T
        B = np.atleast_1d(np.asarray(B, dtype=float))
        energies = np.array([np.linalg.eigvalsh(h0 + b * zeeman) for b in B])
        levels = np.sort([hyperfine_energy(F, builder.I, j, hc.A_Hz, hc.B_Hz)
                          for F in builder.basis.manifolds[0].allowed_F()])
        gap = np.diff(levels)
        gap = gap[gap > 0].min() if np.any(gap > 0) else abs(hc.A_Hz)
        B_label = 1e-4 * gap / np.abs(zeeman).max()       # tesla
        _, vecs = np.linalg.eigh(h0 + B_label * zeeman)
        IJ = builder._ij_operator(j, builder.I)
        F2 = (j * (j + 1) + builder.I * (builder.I + 1)) * np.eye(len(IJ)) + 2 * IJ
        f2 = np.einsum("ik,ij,jk->k", vecs, F2, vecs)
        F = np.round(-1 + np.sqrt(1 + 4 * f2)) / 2
        mF = np.round(2 * (np.array([s.m_f for s in builder.basis]) @ vecs ** 2)) / 2 + 0.0
        return energies, F, mF

    # def init_pairinteraction(self):
    #     if pi.Database.get_global_database() is None:
    #         pi.Database.initialize_global_database(download_missing=True)

    def get_magnetic_field_from_ground_state_transition_frequency(self,
                                                                f1, mf1, f2, mf2, transition_frequency_Hz,
                                                                B_bounds_G=(0., 600.),
                                                                N_interp=10000,
                                                                B_guess=None):
        """Returns the magnetic field(s) (in G) at which the transition from
        (f1,mf1) to (f2,mf2) would occur at frequency 'transition_frequency_Hz'.

        Args:
            f1 (int): State 1 quantum number F.
            mf1 (int): State 1 quantum number mF.
            f2 (int): State 2 quantum number F.
            mf2 (int): State 2 quantum number mF.
            transition_frequency_Hz (float or array-like): Measured transition frequency
                between (f1,mF1) and (f2,mF2).
            B_bounds_G (tuple, optional): Bounds used for field finding. Only
                limited to save time computing all the possible transition frequencies. Defaults to (0.,600.).
            N_interp (int, optional): Number of points used for interpolation. Defaults to 10000.
            B_guess (float, optional): Field (G) used to pick a branch when the
                splitting is non-monotonic and a target has several solutions.

        Raises:
            ValueError: If a target is never reached within the bounds, or if it
                has multiple branches and no ``B_guess`` was given.

        Returns:
            float or np.ndarray: the magnetic field(s) in G.
        """
        # Ground-state (4S_1/2) special case of the general splitting inverter;
        # both states share a manifold, so a single sweep serves all targets.
        return self.get_magnetic_field_from_splitting(
            (4, 0, 0.5, int(f1), int(mf1)),
            (4, 0, 0.5, int(f2), int(mf2)),
            transition_frequency_Hz,
            B_bounds_G=B_bounds_G,
            n_points=N_interp,
            B_guess=B_guess,
        )
    
    def get_ground_state_transition_sensitivity(self,f1,mf1,f2,mf2,B):
        """Returns the ground state transition sensitivity in MHz/G for (f1,mf1) to
        (f2,mf2) at field B.

        Args:
            f1 (int): State 1 quantum number F.
            mf1 (int): State 1 quantum number mF.
            f2 (int): State 2 quantum number F.
            mf2 (int): State 2 quantum number mF.
            B (float): Magnetic field in G.

        Returns:
            float: ground state transition sensitivity in MHz/G.
        """        
        dB = B * 0.001
        # Both fields come from a single sweep (array B) instead of two calls.
        f_B, f_B_plus_dB = self._splitting_mhz(
            (4, 0, 0.5, int(f1), int(mf1)),
            (4, 0, 0.5, int(f2), int(mf2)),
            np.array([B, B + dB]),
        )
        return (f_B_plus_dB - f_B) / dB
    
    def _zeeman_hamiltonian_multi(self, states, B_gauss, B_sweep_steps=500):
        """Run one kamo.hamiltonian sweep covering all requested states.

        Parameters
        ----------
        states : list of ``(n, l, j, m_j, m_i)`` tuples
        B_gauss : scalar or 1-D array

        Returns
        -------
        energies : list of ndarray (MHz), one per entry in *states*
        sweep : MagneticSweepResult
        """
        from kamo.hamiltonian import AtomicStructure
        B_arr = np.atleast_1d(np.asarray(B_gauss, dtype=float))
        B_max = max(float(np.max(B_arr)), 0.01)
        dB = min(B_max / B_sweep_steps, 0.01)
        if dB > 0.1:
            print(f"Sweep steps are large ({dB:1.2e} G per step). Consider increasing sampling if adiabatic state detection suffers.")

        # Collect unique (n, l, j) fine-structure levels spanning all requests
        njl_levels = list(dict.fromkeys((s[0], s[1], s[2]) for s in states))
        model = AtomicStructure(njl_levels)
        res = model.magnetic_sweep(B_max=B_max, dB=dB, diamagnetic=True, include_quadrupole=True)

        energies = [res.get_energy(n, l, j, m_j, m_i, at=B_arr) / 1e6
                    for n, l, j, m_j, m_i in states]
        return energies, res

    def _zeeman_hamiltonian(self, n, l, j, m_j, m_i, B_gauss):
        """Energy track (MHz) via kamo.hamiltonian exact diagonalization.

        Uses the adiabatic (Paschen-Back) connection: ``m_j`` and ``m_i`` are
        the high-field limiting quantum numbers, resolved to a tracked state by
        :meth:`~kamo.hamiltonian.SweepResult.get_energy`.

        ``B_gauss`` may be a scalar or 1-D array; returns ``(energies_MHz, sweep_result)``.
        """
        (energy,), res = self._zeeman_hamiltonian_multi([(n, l, j, m_j, m_i)], B_gauss)
        return energy, res

    def _zeeman_pairinteraction(self, n, l, j, m_j, B_gauss):
        """Zeeman *shift* from zero field (MHz) via pairinteraction.

        Only uses ``m_j`` (pairinteraction has no nuclear spin).
        ``B_gauss`` may be a scalar or 1-D array; returns matching ndarray.
        """
        import pairinteraction as pi
        B_arr = np.atleast_1d(np.asarray(B_gauss, dtype=float))
        shifts = np.zeros(len(B_arr))
        ket = pi.KetAtom("K", n=n, l=l, j=j, m=m_j)
        zero_field_energy = ket.get_energy()
        l_max = min(l + 2, n - 1)
        basis = pi.BasisAtom("K", n=(n - 3, n + 3), l=(l, l_max))
        for idx, b_val in enumerate(B_arr):
            system = pi.SystemAtom(basis)
            system.set_diamagnetism_enabled(True)
            system.set_magnetic_field([0.0, 0.0, float(b_val)], unit="gauss")
            pi.diagonalize([system])
            shifted = system.get_corresponding_energy(ket)
            shifts[idx] = (shifted - zero_field_energy).to("J").magnitude / c.h / 1e6
        return shifts

    def get_semiclassical_polarizability(self,n1,l1,j1,n2,l2,j2,detuning_Hz):
        """See Grimm 1999 equation 8.
        """        
        f0 = np.abs(self.getTransitionFrequency(n1,l1,j1,n2,l2,j2))
        omega0 = 2 * np.pi * f0
        omega = 2 * np.pi * (f0 + detuning_Hz)
        linewidth = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        return 6 * np.pi * c.epsilon0 * c.c**3 * \
            ( linewidth / omega0**2 ) / ( omega0**2 - omega**2 - 1j * (omega**3/omega0**2) * linewidth )
    
    def get_scattering_rate(self,
                            n1,l1,j1,
                            n2,l2,j2,
                            intensity,
                            detuning_Hz=100.e6):
        """Photon scattering rate in s^-1.  See Grimm 1999 equation 9::

            Gamma_sc = Im(alpha) * I / (hbar * eps0 * c)

        The result is a genuine rate (photons per second), not an angular
        frequency: `get_decay_rate` supplies Gamma = 1/tau as a population
        decay constant in s^-1, and eq. 9 carries that through unchanged.
        There is no 1/(2 pi) in Grimm's expression -- dividing by 2 pi here
        would under-report the rate by a factor of 2 pi (it would, for
        example, cap the resonant saturated rate at Gamma/(4 pi) instead of
        the correct Gamma/2).
        """
        alpha = self.get_semiclassical_polarizability(n1,l1,j1,n2,l2,j2,detuning_Hz)
        return 1/(c.hbar * c.epsilon0 * c.c) * np.imag(alpha) * intensity

    # def get_off_resonant_scattering_rate(self,
    #                         n1,l1,j1,
    #                         n2,l2,j2,
    #                         intensity,
    #                         detuning_Hz=100.e6):
    #     omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
    #     linewidth = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
    #     Delta = 2 * np.pi * detuning_Hz
    #     return 3 * np.pi * c.c**2 / (2 * c.hbar * omega0**3) * (linewidth/Delta)**2 * intensity

    def get_decay_rate(self,n1,l1,j1,n2,l2,j2):
        '''
        Returns spontaneous emission rate for the higher of two states in 1/s.
        '''
        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            Gamma = 1/self.getStateLifetime(n2,l2,j2)
        else:
            Gamma = 1/self.getStateLifetime(n1,l1,j1)
        return Gamma

    def get_saturation_intensity(self,
                                 n1,l1,j1,
                                 n2,l2,j2,
                                 detuning_Hz=0.,
                                 convert_to_mW_per_cm2=False):
        '''
        Returns the off-resonant (effective) saturation intensity in W/m^2 for
        the transition between the two given states.

        The on-resonance two-level value is

            I_sat = pi h c Gamma / (3 lambda^3),

        with Gamma the spontaneous decay rate of the upper state (from
        `get_decay_rate`) and lambda the transition wavelength.  A laser detuned
        by Delta saturates the transition more slowly, so the intensity at which
        the excited-state population reaches 1/4 grows as

            I_sat(Delta) = I_sat * ( 1 + (2 Delta / Gamma)^2 ).

        This is the quantity that makes the scattering rate

            R = (Gamma/2) * (I / I_sat(Delta)) / (1 + I / I_sat(Delta))

        equal to the usual detuned two-level result.

        Parameters
        ----------
        n1,l1,j1 : lower/upper state quantum numbers (order does not matter;
            the higher-lying state supplies the linewidth)
        n2,l2,j2 : the other state of the pair
        detuning_Hz: float
            The laser detuning from resonance, as an ordinary frequency in Hz
            (not angular).  Default 0., which gives the resonant I_sat.  Only
            the magnitude matters.
        convert_to_mW_per_cm2: bool
            If True, converts the output to mW/cm^2 before returning.

        Returns
        -------
        float

        Notes
        -----
        This is the two-level (cycling-transition) saturation intensity: it
        carries no Clebsch-Gordan factor for a particular m_F -> m_F' pair, and
        no polarization dependence.  The states are used only to fix Gamma and
        the transition wavelength.
        '''
        Gamma = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        f0 = np.abs(self.getTransitionFrequency(n1,l1,j1,n2,l2,j2))
        wavelength = c.c / f0

        saturation_intensity = np.pi * c.h * c.c * Gamma / (3 * wavelength**3)
        detuning_factor = 1 + (2 * 2 * np.pi * np.asarray(detuning_Hz) / Gamma)**2

        convert_W_per_m2_to_mW_per_cm2 = 0.1
        if convert_to_mW_per_cm2:
            convert = convert_W_per_m2_to_mW_per_cm2
        else:
            convert = 1

        return saturation_intensity * detuning_factor * convert
        
    def lineshape(self,n1=4,l1=0,j1=1/2,n2=4,l2=1,j2=3/2,detuning_Hz=0):
        '''
        Returns the lineshape evaluated at a given detuning for a two-level system. Does not work for excited states.
        '''
        if n2 > 4 or n1 > 4 or l1 > 1 or l2 > 1:
            print("Lineshape not accurate for excited states with n>4.")
        gamma = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        # transition_omega = np.abs( self.getTransitionFrequency(n1,l1,j1,n2,l2,j2) ) / 2 / np.pi
        detuning_omega = 2 * np.pi * detuning_Hz
        return (1/(2*np.pi)) * gamma / ( detuning_omega**2 + gamma**2 / 4 )

    def get_cross_section(self,n1=4,l1=0,j1=1/2,F1=2,n2=4,l2=1,j2=3/2,F2=3,detuning_Hz=0):

        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            A21 = 2*np.pi*self.getTransitionRate(n2,l2,j2,n1,l1,j1,temperature=0.0) 
        else:
            A21 = 2*np.pi*self.getTransitionRate(n1,l1,j1,n2,l2,j2,temperature=0.0)

        g2 = 2*F2 + 1
        g1 = 2*F1 + 1

        # if ordered:
        #     g_ratio = g2/g1
        # else:
        #     g_ratio = g1/g2
        g_ratio = 1

        omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
        lineshape = self.lineshape(n1,l1,j1,n2,l2,j2,detuning_Hz=detuning_Hz)
        scattering_cross_section = g_ratio * np.pi**2 * c.c**2 / omega0**2 * A21 * lineshape
        return scattering_cross_section
    
    def get_zeeman_shift(self, n, l, j, m_j, m_i=None,
                         B=0, return_sweep=False):
        """Return the Zeeman energy (MHz) for state |n l j; m_j [m_i]> at B (Gauss).

        Routing:
        - ``n < 10``: kamo.hamiltonian exact diagonalization.  ``m_i`` required.
          Quantum numbers use the **adiabatic (Paschen-Back) convention**: ``m_j``
          and ``m_i`` are the high-field limiting values.
        - ``n >= 10``: pairinteraction.  Only ``m_j`` is used; ``m_i`` is ignored.

        ``B`` may be a scalar or array; returns matching shape.

        Parameters
        ----------
        return_sweep : bool, optional
            If True, return a ``(energy, sweep)`` tuple where ``sweep`` is the
            :class:`~kamo.hamiltonian.MagneticSweepResult` (n < 10 only; ``None``
            for the pairinteraction path).
        """
        B_arr = np.atleast_1d(np.asarray(B, dtype=float))
        scalar_in = np.ndim(B) == 0

        if n < _HAMILTONIAN_N_THRESHOLD:
            if m_i is None:
                raise ValueError(
                    f"m_i must be provided for n < {_HAMILTONIAN_N_THRESHOLD} "
                    "(kamo.hamiltonian includes nuclear Zeeman)."
                )
            result, sweep = self._zeeman_hamiltonian(n, l, j, m_j, m_i, B_arr)
        else:
            result = self._zeeman_pairinteraction(n, l, j, m_j, B_arr)
            sweep = None

        energy = float(result[0]) if scalar_in else result
        return (energy, sweep) if return_sweep else energy

    def _splitting_mhz(self, state1, state2, B=0):
        """Return |E2 − E1| (MHz) versus field, vectorized over ``B`` (Gauss).

        ``state1``/``state2`` are ``(n, l, j, m_j, m_i)`` tuples.  ``B`` may be a
        scalar or 1-D array (the return matches its shape).  When both states are
        low-n (below :data:`_HAMILTONIAN_N_THRESHOLD`) a single magnetic sweep
        covers both manifolds and the states are followed adiabatically; high-n
        states fall back to per-state :meth:`get_zeeman_shift`.

        This is the vectorized splitting engine used by
        :meth:`get_microwave_transition_frequency` and
        :meth:`get_magnetic_field_from_splitting`, which need array-valued ``B``
        and MHz units that the scalar, Hz-valued
        :meth:`get_transition_frequency` state-tuple API does not provide.
        """
        n1, l1, j1, m_j1, m_i1 = state1
        n2, l2, j2, m_j2, m_i2 = state2
        B_arr = np.atleast_1d(np.asarray(B, dtype=float))
        scalar_in = np.ndim(B) == 0

        if n1 < _HAMILTONIAN_N_THRESHOLD and n2 < _HAMILTONIAN_N_THRESHOLD:
            # One magnetic sweep up to the largest requested field covers both
            # manifolds; each requested field is read back by interpolation
            # (no per-B re-diagonalization).
            [e1_arr, e2_arr], _ = self._zeeman_hamiltonian_multi(
                [tuple(state1), tuple(state2)], B_arr
            )
            result = np.abs(e2_arr - e1_arr)
        else:
            e1 = self.get_zeeman_shift(n1, l1, j1, m_j1, m_i1, B)
            e2 = self.get_zeeman_shift(n2, l2, j2, m_j2, m_i2, B)
            result = np.atleast_1d(np.abs(e2 - e1))

        return float(result[0]) if scalar_in else result

    def get_microwave_transition_frequency(self, n, l, j, m_j1, m_i1, m_j2, m_i2, B=0):
        """|E2 − E1| (MHz) for the ``(m_j1,m_i1) → (m_j2,m_i2)`` transition in
        manifold ``(n, l, j)`` at field ``B`` (Gauss).

        ``B`` is normally a scalar.  Passing a 1-D array is **deprecated**: it
        still works — a single magnetic sweep is run up to ``max(B)`` and each
        requested field is interpolated from that sweep — but prefer a scalar
        ``B``, or drive a :meth:`~kamo.hamiltonian.model.AtomicStructure.magnetic_sweep`
        result directly (e.g. ``SweepResult.field_energy`` / ``get_energy``) for
        full control over the field grid.
        """
        if np.ndim(B) > 0:
            import warnings
            warnings.warn(
                "Supplying a vector B to get_microwave_transition_frequency is "
                "deprecated: one magnetic sweep is run up to max(B) and each "
                "requested field is interpolated from that sweep. Pass a "
                "scalar B, or use a magnetic_sweep result "
                "(SweepResult.field_energy / get_energy) for full control.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._splitting_mhz(
            (n, l, j, m_j1, m_i1), (n, l, j, m_j2, m_i2), B)

    def get_magnetic_field_from_splitting(
        self,
        state1,
        state2,
        transition_frequency_Hz,
        B_bounds_G=(0.0, 600.0),
        n_points=10000,
        B_guess=None,
    ):
        """Return the field(s) (Gauss) at which |E2 − E1| equals ``transition_frequency_Hz``.

        A **single** magnetic sweep of the two states' manifold(s) is run over
        ``B_bounds_G`` (``n_points`` samples) and inverted, so the cost is
        independent of how many target frequencies are requested.

        Monotonic vs non-monotonic
        --------------------------
        When the splitting increases (or decreases) monotonically over
        ``B_bounds_G`` the inversion is a single interpolation.  If the splitting
        curve turns over, a given target can be reached at **several** fields
        (branches).  In that case:

        * pass ``B_guess`` (Gauss) and the branch nearest to it is returned;
        * omit ``B_guess`` and a :class:`ValueError` lists the branch fields so
          you can pick one via ``B_guess``.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) 5-tuples
            The two states, with ``(a, b)`` either ``(F, mF)`` ints or
            ``(m_j, m_i)`` half-integer floats (standard ``kamo`` convention).
        transition_frequency_Hz : float or array-like
            Target splitting |E2 − E1| in Hz.  Array-like returns an array of
            fields (one per target).
        B_bounds_G : (float, float)
            Field-search bounds in Gauss (default ``(0, 600)``).  Only the range
            searched; widen it if the target lies outside.
        n_points : int
            Sweep / interpolation samples across ``B_bounds_G`` (default 10000).
        B_guess : float, optional
            Field (Gauss) used to disambiguate multiple branches: the crossing
            nearest ``B_guess`` is returned.  Required only when the splitting is
            non-monotonic and a target has more than one solution.

        Returns
        -------
        float or np.ndarray
            The field(s) in Gauss.

        Raises
        ------
        ValueError
            If a target is never reached within ``B_bounds_G`` (widen the
            bounds), or if it has multiple branches and no ``B_guess`` was given
            (the error lists the branch fields).
        """
        target_hz = np.atleast_1d(np.asarray(transition_frequency_Hz, dtype=float))
        b = np.linspace(B_bounds_G[0], B_bounds_G[1], n_points)
        freq_MHz = self._splitting_mhz(state1, state2, b)   # one sweep

        diffs = np.diff(freq_MHz)
        monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)

        if monotonic and B_guess is None:
            # single branch → one fast, vectorized interpolation over all targets
            b_grid, f_grid = b, freq_MHz
            if f_grid[0] > f_grid[-1]:                 # np.interp needs ascending x
                f_grid, b_grid = f_grid[::-1], b_grid[::-1]
            B_G = np.interp(target_hz / 1e6, f_grid, b_grid)
            if np.any((B_G == B_bounds_G[0]) | (B_G == B_bounds_G[1])):
                raise ValueError(
                    "One or more target frequencies fall on a bound of "
                    f"B_bounds_G {B_bounds_G} G.  Widen the bounds and re-run."
                )
        else:
            # non-monotonic (or an explicit B_guess): resolve each target from
            # its actual crossing(s) of the splitting curve.
            B_G = np.array([
                self._select_branch(b, freq_MHz, t / 1e6, B_guess, B_bounds_G)
                for t in target_hz
            ])

        return float(B_G[0]) if B_G.size == 1 else B_G

    @staticmethod
    def _crossings(x, y, target):
        """Return every ``x`` where the sampled curve ``y(x)`` equals ``target``.

        Roots are linear interpolations across each sign change of
        ``y - target``; grid points landing exactly on the target are included
        too.  Returns a sorted, de-duplicated ndarray (empty when the target is
        never reached).
        """
        x = np.asarray(x, dtype=float)
        diff = np.asarray(y, dtype=float) - float(target)
        k = np.where(np.diff(np.sign(diff)) != 0)[0]
        roots = []
        for i in k:
            x0, x1, y0, y1 = x[i], x[i + 1], diff[i], diff[i + 1]
            roots.append(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0))
        # include grid points that land exactly on the target
        roots.extend(x[j] for j in np.where(diff == 0)[0])
        return np.unique(np.round(roots, 9))

    @staticmethod
    def _select_branch(b, freq_MHz, target_MHz, B_guess, B_bounds_G):
        """Return the field where ``freq_MHz(b) == target_MHz``.

        Finds every crossing (linear root of ``freq_MHz − target_MHz``) in the
        swept range.  With one crossing it is returned directly; with several,
        the branch nearest ``B_guess`` is chosen, or a :class:`ValueError`
        listing the branch fields is raised when ``B_guess`` is ``None``.
        """
        roots = Potassium39._crossings(b, freq_MHz, target_MHz)

        if len(roots) == 0:
            raise ValueError(
                f"Splitting never equals {target_MHz:.6f} MHz within "
                f"B_bounds_G {B_bounds_G} G.  Widen the bounds and re-run."
            )
        if len(roots) == 1:
            return float(roots[0])
        if B_guess is None:
            branch_str = ", ".join(f"{r:.3f}" for r in roots)
            raise ValueError(
                f"Splitting equals {target_MHz:.6f} MHz at multiple fields "
                f"(non-monotonic curve): branches at [{branch_str}] G.  Pass "
                "B_guess (Gauss) to select the branch nearest a known field."
            )
        return float(roots[np.argmin(np.abs(roots - B_guess))])

    def get_ground_state_transition_frequency(self,f1,m_f1,f2,m_f2,B=0):
        '''
        Returns the ground-state transition frequency |E2 − E1| (MHz) between
        (f1,m_f1) and (f2,m_f2) under external magnetic field B (in Gauss).
        B may be a scalar or 1-D array.  Both states are read from a single
        magnetic sweep.
        '''
        return self._splitting_mhz(
            (4, 0, 0.5, int(f1), int(m_f1)),
            (4, 0, 0.5, int(f2), int(m_f2)), B)

    def get_transition_shift(
        self, n1, l1, j1, m_j1, m_i1, n2, l2, j2, m_j2, m_i2, B=0
    ):
        """Differential Zeeman shift of a transition at field B (MHz).

        Returns ``f(B) − f(0)`` for the transition ``state1 -> state2``, i.e.
        ``ΔE(state2, B) − ΔE(state1, B)`` with ``ΔE(state, B) = E(state, B) −
        E(state, 0)``.  Computed via :meth:`get_transition_frequency` in
        ``relative_mode="magnetic"`` (Hz), converted to MHz.
        """
        shift_hz = self.get_transition_frequency(
            (n1, l1, j1, m_j1, m_i1), (n2, l2, j2, m_j2, m_i2),
            B=B, relative_mode="magnetic",
        )
        return shift_hz / 1e6
    
    def get_transition_frequency(
        self,
        state1,
        state2,
        B=0.0,
        beam=None,
        frequency_Hz=None,
        intensity=None,
        polarization="pi",
        laser_model="auto",
        basis=None,
        n_points=200,
        dB=0.1,
        diamagnetic=True,
        relative_mode=None,
        return_sweep=False,
    ):
        """Transition frequency (Hz) between two states at field ``B``, with an
        optional laser light shift.

        Each state is a 5-tuple ``(n, l, j, a, b)`` whose last two numbers use
        the standard ``kamo`` convention:

        * both **int**   -> coupled-basis ``(F, mF)`` low-field labels;
        * both **float** (half-integer) -> uncoupled adiabatic ``(m_j, m_i)``
          (Paschen-Back) labels.

        The calculation uses ``kamo.hamiltonian`` exact diagonalization with
        eigenshuffle state tracking, so states are followed adiabatically (the
        same magnetic-sweep connection used everywhere else) through avoided
        crossings.

        How the two fields combine
        --------------------------
        * **Magnetic only** (no laser): a magnetic sweep 0 -> ``B`` gives the
          bare transition frequency at ``B``.
        * **Magnetic + laser**: the bare transition at ``B`` is computed from
          the magnetic sweep first; then a laser-intensity sweep *at the same
          field* ``B`` provides the light shift of the transition.  The light
          shift is taken as an intensity *difference* (``f(I) - f(0)``), so it
          is a true lab-frame shift regardless of the rotating frame used
          internally by the RWA model.

        Reference (``relative_mode``)
        -----------------------------
        Let ``f(B, I)`` be the absolute transition frequency ``E2 - E1``.

        * ``"absolute"`` -- return ``f(B, I)`` (the full transition frequency).
        * ``"magnetic"`` -- return ``f(B, I) - f(0, 0)``, i.e. relative to the
          zero-magnetic-field, zero-intensity transition frequency (the Zeeman
          shift of the transition when no laser is present).
        * ``"optical"`` -- return ``f(B, I) - f(B, 0)``, i.e. relative to zero
          intensity at the given (high) magnetic field (the pure light shift).
        * ``None`` (default) -- ``"optical"`` when a laser is supplied,
          otherwise ``"magnetic"``.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) tuples
            Lower/upper states of the transition (result is ``E2 - E1``, signed).
        B : float, optional
            Static magnetic field in Gauss (default 0).
        beam : kamo.GaussianBeam, optional
            Laser beam producing the light shift.  Supplies the laser frequency
            and, unless ``intensity`` is given, the peak intensity ``beam.I0``.
            Provide *either* ``beam`` *or* (``frequency_Hz`` + ``intensity``),
            not both.
        frequency_Hz : float, optional
            Laser frequency (Hz).  Requires ``intensity``.
        intensity : float, optional
            Laser intensity in W/m^2.  Required with ``frequency_Hz``; overrides
            ``beam.I0`` when supplied alongside ``beam``.
        polarization : str, optional
            Laser polarization: "pi", "sigma+", or "sigma-" (default "pi").
        laser_model : {"auto", "perturbative", "rwa", "stark"}, optional
            Light-shift model (default "auto").

            * "perturbative": second-order sum over the exact eigenstates of
              ``h0 + B * Zeeman`` with both rotating terms
              (:mod:`kamo.hamiltonian.perturbative`).  Hyperfine and Zeeman
              resolved, so it gives the qubit's differential shift, and it
              matches the fine-structure polarizabilities far from resonance.
              Linear in intensity.
            * "rwa": rotating-wave dipole coupling, non-perturbative, but
              without the counter-rotating terms (error ~ detuning / 2 f_0).
            * "stark": fine-structure polarizabilities in a diagonal operator;
              fast, no channel manifolds needed, but blind to the substructure
              of the coupled manifolds (error ~ their hyperfine + Zeeman
              spread over the detuning; zero for any within-manifold
              differential).  Never chosen by "auto".
            * "auto" calls :func:`kamo.hamiltonian.choose_laser_model`: "rwa"
              when Rabi/(2 detuning) exceeds 0.1 or when its square (the next
              order of the perturbative sum) exceeds the RWA's counter-rotating
              error estimate; "perturbative" otherwise.  The decision is
              attached to the returned sweep as ``sweep.model_choice``
              (``return_sweep=True``).
        basis : AtomicStructure, optional
            Override the atomic-structure basis.  When omitted, a basis is built
            automatically: just the two states' own manifolds (and their
            fine-structure partners) for a pure magnetic calculation or the
            Stark model, whose operator is diagonal; for the RWA and
            perturbative models, :func:`kamo.hamiltonian.light_shift_basis`
            adds every manifold that carries at least 1e-3 of either state's
            polarizability at the laser wavelength (so 3D and 5S appear for a
            4P state at 1064 nm).
        n_points : int, optional
            Intensity steps for the laser sweep (default 200).
        dB : float, optional
            Magnetic-sweep step in Gauss (default 0.1).
        diamagnetic : bool, optional
            Include the diamagnetic term in the magnetic sweep (default True).
        relative_mode : {None, "absolute", "magnetic", "optical"}, optional
            Reference for the returned frequency (see above).  Default None.
        return_sweep : bool, optional
            If True, also return the underlying sweep result (the laser sweep
            when a laser is supplied, otherwise the magnetic sweep).

        Returns
        -------
        float, or (float, SweepResult) when ``return_sweep`` is True.
        """
        from kamo import GaussianBeam
        from kamo.hamiltonian import (AtomicStructure, choose_laser_model,
                                      light_shift_basis)

        s1 = tuple(state1)
        s2 = tuple(state2)
        if len(s1) != 5 or len(s2) != 5:
            raise ValueError(
                "Each state must be a 5-tuple (n, l, j, a, b) where (a, b) are "
                "either (F, mF) ints or (m_j, m_i) half-integer floats.")

        # ---- validate the laser specification ----
        has_beam = beam is not None
        has_freq = frequency_Hz is not None
        if has_beam and has_freq:
            raise ValueError("Provide either `beam` or `frequency_Hz`, not both.")
        if has_freq and intensity is None:
            raise ValueError("`frequency_Hz` requires `intensity` (W/m^2).")
        use_light = has_beam or has_freq

        # ---- resolve the reference mode ----
        if relative_mode is None:
            relative_mode = "optical" if use_light else "magnetic"
        if relative_mode not in ("absolute", "magnetic", "optical"):
            raise ValueError(
                "relative_mode must be None, 'absolute', 'magnetic', or "
                f"'optical'; got {relative_mode!r}.")

        # ---- choose the laser model ----
        model_choice = None
        if use_light:
            f_laser = beam.frequency() if has_beam else float(frequency_Hz)
            I_choice = (float(intensity) if intensity is not None
                        else (float(beam.I0) if has_beam else None))
            if laser_model == "auto":
                model_choice = choose_laser_model(
                    (s1, s2), f_laser, I_choice, B_gauss=float(B), atom=self)
                laser_model = model_choice.model
            elif laser_model not in ("rwa", "stark", "perturbative"):
                raise ValueError(
                    "laser_model must be 'auto', 'rwa', 'perturbative' or 'stark'; "
                    f"got {laser_model!r}.")

        # ---- build / accept the basis ----
        # The magnetic sweep only needs the states' own manifolds and their
        # fine-structure partners (Zeeman and hyperfine do not couple
        # different (n, l); the diamagnetic l +/- 2 coupling is ~1e-12 at
        # 520 G), so it runs in that small basis unless the caller supplied
        # one.  The RWA and perturbative laser models also need the channel
        # manifolds, which can multiply the dimension by five; the laser sweep
        # identifies its states from the field eigenstates on its own, so it
        # gets the larger basis.
        model_B = model = AtomicStructure(_own_manifolds(s1, s2), atom=self)
        if basis is not None:
            model_B = model = basis
        elif use_light and laser_model in ("rwa", "perturbative"):
            # these models only see channels present in the basis: take every
            # manifold carrying >= 1e-3 of either state's polarizability.  The
            # photon-index loop check only matters for the RWA.
            sel = model_choice.basis if model_choice is not None else None
            if sel is None or (laser_model == "perturbative" and sel.dropped):
                sel = light_shift_basis((s1, s2), f_laser, atom=self, B_gauss=float(B),
                                        check_loops=(laser_model == "rwa"))
            model = AtomicStructure(list(sel.manifolds), atom=self)

        # ---- bare transition frequency at B (magnetic sweep, lab frame) ----
        B = float(B)
        B_max = max(B + dB, dB)
        resB = model_B.magnetic_sweep(B_max=B_max, dB=dB, diamagnetic=diamagnetic)
        f_B0 = resB.get_transition_frequency(s1, s2, at=B)      # f(B, 0)

        # ---- add the laser light shift at the same field ----
        df_light = 0.0
        resL = None
        if use_light:
            if has_beam:
                I_max = (float(intensity) if intensity is not None
                         else float(beam.I0))
            else:
                beam = GaussianBeam(waist=1e-6, frequency=float(frequency_Hz),
                                    power=0.0)
                I_max = float(intensity)
            resL = model.laser_sweep(
                beam, I_max=I_max, n_points=n_points,
                model=laser_model, polarization=polarization, B_gauss=B,
            )
            # intensity difference cancels the RWA rotating-frame offset,
            # leaving the true lab-frame light shift of the transition.
            df_light = resL.transition_frequency_shift(s1, s2, at=I_max)
            resL.model_choice = model_choice

        f_BI = f_B0 + df_light                                  # f(B, I)

        # ---- apply the requested reference ----
        if relative_mode == "absolute":
            result = f_BI
        elif relative_mode == "optical":
            result = f_BI - f_B0                                # pure light shift
        else:  # "magnetic"
            f_00 = resB.get_transition_frequency(s1, s2, at=0.0)  # f(0, 0)
            result = f_BI - f_00

        sweep = resL if use_light else resB
        return (result, sweep) if return_sweep else result

    def get_intensity_from_light_shift(
        self,
        state1,
        state2,
        light_shift_Hz,
        B=0.0,
        beam=None,
        frequency_Hz=None,
        wavelength_m=None,
        polarization="pi",
        laser_model="auto",
        basis=None,
        n_points=200,
        I_max=None,
        I_guess=None,
        max_expansions=8,
        relative_mode=True,
        dB=0.1,
        diamagnetic=True,
        return_sweep=False,
    ):
        """Return the intensity (W/m^2) at which the ``state1 -> state2``
        transition reaches ``light_shift_Hz`` at field ``B``.

        A **single** laser-intensity sweep is run at the fixed magnetic field
        ``B`` and inverted for the requested target(s), so the cost is
        independent of how many targets are requested.

        Reference (``relative_mode``)
        -----------------------------
        ``relative_mode`` selects what ``light_shift_Hz`` means, mirroring the
        ``True``/``False`` behaviour of ``SweepResult.plot``'s
        ``plot_differential``:

        * ``True`` (default) — differential.  Each state's energy at the
          *start* of the intensity sweep is subtracted first, so the target is
          the light **shift** of the transition,

              ``df(I) = f(B, I) - f(B, 0)``     with    ``f = E2 - E1``.

          Taking the shift as an intensity difference cancels the RWA
          rotating-frame offset, so ``df`` is a true lab-frame shift.  This
          inverts :meth:`get_transition_frequency` with
          ``relative_mode="optical"``.
        * ``False`` — absolute.  The target is the full transition frequency
          ``f(B, I)`` itself.  The bare transition frequency ``f(B, 0)`` is
          obtained from a magnetic sweep at the same field (as in
          :meth:`get_transition_frequency`) and added to the light shift, so
          the inverted curve is a genuine lab-frame absolute frequency rather
          than an RWA rotating-frame value.  This inverts
          :meth:`get_transition_frequency` with ``relative_mode="absolute"``.

        The strings ``"optical"`` and ``"absolute"`` are accepted as aliases
        for ``True`` and ``False``, matching
        :meth:`get_transition_frequency`'s vocabulary.

        Sign convention
        ---------------
        ``light_shift_Hz`` is **signed** and refers to ``E2 - E1``.  In
        differential mode a positive value means the laser pushes the two
        levels apart, a negative value that it pulls them together; swapping
        ``state1``/``state2`` flips the sign.

        Unreachable targets
        -------------------
        A target the sweep never reaches is returned as ``NaN`` (rather than
        raising) and a :class:`RuntimeWarning` reports how many were dropped,
        their values, and the span actually covered — usually a sign or
        detuning mistake, or a target beyond ``I_max``.  Because every target
        is inverted from the same sweep, one unreachable entry never discards
        the rest of the vector.

        Intensity range
        ---------------
        The sweep runs over ``I in [0, I_max]``.  When ``I_max`` is omitted it
        is taken from ``beam.I0`` (if the beam carries power), else estimated
        from the low-intensity slope of ``df`` (which is linear in ``I`` for a
        far-detuned laser).  The range is then doubled — up to
        ``max_expansions`` times — while at least one missing target still lies
        beyond the end of the shift curve, since that is the only case the
        doubling can fix.

        Monotonic vs non-monotonic
        --------------------------
        Far from resonance ``df(I)`` is monotonic and the inversion is a single
        interpolation.  Close to resonance the RWA dressed-state curve can turn
        over, so one target can be produced at **several** intensities
        (branches).  In that case:

        * pass ``I_guess`` (W/m^2) and the branch nearest to it is returned;
        * omit ``I_guess`` and a :class:`ValueError` lists the branch
          intensities so you can pick one via ``I_guess``.

        Parameters
        ----------
        state1, state2 : (n, l, j, a, b) 5-tuples
            Lower/upper states of the transition, with ``(a, b)`` either
            ``(F, mF)`` ints or ``(m_j, m_i)`` half-integer floats (standard
            ``kamo`` convention).  The target is referenced to ``E2 - E1``.
        light_shift_Hz : float or array-like
            Target(s) in Hz: a signed light shift when ``relative_mode`` is
            ``True``, or an absolute transition frequency when it is ``False``.
            Array-like returns an array of intensities (one per target), all
            read from the same sweep; unreachable targets come back as ``NaN``
            (see *Unreachable targets*).
        B : float, optional
            Static magnetic field in Gauss (default 0).  The sweep is run at
            this field, so the returned intensity accounts for the Zeeman
            structure of the two states.
        beam : kamo.GaussianBeam, optional
            Laser beam; supplies the laser frequency and, unless ``I_max`` is
            given, the sweep's upper intensity ``beam.I0``.
        frequency_Hz : float, optional
            Laser frequency (Hz), as an alternative to ``beam``.
        wavelength_m : float, optional
            Laser wavelength (m), as an alternative to ``beam``.  Exactly one
            of ``beam``, ``frequency_Hz``, ``wavelength_m`` must be given.
        polarization : str, optional
            Laser polarization: "pi", "sigma+", or "sigma-" (default "pi").
        laser_model : {"auto", "perturbative", "rwa", "stark"}, optional
            "auto" chooses as in :meth:`get_transition_frequency`, using
            ``I_max`` (or the beam's ``I0``) for the perturbativity check.
            Light-shift model (default "rwa").
        basis : AtomicStructure, optional
            Override the atomic-structure basis.  When omitted, the two states'
            manifolds plus their dipole-coupled (Delta l = +-1) neighbours are
            used, as required for a light shift.
        n_points : int, optional
            Intensity steps in the sweep (default 200).  More points give a
            finer inversion grid.
        I_max : float, optional
            Upper intensity of the sweep in W/m^2 (see *Intensity range*).
        I_guess : float, optional
            Intensity (W/m^2) used to disambiguate multiple branches: the
            crossing nearest ``I_guess`` is returned.
        max_expansions : int, optional
            Maximum number of ``I_max`` doublings (default 8).
        relative_mode : bool or {"optical", "absolute"}, optional
            Reference for ``light_shift_Hz`` (see above).  Default ``True``
            (differential / light shift).
        dB : float, optional
            Magnetic-sweep step in Gauss (default 0.1).  Only used when
            ``relative_mode`` is ``False``, where a magnetic sweep supplies the
            bare transition frequency ``f(B, 0)``.
        diamagnetic : bool, optional
            Include the diamagnetic term in that magnetic sweep (default True).
            Only used when ``relative_mode`` is ``False``.
        return_sweep : bool, optional
            If True, also return the underlying
            :class:`~kamo.hamiltonian.LaserSweepResult`.

        Returns
        -------
        float or np.ndarray
            The intensity/intensities in W/m^2, ``NaN`` where a target was not
            reachable, or ``(intensity, sweep)`` when ``return_sweep`` is True.

        Warns
        -----
        RuntimeWarning
            When one or more targets were unreachable and returned as ``NaN``.

        Raises
        ------
        ValueError
            If a target has multiple branches and no ``I_guess`` was given (the
            error lists the branch intensities).

        Examples
        --------
        >>> atom = Potassium39()
        >>> # 30 kHz measured shift of the |1,-1> -> |1,0> clock transition
        >>> # from a 780 nm beam at 100 G:
        >>> atom.get_intensity_from_light_shift(
        ...     (4, 0, 0.5, 1, -1), (4, 0, 0.5, 1, 0),
        ...     light_shift_Hz=30e3, B=100.0, wavelength_m=780e-9)
        >>> # a whole vector of shifts, inverted from one sweep:
        >>> atom.get_intensity_from_light_shift(
        ...     (4, 0, 0.5, 1, -1), (4, 0, 0.5, 1, 0),
        ...     light_shift_Hz=np.linspace(10e3, 50e3, 9),
        ...     B=100.0, wavelength_m=780e-9)
        >>> # the same, but targeting an absolute transition frequency:
        >>> atom.get_intensity_from_light_shift(
        ...     (4, 0, 0.5, 1, -1), (4, 0, 0.5, 1, 0),
        ...     light_shift_Hz=461.75e6, B=100.0, wavelength_m=780e-9,
        ...     relative_mode=False)

        See Also
        --------
        get_transition_frequency : forward direction (intensity -> frequency),
            whose ``relative_mode="optical"`` / ``"absolute"`` this inverts.
        kamo.hamiltonian.LaserSweepResult.intensity_from_splitting_shift :
            same inversion on an existing sweep, using ``|df|``.
        """
        from kamo import GaussianBeam
        from kamo.hamiltonian import (AtomicStructure, choose_laser_model,
                                      light_shift_basis)

        s1 = tuple(state1)
        s2 = tuple(state2)
        if len(s1) != 5 or len(s2) != 5:
            raise ValueError(
                "Each state must be a 5-tuple (n, l, j, a, b) where (a, b) are "
                "either (F, mF) ints or (m_j, m_i) half-integer floats.")

        # ---- validate / build the laser ----
        n_spec = sum(x is not None for x in (beam, frequency_Hz, wavelength_m))
        if n_spec != 1:
            raise ValueError(
                "Provide exactly one of `beam`, `frequency_Hz`, or "
                f"`wavelength_m`; got {n_spec}.")
        if beam is None:
            f_laser = (float(frequency_Hz) if frequency_Hz is not None
                       else c.c / float(wavelength_m))
            # power=0 -> I0=0; the sweep's intensity range is set below.
            beam = GaussianBeam(waist=1e-6, frequency=f_laser, power=0.0)

        # ---- resolve the reference mode ----
        if relative_mode is True or relative_mode == "optical":
            differential = True
        elif relative_mode is False or relative_mode == "absolute":
            differential = False
        else:
            raise ValueError(
                "relative_mode must be True/'optical' (target is a light "
                "shift) or False/'absolute' (target is a full transition "
                f"frequency); got {relative_mode!r}.")

        B = float(B)
        targets = np.atleast_1d(np.asarray(light_shift_Hz, dtype=float))
        scalar_in = np.ndim(light_shift_Hz) == 0

        # ---- choose the laser model ----
        model_choice = None
        I_choice = float(I_max) if I_max is not None else float(getattr(beam, "I0", 0.0))
        if laser_model == "auto":
            model_choice = choose_laser_model(
                (s1, s2), beam.frequency(), I_choice if I_choice > 0 else None,
                B_gauss=B, atom=self)
            laser_model = model_choice.model
        elif laser_model not in ("rwa", "stark", "perturbative"):
            raise ValueError(
                "laser_model must be 'auto', 'rwa', 'perturbative' or 'stark'; "
                f"got {laser_model!r}.")

        # ---- build / accept the basis (see get_transition_frequency) ----
        model_B = model = AtomicStructure(_own_manifolds(s1, s2), atom=self)
        if basis is not None:
            model_B = model = basis
        elif laser_model in ("rwa", "perturbative"):
            sel = model_choice.basis if model_choice is not None else None
            if sel is None or (laser_model == "perturbative" and sel.dropped):
                sel = light_shift_basis((s1, s2), beam.frequency(), atom=self, B_gauss=B,
                                        check_loops=(laser_model == "rwa"))
            model = AtomicStructure(list(sel.manifolds), atom=self)

        def _sweep(i_max, npts):
            return model.laser_sweep(
                beam, I_max=i_max, n_points=npts, model=laser_model,
                polarization=polarization, B_gauss=B,
            )

        # ---- reduce the targets to light shifts ----
        # The laser sweep only ever yields the *shift* df(I) = f(B, I) - f(B, 0)
        # as a true lab-frame quantity (the intensity difference cancels the RWA
        # rotating-frame offset).  In absolute mode the bare transition
        # frequency f(B, 0) comes from a magnetic sweep at the same field, and
        # subtracting it turns the absolute targets into shift targets, so the
        # inversion below is identical in both modes.
        f_B0 = 0.0
        if not differential:
            resB = model_B.magnetic_sweep(
                B_max=max(B + dB, dB), dB=dB, diamagnetic=diamagnetic)
            f_B0 = float(resB.get_transition_frequency(s1, s2, at=B))
        targets_shift = targets - f_B0
        quantity = "Light shift" if differential else "Transition frequency"

        # ---- choose the initial intensity range ----
        I_max = float(I_max) if I_max is not None else float(getattr(beam, "I0", 0.0))
        if I_max <= 0.0:
            # No intensity scale supplied: estimate one from the low-intensity
            # slope of df (linear in I for a far-detuned laser).
            I_probe = 1.0e2                     # 10 mW/cm^2 — safely perturbative
            slope = (_sweep(I_probe, 2)
                     .transition_frequency_shift(s1, s2)[-1] / I_probe)
            scale = float(np.max(np.abs(targets_shift)))
            I_max = 2.0 * scale / abs(slope) if (slope and scale) else I_probe

        # ---- sweep and invert, expanding the range until every target is hit ----
        # All targets share one sweep; the range is doubled only while at least
        # one still-missing target lies beyond the end of the (same-signed)
        # shift curve, since that is the only case doubling can fix.  Anything
        # still missing at the end is unreachable and comes back as NaN.
        resL = None
        roots = []
        n_doublings = max(0, int(max_expansions))
        for attempt in range(n_doublings + 1):
            resL = _sweep(I_max, n_points)
            shift = resL.transition_frequency_shift(s1, s2)      # signed, Hz
            roots = [self._crossings(resL.param, shift, t) for t in targets_shift]
            missing = [k for k, r in enumerate(roots) if len(r) == 0]
            if not missing:
                break
            df_end = float(shift[-1])
            expandable = [k for k in missing
                          if targets_shift[k] * df_end > 0
                          and abs(targets_shift[k]) > abs(df_end)]
            if not expandable or attempt == n_doublings:
                break
            I_max *= 2.0

        # ---- pick a branch per target (NaN where none was found) ----
        out = np.full(len(targets_shift), np.nan)
        unreachable = []
        for k, (t, r) in enumerate(zip(targets, roots)):
            if len(r) == 0:
                unreachable.append(t)
            elif len(r) == 1:
                out[k] = float(r[0])
            elif I_guess is not None:
                out[k] = float(r[np.argmin(np.abs(r - float(I_guess)))])
            else:
                shown = ", ".join(f"{v:.4e}" for v in r[:8])
                if len(r) > 8:
                    shown += f", ... ({len(r)} total)"
                hint = ""
                if len(r) > 4:
                    hint = (
                        f"  So many branches usually means {t:.4e} Hz is below "
                        "the numerical resolution of the sweep (the two states "
                        "shift almost identically), in which case no intensity "
                        "is well determined."
                    )
                raise ValueError(
                    f"{quantity} of {t:.4e} Hz occurs at multiple "
                    f"intensities (non-monotonic curve): branches at "
                    f"[{shown}] W/m^2.  Pass I_guess (W/m^2) to select the "
                    f"branch nearest a known intensity.{hint}"
                )

        if unreachable:
            import warnings
            miss_str = ", ".join(f"{t:.4e}" for t in unreachable[:8])
            if len(unreachable) > 8:
                miss_str += f", ... ({len(unreachable)} total)"
            warnings.warn(
                f"{len(unreachable)} of {len(targets)} target(s) returned as "
                f"NaN: {quantity.lower()}(s) [{miss_str}] Hz never occur for I "
                f"in [0, {I_max:.4e}] W/m^2, where the reachable range is "
                f"[{np.min(shift) + f_B0:.4e}, {np.max(shift) + f_B0:.4e}] Hz "
                f"after {attempt} doubling(s).  Check the sign of "
                "light_shift_Hz (it is "
                + ("f(B, I) - f(B, 0)" if differential else "f(B, I)")
                + " with f = E2 - E1) and the laser detuning; if the target "
                "is simply beyond the swept range, pass a larger I_max or "
                "raise max_expansions.",
                RuntimeWarning,
                stacklevel=2,
            )

        I_out = out
        result = float(I_out[0]) if scalar_in else I_out
        return (result, resL) if return_sweep else result

    def state_lookup(self, n, l, j, m1, m2):
        """Both label sets of one state (deprecated).

        Use :func:`kamo.hamiltonian.state_label` for labels, or
        ``Manifold(n, l, j).state_for(F, mF)`` / ``.label_for(m_j, m_i)`` to
        convert quantum numbers.

        Args:
            n, l, j: fine-structure quantum numbers.
            m1, m2: ``(F, mF)`` integers or ``(m_J, m_I)`` half-integers.

        Returns:
            dict: ``"hf"`` ``(m_J, m_I)``, ``"lf"`` ``(F, mF)`` (the two ends
            of the Paschen-Back adiabatic connection), and ``"hf_str"`` /
            ``"lf_str"``, the matching TeX kets.
        """
        import warnings
        from kamo.hamiltonian.state_labels import _manifold, state_label
        warnings.warn(
            "Potassium39.state_lookup is deprecated; use "
            "kamo.hamiltonian.state_label or Manifold.state_for/label_for.",
            DeprecationWarning, stacklevel=2)
        man = _manifold(n, l, j)
        try:
            if int(m1) == m1 and int(m2) == m2:
                F, mF = int(m1), int(m2)
                m_j, m_i = man.state_for(F, mF)
            else:
                m_j, m_i = float(m1), float(m2)
                F, mF = man.label_for(m_j, m_i)
        except KeyError as err:
            raise ValueError(err.args[0]) from None
        return {
            "hf": (float(m_j), float(m_i)),
            "lf": (int(F), int(mF)),
            "hf_str": state_label(n, l, j, float(m_j), float(m_i), term=False),
            "lf_str": state_label(n, l, j, int(F), int(mF), term=False),
        }

    def get_scattering_length(self, f, mf, b,
                            f2=None, mf2=None,
                            interp=True,
                            method='table',
                            return_complex=False):
        """s-wave scattering length (a0) of the pair |f,mf> + |f2,mf2> at field b (G).

        Thin wrapper around :func:`kamo.scattering.lookup.scattering_length`.

        Args:
            f (int), mf (int): hyperfine state of the first atom.
            b (float or array): magnetic field in Gauss.
            f2 (int), mf2 (int), optional: state of the second atom.  Omit both
                for two atoms in |f,mf>.
            interp (bool, optional): method='kokkelmans' only; interpolate the
                0.5 G table instead of taking the nearest point.
            method (str, optional):
                'table': (default) the calibrated coupled-channels model, precomputed for all
                    36 pairs of ground states on 0-1000 G and shipped with kamo.
                    Instant, and the only method covering every pair.  Accurate to
                    ~1e-3 relative above 0.01 G and exact at b = 0; in
                    0 < b < 0.01 G unresolved channel-opening jumps make it
                    indicative only -- use 'cc' there.
                'kokkelmans': S. Kokkelmans' tables on the
                    Tweezers G: drive, same-state pairs only, 1-1000 G.  Needs the
                    share mounted, and differs from the calibrated model by ~2% at
                    some fields.  NOTE: this default disagrees with 'table' below;
                    pass method explicitly if it matters.
                'cc': the same model computed directly (~1.5 s setup, then
                    ~40-90 ms per field, memoised).
                'empirical': measured-resonance model.  Instant, but only for the
                    F=1 channels with measured resonances.
            return_complex (bool, optional): return a_re - i a_im (lossy
                channels, e.g. F=2) instead of the real part.

        Returns:
            float for scalar b, else ndarray of b's shape.

        Raises:
            ValueError: invalid states, only one of f2/mf2 given, no data for
                the pair with this method, or b outside the method's range.
        """
        from kamo.scattering.lookup import scattering_length
        if (f2 is None) != (mf2 is None):
            raise ValueError("give both f2 and mf2, or neither (same-state pair)")
        second = None if f2 is None else (f2, mf2)
        return scattering_length((f, mf), second, b, method=method, interp=interp,
                                 return_complex=return_complex)
    
    def state_label(self,
                    n,l,j,
                    m1=None,m2=None,
                    skip_njl = False,
                    force_hf_lf = None,
                    force_skip_spin = False,
                    tex_formatting=True):
        r"""Spectroscopic label of a state, e.g. ``4S_{1/2}|F=1, m_F=-1\rangle``.

        Thin wrapper around :func:`kamo.hamiltonian.state_label`, returned
        *without* ``$`` delimiters so it can sit inside a larger math string.

        Args:
            n, l, j: fine-structure quantum numbers.
            m1, m2 (optional): ``(F, mF)`` integers or ``(m_J, m_I)``
                half-integers.  Give only ``m1`` for a one-number ket
                (``|F=2>`` or ``|m_J=+1/2>``).
            skip_njl (bool): leave out the term symbol ``nL_J``.
            force_hf_lf ({None, 'hf', 'lf'}): print the uncoupled ``(m_J, m_I)``
                ('hf') or coupled ``(F, mF)`` ('lf') ket, converting through the
                Paschen-Back adiabatic connection.  None keeps the input's basis.
            force_skip_spin (bool): leave out the ket.
            tex_formatting (bool): TeX (default) or plain text.

        Returns:
            str: e.g. ``4S_{1/2}|m_J=-1/2, m_I=+3/2\rangle``.
        """
        from kamo.hamiltonian import state_label
        bases = {None: None, 'hf': 'uncoupled', 'lf': 'coupled'}
        if force_hf_lf not in bases:
            raise ValueError(
                f"force_hf_lf must be None, 'hf', or 'lf'; got {force_hf_lf!r}.")
        spins = () if force_skip_spin else tuple(m for m in (m1, m2) if m is not None)
        return state_label(n, l, j, *spins,
                           basis=bases[force_hf_lf] if len(spins) == 2 else None,
                           math=False, term=not skip_njl, tex=tex_formatting)
