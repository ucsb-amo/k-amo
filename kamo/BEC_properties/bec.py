import arc
import numpy as np
import kamo.constants as c

class BEC():
    def __init__(self):
        super().__init__()
    
    def oscillator_len(trap_freq):
        """
        Calculate oscillator length for a given trap frequency.

        Parameters:
        trap_freq (float): Trap frequency in 2pi * Hz.

        Returns:
        float: oscillator length in m.
        """
        return np.sqrt(c.hbar / (c.m_K * trap_freq))
    
    def get_axial_trap_frequency(radial_trap_frequency,lmbda=1064.e-9,waist=3.8e-6):
        """
        Calculate the axial trap frequency for an optical dipole trap.

        Parameters:
        rad_freq (float): Radial trap frequency in 2pi*Hz.
        waist (float): Beam waist in meters.
        lmbda (float): Wavelength of the trapping light in meters.

        Returns:
        float: Axial trap frequency in 2pi*Hz.
        """
        return radial_trap_frequency * (lmbda / (np.sqrt(2)*np.pi * waist))
    
    def chemical_potential(n_atoms,a_scattering,radial_trap_frequency):
        """
        Calculate the chemical potential for an interacting BEC.

        Parameters:
        n_atoms (float): number of atoms.
        a_scattering (float): Scattering length in Bohr radii.
        radial_trap_frequency (float): radial trap frequency in 2pi*Hz.

        Returns:
        float: chemical potential.
        """
        a_scat = a_scat * 5.29177210544e-11
        ax_trap = self.get_axial_trap_frequency(radial_trap_frequency=radial_trap_frequency)
        return ((15**(2/5))/2) * ((n_atoms*a_scattering)/(np.sqrt(c.hbar / (c.m_K*(radial_trap_frequency*radial_trap_frequency*ax_trap)**(1/3)))))**(2/5) * c.hbar * (radial_trap_frequency*radial_trap_frequency*ax_trap)**(1/3)
    
    def thomas_fermi_radius(n_atoms,a_scattering,radial_trap_frequency,trap_frequency):
        """
        Calculate the Thomas-Fermi radius for one dimension of an interacting BEC with given trap frequency.

        Parameters:
        n_atoms (float): number of atoms.
        a_scattering (float): Scattering length in Bohr radii.
        radial_trap_frequency (float): radial trap frequency in 2pi*Hz.
        trap_frequency (float): trap frequency of the desired dimension in 2pi*Hz.

        Returns:
        float: chemical potential.
        """
        chem_pot = self.chemical_potential(n_atoms=n_atoms,a_scattering=a_scattering,radial_trap_frequency=radial_trap_frequency)
        return np.sqrt((2*chem_pot)/(c.m_K*(trap_frequency**2)))