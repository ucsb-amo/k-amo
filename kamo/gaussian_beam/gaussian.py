import numpy as np
from kamo import constants as c

class GaussianBeam():
    ''''
    A gaussian beam object.

    Parameters
    ----------
    waist (m)
    wavelength (m)
    power (W)
    n_medium
    include_trap_properties (defaults to False)

    Attributes
    ----------
    waist: float
        Waist (in m)
    wavelength: float
        wavelength (in m)
    power: float
        power (in W)
    n_medium: float
        The optical index of the medium

    I0: float
        The peak intensity (in W/m^2)
    w0: float
        An alias for the beam waist
    zR: float
        An alias for the rayleigh range.

    Methods
    -------
    beam_radius
    intensity
    '''
    def __init__(self,waist,wavelength,power=0.,n_medium=1.,include_trap_properties=False):
        self.waist = waist
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.rayleigh_range = np.pi * self.waist**2 * self.n_medium / self.wavelength
        self.divergence_angle = self.wavelength / np.pi / self.n_medium / self.waist
        self.power = power
        self.peak_intensity = self.intensity()

        # aliases for commonly used parameters
        self.I0 = self.peak_intensity
        self.w0 = self.waist
        self.zR = self.rayleigh_range

        self.include_trap_properties = include_trap_properties
        if include_trap_properties:
            from kamo import light_shift
            self.polarizability_ground_state = \
                light_shift.compute_complete_polarizability(4,0,1/2,2,2,self.wavelength) \
                    * c.convert_polarizability_au_to_SI
        
    def beam_radius(self,z):
        '''
        Returns the beam radius at a distance z from the waist
        
        Parameters:
        -----------
        z: float
            distance from the waist

        Returns:
        --------
        float
        '''
        return self.waist * np.sqrt( 1 + (z / self.rayleigh_range)**2 )
    
    def intensity(self,power=-0.1,r=0.,z=0.):
        '''
        Returns the intensity of the gaussian beam at (r,z)

        Parameters
        ----------
        power: float
            The power (in Watts) in the beam (default = -0.1, uses power =
            self.power)
        r: float
            The radial position (in m) from the beam axis (default = 0.)
        z: float
            The axial position (in m) from the beam waist (default = 0.)
        '''
        if power == -0.1:
            power = self.power
        wz = self.beam_radius(z)
        return 2 * power / np.pi / wz**2 * np.exp(-2 * r**2 / wz )
    
    def power_from_intensity(self,intensity_mW_per_cm2,r=0.,z=0.):
        '''
        Returns the power of the gaussian beam which gives I(r,z).

        Parameters
        ----------
        intensity_mW_per_cm2: float
            The intensity given in units of mW per cm^2

        Returns
        -------
        float
        '''
        w0 = self.waist
        wz = self.beam_radius(z)
        convert_W_per_m2_to_mW_per_cm2 = 0.1
        intensity_W_per_m2 = intensity_mW_per_cm2 / convert_W_per_m2_to_mW_per_cm2
        return intensity_W_per_m2 / self.intensity(1,r,z)
    
    def trap_frequency(self,power,trap_length,polarizability):
        return np.sqrt( \
            2 * self.peak_intensity * self.polarizability_ground_state ) \
            / np.sqrt( c.c * c.m_K * c.epsilon_0 ) / trap_length

    def trap_frequency_radial(self,power=-0.1,polarizability=0.):
        '''
        Returns the radial trap frequency for a potassium atom's ground state.
        '''
        if (not self.include_trap_properties) and polarizability == 0.:
            raise ValueError("Trap properties were not included in the initialization of the class, so polarizability data is not available.")
        if polarizability == 0.:
            polarizability = self.polarizability_ground_state
        if power == -0.1:
            power = self.power
        return self.trap_frequency(power,self.waist,polarizability)
    
    def trap_frequency_axial(self,power=-0.1,polarizability=0.):
        '''
        Returns the radial trap frequency for a potassium atom's ground state.
        '''
        if (not self.include_trap_properties) and polarizability == 0.:
            raise ValueError("Trap properties were not included in the initialization of the class, so polarizability data is not available.")
        if polarizability == 0.:
            polarizability = self.polarizability_ground_state
        if power == -0.1:
            power = self.power
        return self.trap_frequency(power,self.zR,polarizability)
    
    def potential_depth(self,power=-0.1,r=0.,z=0.,polarizability=0.):
        if (not self.include_trap_properties) and polarizability == 0.:
            raise ValueError("Trap properties were not included in the initialization of the class, so polarizability data is not available.")
        if polarizability == 0.:
            polarizability = self.polarizability_ground_state
        if power == -0.1:
            power = self.power
        return - 1/(2*c.c*c.epsilon_0) * polarizability * self.intensity(power,r,z)
        