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
            cp = light_shift.compute_polarizabilities.ComputePolarizabilities()
            self.polarizability_ground_state = \
                float(cp.compute_complete_polarizability(4,0,1/2,1,-1,self.wavelength)[0]) \
                    * c.convert_polarizability_au_to_SI
            
    def frequency(self):
        return c.c / self.wavelength
        
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
    
    def intensity(self,power=-0.1,r=0.,z=0.,
                  convert_to_mW_per_cm2=False):
        '''
        Returns the intensity of the gaussian beam at (r,z).

        Parameters
        ----------
        power: float
            The power (in Watts) in the beam (default = -0.1, uses power =
            self.power)
        r: float
            The radial position (in m) from the beam axis (default = 0.)
        z: float
            The axial position (in m) from the beam waist (default = 0.)
        convert_to_mW_per_cm2: bool
            If true, converts the output to mW/cm^2 before returning.
        '''
        if power == -0.1:
            power = self.power
        wz = self.beam_radius(z)

        convert_W_per_m2_to_mW_per_cm2 = 0.1
        if convert_to_mW_per_cm2:
            convert = convert_W_per_m2_to_mW_per_cm2
        else:
            convert = 1

        return 2 * power / np.pi / wz**2 * np.exp(-2 * (r / wz)**2 ) * convert
    
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
        '''
        Returns the trap frequency (rad/s) for a potassium atom's ground state
        in the gaussian beam.

        trap_length refers to either the trap waist or the Rayleigh range of the
        beam, depending on if the user wants the radial or the axial trap
        frequency.
        '''
        intensity = self.intensity(power)
        omega = np.sqrt( 2 * intensity * self.polarizability_ground_state ) \
            / np.sqrt( c.c * c.m_K * c.epsilon_0 ) / trap_length
        return omega

    def trap_frequency_radial(self,power=-0.1,polarizability=0.):
        '''
        Returns the radial trap frequency (rad/s) for a potassium atom's ground
        state in the given gaussian beam.
        '''
        power, polarizability = self._handle_trap_args(power,polarizability)
        return self.trap_frequency(power,self.waist,polarizability)
    
    def trap_frequency_axial(self,power=-0.1,polarizability=0.):
        '''
        Returns the axial trap frequency (rad/s) for a potassium atom's ground
        state in the given gaussian beam.
        '''
        power, polarizability = self._handle_trap_args(power,polarizability)
        return self.trap_frequency(power,self.zR,polarizability)
    
    def trap_depth(self,power=-0.1,r=0.,z=0.,polarizability=0.):
        '''
        Returns the trap depth in K.
        '''
        power, polarizability = self._handle_trap_args(power,polarizability)
        return - 1/(2*c.c*c.epsilon_0) * polarizability * self.intensity(power,r,z) / c.kB
    
    def power_for_given_trap_depth(self,trap_depth_K=0.,r=0.,z=0.,polarizability=0.):
        _, polarizability = self._handle_trap_args(0.,polarizability)
        return trap_depth_K / np.abs(self.trap_depth(1.,r,z,polarizability))

    def _handle_trap_args(self,power,polarizability):
        if (not self.include_trap_properties) and polarizability == 0.:
            raise ValueError("Trap properties were not included in the initialization of the class, so polarizability data is not available.")
        if polarizability == 0.:
            polarizability = self.polarizability_ground_state
        if power == -0.1:
            power = self.power
        return power, polarizability
        