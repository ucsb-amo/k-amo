import numpy as np
from kamo import constants as c

class GaussianBeam():
    ''''
    A gaussian beam object.

    The beam strength may be specified either by the power or by the peak
    intensity, but not both (providing both raises a ValueError). If neither is
    given, the power defaults to 0 W.

    Parameters
    ----------
    waist (m)
    wavelength (m)
    power (W)
    n_medium
    include_trap_properties (defaults to False)
    peak_intensity (W/m^2)

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
    def __init__(self,waist,wavelength=None,power=None,n_medium=1.,include_trap_properties=False,frequency=None,
                 peak_intensity=None):
        if wavelength is None and frequency is None:
            raise ValueError("Must provide either wavelength or frequency.")
        if wavelength is not None and frequency is not None:
            raise ValueError("Provide either wavelength or frequency, not both.")
        if power is not None and peak_intensity is not None:
            raise ValueError("Provide either power or peak_intensity, not both.")
        if frequency is not None:
            wavelength = c.c / frequency
        self.waist = waist
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.rayleigh_range = np.pi * self.waist**2 * self.n_medium / self.wavelength
        self.divergence_angle = self.wavelength / np.pi / self.n_medium / self.waist
        if peak_intensity is not None:
            self.power = self.power_from_peak_intensity(peak_intensity)
        else:
            self.power = 0. if power is None else power
        self.peak_intensity = self.intensity()

        # aliases for commonly used parameters
        self.I0 = self.peak_intensity
        self.w0 = self.waist
        self.zR = self.rayleigh_range

        self.include_trap_properties = include_trap_properties
        if include_trap_properties:
            from kamo import light_shift
            cp = light_shift.compute_polarizabilities.ComputePolarizabilities(force_arc=True)
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
        # w0 = self.waist
        # wz = self.beam_radius(z)
        convert_W_per_m2_to_mW_per_cm2 = 0.1
        intensity_W_per_m2 = intensity_mW_per_cm2 / convert_W_per_m2_to_mW_per_cm2
        return intensity_W_per_m2 / self.intensity(1,r,z)

    def power_from_peak_intensity(self,peak_intensity):
        '''
        Returns the power of the gaussian beam whose peak intensity (at the
        waist, on axis) is peak_intensity.

        Parameters
        ----------
        peak_intensity: float
            The peak intensity in W/m^2

        Returns
        -------
        float
        '''
        return peak_intensity * np.pi * self.waist**2 / 2

    def trap_frequency(self,power,trap_length,polarizability,curvature_factor=4.):
        '''
        Returns the trap frequency (rad/s) for a potassium atom's ground state
        in the gaussian beam.

        The trap frequency is omega = sqrt(curvature_factor * U0 / (m L^2)),
        where U0 = polarizability * I0 / (2 c epsilon_0) is the trap depth and
        L is trap_length.  The curvature factor is *not* the same along the two
        axes, because the beam profile is not:

          radial:  U(r) = -U0 exp(-2 r^2 / w0^2)  ~  -U0 (1 - 2 r^2 / w0^2)
                   => omega_r = sqrt(4 U0 / (m w0^2)),   L = w0,  factor 4
          axial:   U(z) = -U0 / (1 + (z/zR)^2)    ~  -U0 (1 -   z^2 / zR^2)
                   => omega_z = sqrt(2 U0 / (m zR^2)),   L = zR,  factor 2

        Parameters
        ----------
        power: float
            The power (in Watts) in the beam.
        trap_length: float
            Either the waist (radial) or the Rayleigh range (axial), in m.
        polarizability: float
            The ground-state polarizability in SI units (C m^2 / V).
        curvature_factor: float
            4 for the radial direction (default), 2 for the axial direction.
            See above.
        '''
        intensity = self.intensity(power)
        trap_depth = polarizability * intensity / (2 * c.c * c.epsilon_0)
        omega = np.sqrt( curvature_factor * trap_depth / c.m_K ) / trap_length
        return omega

    def trap_frequency_radial(self,power=-0.1,polarizability=0.):
        '''
        Returns the radial trap frequency (rad/s) for a potassium atom's ground
        state in the given gaussian beam.
        '''
        power, polarizability = self._handle_trap_args(power,polarizability)
        return self.trap_frequency(power,self.waist,polarizability,
                                   curvature_factor=4.)

    def trap_frequency_axial(self,power=-0.1,polarizability=0.):
        '''
        Returns the axial trap frequency (rad/s) for a potassium atom's ground
        state in the given gaussian beam.

        This is smaller than trap_frequency_radial by w0 / (sqrt(2) zR)
        = lambda / (sqrt(2) pi w0), matching the convention used in
        kamo.BEC_properties.
        '''
        power, polarizability = self._handle_trap_args(power,polarizability)
        return self.trap_frequency(power,self.zR,polarizability,
                                   curvature_factor=2.)
    
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
        