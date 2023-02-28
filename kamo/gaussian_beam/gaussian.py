import numpy as np

class GaussianBeam():
    ''''
    A gaussian beam object.

    Attributes
    ----------
    waist: float
        Waist (in m)
    wavelength: float
        wavelength (in m)
    n_medium: float
        The optical index of the medium
    divergence_angle: float

    Methods
    -------
    beam_radius
    peak_intensity
    '''
    def __init__(self,waist,wavelength,n_medium=1.):
        self.waist = waist
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.rayleigh_range = np.pi * self.waist**2 * self.n_medium / self.wavelength
        self.divergence_angle = self.wavelength / np.pi / self.n_medium / self.waist
        
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
    
    def peak_intensity(self,power,r=0.,z=0.):
        '''
        Returns the intensity of the gaussian beam at (r,z)

        Parameters
        ----------
        power: float
            The power (in Watts) in the beam
        r: float
            The radial position (in m) from the beam axis (default = 0.)
        z: float (default z=0.)
            The axial position (in m) from the beam waist (default = 0.)
        '''
        w0 = self.waist
        wz = self.beam_radius(z)
        return 2 * power / np.pi / wz**2 * np.exp(-2 * r**2 / wz )