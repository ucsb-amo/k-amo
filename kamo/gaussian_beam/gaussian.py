import numpy as np

class GaussianBeam():
    def __init__(self,waist=0.,wavelength=0.,n_medium=1.):
        self.waist = waist
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.power = power
        self.rayleigh_range = np.pi * self.waist^2 * self.n_medium / self.wavelength
        self.divergence_angle = self.wavelength / np.pi / self.n_medium / self.waist
        
    def beam_radius(self,z):
        return self.waist * np.sqrt( 1 + (z / self.rayleigh_range)**2 )
    
    def peak_intensity(self,power,r=0.,z=0.):
        '''
        Returns the intensity of the gaussian beam at (r,z)

        Parameters
        ----------

        '''
        w0 = self.waist
        wz = self.beam_radius(z)
        return 2 * power / np.pi / wz**2 * np.exp(-2 * r**2 / wz )