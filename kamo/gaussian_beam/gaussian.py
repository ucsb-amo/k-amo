import numpy as np

class GaussianBeam():
    def __init__(self,power=0.,waist=0.,wavelength=0.,n_medium=1.):
        self.power = power
        self.waist = waist
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.rayleigh_range = np.pi * self.waist^2 * self.n_medium / self.wavelength
        self.divergence_angle = self.wavelength / np.pi / self.n_medium / self.waist
        
    def beam_radius(self,z=0.):
        return self.waist * np.sqrt( 1 + (z / self.rayleigh_range)**2 )
    
    def peak_intensity(self,z=0.):
        return 2 * self.power / np.pi / self.beam_radius(z)