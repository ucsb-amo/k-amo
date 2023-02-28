import numpy as np
from gaussian import GaussianBeam

class ThinLensGaussian():
    '''
    For computing the effect of a thin lens on a gaussian beam. Results are
    taken from "Fundamentals of Photonics" by Saleh and Teich

    Parameters
    ----------
    focus: float
        The lens focal length (m)

    Methods
    -------
    output_beam
    '''
    def __init__(self,focus=0.):
        self.focus = focus

    def output_beam(self, input_beam, z_input_waist_to_lens):
        '''
        Return the waist 

        Parameters
        ----------
        input_beam: GaussianBeam
            The input beam
        z_input_waist_to_lens: float
            The distance (m) from the beam waist to the thin lens

        Returns
        -------
        GaussianBeam
            The output beam with new waist.
        '''
        Mr = np.abs( self.focus / ( z_input_waist_to_lens - self.focus ) )
        r = input_beam.rayleigh_range / ( z_input_waist_to_lens - self.focus )
        M = Mr / np.sqrt( 1 + r**2 )
        out_beam = GaussianBeam(waist = M*input_beam.waist, wavelength=input_beam.wavelength)
        # z_output_waist_to_lens = 
        return out_beam
    
class Objective():
    '''
    A microscope objective.

    Parameters
    ----------
    NA: float
        The numerical aperture of the objective
    working_distance: float
        The working distance (in m) of the objective
    focal_length: float
        The focal length (in m) of the objective
    pupil_diameter: float
        The pupil diameter (in m) of the objective
    '''
    def __init__(self, NA, working_distance, focal_length, pupil_diameter):
        self.NA = NA
        self.working_distance = working_distance
        self.focal_length = focal_length
        self.pupil_diameter = pupil_diameter
    def output_beam(self,input_beam):
        '''
        Returns the output gaussian beam for a given input beam, assuming
        operation away from the limits of the objective.

        Parameters
        ----------
        input_beam: GaussianBeam
            The input beam

        Outputs
        -------
        GaussianBeam
        '''
        
