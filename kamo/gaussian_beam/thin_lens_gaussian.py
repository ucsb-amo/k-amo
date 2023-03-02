import numpy as np
from kamo.gaussian_beam.gaussian import GaussianBeam

class ThinLensGaussian():
    '''
    For computing the effect of a thin lens on a gaussian beam. Results are
    taken from "Fundamentals of Photonics" by Saleh and Teich

    Attributes
    ----------
    focus: float
        The lens focal length (m)

    Methods
    -------
    beam_after_lens
    '''
    def __init__(self,focus=0.):
        self.focus = focus

    def beam_after_lens(self, input_beam, z_input_waist_to_lens=0.):
        '''
        Return the waist 

        Parameters
        ----------
        input_beam: GaussianBeam
            The input beam
        z_input_waist_to_lens: float
            The distance (m) from the beam waist to the thin lens. Default = 0.

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

    Attributes
    ----------
    NA: float
        The numerical aperture of the objective
    working_distance: float
        The working distance (in m) of the objective (WD)
    focal_length: float
        The effective focal length (in m) of the objective (EFL)
    entrance_aperture_diameter
        The entrance pupil diameter at the back aperture of the objective (EP =
        2*NA*EFL)
    '''

    def __init__(self, NA, working_distance, focal_length):
        self.NA = NA
        self.working_distance = working_distance
        self.focal_length = focal_length
        self.entrance_aperture_diameter = 2 * self.NA * self.focal_length

    # def input_beam_from_output_waist(self,output_waist,wavelength): '''
    #     Returns the required input gaussian beam for a given output beam
    #     waist, assuming operation away from the limits of the objective.

    #     Parameters
    #     ----------
    #     output_waist: float The 1/e^2 spot waist at the working distance from
    #         the objective

    #     Outputs
    #     -------
    #     GaussianBeam '''

    #     output_beam = GaussianBeam(waist=output_waist, wavelength=wavelength)
    #     eff_lens = ThinLensGaussian(self.focal_length) input_beam =
    #     eff_lens.beam_after_lens(output_beam,z_input_waist_to_lens=self.focal_length)
    #     return input_beam

    def spot_waist_from_input_beam(self, input_beam):
        beam_diameter = input_beam.waist * 2
        wavelength = input_beam.wavelength
        spot_waist = 2 * wavelength * self.focal_length / np.pi / beam_diameter
    
        if beam_diameter > self.entrance_aperture_diameter:
                print(f"Input beam diameter ({beam_diameter}) is larger than the "+
                    "entrance pupil of the objective ({self.entrance_aperture_diameter})")
                
        return spot_waist
    
    def input_waist_from_spot_beam(self, spot_beam):
        spot_waist = spot_beam.waist * 2
        wavelength = spot_beam.wavelength
        input_waist = 2 * wavelength * self.focal_length / np.pi / spot_waist

        if 2*input_waist > self.entrance_aperture_diameter:
            print(f"Required input beam diameter ({2*input_waist}) is larger than the "+
                  "entrance pupil of the objective ({self.entrance_aperture_diameter})")

        return input_waist