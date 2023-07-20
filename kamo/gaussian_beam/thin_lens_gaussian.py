import numpy as np
from kamo import GaussianBeam
from numpy.typing import ArrayLike

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

    def __init__(self, NA=0., working_distance=0., focal_length=0.):
        self.NA = NA
        self.working_distance = working_distance
        self.focal_length = focal_length
        self.entrance_aperture_diameter = 2 * self.NA * self.focal_length

    def spot_waist_from_input_beam(self, input_beam, aperture_warning = False):

        beam_diameter = input_beam.waist * 2
        wavelength = input_beam.wavelength

        if type(beam_diameter) != ArrayLike:
            beam_diameter = np.array([beam_diameter])

        spot_waist = wavelength * self.focal_length / np.pi / input_beam.waist
        spot_beam = GaussianBeam(waist=spot_waist,wavelength=wavelength,power=input_beam.power)

        if aperture_warning:
            for diam in beam_diameter:
                if beam_diameter > self.entrance_aperture_diameter:
                        print(f"Input beam diameter ({diam:1.2e}) is larger than the "+
                            f"entrance pupil of the objective ({self.entrance_aperture_diameter:1.2e})")
                
        return spot_beam
    
    def input_waist_from_spot_beam(self, spot_beam, aperture_warning = False):
        spot_waist = spot_beam.waist
        wavelength = spot_beam.wavelength
        
        if type(spot_waist) != ArrayLike:
            spot_waist = np.array([spot_waist])

        input_waist = wavelength * self.focal_length / np.pi / spot_waist
        input_beam = GaussianBeam(waist=input_waist,wavelength=wavelength,power=spot_beam.power)

        if aperture_warning:
            for idx in range(len(input_waist)):
                inwaist = input_waist[idx]
                if 2*inwaist > self.entrance_aperture_diameter:
                    print(f"Required input beam diameter ({2*inwaist:1.2e}) to achieve "+
                        f"spot diameter ({2*spot_waist[idx]:1.2e}) is larger than the "+
                        f"entrance pupil of the objective ({self.entrance_aperture_diameter:1.2e}).")

        return input_beam