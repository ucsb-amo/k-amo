import numpy as np
from kamo import Potassium39
from kamo.light_shift.compute_polarizabilities import ComputePolarizabilities
from kamo.light_shift.parse_portal_data import PortalDataParser
from kamo import constants as c

class ComputeLightShift():

    def __init__(self,
                atom=Potassium39(),
                force_arc=False,
                portal_data_parser:PortalDataParser = None):
        if portal_data_parser == None:
            self.pdp = PortalDataParser(atom=atom,force_arc=force_arc)
        else:
            if isinstance(portal_data_parser,PortalDataParser):
                self.pdp = portal_data_parser
            else:
                raise ValueError("Invalid class for argument `portal_data_parser` -- must have class kamo.light_shift.PortalDataParser")
            
        self.ComputePolarizabilities = ComputePolarizabilities(atom=atom,
                                          force_arc=force_arc,
                                          portal_data_parser=self.pdp)

    def compute_state_shift(self,
                            n,l,j,F,mF,
                            wavelength_m,
                            intensity,
                            polarization=[1,0],
                            I=3/2):
        """
        Computes the energy shift in Hz of a given state in a beam of the given
        wavelength, intensity, and polarization.

        Args:
            n (int): n quantum number for the state.
            l (int): l quantum number for the state.
            j (float): j quantum number for the state.
            F (int): F quantum number for the state.
            mF (int): mF quantum number for the state.
            wavelength_m (_type_): The wavelength of the field.
            intensity (float): The light field intensity at the position of the atom.
            polarization (list, optional): The polarization of the light relative t
            the quantization axis (+x). Defaults to [1,0].
            I (float, optional): The nuclear spin. Defaults to 3/2.

        Returns:
            float: the energy shift in Hz of the state due to the incident light field. 
        """
        alpha_F = self.ComputePolarizabilities.compute_complete_polarizability(n,l,j,F,mF,wavelength_m,polarization,I)
        alpha_F_SI = alpha_F * c.convert_polarizability_au_to_SI
        state_shift_J = -1/(2*c.c*c.epsilon0) * alpha_F_SI * intensity
        state_shift_Hz = state_shift_J.astype('float64') / c.h
        return state_shift_Hz

    def compute_transition_shift(self,ni,li,ji,Fi,mFi,
                                nf,lf,jf,Ff,mFf,
                                wavelength_m,intensity,
                                polarization=[1,0],I=3/2):
        """_summary_

        Args:
            ni (int): n quantum number for state i.
            li (int): l quantum number for state i.
            ji (float): j quantum number for state i.
            Fi (int): F quantum number for state i.
            mFi (int): mF quantum number for state i.
            nf (int): n quantum number for state f.
            lf (int): l quantum number for state f.
            jf (float): j quantum number for state f.
            Ff (int): F quantum number for state f.
            mFf (int): mF quantum number for state f.
            wavelength_m (float): The wavelength of the field.
            intensity (float): The light field intensity at the position of the atom.
            polarization (list, optional): The polarization of the light relative t
            the quantization axis (+x). Defaults to [1,0].
            I (float, optional): The nuclear spin. Defaults to 3/2.

        Returns:
            float: the energy shift in Hz of the transition (defined as shift_f -
            shift_i) due to the incident light field. 
        """    
        state_shift_i_Hz = self.compute_state_shift(ni,li,ji,Fi,mFi,wavelength_m,intensity,polarization,I)
        state_shift_f_Hz = self.compute_state_shift(nf,lf,jf,Ff,mFf,wavelength_m,intensity,polarization,I)
        transition_shift_Hz = state_shift_f_Hz - state_shift_i_Hz
        transition_shift_Hz = transition_shift_Hz.astype('float64')
        return transition_shift_Hz

