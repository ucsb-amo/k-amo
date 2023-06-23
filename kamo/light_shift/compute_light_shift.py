import numpy as np
import kamo.light_shift.compute_polarizability as cp
from kamo import constants as c

def compute_state_shift(n,l,j,F,mF,wavelength_m,intensity,polarization=[1,0],I=3/2):
    """
    Computes the energy shift of a given state in a beam of the given
    wavelength, intensity, and polarization.

    Args:
        n (_type_): n quantum number.
        l (_type_): l quantum number.
        j (_type_): j quantum number.
        F (_type_): F quantum number.
        mF (_type_): mF quantum number.
        wavelength_m (_type_): The wavelength of the field.
        intensity (float): The light field intensity at the position of the atom.
        polarization (list, optional): The polarization of the light relative t
        the quantization axis (+x). Defaults to [1,0].
        I (float, optional): The nuclear spin. Defaults to 3/2.

    Returns:
        float: the energy shift in Hz of the state due to the incident light field. 
    """
    alpha_F = cp.compute_complete_polarizability(n,l,j,F,mF,wavelength_m,polarization)
    alpha_F_SI = alpha_F * c.convert_polarizability_au_to_SI
    state_shift_J = -1/(2*c.c*c.epsilon0) * alpha_F_SI * intensity
    state_shift_Hz = state_shift_J.astype('float64') / c.h
    return state_shift_Hz

def compute_transition_shift(ni,li,ji,Fi,mFi,
                             nf,lf,jf,Ff,mFf,
                             wavelength_m,intensity,polarization=[1,0],I=3/2):
    """_summary_

    Args:
        ni (_type_): n quantum number for state i.
        li (_type_): l quantum number for state i.
        ji (_type_): j quantum number for state i.
        Fi (_type_): F quantum number for state i.
        mFi (_type_): mF quantum number for state i.
        nf (_type_): n quantum number for state f.
        lf (_type_): l quantum number for state f.
        jf (_type_): j quantum number for state f.
        Ff (_type_): F quantum number for state f.
        mFf (_type_): mF quantum number for state f.
        wavelength_m (_type_): The wavelength of the field.
        intensity (float): The light field intensity at the position of the atom.
        polarization (list, optional): The polarization of the light relative t
        the quantization axis (+x). Defaults to [1,0].
        I (float, optional): The nuclear spin. Defaults to 3/2.

    Returns:
        float: the energy shift in Hz of the transition (defined as shift_f -
        shift_i) due to the incident light field. 
    """    
    state_shift_i_Hz = compute_state_shift(ni,li,ji,Fi,mFi,wavelength_m,intensity,polarization,I)
    state_shift_f_Hz = compute_state_shift(nf,lf,jf,Ff,mFf,wavelength_m,intensity,polarization,I)
    transition_shift_Hz = state_shift_i_Hz - state_shift_f_Hz
    transition_shift_Hz = transition_shift_Hz.astype('float64')
    return transition_shift_Hz

