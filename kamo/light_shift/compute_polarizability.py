import numpy as np
from kamo import constants as c
import kamo.light_shift.parse_portal_data as ppd
from sympy.physics import wigner

def compute_polarizability(n,l,j,F,wavelength_m,I=3/2):
    """Computes the hyperfine polarizabilities for the input state for the given wavelength(s).

    Args:
        n (int): the n quantum number.
        l (int): the l quantum number
        j (float): the J quantum number
        F (float): the F quantum number.
        wavelength_m (ndarray): the wavelength of the light field.
        I (float, optional): the nuclear spin of the atom. Defaults to 3/2.

    Returns:
        float: the scalar polarizability in atomic units of the given hyperfine
        state.
        float: the vector polarizability in atomic units of the given hyperfine
        state.
        float: the tensor polarizability in atomic units of the given hyperfine
        state.
    """    

    if isinstance(wavelength_m,list):
        wavelength_m = np.array(wavelength_m)
    elif isinstance(wavelength_m,float):
        wavelength_m = np.array([wavelength_m])

    alpha_j_scalar = np.zeros(np.shape(wavelength_m))
    alpha_j_vector = np.zeros(np.shape(wavelength_m))
    alpha_j_tensor = np.zeros(np.shape(wavelength_m))

    for ii in range(len(wavelength_m)):

        laser_energy_J = c.h * c.c / wavelength_m[ii]

        transition_table, allowed_final_states = ppd.reduced_dipole_matrix_element_table(n,l,j)

        for state_f in allowed_final_states:

            nf, lf, jf = ppd.state_label_to_quantum_numbers(state_f)

            matrix_element, transition_energy_J  = ppd.matrix_element_from_transition_table(nf,lf,jf,transition_table)
            matrix_element_SI = matrix_element * c.a0 * c.e

            common_factor = matrix_element_SI**2 / ( transition_energy_J**2 - laser_energy_J**2 )
            
            alpha_j_scalar[ii] += common_factor * transition_energy_J
            alpha_j_vector[ii] += common_factor * (-1)**(j+jf+1) * laser_energy_J * wigner.wigner_6j(j,1,j,1,jf,1)
            alpha_j_tensor[ii] += common_factor * (-1)**(j+jf) * transition_energy_J * wigner.wigner_6j(j,2,j,1,jf,1)

    alpha_j_scalar = alpha_j_scalar * (2/3) * 1/(2*j+1)
    alpha_j_vector = alpha_j_vector * np.sqrt( 24*j/(j+1)/(2*j+1) )
    alpha_j_tensor = alpha_j_tensor * np.sqrt( 40*j*(2*j-1)/( 3*(j+1)*(2*j+3)*(2*j+1) ) )

    coeff_F_vector = (-1)**(j+F+I+1) * wigner.wigner_6j(F,j,I,j,F,1) * \
          np.sqrt( F*(2*F+1)*(2*j+1)*(j+1)/j/(F+1) )
    if j != 1/2:
        coeff_F_tensor = (-1)**(j+F+I) * wigner.wigner_6j(F,j,I,j,F,2) * \
            np.sqrt( F*(2*F-1)*(2*F+1)/(2*F+3)/(F+1) ) * \
            np.sqrt((2*j+3)*(2*j+1)*(j+1)/j/(2*j-1))
    else:
        coeff_F_tensor = 0

    alpha_F_scalar = alpha_j_scalar
    alpha_F_vector = alpha_j_vector * coeff_F_vector
    alpha_F_tensor = alpha_j_tensor * coeff_F_tensor

    alpha_F_scalar = alpha_F_scalar / c.convert_polarizability_au_to_SI
    alpha_F_vector = alpha_F_vector / c.convert_polarizability_au_to_SI
    alpha_F_tensor = alpha_F_tensor / c.convert_polarizability_au_to_SI

    return alpha_F_scalar, alpha_F_vector, alpha_F_tensor

def compute_complete_polarizability(n,l,j,F,mF,wavelength_m,polarization=[1,0],I=3/2):
    """
    Computes the total hyperfine polarizability for the input state for the
    given wavelength(s). This number is proportional to the energy shift of the
    atom in an AC light field.

    Args:
        n (int): the n quantum number.
        l (int): the l quantum number
        j (float): the J quantum number
        F (float): the F quantum number.
        wavelength_m (ndarray): the wavelength of the light field.
        I (float, optional): the nuclear spin of the atom. Defaults to 3/2.

    Returns:
        float: the complete polarizability in atomic units of the given hyperfine
        state.
    """   
    
    if isinstance(wavelength_m,list):
        wavelength_m = np.array(wavelength_m)
    elif isinstance(wavelength_m,float):
        wavelength_m = np.array([wavelength_m])

    if isinstance(polarization,list):
        polarization = np.array(polarization)

    if len(polarization) == 2:
        polarization = np.append(polarization,0)

    polarization = polarization / np.sqrt(np.sum(np.abs(polarization)**2))
    beta = np.imag( np.cross(polarization, np.conj(polarization)) )[0]
    gamma = (3 * np.conj(polarization[0]) * polarization[0] - 1)/2

    alpha_F_scalar, alpha_F_vector, alpha_F_tensor = compute_polarizability(n,l,j,F,wavelength_m,I)

    alpha_F = alpha_F_scalar - beta * mF / (2*F) * alpha_F_vector + \
        gamma * (3*mF**2 - F*(F+1)) / (F*(2*F-1)) * alpha_F_tensor
    
    return alpha_F