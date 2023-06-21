import os
import pandas as pd
import numpy as np
import arc

from kamo import constants as c

def load_portal_data(portal_data_folder = r"B:\_K\Resources\udel_potassium_matrix_elements",
                     portal_data_relpath = "K1MatrixElements_complete.csv"):
    """
    Returns the data for all matrix elements in potassium from the UDel atomic physics portal.

    Args:
        portal_data_folder (str, optional): The folder where the complete matrix
        elements csv file is stored. Defaults to
        r"B:\\_K\\Resources\\udel_potassium_matrix_elements".
        
        portal_data_relpath (str, optional): The filename of the complete matrix
        elements csv file. Defaults to "K1MatrixElements_complete.csv".

    Returns:
        DataFrame: the matrix element data for all transitions.
    """    
    data_fullpath = os.path.join(portal_data_folder,portal_data_relpath)
    data = pd.read_csv(data_fullpath)
    return data

def quantum_numbers_to_state_label(n,l,j):
    """
    Converts a set of quantum numbers to the Russel-Saunders string that
    appears in the UDel portal csv.

    Args:
        n (int): The n quantum number for the specified state.
        l (int): The l quantum number for the specified state.
        j (float): The j quantum number for the specified state.

    Returns:
        str: A string labeling the state in Russel-Saunders notation, for use in
        filtering the UDel atomic physics portal data.
    """

    if l == 0:
        Lstr = "s"
    elif l == 1:
        Lstr = "p"
    elif l == 2:
        Lstr == "d"
    elif l == 3:
        Lstr == "f"
    else:
        raise ValueError("The quantum number 'l' must be between 0 and 3 -- maximum state supported is f")
    
    if l == 0:
        Jbounds = [1/2]
    else:
        Jbounds = [l-1/2,l+1/2]

    if j not in Jbounds:
        raise ValueError("The quantum number 'j' must take the value l-1/2 or l+1/2")
    
    if j == 0.5:
        jStr = "1/2"
    elif j == 1.5:
        jStr = "3/2"
    elif j == 2.5:
        jStr = "5/2"
    elif j == 3.5:
        jStr = "7/2"
    elif j == 4.5:
        jStr = "9/2"

    return f"{n}{Lstr}{jStr}"

def reduced_dipole_matrix_element(n1,l1,j1,n2,l2,j2,portal_data:pd.DataFrame = None):
    """Returns the reduced dipole matrix element and energy for the specified transition.

    Args:
        n1 (int): The n quantum number for the initial state.
        l1 (int): The l quantum number for the initial state.
        j1 (float): The j quantum number for the initial state.
        n2 (int): The n quantum number for the final state.
        l2 (int): The l quantum number for the final state.
        j2 (float): The j quantum number for the final state.
        portal_data (pd.DataFrame, optional): If the matrix element DataFrame
        has already been loaded, provide it here. If unspecified, loads from
        file. Defaults to None.

    Returns:
        tuple: The dipole matrix element (in SI units), and the signed
        transition energy (in J). Transition energy is defined as E_final -
        E_initial.
    """    
    if portal_data == None:
        portal_data = load_portal_data()
    elif not isinstance(portal_data,pd.DataFrame):
        raise ValueError("portal_data must be a pandas.DataFrame.")
    
    state_i = quantum_numbers_to_state_label(n1,l1,j1)
    state_f = quantum_numbers_to_state_label(n2,l2,j2)

    elems_from_i = portal_data.loc[ portal_data['Initial'] == state_i ]
    if elems_from_i.empty:
        raise ValueError(f"No matrix element found from state {state_i}. Check the portal_data object.")

    elem_i_to_f = elems_from_i.loc[ elems_from_i['Final'] == state_f ]
    if elem_i_to_f.empty:
        raise ValueError(f"No matrix element found from initial state ({state_i}) to {state_f}. Check the portal_data object.")
    
    matrix_element_au = elem_i_to_f['Matrix element (a.u.)'].values
    transition_wavelength_m = elem_i_to_f['Wavelength (nm)'].values * 1.e-9

    transition_energy_J = c.h * c.c / transition_wavelength_m

    # flip the sign on the transition energy if E_i > E_f
    atom = arc.Potassium39()
    if atom.getEnergy(n1,l1,j1) > atom.getEnergy(n2,l2,j2):
        transition_energy_J = -1 * transition_energy_J

    return matrix_element_au, transition_energy_J