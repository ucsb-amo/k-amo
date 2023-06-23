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

def fraction_str_to_decimal(frac_str):
    num,den = frac_str.split('/')
    return float(num)/float(den)

def state_label_to_quantum_numbers(state_str):

    delims = ["s","p","d","f"]

    Lstr = ""
    for d in delims:
        st_split = state_str.split(d)
        if len(st_split) > 1:
            Lstr = d
            break

    n = int(st_split[0])
    j = fraction_str_to_decimal(st_split[-1])
    
    if Lstr == "s": l = 0
    elif Lstr == "p": l = 1
    elif Lstr == "d": l = 2
    elif Lstr == "f": l = 3
    else: 
        print(Lstr)
        raise ValueError("The state label string did not contain s, p, d, or f.")
    
    return n, l, j

def reduced_dipole_matrix_element_table(n,l,j,portal_data:pd.DataFrame = None):
    """Returns the reduced dipole matrix element and energy for the specified transition.

    Args:
        n (int): The n quantum number for the initial state.
        l (int): The l quantum number for the initial state.
        j (float): The j quantum number for the initial state.
        portal_data (pd.DataFrame, optional): If the matrix element DataFrame
        has already been loaded, provide it here. If unspecified, loads from
        file. Defaults to None.

    Returns:
        pd.DataFrame: a dataframe containing the information about transitions
        from the specified initial state.
        np.ndarray: an ndarray containing the state-labeling string 
    """    
    if portal_data == None:
        portal_data = load_portal_data()
    elif not isinstance(portal_data,pd.DataFrame):
        raise ValueError("portal_data must be a pandas.DataFrame.")
    
    state_i = quantum_numbers_to_state_label(n,l,j)

    elems_from_i = portal_data.loc[ portal_data['Initial'] == state_i ]
    if elems_from_i.empty:
        raise ValueError(f"No matrix element found from state {state_i}. Check the portal_data object.")
    
    allowed_final_states = elems_from_i['Final'].values

    return elems_from_i, allowed_final_states

def matrix_element_from_transition_table(nf,lf,jf,matrix_element_table=None):
    """_summary_

    Args:
        nf (int): The final n quantum number
        lf (int): The final l quantum number
        jf (float): The final j quantum number
        matrix_element_table (pd.DataFrame): The dataframe that
        contains all the rows corresponding to transitions from a desired
        initial state. 

    Returns:
        float: The reduced electric dipole matrix element for the transition
        from the initial state to the final state in atomic units.
        float: The signed transition energy in Joules, defined as E_f - E_i.
    """
    
    state_i = matrix_element_table['Initial'].values[0]
    ni,li,ji = state_label_to_quantum_numbers("4s1/2")

    atom = arc.Potassium39()
    energy_i = atom.getEnergy(ni,li,ji)
    energy_f = atom.getEnergy(nf,lf,jf)

    state_f = quantum_numbers_to_state_label(nf,lf,jf)

    elem_i_to_f = matrix_element_table.loc[ matrix_element_table['Final'] == state_f ]
    if elem_i_to_f.empty:
        raise ValueError(f"No matrix element found from initial state ({state_i}) to {state_f}. Check the portal_data object.")
    
    matrix_element_au = elem_i_to_f['Matrix element (a.u.)'].values
    transition_wavelength_m = elem_i_to_f['Wavelength (nm)'].values * 1.e-9

    transition_energy_J = c.h * c.c / transition_wavelength_m

    # flip the sign on the transition energy if E_i > E_f
    
    if energy_i > energy_f:
        transition_energy_J = -1 * transition_energy_J

    return matrix_element_au, transition_energy_J