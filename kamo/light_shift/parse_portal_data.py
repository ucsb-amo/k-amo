import os
import pandas as pd

def load_portal_data(portal_data_folder = r"B:\_K\Resources\udel_potassium_matrix_elements",
                     portal_data_relpath = "K1MatrixElements_complete.csv"):
    """
    Returns the data for all matrix elements in potassium from the UDel atomic physics portal.

    Args:
        portal_data_folder (str, optional): The folder where the complete matrix
        elements csv file is stored. Defaults to
        r"B:\_K\Resources\\udel_potassium_matrix_elements".
        
        portal_data_relpath (str, optional): The filename of the complete matrix
        elements csv file. Defaults to "K1MatrixElements_complete.csv".

    Returns:
        DataFrame: the matrix element data for all transitions.
    """    
    data_fullpath = os.path.join(portal_data_folder,portal_data_relpath)
    data = pd.read_csv(data_fullpath)
    return data

import numpy as np

def quantum_numbers_to_state_label(n,l,j):
    """
    Converts a set of quantum numbers to the Russel-Saunders string that
    appears in the UDel portal csv.

    Args:
        n (int): The n quantum number for the specified state.
        l (int): The l quantum number for the specified state.
        j (float): The j quantum number for the specified state.
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
        jStr == "3/2"
    elif j == 2.5:
        jStr == "5/2"
    elif j == 3.5:
        jStr == "7/2"
    elif j == 4.5:
        jStr == "9/2"

    return f"{n}{Lstr}{jStr}"
    
