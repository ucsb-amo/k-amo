import os
import pandas as pd
import numpy as np
import arc

from kamo import constants as c
from kamo import Potassium39

import time

class PortalDataParser():

    def __init__(self, portal_data:pd.DataFrame = None, n_max=16, n_min=3):
        self.N_MAX = n_max
        self.N_MIN = n_min

        self.atom = Potassium39()
        self.state_energy_list = self._get_state_energy_list()

        if not isinstance(portal_data,pd.DataFrame):
            if portal_data == None:
                self.portal_data = self.load_portal_data()
            else:
                raise ValueError("portal_data must be a pandas.DataFrame.")
        else:
            self.portal_data = portal_data

    def load_portal_data(self, 
                        portal_data_folder = r"B:\_K\Resources\udel_potassium_matrix_elements",
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

    def quantum_numbers_to_state_label(self,n,l,j):
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
            Lstr = "d"
        elif l == 3:
            Lstr = "f"
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

    def fraction_str_to_decimal(self,frac_str):
        num,den = frac_str.split('/')
        return float(num)/float(den)

    def state_label_to_quantum_numbers(self,state_str):

        delims = ["s","p","d","f"]

        Lstr = ""
        for d in delims:
            st_split = state_str.split(d)
            if len(st_split) > 1:
                Lstr = d
                break

        n = int(st_split[0])
        j = self.fraction_str_to_decimal(st_split[-1])
        
        if Lstr == "s": l = 0
        elif Lstr == "p": l = 1
        elif Lstr == "d": l = 2
        elif Lstr == "f": l = 3
        else: 
            raise ValueError("The state label string did not contain s, p, d, or f.")
        
        return n, l, j
    
    def determine_allowed_final_states(self,n,l,j):

        # selection rule for Delta l = +/- 1
        if l == 0:
            lrange = [l+1]
        else:
            lrange = [l-1, l+1]

        n_range = range(self.N_MIN,self.N_MAX+1)

        states = []

        for nf in n_range:

            # impose 0 < lf < nf
            lrange_for_this_n = range(nf)
            # impose 0 < lf < nf
            lrange = np.array(list(set(lrange) & set(lrange_for_this_n)))

            for lf in lrange:
                if ((nf == 3) & (lf < 2)):
                    # no transitions to 3p or lower states
                    pass
                else:
                    # selection rule for Delta j = 0, +/- 1
                    jRange1 = [j-1, j, j+1]
                    # angular momentum addition bounds
                    jRange2 = [lf-1/2,lf+1/2]
                    # union of two j rules
                    jRange = np.array(list(set(jRange1) & set(jRange2)))
                    # j > 0
                    jRange = jRange[jRange>=0]
                    for jf in jRange:
                        states.append(self.quantum_numbers_to_state_label(nf,lf,jf))
        
        return states

    def reduced_dipole_matrix_element_table(self,n,l,j):
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
        
        state_i = self.quantum_numbers_to_state_label(n,l,j)

        elems_from_i = self.portal_data.loc[ self.portal_data['Initial'] == state_i ]
        if elems_from_i.empty:
            raise ValueError(f"No matrix elements found from state {state_i}. Check the portal_data object.")
        
        # allowed_final_states = elems_from_i['Final'].values
        allowed_final_states = self.determine_allowed_final_states(n,l,j)

        return elems_from_i, allowed_final_states           

    def matrix_element_from_transition_table(self,nf,lf,jf,matrix_element_table=None):
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
        state_f = self.quantum_numbers_to_state_label(nf,lf,jf)

        elem_i_to_f = matrix_element_table.loc[ matrix_element_table['Final'] == state_f ]
        if elem_i_to_f.empty:
            # print(f"No portal data for {state_i} to {state_f} -- substituting ARC data.")
            ni,li,ji = self.state_label_to_quantum_numbers(state_i)
            elem_i_to_f = self.atom.getReducedMatrixElementJ(ni,li,ji,nf,lf,jf)
            matrix_element_au = elem_i_to_f * c.e * c.a0

            ei = self.atom.getEnergy(ni,li,ji)
            ef = self.atom.getEnergy(nf,lf,jf)
            transition_energy_J = (ef-ei) * c.convert_joules_per_electronvolt
        else:
            matrix_element_au = elem_i_to_f['Matrix element (a.u.)'].values
            transition_wavelength_m = elem_i_to_f['Wavelength (nm)'].values * 1.e-9

            transition_energy_J = c.h * c.c / transition_wavelength_m

            # flip the sign on the transition energy if E_i > E_f
            state_ordering = self.get_state_energy_order(state_f,state_i)
            transition_energy_J = state_ordering * transition_energy_J

        return matrix_element_au, transition_energy_J

    def get_state_energy_order(self,state_f,state_i):
        '''Returns +1 if state f is higher energy than state i. Otherwise, returns -1.'''
        idx_i = self.state_energy_list.index(state_i)
        idx_f = self.state_energy_list.index(state_f)
        if idx_f > idx_i:
            state_order = 1
        elif idx_i > idx_f:
            state_order = -1
        return state_order
    
    def _get_state_energy_list(self,l_max=3):
        '''Uses ARC to compute the energy level order.'''
        levels = []
        for n in range(self.N_MIN,self.N_MAX+1):
            for l in range(0,n):
                if l < (l_max+1):
                    if ((n == 3) & (l < 2)):
                        jrange = []
                    else:
                        jrange = np.array([l-1/2,l+1/2])
                        jrange = jrange[jrange>=0]
                    for j in jrange:
                        energy = self.atom.getEnergy(n,l,j)
                        levels.append((n,l,j,energy))
        levels.sort(key=lambda x: x[3])
        
        states = []
        for level in levels:
            n, l, j = level[0], level[1], level[2]
            state = self.quantum_numbers_to_state_label(n,l,j)
            states.append(state)

        return states