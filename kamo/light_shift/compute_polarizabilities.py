import numpy as np
from kamo import constants as c
from kamo.light_shift.parse_portal_data import PortalDataParser
from sympy.physics import wigner

class ComputePolarizabilities():
    def __init__(self,
                atom=None,
                force_arc=False,
                portal_data_parser:PortalDataParser = None,
                n_max=None,
                n_min=None,
                include_core=True,
                portal_species=None):
        """
        Args:
            atom: a kamo atom (default kamo's default atom, 39K, built with
            ``use_portal=not force_arc``).
            include_core (bool, optional): Add the ionic-core polarizability
            (see `return_ionic_core_contribution`) to the scalar part. Defaults
            to True.
            n_min, n_max: see :class:`PortalDataParser` (defaults from the atom).
            portal_species (str, optional): UDel portal species whose matrix
            elements are used when `force_arc` is False. Defaults to the
            atom's (``"K1"`` for potassium).
        """

        if atom is None:
            from kamo.atom_properties.alkali import default_atom
            from kamo.atom_properties.k39 import Potassium39
            atom = default_atom() if not force_arc else Potassium39(use_portal=False)

        self.include_core = include_core

        if portal_data_parser == None:
            self.pdp = PortalDataParser(atom=atom, n_max=n_max, n_min=n_min,
                                        force_arc=force_arc,
                                        portal_species=portal_species)
        else:
            if isinstance(portal_data_parser,PortalDataParser):
                self.pdp = portal_data_parser
            else:
                raise ValueError("Invalid class for argument `portal_data_parser` -- must have class kamo.light_shift.PortalDataParser")
            
        self.atom = atom
            
    def _handle_wavelength_arraylike(self,wavelength_m):
        wavelength_m = np.atleast_1d(wavelength_m)

        WAVELENGTH_M_MAX = 20.e-6
        WAVELENGTH_M_MIN = 100.e-9
        if any(wavelength_m > WAVELENGTH_M_MAX) or any(wavelength_m < WAVELENGTH_M_MIN):
            raise ValueError(f"Wavelength must be between {WAVELENGTH_M_MIN} and {WAVELENGTH_M_MAX}")
        
        return wavelength_m

    def compute_fine_structure_polarizability(self,n,l,j,
                                              wavelength_m):
        """Computes the fine structure polarizabilities for the input state for the
        given wavelength(s).

        Args:
            n (int): the n quantum number.
            l (int): the l quantum number
            j (float): the J quantum number
            wavelength_m (ndarray): the wavelength of the light field.

        Returns:
            float: the scalar polarizability in atomic units of the given fine structure
            state.
            float: the vector polarizability in atomic units of the given fine structure
            state.
            float: the tensor polarizability in atomic units of the given fine structure
            state.
        """    
    
        wavelength_m = self._handle_wavelength_arraylike(wavelength_m)

        alpha_j_scalar = np.zeros(np.shape(wavelength_m))
        alpha_j_vector = np.zeros(np.shape(wavelength_m))
        alpha_j_tensor = np.zeros(np.shape(wavelength_m))

        out = []

        for ii in range(len(wavelength_m)):

            laser_energy_J = c.h * c.c / wavelength_m[ii]

            allowed_final_states = self.pdp.determine_allowed_final_states(l,j)

            if not self.pdp.arc:
                transition_table = self.pdp.reduced_dipole_matrix_element_table(n,l,j)
            
            for state_f in allowed_final_states:

                nf, lf, jf = self.pdp.state_label_to_quantum_numbers(state_f)
                if self.pdp.arc:
                    matrix_element, transition_energy_J = self.pdp.matrix_element_arc(n,l,j,nf,lf,jf)
                else:
                    matrix_element, transition_energy_J = self.pdp.matrix_element_from_transition_table(nf,lf,jf,transition_table)
                matrix_element_SI = matrix_element * c.a0 * c.e

                common_factor = matrix_element_SI**2 / ( transition_energy_J**2 - laser_energy_J**2 )
                
                alpha_j_scalar[ii] += common_factor * transition_energy_J
                alpha_j_vector[ii] += common_factor * (-1)**(j+jf+1) * laser_energy_J * wigner.wigner_6j(j,1,j,1,jf,1)
                alpha_j_tensor[ii] += common_factor * (-1)**(j+jf) * transition_energy_J * wigner.wigner_6j(j,2,j,1,jf,1)
            
        alpha_j_scalar = alpha_j_scalar * (2/3) * 1/(2*j+1)
        alpha_j_vector = alpha_j_vector * np.sqrt( 24*j/(j+1)/(2*j+1) )
        alpha_j_tensor = alpha_j_tensor * np.sqrt( 40*j*(2*j-1)/( 3*(j+1)*(2*j+3)*(2*j+1) ) )

        alpha_j_scalar = alpha_j_scalar / c.convert_polarizability_au_to_SI
        alpha_j_vector = alpha_j_vector / c.convert_polarizability_au_to_SI
        alpha_j_tensor = alpha_j_tensor / c.convert_polarizability_au_to_SI

        if self.include_core:
            alpha_j_scalar = alpha_j_scalar + self.return_ionic_core_contribution()

        return alpha_j_scalar, alpha_j_vector, alpha_j_tensor

    def return_ionic_core_contribution(self):
        '''Returns the ionic core contribution to the polarizability in a.u.
        (K+: 5.457, from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.052504;
        the other alkali ions from Safronova, Johnson & Derevianko, PRA 60,
        4476 (1999), via ``atom.core_polarizability_au``).

        Treated as static: the core's resonances are near 20 eV, so its
        frequency dependence is negligible for wavelengths above ~300 nm.'''
        core = getattr(self.atom, "core_polarizability_au", None)
        if core is None or not np.isfinite(core):
            from kamo.atom_properties.alkali import CORE_POLARIZABILITY_AU
            core = CORE_POLARIZABILITY_AU.get(getattr(self.atom, "elementName", "K")[:2].rstrip("0123456789"), 5.457)
        return float(core)

    def _nuclear_spin(self, I):
        """``I`` as given, else the atom's nuclear spin (3/2, the historical
        default, for a calculator built without an atom)."""
        if I is not None:
            return I
        return float(getattr(getattr(self, "atom", None), "I", 1.5))

    def compute_polarizability(self,
                               n,l,j,F,
                               wavelength_m,
                               I=None):
        """Computes the hyperfine polarizabilities for the input state for the given wavelength(s).
        I = self._nuclear_spin(I)

        Args:
            n (int): the n quantum number.
            l (int): the l quantum number
            j (float): the J quantum number
            F (float): the F quantum number.
            wavelength_m (ndarray): the wavelength of the light field.

        Returns:
            float: the scalar polarizability in atomic units of the given hyperfine
            state.
            float: the vector polarizability in atomic units of the given hyperfine
            state.
            float: the tensor polarizability in atomic units of the given hyperfine
            state.
        """    

        wavelength_m = self._handle_wavelength_arraylike(wavelength_m)

        alpha_j_scalar, alpha_j_vector, alpha_j_tensor = self.compute_fine_structure_polarizability(n,l,j,wavelength_m)
        coeff_F_vector = (-1)**(j+F+I+1) * float(wigner.wigner_6j(F,j,I,j,F,1,prec=10)) * \
            np.sqrt( F*(2*F+1)*(2*j+1)*(j+1)/j/(F+1) )
        if j != 1/2:
            coeff_F_tensor = (-1)**(j+F+I) * float(wigner.wigner_6j(F,j,I,j,F,2,prec=10)) * \
                np.sqrt( F*(2*F-1)*(2*F+1)/(2*F+3)/(F+1) ) * \
                np.sqrt((2*j+3)*(2*j+1)*(j+1)/j/(2*j-1))
        else:
            coeff_F_tensor = 0

        alpha_F_scalar = (alpha_j_scalar).astype(float)
        alpha_F_vector = (alpha_j_vector * coeff_F_vector).astype(float)
        alpha_F_tensor = (alpha_j_tensor * coeff_F_tensor).astype(float)

        return alpha_F_scalar, alpha_F_vector, alpha_F_tensor

    def compute_complete_polarizability(self,
                                        n,l,j,F,mF,
                                        wavelength_m,
                                        polarization=[1,0],
                                        I=None):
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
            I (float, optional): the nuclear spin of the atom. Defaults to the atom's.

        Returns:
            float: the complete polarizability in atomic units of the given hyperfine
            state.
        """
        I = self._nuclear_spin(I)   
        
        wavelength_m = self._handle_wavelength_arraylike(wavelength_m)

        if isinstance(polarization,list):
            polarization = np.array(polarization)

        if len(polarization) == 2:
            polarization = np.append(polarization,0)

        polarization = polarization / np.sqrt(np.sum(np.abs(polarization)**2))
        beta = np.imag( np.cross(polarization, np.conj(polarization)) )[0]
        gamma = (3 * np.conj(polarization[0]) * polarization[0] - 1)/2

        alpha_F_scalar, alpha_F_vector, alpha_F_tensor = self.compute_polarizability(n,l,j,F,wavelength_m,I)

        alpha_F = alpha_F_scalar - beta * mF / (2*F) * alpha_F_vector + \
            gamma * (3*mF**2 - F*(F+1)) / (F*(2*F-1)) * alpha_F_tensor
        
        return alpha_F