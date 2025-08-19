import arc
import numpy as np
import kamo.constants as c
import csv

dv = -1000.

class Potassium39(arc.Potassium39):
    def __init__(self):
        super().__init__()
        self.cross_section = self.get_cross_section()

    def get_magnetic_field_from_ground_state_transition_frequency(self,
                                                                f1, mf1, f2, mf2, transition_frequency_Hz,
                                                                B_bounds_G=(0., 600.),
                                                                N_interp=10000):
        """Returns the magnetic field(s) (in G) at which the transition from
        (f1,mf1) to (f2,mf2) would occur at frequency 'transition_frequency_Hz'.

        Args:
            f1 (int): State 1 quantum number F.
            mf1 (int): State 1 quantum number mF.
            f2 (int): State 2 quantum number F.
            mf2 (int): State 2 quantum number mF.
            transition_frequency_Hz (float or array-like): Measured transition frequency
                between (f1,mF1) and (f2,mF2).
            B_bounds_G (tuple, optional): Bounds used for field finding. Only
                limited to save time computing all the possible transition frequencies. Defaults to (0.,600.).
            N_interp (int, optional): Number of points used for interpolation. Defaults to 10000.

        Raises:
            ValueError: If any returned value is one of the bounds of the
                magnetic field specified, raises an error.

        Returns:
            float or np.ndarray: the magnetic field(s) in G.
        """

        if isinstance(transition_frequency_Hz,float) or isinstance(transition_frequency_Hz,int):
            transition_frequency_Hz = [transition_frequency_Hz]
        if isinstance(transition_frequency_Hz,list):
            transition_frequency_Hz = np.array(transition_frequency_Hz)

        b = np.linspace(B_bounds_G[0], B_bounds_G[1], N_interp)
        f_transitions_MHz = self.get_ground_state_transition_frequency(f1, mf1, f2, mf2, b)
        
        B_G = np.interp(transition_frequency_Hz, f_transitions_MHz * 1.e6, b)

        if np.any((B_G == B_bounds_G[0]) | (B_G == B_bounds_G[1])):
            raise ValueError("One or more transition frequencies correspond with one of the bounds of the magnetic field specified in the 'B_bounds_G' argument. Update this argument and re-run.")

        if B_G.size == 1:
            return B_G[0]
        return B_G
    
    def get_ground_state_transition_sensitivity(self,f1,mf1,f2,mf2,B):
        """Returns the ground state transition sensitivity in MHz/G for (f1,mf1) to
        (f2,mf2) at field B.

        Args:
            f1 (int): State 1 quantum number F.
            mf1 (int): State 1 quantum number mF.
            f2 (int): State 2 quantum number F.
            mf2 (int): State 2 quantum number mF.
            B (float): Magnetic field in G.

        Returns:
            float: ground state transition sensitivity in MHz/G.
        """        
        dB = B * 0.001
        f_B_plus_dB = self.get_ground_state_transition_frequency(f1,mf1,f2,mf2,B+dB)
        f_B = self.get_ground_state_transition_frequency(f1,mf1,f2,mf2,B)
        return (f_B_plus_dB - f_B) / dB
    
    def get_semiclassical_polarizability(self,n1,l1,j1,n2,l2,j2,detuning_Hz):
        """See Grimm 1999 equation 8.
        """        
        f0 = np.abs(self.getTransitionFrequency(n1,l1,j1,n2,l2,j2))
        omega0 = 2 * np.pi * f0
        omega = 2 * np.pi * (f0 + detuning_Hz)
        linewidth = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        return 6 * np.pi * c.epsilon0 * c.c**3 * \
            ( linewidth / omega0**2 ) / ( omega0**2 - omega**2 - 1j * (omega**3/omega0**2) * linewidth )
    
    def get_scattering_rate(self,
                            n1,l1,j1,
                            n2,l2,j2,
                            intensity,
                            detuning_Hz=100.e6):
        """See Grimm 1999 equation 9.
        """        
        alpha = self.get_semiclassical_polarizability(n1,l1,j1,n2,l2,j2,detuning_Hz)
        return 1/(c.hbar * c.epsilon0 * c.c) * np.imag(alpha) * intensity / (2 * np.pi)

    # def get_off_resonant_scattering_rate(self,
    #                         n1,l1,j1,
    #                         n2,l2,j2,
    #                         intensity,
    #                         detuning_Hz=100.e6):
    #     omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
    #     linewidth = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
    #     Delta = 2 * np.pi * detuning_Hz
    #     return 3 * np.pi * c.c**2 / (2 * c.hbar * omega0**3) * (linewidth/Delta)**2 * intensity

    def get_decay_rate(self,n1,l1,j1,n2,l2,j2):
        '''
        Returns spontaneous emission rate for the higher of two states in 1/s.
        '''
        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            Gamma = 1/self.getStateLifetime(n2,l2,j2)
        else:
            Gamma = 1/self.getStateLifetime(n1,l1,j1)
        return Gamma
        
    def lineshape(self,n1=4,l1=0,j1=1/2,n2=4,l2=1,j2=3/2,detuning_Hz=0):
        '''
        Returns the lineshape evaluated at a given detuning for a two-level system. Does not work for excited states.
        '''
        if n2 > 4 or n1 > 4 or l1 > 1 or l2 > 1:
            print("Lineshape not accurate for excited states with n>4.")
        gamma = self.get_decay_rate(n1,l1,j1,n2,l2,j2)
        # transition_omega = np.abs( self.getTransitionFrequency(n1,l1,j1,n2,l2,j2) ) / 2 / np.pi
        detuning_omega = 2 * np.pi * detuning_Hz
        return (1/(2*np.pi)) * gamma / ( detuning_omega**2 + gamma**2 / 4 )

    def get_cross_section(self,n1=4,l1=0,j1=1/2,F1=2,n2=4,l2=1,j2=3/2,F2=3,detuning_Hz=0):

        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            A21 = 2*np.pi*self.getTransitionRate(n2,l2,j2,n1,l1,j1,temperature=0.0) 
        else:
            A21 = 2*np.pi*self.getTransitionRate(n1,l1,j1,n2,l2,j2,temperature=0.0)

        g2 = 2*F2 + 1
        g1 = 2*F1 + 1

        # if ordered:
        #     g_ratio = g2/g1
        # else:
        #     g_ratio = g1/g2
        g_ratio = 1

        omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
        lineshape = self.lineshape(n1,l1,j1,n2,l2,j2,detuning_Hz=detuning_Hz)
        scattering_cross_section = g_ratio * np.pi**2 * c.c**2 / omega0**2 * A21 * lineshape
        return scattering_cross_section
    
    def get_zeeman_shift(self,n,l,j,f,m_f,B):
        '''
        Returns the zeeman energy in units of MHz as a function of B field (in
        Gauss) for a given F, m_f (will also accept mj mi basis) sublevel in the
        specified fine structure manifold.
        '''
        
        if isinstance(B,float) or isinstance(B,int):
            B = np.array([B])
        if isinstance(B,list):
            B = np.array(B)

        # nuclear spin
        n_s = 1.5
        # convert B field in gauss to Tesla
        B = B / 1.e4

        # lookup the input state and reassign quantum numbers if input is given in mj mi basis
        state = self.state_lookup(n,l,j,f,m_f)

        (f,m_f) = state['lf']
        (F1_arc,mf1_arc) = state['lf_arc']

        #for some reason ARCs breit-rabi function doesn't work for K39 ground state, use this instead:
        if l==0:
            if f==1:
                return (((-c.get_hyperfine_constant(0,.5) / 4) 
                        + c.g_I * c.mu_b * m_f * B
                        - (c.get_hyperfine_constant(0,.5)*(n_s+.5) / 2)
                        * np.sqrt(1 + (4 * m_f * (c.get_total_electronic_g_factor(0,.5) - c.g_I) * c.mu_b * B) / (((2 * n_s) + 1) * c.get_hyperfine_constant(0,.5) * (n_s + .5)) 
                                   + (((c.get_total_electronic_g_factor(0,.5) - c.g_I) * c.mu_b * B) / (c.get_hyperfine_constant(0,.5) * (n_s + .5)))**2))
                                   / (c.h * 1.e6))
            if f==2:
                if m_f != -2:
                    return (((-c.get_hyperfine_constant(0,.5) / 4) 
                            + c.g_I * c.mu_b * m_f * B
                            + (c.get_hyperfine_constant(0,.5)*(n_s+.5) / 2)
                            * np.sqrt(1 + (4 * m_f * (c.get_total_electronic_g_factor(0,.5) - c.g_I) * c.mu_b * B) / (((2 * n_s) + 1) * c.get_hyperfine_constant(0,.5) * (n_s + .5)) 
                                    + (((c.get_total_electronic_g_factor(0,.5) - c.g_I) * c.mu_b * B) / (c.get_hyperfine_constant(0,.5) * (n_s + .5)))**2))
                                    / (c.h * 1.e6))
                elif m_f == -2:
                    return (((-c.get_hyperfine_constant(0,.5) / 4) 
                            + c.g_I * c.mu_b * m_f * B
                            + (c.get_hyperfine_constant(0,.5)*(n_s+.5) / 2)
                            * (1 - ((c.get_total_electronic_g_factor(0,.5) - c.g_I) * c.mu_b * B) / (c.get_hyperfine_constant(0,.5) * (n_s + .5))))
                                    / (c.h * 1.e6))
        #for all others use ARC breit rabi function:
        if l!=0:
            zeeman_Evs = self.breitRabi(n, l, j, B)
            zeeman_Es = np.transpose(zeeman_Evs[0])
            for idx in range(len(zeeman_Evs[1])):
                # loop through all the F, mF until you have the right one
                if zeeman_Evs[1][idx] == F1_arc:
                    if zeeman_Evs[2][idx] == mf1_arc:
                        return zeeman_Es[idx] / 1.e6
            
    def get_ground_state_transition_frequency(self,f1,m_f1,f2,m_f2,B=0):
        '''
        Returns the amount of shift in MHz of a given ground state transition
        under external magnetic field B (in Gauss). 
        '''
        n = 4
        l = 0
        j = 1/2
        transition_frequency = abs(self.get_zeeman_shift(n,l,j,f2,m_f2,B) - self.get_zeeman_shift(n,l,j,f1,m_f1,B))

        return transition_frequency

    def get_transition_shift(self,n1,l1,j1,f1,m_f1,n2,l2,j2,f2,m_f2,B=0):
        '''
        Subtracts the calculated Zeeman shift of the excited state (n2,l2,j2,f2,m2) from that of the ground state (n1,l1,j1,f1,m1)
        Returns the amount of shift in MHz of a given optical transition under external magnetic field B (in Gauss). 
        '''
        state1 = self.state_lookup(n1,l1,j1,f1,m_f1)
        state2 = self.state_lookup(n2,l2,j2,f2,m_f2)

        (F1_arc,mf1_arc) = state1['lf_arc']
        (F2_arc,mf2_arc) = state2['lf_arc']

        state1_shift = self.get_zeeman_shift(n1,l1,j1,F1_arc,mf1_arc,B) - self.get_zeeman_shift(n1,l1,j1,F1_arc,mf1_arc,0)
        state2_shift = self.get_zeeman_shift(n2,l2,j2,F2_arc,mf2_arc,B) - self.get_zeeman_shift(n2,l2,j2,F2_arc,mf2_arc,0)

        transition_frequency_shift = state2_shift - state1_shift

        return transition_frequency_shift
    
    def state_dicts(self,n,l,j,hf=True) -> dict:
        '''
        n: principle quantum number (unused for now, but kept for consistency)
        l: angular momentum
        j: total angular momentum
        hf: specify high or low field. For hf=True, the method returns a dict to be keyed with mj, mi quantum numbers, otherwise, method returns a dict to be keyed with f, mf quantum numbers.

        Contains state dicts for 4s.5, 4p.5 and 4p1.5.

        the method returns the dictionary of states for the given manifold, to be keyed either by high or low field quantum numbers depending on hf
        '''

        state_4s1_lf = {
            '2, -2': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-.5,-1.5),
                "lf": (2,-2),
                "lf_arc": (2,-2)
                },
            '1, -1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 1, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (1,-1),
                "lf_arc": (1,-1)
                },
            '1, 0': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (1,0)
                },
            '1, 1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = 1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (1,1)
                },

            '2, -1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (2,-1),
                "lf_arc": (2,-1)
                },
            '2, 0': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (2,0)
                },
            '2, 1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (2,1)
                },
            '2, 2': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (2,2)
                },
        }

        state_4p1_lf = {
            '2, -2': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-.5,-1.5),
                "lf": (2,-2),
                "lf_arc": (2,-2)
                },
            '1, -1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 1, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (1,-1),
                "lf_arc": (1,-1)
                },
            '1, 0': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (1,0)
                },
            '1, 1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = 1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (1,1)
                },

            '2, -1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (2,-1),
                "lf_arc": (2,-1)
                },
            '2, 0': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (2,0)
                },
            '2, 1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (2,1)
                },
            '2, 2': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (2,2)
                },
        }

        state_4p3_lf = {
            '3, -3': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -3',
                "hf": (-1.5,-1.5),
                "lf": (3,-3),
                "lf_arc": (3,-3)
                },
            '2, -2': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-1.5,-.5),
                "lf": (2,-2),
                "lf_arc": (2,-2)
                },
            '1, -1': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = -1',
                "hf": (-1.5,.5),
                "lf": (1,-1),
                "lf_arc": (1,-1)
                },
            '0, 0': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = 1.5',
                "lf_str": r'F = 0,$m_f$ = 0',
                "hf": (-1.5,1.5),
                "lf": (0,0),
                "lf_arc": (0,0)
                },

            '3, -2': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -2',
                "hf": (-.5,-1.5),
                "lf": (3,-2),
                "lf_arc": (3,-2)
                },
            '2, -1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (2,-1),
                "lf_arc": (2,-1)
                },
            '1, 0': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (1,0)
                },
            '1, 1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = 1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (1,1)
                },

            '3, -1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (3,-1),
                "lf_arc": (3,-1)
                },
            '2, 0': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (2,0)
                },
            '2, 1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (2,1)
                },
            '2, 2': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (2,2)
                },

            '3, 0': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = 0',
                "hf": (-.5,-1.5),
                "lf": (3,0),
                "lf_arc": (3,0)
                },
            '3, 1': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = -.5',
                "lf_str": r'F = 3, $m_f$ = 1',
                "hf": (-.5,-.5),
                "lf": (3,1),
                "lf_arc": (3,1)
                },
            '3, 2': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = .5',
                "lf_str": r'F = 3, $m_f$ = 2',
                "hf": (-.5,.5),
                "lf": (3,2),
                "lf_arc": (3,2)
                },
            '3, 3': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = 1.5',
                "lf_str": r'F = 3,$m_f$ = 3',
                "hf": (-.5,1.5),
                "lf": (3,3),
                "lf_arc": (3,3)
                },
        }

        state_4s1_hf = {
            '-0.5, -1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-.5,-1.5),
                "lf": (2,-2),
                "lf_arc": (2,-2)
                },
            '-0.5, -0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 1, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (1,-1),
                "lf_arc": (1,-1)
                },
            '-0.5, 0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (1,0)
                },
            '-0.5, 1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = 1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (1,1)
                },

            '0.5, -1.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (2,-1),
                "lf_arc": (2,-2)
                },
            '0.5, -0.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (2,0)
                },
            '0.5, 0.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (2,1)
                },
            '0.5, 1.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (2,2)
                },
        }

        state_4p1_hf = {
            '-0.5, -1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-.5,-1.5),
                "lf": (2,-2),
                "lf_arc": (2,-2)
                },
            '-0.5, -0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 1, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (1,-1),
                "lf_arc": (1,-1)
                },
            '-0.5, 0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (1,0)
                },
            '-0.5, 1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = -1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (1,1)
                },

            '0.5, -1.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (2,-1),
                "lf_arc": (2,-2)
                },
            '0.5, -0.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (2,0)
                },
            '0.5, 0.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (2,1)
                },
            '0.5, 1.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (2,2)
                },
        }

        state_4p3_hf = {
            '-1.5, -1.5': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -3',
                "hf": (-1.5,-1.5),
                "lf": (3,-3),
                "lf_arc": (1,1)
                },
            '-1.5, -0.5': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-1.5,-.5),
                "lf": (2,-2),
                "lf_arc": (1,0)
                },
            '-1.5, 0.5': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = -1',
                "hf": (-1.5,.5),
                "lf": (1,-1),
                "lf_arc": (1,-1)
                },
            '-1.5, 1.5': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = 1.5',
                "lf_str": r'F = 0,$m_f$ = 0',
                "hf": (-1.5,1.5),
                "lf": (0,0),
                "lf_arc": (0,0)
                },

            '-0.5, -1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -2',
                "hf": (-.5,-1.5),
                "lf": (3,-2),
                "lf_arc": (2,1)
                },
            '-0.5, -0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (2,-1),
                "lf_arc": (2,0)
                },
            '-0.5, 0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (2,-1)
                },
            '-0.5, 1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = 1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (2,-2)
                },

            '0.5, -1.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (3,-1),
                "lf_arc": (2,2)
                },
            '0.5, -0.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (3,-3)
                },
            '0.5, 0.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (3,-2)
                },
            '0.5, 1.5': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (3,-1)
                },

            '1.5, -1.5': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = 0',
                "hf": (-.5,-1.5),
                "lf": (3,0),
                "lf_arc": (3,0)
                },
            '1.5, -0.5': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = -.5',
                "lf_str": r'F = 3, $m_f$ = 1',
                "hf": (-.5,-.5),
                "lf": (3,1),
                "lf_arc": (3,1)
                },
            '1.5, 0.5': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = .5',
                "lf_str": r'F = 3, $m_f$ = 2',
                "hf": (-.5,.5),
                "lf": (3,2),
                "lf_arc": (3,2)
                },
            '1.5, 1.5': {
                "hf_str": r'$m_j$ = 1.5, $m_i$ = 1.5',
                "lf_str": r'F = 3,$m_f$ = 3',
                "hf": (-.5,1.5),
                "lf": (3,3),
                "lf_arc": (3,3)
                },
        }

        if hf==True:
            if l==0:
                return state_4s1_hf
            elif l==1:
                if j==.5:
                    return state_4p1_hf
                elif j==1.5:
                    return state_4p3_hf
        else:
            if l==0:
                return state_4s1_lf
            elif l==1:
                if j==.5:
                    return state_4p1_lf
                elif j==1.5:
                    return state_4p3_lf

    def state_lookup(self,n,l,j,m1,m2):
        """_summary_

        Args:
            n (int): The n quantum number for the state of interest.
            l (int): The l quantum number for the state of interest.
            j (float): The j quantum number for the state of interest.
            m1 (int or float): The first quantum number to specify the state, either F or mJ (depending on regime).
            m2 (int or float): The second quantum number to specify the state, either mF or mI (depending on regime).

        Returns:
            dict: a dict containing state information.
        """        
        if abs(m1) == .5 or abs(m1) == 1.5:
            dct = self.state_dicts(n,l,j)
        else:
            dct = self.state_dicts(n,l,j, hf=False)

        key = str((m1, m2))
        key = key[:-1]
        key = key[1:]
        
        return dct[key]

    def get_scattering_length(self,f,mf,b,
                              interp=False):
        """Get the scattering length for the state (f,mf) at the given field b (G).

        Args:
            f (int): The nuclear quantum number F.
            mf (int): The magnetic sublevel quantum number m_F.
            b (float): The magnetic bias field in Gauss.
            interp (bool, optional): If True, will interpolate the scattering
            length. Defaults to False.
        """

        if not isinstance(b,np.ndarray) or isinstance(b,list):
            b = np.array([b])
        elif isinstance(b,list):
            b = np.array(b)

        def find_nearest_b_idx(b,b_list):
            return np.argmin(np.abs(b_list - b ))
        
        def read_data(path,bdata=False):
            with open(path,'r') as fd:
                reader = csv.reader(fd)
                d = []
                for row in reader:
                    d.append(row)
                float_data = []
                for n in range(len(d)):
                    if bdata:
                        float_data.append(float(d[n][0]))
                    else:
                        float_data.append(float(d[n][0][:25]))
                float_data = np.array(float_data)
            return float_data

        Bval = read_data('B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/aa_1G_1000G/Bval.txt',bdata=True)
        Bval = np.array(Bval)

        if f==1:
            if mf==-1:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/cc_1G_1000G/data.txt'

            elif mf==0:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/bb_1G_1000G/data.txt'

            elif mf==1:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/aa_1G_1000G/data.txt'
        
        elif f==2:
            if mf==-2:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/dd_1G_1000G/data.txt'
        
            elif mf==-1:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/ee_1G_1000G/data.txt'

            elif mf==0:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/ff_1G_1000G/data.txt'

            elif mf==1:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/gg_1G_1000G/data.txt'

            elif mf==2:
                dpath = 'B:/_K/Resources/scattering_lengths/Kokkelmans_data_2/hh_1G_1000G/data.txt'

        data = read_data(dpath)

        if interp:
            scattering_length = np.interp(b, Bval, data)
        else:
            scattering_length = np.zeros(b.shape)
            for n in range(len(b)):
                scattering_length[n] = data[find_nearest_b_idx(b[n],Bval)]

        if len(scattering_length) == 1:
            scattering_length = scattering_length[0]

        return scattering_length