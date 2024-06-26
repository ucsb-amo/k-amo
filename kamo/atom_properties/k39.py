import arc
import numpy as np
import kamo.constants as c

class Potassium39(arc.Potassium39):
    def __init__(self):
        super().__init__()
        self.cross_section = self.get_cross_section()

    def get_decay_rate(self,n1,l1,j1,n2,l2,j2):
        '''
        Returns spontaneous emission rate for the higher of two states.
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
        return (1/(2/np.pi)) * gamma / ( detuning_omega**2 + gamma**2 / 4 )

    def get_cross_section(self,n1=4,l1=0,j1=1/2,F1=2,n2=4,l2=1,j2=3/2,F2=3,detuning_Hz=0):

        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            A21 = 2*np.pi*self.getTransitionRate(n2,l2,j2,n1,l1,j1,temperature=0.0) 
        else:
            A21 = 2*np.pi*self.getTransitionRate(n1,l1,j1,n2,l2,j2,temperature=0.0)

        g2 = 2*F2 + 1
        g1 = 2*F1 + 1

        if ordered:
            g_ratio = g2/g1
        else:
            g_ratio = g1/g2

        omega0 = 2 * np.pi * self.getTransitionFrequency(n1,l1,j1,n2,l2,j2)
        lineshape = self.lineshape(n1,l1,j1,n2,l2,j2,detuning_Hz=detuning_Hz)
        scattering_cross_section = g_ratio * np.pi**2 * c.c**2 / omega0**2 * A21 * lineshape
        return scattering_cross_section
    
    def get_zeeman_shift(self,n,l,j,f,m_f,B):
        '''
        Returns the zeeman energy in units of MHz as a function of B field (in Gauss) for a given F, m_f sublevel in the specified fine structure manifold.
        '''
        #nuclear spin
        n_s = 1.5
        #convert B field in gauss to Tesla
        B = B / 1.e4

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
            zeeman_Evs = self.breitRabi(n, l, j, np.array([B]))
            zeeman_Es = np.transpose(zeeman_Evs[0])
            for idx in range(len(zeeman_Evs[1])):
                if zeeman_Evs[1][idx] == f:
                    if zeeman_Evs[2][idx] == m_f:
                        return zeeman_Es[idx] / 1.e6
            
    def get_microwave_transition_frequency(self,n,l,j,f1,m_f1,f2,m_f2,B=0):
        '''
        Returns the amount of shift in MHz of a given microwave transition under external magnetic field B (in Gauss). 
        '''
        transition_frequency = abs(self.get_zeeman_shift(n,l,j,f2,m_f2,B) - self.get_zeeman_shift(n,l,j,f1,m_f1,B))

        return transition_frequency
    
    def get_transition_shift(self,n1,l1,j1,f1,m_f1,n2,l2,j2,f2,m_f2,B=0):
        '''
        Returns the amount of shift in MHz of a given optical transition under external magnetic field B (in Gauss). 
        '''
        transition_frequency = (self.get_zeeman_shift(n1,l1,j1,f1,m_f1,B) - self.get_zeeman_shift(n1,l1,j1,f1,m_f1,0)) + (self.get_zeeman_shift(n2,l2,j2,f2,m_f2,B) - self.get_zeeman_shift(n2,l2,j2,f2,m_f2,0))

        return transition_frequency
    
    def state_dicts(self,n,l,j,hf=True):
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
                "lf": (2,-2),
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
                "lf_arc": (2,-2)
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
                "lf_arc": (1,1)
                },
            '2, -2': {
                "hf_str": r'$m_j$ = -1.5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = -2',
                "hf": (-1.5,-.5),
                "lf": (2,-2),
                "lf_arc": (1,0)
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
                "lf_arc": (2,1)
                },
            '2, -1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = -1',
                "hf": (-.5,-.5),
                "lf": (2,-1),
                "lf_arc": (2,0)
                },
            '1, 0': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 1, $m_f$ = 0',
                "hf": (-.5,.5),
                "lf": (1,0),
                "lf_arc": (2,-1)
                },
            '1, 1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
                "lf_str": r'F = 1,$m_f$ = 1',
                "hf": (-.5,1.5),
                "lf": (1,1),
                "lf_arc": (2,-2)
                },

            '.3, -1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = -1',
                "hf": (.5,-1.5),
                "lf": (3,-1),
                "lf_arc": (2,2)
                },
            '2, 0': {
                "hf_str": r'$m_j$ = .5, $m_i$ = -.5',
                "lf_str": r'F = 2, $m_f$ = 0',
                "hf": (.5,-.5),
                "lf": (2,0),
                "lf_arc": (3,-3)
                },
            '2, 1': {
                "hf_str": r'$m_j$ = .5, $m_i$ = .5',
                "lf_str": r'F = 2, $m_f$ = 1',
                "hf": (.5,.5),
                "lf": (2,1),
                "lf_arc": (3,-2)
                },
            '2, 2': {
                "hf_str": r'$m_j$ = .5, $m_i$ = 1.5',
                "lf_str": r'F = 2,$m_f$ = 2',
                "hf": (.5,1.5),
                "lf": (2,2),
                "lf_arc": (3,-1)
                },

            '3, 0': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = 0',
                "hf": (-.5,-1.5),
                "lf": (3,0),
                "lf_arc": (3,0)
                },
            '3, 1': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 3, $m_f$ = 1',
                "hf": (-.5,-.5),
                "lf": (3,1),
                "lf_arc": (3,1)
                },
            '3, 2': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 3, $m_f$ = 2',
                "hf": (-.5,.5),
                "lf": (3,2),
                "lf_arc": (3,2)
                },
            '3, 3': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
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
                "hf_str": r'$m_j$ = -.5, $m_i$ = -1.5',
                "lf_str": r'F = 3, $m_f$ = 0',
                "hf": (-.5,-1.5),
                "lf": (3,0),
                "lf_arc": (3,0)
                },
            '1.5, -0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = -.5',
                "lf_str": r'F = 3, $m_f$ = 1',
                "hf": (-.5,-.5),
                "lf": (3,1),
                "lf_arc": (3,1)
                },
            '1.5, 0.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = .5',
                "lf_str": r'F = 3, $m_f$ = 2',
                "hf": (-.5,.5),
                "lf": (3,2),
                "lf_arc": (3,2)
                },
            '1.5, 1.5': {
                "hf_str": r'$m_j$ = -.5, $m_i$ = 1.5',
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
        if abs(m1) == .5 or abs(m1) == 1.5:
            dct = self.state_dicts(l=l,j=j)
        else:
            dct = self.state_dicts(l=l,j=j, hf=False)

        key = str((m1, m2))
        key = key[:-1]
        key = key[1:]
        
        return dct[key]
