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
            zeeman_Es = self.breitRabi(n, l, j, np.array([B]))
            zeeman_Es = np.transpose(zeeman_Es[0])
            if f==0:
                return zeeman_Es[m_f] / 1.e6
            if f==1:
                return zeeman_Es[m_f+1] / 1.e6
            if f==2:
                return zeeman_Es[m_f+2] / 1.e6
            if f==3:
                return zeeman_Es[m_f+3] / 1.e6
                


