import arc
import numpy as np
import kamo.constants as c

class Potassium39(arc.Potassium39):
    def __init__(self):
        super().__init__()
        self.cross_section = self.get_cross_section()

    def get_decay_rate(self,n1,l1,j1,n2,l2,j2):
        '''
        Returns decay rate (2*pi*(1/tau), units of radians/second) for transition from |n1,l1,j1> to |n2,l2,j2>. 
        Automatically assumes atom is in higher energy state.
        '''
        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            Gamma = self.getTransitionRate(n2,l2,j2,n1,l1,j1)
        else:
            Gamma = self.getTransitionRate(n1,l1,j1,n2,l2,j2)
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
        return (1/2/np.pi) * gamma / ( detuning_omega**2 + gamma**2 / 4 )

    def get_cross_section(self,n1=4,l1=0,j1=1/2,F1=2,n2=4,l2=1,j2=3/2,F2=3,detuning_Hz=0):

        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            A21 = 1/self.getStateLifetime(n2,l2,j2)
        else:
            A21 = 1/self.getStateLifetime(n1,l1,j1)

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

