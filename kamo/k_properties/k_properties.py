import arc
import numpy as np

class Potassium39(arc.Potassium39):
    def __init__(self):
        super().__init__()

    def getDecayRate(self,n1,l1,j1,n2,j2,l2):
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
            print("Lineshape not accurate for excited states.")
        gamma = self.getDecayRate(n1,l1,j1,n2,l2,j2)
        # transition_omega = np.abs( self.getTransitionFrequency(n1,l1,j1,n2,l2,j2) ) / 2 / np.pi
        detuning_omega = 2 * np.pi * detuning_Hz
        return (1/2/np.pi) * gamma / ( detuning_omega**2 + gamma**2 / 4 )

    def getCrossSection(self,n1=4,l1=0,j1=1/2,F1=2,n2=4,l2=1,j2=3/2,F2=3,detuning_Hz=0):

        ordered = self.getEnergy(n1,l1,j1) < self.getEnergy(n2,l2,j2)
        if ordered:
            A21 = 1/self.getStateLifetime(n2,l2,j2)
        else:
            A21 = 1/self.getStateLifetime(n1,l1,j1)

        g2 = 2*F2 + 1
        g1 = 2*F1 + 1

        if ordered:
            g_ratio = g2/g1
        else
            
        

