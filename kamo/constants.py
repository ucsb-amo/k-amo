import arc
import scipy.constants as scon
import numpy as np

atom_K39 = arc.Potassium39()

c = scon.c
kB = scon.k
h = scon.h
a0 = scon.physical_constants['Bohr radius'][0]
hbar = h / 2 / np.pi
e = scon.e

m_K = atom_K39.mass
m_e = scon.m_e
epsilon0 = scon.epsilon_0
epsilon_0 = scon.epsilon_0

convert_polarizability_au_to_SI = 4 * np.pi * epsilon0 * a0**3

#bohr magneton in J / T
mu_b = 9.2740100783e-24

#K39 total nuclear g-factor
g_I = -0.00014193489

#hyperfine constants
def get_hyperfine_constant(l,j,iso=39):
    """Args:
        l (int): the l quantum number, must be either 0 or 1
        j (float): the J quantum number, must be either .5 or 1.5
        iso (optional): the isotope, either 39, 40, or 41, 39 by default.

    Returns:
        hyperfine constant for the specified fine structure manifold and isotope in unites of Hz
    """  
    if iso==39:
        if l==0:
            return h*230.8598601e6
        if l==1:
            if j==.5:
                return h*27.775e6
            elif j==1.5:
                return h*6.093e6
    elif iso==40:
        if l==0:
            return h*-285.7308e6
        if l==1:
            if j==.5:
                return h*-34.523e6
            elif j==1.5:
                return h*-7.585e6
    elif iso==41:
        if l==0:
            return h*127.0069352e6
        if l==1:
            if j==.5:
                return h*15.245e6
            elif j==1.5:
                return h*3.363e6
            
#total electronic g-factors
def get_total_electronic_g_factor(l,j):
    if l==0:
        return 2.00229421
    if l==1:
        if j==.5:
            return 2/3
        if j==1.5:
            return 4/3