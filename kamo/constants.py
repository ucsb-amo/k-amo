import arc
import scipy.constants as scon
import numpy as np

atom_K39 = arc.Potassium39()

c = scon.c
kB = scon.k
h = scon.h
a0 = scon.physical_constants['Bohr radius'][0]
hbar = h / 2 / np.pi

m_K = atom_K39.mass
m_e = scon.m_e
epsilon0 = scon.epsilon_0

convert_polarizability_au_to_SI = 4 * np.pi * epsilon0 * a0**3

