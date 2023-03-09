import arc
import scipy.constants as scon
import numpy as np

atom_K39 = arc.Potassium39()

c = scon.c
kB = scon.k
h = scon.h
hbar = h / 2 / np.pi

m_K = atom_K39.mass
m_e = scon.m_e
epsilon0 = scon.epsilon_0

