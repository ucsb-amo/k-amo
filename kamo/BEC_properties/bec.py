"""Harmonic Thomas-Fermi estimates for a single-tweezer BEC (legacy interface).

``BEC`` keeps its original methods and units -- angular frequencies in rad/s,
scattering lengths in Bohr radii -- but carries no physics of its own:

* the axial trap frequency comes from the Hessian of a :class:`kamo.trap.Trap`
  built from a :class:`kamo.trap.Tweezer` of the configured wavelength and waist
  (default 1064 nm and 3.8 um, as before);
* the chemical potential and radii are the harmonic closed forms of
  :mod:`kamo.trap.thomas_fermi`.

The results agree with the previous closed-form implementation to ~1e-12.  The
reference trap has no gravity and a placeholder polarizability and mass: a
Gaussian beam's frequency *ratio* depends on neither, so nothing here touches the
UDel portal.  For the sagged, anharmonic, state-dependent answer use
``kamo.trap`` directly (``Trap``, ``solve``).
"""
import functools

import numpy as np
from scipy.constants import atomic_mass

import kamo.constants as c

# Placeholders for the reference trap: the axial/radial frequency ratio of a
# Gaussian beam is independent of the power, the polarizability and the mass.
_REF_POWER_W = 1e-3
_REF_ALPHA_SI = 600.0 * c.convert_polarizability_au_to_SI   # ~ K 4S at 1064 nm
_REF_MASS_KG = 39.0 * atomic_mass


@functools.lru_cache(maxsize=128)
def _axial_over_radial(lmbda: float, waist: float) -> float:
    """omega_axial / omega_radial of a tweezer, from the kamo.trap Hessian."""
    from kamo.trap import Trap, Tweezer
    tweezer = Tweezer(waist=waist, wavelength_m=lmbda, power=_REF_POWER_W)
    tf = Trap(tweezer, polarizability_SI=_REF_ALPHA_SI, mass=_REF_MASS_KG,
              gravity=False).trap_frequencies()
    axial = int(np.argmax(np.abs(tf.axes @ tweezer.propagation_direction)))
    radial = 0 if axial != 0 else 1
    return float(tf.frequencies_Hz[axial] / tf.frequencies_Hz[radial])


def _ratio(lmbda, waist):
    if np.ndim(lmbda) == 0 and np.ndim(waist) == 0:
        return _axial_over_radial(float(lmbda), float(waist))
    return np.vectorize(lambda l, w: _axial_over_radial(float(l), float(w)))(lmbda, waist)


class BEC():
    """Harmonic TF estimates for a BEC in one tweezer of wavelength ``lmbda`` and
    waist ``waist`` (m); see the module docstring."""

    def __init__(self, lmbda=1064.e-9, waist=3.8e-6):
        self.lmbda = lmbda
        self.waist = waist

    def oscillator_len(self,trap_freq):
        """
        Calculate oscillator length for a given trap frequency.

        Parameters:
        trap_freq (float): Trap frequency in 2pi * Hz.

        Returns:
        float: oscillator length in m.
        """
        return np.sqrt(c.hbar / (c.m_K * trap_freq))

    def get_axial_trap_frequency(self,radial_trap_frequency,lmbda=None,waist=None):
        """
        Calculate the axial trap frequency for an optical dipole trap.

        Parameters:
        radial_trap_frequency (float): Radial trap frequency in 2pi*Hz.
        lmbda (float): Wavelength of the trapping light in meters (default: self.lmbda).
        waist (float): Beam waist in meters (default: self.waist).

        Returns:
        float: Axial trap frequency in 2pi*Hz.
        """
        lmbda = self.lmbda if lmbda is None else lmbda
        waist = self.waist if waist is None else waist
        return radial_trap_frequency * _ratio(lmbda, waist)

    def chemical_potential(self,n_atoms,a_scattering,radial_trap_frequency):
        """
        Calculate the chemical potential for an interacting BEC.

        Parameters:
        n_atoms (float): number of atoms.
        a_scattering (float): Scattering length in Bohr radii.
        radial_trap_frequency (float): radial trap frequency in 2pi*Hz.

        Returns:
        float: chemical potential in 2pi Hz.
        """
        from kamo.trap.thomas_fermi import harmonic_chemical_potential_J
        ax_trap = self.get_axial_trap_frequency(radial_trap_frequency=radial_trap_frequency)
        bar_omega = (radial_trap_frequency*radial_trap_frequency*ax_trap)**(1/3)
        mu_J = harmonic_chemical_potential_J(n_atoms, a_scattering * c.a0, bar_omega, c.m_K)
        return mu_J / c.hbar

    def thomas_fermi_radius(self,n_atoms,a_scattering,radial_trap_frequency,trap_frequency):
        """
        Calculate the Thomas-Fermi radius for one dimension of an interacting BEC with given trap frequency.

        Parameters:
        n_atoms (float): number of atoms.
        a_scattering (float): Scattering length in Bohr radii.
        radial_trap_frequency (float): radial trap frequency in 2pi*Hz.
        trap_frequency (float): trap frequency of the desired dimension in 2pi*Hz.

        Returns:
        float: Thomas-Fermi radius
        """
        from kamo.trap.thomas_fermi import harmonic_tf_radius
        chem_pot = c.hbar * self.chemical_potential(n_atoms=n_atoms,a_scattering=a_scattering,radial_trap_frequency=radial_trap_frequency)
        return harmonic_tf_radius(chem_pot, trap_frequency, c.m_K)
