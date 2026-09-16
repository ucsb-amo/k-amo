"""Detected-mode projection: what the phase-contrast camera measures.

Every observable before this module is a property of the SCATTERED field -- the
on-axis amplitude ``E_f = sum_j beta_j e^{-ikx_j}`` or the far-field pattern.
The camera measures something else: ``|E_probe e^{i theta} + E_scat,NA|^2`` on
the image plane, where the scattered light has been clipped to the collection
numerical aperture and the unscattered probe retarded by a phase plate.  For a
needle of diameter ~lambda the two differ materially, and every "signal
suppression" or "propagation compression" percentage quoted for the scattered
field is NOT a percentage of the measured phase.

Route: the far-field amplitude ``F(n)`` (exact, no coarse-graining ambiguity) IS
the plane-wave spectrum of the scattered field.  With the scattered field
tending to ``E0 (3/2) F(n) e^{ikr} / (kr)`` and the Weyl identity
``e^{ikr}/r = (i/2pi) int d^2k_perp e^{i(k_perp.rho + k_x |x|)} / k_x``, the
spectrum referred to the plane ``x = 0`` is

    E~(k_perp) / E0 = (3 pi i / (k k_x)) F(n),    n = (k_x, k_perp) / k,

and the (refocused, propagating-mode) field on that plane is
``E_s(rho) = (2pi)^{-2} int E~ e^{i k_perp.rho} d^2k_perp`` over the collection
aperture -- the same angular-spectrum machinery as :mod:`kamo.imaging.readout`,
on the same :class:`kamo.imaging.grid.TransverseGrid`.  Test T21 checks the
identity against the microscopic field on a downstream plane.

Phase plate: the unscattered probe (``k_perp = 0``) and any scattered mode with
``sin(theta) < dimple_NA`` get ``e^{i theta_pp}``.  Idealised: no aberrations,
paraxial polarization mapping (the longitudinal ``E_x`` of the far field is
dropped), unit magnification.

Atom-equivalent units: the signal divided by the slope per ``S_z`` of the SAME
signal for independent atoms at the SAME positions,
``(signal(all up) - signal(all dn)) / N`` -- the only unit in which these
numbers are interpretable across NA, phase-plate and detuning choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.fft as sfft

from kamo.imaging.grid import TransverseGrid

from .cloud import Configuration
from .fields import far_field_amplitude
from .solver import SolveResult, independent_solution
from .system import IncidentField, OperatingPoint, default_incident


@dataclass
class ImagingSystem:
    """Collection optics and phase plate."""

    NA: float = 0.42
    phase_shift: float = np.pi / 2       #: retardation of the unscattered probe
    dimple_NA: float = 0.0               #: scattered modes inside this NA are retarded too
    n_grid: int = 256
    L_box: float = 40e-6                 #: image-plane box (m); resolution lambda/(2 NA) needs L/n < that
    x_focus: float = 0.0                 #: object plane the system is focused on
    polarizer: Optional[np.ndarray] = None   #: transverse analyzer; None = total intensity

    def grid(self, k: float) -> TransverseGrid:
        return TransverseGrid(self.n_grid, self.L_box, k)


@dataclass
class ImagePlane:
    grid: TransverseGrid
    E: np.ndarray            #: (n, n, 2) transverse (y, z) components, / E0
    probe: np.ndarray        #: (2,) the retarded, unscattered probe
    intensity: np.ndarray    #: (n, n)  I / I0

    @property
    def signal_map(self) -> np.ndarray:
        """``I/I0 - 1``."""
        return self.intensity - 1.0

    def on_axis(self) -> float:
        c = self.grid.center
        return float(self.signal_map[c, c])

    def disk_mean(self, radius: float) -> float:
        """Mean of ``I/I0 - 1`` over a disk about the axis (a finite pixel)."""
        m = (self.grid.Y ** 2 + self.grid.Z ** 2) <= radius ** 2
        return float(np.mean(self.signal_map[m]))

    def integrated(self) -> float:
        """``int (I/I0 - 1) d^2rho`` (m^2): the total excess photon count per unit
        incident photon flux density."""
        return float(np.sum(self.signal_map) * self.grid.d ** 2)


def scattered_spectrum(positions, beta, k: float, e_hat, grid: TransverseGrid,
                       NA: float, x_focus: float = 0.0) -> np.ndarray:
    """``E~(k_perp)`` on the grid's modes inside the NA, referred to ``x = x_focus``.

    Shape (n, n, 3); zero outside the aperture.
    """
    mask = grid.na_mask(NA)
    nhat = np.stack([grid.KX[mask], grid.KY[mask], grid.KZ[mask]], axis=1) / k
    F = far_field_amplitude(nhat, positions, beta, k, e_hat)
    Et = np.zeros(grid.KX.shape + (3,), dtype=complex)
    pref = 3 * np.pi * 1j / (k * grid.KX[mask]) * np.exp(1j * grid.KX[mask] * x_focus)
    Et[mask] = pref[:, None] * F
    return Et


def image_field(positions, beta, op: OperatingPoint, system: ImagingSystem,
                incident: Optional[IncidentField] = None) -> ImagePlane:
    """Image-plane field and intensity for a set of dipole amplitudes."""
    inc = default_incident(op) if incident is None else incident
    g = system.grid(op.k)
    Et = scattered_spectrum(positions, beta, op.k, op.e_hat, g, system.NA, system.x_focus)
    plate = np.exp(1j * system.phase_shift)
    dimple = g.KT2 <= (op.k * system.dimple_NA) ** 2
    dimple[0, 0] = True
    Et[dimple] *= plate
    # The grid puts rho = 0 at index n//2 while the DFT puts it at index 0:
    # shift so the image is on the grid's own axes (grid.center is on axis).
    Es = sfft.fftshift(sfft.ifft2(Et, axes=(0, 1), workers=-1), axes=(0, 1)) / g.d ** 2
    probe = inc.field(np.array([[system.x_focus, 0.0, 0.0]]))[0] * plate
    E = Es[..., 1:] + probe[None, None, 1:]
    if system.polarizer is not None:
        pz = np.asarray(system.polarizer, dtype=complex)[1:]
        pz = pz / np.linalg.norm(pz)
        amp = E @ np.conj(pz)
        I = np.abs(amp) ** 2
    else:
        I = np.sum(np.abs(E) ** 2, axis=-1)
    return ImagePlane(g, E, probe[1:], I)


@dataclass
class DetectionResult:
    """The detected signal next to the scattered-field proxy, for one configuration.

    The phase-contrast response of a 500-atom cloud at 9 Gamma is far from
    linear in ``S_z`` (the all-up image is nearly dark on axis), so the
    atom-equivalent unit is the LOCAL derivative: the change of the
    independent-atom signal, at these positions and spins, per atom flipped
    (``d sig / d S_z``, averaged over which atom flips -- exact for independent
    atoms since the image field is linear in the dipoles).

    ``atoms = S_z + (sig - sig_indep) / slope``: independent atoms read
    exactly ``S_z``; interactions move the reading by ``deviation_atoms``.
    """

    on_axis: float                 #: I/I0 - 1 at the image centre
    disk: float                    #: mean over a resolution-sized disk
    integrated: float              #: int (I/I0 - 1) d^2rho, m^2 (energy bookkeeping only)
    forward_amplitude: complex     #: the old proxy, sum beta e^{-ikx}
    on_axis_indep: float           #: independent atoms, same positions and spins
    disk_indep: float
    slope_on_axis: float           #: local d sig / d S_z for independent atoms
    slope_disk: float
    S_z: float

    @property
    def deviation_atoms_on_axis(self) -> float:
        """``(sig - sig_indep) / slope``: what interactions did, in S_z units."""
        return (self.on_axis - self.on_axis_indep) / self.slope_on_axis

    @property
    def deviation_atoms_disk(self) -> float:
        return (self.disk - self.disk_indep) / self.slope_disk

    @property
    def atoms_on_axis(self) -> float:
        """The reading in atom-equivalent units of ``S_z``."""
        return self.S_z + self.deviation_atoms_on_axis

    @property
    def atoms_disk(self) -> float:
        return self.S_z + self.deviation_atoms_disk


def independent_reference(config: Configuration, op: OperatingPoint, system: ImagingSystem,
                          incident: Optional[IncidentField] = None,
                          disk_radius: Optional[float] = None):
    """``(sig_on_axis, sig_disk, slope_on_axis, slope_disk)`` for independent atoms.

    The slopes are the mean change of the signal per atom flipped from dn to up:
    ``2 Re[conj(E_img) . (E_s,up - E_s,dn) / N]`` with ``E_s,up/dn`` the
    independent-atom scattered image fields of an all-up / all-dn cloud at the
    same positions (linear in the dipoles, so this is exact).
    """
    inc = default_incident(op) if incident is None else incident
    r = system_resolution(op, system) if disk_radius is None else disk_radius
    b = independent_solution(config, op, inc)
    im = image_field(config.positions, b, op, system, inc)
    g = im.grid
    disk = (g.Y ** 2 + g.Z ** 2) <= r ** 2
    Es = {}
    for spin in (1, -1):
        c = Configuration(config.positions, np.full(config.N, spin, dtype=np.int8), config.theta)
        imp = image_field(c.positions, independent_solution(c, op, inc), op, system, inc)
        Es[spin] = imp.E - imp.probe[None, None, :]
    dE = (Es[1] - Es[-1]) / config.N
    if system.polarizer is not None:
        pz = np.asarray(system.polarizer, dtype=complex)[1:]
        pz = pz / np.linalg.norm(pz)
        dI = 2 * np.real(np.conj(im.E @ np.conj(pz)) * (dE @ np.conj(pz)))
    else:
        dI = 2 * np.real(np.sum(np.conj(im.E) * dE, axis=-1))
    c = g.center
    return (im.on_axis(), im.disk_mean(r), float(dI[c, c]), float(np.mean(dI[disk])))


def system_resolution(op: OperatingPoint, system: ImagingSystem) -> float:
    """``0.61 lambda / NA`` (m)."""
    return 0.61 * op.wavelength / system.NA


def detect(result: SolveResult, system: ImagingSystem, disk_radius: Optional[float] = None,
           reference=None) -> DetectionResult:
    """Detected-mode signal of a solved configuration, raw and in atom-equivalent units."""
    r = system_resolution(result.op, system) if disk_radius is None else disk_radius
    im = image_field(result.config.positions, result.beta, result.op, system, result.incident)
    if reference is None:
        reference = independent_reference(result.config, result.op, system, result.incident, r)
    return DetectionResult(im.on_axis(), im.disk_mean(r), im.integrated(),
                           result.forward_amplitude(), *reference, result.config.S_z)
