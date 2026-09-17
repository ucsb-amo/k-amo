"""The electromagnetic field everywhere, reconstructed from the dipole amplitudes.

Once the ``beta_j`` are known the field at ANY point is exact:

    E(r) / E0 = E_inc(r) / E0 + sum_j beta_j Gt(r - r_j) . e_hat

with the dimensionless dyadic ``Gt = (3/2) (e^{ix}/x) [(1 - r r) + (i/x - 1/x^2)(1 - 3 r r)]``
(the ``3/2`` is the prefactor the build specification asked to have verified
against the single-atom cross section, S3, and the analytic dipole field, T6).

Three distinct fields -- do not conflate them
---------------------------------------------
1. **Microscopic field** ``E(r)`` (:func:`field_at`, :func:`field_on_grid`).
   Exact and genuinely spiky: inside the cloud it is dominated by the ``1/r^3``
   near field of whichever atom is nearest and it diverges AT the atoms.  That is
   physically correct, not an artefact.  Points within ``mask_radius`` (default
   10 nm) of an atom are flagged; the field there is enormous, real, and
   grid-resolution dependent.

2. **Coherent (macroscopic) field** ``<E(r)>`` (:func:`coherent_field`), the
   CONFIGURATION average at a fixed point.  This is what a susceptibility-based
   propagation computes and the only thing comparable to one.  It is smooth and
   finite everywhere, including at atomic positions, and it carries the lensing.

   Average over configurations, not over space.  Spatial coarse-graining needs
   ``n^{-1/3} << R << lambda`` and there is no room: ``n^{-1/3} ~ 130 nm``,
   ``lambda = 767 nm``, and a 250-350 nm cell holds 2-8 atoms (35-70%
   granularity noise).  Configuration averaging has no such requirement and
   converges on an arbitrarily fine grid, because the angular average of the
   near-field factor ``B = 1 - (3/2) sin^2 theta`` over a sphere is exactly
   zero: the ``1/r^3`` term cancels in the mean while surviving in every shot.
   :func:`coarse_grain` exists but warns with the actual atoms per cell.

   Convergence is slow without an exclusion radius.  The near-field spikes have
   divergent variance, so 40 configurations leaves speckle in the mean.  Passing
   ``R_exc`` omits dipoles within ``R_exc`` of the observation point; the
   vanishing angular average means this does not bias the mean (to the extent
   the density is uniform over ``R_exc``) but removes most of the variance.
   ``R_exc = 150 nm ~`` the interparticle spacing gives clean maps at 40
   configurations (test T19).  Default 0: the exact microscopic field.

3. **Exciting field at an atom** (:func:`exciting_field_at_atoms`): the
   microscopic field minus that atom's own field -- the one field that is finite
   and unambiguous at an atomic position, and the one satisfying
   ``beta_j = alpha_j conj(e_hat) . E_exc(r_j)`` (S8 / T5).

Memory: a ``200 x 200`` grid with ``N = 1000`` atoms is ``4e4 x 1e3 x 3``
complex numbers per component if built at once.  Everything here is chunked over
observation points (``chunk`` points at a time, ``chunk * N * 3 * 16`` bytes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .system import IncidentField, OperatingPoint, default_incident

DEFAULT_MASK_RADIUS = 10e-9
DEFAULT_CHUNK = 1024


class CoarseGrainWarning(UserWarning):
    pass


# ---------------------------------------------------------- dipole fields


def dipole_field_vectors(dvec, k: float, e_hat, xp=None):
    """``Gt(dvec) . e_hat`` for separation vectors ``dvec`` (..., 3) -> (..., 3).

    Array-module generic (numpy or torch) so the GPU path shares this code.
    Entries with ``|dvec| = 0`` are returned as zero (the self term).
    """
    if xp is None:
        xp = np
    e = e_hat
    r = xp.sqrt(xp.sum(dvec * dvec, -1))
    safe = xp.where(r > 0, r, 1.0)
    rhat = dvec / safe[..., None]
    x = k * safe
    re = xp.sum(rhat * e, -1)                      # rhat . e_hat (complex scalar; torch-safe)
    pref = 1.5 * xp.exp(1j * x) / x
    coef = 1j / x - 1.0 / (x * x)
    term = (e - rhat * re[..., None]) + coef[..., None] * (e - 3.0 * rhat * re[..., None])
    out = pref[..., None] * term
    return xp.where((r > 0)[..., None], out, 0.0 * out)


def analytic_dipole_field(points, position, beta: complex, k: float, e_hat) -> np.ndarray:
    """Closed-form field of one dipole ``beta e_hat`` at ``position`` (T6 reference).

    Written independently of :func:`dipole_field_vectors`, straight from the
    textbook near/intermediate/far-zone expansion of a radiating dipole:

        E = (k^2 / eps0) e^{ikr}/(4 pi r) [ (1 + i/kr - 1/(kr)^2) p
            - (1 + 3i/kr - 3/(kr)^2) r(r.p) ]

    in units where ``p = beta e_hat`` carries ``6 pi eps0 E0 / k^3``, so that the
    prefactor becomes ``(3/2) e^{ikr} / (kr)``.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    d = pts - np.asarray(position, dtype=float)
    r = np.linalg.norm(d, axis=1)
    rhat = d / r[:, None]
    x = k * r
    p = beta * np.asarray(e_hat, dtype=complex)
    rp = rhat @ p
    a = 1 + 1j / x - 1 / x ** 2
    b = 1 + 3j / x - 3 / x ** 2
    return 1.5 * np.exp(1j * x)[:, None] / x[:, None] * (a[:, None] * p - b[:, None] * rhat * rp[:, None])


def scattered_field(points, positions, beta, k: float, e_hat, R_exc: float = 0.0,
                    chunk: int = DEFAULT_CHUNK, backend: str = "cpu") -> np.ndarray:
    """``sum_j beta_j Gt(r - r_j) . e_hat`` at ``points`` (M, 3) -> (M, 3).

    Dipoles closer than ``R_exc`` to an observation point are omitted (see the
    module docstring).  With ``R_exc = 0`` only the exactly coincident self term
    is dropped, which is what makes ``points = positions`` give the exciting
    field.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float)
    beta = np.asarray(beta, dtype=complex)
    e = np.asarray(e_hat, dtype=complex)
    M = pts.shape[0]
    if backend == "gpu":
        import torch
        dev = torch.device("cuda")
        posd = torch.as_tensor(pos, device=dev)
        betad = torch.as_tensor(beta, device=dev)
        ed = torch.as_tensor(e, device=dev)
        out = np.empty((M, 3), dtype=complex)
        for c0 in range(0, M, chunk):
            c1 = min(c0 + chunk, M)
            p = torch.as_tensor(pts[c0:c1], device=dev)
            dvec = p[:, None, :] - posd[None, :, :]
            V = dipole_field_vectors(dvec, k, ed, xp=torch)
            if R_exc > 0:
                r = torch.sqrt(torch.sum(dvec * dvec, -1))
                V = torch.where((r >= R_exc)[..., None], V, torch.zeros_like(V))
            out[c0:c1] = (torch.einsum("j,mjc->mc", betad, V)).cpu().numpy()
        return out
    out = np.empty((M, 3), dtype=complex)
    for c0 in range(0, M, chunk):
        c1 = min(c0 + chunk, M)
        dvec = pts[c0:c1, None, :] - pos[None, :, :]
        V = dipole_field_vectors(dvec, k, e)
        if R_exc > 0:
            r = np.linalg.norm(dvec, axis=-1)
            V[r < R_exc] = 0.0
        out[c0:c1] = np.einsum("j,mjc->mc", beta, V)
    return out


# --------------------------------------------------------------- products


@dataclass
class FieldSample:
    """A field evaluated at points: ``E / E0`` (M, 3), the incident part, and the
    mask of points that sit within ``mask_radius`` of an atom."""

    points: np.ndarray
    E: np.ndarray
    E_inc: np.ndarray
    masked: np.ndarray
    R_exc: float = 0.0

    @property
    def E_scat(self) -> np.ndarray:
        return self.E - self.E_inc

    @property
    def intensity(self) -> np.ndarray:
        """``|E|^2 / E0^2`` per point."""
        return np.sum(np.abs(self.E) ** 2, axis=-1)


def _mask_near_atoms(points, positions, radius):
    if radius <= 0:
        return np.zeros(points.shape[0], dtype=bool)
    d, _ = cKDTree(positions).query(points, k=1)
    return d < radius


def field_at(result, points, R_exc: float = 0.0, mask_radius: float = DEFAULT_MASK_RADIUS,
             chunk: int = DEFAULT_CHUNK, backend: str = "cpu",
             incident: Optional[IncidentField] = None) -> FieldSample:
    """The MICROSCOPIC field ``E(r) / E0`` at arbitrary points (M, 3).

    ``R_exc > 0`` omits dipoles within that distance of each point (for
    coherent-field averaging); ``mask_radius`` flags points near atoms where
    the exact field is dominated by one near-field singularity.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    inc = result.incident if incident is None else incident
    E_inc = inc.field(pts)
    Es = scattered_field(pts, result.config.positions, result.beta, result.op.k,
                         result.op.e_hat, R_exc=R_exc, chunk=chunk, backend=backend)
    return FieldSample(pts, E_inc + Es, E_inc, _mask_near_atoms(pts, result.config.positions,
                                                                 mask_radius), R_exc)


def grid_points(x, y, z):
    """Regular grid from three axes -> points (nx*ny*nz, 3), C order."""
    X, Y, Z = np.meshgrid(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z), indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def field_on_grid(result, x, y, z, **kw) -> FieldSample:
    """:func:`field_at` on a regular grid; ``E`` is returned with shape (nx, ny, nz, 3)."""
    x, y, z = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x, y, z))
    fs = field_at(result, grid_points(x, y, z), **kw)
    shape = (x.size, y.size, z.size)
    fs.E = fs.E.reshape(shape + (3,))
    fs.E_inc = fs.E_inc.reshape(shape + (3,))
    fs.masked = fs.masked.reshape(shape)
    fs.points = fs.points.reshape(shape + (3,))
    return fs


def exciting_field_at_atoms(result, chunk: int = DEFAULT_CHUNK, backend: str = "cpu") -> np.ndarray:
    """``E_exc(r_j) / E0`` (N, 3): the field each atom actually sees, its own excluded.

    Satisfies ``beta_j = alpha_tilde_j conj(e_hat) . E_exc(r_j)`` (S8).  Its
    intensity is the local saturation an atom experiences and
    ``Re(alpha) |E_exc|^2 / 2`` its light shift.
    """
    pos = result.config.positions
    E_inc = result.incident.field(pos)
    return E_inc + scattered_field(pos, pos, result.beta, result.op.k, result.op.e_hat,
                                   R_exc=0.0, chunk=chunk, backend=backend)


def self_consistency_residual(result, E_exc: Optional[np.ndarray] = None) -> float:
    """S8: ``max |beta_j - alpha_j conj(e) . E_exc_j| / max |beta|`` (must be ~1e-15)."""
    if E_exc is None:
        E_exc = exciting_field_at_atoms(result)
    alpha = result.op.polarizability_scalar(result.detunings)
    if result.rg is not None and result.rg.tracked:
        alpha = -0.5 / (result.detunings + 0.5j * result.rg.gamma)
    pred = alpha * (E_exc @ np.conj(result.op.e_hat))
    # the RG/'far'/'independent' variants use a kernel that is NOT the vacuum
    # dyadic, so their exciting field is defined through their own kernel only
    if result.variant != "full":
        return float("nan")
    return float(np.max(np.abs(result.beta - pred)) / np.max(np.abs(result.beta)))


def far_field_amplitude(nhat, positions, beta, k: float, e_hat) -> np.ndarray:
    """``F(n) = sum_j beta_j e^{-i k n.r_j} (e_hat - n (n.e_hat))`` (M, 3).

    The scattered field tends to ``E0 (3/2) F(n) e^{ikr} / (kr)``; the total
    scattered power is ``sigma0 I (3 / 8 pi) int |F|^2 dOmega = sigma0 I beta^dag Gamma beta``.
    """
    n = np.asarray(nhat, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float)
    e = np.asarray(e_hat, dtype=complex)
    phase = np.exp(-1j * k * (n @ pos.T))                       # (M, N)
    S = phase @ np.asarray(beta, dtype=complex)                  # (M,)
    ne = n @ e
    return S[:, None] * (e[None, :] - n * ne[:, None])


def far_field_power(result, n_u: int = 200, n_phi: int = 256):
    """S7: ``(3/8pi) int |F|^2 dOmega`` against ``beta^dag Gamma beta``.

    Returns ``(quadrature, quadratic_form, relative_difference)``.  The
    quadrature is computed with the analytic prefactor and NOT normalised to the
    total power, so an error in the kernel or the quadrature shows up here.
    ``n_u, n_phi`` are the convergence parameters.
    """
    from kamo.imaging.farfield import sphere_quadrature
    nhat, w = sphere_quadrature(n_u, n_phi)
    nhat = nhat.reshape(-1, 3)
    F = far_field_amplitude(nhat, result.config.positions, result.beta, result.op.k,
                            result.op.e_hat)
    quad = 3 / (8 * np.pi) * float(np.sum(w.ravel() * np.sum(np.abs(F) ** 2, axis=1)))
    qf = result.radiated_power
    return quad, qf, abs(quad - qf) / abs(qf)


def angular_distribution(result, nhat) -> np.ndarray:
    """``dP/dOmega`` in units of ``sigma0 I``: ``(3/8pi) |F(n)|^2``."""
    F = far_field_amplitude(nhat, result.config.positions, result.beta, result.op.k,
                            result.op.e_hat)
    return 3 / (8 * np.pi) * np.sum(np.abs(F) ** 2, axis=1)


# ---------------------------------------------------------- transmitted plane


@dataclass
class PlaneField:
    """The field on a transverse plane ``x = x_plane``."""

    x_plane: float
    y: np.ndarray
    z: np.ndarray
    E: np.ndarray            #: (ny, nz, 3), E / E0
    E_inc: np.ndarray        #: (ny, nz, 3)
    R_exc: float = 0.0

    @property
    def psi(self) -> np.ndarray:
        """``E_pol / E_inc,pol`` -- the polarization component of the incident
        field, normalised to the incident field: directly comparable to
        ``PropagationResult.psi_exit`` of the BPM."""
        pol = self.E_inc.reshape(-1, 3)
        pol = pol[np.argmax(np.linalg.norm(pol, axis=1))]
        pol = pol / np.linalg.norm(pol)
        E_p = self.E @ np.conj(pol)
        E_ip = self.E_inc @ np.conj(pol)
        return E_p / E_ip

    @property
    def transmission(self) -> np.ndarray:
        return np.abs(self.psi) ** 2


def transmitted_plane(result, x_plane: float, y, z, R_exc: float = 0.0,
                      chunk: int = DEFAULT_CHUNK, backend: str = "cpu") -> PlaneField:
    """Microscopic field on the plane ``x = x_plane`` over axes ``y, z``."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    fs = field_on_grid(result, [x_plane], y, z, R_exc=R_exc, mask_radius=0.0, chunk=chunk,
                       backend=backend)
    return PlaneField(float(x_plane), y, z, fs.E[0], fs.E_inc[0], R_exc)


# --------------------------------------------------------- coherent field


@dataclass
class CoherentField:
    """Configuration statistics of the field at fixed points.

    ``mean`` is ``<E>/E0`` (M, 3); ``intensity_coherent = |<E>|^2`` and
    ``intensity_mean = <|E|^2>``; their difference is the GRANULAR part, the
    fluctuating pair-potential landscape that no smooth propagation sees.
    ``sem`` is the standard error of ``mean`` per component (from the
    configuration scatter).
    """

    points: np.ndarray
    mean: np.ndarray
    intensity_mean: np.ndarray
    n_config: int
    sem: np.ndarray
    R_exc: float
    E_inc: np.ndarray

    @property
    def intensity_coherent(self) -> np.ndarray:
        return np.sum(np.abs(self.mean) ** 2, axis=-1)

    @property
    def intensity_granular(self) -> np.ndarray:
        return self.intensity_mean - self.intensity_coherent

    @property
    def psi(self) -> np.ndarray:
        """Coherent field projected on the incident polarization, over the incident field."""
        pol = self.E_inc.reshape(-1, 3)
        pol = pol[np.argmax(np.linalg.norm(pol, axis=1))]
        pol = pol / np.linalg.norm(pol)
        return (self.mean @ np.conj(pol)) / (self.E_inc @ np.conj(pol))

    def relative_sem(self) -> float:
        """Typical ``|sem| / |<E>|`` over the points: the speckle left in the map."""
        return float(np.median(np.linalg.norm(self.sem, axis=-1)
                               / np.maximum(np.linalg.norm(self.mean, axis=-1), 1e-300)))


def coherent_field(results: Sequence, points, R_exc: float = 0.0, chunk: int = DEFAULT_CHUNK,
                   backend: str = "cpu", incident: Optional[IncidentField] = None,
                   progress=None) -> CoherentField:
    """Configuration average ``<E(r)>`` over a list of :class:`SolveResult`.

    Every result must share the incident field.  Points are held fixed; the
    atoms move from configuration to configuration.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(results)
    if n == 0:
        raise ValueError("no configurations")
    inc = results[0].incident if incident is None else incident
    E_inc = inc.field(pts)
    s1 = np.zeros((pts.shape[0], 3), dtype=complex)
    s2 = np.zeros((pts.shape[0], 3))
    for i, res in enumerate(results):
        if progress is not None:
            progress(i, n)
        E = E_inc + scattered_field(pts, res.config.positions, res.beta, res.op.k, res.op.e_hat,
                                    R_exc=R_exc, chunk=chunk, backend=backend)
        s1 += E
        s2 += np.abs(E) ** 2
    mean = s1 / n
    var = np.maximum(s2 / n - np.abs(mean) ** 2, 0.0)
    sem = np.sqrt(var / max(n - 1, 1))
    return CoherentField(pts, mean, np.sum(s2, axis=-1) / n, n, sem, float(R_exc), E_inc)


def coherent_plane(results, x_plane: float, y, z, R_exc: float = 0.0, **kw):
    """:func:`coherent_field` on a transverse plane; returns ``(CoherentField, psi (ny, nz))``."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    cf = coherent_field(results, grid_points([x_plane], y, z), R_exc=R_exc, **kw)
    return cf, cf.psi.reshape(y.size, z.size)


# ------------------------------------------------------------- landscapes


def light_shift_landscape(intensity_rel, op: OperatingPoint, delta: float, s0: float):
    """Ground-state light shift (Hz) from a relative intensity map ``|E|^2/E0^2``.

    Uses the same closed form as :mod:`kamo.imaging` (``TwoLevelResponse.light_shift_Hz``)
    with the LOCAL saturation ``s0 |E|^2 / E0^2`` -- so the coherent and the
    granular intensity maps of :class:`CoherentField` give the smooth potential
    the whole cloud feels and the fluctuating pair potential respectively.
    """
    return op.response().light_shift_Hz(2 * float(delta), s0 * np.asarray(intensity_rel))


def coarse_grain(E_grid: np.ndarray, axes, cell: float, positions=None):
    """Box-average a gridded field over cubes of side ``cell``.

    Provided for completeness and NOT the recommended route to a macroscopic
    field: it warns with the actual number of atoms per cell, which at the
    operating point is 2-8 for any cell small enough to resolve the cloud.  Use
    :func:`coherent_field` instead.
    """
    from scipy.ndimage import uniform_filter
    x, y, z = (np.asarray(a, dtype=float) for a in axes)
    d = np.array([np.diff(a).mean() if a.size > 1 else np.inf for a in (x, y, z)])
    size = [max(int(round(cell / di)), 1) if np.isfinite(di) else 1 for di in d]
    if positions is not None:
        n_peak_cell = None
        pos = np.asarray(positions)
        tree = cKDTree(pos)
        centre = pos.mean(axis=0)
        n_peak_cell = len(tree.query_ball_point(centre, 0.62 * cell))   # sphere of equal volume
        warnings.warn(f"coarse_grain: cell = {cell * 1e9:.0f} nm holds ~{n_peak_cell} atoms at "
                      f"the cloud centre; the granularity noise is ~{1 / np.sqrt(max(n_peak_cell, 1)):.0%}. "
                      "Prefer configuration averaging (coherent_field).", CoarseGrainWarning,
                      stacklevel=2)
    out = np.empty_like(E_grid)
    for c in range(E_grid.shape[-1]):
        out[..., c] = (uniform_filter(E_grid[..., c].real, size=size, mode="nearest")
                       + 1j * uniform_filter(E_grid[..., c].imag, size=size, mode="nearest"))
    return out
