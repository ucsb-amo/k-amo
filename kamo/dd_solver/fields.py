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

2. **Coherent field** ``<E(r)>`` (:func:`coherent_field`), the CONFIGURATION
   average at a fixed point.  It is smooth and finite everywhere and carries the
   lensing.  OUTSIDE the cloud -- which is where the A/B against
   :mod:`kamo.imaging` is done -- it is exactly what a susceptibility-based
   propagation computes.  INSIDE the cloud it is not; see the two corrections
   below.

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
   ``R_exc`` omits dipoles within ``R_exc`` of the observation point, which
   removes most of the variance.  ``R_exc = 150 nm ~`` the interparticle spacing
   gives clean maps at 40 configurations (test T19).  Default 0: the exact
   microscopic field.

   **Two corrections that apply INSIDE the cloud** (established 2026-09-17; both
   are O(10 %) at the operating density and neither affects the exit-plane A/B,
   where ``n ~ 0``):

   a. ``R_exc`` DOES bias the mean.  Only the ``1/r^3`` term angle-averages to
      zero; the angular average of ``Gt . e`` is ``(e^{ix}/x) e``, which does
      not.  For locally uniform density the excluded sphere removes
      ``f(X) P / (3 eps0)`` with ``f(X) = 2[(1 - iX) e^{iX} - 1]`` and
      ``X = k R_exc``.  At ``R_exc = 150 nm``, ``X = 1.23`` and
      ``f = 0.99 + 1.06i``: the removed radiative part is as large as the
      Lorentz term itself.  Measured at the cloud centre, the relative intensity
      moves by about 7 % between ``R_exc = 50`` and ``150 nm`` and about 12 % out
      to 250 nm, with the sign set by the detuning.  Quote in-cloud numbers with
      the ``R_exc`` that produced them.

   b. The ``R_exc -> 0`` limit is the LORENTZ LOCAL field, not the Maxwell field.
      The dipole sum omits the contact term ``-p delta(r) / 3 eps0``, so the
      configuration average at a vacant point is
      ``E_Maxwell + P / (3 eps0)``.  At the cloud centre ``chi_yy / 3 = 0.053``,
      i.e. about +11 % in intensity for the red-detuned species and -10 % for
      the blue one.  A susceptibility propagation reports ``E_Maxwell``.  This is
      the same Lorentz-Lorenz term :mod:`kamo.imaging` exposes as
      ``TwoLevelResponse(local_field=True)``, and it is the reason a mean-field
      index comparison must be made outside the sample or corrected for
      explicitly.

3. **Exciting field at an atom** (:func:`exciting_field_at_atoms`): the
   microscopic field minus that atom's own field -- the one field that is finite
   and unambiguous at an atomic position, and the one satisfying
   ``beta_j = alpha_j conj(e_hat) . E_exc(r_j)`` (S8 / T5).

Memory: a ``200 x 200`` grid with ``N = 1000`` atoms is ``4e4 x 1e3 x 3``
complex numbers per component if built at once.  Everything here is chunked over
observation points (``chunk`` points at a time).  The working set is about
``6.5 x chunk * N * 3 * 16`` bytes: the separation vectors, their norms, the unit
vectors and the complex radial weights are all live at once, so a 1024-point
chunk at N = 1000 peaks near 160 MB rather than the 25 MB the naive count gives.
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

    ``beta`` may be ``(N,)`` or ``(V, N)``.  In the second form the V dipole
    sets share ONE geometry pass and the result is ``(V, M, 3)`` -- the way to
    evaluate several kernel variants of the same configuration on the same
    plane, which is what the A/B sweep does.  This is where the CPU time of a
    density sweep goes: 29241 window points x 500 atoms is 5.7 s per
    configuration per variant, against 5 s for ALL the solves of a density point
    (measured 2026-09-17).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float)
    beta = np.asarray(beta, dtype=complex)
    stacked = beta.ndim > 1
    betas = beta if stacked else beta[None, :]
    e = np.asarray(e_hat, dtype=complex)
    M = pts.shape[0]
    if backend != "gpu":
        # Lean numpy path: scalar radial weights instead of the (M, N, 3) complex
        # unit-vector and term arrays, which carried 6.5x the documented memory.
        #   Gt . e = A e - (B / r^2) d (d . e),
        #   A = (3/2)(e^{ix}/x)(1 + c),  B = (3/2)(e^{ix}/x)(1 + 3c),  c = i/x - 1/x^2
        out = np.zeros((betas.shape[0], M, 3), dtype=complex)
        for c0 in range(0, M, chunk):
            c1 = min(c0 + chunk, M)
            d = pts[c0:c1, None, :] - pos[None, :, :]
            r2 = np.einsum("mjc,mjc->mj", d, d)
            live = r2 > 0.0
            if R_exc > 0:
                live &= r2 >= R_exc * R_exc
            r = np.sqrt(np.where(live, r2, 1.0))
            x = k * r
            pref = 1.5 * np.exp(1j * x) / x
            cc = 1j / x - 1.0 / (x * x)
            A = np.where(live, pref * (1.0 + cc), 0.0)
            W = np.where(live, pref * (1.0 + 3.0 * cc) / np.where(live, r2, 1.0), 0.0)
            de = d @ e                                        # (m, N) complex
            for v in range(betas.shape[0]):
                bv = betas[v]
                out[v, c0:c1] = (np.einsum("mj,j->m", A, bv)[:, None] * e
                                 - np.einsum("mj,mjc->mc", W * de * bv[None, :], d))
        return out if stacked else out[0]
    if backend == "gpu":
        import torch
        dev = torch.device("cuda")
        posd = torch.as_tensor(pos, device=dev)
        betad = torch.as_tensor(betas, device=dev)
        ed = torch.as_tensor(e, device=dev)
        out = np.empty((betas.shape[0], M, 3), dtype=complex)
        for c0 in range(0, M, chunk):
            c1 = min(c0 + chunk, M)
            p = torch.as_tensor(pts[c0:c1], device=dev)
            dvec = p[:, None, :] - posd[None, :, :]
            V = dipole_field_vectors(dvec, k, ed, xp=torch)
            if R_exc > 0:
                r = torch.sqrt(torch.sum(dvec * dvec, -1))
                V = torch.where((r >= R_exc)[..., None], V, torch.zeros_like(V))
            out[:, c0:c1] = (torch.einsum("vj,mjc->vmc", betad, V)).cpu().numpy()
        return out if stacked else out[0]


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

    Satisfies ``beta_j = alpha_tilde_j conj(e_hat) . E_exc(r_j)`` (S8).

    .. warning::
       ``|E_exc|^2`` is the TOTAL vector intensity and is **not** what saturates
       or shifts the atom (corrected 2026-09-17).  A closed sigma- line responds
       only to the projection on its own dipole,
       ``|conj(e_hat) . E_exc|^2 = |beta_j / alpha_j|^2`` -- use
       :func:`driving_intensity`.  The difference is large, not cosmetic: in the
       near field of a neighbour ``|E|^2`` goes as ``1 + 3c`` while the driving
       part goes as ``(1 - 3c)^2``, so most of a near-field spike sits in the
       ``e_+`` and ``z`` components, which this atom can reach only through the
       pi and sigma+ lines 152-333 linewidths away.  Over 30 configurations the
       total intensity has mean 5.6 and 2.0 % of atoms above 10x incident, while
       the driving intensity has mean 1.8 and 0.75 % above 10x.
    """
    pos = result.config.positions
    E_inc = result.incident.field(pos)
    return E_inc + scattered_field(pos, pos, result.beta, result.op.k, result.op.e_hat,
                                   R_exc=0.0, chunk=chunk, backend=backend)


def driving_intensity(E, e_hat, incident=None) -> np.ndarray:
    """``|conj(e_hat) . E|^2`` -- the part of a field that drives the sigma- line.

    This is the quantity that sets the excited population, the saturation and
    the light shift of a closed two-level line, and the one that satisfies
    ``|beta_j|^2 = |alpha_j|^2 x driving_intensity``.

    Parameters
    ----------
    E : (..., 3) complex
        Field in units of ``E0``.
    e_hat : (3,) complex
        Driven dipole unit vector (``op.e_hat``).
    incident : IncidentField, optional
        When given, the result is normalised to the driving intensity of the
        INCIDENT beam (``|conj(e_hat) . pol|^2``, 1/2 for the lab's y-polarized
        Voigt probe), so 1.0 means "as strongly driven as an isolated atom in
        the probe".  Without it the result is in units of ``|E0|^2``.
    """
    val = np.abs(np.asarray(E) @ np.conj(np.asarray(e_hat))) ** 2
    if incident is None:
        return val
    return val / float(incident.projection(e_hat))


def self_consistency_residual(result, E_exc: Optional[np.ndarray] = None) -> float:
    """S8: ``max |beta_j - alpha_j conj(e) . E_exc_j| / max |beta|`` (must be ~1e-15)."""
    if E_exc is None:
        E_exc = exciting_field_at_atoms(result)
    f = result.op.strengths(result.config.spins)
    alpha = result.op.polarizability_scalar(result.detunings, f)
    if result.rg is not None and result.rg.tracked:
        alpha = -0.5 * f / (result.detunings + 0.5j * result.rg.gamma)
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


def far_field_power(result, n_u: Optional[int] = None, n_phi: Optional[int] = None,
                    chunk: int = 4096):
    """S7: ``(3/8pi) int |F|^2 dOmega`` against ``beta^dag Gamma beta``.

    Returns ``(quadrature, quadratic_form, relative_difference)``.  The
    quadrature is computed with the analytic prefactor and NOT normalised to the
    total power, so an error in the kernel or the quadrature shows up here.

    ``n_u, n_phi`` default to the band limit of the sample,
    ``n_phi >= 2 k R_max + 16`` and ``n_u = n_phi``, which is where the spectral
    convergence has already reached rounding error; the previous fixed
    ``200 x 256`` spent about 4x the time and, because
    :func:`far_field_amplitude` built the whole ``(M_directions, N)`` phase
    matrix at once, 821 MB at N = 500 and ~5.7 GB at N = 3500.  The sum is now
    chunked over directions (2026-09-17).
    """
    from kamo.imaging.farfield import sphere_quadrature
    pos = result.config.positions
    if n_phi is None:
        R_max = float(np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))
        n_phi = int(2 ** np.ceil(np.log2(max(2 * result.op.k * R_max + 16, 32))))
    if n_u is None:
        n_u = n_phi
    nhat, w = sphere_quadrature(n_u, n_phi)
    nhat = nhat.reshape(-1, 3)
    w = w.ravel()
    total = 0.0
    for i0 in range(0, nhat.shape[0], chunk):
        sl = slice(i0, min(i0 + chunk, nhat.shape[0]))
        F = far_field_amplitude(nhat[sl], pos, result.beta, result.op.k, result.op.e_hat)
        total += float(np.sum(w[sl] * np.sum(np.abs(F) ** 2, axis=1)))
    quad = 3 / (8 * np.pi) * total
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
    _wavelength: Optional[float] = None

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
        """Typical ``|sem| / |<E>|`` over the points: the speckle left in the map.

        Normalised by the TOTAL field, which is ~1 on a transmission plane.  To
        compare with a difference between two fields (an rms residual against a
        propagation code), use :meth:`sem_versus` instead -- the two differ by
        ``1 / rms|psi - 1|``, a factor of 9 to 900 in the A/B sweep.
        """
        return float(np.median(np.linalg.norm(self.sem, axis=-1)
                               / np.maximum(np.linalg.norm(self.mean, axis=-1), 1e-300)))

    def sem_versus(self, reference) -> float:
        """Noise floor of an rms comparison against ``reference`` (a ``psi`` array).

        ``sqrt(mean |sem_psi|^2) / rms|reference - 1|``, i.e. the per-pixel
        configuration noise of :attr:`psi` expressed in the SAME units as an rms
        difference normalised by ``rms|psi_ref - 1|``.  An rms difference within
        about 2x of this number is noise, not physics (added 2026-09-17: the
        A/B table previously printed :meth:`relative_sem` beside such an rms,
        understating the floor by up to 900x and making the balanced-cloud
        residuals look like a real full-versus-far difference).
        """
        pol = self.E_inc.reshape(-1, 3)
        pol = pol[np.argmax(np.linalg.norm(pol, axis=1))]
        pol = pol / np.linalg.norm(pol)
        sem_psi = np.abs(self.sem @ np.conj(pol)) / np.abs(self.E_inc @ np.conj(pol))
        ref = np.asarray(reference)
        denom = float(np.sqrt(np.mean(np.abs(ref - 1.0) ** 2)))
        return float(np.sqrt(np.mean(sem_psi.reshape(-1) ** 2)) / max(denom, 1e-300))

    def lorentz_correction(self, density, beta_mean: complex, e_hat) -> np.ndarray:
        """``<E>`` minus the Maxwell field, for locally uniform density (M, 3).

        Returns ``[1 - f(k R_exc)] P / (3 eps0 E0)`` along ``e_hat``, with
        ``f(X) = 2[(1 - iX) e^{iX} - 1]`` the part of the excluded sphere's
        contribution that is radiative rather than static, and
        ``P / (3 eps0 E0) = (4 pi / k^3) n <beta> / 3 x (3/2) ... ``

        Concretely, a uniform density ``n`` of dipoles of mean amplitude
        ``<beta>`` gives ``P/(3 eps0 E0) = 2 pi n <beta> / k^3`` in these units.
        Subtract this from :attr:`mean` to obtain the Maxwell (macroscopic) field
        that a susceptibility propagation reports.  Outside the cloud
        ``density = 0`` and the correction vanishes, which is why the exit-plane
        A/B needs none.

        Parameters
        ----------
        density : float or (M,) array
            Local number density (m^-3) at the observation points.
        beta_mean : complex
            Configuration-mean dipole amplitude of the atoms near those points.
        e_hat : (3,) complex
            Driven dipole direction.
        """
        if self._wavelength is None:
            raise ValueError("wavelength unknown; use a CoherentField built by "
                             "coherent_field(), which records it")
        k = 2 * np.pi / self._wavelength
        X = k * self.R_exc
        f = 2.0 * ((1.0 - 1j * X) * np.exp(1j * X) - 1.0)
        P_over_3eps0 = 2 * np.pi * np.asarray(density) * beta_mean / k ** 3
        return np.multiply.outer((1.0 - f) * P_over_3eps0, np.asarray(e_hat))


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
    cf = CoherentField(pts, mean, np.sum(s2, axis=-1) / n, n, sem, float(R_exc), E_inc)
    cf._wavelength = 2 * np.pi / results[0].op.k
    return cf


def coherent_plane(results, x_plane: float, y, z, R_exc: float = 0.0, **kw):
    """:func:`coherent_field` on a transverse plane; returns ``(CoherentField, psi (ny, nz))``."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    cf = coherent_field(results, grid_points([x_plane], y, z), R_exc=R_exc, **kw)
    return cf, cf.psi.reshape(y.size, z.size)


def coherent_field_multi(results_by_variant, points, R_exc: float = 0.0,
                         chunk: int = DEFAULT_CHUNK, backend: str = "cpu",
                         incident: Optional[IncidentField] = None, progress=None):
    """:func:`coherent_field` for several variants at once, sharing the geometry.

    ``results_by_variant`` maps a variant name to its list of
    :class:`~kamo.dd_solver.solver.SolveResult`; all lists must be over the SAME
    configurations in the same order, which is what
    :func:`kamo.dd_solver.ensemble.run_ensemble` returns.  The separations to the
    observation points depend only on the positions, so evaluating V variants
    costs barely more than one (measured 1.13x for two, against 2x before).
    """
    variants = list(results_by_variant)
    if not variants:
        raise ValueError("no variants")
    lists = [results_by_variant[v] for v in variants]
    n = len(lists[0])
    if n == 0:
        raise ValueError("no configurations")
    if any(len(L) != n for L in lists):
        raise ValueError("every variant must cover the same configurations")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    inc = lists[0][0].incident if incident is None else incident
    E_inc = inc.field(pts)
    V = len(variants)
    s1 = np.zeros((V, pts.shape[0], 3), dtype=complex)
    s2 = np.zeros((V, pts.shape[0], 3))
    for i in range(n):
        if progress is not None:
            progress(i, n)
        res0 = lists[0][i]
        if any(L[i].config is not res0.config
               and not np.array_equal(L[i].config.positions, res0.config.positions)
               for L in lists):
            raise ValueError("the variants are not on the same configurations")
        betas = np.stack([L[i].beta for L in lists])
        E = E_inc[None, :, :] + scattered_field(pts, res0.config.positions, betas, res0.op.k,
                                                res0.op.e_hat, R_exc=R_exc, chunk=chunk,
                                                backend=backend)
        s1 += E
        s2 += np.abs(E) ** 2
    out = {}
    for vi, name in enumerate(variants):
        mean = s1[vi] / n
        var = np.maximum(s2[vi] / n - np.abs(mean) ** 2, 0.0)
        cf = CoherentField(pts, mean, np.sum(s2[vi], axis=-1) / n, n,
                           np.sqrt(var / max(n - 1, 1)), float(R_exc), E_inc)
        cf._wavelength = 2 * np.pi / lists[vi][0].op.k
        out[name] = cf
    return out


def coherent_plane_multi(results_by_variant, x_plane: float, y, z, R_exc: float = 0.0, **kw):
    """:func:`coherent_field_multi` on a transverse plane.

    Returns ``{variant: (CoherentField, psi (ny, nz))}``.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    cfs = coherent_field_multi(results_by_variant, grid_points([x_plane], y, z),
                               R_exc=R_exc, **kw)
    return {v: (cf, cf.psi.reshape(y.size, z.size)) for v, cf in cfs.items()}


# ------------------------------------------------------------- landscapes


def light_shift_landscape(driving, op: OperatingPoint, delta: float, s0: float,
                          strength: Optional[float] = None):
    """Ground-state light shift (Hz) from a DRIVING-intensity map.

    ``U/h = delta * Gamma_Hz * rho_ee`` with ``rho_ee = (s0/2) |alpha|^2 driving``
    -- the low-saturation limit, exact to O(s) and good to better than 1 % at the
    lab's ``s0 = 0.21-0.34`` and ``|delta| = 9``.

    Parameters
    ----------
    driving : array
        ``|conj(e_hat) . E|^2 / |E0|^2``, what :func:`driving_intensity` returns
        WITHOUT ``incident=``.  An isolated atom in the lab's y-polarized Voigt
        probe has 0.5, not 1, because only half the field drives the sigma- line.
    op, delta : OperatingPoint, float
        Operating point and the detuning of the species (Gamma units).
    s0 : float
        Saturation parameter of the incident probe, ``I / I_sat``, with
        ``I_sat`` built on the full cycling ``sigma0`` and NO polarization
        projection -- the lab convention, and :mod:`kamo.imaging`'s.
    strength : float, optional
        Oscillator strength of the line; defaults to the operating point's value
        for this detuning.  The shift scales as ``f^2``.

    .. note::
       Corrected 2026-09-17.  This used to take the TOTAL relative intensity
       ``|E|^2/E0^2`` and feed ``s0 |E|^2`` to the ideal two-level closed form,
       which double counts the polarization: the shift came out exactly 2x too
       large for an isolated atom (24.65 kHz against the correct 11.81 kHz at
       ``s0 = 0.30``, ``|delta| = 9.139``) and up to ~8x too large on granular
       near-field maps, where most of ``|E|^2`` is in the ``e_+`` and ``z``
       components that this line cannot absorb.  The ideal closed form also
       omits the ``f^2 = 0.955`` of the real line.
    """
    if strength is None:
        strength = op.strength_up if float(delta) > 0 else op.strength_dn
    alpha = op.polarizability_scalar(float(delta), float(strength))
    rho_ee = 0.5 * float(s0) * np.abs(alpha) ** 2 * np.asarray(driving)
    return float(delta) * op.linewidth_Hz * rho_ee


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
    # Odd windows only: uniform_filter is not centred for an even size, which
    # displaces the "box average" by half a grid step.  A cell below 1.5 grid
    # steps rounds to size 1 -- the identity -- which used to happen silently
    # (including in this package's own test case); warn instead (2026-09-17).
    size = []
    for di in d:
        if not np.isfinite(di):
            size.append(1)
            continue
        s = int(round(cell / di))
        if s % 2 == 0:
            s += 1
        size.append(max(s, 1))
    if any(s == 1 and np.isfinite(di) and di > 0 for s, di in zip(size, d)):
        warnings.warn(f"coarse_grain: cell = {cell * 1e9:.0f} nm is below 1.5 grid steps on at "
                      f"least one axis ({d[np.isfinite(d)] * 1e9}); that axis is NOT averaged. "
                      "The applied window is " + str([int(s) for s in size]) + " samples.",
                      CoarseGrainWarning, stacklevel=2)
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
