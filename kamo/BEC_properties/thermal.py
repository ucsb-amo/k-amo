"""Ideal Bose gas in a harmonic trap: Bose functions, the finite-N series, and the
closed-form bimodal cloud :class:`IdealHarmonicBoseGas`.

This is the T > 0 analogue of :mod:`kamo.BEC_properties.variational`: no grid, no
solver, closed forms only.  It is the quick "is 50 nK a lot here?" estimate, the
reference that :meth:`kamo.trap.cloud.TrapCloud.harmonic_reference` returns at
``T > 0``, the bimodal model a time-of-flight fit assumes, and the analytic oracle
the finite-temperature solver in :mod:`kamo.trap.finite_temperature` is tested
against.

Quick start
-----------
>>> from kamo.BEC_properties.thermal import IdealHarmonicBoseGas
>>> import numpy as np
>>> gas = IdealHarmonicBoseGas(N=500, omega=2 * np.pi * np.array([79, 993, 979]), T_K=50e-9)
>>> gas.T_c_nK, gas.condensate_fraction, gas.sigma_thermal * 1e6
>>> gas.exact_condensate_fraction()          # the finite-N sum, no semiclassics
>>> print(gas.summary())

Model
-----
Semiclassical (local-density) thermal cloud with the exact oscillator ground state
as the condensate::

    k T_c = hbar wbar (N / zeta(3))^(1/3)                   zeta(3) = 1.2020569
    N_0 / N = 1 - (T / T_c)^3                                (T < T_c, z = 1)
    n_th(r) = g_{3/2}(z e^{-V(r)/kT}) / lambda_dB^3,        lambda_dB = h / sqrt(2 pi m kT)
    n_0(r)  = N_0 prod_k sqrt(m w_k / (pi hbar)) exp(-m w_k x_k^2 / hbar)

with ``V(r) = sum_k m w_k^2 x_k^2 / 2`` measured from the trap bottom, ``z = 1``
below ``T_c`` and ``z < 1`` above it from ``g_3(z) (kT / hbar wbar)^3 = N``.  The
thermal rms widths carry the Bose enhancement, ``<x_k^2> = (kT / m w_k^2) g_4(z) / g_3(z)``,
and reduce to the Boltzmann ``sqrt(kT / m w_k^2)`` as ``T >> T_c``.

Two corrections are *reported*, never folded into the density (a reference that
corrects itself is no longer a reference): the finite-size shift
``dT_c / T_c = -(zeta(2) / (2 zeta(3)^(2/3))) (w_mean / wbar) N^(-1/3) = -0.7275 (w_mean/wbar) N^(-1/3)``
and the mean-field shift ``-1.33 (a / a_ho) N^(1/6)`` (Dalfovo, Giorgini, Pitaevskii
and Stringari, Rev. Mod. Phys. 71, 463 (1999), Eqs. 15 and 16).

The finite-size formula is itself optimistic at small N: for the K-team tweezer
(993, 979, 79 Hz; N = 500) it gives ``T_c = 129.6 nK`` where the exact finite-N sum
gives ``124.1 nK`` against ``T_c0 = 152.1 nK``.  That exact sum is
:func:`ideal_harmonic_condensate`::

    N = sum_{j >= 1} e^{j mu / kT} prod_k [1 - e^{-j hbar w_k / kT}]^{-1},   mu <= 0 from E_0

(the sum over all product states, ground state included), and it is what
:meth:`IdealHarmonicBoseGas.exact_condensate_fraction` returns.  It carries no
semiclassical approximation and is the test oracle for the hybrid solver in
:mod:`kamo.trap.finite_temperature` on a ``HarmonicTrap``.

Units: SI in and out (``omega`` in rad/s, ``T_K`` in kelvin, ``a_scattering`` in
metres); ``_nK`` views where people read them that way.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import spence

import kamo.constants as kc

ZETA_2 = 1.6449340668482264      # zeta(2) = pi^2 / 6
ZETA_3 = 1.2020569031595943      # zeta(3), Apery's constant
ZETA_3_2 = 2.6123753486854883    # zeta(3/2) = g_{3/2}(1)
ZETA_4 = 1.0823232337111382      # zeta(4) = pi^4 / 90
FINITE_SIZE_COEFF = ZETA_2 / (2.0 * ZETA_3 ** (2.0 / 3.0))   # 0.7275: Dalfovo RMP 71, 463 (1999) Eq. 15
MEAN_FIELD_COEFF = 1.33                                     # Dalfovo RMP 71, 463 (1999) Eq. 16

# Robinson expansion Li_s(e^-t) = Gamma(1 - s) t^(s-1) + sum_k zeta(s - k) (-t)^k / k!,
# radius of convergence 2 pi; used for t = -ln z <= 1.  zeta(s - k) for k = 0..18,
# generated once with mpmath at 30 digits (2026-09-13).  The list for s = 1/2 is the
# s = 3/2 list shifted by one.
_ZETA_S_MINUS_K_3_2 = np.array([
    2.6123753486854883433, -1.4603545088095868129, -0.20788622497735456602,
    -0.02548520188983303595, 0.0085169287778503305424, 0.0044410113354794319585,
    -0.0030916692472158338448, -0.002671458019899224599, 0.0027467679395368687584,
    0.0032690395726002200217, -0.0044160328730048898084, -0.0066721722964666407568,
    0.011146122473942814136, 0.020396978715942792056, -0.04057496748119457841,
    -0.087175255906217251469, 0.20117404938422688243, 0.49627121991205760787,
    -1.3032292507051139539])
_ZETA_S_MINUS_K_1_2 = np.array([
    -1.4603545088095868129, -0.20788622497735456602, -0.02548520188983303595,
    0.0085169287778503305424, 0.0044410113354794319585, -0.0030916692472158338448,
    -0.002671458019899224599, 0.0027467679395368687584, 0.0032690395726002200217,
    -0.0044160328730048898084, -0.0066721722964666407568, 0.011146122473942814136,
    0.020396978715942792056, -0.04057496748119457841, -0.087175255906217251469,
    0.20117404938422688243, 0.49627121991205760787, -1.3032292507051139539,
    -3.6297592997745741279])
_ZETA_S_MINUS_K_5_2 = np.concatenate([[1.3414872572509171798], _ZETA_S_MINUS_K_3_2[:-1]])
_FACT = np.cumprod(np.concatenate([[1.0], np.arange(1.0, 19.0)]))     # k! for k = 0..18
_ROBINSON = {0.5: (_ZETA_S_MINUS_K_1_2, np.sqrt(np.pi)),          # Gamma(1/2)
             1.5: (_ZETA_S_MINUS_K_3_2, -2.0 * np.sqrt(np.pi)),   # Gamma(-1/2)
             2.5: (_ZETA_S_MINUS_K_5_2, 4.0 / 3.0 * np.sqrt(np.pi))}   # Gamma(-3/2)
_SERIES_TERMS = 40           # at z = 1/e the 40th term is e^-40 ~ 4e-18
_INTEGER_TERMS = 20_000      # integer orders: direct series plus an analytic tail


def bose_g(z, order: float = 1.5):
    """Bose (polylogarithm) function ``g_s(z) = sum_{k>=1} z^k / k^s`` for ``0 <= z <= 1``.

    ``order`` in {1/2, 3/2, 5/2}: a two-branch vectorized evaluation, the direct
    series below ``z = 1/e`` and the 19-term Robinson expansion above, accurate to
    ~1e-15; ``g_{1/2}(1) = inf``.  ``order`` 2 is ``scipy.special.spence``; orders
    3 and 4 are the direct series with an analytic tail (~1e-12), meant for
    scalars and small arrays (the fugacity of a harmonic gas), not for grids.
    ``g_s(1) = zeta(s)``.
    """
    z = np.asarray(z, dtype=float)
    if np.any(z < 0) or np.any(z > 1.0 + 1e-12):
        raise ValueError("bose_g needs 0 <= z <= 1")
    z = np.clip(z, 0.0, 1.0)
    s = float(order)
    if s == 2.0:
        return spence(1.0 - z)
    if s in _ROBINSON:
        coeff, gamma_1ms = _ROBINSON[s]
        out = np.zeros_like(z)
        big = z > np.exp(-1.0)
        small = ~big & (z > 0)
        if np.any(small):
            zs = z[small]
            acc, zp = np.zeros_like(zs), np.ones_like(zs)
            for j in range(1, _SERIES_TERMS + 1):
                zp = zp * zs
                acc += zp / j ** s
            out[small] = acc
        if np.any(big):
            with np.errstate(divide="ignore"):
                t = -np.log(z[big])
            acc, tk = np.zeros_like(t), np.ones_like(t)
            for k in range(coeff.size):
                acc += coeff[k] * tk / _FACT[k]
                tk = tk * (-t)
            with np.errstate(divide="ignore"):
                lead = gamma_1ms * t ** (s - 1.0)
            out[big] = lead + acc
        return out
    if s in (3.0, 4.0):
        j = np.arange(1, _INTEGER_TERMS + 1, dtype=float)
        zz = z.reshape(-1)
        acc = np.array([float(np.sum(zi ** j / j ** s)) for zi in zz])
        J = _INTEGER_TERMS + 0.5
        tail = zz ** (_INTEGER_TERMS + 1) / ((s - 1.0) * J ** (s - 1.0))
        exact = {3.0: ZETA_3, 4.0: ZETA_4}[s]
        at_one = zz >= 1.0
        acc = np.where(at_one, exact, acc + tail)
        return acc.reshape(z.shape)
    raise ValueError("order must be one of 1/2, 3/2, 2, 5/2, 3, 4")


# ---------------------------------------------------------------- the finite-N series

def _harmonic_series(x: float, a: np.ndarray, thermal_only: bool) -> float:
    """``sum_j e^{-j x} [prod_k (1 - e^{-j a_k})^{-1} - (1 if thermal_only)]``,
    ``x = -mu / kT >= 0``, ``a_k = hbar w_k / kT``."""
    j_max = int(np.ceil(40.0 / float(np.min(a)))) + 1
    if not thermal_only and x <= 0:
        return float("inf")
    j = np.arange(1, j_max + 1, dtype=float)
    prod = np.prod(1.0 / (1.0 - np.exp(-np.outer(j, a))), axis=1)
    if thermal_only:
        prod = prod - 1.0
    terms = np.exp(-j * x) * prod
    total = float(np.sum(terms))
    if not thermal_only and x > 0:                       # geometric tail of the condensate part
        total += np.exp(-(j_max + 1) * x) / (1.0 - np.exp(-x))
    return total


def ideal_harmonic_atom_number(T_K: float, mu_offset_J: float, omega, *,
                               thermal_only: bool = False) -> float:
    """Exact finite-N ideal-gas atom number in a harmonic trap at ``T_K`` and chemical
    potential ``mu_offset_J <= 0`` measured from the ground-state energy
    ``E_0 = sum hbar w_k / 2``: ``sum_j e^{j mu/kT} prod_k (1 - e^{-j hbar w_k/kT})^{-1}``.
    With ``thermal_only`` the ground state is left out.  ``omega`` in rad/s; the
    result does not depend on the mass."""
    kT = kc.kB * float(T_K)
    a = kc.hbar * np.asarray(omega, dtype=float) / kT
    return _harmonic_series(-float(mu_offset_J) / kT, a, thermal_only)


def ideal_harmonic_condensate(N: float, T_K: float, omega) -> tuple:
    """``(N_0, mu_offset_J)`` of ``N`` ideal bosons in a harmonic trap at ``T_K``: the
    exact finite-N sum, ``mu`` from the ground state (always ``< 0``).  No
    semiclassical approximation and no sharp transition."""
    N, kT = float(N), kc.kB * float(T_K)
    a = kc.hbar * np.asarray(omega, dtype=float) / kT

    def f(logx):
        x = np.exp(logx)
        return 1.0 / np.expm1(x) + _harmonic_series(x, a, True) - N

    lo, hi = np.log(1e-3 / N), np.log(60.0)
    for _ in range(60):                                  # widen until bracketed
        if f(lo) > 0 > f(hi):
            break
        lo, hi = lo - 2.0, hi + 2.0
    x = np.exp(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))
    return 1.0 / np.expm1(x), -x * kT


def harmonic_thermal_density(u1, u2, u3, T_K: float, mu_offset_J: float, omega,
                             mass: Optional[float] = None, j_max: Optional[int] = None):
    """Exact ideal-gas thermal density (1/m^3) in a harmonic trap at ``T_K`` and
    ``mu_offset_J <= 0`` from ``E_0``, on principal coordinates ``u_k`` (m, broadcast):
    the Mehler kernel summed over the fugacity series, with the ground state removed::

        n_th = sum_j z^j [ prod_k e^{j a_k/2} sqrt(m w_k / (2 pi hbar sinh(j a_k)))
                           exp(-(m w_k / hbar) tanh(j a_k / 2) u_k^2)  -  |phi_0|^2 ]

    ``a_k = hbar w_k / kT``, ``z = e^{mu/kT}``.  No semiclassics: the oracle for the
    product basis on a ``HarmonicTrap``."""
    m = float(kc.m_K if mass is None else mass)
    kT = kc.kB * float(T_K)
    w = np.asarray(omega, dtype=float)
    a = kc.hbar * w / kT
    z = np.exp(float(mu_offset_J) / kT)
    if j_max is None:
        j_max = int(np.ceil(40.0 / float(np.min(a)))) + 1
    us = (np.asarray(u1, dtype=float), np.asarray(u2, dtype=float), np.asarray(u3, dtype=float))
    ground = 1.0
    for k in range(3):
        ground = ground * np.sqrt(m * w[k] / (np.pi * kc.hbar)) * np.exp(-m * w[k] / kc.hbar * us[k] ** 2)
    out = 0.0
    for j in range(1, j_max + 1):
        term = 1.0
        for k in range(3):
            ja = j * a[k]
            # e^{ja/2} / sqrt(sinh ja) = sqrt(2 / (1 - e^{-2 ja})): no overflow at large ja
            term = term * (np.sqrt(m * w[k] / (2.0 * np.pi * kc.hbar))
                           * np.sqrt(2.0 / -np.expm1(-2.0 * ja))
                           * np.exp(-(m * w[k] / kc.hbar) * np.tanh(0.5 * ja) * us[k] ** 2))
        out = out + z ** j * (term - ground)
    return out


# ------------------------------------------------------------ T_c closed forms

def critical_temperature_K(N: float, omega, mass: Optional[float] = None) -> float:
    """Ideal harmonic ``T_c``: ``k T_c = hbar wbar (N / zeta(3))^(1/3)``."""
    wbar = float(np.prod(np.asarray(omega, dtype=float))) ** (1.0 / 3.0)
    return kc.hbar * wbar * (float(N) / ZETA_3) ** (1.0 / 3.0) / kc.kB


def finite_size_shift(N: float, omega) -> float:
    """``dT_c / T_c = -0.7275 (w_mean / wbar) N^(-1/3)`` (Dalfovo et al. 1999, Eq. 15)."""
    w = np.asarray(omega, dtype=float)
    return -FINITE_SIZE_COEFF * (float(np.mean(w)) / float(np.prod(w)) ** (1.0 / 3.0)) * float(N) ** (-1.0 / 3.0)


def mean_field_shift(N: float, omega, a_scattering: float, mass: float) -> float:
    """``dT_c / T_c = -1.33 (a / a_ho) N^(1/6)`` (Dalfovo et al. 1999, Eq. 16)."""
    wbar = float(np.prod(np.asarray(omega, dtype=float))) ** (1.0 / 3.0)
    a_ho = np.sqrt(kc.hbar / (float(mass) * wbar))
    return -MEAN_FIELD_COEFF * float(a_scattering) / a_ho * float(N) ** (1.0 / 6.0)


# --------------------------------------------------------------- the cloud

class IdealHarmonicBoseGas:
    """Ideal Bose gas in an anisotropic harmonic trap: closed forms, no grid.

    Parameters
    ----------
    N : float
    omega : (3,) sequence
        Trap frequencies (rad/s), ordered (x, y, z) -- the axes the density is
        evaluated on.
    T_K : float
        Temperature (K).  ``0`` is the pure oscillator ground state.
    mass : float, optional
        Atomic mass (kg); default K-39.
    a_scattering : float, optional
        Used only for the reported mean-field ``T_c`` shift; the density is ideal.

    See the module docstring for the model.  ``density``/``column_density`` take
    ``component="total" | "condensate" | "thermal"``.  Meets the imaging cloud
    contract (``N``, ``widths``, ``density(x, y, z)``).
    """

    def __init__(self, N: float, omega: Sequence[float], T_K: float,
                 mass: Optional[float] = None, a_scattering: Optional[float] = None):
        self.N = float(N)
        self.omega = np.asarray(omega, dtype=float)
        if self.omega.shape != (3,) or not np.all(self.omega > 0):
            raise ValueError("omega must be three positive trap frequencies (rad/s)")
        self.T_K = float(T_K)
        if self.T_K < 0 or not np.isfinite(self.T_K):
            raise ValueError(f"T_K must be finite and >= 0; got {T_K}")
        self.mass = float(kc.m_K if mass is None else mass)
        self.a_scattering = None if a_scattering is None else float(a_scattering)
        self._z: Optional[float] = None

    # -------------------------------------------------------------- scales
    @property
    def omega_bar(self) -> float:
        return float(np.prod(self.omega)) ** (1.0 / 3.0)

    @property
    def kT(self) -> float:
        return kc.kB * self.T_K

    @property
    def T_nK(self) -> float:
        return self.T_K * 1e9

    @property
    def T_c_K(self) -> float:
        """Ideal harmonic ``T_c`` (K)."""
        return critical_temperature_K(self.N, self.omega)

    @property
    def T_c_nK(self) -> float:
        return self.T_c_K * 1e9

    @property
    def T_c_finite_size_K(self) -> float:
        """``T_c`` with the leading finite-size correction (K)."""
        return self.T_c_K * (1.0 + finite_size_shift(self.N, self.omega))

    @property
    def T_c_finite_size_nK(self) -> float:
        return self.T_c_finite_size_K * 1e9

    @property
    def T_c_interaction_shift(self) -> float:
        """Relative mean-field shift of ``T_c``; NaN without a scattering length."""
        if self.a_scattering is None:
            return float("nan")
        return mean_field_shift(self.N, self.omega, self.a_scattering, self.mass)

    @property
    def thermal_wavelength_m(self) -> float:
        """``h / sqrt(2 pi m k T)`` (m); inf at T = 0."""
        if self.T_K == 0:
            return float("inf")
        return kc.h / np.sqrt(2.0 * np.pi * self.mass * self.kT)

    @property
    def oscillator_lengths(self) -> np.ndarray:
        return np.sqrt(kc.hbar / (self.mass * self.omega))

    # ---------------------------------------------------------- occupation
    @property
    def fugacity(self) -> float:
        """``z = e^{mu/kT}``: 1 below ``T_c``, from ``g_3(z) (kT/hbar wbar)^3 = N`` above."""
        if self._z is None:
            if self.T_K == 0 or self.T_K <= self.T_c_K:
                self._z = 1.0
            else:
                target = self.N * (kc.hbar * self.omega_bar / self.kT) ** 3
                self._z = float(brentq(lambda z: float(bose_g(z, 3.0)) - target, 1e-300, 1.0,
                                       xtol=1e-16, rtol=1e-14))
        return self._z

    @property
    def chemical_potential(self) -> float:
        """``mu`` from the trap bottom (J): 0 below ``T_c``, ``kT ln z`` above."""
        return 0.0 if self.fugacity >= 1.0 else self.kT * np.log(self.fugacity)

    @property
    def N_th(self) -> float:
        """Semiclassical thermal number ``(kT / hbar wbar)^3 g_3(z)``, capped at N."""
        if self.T_K == 0:
            return 0.0
        return min(self.N, (self.kT / (kc.hbar * self.omega_bar)) ** 3 * float(bose_g(self.fugacity, 3.0)))

    @property
    def N_0(self) -> float:
        return self.N - self.N_th

    @property
    def condensate_fraction(self) -> float:
        """``1 - (T/T_c)^3`` below ``T_c``, 0 above (the semiclassical closed form)."""
        return self.N_0 / self.N

    def exact_condensate_fraction(self) -> float:
        """The finite-N sum's condensate fraction (no semiclassics, no sharp transition)."""
        if self.T_K == 0:
            return 1.0
        return ideal_harmonic_condensate(self.N, self.T_K, self.omega)[0] / self.N

    def temperature_for_condensate_fraction(self, fraction: float) -> float:
        """``T = T_c (1 - fraction)^(1/3)`` (K): the inverse question, in closed form."""
        f = float(fraction)
        if not 0.0 < f <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        return self.T_c_K * (1.0 - f) ** (1.0 / 3.0)

    # --------------------------------------------------------------- widths
    @property
    def sigma_condensate(self) -> np.ndarray:
        """Oscillator ground-state rms widths ``sqrt(hbar / 2 m w_k)`` (m)."""
        return np.sqrt(kc.hbar / (2.0 * self.mass * self.omega))

    @property
    def sigma_thermal(self) -> np.ndarray:
        """Bose-enhanced thermal rms widths ``sqrt(kT / m w_k^2) sqrt(g_4(z) / g_3(z))`` (m);
        NaN at T = 0."""
        if self.T_K == 0:
            return np.full(3, np.nan)
        z = self.fugacity
        ratio = float(bose_g(z, 4.0)) / float(bose_g(z, 3.0))
        return np.sqrt(self.kT / (self.mass * self.omega ** 2) * ratio)

    @property
    def sigma(self) -> np.ndarray:
        """rms widths of the total (bimodal) density about the centre (m)."""
        f = self.condensate_fraction
        if f >= 1.0:
            return self.sigma_condensate
        return np.sqrt(f * self.sigma_condensate ** 2 + (1.0 - f) * self.sigma_thermal ** 2)

    @property
    def widths(self) -> np.ndarray:
        """``sqrt(2) sigma`` of the total density: the imaging cloud contract (m)."""
        return np.sqrt(2.0) * self.sigma

    @property
    def widths_condensate(self) -> np.ndarray:
        return np.sqrt(2.0) * self.sigma_condensate

    @property
    def widths_thermal(self) -> np.ndarray:
        return np.sqrt(2.0) * self.sigma_thermal

    # -------------------------------------------------------------- density
    def _check_component(self, component: str) -> str:
        if component not in ("total", "condensate", "thermal"):
            raise ValueError("component must be 'total', 'condensate' or 'thermal'")
        return component

    def _V_over_kT(self, x, y, z):
        m, w = self.mass, self.omega
        return 0.5 * m * (w[0] ** 2 * x ** 2 + w[1] ** 2 * y ** 2 + w[2] ** 2 * z ** 2) / self.kT

    def density(self, x, y, z, *, component: str = "total"):
        """``n(x, y, z)`` (1/m^3) about the trap centre."""
        component = self._check_component(component)
        x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
        out = 0.0
        if component in ("total", "condensate"):
            s2 = self.sigma_condensate ** 2
            norm = self.N_0 / ((2.0 * np.pi) ** 1.5 * np.prod(np.sqrt(s2)))
            out = out + norm * np.exp(-0.5 * (x ** 2 / s2[0] + y ** 2 / s2[1] + z ** 2 / s2[2]))
        if component in ("total", "thermal") and self.T_K > 0:
            out = out + (bose_g(self.fugacity * np.exp(-self._V_over_kT(x, y, z)), 1.5)
                         / self.thermal_wavelength_m ** 3)
        return out

    def column_density(self, y, z, *, component: str = "total"):
        """Column density along ``x`` at ``(y, z)`` (1/m^2)."""
        component = self._check_component(component)
        y, z = (np.asarray(v, dtype=float) for v in (y, z))
        out = 0.0
        if component in ("total", "condensate"):
            s2 = self.sigma_condensate ** 2
            norm = self.N_0 / (2.0 * np.pi * np.sqrt(s2[1] * s2[2]))
            out = out + norm * np.exp(-0.5 * (y ** 2 / s2[1] + z ** 2 / s2[2]))
        if component in ("total", "thermal") and self.T_K > 0:
            lam = self.thermal_wavelength_m
            lx = np.sqrt(2.0 * np.pi * self.kT / (self.mass * self.omega[0] ** 2))
            v = self._V_over_kT(0.0, y, z)
            out = out + lx / lam ** 3 * bose_g(self.fugacity * np.exp(-v), 2.0)
        return out

    @property
    def peak_density(self) -> float:
        return float(self.density(0.0, 0.0, 0.0))

    @property
    def peak_density_thermal(self) -> float:
        return float(self.density(0.0, 0.0, 0.0, component="thermal"))

    @property
    def peak_column_density(self) -> float:
        return float(self.column_density(0.0, 0.0))

    @property
    def peak_degeneracy(self) -> float:
        """``n_th(0) lambda_dB^3``: ``zeta(3/2) = 2.612`` at and below ``T_c``."""
        if self.T_K == 0:
            return float("inf")
        return self.peak_density_thermal * self.thermal_wavelength_m ** 3

    # ------------------------------------------------------------- variants
    def with_temperature(self, T_K: float) -> "IdealHarmonicBoseGas":
        return IdealHarmonicBoseGas(self.N, self.omega, T_K, self.mass, self.a_scattering)

    def with_atom_number(self, N: float) -> "IdealHarmonicBoseGas":
        return IdealHarmonicBoseGas(N, self.omega, self.T_K, self.mass, self.a_scattering)

    @classmethod
    def from_trap(cls, trap, N: float, T_K: float, a_scattering: Optional[float] = None
                  ) -> "IdealHarmonicBoseGas":
        """From a :class:`kamo.trap.Trap` or ``HarmonicTrap``: its principal frequencies,
        in lab order when the principal axes are lab axes (the single tweezer), else in
        principal order -- the thermodynamics are rotation invariants, but ``density``
        then factorizes on the principal axes, not the lab ones, and a warning says so."""
        tf = trap.trap_frequencies()
        H = np.asarray(tf.hessian, dtype=float)
        off = float(np.max(np.abs(H - np.diag(np.diag(H)))))
        if off <= 1e-9 * float(np.max(np.abs(np.diag(H)))):
            omega = np.sqrt(np.diag(H) / trap.mass)
        else:
            omega = tf.omega
            warnings.warn("the trap's principal axes are not lab axes: the ideal-gas density "
                          "is on the principal axes (frequencies in principal order).",
                          UserWarning, stacklevel=2)
        return cls(N, omega, T_K, mass=trap.mass, a_scattering=a_scattering)

    # --------------------------------------------------------------- report
    def summary(self) -> str:
        w = self.omega / (2.0 * np.pi)
        lines = [f"IdealHarmonicBoseGas: N = {self.N:.0f}, f = ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}) Hz, "
                 f"T = {self.T_nK:.2f} nK",
                 f"  T_c = {self.T_c_nK:.2f} nK (ideal), {self.T_c_finite_size_nK:.2f} nK with the "
                 f"finite-size shift ({100 * finite_size_shift(self.N, self.omega):+.1f}%)"]
        if self.a_scattering is not None:
            lines.append(f"  mean-field T_c shift {100 * self.T_c_interaction_shift:+.2f}% "
                         f"(a = {self.a_scattering / kc.a0:+.2f} a0)")
        lines.append(f"  N0/N = {self.condensate_fraction:.4f} (semiclassical), "
                     f"{self.exact_condensate_fraction():.4f} (exact finite-N sum)")
        if self.T_K > 0:
            st, sc = self.sigma_thermal * 1e6, self.sigma_condensate * 1e6
            lines.append(f"  sigma thermal ({st[0]:.3f}, {st[1]:.3f}, {st[2]:.3f}) um, "
                         f"condensate ({sc[0]:.3f}, {sc[1]:.3f}, {sc[2]:.3f}) um, "
                         f"lambda_dB = {self.thermal_wavelength_m * 1e6:.3f} um")
            lines.append(f"  z = {self.fugacity:.6f}, n_th(0) lambda^3 = {self.peak_degeneracy:.3f}, "
                         f"peak n = {self.peak_density * 1e-6:.3e} cm^-3")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"IdealHarmonicBoseGas(N={self.N:.0f}, T={self.T_nK:.1f} nK, "
                f"N0/N={self.condensate_fraction:.3f})")
