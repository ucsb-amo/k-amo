"""Real Tiemann/Falke K2 Born-Oppenheimer ground-state potentials for K39.

Analytic three-region parameterization from Falke et al., PRA 78, 012503 (2008),
Tables IV (singlet X^1 Sigma_g+) and V (triplet a^3 Sigma_u+), "case A" (no
Born-Oppenheimer correction).  Energies in cm^-1, distances in Angstrom in the
native form; :func:`potential_hartree` returns Hartree for R in Bohr.

    R < Rinn:            U = A + B/R^Ns
    Rinn <= R <= Rout:   U = sum_i a_i * xi^i,   xi = (R - Rm)/(R + b*Rm)
    R > Rout:            U = U_inf - C6/R^6 - C8/R^8 - C10/R^10 - A_ex*R^gamma*exp(-beta*R)

The exchange term carries the tabulated sign of A_ex (+ singlet lowers, - triplet
raises); the single overall minus sign is fixed by validation against Falke
Table VIII scattering lengths (a_S~138.80, a_T~-33.41 a0).
"""

from __future__ import annotations

import numpy as np

from kamo import constants as c

# unit conversions
HARTREE_CM1 = 219474.6313702          # cm^-1 per Hartree
BOHR_ANG = 0.52917721067              # Angstrom per Bohr

# ---- Table IV: singlet X^1 Sigma_g+ (no BO correction) ----
SINGLET = dict(
    Rinn=2.870, Rout=12.000, b=-0.40, Rm=3.92436437, Ns=12,
    A=-0.262878738e4, B=0.8129033720e9,
    C6=0.1892046304e8, C8=0.5700273275e9, C10=0.1866135374e11,
    A_ex=0.97014411e4, gamma=5.19500, beta=2.13539, U_inf=0.0,
    a=[-4450.899108, 0.027435192082021, 0.13671215240591e5, 0.10750901039993e5,
       -0.20933147904789e4, -0.19385880603136e5, -0.49208904259548e5,
       0.11026640034823e6, 0.72867340031285e6, -0.29310679230619e7,
       -0.12407070105941e8, 0.40333947204823e8, 0.13229848708507e9,
       -0.37617673800749e9, -0.95250413278553e9, 0.24655585743928e10,
       0.47848257694561e10, -0.11582132110109e11, -0.17022518297748e11,
       0.39469335034300e11, 0.43141949844175e11, -0.97616955325590e11,
       -0.77417530686085e11, 0.17314133615536e12, 0.96118849114885e11,
       -0.21425463041524e12, -0.78513081753454e11, 0.17539493131261e12,
       0.37939637010974e11, -0.85271868689619e11, -0.82123523177698e10,
       0.18626451758590e11],
)

# ---- Table V: triplet a^3 Sigma_u+ (no BO correction) ----
TRIPLET = dict(
    Rinn=4.750, Rout=12.000, b=-0.300, Rm=5.73392370, Ns=6,
    A=-0.672898984e3, B=0.7735201466e7,
    C6=0.1892046304e8, C8=0.5700273275e9, C10=0.1866135374e11,
    A_ex=-0.97014411e4, gamma=5.19500, beta=2.13539, U_inf=0.0,
    a=[-255.016075, -0.83437034991917e1, 0.20960239701879e4,
       -0.17090691582228e4, -0.17873986188680e4, 0.29450770829461e4,
       -0.20200111692363e5, -0.35699427038012e5, 0.59869069169566e6,
       -0.71054314902491e6, -0.61771835715388e7, 0.19365507918230e8,
       0.67930591036665e7, -0.12020061749490e9, 0.21603960091887e9,
       -0.63531970658436e8, -0.52391212911571e9, 0.15913304556368e10,
       -0.24792546801660e10, 0.20326032002627e10, -0.68044505933607e9],
)


def potential_cm1(R_ang, params):
    """Tiemann potential (cm^-1) at internuclear distance R_ang (Angstrom).

    Vectorized over R_ang.  Piecewise: inner wall / power series / long range.
    """
    R = np.asarray(R_ang, dtype=float)
    out = np.empty_like(R)
    Rinn, Rout, Rm, b = params["Rinn"], params["Rout"], params["Rm"], params["b"]

    inner = R < Rinn
    outer = R > Rout
    mid = ~inner & ~outer

    # inner wall
    if np.any(inner):
        Ri = R[inner]
        out[inner] = params["A"] + params["B"] / Ri ** params["Ns"]
    # intermediate power series in xi
    if np.any(mid):
        Rmid = R[mid]
        xi = (Rmid - Rm) / (Rmid + b * Rm)
        s = np.zeros_like(Rmid)
        for i, ai in enumerate(params["a"]):
            s = s + ai * xi ** i
        out[mid] = s
    # long range
    if np.any(outer):
        Ro = R[outer]
        disp = (params["U_inf"] - params["C6"] / Ro ** 6 - params["C8"] / Ro ** 8
                - params["C10"] / Ro ** 10)
        exch = params["A_ex"] * Ro ** params["gamma"] * np.exp(-params["beta"] * Ro)
        out[outer] = disp - exch
    return out if out.ndim else float(out)


def potential_hartree(R_bohr, params):
    """Tiemann potential (Hartree) at R_bohr (Bohr), referenced to the asymptote."""
    R_ang = np.asarray(R_bohr, dtype=float) * BOHR_ANG
    return potential_cm1(R_ang, params) / HARTREE_CM1


def scattering_length(params, mu=None, r_in_bohr=4.0, r_out_bohr=1500.0,
                      h=0.005, fit_from=1200.0) -> float:
    """Zero-energy s-wave scattering length (a0) of a Tiemann potential.

    Numerov integration of u'' = 2 mu V(r) u outward from ``r_in_bohr``;
    ``a = -d/c`` from the asymptotic linear fit ``u ~ c*r + d``.
    """
    from .potentials import MU_AU
    if mu is None:
        mu = MU_AU
    n = int(round((r_out_bohr - r_in_bohr) / h)) + 1
    r = r_in_bohr + h * np.arange(n)
    f = 2.0 * mu * potential_hartree(r, params)
    w = 1.0 - (h * h / 12.0) * f
    u = np.zeros(n)
    u[1] = 1e-14
    h2 = h * h
    for i in range(1, n - 1):
        u[i + 1] = (2.0 * u[i] * (1.0 + 5.0 * h2 / 12.0 * f[i])
                    - u[i - 1] * w[i - 1]) / w[i + 1]
        if abs(u[i + 1]) > 1e120:
            u[:i + 2] /= 1e120
    m = r >= fit_from
    A = np.vstack([r[m], np.ones(m.sum())]).T
    cc, dd = np.linalg.lstsq(A, u[m], rcond=None)[0]
    return float(-dd / cc)
