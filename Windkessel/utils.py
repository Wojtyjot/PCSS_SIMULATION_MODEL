import os
import numpy as np
from typing import float, Optional

# DLA SIMVASCULAR WSZYSTKIE WARTOSCI TRZEBA ZMIENIC NA CGS SYSTEM
# 1 EXP, stałe: HR, SV, P_dia, P_sys, tau
"""
SI to CGS conversion
length[cm = 0.01m]
mass [g = 0.001 kg]
time [s]
velocity [cm/s = 0.01 m/s]
acceleration [Gal == cm/s^2 = 0.01 m/s^2]
pressure [Ba == g/(cm * s^2) = 0.1 Pa]
dynamic viscosity [P == g/(cm *s) = 0.1 Pa*s]
"""


def compute_R_tot(P_sys: float, P_dia: float, SV: float, HR: float) -> float:
    """
    Function computes total resistance of the cardiovascular system

    Args:
        P_sys: [Ba] systolic pressure
        P_dia: [Ba] diastolic pressure
        SV: [cm^3] stroke volume
        HR: [1/s] heart rate
    """
    return (1 / 3 * P_sys + 2 / 3 * P_dia) / (SV * HR)


def compute_C_tot(R_tot: float, tau: Optional[float] = 1.34) -> float:
    """
    Function computes total compliance

    Args:
        R_tot: [Ba s / cm^3] total resistance
        tau: [s] aortic pressure decay
    """
    return tau / R_tot


def compute_c0(r0: float, beta: float, rho: float) -> float:
    """
    Function computes  pulswave propagation speed at eq

    Args:
        r0: [cm] radius of artery
        beta: vessel wall elasticity
        rho: [g/cm^3] density of blood
    """
    return np.sqrt(beta / (2 * rho * np.sqrt(np.pi * r0**2)))


def compute_beta(r0: float) -> float:
    """
    Function computes Eh from empirical relation Olufsen

    Args:
        r0: [cm] radius of artery
    """
    k1 = 2e7  # Ba
    k2 = -22.53  # cm^-1
    k3 = 8.65e5  # Ba
    Eh = r0 * (k1 * np.exp(k2 * r0) + k3)
    return 4 / 3 * np.sqrt(np.pi) * Eh


def compute_R1(rho: float, c0: float, r0: float) -> float:
    """
    Function computes R1

    Args:
        rho: [g/cm^3] density of blood
        c0: [cm/s] pulse wave propagation speed
        r0: [cm] radius of artery
    """
    return rho * c0 / (np.pi * r0**2)


def compute_R_t(R_tot: float, COf: float, sigma_r0: float, r0: float) -> float:
    """
    Function computes R_t

    Args:
        R_tot: [Ba s / cm^3] total resistance
        COf: cardiac output fraction for brain vessels = 0.12
        sigma_r0: [cm^2] sum of radii of all arteries
        r0: [cm] radius of artery
    """
    return R_tot - COf * sigma_r0 / r0


def compute_R_2(R_T: float, R_1: float) -> float:
    """
    Function computes R_2

    Args:
        R_T: [Ba s / cm^3] total resistance
        R_1: [Ba s / cm^3] upstream resistance
    """
    return R_T - R_1


def compute_C_t(C_tot: float, R_tot: float, R_T: float) -> float:
    """
    Function computes C_t

    Args:
        C_tot: [cm^3 / Ba] total compliance
        R_tot: [Ba s / cm^3] total resistance
        R_T: [Ba s / cm^3] total resistance at output
    """
    return C_tot * R_tot / R_T


def mmHg_to_unit(p):
    """
    Converts pressure value in mmHg to g cm-1 s-2.

    Arguments
    ---------
    p : float
        Pressure value in mmHg

    Returns
    -------
    return : float
        Pressure value in g cm-1 s-2
    """
    return 101325 / 76 * p


def unit_to_mmHg(p):
    """
    Converts pressure value in g cm-1 s-2 to mmHg.

    Arguments
    ---------
    p : float
        Pressure value in g cm-1 s-2

    Returns
    -------
    return : float
        Pressure value in mmHg
    """
    return 76 / 101325 * p
