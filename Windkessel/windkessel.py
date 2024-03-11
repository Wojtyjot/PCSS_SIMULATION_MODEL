# from utils import *
import numpy as np
import pandas as pd
from typing import *

# ZAŁOŻENIE dane w formie padas dtataframe i na tej podstawie tworzymy skrypt
# WSZYTKO W CGS
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

CT = SI * 10^5
R= SI * 10^-5
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
    r0 = r0 * 0.01
    return np.sqrt(beta / (2 * rho * np.sqrt(np.pi * r0**2)))


def compute_beta(df: pd.DataFrame, olufsen: bool, id: int) -> float:
    """
    Function computes Eh from empirical relation Olufsen

    Args:
        r0: [cm] radius of artery
    """
    # k1 = 2e7  # Ba
    # k2 = -22.53  # cm^-1
    # k3 = 8.65e5  # Ba
    #####################################################
    # ZMIANA Z OLUFSEN NA LINEAR
    # ##############################################
    if olufsen:
        k1 = 2e6
        k2 = -2253
        k3 = 86.5e3
        # change r0 from cm to m
        r0 = df[df.id == id].r0_out.iloc[0]
        r0 = r0 * 0.01
        Eh = r0 * (k1 * np.exp(k2 * r0) + k3)
        return 4 / 3 * np.sqrt(np.pi) * Eh
    else:
        # change r0 from cm to m
        # change modulus from Ba to Pa
        # change h from cm to m
        # r0 = df[df.id == id].r0_out.iloc[0] * 0.01
        h = df[df.id == id].thickness.iloc[0] * 0.01
        E = df[df.id == id].modulus.iloc[0] * 0.1  # modulus in script in BA
        # r0 = r0 * 0.01
        # Eh = 1.5e6 * r0
        return 4 / 3 * np.sqrt(np.pi) * E * h


def compute_R1(rho: float, c0: float, r0: float) -> float:
    """
    Function computes R1

    Args:
        rho: [g/cm^3] density of blood
        c0: [cm/s] pulse wave propagation speed
        r0: [cm] radius of artery
    """
    r0 = r0 * 0.01
    return rho * c0 / (np.pi * r0**2)


def compute_R_t(R_tot: float, COf: float, sigma_r0: float, r0: float) -> float:
    """
    Function computes R_t

    Args:
        R_tot: [Ba s / cm^3] total resistance
        COf: cardiac output fraction for brain vessels = 0.12
        sigma_r0: [cm] sum of radii of all arteries
        r0: [cm] radius of artery
    """
    r0 = r0 * 0.01
    sigma_r0 = sigma_r0 * 0.01
    return R_tot / COf * sigma_r0 / r0


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

def mmhg_to_Pa(p):
    """
    function converts pressure from mmhg to Pa
    """
    return p * 133.3223684


def compute_windkessel(
    df: pd.DataFrame,
    SV: float,
    P_sys: Optional[float] = 17331.91,
    P_dia: Optional[float] = 10132.50,
    # SV: Optional[float] = 70 * 1e-6,
    HR: Optional[float] = 1,
    rho: Optional[float] = 1050,
    tau: Optional[float] = 1.34,
    COf: Optional[float] = 0.12,
) -> pd.DataFrame:
    """
    Function estimates windkessel parameters using strategy from paper
    M. Sarabian, H. Babaee and K. Laksari,
    "Physics-Informed Neural Networks for Brain Hemodynamic Predictions Using Medical Imaging,"
    in IEEE Transactions on Medical Imaging,
    vol. 41, no. 9, pp. 2285-2303, Sept. 2022,
    doi: 10.1109/TMI.2022.3161653.

    Args:
        df: [DataFrame] pandas dataframe with arteries
        P_sys: [Ba] systolic pressure
        P_dia: [Ba] diastolic pressure
        SV: [cm^3] stroke volume
        HR: [1/s] heart rate
        rho: [g/cm^3] density of blood
        tau: [s] aortic pressure decay
        COf: cardiac output fraction for brain vessels = 0.12

    returns dataframe with windkessel parameters
    """
    ### w tej formie nadaje sie tylko do estymacji COW
    ### aby estymowaac w t ramiennych i aorcie trzeba zmienic
    ### plik csv moze miec tylko outlety w glowie
    # Trzeba wziać pod uwagę outlety w brachial i aorcie ...
    # w df dodajmy outlet = 2 dla brachials
    # i outlet = 3 dla aorty cof aorta =
    # cof brain = 0.12 / 0.15
    # cof arm = 0.05, each
    # cof thc aorta = 0.75

    R1, R2, C = list(), list(), list()
    r0 = list()
    r02 = list()
    ##############################
    R_tot = compute_R_tot(P_sys, P_dia, SV, HR)
    C_tot = compute_C_tot(R_tot, tau)
    ##################################
    # R_tot = 1.34e8
    # C_tot = 9.45e-9
    print("R_tot: ", R_tot)
    print("C_tot: ", C_tot)
    with open("Windkessel.txt", "w") as f:
        f.write(str(R_tot))
        f.write("\n")
        f.write(str(C_tot))
        f.close()

    for id in df.id:
        if (df[df.id == id].outlet == 1).bool():
            r0.append(df[df.id == id].r0_out.iloc[0])
        elif (df[df.id == id].outlet == 2).bool():
            r02.append(df[df.id == id].r0_out.iloc[0])
    sigma_r0 = np.sum(np.array(r0))
    sigma_r02 = np.sum(np.array(r02))
    for id in df.id:
        if (df[df.id == id].outlet == 1).bool():
            r_0 = df[df.id == id].r0_out.iloc[0]
            beta = compute_beta(df, olufsen=True, id=id)
            c0 = compute_c0(r_0, beta, rho)
            R_1 = compute_R1(rho, c0, r_0)
            R_T = compute_R_t(R_tot, COf=0.15, sigma_r0=sigma_r0, r0=r_0)
            R_2 = compute_R_2(R_T, R_1)
            C_T = compute_C_t(C_tot, R_tot, R_T)
            R1.append(R_1 * 1e-5)
            R2.append(R_2 * 1e-5)
            C.append(C_T * 1e5)
        elif (df[df.id == id].outlet == 2).bool():
            r_0 = df[df.id == id].r0_out.iloc[0]
            beta = compute_beta(df, olufsen=True, id=id)
            c0 = compute_c0(r_0, beta, rho)
            R_1 = compute_R1(rho, c0, r_0)
            R_T = compute_R_t(R_tot, COf=0.1, sigma_r0=sigma_r02, r0=r_0)
            R_2 = compute_R_2(R_T, R_1)
            C_T = compute_C_t(C_tot, R_tot, R_T)
            R1.append(R_1 * 1e-5)
            R2.append(R_2 * 1e-5)
            C.append(C_T * 1e5)
            # R1.append(13.37e2)
            # R2.append(2.5462)
            # C.append(2.58e-5)
        elif (df[df.id == id].outlet == 3).bool():
            r_0 = df[df.id == id].r0_out.iloc[0]
            beta = compute_beta(df, olufsen=True, id=id)
            c0 = compute_c0(r_0, beta, rho)
            R_1 = compute_R1(rho, c0, r_0)
            R_T = compute_R_t(R_tot, COf=0.75, sigma_r0=r_0, r0=r_0)
            R_2 = compute_R_2(R_T, R_1)
            C_T = compute_C_t(C_tot, R_tot, R_T)
            R1.append(R_1 * 1e-5)
            R2.append(R_2 * 1e-5)
            C.append(C_T * 1e5)
            # R1.append(1.77e2)
            # R2.append(0.1622e4)
            # C.append(38.7e-5)
        else:
            R1.append(0)
            R2.append(0)
            C.append(0)
    df["R1"] = R1
    df["R2"] = R2
    df["C"] = C
    return df
