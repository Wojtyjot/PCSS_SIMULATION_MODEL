from scipy import integrate
import pandas as pd
import numpy as np
from typing import *
import os
import time

# sampling all params and adding them to da
from pathlib import Path

# sampling of COW topologies
# odpalamy program w pcss z scratch?
# WARTOŚCI MUSZĄ BYĆ W CGS
# 1 EKSPERYMENT Z 1 COW i Z STAŁYMI BC
# stały R_tot


def sample_template(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Function samples COW topology from predefined templates

    Args:
        path: path to templates folder
    returns tuple of two dataframes:
        - df: dataframe with topology (names, nodes)
        - df_joints: dataframe with parameters of topology (joints)
    """

    templates = [f"{i}" for i in range(6)]  # zmienić do literatury

    prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # zmienić z literatura

    template = np.random.choice(templates)
    # zmienic jesze path po testach w pcss
    path_segments = path / f"TEMPLATE_{template}.csv"
    path_joints = path / f"TEMPLATE_{template}_TOPO.csv"

    df = pd.read_csv(path_segments)
    df_joints = pd.read_csv(path_joints)
    return (df, df_joints, template)


def sample_segment_params(df: pd.DataFrame, template: str) -> pd.DataFrame:
    """
    Function samples segment parameters from predefined multivariate
    gaussian distribution

    Args:
        df: dataframe with topology (names, nodes)
        template: string with template name

    returns dataframe with sampled segment parameters
    """

    # sampling L,r0_in, r0_out

    cov_matrix = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    )  # zmienić jak policzymy z danych n_COW*3 X n_COW*3

    mean = np.array([1.0, 0.5, 0.5])  # zmienić jak policzymy z danych
    # sampling every arttery
    out = np.random.multivariate_normal(mean, cov_matrix, size=len(df))
    # w out zmienić size na ilosc naczyn w COW
    L, r0_in, r0_out = out[:, 0], out[:, 1], out[:, 2]

    # maskowanie zmienic na podstawie danych
    if template == "0":
        # maska w zaleznosci od topologii
        mask = np.array([1, 0, 0, 0, 0, 0])

    elif template == "1":
        mask = np.array([0, 1, 0, 0, 0, 0])

    elif template == "2":
        mask = np.array([0, 0, 1, 0, 0, 0])

    elif template == "3":
        mask = np.array([0, 0, 0, 1, 0, 0])

    # apply mask
    L = L * mask
    r0_in = r0_in * mask
    r0_out = r0_out * mask
    L = L[L != 0]
    r0_in = r0_in[r0_in != 0]
    r0_out = r0_out[r0_out != 0]

    # add to dataframe
    df["L"] = L
    df["r0_in"] = r0_in
    df["r0_out"] = r0_out
    df["area_inlet"] = np.pi * r0_in**2
    df["area_outlet"] = np.pi * r0_out**2

    return df


def sample_system_params() -> Tuple[np.float, np.float, np.float, np.float]:
    """
    Function samples system parameters from predefined uniform distribution

    returns tuple of system parameters:
        - HR: heart rate
        - SV: stroke volume
        - P_sys: systolic pressure
        - P_dia: diastolic pressure
    """
    # ZAstanowić się nad rozkładem czy uniform czy normal
    # wartości jak dostarczy jacob
    # dodac jeszcze tau i Q do modelowania serca
    HR = np.random.uniform(60, 100)
    SV = np.random.uniform(60, 100)
    P_sys = np.random.uniform(120, 140)
    P_dia = np.random.uniform(80, 90)

    return (HR, SV, P_sys, P_dia)


def sample_Q_hat() -> np.float:
    """
    Function samples Q_hat from predefined uniform distribution
    pfv [cm/s]?

    returns Q_hat
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    pfv = np.random.uniform(100, 140)
    A = np.pi * 1.2**2

    return pfv * A


def generate_flow_file(
    Q_hat: float, T: float, dt: float, tau: float, filename: str
) -> None:
    """
    Function creates inlet flow file for 1D simulaion

    Q(t) = Q_hat*sin(pi*t/tau) if t < T else 0

    and saves as txt where first column is time and second is flow seperated by space
    """
    t = np.arange(0, T, dt)
    Q = Q_hat * np.sin(np.pi * t / tau)
    Q = np.where(t < tau, Q, 0)
    flow = np.column_stack((t, Q))
    np.savetxt(filename, flow, delimiter=" ")
    return None


def get_Q(Q_hat: float, t: np.array, tau: float) -> np.array:
    """
    Function creates inlet flow file for 1D simulaion

    Q(t) = Q_hat*sin(pi*t/tau) if t < T else 0

    and saves as txt where first column is time and second is flow seperated by space
    """
    Q = Q_hat * np.sin(np.pi * t / tau)
    Q = np.where(t < tau, Q, 0)
    return Q


def get_Q_v2(Q_hat: float, tau: float, t: np.array) -> np.array:
    """
    Test for inflow from chuj wie co
    """
    import matplotlib.pyplot as plt

    T = 1
    Q = Q_hat * np.sin(np.pi * t / tau)
    Q = np.where(t < tau, Q, 0)
    Q_2 = Q_hat / (t * 12) * np.log(t / tau)
    Q_2 = np.where(t > tau, Q_2, 0)
    Q = Q + Q_2
    # plt.plot(t, Q)
    # plt.grid(True)
    # plt.show()
    return Q


def get_Q_Zikic(Q_hat: float, t: np.array, t1: float, t2: float) -> np.array:
    """
    Function model inflow to ascending aorta following Zikic 2016
    """
    import matplotlib.pyplot as plt
    import sys

    theta1 = 93.0
    omega1 = 14.95
    theta2 = 7.67
    omega2 = 29.9
    phi = 0.6
    Q_1 = 73.689
    theta3 = 1.5
    omega3 = 104.66
    zeta = 0.3
    alpha = 0.95
    Q_0 = -24.5
    # t = np.linspace(0,0.6, 100)

    Q = theta1 * omega1**2 * t * np.exp(-omega1 * t)
    Q = np.where(t < t1, Q, 0)
    # Q += np.where(
    #    (t1 < t) & (t <t2), -theta2 * omega2 * np.cos(omega2 * t + phi) + Q_1, 0
    # )
    Q += np.where((t1 < t) & (t < t2), 485 * np.log(t / t1), 0)
    Q += np.where(
        t2 <= t,
        (
            theta3
            * omega3
            * (
                1
                - 1
                / alpha
                * np.exp(-zeta * omega3 * t)
                * np.sin(alpha * omega3 * t + phi)
            )
            + Q_0
        ),
        0,
    )
    plt.plot(t, Q)
    plt.grid(True)
    plt.show()
    sys.exit()
    return Q


def get_Q_database() -> np.array:
    """
    test case for physio ascending aorta
    """
    Q = np.load("/home/wojciech/Doppler/Aorta_data/test_q.npy")
    Q_last = Q[-1].repeat(999 - 955)
    Q = np.concatenate((Q, Q_last))
    print(Q.shape)
    return Q


def get_Q_SV(
    Q_hat: float, T: float, tau: float, dt: float
) -> Tuple[np.array, np.float]:
    """
    Function calculates the stroke volume by integrating the flow over time

    SV = ∫Q(t) dt from 0 to T
    """
    t = np.arange(0, T, dt)
    Q = get_Q(Q_hat, t, tau)
    # Q = get_Q_v2(Q_hat, tau, t)
    # Q = get_Q_Zikic(Q_hat, t, 0.3, 0.35)
    # Q = get_Q_database()
    Q_z = np.where(t < tau, Q, 0)
    SV = integrate.trapz(Q, t)
    return (Q, SV)


def sample_flow(
    T: Optional[float] = 1.0, dt: Optional[float] = 0.001, tau: Optional[float] = 0.35
) -> Tuple[np.ndarray, np.float]:
    """
    Function samples flow from hardcoded distribution and SV[ml]
    """
    t = np.arange(0, T, dt)
    Q_hat = sample_Q_hat()
    # Q_hat = 485.0
    Q, SV = get_Q_SV(Q_hat, T, tau, dt)
    flow = np.column_stack((t, Q))
    return (flow, SV)


def sample_vessel_params() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function samples vessel lengths and inlet diameters from
    uniform distribution

    params of all vesels outside of COW are constant

    only params from COW are sampled
    """
    # np.random.seed(2137420)
    np.random.seed()
    L = np.ones(33)
    r0_in = np.ones(33)
    r0_out = np.ones(33)
    ids = [
        10,
        11,
        13,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
    ]
    # removed last
    L_range = [
        (12.6, 17.4),
        (12.4, 17.6),
        (14.7, 14.9),
        (14.6, 14.8),
        (0.5, 0.5),
        (0.5, 1.8),
        (0.5, 1.8),
        (0.5, 0.5),
        (2.2, 3.3),
        (0.7, 2.1),
        (0.7, 2.1),
        (0.8, 2.2),
        (0.7, 2.3),
        (0.5, 0.8),
        (0.5, 0.8),
        (3.4, 3.6),
        (3.4, 3.7),
        (0.2, 0.6),
        (3.0, 3.8),
        (3.0, 3.8),
    ]
    r0_in_range = [
        (0.203, 0.25),
        (0.203, 0.25),
        (0.1325 * 0.7, 0.2025 * 0.7),
        (0.1325 * 0.7, 0.2025 * 0.7),
        (0.097, 0.174),
        (0.02, 0.15),
        (0.02, 0.15),
        (0.097, 0.174),
        (0.125, 0.185),
        (0.137, 0.246),
        (0.137, 0.246),
        (0.1225, 0.1525),
        (0.126, 0.162),
        (0.093, 0.133),
        (0.0935, 0.1335),
        (0.0985, 0.1605),
        (0.0945, 0.1575),
        (0.0295, 0.105),
        (0.093, 0.133),
        (0.0925, 0.13),
    ]
    r0_out_range = [
        (0.203, 0.25),
        (0.203, 0.25),
        (0.1325, 0.2025),
        (0.1325, 0.2025),
        (0.097, 0.174),
        (0.02, 0.15),
        (0.02, 0.15),
        (0.097, 0.174),
        (0.125, 0.185),
        (0.137, 0.246),
        (0.137, 0.246),
        (0.1225, 0.1525),
        (0.126, 0.162),
        (0.093, 0.133),
        (0.0935, 0.1335),
        (0.0985, 0.1605),
        (0.0945, 0.1575),
        (0.0295, 0.105),
        (0.093, 0.133),
        (0.0925, 0.13),
    ]
    for i, idx in enumerate(ids):
        L[idx] = np.random.uniform(L_range[i][0], L_range[i][1])
        r0_out[idx] = r0_in[idx] = np.random.uniform(
            r0_in_range[i][0], r0_in_range[i][1]
        )
        # r0_out[idx] = np.random.uniform(r0_out_range[i][0], r0_out_range[i][1])

    # hardcoding non COW vessels
    # ascending aorta
    L[0] = 4.0
    r0_in[0] = 1.2
    r0_out[0] = 1.2

    # aortic arch I
    L[1] = 2.0
    r0_in[1] = 1.12
    r0_out[1] = 1.12

    # brachiocephalic
    L[2] = 3.4
    r0_in[2] = 0.62
    r0_out[2] = 0.62

    # ao arch II
    L[3] = 3.9
    r0_in[3] = 1.07
    r0_out[3] = 1.07

    # left common carotid
    L[4] = 20.8
    r0_in[4] = 0.250
    r0_out[4] = 0.250

    # R common carotid
    L[5] = 17.7
    r0_in[5] = 0.250
    r0_out[5] = 0.250

    # R subclaian
    L[6] = 3.4
    r0_in[6] = 0.423
    r0_out[6] = 0.423

    # Thoracic aorta
    L[7] = 15.6
    r0_in[7] = 0.999
    r0_out[7] = 0.999

    # L subclavian
    L[8] = 3.4
    r0_in[8] = 0.423
    r0_out[8] = 0.423

    # L ext carotid
    L[9] = 17.7
    r0_in[9] = 0.150
    r0_out[9] = 0.150

    # ICA???? I

    # R ext carotid
    L[12] = 17.7
    r0_in[12] = 0.150
    r0_out[12] = 0.150

    # R vertebral
    # TODO

    # R Brachial
    L[14] = 42.2
    r0_in[14] = 0.403
    r0_out[14] = 0.403

    # L brachial
    L[15] = 42.2
    r0_in[15] = 0.403
    r0_out[15] = 0.403

    # r int carotid 2
    r0_in[20] = r0_out[20] = r0_in[11]

    # l int carotid 2
    r0_in[17] = r0_out[17] = r0_in[10]

    return (L, r0_in, r0_out)


def params_Ala() -> Tuple[list, list, list]:

    df = pd.read_csv(
        #"/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data/PCSS_SIMULATION_MODEL/Data/Alstruey2007.csv"
        "/home/wojciech/Doppler/PCSS_SIMULATION_MODEL/Data/Alstruey2007.csv"
    )
    L = df["L"].to_numpy() * 100
    r0_out = r0_in = df["Rp"].to_numpy() * 100
    return (L, r0_in, r0_out)


def get_thickness(r0_in: np.ndarray) -> np.ndarray:
    """
    Function calculates thickness of vessel wall
    """
    h = 0.25 * r0_in
    h[0] = 0.163
    h[1] = 0.126
    h[2] = 0.080
    h[3] = 0.115
    h[4] = 0.063
    h[5] = 0.063
    h[6] = 0.067
    h[7] = 0.110
    h[8] = 0.067
    h[9] = 0.038
    h[10] = 0.038
    h[14] = 0.067
    h[15] = 0.067
    return h


def sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.float]:
    """
    Function samples COW and flow parameters from predefined uniform
    distributions and adds them to df
    """
    # L, r0_in, r0_out = sample_vessel_params()
    L, r0_in, r0_out = params_Ala()
    path = "/home/wojciech/Doppler/PCSS_SIMULATION_MODEL/Data/MRI_COW_v2.csv"
    # L, r0_in, r0_out = sample_normal(path)
    L, r0_in, r0_out = add_gaussian_noise(L, r0_in, r0_out)
    df["L"] = L
    df["r0_in"] = r0_in
    df["r0_out"] = r0_out
    df["area_inlet"] = np.pi * r0_in**2
    df["area_outlet"] = np.pi * r0_out**2
    df["thickness"] = get_thickness(r0_in)

    flow, SV = sample_flow()

    return (df, flow, SV)


def get_covariance_matrix(df_path: str) -> np.ndarray:
    """
    Function calculate covariance matrix
    for all variables in data frame collumns
    """
    df = pd.read_csv(df_path)
    df = df / 10
    cov = df.cov().to_numpy()
    return cov


def get_mean(df_path: str) -> np.ndarray:
    """
    Function calculate mean vector
    for all variables in data frame collumns
    """
    df = pd.read_csv(df_path)
    df = df / 10
    mean = df.mean().to_numpy()
    return mean


def add_gaussian_noise(
    L: np.ndarray, r0_in: np.ndarray, r0_out: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function adds gaussian noise to sampled params
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    ids = [
        10,
        11,
        13,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
    ]
    for i in ids:
        if i in [22, 23]:
            L[i] = L[i] + np.random.normal(0, 0.05 * (L[i] - 9.5)) - 9.5
            r0_in[i] = r0_out[i] = r0_in[i] + np.random.normal(0, 0.05 * r0_in[i])
        elif i in [28, 29]:
            L[i] = L[i] - 7.0 + np.random.normal(0, 0.05 * (L[i] - 7.0))
            r0_in[i] = r0_out[i] = r0_in[i] + np.random.normal(0, 0.05 * r0_in[i])
        else:
            L[i] = L[i] + np.random.normal(0, 0.05 * L[i])
            r0_in[i] = r0_out[i] = r0_in[i] + np.random.normal(0, 0.05 * r0_in[i])
        # r0_out[i] = np.random.normal(0, 0.1*r0_out[i])
    return (L, r0_in, r0_out)


def sample_conditional_uniform(
    mean: np.ndarray,
    sample: np.ndarray,
    L: np.ndarray,
    r0_in: np.ndarray,
    r0_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function samples params from predefined distributions

    given if sample is bigger than mean or not
    """
    sample_ids = [
        0,
        # 6,
        4,
        # 4,
        2,
        # 8,
        0,
    ]
    ids = [
        10,
        # 11,
        13,
        # 16,
        28,
        # 29,
        30,
    ]
    # removed last
    L_range = [
        (12.6, 17.4),
        # (12.4, 17.6),
        (14.7, 14.9),
        # (14.6, 14.8),
        (3.4, 3.6),
        # (3.4, 3.7),
        (0.2, 0.6),
    ]
    r0_in_range = [
        (0.203, 0.25),
        # (0.203, 0.25),
        (0.1325 * 0.7, 0.2025 * 0.7),
        # (0.1325 * 0.7, 0.2025 * 0.7),
        (0.0985, 0.1605),
        # (0.0945, 0.1575),
        (0.03, 0.105),
    ]

    for i, idx in enumerate(ids):
        L[idx] = np.random.uniform(L_range[i][0], L_range[i][1])
        if sample[sample_ids[i]] > mean[sample_ids[i]]:
            r0_out[idx] = r0_in[idx] = np.random.uniform(
                np.mean(r0_in_range[i][0] + r0_in_range[i][1]), r0_in_range[i][1]
            )
        else:
            r0_out[idx] = r0_in[idx] = np.random.uniform(
                r0_in_range[i][0], np.mean(r0_in_range[i][0] + r0_in_range[i][1])
            )
    r0_in[28] = r0_out[28] = r0_in[24]
    r0_in[29] = r0_out[29] = r0_in[25]
    L[11] = L[10]
    r0_in[11] = r0_out[11] = r0_in[10]
    L[16] = L[13]
    r0_in[16] = r0_out[16] = r0_in[13]
    L[29] = L[28]
    # r0_in[29] = r0_out[29] = r0_in[28]

    return (L, r0_in, r0_out)


def sample_normal(mesurements_path: str) -> np.ndarray:
    """
    Function samples normal distribution
    from given mean and covariance matrix
    """
    cov = get_covariance_matrix(mesurements_path)
    mean = get_mean(mesurements_path)
    sample = np.random.multivariate_normal(mean, cov)
    L = np.ones(33)
    r0_in = np.ones(33)
    r0_out = np.ones(33)
    # encoding sampled values
    # L_ICA_II
    L[17] = sample[11]
    r0_in[17] = sample[0]
    r0_out[17] = sample[0]

    # L_MCA
    L[22] = sample[12]
    r0_in[22] = sample[1]
    r0_out[22] = sample[1]

    # L_ACA_A1
    L[24] = sample[13]
    r0_in[24] = sample[2]
    r0_out[24] = sample[2]

    # L_PCA_P1
    L[26] = sample[14]
    r0_in[26] = sample[3]
    r0_out[26] = sample[3]

    # BA
    L[21] = sample[15]
    r0_in[21] = sample[4]
    r0_out[21] = sample[4]

    # L_PcoA
    L[18] = sample[16]
    r0_in[18] = sample[5]
    r0_out[18] = sample[5]

    # R_ICA_II
    L[20] = sample[17]
    r0_in[20] = sample[6]
    r0_out[20] = sample[6]

    # R_MCA
    L[23] = sample[18]
    r0_in[23] = sample[7]
    r0_out[23] = sample[7]

    # R_ACA_A1
    L[25] = sample[19]
    r0_in[25] = sample[8]
    r0_out[25] = sample[8]

    # R_PCA_P1
    L[27] = sample[20]
    r0_in[27] = sample[9]
    r0_out[27] = sample[9]

    # R_PcoA
    L[19] = sample[21]
    r0_in[19] = sample[10]
    r0_out[19] = sample[10]

    # sampling pcas and rs of pcas
    r0_in[31] = r0_out[31] = sample[3]
    r0_in[32] = r0_out[32] = sample[9]

    # sampling lengths of pcas
    L[31] = np.random.uniform(2.0, 2.8)
    L[32] = np.random.uniform(2.0, 2.8)

    # hardcoding non COW vessels
    # ascending aorta
    L[0] = 4.0
    r0_in[0] = 1.2
    r0_out[0] = 1.2

    # aortic arch I
    L[1] = 2.0
    r0_in[1] = 1.12
    r0_out[1] = 1.12

    # brachiocephalic
    L[2] = 3.4
    r0_in[2] = 0.62
    r0_out[2] = 0.62

    # ao arch II
    L[3] = 3.9
    r0_in[3] = 1.07
    r0_out[3] = 1.07

    # left common carotid
    L[4] = 20.8
    r0_in[4] = 0.250
    r0_out[4] = 0.250

    # R common carotid
    L[5] = 17.7
    r0_in[5] = 0.250
    r0_out[5] = 0.250

    # R subclaian
    L[6] = 3.4
    r0_in[6] = 0.423
    r0_out[6] = 0.423

    # Thoracic aorta
    L[7] = 15.6
    r0_in[7] = 0.999
    r0_out[7] = 0.999

    # L subclavian
    L[8] = 3.4
    r0_in[8] = 0.423
    r0_out[8] = 0.423

    # L ext carotid
    L[9] = 17.7
    r0_in[9] = 0.150
    r0_out[9] = 0.150

    # ICA???? I

    # R ext carotid
    L[12] = 17.7
    r0_in[12] = 0.150
    r0_out[12] = 0.150

    # R vertebral
    # TODO

    # R Brachial
    L[14] = 42.2
    r0_in[14] = 0.403
    r0_out[14] = 0.403

    # L brachial
    L[15] = 42.2
    r0_in[15] = 0.403
    r0_out[15] = 0.403

    # r int carotid 2
    # r0_in[20] = r0_out[20] = r0_in[11]

    # l int carotid 2
    # r0_in[17] = r0_out[17] = r0_in[10]

    L, r0_in, r0_out = sample_conditional_uniform(mean, sample, L, r0_in, r0_out)

    return (L, r0_in, r0_out)
