import os
import numpy as np
import pandas as pd
from typing import *
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

    returns Q_hat
    """
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


def sample_flow(
    T: Optional[float] = 1.0, dt: Optional[float] = 0.001, tau: Optional[float] = 0.3
) -> np.ndarray:
    """
    Function samples flow from hardcoded distribution
    """
    t = np.arrange(0, T, dt)
    Q_hat = sample_Q_hat()
    Q = Q_hat * np.sin(np.pi * t / tau)
    Q = np.where(t < tau, Q, 0)
    flow = np.column_stack((t, Q))
    return flow


def sample_vessel_params() -> Tuple[list, list, list]:
    """
    Function samples vessel lengths and inlet diameters from
    uniform distribution

    params of all vesels outside of COW are constant

    only params from COW are sampled
    """
    L = np.ones(32)
    r0_in = np.ones(32)
    r0_out = np.ones(32)
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
        r0_in[idx] = np.random.uniform(r0_in_range[i][0], r0_in_range[i][1])
        r0_out[idx] = np.random.uniform(r0_out_range[i][0], r0_out_range[i][1])
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

    return (L, r0_in, r0_out)
