import numpy as np
#import torch
import os
import pandas as pd
import sys
from typing import *
from pathlib import Path

# Script for creating dataset from the raw data compatibile with GNOT
# dataset in format dataset = [
#            [X, Y, theta_i, (in_funcs)]
#    ],
# X (nx * nt, 2) - input grid or (nx * (nt - 1), 2) for the prediction task
# Y (nx * nt, 3) - output  or (nx * (nt - 1), 2) for the prediction task
# theta_i - global parameters one-hot encoded artery + lengths in COW + r0's in COW + SV
# in_funcs - initial conditions, boundary conditions, forcing functions


def load_dat_file(file: str) -> np.ndarray:
    """
    Function loads data from .dat file and returns it as numpy array
    """
    # nie wiem dlaczego ale jest szybciej niz np.loadtxt XD
    df = pd.read_csv(file, sep=" ")
    p = df.to_numpy()
    shape = p.shape
    new_shape = (shape[0], shape[1] - 1)
    p = p[~np.isnan(p)].reshape(new_shape)
    p = p[:, -100:] # take only last 100 time steps == last cycle 
    return p


def create_X(L: float) -> np.ndarray:
    """
    Function creates X matrix from the input array

    Parameters
    ----------
    L : float
        Length of the artery
    """
    t = np.linspace(0, 1, 100)  # nt = 100
    x = np.linspace(0, L, 200)  # nx = 200
    num_nodes = 200 * 100  # nx * nt

    [X, T] = np.meshgrid(x, t)

    X = X.reshape(num_nodes, 1)
    T = T.reshape(num_nodes, 1)

    out = np.concatenate((X, T), axis=-1)
    return out


def encode_artery(artery: str) -> np.ndarray:
    """
    Function encodes artery type as one-hot vector

    Parameters
    ----------
    artery : str
        Artery type must be one of the following:
        "VA", "ICA_1", "BA", "MCA", "ACA_A1", "ACA_A2",
        "PCA_P1", "PCA_P2", "PCOA", "ACOA", "ICA_2"
    """
    arteries = [
        "VA",
        "ICA_1",
        "BA",
        "MCA",
        "ACA_A1",
        "ACA_A2",
        "PCA_P1",
        "PCA_P2",
        "PCOA",
        "ACOA",
        "ICA_2",
    ]
    out = np.zeros(len(arteries))
    index = arteries.index(artery)
    out[index] = 1
    return out

def decode_artery(artery: np.ndarray) -> str:
    """
    Function decodes artery one-hot vector to artery name
    """
    arteries = [
        "VA",
        "ICA_1",
        "BA",
        "MCA",
        "ACA_A1",
        "ACA_A2",
        "PCA_P1",
        "PCA_P2",
        "PCOA",
        "ACOA",
        "ICA_2",
    ]
    index = np.argmax(artery)
    return arteries[index]

def get_name_for_encoding(artery: str) -> str:
    """
    Function returns the name of the artery for encoding
    """
    if artery == "L_int_carotid_I" or artery == "R_int_carotid_I":
        return "ICA_1"
    elif artery == "R_vertebral" or artery == "L_vertebral":
        return "VA"
    elif artery == "Basilar":
        return "BA"
    elif artery == "L_int_carotid_II" or artery == "R_int_carotid_II":
        return "ICA_2"
    elif artery == "L_PcoA" or artery == "R_PcoA":
        return "PCOA"
    elif artery == "L_MCA" or artery == "R_MCA":
        return "MCA"
    elif artery == "L_ACA_A1" or artery == "R_ACA_A1":
        return "ACA_A1"
    elif artery == "L_PCA_P1" or artery == "R_PCA_P1":
        return "PCA_P1"
    elif artery == "L_ACA_A2" or artery == "R_ACA_A2":
        return "ACA_A2"
    elif artery == "AcoA":
        return "ACOA"
    elif artery == "L_PCA_P2" or artery == "R_PCA_P2":
        return "PCA_P2"

def create_Y(pressure: np.ndarray, area: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
    Function creates Y matrix for the output array

    output array has shape (nx * nt, 3)
    withe entries: [
        [p(x0, t0), A(x0, t0), u(x0, t0)],
        [p(x0, t1), A(x0, t1), u(x0, t1)],
        ...
    ]

    Parameters
    ----------
    pressure : np.ndarray
        Pressure array (nx, nt)
    area : np.ndarray
        Area array (nx, nt)
    velocity : np.ndarray
        Velocity array (nx, nt)
    """
    pressure = pressure.reshape(-1, 1)
    area = area.reshape(-1, 1)
    velocity = velocity.reshape(-1, 1)
    out = np.concatenate((pressure, area, velocity), axis=-1)
    return out

def get_velocity(area: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Function computes velocity from the area and flow arrays
    """
    velocity = flow / area
    return velocity

def create_theta(artery: str, lengths: np.ndarray, r0s: np.ndarray, SV: np.ndarray) -> np.ndarray:
    """
    Function creates theta vector for the global parameters

    Parameters
    ----------
    artery : str
        Artery type
    lengths : np.ndarray
        Lengths of the arteries in the COW
    r0s : np.ndarray
        r0s of the arteries in the COW
    SV : float
        Stroke volume
    """
    artery = get_name_for_encoding(artery)
    artery = encode_artery(artery)
    out = np.concatenate((artery, lengths, r0s, SV))
    return out

def get_inflow_BC(velocity: np.ndarray) -> np.ndarray:
    """
    Function computes inflow boundary conditon from velocity array

    BC in form [
        [x0, t0, u(x0, t0)],
        [x0, t1, u(x0, t1)],
        ...
    ]
    dim = nt X 3

    Parameters
    ----------
    velocity : np.ndarray
        Velocity array (nx, nt)
    """
    x = np.zeros((100, 1))
    t = np.linspace(0, 1, 100)
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)    
    out = np.concatenate((x, t, velocity[0, :].reshape(-1, 1)), axis=-1)
    return out

def get_IC(pressure: np.ndarray, area: np.ndarray, velocity: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function computes initial condition from the pressure, area and velocity arrays

    IC in form [
        [x0, t0, p/u/a(x0, t0)],
        [x1, t0, p/u/a(x1, t0)],
        ...
    ]

    Parameters
    ----------
    pressure : np.ndarray
        Pressure array (nx, nt)
    area : np.ndarray
        Area array (nx, nt)
    velocity : np.ndarray
        Velocity array (nx, nt)
    L : float
        length of vessel

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Initial condition for pressure, velocity and area
    """
    x = np.linspace(0, L, 200) #nx = 200
    t = np.zeros((200,1))
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    p0 = np.concatenate((x, t, pressure[:, 0].reshape(-1, 1)), axis=-1)
    u0 = np.concatenate((x, t, velocity[:, 0].reshape(-1, 1)), axis=-1)
    a0 = np.concatenate((x, t, area[:, 0].reshape(-1, 1)), axis=-1)
    return p0, u0, a0

def main():
    """
    Main function for creating dataset on PCSS
    """
    data_dir = Path("/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data")

    dataset_name = "GNOT_dataset_no_end_BC"
    arteries = [
        "L_int_carotid_I",
        "R_int_carotid_I",
        #"R_vertebral",
        #"L_vertebral",
        "Basilar",
        "L_int_carotid_II",
        "R_int_carotid_II",
        "L_PcoA",
        "R_PcoA",
        "L_MCA",
        "R_MCA",
        "L_ACA_A1",
        "R_ACA_A1",
        "L_PCA_P1",
        "R_PCA_P1",
        "L_ACA_A2",
        "R_ACA_A2",
        "AcoA",
        "L_PCA_P2",
        "R_PCA_P2",
    ]
    dataset = []

    for folder in ["DATA", "DATA_2", "DATA_3", "DATA_4"]:
        path = data_dir / folder
        for i in range(1200):
            results_dir = path / f"RESULTS_{i}"
            if results_dir.exists():
                if len(list(results_dir.glob("*.dat"))) > 0:
                    params_df = pd.read_csv(results_dir / f"PARAMS_{i}.csv")
                    SV = np.loadtxt(results_dir / f"SV.txt").reshape(1)

                    # Load parameters of given Circle of Willis
                    # Remember to load COW ONLY
                    lengths = list()
                    r0s = list()
                    for artery in arteries:
                        lengths.append(params_df[params_df["name"] == artery]["L"].values[0])
                        r0s.append(params_df[params_df["name"] == artery]["r0_in"].values[0])
                    
                    lengths = np.array(lengths)                    
                    r0s = np.array(r0s)

                    for artery in arteries:
                        flow_file = results_dir / f"RUN_{i}{artery}_flow.dat"
                        pressure_file = results_dir / f"RUN_{i}{artery}_pressure.dat"
                        area_file = results_dir / f"RUN_{i}{artery}_area.dat"

                        ########################################
                        # load data and create dataset entry
                        flow = load_dat_file(flow_file)
                        pressure = load_dat_file(pressure_file)
                        area = load_dat_file(area_file)
                        velocity = get_velocity(area, flow)

                        L = params_df[params_df["name"] == artery]["L"].values[0]
                        X = create_X(L)
                        Y = create_Y(pressure, area, velocity)

                        theta = create_theta(artery, lengths, r0s, SV)

                        in_BC = get_inflow_BC(velocity)

                        p0, u0, a0 = get_IC(pressure, area, velocity, L)

                        in_funcs = (in_BC, p0, u0, a0)

                        dataset.append([X, Y, theta, in_funcs])


    np.save(data_dir / f"{dataset_name}.npy", dataset)

if __name__ == "__main__":
    main()