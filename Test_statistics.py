import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import *
import matplotlib.pyplot as plt


def load_dat_file(filename: str) -> pd.DataFrame:
    """Load a .dat file into a Pandas DataFrame"""
    df = pd.read_csv(filename, sep=" ")
    p = df.to_numpy()
    shape = p.shape
    new_shape = (shape[0], shape[1] - 1)
    p = p[~np.isnan(p)].reshape(new_shape)
    return p


def compute_velocity(df_flow: np.ndarray, df_area: np.ndarray) -> np.ndarray:
    """
    Function to compute velocity from flow and area
    """
    return df_flow / df_area


def compute_MFV(df_velocity: np.ndarray) -> float:
    """
    Function to compute mean flow velocity
    (max + (min *2))/3
    """

    return np.min(df_velocity) + (np.max(df_velocity) - np.min(df_velocity)) / 3
    return (np.max(df_velocity) + (np.min(df_velocity) * 2)) / 3


def compute_MAP(df_pressure: np.ndarray) -> float:
    """
    Function to compute mean arterial pressure

    MAP = (SBP + (2 * DBP)) / 3

    """

    return (np.max(df_pressure) + (2 * np.min(df_pressure))) / 3


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


def compute_PI(df_velocity: np.ndarray, MFV: float) -> float:
    """
    Function to compute pulsatility index

    PI = (Vmax - Vmin) / MFV

    """
    return (np.max(df_velocity) - np.min(df_velocity)) / MFV


def create_velocity_plots(result_folder: str, arteries: List[str]) -> None:
    """
    Function creates aggregated plot af all waveforms for every artery in arteries list
    """
    result_folder = Path(result_folder)
    for artery in arteries:
        fig, ax = plt.subplots()
        for i in range(100):
            folder = result_folder / f"RESULTS_{i}"
            flow = folder / f"TEST_{i}{artery}_flow.dat"
            area = folder / f"TEST_{i}{artery}_area.dat"
            try:
                df_flow = load_dat_file(flow)
                df_area = load_dat_file(area)
                df_velocity = compute_velocity(df_flow, df_area)
                df_velocity = df_velocity[:, -100:]
                ax.plot(np.linspace(0, 1, 100), df_velocity[0, :], label=f"TEST_{i}")
            except:
                continue
        ax.set_title(f"{artery} Velocity")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Velocity [cm/s]")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_folder / "Figures" / f"{artery}_velocity.png")
        plt.close(fig)


def create_area_plots(result_folder: str, arteries: List[str]) -> None:
    """
    Function creates aggregated plot af all waveforms for every artery in arteries list
    """
    result_folder = Path(result_folder)
    for artery in arteries:
        fig, ax = plt.subplots()
        for i in range(100):
            folder = result_folder / f"RESULTS_{i}"
            area = folder / f"TEST_{i}{artery}_area.dat"
            try:
                df_area = load_dat_file(area)
                df_area = df_area[:, -100:]
                ax.plot(np.linspace(0, 1, 100), df_area[0, :], label=f"TEST_{i}")
            except:
                continue
        ax.set_title(f"{artery} Area")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Area [cm2]")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_folder / "Figures" / f"{artery}_area.png")
        plt.close(fig)


def create_pressure_plots(result_folder: str, arteries: List[str]) -> None:
    """
    Function creates aggregated plot af all waveforms for every artery in arteries list
    """
    result_folder = Path(result_folder)
    for artery in arteries:
        fig, ax = plt.subplots()
        for i in range(100):
            folder = result_folder / f"RESULTS_{i}"
            pressure = folder / f"TEST_{i}{artery}_pressure.dat"
            try:
                df_pressure = load_dat_file(pressure)
                df_pressure = df_pressure[:, -100:]
                ax.plot(
                    np.linspace(0, 1, 100),
                    unit_to_mmHg(df_pressure[0, :]),
                    label=f"TEST_{i}",
                )
            except:
                continue
        ax.set_title(f"{artery} Pressure")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Pressure [mmHg]")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_folder / "Figures" / f"{artery}_pressure.png")
        plt.close(fig)


def create_flow_plots(result_folder: str, arteries: List[str]) -> None:
    """
    Function creates aggregated plot of all waveforms for every artery in list
    """
    result_folder = Path(result_folder)
    for artery in arteries:
        fig, ax = plt.subplots()
        for i in range(100):
            folder = result_folder / f"RESULTS_{i}"
            flow = folder / f"TEST_{i}{artery}_flow.dat"
            try:
                df_flow = load_dat_file(flow)
                df_flow = df_flow[:, -100:]
                ax.plot(np.linspace(0, 1, 100), df_flow[0, :], label=f"TEST_{i}")
            except:
                continue
        ax.set_title(f"{artery} Flow")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Flow [ml/s]")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_folder / "Figures" / f"{artery}_flow.png")
        plt.close(fig)


def create_diameter_plots(result_folder: str, arteries: List[str]) -> None:
    """
    Function creates aggregated plot of all waveforms for diameter
    """
    result_folder = Path(result_folder)
    for artery in arteries:
        fig, ax = plt.subplots()
        for i in range(100):
            folder = result_folder / f"RESULTS_{i}"
            area = folder / f"TEST_{i}{artery}_area.dat"
            try:
                df_diameter = load_dat_file(area)
                df_diameter = df_diameter[:, -100:]
                df_diameter = np.sqrt(df_diameter / np.pi) * 2
                ax.plot(np.linspace(0, 1, 100), df_diameter[0, :], label=f"TEST_{i}")
            except:
                continue
        ax.set_title(f"{artery} Diameter")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Diameter [cm]")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_folder / "Figures" / f"{artery}_diameter.png")
        plt.close(fig)


def main(result_folder: str, arteries: List[str]) -> None:
    """
    function iterates over Test folder and its subfloders RESULTS_i,
    loads area pressure and velocity .dat files of arteries in arteries list
    every dat file has following name TEST_{i}{artery}_area.dat etc.

    function computes mean and std of area, MAP and MVF for every artery across different results folders

    and saves it in table format to result_folder can be csv

    """
    result_folder = Path(result_folder)
    result_dict = dict()
    failed = 0
    for artery in arteries:
        result_dict[f"{artery}_MAP"] = list()
        result_dict[f"{artery}_MFV"] = list()
        result_dict[f"{artery}_area"] = list()
        result_dict[f"{artery}_minV"] = list()
        result_dict[f"{artery}_maxV"] = list()
        result_dict[f"{artery}_PI"] = list()
        result_dict[f"{artery}_flow"] = list()
        result_dict[f"{artery}_L"] = list()
        result_dict[f"{artery}_r0_in"] = list()

    for artery in arteries:
        for i in range(100):
            folder = result_folder / f"RESULTS_{i}"
            area = folder / f"TEST_{i}{artery}_area.dat"
            pressure = folder / f"TEST_{i}{artery}_pressure.dat"
            flow = folder / f"TEST_{i}{artery}_flow.dat"
            params = folder / f"PARAMS_{i}.csv"
            try:
                df_area = load_dat_file(area)
                df_pressure = load_dat_file(pressure)
                df_flow = load_dat_file(flow)
                df_velocity = compute_velocity(df_flow, df_area)
                df_area = df_area[:, -100:]
                df_pressure = df_pressure[:, -100:]
                df_velocity = df_velocity[:, -100:]

                params_df = pd.read_csv(params)

                result_dict[f"{artery}_minV"].append(np.min(df_velocity))
                result_dict[f"{artery}_maxV"].append(np.max(df_velocity))
                result_dict[f"{artery}_MAP"].append(
                    unit_to_mmHg(compute_MAP(df_pressure))
                )
                result_dict[f"{artery}_MFV"].append(compute_MFV(df_velocity))
                result_dict[f"{artery}_area"].append(np.mean(df_area))
                result_dict[f"{artery}_PI"].append(
                    compute_PI(df_velocity, compute_MFV(df_velocity))
                )
                result_dict[f"{artery}_flow"].append(np.mean(df_flow[:, -100:]))
                result_dict[f"{artery}_L"].append(
                    params_df[params_df.name == artery].L.to_numpy()
                )
                result_dict[f"{artery}_r0_in"].append(
                    params_df[params_df.name == artery].r0_in.to_numpy()
                )
            except:
                failed += 1
                print(f"Failed to load {artery} in {folder}")
                continue
    # compute means and std form results
    for artery in arteries:
        result_dict[f"{artery}_MAP_mean"] = np.mean(result_dict[f"{artery}_MAP"])
        result_dict[f"{artery}_MAP_std"] = np.std(result_dict[f"{artery}_MAP"])
        result_dict[f"{artery}_MFV_mean"] = np.mean(result_dict[f"{artery}_MFV"])
        result_dict[f"{artery}_MFV_std"] = np.std(result_dict[f"{artery}_MFV"])
        result_dict[f"{artery}_area_mean"] = np.mean(result_dict[f"{artery}_area"])
        result_dict[f"{artery}_area_std"] = np.std(result_dict[f"{artery}_area"])

    # create data frame of only means and stds coll names Map_mean, Map_std etc.
    new_results = dict()
    for artery in arteries:
        new_results[f"{artery}_MAP_mean"] = result_dict[f"{artery}_MAP_mean"]
        new_results[f"{artery}_MAP_std"] = result_dict[f"{artery}_MAP_std"]
        new_results[f"{artery}_MFV_mean"] = result_dict[f"{artery}_MFV_mean"]
        new_results[f"{artery}_MFV_std"] = result_dict[f"{artery}_MFV_std"]
        new_results[f"{artery}_area_mean"] = result_dict[f"{artery}_area_mean"]
        new_results[f"{artery}_area_std"] = result_dict[f"{artery}_area_std"]
        new_results[f"{artery}_minV_mean"] = np.mean(result_dict[f"{artery}_minV"])
        new_results[f"{artery}_minV_std"] = np.std(result_dict[f"{artery}_minV"])
        new_results[f"{artery}_maxV_mean"] = np.mean(result_dict[f"{artery}_maxV"])
        new_results[f"{artery}_maxV_std"] = np.std(result_dict[f"{artery}_maxV"])
        new_results[f"{artery}_PI_mean"] = np.mean(result_dict[f"{artery}_PI"])
        new_results[f"{artery}_PI_std"] = np.std(result_dict[f"{artery}_PI"])
        new_results[f"{artery}_flow_mean"] = np.mean(result_dict[f"{artery}_flow"])
        new_results[f"{artery}_flow_std"] = np.std(result_dict[f"{artery}_flow"])
        new_results[f"{artery}_L_mean"] = np.mean(result_dict[f"{artery}_L"])
        new_results[f"{artery}_L_std"] = np.std(result_dict[f"{artery}_L"])
        new_results[f"{artery}_r0_in_mean"] = np.mean(result_dict[f"{artery}_r0_in"])
        new_results[f"{artery}_r0_in_std"] = np.std(result_dict[f"{artery}_r0_in"])

    # create table from rsults rows = names of artries, cols = mean and std of MAP, MFV and area
    df = pd.DataFrame.from_dict(new_results, orient="index")
    df.to_csv(result_folder / "RESULTS_EPSILON_ALA_TOPO_v3_08_pfv_100_140_tau_035.csv")
    print(f"Failed to load {failed} files")


if __name__ == "__main__":
    arteries = [
        "L_MCA",
        "L_PCA_P1",
        "L_PCA_P2",
        "L_ACA_A1",
        "L_ACA_A2",
        "Basilar",
        "L_int_carotid_II",
        "R_MCA",
        "R_PCA_P1",
        "R_PCA_P2",
        "R_ACA_A1",
        "R_ACA_A2",
        "R_int_carotid_II",
        "L_PcoA",
        "R_PcoA",
        "AcoA",
        "Ascending_aorta",
    ]
    result_folder = "/home/wojciech/Doppler/PCSS_SIMULATION_MODEL/Test"
    main(result_folder, arteries)
    create_velocity_plots(result_folder, arteries)
    create_area_plots(result_folder, arteries)
    create_pressure_plots(result_folder, arteries)
    create_flow_plots(result_folder, arteries)
    create_diameter_plots(result_folder, arteries)
