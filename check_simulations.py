import os
import sys
import numpy as np
import pandas as pd
from typing import *
from pathlib import Path
import matplotlib.pyplot as plt


def check_num_converged(path_folder: str) -> int:
    """
    Check the number of converged simulations in a given directory
    by checking if in each subdirectory RESULTS_i there exist a file ending
    with .dat
    """
    path_folder = Path(path_folder)
    num_converged = 0
    for folder in ["DATA"]:
        path = path_folder / folder
        for i in range(1, 1200):
            results_dir = path / f"RESULTS_{i}"
            if results_dir.exists():
                for file in results_dir.glob("*.dat"):
                    num_converged += 1
                    break
    return num_converged

def load_dat_file(filename: str) -> np.ndarray:
    """
    Function loads results .dat file into pandas dataframe
    """
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

def create_velocity_plots(results_folder: str, arteries: List[str], figures_path: str) -> None:
    """
    Function creates velocity plots for each artery
    """
    results_folder = Path(results_folder)
    figures_path = Path(figures_path)
    for artery in arteries:
        fig, ax = plt.subplots()
        for folder in ["DATA"]:
            path = results_folder / folder
            for i in range(1, 1200):
                results_dir = path / f"RESULTS_{i}"
                flow = results_dir / f"RUN_{i}{artery}_flow.dat"
                area = results_dir / f"RUN_{i}{artery}_area.dat"
                try:
                    df_flow = load_dat_file(results_dir / flow)
                    df_area = load_dat_file(results_dir / area)
                    df_velocity = compute_velocity(df_flow, df_area)
                    df_velocity = df_velocity[:, -100:]
                    ax.plot(np.linspace(0,1,100), df_velocity[0,:])
                except:
                    continue
        ax.set_title(f"{artery} velocity")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [cm/s]")
        ax.grid(True)
        fig.savefig(figures_path / "Velocity" /f"{artery}_velocity.png")
        plt.close(fig)
           
def create_pressure_plots(results_folder: str, arteries: List[str], figures_path: str) -> None:
    """
    Function creates pressure plots for each artery
    """
    results_folder = Path(results_folder)
    figures_path = Path(figures_path)
    for artery in arteries:
        fig, ax = plt.subplots()
        for folder in ["DATA"]:
            path = results_folder / folder
            for i in range(1, 1200):
                results_dir = path / f"RESULTS_{i}"
                pressure = results_dir / f"RUN_{i}{artery}_pressure.dat"
                try:
                    df_pressure = load_dat_file(results_dir / pressure)
                    df_pressure = df_pressure[:, -100:]
                    ax.plot(np.linspace(0,1,100), unit_to_mmHg(df_pressure[0,:]))
                except:
                    continue
        ax.set_title(f"{artery} pressure")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [mmHg]")
        ax.grid(True)
        fig.savefig(figures_path / "Pressure" /f"{artery}_pressure.png")
        plt.close(fig)

def create_area_plots(results_folder: str, arteries: List[str], figures_path: str) -> None:
    """
    Function creates area plots for each artery
    """
    results_folder = Path(results_folder)
    figures_path = Path(figures_path)
    for artery in arteries:
        fig, ax = plt.subplots()
        for folder in ["DATA"]:
            path = results_folder / folder
            for i in range(1, 1200):
                results_dir = path / f"RESULTS_{i}"
                area = results_dir / f"RUN_{i}{artery}_area.dat"
                try:
                    df_area = load_dat_file(results_dir / area)
                    df_area = df_area[:, -100:]
                    ax.plot(np.linspace(0,1,100), df_area[0,:])
                except:
                    continue
        ax.set_title(f"{artery} area")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Area [cm^2]")
        ax.grid(True)
        fig.savefig(figures_path / "Area" /f"{artery}_area.png")
        plt.close(fig)

def create_flow_plots(results_folder: str, arteries: List[str], figures_path: str) -> None:
    """
    Function creates flow plots for each artery
    """
    results_folder = Path(results_folder)
    figures_path = Path(figures_path)
    for artery in arteries:
        fig, ax = plt.subplots()
        for folder in ["DATA"]:
            path = results_folder / folder
            for i in range(1, 1200):
                results_dir = path / f"RESULTS_{i}"
                flow = results_dir / f"RUN_{i}{artery}_flow.dat"
                try:
                    df_flow = load_dat_file(results_dir / flow)
                    df_flow = df_flow[:, -100:]
                    ax.plot(np.linspace(0,1,100), df_flow[0,:])
                except:
                    continue
        ax.set_title(f"{artery} flow")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flow [cm^3/s]")
        ax.grid(True)
        fig.savefig(figures_path / "Flow" /f"{artery}_flow.png")
        plt.close(fig)

def compute_dataset_parameters(results_folder: str, arteries: List[str], figures_path: str) -> None:
    """
    Function computes dataset parametersf low pi itp
    """
    figures_path = Path(figures_path)
    results_folder = Path(results_folder)
    result_dict = dict()
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
        for folder in ["DATA"]:
            path = results_folder / folder
            for i in range(1, 1200):
                results_dir = path / f"RESULTS_{i}"
                flow = results_dir / f"RUN_{i}{artery}_flow.dat"
                area = results_dir / f"RUN_{i}{artery}_area.dat"
                pressure = results_dir / f"RUN_{i}{artery}_pressure.dat"
                params = results_dir / f"PARAMS_{i}.csv"
                try:
                    df_flow = load_dat_file(results_dir / flow)
                    df_area = load_dat_file(results_dir / area)
                    df_pressure = load_dat_file(results_dir / pressure)
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
    df.to_csv(figures_path / "RESULTS_RANDOM.csv")
    

def main():
    #bedziemy w project data i sprawdzic DATA i DATA_2 gdzie sa RESULTS_i
    #potem sprawdzic czy w RESULTS_i jest plik .dat

    # checking pat
    figures_path = "/media/wojciech/4C4A0BCF4A0BB52C/Figures"
    results_folder = "/media/wojciech/4C4A0BCF4A0BB52C"
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

    create_area_plots(results_folder, arteries, figures_path)
    create_velocity_plots(results_folder, arteries, figures_path)
    create_pressure_plots(results_folder, arteries, figures_path)
    create_flow_plots(results_folder, arteries, figures_path)
    compute_dataset_parameters(results_folder, arteries, figures_path)
    num_converged = check_num_converged(results_folder)
    with open(Path(results_folder) / "num_converged.txt", "w") as f:
        f.write(str(num_converged))
        f.close()

    

    
if __name__ == "__main__":
    main()

