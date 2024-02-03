import os
import sys
import numpy as np
import pandas as pd
from typing import *
from pathlib import Path
from Windkessel.windkessel import compute_windkessel
from Sampling.sampling import sample
from Scripting.scripting import create_script
import time


# main function for creating dataset


def main(
    solver_path: str,
    output_path: str,
    num_samples: int,
    templates: str,
    NUM_WORKERS: int,
) -> None:
    """
    Main function for creating dataset, running simulation,
    sampling parameters and saving results
    """
    output_path = Path(output_path)
    templates = Path(templates)
    for i in range(num_samples):
        # create results folder in outptu path if path does not exist
        if not os.path.exists(output_path / f"RESULTS_{i}"):
            os.makedirs(output_path / f"RESULTS_{i}")

        os.chdir(output_path / f"RESULTS_{i}")

        # create df and df_joints from template
        df = pd.read_csv(templates / f"TEMPLATE_1.csv")
        df_joints = pd.read_csv(templates / f"TEMPLATE_1_TOPO.csv")

        # sample parameters
        df, flow = sample(df)
        df = compute_windkessel(df)
        create_script(df, df_joints, f"SCRIPT_{i}", "TEST", flow)  # zmienic potem
        os.system(f"./{solver_path} SCRIPT_{i}.txt")


from multiprocessing import Pool


def process_sample(sample_index, solver_path, output_path, templates):
    output_folder = output_path / f"RESULTS_{sample_index}"

    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    os.chdir(output_folder)

    df = pd.read_csv(templates / f"TEMPLATE_1.csv")
    df_joints = pd.read_csv(templates / f"TEMPLATE_1_TOPO.csv")

    df, flow, SV = sample(df)
    # create txt file with Sv and save
    with open("SV.txt", "w") as f:
        f.write(str(SV))
        f.close()
    df = compute_windkessel(df, SV=SV * 1e-6)
    create_script(
        df,
        df_joints,
        f"SCRIPT_{sample_index}",
        f"RUN_{sample_index}",
        flow,
        olufsen=True,
    )
    df.to_csv(f"PARAMS_{sample_index}.csv")
    os.system(f"{solver_path} SCRIPT_{sample_index}.txt")
    with open("JOB_DONE.txt", "w") as f:
        f.write("DONE")
        f.close()



def main_2(
    solver_path: str,
    output_path: str,
    num_samples: int,
    templates: str,
    NUM_WORKERS: int,
) -> None:
    output_path = Path(output_path)
    templates = Path(templates)
    st = time.time()
    with Pool(NUM_WORKERS) as pool:
        pool.starmap(
            process_sample,
            [(i, solver_path, output_path, templates) for i in range(num_samples)],
        )
    print(f"Total time: {time.time() - st}")


if __name__ == "__main__":
    main_2(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
