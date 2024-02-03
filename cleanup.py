import os
import sys
from pathlib import Path
import shutil

def main():
    """
    Function iterates thorugh data folders and if in RESULTS_i foldeqr there is a .dat file it moves it to the DATASET folder
    """
    project_dir = Path("/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data")
    dataset_path = project_dir / "DATASET"
    
    
    #/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data/DATA/RESULTS_
    #cd RESULTS_
    for folder in ["DATA", "DATA_2", "DATA_3", "DATA_4"]:
        path = project_dir / folder
        num_result = 61
        for i in range(1200):
            results_dir = path / f"RESULTS_{i}"
            if results_dir.exists():
                if len(list(results_dir.glob("*.dat"))) > 0:
                    if not Path(dataset_path/ f"RESULTS_{num_result}").exists():
                        os.mkdir(dataset_path/ f"RESULTS_{num_result}")
                    for file in results_dir.glob("*.dat"):
                        shutil.move(file, dataset_path/ f"RESULTS_{num_result}")
                    num_result += 1
                else:
                    continue

def move_back():
    """
    Function moves the files back to the original folders
    """

def get_num(file:str):
    """
    Function returns the number from the file name
    with following structure: RUN_iNAME_flow.dat
    """
    return int(file.split("_")[1])
    
    
        
if __name__ == "__main__":
    main()
                    
            





    
