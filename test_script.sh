#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4gb
#SBATCH --time=0:40:00

module load gcc
module load python/3.10.7

export PYTHONPATH=$PYTHONPATH:/home/users/$USER/.local/lib/python3.10/site-packages/

SOLVER_PATH="/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data/build/bin/OneDSolver"
OUTPUT_PATH="/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/scratch/Test"
TAMPLATES_PATH="/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data/PCSS_SIMULATION_MODEL/Templates"
MAIN_PATH="/mnt/storage_4/home/wojciech.kaczmarek/pl0110-01/project_data/PCSS_SIMULATION_MODEL/main.py"

NUM_SIM=2
NUM_WORKERS=1

python3 $MAIN_PATH $SOLVER_PATH $OUTPUT_PATH $NUM_SIM $TAMPLATES_PATH $NUM_WORKERS
