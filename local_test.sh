#!/bin/bash

SOLVER_PATH="/home/wojciech/sv/build/bin/OneDSolver"
OUTPUT_PATH="/home/wojciech/Doppler/PCSS_SIMULATION_MODEL/Test"
TAMPLATES_PATH="/home/wojciech/Doppler/PCSS_SIMULATION_MODEL/Templates"
MAIN_PATH="/home/wojciech/Doppler/PCSS_SIMULATION_MODEL/main.py"

NUM_SIM=10
NUM_WORKERS=4
python3 $MAIN_PATH $SOLVER_PATH $OUTPUT_PATH $NUM_SIM $TAMPLATES_PATH $NUM_WORKERS
