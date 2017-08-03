#!/bin/bash

#SBATCH --job-name=arrayJob
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --partition=sandyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

python ../qubit_simulation.py --task_id $SLURM_ARRAY_TASK_ID
