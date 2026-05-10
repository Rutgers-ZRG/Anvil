#!/bin/bash
#SBATCH --job-name=anvil_smoke_train
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lz432/anvil_smoke_train_%j.out

# Orchestrator runs on a CPU node; it submits 3 GPU child jobs and polls them.
set -e
module load intel/17.0.4
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python /home/lz432/Anvil/scripts/smoke_train_ensemble.py "$@"
