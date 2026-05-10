#!/bin/bash
#SBATCH --job-name=anvil_smoke_tier2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lz432/anvil_smoke_tier2_%j.out

set -e
module load intel/17.0.4
module load cuda/12.1.0 2>/dev/null || true
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python /home/lz432/Anvil/scripts/smoke_tier2_eval.py "$@"
