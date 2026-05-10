#!/bin/bash
#SBATCH --job-name=anvil_smoke_3pool
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lz432/anvil_smoke_3pool/smoke_%j.out

set -e
module load intel/17.0.4
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /scratch/lz432/anvil_smoke_3pool
cd /scratch/lz432/anvil_smoke_3pool
python /home/lz432/Anvil/scripts/smoke_3pool_bootstrap.py
