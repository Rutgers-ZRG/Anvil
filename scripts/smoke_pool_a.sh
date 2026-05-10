#!/bin/bash
#SBATCH --job-name=anvil_smoke_pool_a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lz432/anvil_smoke_pool_a/smoke_%j.out

set -e
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd /scratch/lz432/anvil_smoke_pool_a 2>/dev/null || mkdir -p /scratch/lz432/anvil_smoke_pool_a
cd /scratch/lz432/anvil_smoke_pool_a
python /home/lz432/Anvil/scripts/smoke_pool_a.py
