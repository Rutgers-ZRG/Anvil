#!/bin/bash
#SBATCH --job-name=anvil_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --chdir=/scratch/lz432/carbon_ablation_n200
#SBATCH --output=/scratch/lz432/carbon_ablation_n200/row_%x_%j.out

# amarel3 variant. amarel3 default chdir is /cache/home (cleared post-job),
# so --chdir to a real scratch path is mandatory.

set -e
module load intel/17.0.4 2>/dev/null || true
module load cuda/12.1.0 2>/dev/null || true
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /scratch/lz432/carbon_ablation_n200
ROW_NAME="${1:?row name required (A_only | A_plus_B | A_plus_C | full)}"
echo "=== ablation row: $ROW_NAME (amarel3) ==="
python /home/lz432/Anvil/scripts/ablation_carbon_n200.py "$ROW_NAME" --cluster amarel3
