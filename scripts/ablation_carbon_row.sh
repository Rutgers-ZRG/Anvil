#!/bin/bash
#SBATCH --job-name=anvil_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/lz432/carbon_ablation_n200/row_%x_%j.out

# One-row driver. Holds a GPU for the full row (foundation load + Pool A/B
# generation + final ensemble eval). DFT and training children go to other
# partitions but the driver waits.
#
# Submit one per row:
#   sbatch --job-name=ab_A_only   ablation_carbon_row.sh A_only
#   sbatch --job-name=ab_A_plus_B ablation_carbon_row.sh A_plus_B
#   sbatch --job-name=ab_A_plus_C ablation_carbon_row.sh A_plus_C
#   sbatch --job-name=ab_full     ablation_carbon_row.sh full

set -e
module load intel/17.0.4
module load cuda/12.1.0 2>/dev/null || true
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /scratch/lz432/carbon_ablation_n200
ROW_NAME="${1:?row name required (A_only | A_plus_B | A_plus_C | full)}"
echo "=== ablation row: $ROW_NAME ==="
python /home/lz432/Anvil/scripts/ablation_carbon_n200.py "$ROW_NAME"
