#!/bin/bash
#SBATCH --job-name=anvil_ablation
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/lz432/carbon_ablation_n200/daemon_%j.out

# Long-running Anvil orchestrator daemon for the carbon ablation feasibility
# round. CPU node — submits VASP + GPU training children to other partitions.
#
# Total budget: 4 rows × 200 DFT (r2SCAN+rVV10 carbon, ~1 hr each) +
#               4 × 3 = 12 GPU training jobs.
# Wallclock: 12-24 hr realistic; 48 hr cap for safety.

set -e
module load intel/17.0.4
eval "$(/home/lz432/miniconda3/bin/conda shell.bash hook)"
conda activate nequip
export PYTHONPATH="/home/lz432/Anvil:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /scratch/lz432/carbon_ablation_n200
python /home/lz432/Anvil/scripts/ablation_carbon_n200.py
