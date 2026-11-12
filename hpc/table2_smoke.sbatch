#!/bin/bash
# Short Slurm job: one BCIC IV 2a subject, 4 Hz band, a few epochs.
#
#   mkdir -p logs && sbatch hpc/table2_smoke.sbatch
#
#SBATCH --job-name=table2-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
##SBATCH --partition=scavenge

set -euo pipefail

module load Anaconda3
conda activate braindecode
export MNE_LOGGING_LEVEL=ERROR

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Started $(date)"
echo "Directory $(pwd)"
echo "Python $(command -v python)"
echo "Running smoke test: BCIC IV 2a, subject 1, 4 Hz, 4 epochs"

python run_all.py --datasets BCIC_IV_2a --low-cuts 4 --subjects 1 \
    --max-epochs 4 --max-increase-epochs 2 "$@"

echo "Finished $(date)"
