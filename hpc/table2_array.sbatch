#!/bin/bash
# One Slurm job that runs the whole Table 2 pipeline on this node.
#
#   mkdir -p logs && sbatch hpc/table2_array.sbatch
#
# Extra arguments are passed to run_all.py, for example:
#   sbatch hpc/table2_array.sbatch --datasets BCIC_IV_2a --low-cuts 4 --subjects 1 --max-epochs 4
#
#SBATCH --job-name=table2-deep4-eeg-cnn-reproduce
#SBATCH --gres=gpu:rtx2070:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/job_%j.out
##SBATCH --partition=scavenge

set -euo pipefail
trap 'printf "Wall clock %d:%02d:%02d\n" $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60))' EXIT

module load Anaconda3
source activate new_braindecode
export MNE_LOGGING_LEVEL=ERROR

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Started $(date)"
echo "Directory $(pwd)"
echo "Python $(command -v python)"
echo "Running: python run_all.py $*"

python run_all.py "$@"

echo "Finished $(date)"
