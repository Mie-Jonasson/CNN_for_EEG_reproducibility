#!/bin/bash
# Slurm job array running all 46 Table 2 trainings, one (dataset, band,
# subject) per task: indices 0-17 are BCIC IV 2a, 18-45 HGD.
#
# Before submitting, from the project folder on the login node:
#   1. python -m venv .venv && .venv/bin/pip install -r requirements.txt
#   2. Download the data (compute nodes may have no internet):
#      .venv/bin/python -c "from data_loading import download; download('BCIC_IV_2a'); download('HGD')"
#   3. mkdir -p logs && sbatch hpc/table2_array.sbatch
# Afterwards: .venv/bin/python reproduce_table2_deepconvnet.py summarize
#
# TODO: set partition/account and module loads for your cluster.
#SBATCH --job-name=table2-deep4
#SBATCH --array=0-45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --partition=<gpu partition>
##SBATCH --account=<account>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load <python/cuda modules>
source .venv/bin/activate
export MNE_LOGGING_LEVEL=ERROR

python reproduce_table2_deepconvnet.py train --job-index "$SLURM_ARRAY_TASK_ID"
