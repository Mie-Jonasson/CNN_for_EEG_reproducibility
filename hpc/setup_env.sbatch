#!/bin/bash

#SBATCH --job-name=pytorch-gpu-condaenv    # Job name
#SBATCH --output=logs/installs_%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=2        # Schedule 8 cores (includes hyperthreading)
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --gres=gpu:rtx2070:1
#SBATCH --partition=scavenge    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END
echo "Running on $(hostname):"
module load Anaconda3
source activate new_braindecode
pip install -r requirements.txt
# requirements.txt leaves an already-installed torch alone. This driver is
# CUDA 12.6, so replace whatever wheel is there with the matching build.
pip install --force-reinstall "torch==2.14.1" \
    --index-url https://download.pytorch.org/whl/cu126

# Verify install
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
