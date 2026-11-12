#!/bin/bash

#SBATCH --job-name=pytorch-gpu-condaenv    # Job name
#SBATCH --output=logs/installs_%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=2        # Schedule 8 cores (includes hyperthreading)
#SBATCH --time=04:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --gres=gpu:rtx2070:1
#SBATCH --partition=scavenge    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END
echo "Running on $(hostname):"
module load Anaconda3
source activate new_braindecode
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# Verify install
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
