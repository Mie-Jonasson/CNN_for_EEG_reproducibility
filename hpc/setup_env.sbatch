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

echo "Removing existing environment..."
conda env remove -n new_braindecode -y

echo "Creating new environment with Pytorch-CUDA 11.8..."
conda create -y -n new_braindecode python=3.10 \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia \
    --override-channels --strict-channel-priority

echo "Activating new environment..."
conda activate new_braindecode

echo "Installing additional requirements..."
pip install --upgrade-strategy only-if-needed -r requirements.txt

echo "Verifying installation..."
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

echo "Installation complete!"