#!/bin/bash
# Set job requirements
#SBATCH --gpus=1
#SBATCH --job-name={{{JOB_NAME}}}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

# Initialize Conda
source ~/.bashrc
conda activate ~/.conda/envs/jim

# Check the GPU model
nvidia-smi

# Run the script
python run.py --event-id {{{GW_ID}}} --outdir default
