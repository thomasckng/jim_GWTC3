#!/bin/bash
# Set job requirements
#SBATCH --gpus=1
#SBATCH --job-name={{{JOB_NAME}}}
#SBATCH --output={{{OUTDIR}}}/{{{GW_ID}}}/%x.out
#SBATCH --error={{{OUTDIR}}}/{{{GW_ID}}}/%x.err

# Initialize Conda
source ~/.bashrc
source /home/user/ckng/.venv/jim/bin/activate

# Check the GPU model
nvidia-smi

# Run the script
python run.py --event-id {{{GW_ID}}} --outdir default
