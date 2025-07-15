#!/bin/bash

#SBATCH --job-name=run
#SBATCH --cpus-per-task=15
#SBATCH --mem=250G
#SBATCH --mail-user=tim.nico.grams@tu-clausthal.de
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-vram-48gb

source /home/tgrams/miniconda3/etc/profile.d/conda.sh
conda activate textarena0.69
export WANDB_ENTITY=stlm
python scripts/exploration.py --checkpoints_dir "/work/tgrams/selfplay/UnstableBaselines/outputs/2025-07-13/23-16-35/exploration-Qwen3-1.7B-Base-['ConnectFour-v0-train']-1752441380/checkpoints" --env_id "ConnectFour-v0-train"