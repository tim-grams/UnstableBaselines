#!/bin/bash

#SBATCH --job-name=run
#SBATCH --cpus-per-task=15
#SBATCH --mem=200G
#SBATCH --mail-user=tim.nico.grams@tu-clausthal.de
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-vram-94gb

source /home/tgrams/miniconda3/etc/profile.d/conda.sh
conda activate textarena0.69
export WANDB_ENTITY=stlm
python scripts/exploration.py