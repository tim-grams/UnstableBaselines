#!/bin/bash

#SBATCH --job-name=run
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --mail-user=tim.nico.grams@tu-clausthal.de
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --partition=gpu-vram-48gb

source /home/tgrams/miniconda3/etc/profile.d/conda.sh
conda activate textarena0.69
export WANDB_ENTITY=stlm
python example2.py