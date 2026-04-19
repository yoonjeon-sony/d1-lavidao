#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=MMADA-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=./slurm-logs/output.%j.log
#SBATCH --error=./slurm-logs/error.%j.log

set -euo pipefail

cd /music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M

export PYTHONPATH="/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export HF_HOME=/home/yoonjeon.kim/.cache/huggingface
export HF_HUB_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub

python -u training/train_interleave.py \
    config=configs/mmada_interleave_thinkmorph_zebracot.yaml \
    training.max_train_steps=2 \
    training.gradient_accumulation_steps=1 \
    experiment.log_every=1 \
    experiment.save_every=100000
