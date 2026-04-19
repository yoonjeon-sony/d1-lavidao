#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=MMADA-interleave-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=100:00:00
#SBATCH --requeue
#SBATCH --output=./slurm-logs/output.%j.log
#SBATCH --error=./slurm-logs/error.%j.log

set -euo pipefail

cd /music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M

export PYTHONPATH="/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HF_HOME=/home/yoonjeon.kim/.cache/huggingface
export HF_HUB_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub
export WANDB_MODE=offline

python -u -m accelerate.commands.launch --num_processes 8 \
    training/train_interleave.py \
    config=configs/mmada_interleave_thinkmorph_zebracot.yaml \
    training.batch_size=64 \
    training.max_train_steps=5 \
    experiment.log_every=1 \
    experiment.save_every=100000
