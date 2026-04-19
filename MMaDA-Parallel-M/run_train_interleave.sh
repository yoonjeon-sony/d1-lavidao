#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=MMADA-interleave-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --requeue
#SBATCH --output=/home/yoonjeon.kim/d1/MMaDA/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/d1/MMaDA/slurm-logs/error.%j.log

set -euo pipefail

cd /music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M

export PYTHONPATH="/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HF_HOME=/home/yoonjeon.kim/.cache/huggingface
export HF_HUB_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub

CONFIG="${CONFIG:-configs/mmada_interleave_thinkmorph_zebracot.yaml}"
GPUS="${GPUS:-1}"

if [[ "$GPUS" -gt 1 ]]; then
    python -u -m accelerate.commands.launch --num_processes "$GPUS" \
        training/train_interleave.py config="$CONFIG"
else
    python -u training/train_interleave.py config="$CONFIG"
fi
