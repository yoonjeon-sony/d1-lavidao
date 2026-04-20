#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=MMADA-t2i-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --requeue
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

set -euo pipefail

cd /music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M

export PYTHONPATH="/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M:/music-home-shared-disk/user/yoonjeon.kim/d1/diffu-grpo:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true

python -u infer_t2i_grid.py
