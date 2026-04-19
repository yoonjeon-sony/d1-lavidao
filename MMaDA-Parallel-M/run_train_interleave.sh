#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=MMADA-interleave-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --requeue
#SBATCH --output=./slurm-logs/output.%j.log
#SBATCH --error=./slurm-logs/error.%j.log

set -euo pipefail

REPO_DIR="/music-home-shared-disk/user/yoonjeon.kim/d1/.claude/worktrees/mmada-parallel/MMaDA-Parallel-M"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HF_HOME=/home/yoonjeon.kim/.cache/huggingface
export HF_HUB_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub

CONFIG="${CONFIG:-configs/mmada_interleave_thinkmorph_zebracot.yaml}"
GPUS="${GPUS:-1}"
PORT="${MASTER_PORT:-$((29500 + RANDOM % 10000))}"

TRAIN_SCRIPT="$REPO_DIR/training/train_interleave.py"

DS_CONFIG="$REPO_DIR/configs/ds_zero2.json"

if [[ "$GPUS" -gt 1 ]]; then
    python -u -m accelerate.commands.launch \
        --num_processes "$GPUS" \
        --num_machines 1 \
        --machine_rank 0 \
        --main_process_ip 127.0.0.1 \
        --main_process_port "$PORT" \
        --use_deepspeed \
        --zero_stage 2 \
        --deepspeed_config_file "$DS_CONFIG" \
        --mixed_precision bf16 \
        --gradient_accumulation_steps 16 \
        --gradient_clipping 1.0 \
        "$TRAIN_SCRIPT" config="$CONFIG"
else
    python -u "$TRAIN_SCRIPT" config="$CONFIG"
fi
