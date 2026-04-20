#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=MMADA-interleave-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=100:00:00
#SBATCH --requeue
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export HF_HOME=/home/yoonjeon.kim/.cache/huggingface
export HF_HUB_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/yoonjeon.kim/.cache/huggingface/hub

CONFIG="${CONFIG:-configs/mmada_interleave_thinkmorph_zebracot.yaml}"
GPUS="${GPUS:-1}"
PORT="${MASTER_PORT:-$((29500 + RANDOM % 10000))}"

TRAIN_SCRIPT="./training/train_interleave.py"

DS_CONFIG="./configs/ds_zero2.json"

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
