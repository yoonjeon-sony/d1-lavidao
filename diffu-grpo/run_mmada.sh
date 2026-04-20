#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --account=dgm
#SBATCH --job-name=RL-mmada-diffuGRPO
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

# Activate the project venv so the SLURM job uses the same Python as the
# submitting shell. sbatch does not auto-source user rc files.
source /home/yoonjeon.kim/dLLM-RL/train_sft/.venv/bin/activate

export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-${USER}/${SLURM_JOB_ID:-$$}-${LOCAL_RANK:-0}"
DEBUG="${DEBUG:-0}"

mkdir -p "$TRITON_CACHE_DIR"
chmod 700 "$TRITON_CACHE_DIR"
DATASET="thinkmorph_answer" # Options: thinkmorph_interleave, thinkmorph_answer, thinkmorph_edit

# MMaDA-Parallel has no grounding head — region-edit is always false here.
REGION_EDIT=false

RUN_NAME="${DATASET}-MMaDA-MixCoT"
# MODEL_PATH="/group2/dgm/yoonjeon/MMaDA-8B-MixCoT"
MODEL_PATH="/group2/dgm/yoonjeon/ckpts/sft_MMaDA-PM-thinkmorph_zebracot/checkpoint-4000/unwrapped_model"
OUTPUT_DIR="/scratch2/yoonjeon.kim/rl-mmadaMixCoT-thinkmorph/$RUN_NAME"

# ----------------------------
# Optimizer / scheduler configs
# ----------------------------
LEARNING_RATE=1e-5
ADAM_BETA1=0.9
ADAM_BETA2=0.99
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0
LR_SCHEDULER_TYPE="constant_with_warmup"
WARMUP_RATIO=0.0001

# ----------------------------
# Sampling configs (MMaDA-Parallel layout: max_seq_length=256, num_vq_tokens=1024)
# ----------------------------
TEMPERATURE=0.6
MAX_PROMPT_LENGTH=256
MAX_COMPLETION_LENGTH=256
BLOCK_LENGTH=32
DIFFUSION_STEPS=128
REMASKING="low_confidence"
CFG_SCALE=0.0
RANDOM_MASKING="true"
P_MASK_PROMPT=0.15
GENERATION_BATCH_SIZE=16

# ----------------------------
# GRPO model update configs
# ----------------------------
NUM_ITER=2
BETA=0.04
EPSILON=0.2
SYNC_REF_MODEL="false"
REF_MODEL_SYNC_STEPS=64

if [[ "${DEBUG}" == "1" || "${DEBUG,,}" == "true" ]]; then
    echo "Running in debug mode!!!!"
    export DIFFU_GRPO_DEBUG=1
    export DIFFU_GRPO_STEP0_ASSERT=1
    export DIFFU_GRPO_STEP0_STRICT=1
    BATCH_SIZE=16
    NUM_PROCESSES=4
    NUM_GENERATIONS=2
    PER_DEVICE_BATCH_SIZE=1
    MAX_STEPS=20
    LOGGING_STEPS=1
    SAVE_STEPS=100000
    MAX_COMPLETION_LENGTH=128
    DIFFUSION_STEPS=64
    RETURN_DEBUG_ARTIFACTS=true
    RESUME=false
else
    BATCH_SIZE=64
    NUM_PROCESSES=8
    NUM_GENERATIONS=8
    # Chunk size during backward — each grad-accum micro-step retains
    # PER_DEVICE_BATCH_SIZE autograd graphs (gen + und) simultaneously in
    # _compute_loss. GRAD_ACCUM_STEPS is auto-recomputed below so the
    # effective batch size (BATCH_SIZE * NUM_GENERATIONS = 512) is unchanged.
    PER_DEVICE_BATCH_SIZE=1
    MAX_STEPS="${MAX_STEPS:-}"
    LOGGING_STEPS=1
    SAVE_STEPS=50
    RETURN_DEBUG_ARTIFACTS=false
    RESUME=false
fi

GRAD_ACCUM_STEPS=$(
  echo $((
    BATCH_SIZE * NUM_GENERATIONS
    / NUM_PROCESSES
    / PER_DEVICE_BATCH_SIZE
  ))
)

python -m accelerate.commands.launch \
    --config_file ./diffu-grpo/accelerate.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines 1 \
    --machine_rank 0 \
    diffu-grpo/diffu_grpo_train.py \
    --config diffu-grpo/slurm_scripts/train.yaml \
    --model_type mmada \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    \
    --learning_rate $LEARNING_RATE \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_ratio $WARMUP_RATIO \
    \
    --temperature $TEMPERATURE \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --block_length $BLOCK_LENGTH \
    --diffusion_steps $DIFFUSION_STEPS \
    --remasking $REMASKING \
    --cfg_scale $CFG_SCALE \
    --random_masking $RANDOM_MASKING \
    --p_mask_prompt $P_MASK_PROMPT \
    --generation_batch_size $GENERATION_BATCH_SIZE \
    \
    --num_iterations $NUM_ITER \
    --beta $BETA \
    --epsilon $EPSILON \
    --sync_ref_model $SYNC_REF_MODEL \
    --ref_model_sync_steps $REF_MODEL_SYNC_STEPS \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --report_to wandb \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --region_edit $REGION_EDIT \
    --text_rollout_use_gen_image true
