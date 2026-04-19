#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=RL-d1-diffuGRPO
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:8
#SBATCH --time=100:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

export DEBUG_FIX_PADDING=1
export NOT_ALWASY_DO_2DPOOL=1
export SKIP_COMPLEMENTARY_MASKING=1
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-${USER}/${SLURM_JOB_ID:-$$}-${LOCAL_RANK:-0}"
DEBUG="${DEBUG:-0}"

mkdir -p "$TRITON_CACHE_DIR"
chmod 700 "$TRITON_CACHE_DIR"
DATASET="thinkmorph_interleave" # thinkmorph_interleave thinkmorph_answer thinkmorph_edit

REGION_EDIT=true
if [ "$REGION_EDIT" = true ]; then
    DATA_NAME="${DATASET}-region-edit"
else
    DATA_NAME="${DATASET}"
fi
RUN_NAME=${DATA_NAME}-LavidaO
MODEL_PATH="/scratch2/yoonjeon.kim/sft_LaViDa-O-thinkmorph_zebracot-step9000"
OUTPUT_DIR=/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/$RUN_NAME
# MODEL_PATH="/group2/dgm/yoonjeon/ckpts/sft_LaViDa-O-thinkmorph_zebracot/checkpoint-9000"
# OUTPUT_DIR="/group2/dgm/yoonjeon/ckpts/rl-lavidao-thinkmorph/$RUN_NAME"

# ----------------------------
# Model initialization configs
# ----------------------------
VERSION="llada"
LOAD_VLM="true"
PREFIX_LM="true"
UNIFIED_GEN="true"
VISION_TOWER="google/siglip-so400m-patch14-384"
MM_TUNABLE_PARTS="mm_vision_tower,mm_mlp_adapter,mm_vision_resampler,mm_language_model,mm_language_model_vision_parms"
MM_VISION_TOWER_LR=2e-6
DUAL_TOWER_LAYERS=16
ENC_USE_IMAGE_BRANCH="true"
MM_VISION_SELECT_LAYER=-2
MM_USE_IM_START_END="false"
MM_USE_IM_PATCH_TOKEN="false"
MM_PATCH_MERGE_TYPE="spatial_unpad"
MM_RESAMPLER_TYPE="spatial_pool"
MM_SPATIAL_POOL_MODE="conv"
MM_SPATIAL_POOL_STRIDE=2
MM_SPATIAL_POOL_OUT_CHANNELS=1152
IMAGE_ASPECT_RATIO="anyres"
IMAGE_GRID_PINPOINTS="[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]"
IMAGE_GEN_SIZE=1024
NUM_GEN_IMAGE_TOKENS=1024
VQVAE="Meissonic/vqvae"

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
# Sampling configs
# ----------------------------
TEMPERATURE=0.6
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=256
BLOCK_LENGTH=32
DIFFUSION_STEPS=128
REMASKING="low_confidence"
CFG_SCALE=0.0
RANDOM_MASKING="true"
P_MASK_PROMPT=0.15
GENERATION_BATCH_SIZE=16
TEXT_ROLLOUT_FORCE_PREFIX_LM="true"

# image-edit rollout defaults
IMAGE_EDIT_SAMPLE_POLICY="multinomial"
IMAGE_EDIT_CONFIDENCE_POLICY="stratified"
IMAGE_EDIT_GUIDANCE_SCALE=0.0
IMAGE_EDIT_RESOLUTION=1024
IMAGE_EDIT_SHIFT=5
IMAGE_EDIT_N_STEPS=64
IMAGE_EDIT_SCHEDULE="shift"
IMAGE_EDIT_ALG_TEMP=5.0
IMAGE_EDIT_DYNAMIC_TEMPERATURE="true"
IMAGE_EDIT_SCHEDULE_TEMP="cosine2"
IMAGE_EDIT_MIN_TEMPERATURE=0.5

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
    # Turn on GRPO loss instrumentation (shape / coef_1 / KL / reduced-loss
    # logging in diffu_grpo_trainer._grpo_loss) so blow-ups are localized.
    export DIFFU_GRPO_DEBUG=1
    export DIFFU_GRPO_STEP0_ASSERT=1
    export DIFFU_GRPO_STEP0_STRICT=1
    BATCH_SIZE=16
    NUM_PROCESSES=8
    NUM_GENERATIONS=2
    PER_DEVICE_BATCH_SIZE=1
    MAX_STEPS=20
    LOGGING_STEPS=1
    SAVE_STEPS=100000
    IMAGE_EDIT_N_STEPS=64
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
    # _compute_loss. Dropping from 4 → 1 gives a ~4× reduction in peak
    # backward activation memory. GRAD_ACCUM_STEPS is auto-recomputed below
    # so the effective batch size (BATCH_SIZE * NUM_GENERATIONS = 512) is
    # unchanged.
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
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    \
    --version $VERSION \
    --load_vlm $LOAD_VLM \
    --prefix_lm $PREFIX_LM \
    --unified_gen $UNIFIED_GEN \
    --vision_tower $VISION_TOWER \
    --mm_tunable_parts "$MM_TUNABLE_PARTS" \
    --mm_vision_tower_lr $MM_VISION_TOWER_LR \
    --dual_tower_layers $DUAL_TOWER_LAYERS \
    --enc_use_image_branch $ENC_USE_IMAGE_BRANCH \
    --mm_vision_select_layer $MM_VISION_SELECT_LAYER \
    --mm_use_im_start_end $MM_USE_IM_START_END \
    --mm_use_im_patch_token $MM_USE_IM_PATCH_TOKEN \
    --mm_patch_merge_type $MM_PATCH_MERGE_TYPE \
    --mm_resampler_type $MM_RESAMPLER_TYPE \
    --mm_spatial_pool_mode $MM_SPATIAL_POOL_MODE \
    --mm_spatial_pool_stride $MM_SPATIAL_POOL_STRIDE \
    --mm_spatial_pool_out_channels $MM_SPATIAL_POOL_OUT_CHANNELS \
    --image_aspect_ratio $IMAGE_ASPECT_RATIO \
    --image_edit_batch_size 4 \
    --image_grid_pinpoints "$IMAGE_GRID_PINPOINTS" \
    --image_gen_size $IMAGE_GEN_SIZE \
    --num_gen_image_tokens $NUM_GEN_IMAGE_TOKENS \
    --vqvae $VQVAE \
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
    --text_rollout_force_prefix_lm $TEXT_ROLLOUT_FORCE_PREFIX_LM \
    --image_edit_sample_policy $IMAGE_EDIT_SAMPLE_POLICY \
    --image_edit_confidence_policy $IMAGE_EDIT_CONFIDENCE_POLICY \
    --image_edit_guidance_scale $IMAGE_EDIT_GUIDANCE_SCALE \
    --image_edit_resolution $IMAGE_EDIT_RESOLUTION \
    --image_edit_shift $IMAGE_EDIT_SHIFT \
    --image_edit_n_steps $IMAGE_EDIT_N_STEPS \
    --image_edit_schedule $IMAGE_EDIT_SCHEDULE \
    --image_edit_alg_temp $IMAGE_EDIT_ALG_TEMP \
    --image_edit_dynamic_temperature $IMAGE_EDIT_DYNAMIC_TEMPERATURE \
    --image_edit_schedule_temp $IMAGE_EDIT_SCHEDULE_TEMP \
    --image_edit_min_temperature $IMAGE_EDIT_MIN_TEMPERATURE \
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