export WANDB_ENTITY="jeoni"
export WANDB_PROJECT="rl-lavidao-thinkmorph"
export MASTER_ADDR=127.0.0.1
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
DATASET="${DATASET:-thinkmorph_edit}" # thinkmorph, thinkmorph_edit, thinkmorph_grounding
DEBUG="${DEBUG:-0}"
if [[ "${DATASET}" == "thinkmorph" ]]; then
    STEPS_PER_GENERATION=1
else
    STEPS_PER_GENERATION=3
fi

USE_BBOX="${USE_BBOX:-true}"
# MODEL_PATH="/scratch2/yoonjeon.kim/sft-lavidao-thinkmorph-edit/"
MODEL_PATH="/scratch2/yoonjeon.kim/sft_LaViDa-O-thinkmorph_zebracot-step3000/"
BETA=0.04
RUN_NAME_BASE="thinkmorph-interleaved_reasoning-multimodal_reward-beta${BETA}_attnFixed-SFT_NEW"
if [[ "${USE_BBOX,,}" == "true" || "${USE_BBOX}" == "1" ]]; then
    BBOX_SUFFIX="yes_bbox"
else
    BBOX_SUFFIX="no_bbox"
fi
RUN_NAME="${RUN_NAME_BASE}-${BBOX_SUFFIX}"
NUM_PROCESSES=8
# Explicitly check for DEBUG being set to string "1" or "true", nothing else will trigger debug mode.
if [[ "${DEBUG}" == "1" || "${DEBUG,,}" == "true" ]]; then
    echo "Running in debug mode!!!!"
    BATCH_SIZE=8
    NUM_GENERATION=2
    PER_DEVICE_BATCH_SIZE=1
    MAX_STEPS=10
    LOGGING_STEPS=1
    SAVE_STEPS=100000
    RETURN_DEBUG_ARTIFACTS=true
    RESUME=false
else
    BATCH_SIZE=64
    NUM_GENERATION=8
    PER_DEVICE_BATCH_SIZE=4
    MAX_STEPS="${MAX_STEPS:-}"
    LOGGING_STEPS="${LOGGING_STEPS:-}"
    SAVE_STEPS="${SAVE_STEPS:-}"
    RETURN_DEBUG_ARTIFACTS=false
    RESUME=false
fi

GRADIENT_ACCUMULATION_STEPS=$(
  echo $(( 
    BATCH_SIZE * NUM_GENERATION 
    / NUM_PROCESSES 
    / PER_DEVICE_BATCH_SIZE
  ))
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${LOGGING_STEPS}" ]]; then
    EXTRA_ARGS+=(--logging_steps "$LOGGING_STEPS")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
    EXTRA_ARGS+=(--save_steps "$SAVE_STEPS")
fi

.venv/bin/python -m accelerate.commands.launch \
    --config_file ./scripts/accelerate_configs/config.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    train.py \
    --config ./scripts/rl_train.yaml \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_generations $NUM_GENERATION \
    --num_iterations 2 \
    --steps_per_generation $STEPS_PER_GENERATION \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --resume_from_checkpoint $RESUME \
    --guidance_scale 0 \
    --use_bbox $USE_BBOX \
    --data_root /home/yoonjeon.kim/dllm-RL/data/ \
    --image_root /scratch2/yoonjeon.kim/data/ \
    --return_debug_artifacts $RETURN_DEBUG_ARTIFACTS \
    --output_dir /scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/$RUN_NAME \
    --beta $BETA \
    "${EXTRA_ARGS[@]}" \
    "$@"
