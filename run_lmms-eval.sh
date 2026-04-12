#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=lmmseval-SFT2step7000
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

# LIMIT=2 TASKS=chartqa CHAT_MODE=text_gen NUM_GPUS=1 BATCH_SIZE=1 sbatch run_lmms-eval.sh                                         
#   LIMIT=2 TASKS=thinkmorph_visual_search CHAT_MODE=image_gen NUM_GPUS=1 BATCH_SIZE=1 bash run_lmms-eval.sh
# USE_BBOX=False CHAT_MODE="image_gen" TASKS="blink_jigsaw,vstar_bench,cv_bench,chartqa,mmvet,VisPuzzle_direct" BATCH_SIZE=16 sbatch scripts/run_lmms-eval.sh
CHAT_MODE=${CHAT_MODE:-image_gen} # text_gen,image_gen
USE_BBOX=${USE_BBOX:-False}
BATCH_SIZE=${BATCH_SIZE:-16}

# CKPT="/group2/dgm/yoonjeon/LaViDa-O"
# CKPT="/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph-complete/checkpoint-2420" # SFT model 1
CKPT="/group2/dgm/yoonjeon/ckpts/sft_LaViDa-O-thinkmorph_zebracot/checkpoint-7000" # SFT model 2
# CKPT="yjyjyj98/thinkmorph-interleaved_reasoning-multimodal_reward-beta0.04_attnFixed-SFT_NEW-yes_bbox_ckpt100"
# CKPT="yjyjyj98/thinkmorph-interleaved_reasoning-multimodal_reward-beta0.04_attnFixed-SFT_NEW-yes_bbox_ckpt200"
# CKPT=/group2/dgm/yoonjeon/ckpts/rl-lavidao-thinkmorph/thinkmorph-interleaved_reasoning-multimodal_reward-beta0_attnFixed-yes_bbox/checkpoint-50
LIMIT=${LIMIT:-}
NUM_GPUS=${NUM_GPUS:-2}
TASKS=${TASKS:-} 
# "blink_jigsaw,vstar_bench,cv_bench,chartqa,mmvet,VisPuzzle_direct,
# mathvista_testmini_cot,mathverse_testmini,dynamath_reasoning"
# thinkmorph_visual_search,thinkmorph_spatial_navigation,thinkmorph_jigsaw_assembly,thinkmorph_chart_refocus

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
BLOCK_LENGTH=${BLOCK_LENGTH:-256}
STEP_PER_BLOCK=${STEP_PER_BLOCK:-${BLOCK_LENGTH}}
TEMPERATURE=${TEMPERATURE:-0}
SAVE_PARITY_CASES=${SAVE_PARITY_CASES:-0}
PARITY_CASES_ROOT=${PARITY_CASES_ROOT:-DEBUG/parity_text_gen}
PARITY_CASES_MAX_PER_TASK=${PARITY_CASES_MAX_PER_TASK:-4}

export NOT_ALWASY_DO_2DPOOL=1
export DEBUG_PRINT_IMAGE_RES=1
export DEBUG_FIX_PADDING=1

MODEL_NAME=$(basename "$(dirname "$CKPT")")-$(basename "$CKPT")
BASE_DIR="testing"
# BASE_DIR="outputs/eval_generate_logs/"
# BASE_DIR="DEBUG"
OUTPUT_DIR="${BASE_DIR}/${CHAT_MODE}_usebbox${USE_BBOX}_tok${MAX_NEW_TOKENS}_blk${BLOCK_LENGTH}_step${STEP_PER_BLOCK}_t${TEMPERATURE}/${MODEL_NAME}"

if [ "${NUM_GPUS}" -eq 1 ]; then
    LAUNCH_CMD="python"
    LAUNCH_ARGS="-m lmms_eval"
else
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK
    echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NUM_GPUS=${NUM_GPUS}"

    LAUNCH_CMD="accelerate launch --num_machines=1 --machine_rank=0 --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --num_processes=${NUM_GPUS}"
    LAUNCH_ARGS="-m lmms_eval"
fi

echo "Running with TASKS=${TASKS} CKPT=${CKPT} BATCH_SIZE=${BATCH_SIZE} CHAT_MODE=${CHAT_MODE}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" && "${LIMIT,,}" != "none" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi
PARITY_ARGS=()
if [[ "${SAVE_PARITY_CASES}" == "1" || "${SAVE_PARITY_CASES}" == "true" || "${SAVE_PARITY_CASES}" == "True" ]]; then
    PARITY_ARGS=(--save_parity_cases --parity_cases_root "${PARITY_CASES_ROOT}" --parity_cases_max_per_task "${PARITY_CASES_MAX_PER_TASK}")
fi

${LAUNCH_CMD} ${LAUNCH_ARGS} \
    --model llava_llada \
    --model_args pretrained=$CKPT,conv_template=llada,model_name=llava_llada${CHAT_MODE:+,chat_mode=${CHAT_MODE},use_bbox=${USE_BBOX}},gen_img_dir=${OUTPUT_DIR}/gen_imgs\
    --tasks "$TASKS" \
    --batch_size ${BATCH_SIZE} \
    --gen_kwargs prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},temperature=${TEMPERATURE} \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ${OUTPUT_DIR} --verbosity=DEBUG \
    --wandb_args "project=lmms-eval,job_type=eval,name=${EVAL_RUN:-debug}" \
    "${LIMIT_ARGS[@]}" \
    "${PARITY_ARGS[@]}" \
    "$@"
