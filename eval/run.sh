#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=lmms-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

set -eu

LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-4}"
BLOCK_LENGTH="${BLOCK_LENGTH:-128}"
STEP_PER_BLOCK="${STEP_PER_BLOCK:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
DO_IMAGE_ROLLOUT="${DO_IMAGE_ROLLOUT:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

export NCCL_P2P_DISABLE=1
export NOT_ALWASY_DO_2DPOOL=1
export DEBUG_PRINT_IMAGE_RES=1
export DEBUG_FIX_PADDING=1

# declare -a CKPTS=(
#   "/group2/dgm/yoonjeon/ckpts/sft-lavidao-thinkmorph-edit"
#   "/group2/dgm/yoonjeon/LaViDa-O" # mmmu_val,vstar_bench,blink,chartqa done => add scienceqa_img,cv_bench,VisualPuzzles_cot,mmstar
#   "/group2/dgm/yoonjeon/ckpts/sft_LaViDa-O-thinkmorph_zebracot/checkpoint-9000" # mmmu_val,vstar_bench,blink,chartqa done => add scienceqa_img,cv_bench,VisualPuzzles_cot,mmstar
#   # "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-LavidaO/checkpoint-50" # add mmmu_val(900),vstar_bench(191),blink(1,901),chartqa(2,500),scienceqa_img(2,017),cv_bench(2,638),VisualPuzzles_cot(1,168),mmstar(1,500)
#   "yjyjyj98/thinkmorph_interleave-Unified-LavidaO-ckpt50" # add mmmu_val,vstar_bench,blink,chartqa,scienceqa_img,cv_bench,VisualPuzzles_cot,mmstar
#   # "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_edit-LavidaO/checkpoint-50"
#   # "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_answer-LavidaO/checkpoint-50"
# )
declare -a CKPTS=(
  "/scratch2/yoonjeon.kim/LaViDa-O" # mmmu_val,vstar_bench,blink,chartqa done => add scienceqa_img,cv_bench,VisualPuzzles_cot,mmstar
  # "/scratch2/yoonjeon.kim/sft_LaViDa-O-thinkmorph_zebracot-step9000" # mmmu_val,vstar_bench,blink,chartqa done => add scienceqa_img,cv_bench,VisualPuzzles_cot,mmstar
  "/scratch2/yoonjeon.kim/sft-lavidao-thinkmorph-edit/" # mmmu_val,blink,VisualPuzzles_cot,mmstar
  "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-LavidaO/checkpoint-50" # add mmmu_val(900),vstar_bench(191),blink(1,901),chartqa(2,500),scienceqa_img(2,017),cv_bench(2,638),VisualPuzzles_cot(1,168),mmstar(1,500)
  "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-Unified-LavidaO/checkpoint-50" # add mmmu_val,vstar_bench,blink,chartqa,scienceqa_img,cv_bench,VisualPuzzles_cot,mmstar
  # "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_edit-LavidaO/checkpoint-50"
  # "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_answer-LavidaO/checkpoint-50"
)

model_name_of() {
  local ckpt="$1"
  echo "$(basename "$(dirname "$ckpt")")-$(basename "$ckpt")"
}

out_dir_of() {
  local ckpt="$1"
  local tok="$2"
  local mn
  mn="$(model_name_of "$ckpt")"
  local suf=""
  if [[ "$DO_IMAGE_ROLLOUT" == "true" ]]; then
    suf="_image"
  fi
  echo "TEST/tok${tok}_blk${BLOCK_LENGTH}_step${STEP_PER_BLOCK}_t${TEMPERATURE}/${mn}${suf}"
}

filter_tasks() {
  local out_dir="$1"; shift
  local kept=()
  for t in "$@"; do
    # Match exact "<task>.jsonl" OR group-style "<task>_*.jsonl" (e.g. blink_art_style).
    if find "$out_dir" -type f \( -name "*_samples_${t}.jsonl" -o -name "*_samples_${t}_*.jsonl" \) 2>/dev/null | grep -q .; then
      echo "[skip] ${out_dir} / ${t}" >&2
    else
      kept+=("$t")
    fi
  done
  (IFS=,; echo "${kept[*]:-}")
}

run_shard() {
  local ckpt="$1"
  local tok="$2"
  shift 2
  local tasks=("$@")
  # When max_new_tokens < block_length the model asserts gen_length % block_length == 0.
  # Rule: if MAX_NEW_TOKENS < 128, set block_length = MAX_NEW_TOKENS and step_per_block = MAX_NEW_TOKENS / 2.
  local blk="$BLOCK_LENGTH" steps="$STEP_PER_BLOCK"
  if (( MAX_NEW_TOKENS < 128 )); then
    blk="$MAX_NEW_TOKENS"
    steps=$(( MAX_NEW_TOKENS / 2 ))
  fi
  local out_dir
  out_dir=$(BLOCK_LENGTH="$blk" STEP_PER_BLOCK="$steps" out_dir_of "$ckpt" "$tok")
  mkdir -p "$out_dir"
  local remaining
  remaining=$(filter_tasks "$out_dir" "${tasks[@]}")
  if [[ -z "$remaining" ]]; then
    echo "[done] GPU=${GPU_ID:-0} ckpt=${ckpt} tok=${tok} tasks=[${tasks[*]}] nothing to run"
    return 0
  fi
  echo "[run ] GPU=${GPU_ID:-0} ckpt=${ckpt} tok=${tok} blk=${blk} steps=${steps} tasks=${remaining} out=${out_dir}"
  local port=$((20000 + RANDOM % 10000))
  accelerate launch --num_processes=1 \
    --num_machines=1 \
    --main_process_port "$port" \
    -m lmms_eval \
    --model llava_llada \
    --system_instruction "Think step-by-step and answer the question. " \
    --model_args "pretrained=${ckpt},conv_template=llada,model_name=llava_llada,attn_implementation=flash_attention_2" \
    --tasks "$remaining" \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --gen_kwargs "prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},block_length=${blk},step_per_block=${steps},temperature=${TEMPERATURE},do_image_rollout=${DO_IMAGE_ROLLOUT},image_edit_resolution=512,image_edit_n_steps=32" \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path "$out_dir" --verbosity=DEBUG \
    --wandb_args "project=lmms-eval,job_type=eval"
}

run_shard_defaults_one() {
  # Args: ckpt, block_length, task
  local ckpt="$1" blk="$2" task="$3"
  # Rule: if blk < 128 (matches max_new_tokens since blk is set to task default in this code path),
  # use step_per_block = blk / 2; otherwise use STEP_PER_BLOCK env (default 64).
  local steps="$STEP_PER_BLOCK"
  if (( blk < 128 )); then
    steps=$(( blk / 2 ))
  fi
  local mn
  mn="$(model_name_of "$ckpt")"
  local suf=""
  if [[ "$DO_IMAGE_ROLLOUT" == "true" ]]; then
    suf="_image"
  fi
  local out_dir="TEST/tokdefault_noSI_blk${blk}_step${steps}_t${TEMPERATURE}/${mn}${suf}"
  mkdir -p "$out_dir"
  local remaining
  remaining=$(filter_tasks "$out_dir" "$task")
  if [[ -z "$remaining" ]]; then
    echo "[done-def] ckpt=${ckpt} task=${task} blk=${blk} steps=${steps} nothing to run"
    return 0
  fi
  echo "[run-def] ckpt=${ckpt} task=${task} blk=${blk} steps=${steps} out=${out_dir}"
  local port=$((20000 + RANDOM % 10000))
  accelerate launch --num_processes=1 \
    --num_machines=1 \
    --main_process_port "$port" \
    -m lmms_eval \
    --model llava_llada \
    --model_args "pretrained=${ckpt},conv_template=llada,model_name=llava_llada,attn_implementation=flash_attention_2" \
    --tasks "$task" \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --gen_kwargs "prefix_lm=True,block_length=${blk},step_per_block=${steps},temperature=${TEMPERATURE},do_image_rollout=${DO_IMAGE_ROLLOUT},image_edit_resolution=512,image_edit_n_steps=32" \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path "$out_dir" --verbosity=DEBUG \
    --wandb_args "project=lmms-eval,job_type=eval"
}

run_cv_bench_long() {
  # cv_bench with explicit max_new_tokens=2048, no system_instruction.
  local ckpt="$1"
  local blk=128
  local mnt=2048
  local mn
  mn="$(model_name_of "$ckpt")"
  local suf=""
  if [[ "$DO_IMAGE_ROLLOUT" == "true" ]]; then
    suf="_image"
  fi
  local out_dir="TEST/tok${mnt}_noSI_blk${blk}_step${STEP_PER_BLOCK}_t${TEMPERATURE}/${mn}${suf}"
  mkdir -p "$out_dir"
  local remaining
  remaining=$(filter_tasks "$out_dir" "cv_bench")
  if [[ -z "$remaining" ]]; then
    echo "[done-cv2048] ckpt=${ckpt} nothing to run"
    return 0
  fi
  echo "[run-cv2048] ckpt=${ckpt} mnt=${mnt} blk=${blk} out=${out_dir}"
  local port=$((20000 + RANDOM % 10000))
  accelerate launch --num_processes=1 \
    --num_machines=1 \
    --main_process_port "$port" \
    -m lmms_eval \
    --model llava_llada \
    --model_args "pretrained=${ckpt},conv_template=llada,model_name=llava_llada,attn_implementation=flash_attention_2" \
    --tasks cv_bench \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --gen_kwargs "prefix_lm=True,max_new_tokens=${mnt},block_length=${blk},step_per_block=${STEP_PER_BLOCK},temperature=${TEMPERATURE},do_image_rollout=${DO_IMAGE_ROLLOUT},image_edit_resolution=512,image_edit_n_steps=32" \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path "$out_dir" --verbosity=DEBUG \
    --wandb_args "project=lmms-eval,job_type=eval"
}

CKPT_IDX="${CKPT_IDX:-0}"
CKPT="${CKPTS[$CKPT_IDX]}"
MODE="${MODE:-standard}"
if [[ "$MODE" == "defaults" ]]; then
  # Per-task block_length must divide max_new_tokens (chartqa=16, cv_bench=1024).
  run_shard_defaults_one "$CKPT" 16  chartqa
  run_shard_defaults_one "$CKPT" 128 cv_bench
elif [[ "$MODE" == "cv_2048" ]]; then
  run_cv_bench_long "$CKPT"
else
  MAX_NEW_TOKENS=128 run_shard "$CKPT" 128 mmmu_val vstar_bench blink ai2d_lite
  MAX_NEW_TOKENS=256 run_shard "$CKPT" 256 scienceqa_img cv_bench VisualPuzzles_cot mmstar chartqa
fi