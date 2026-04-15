#!/bin/bash
#SBATCH --partition=dgm
#SBATCH --account=dgm
#SBATCH --job-name=lmms-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task per GPU
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00               # Max time
#SBATCH --requeue                     # allow requeue if preempted
#SBATCH --output=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/output.%j.log
#SBATCH --error=/home/yoonjeon.kim/dLLM-RL/train_sft/slurm-logs/error.%j.log

CKPT=/group2/dgm/yoonjeon/LaViDa-O
# CKPT="/scratch2/yoonjeon.kim/sft_LaViDa-O-thinkmorph_zebracot-step9000"
# CKPT="/scratch2/yoonjeon.kim/LaViDa-O"
LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
set -x
LIMIT=8

export TASKS=${TASKS:-"mme"}
# ,mmmu_val,mmbench_en_dev,textvqa_val,docvqa_val,chartqa,infovqa_val,scienceqa_full,ai2d,mathverse_testmini_vision_dominant,mathvista_testmini_format
export NOT_ALWASY_DO_2DPOOL=1
export DEBUG_PRINT_IMAGE_RES=1
export DEBUG_FIX_PADDING=1 # new runs must have this !!!!!!!!!!!!
export NCCL_P2P_DISABLE=1
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
BLOCK_LENGTH=${BLOCK_LENGTH:-256}
STEP_PER_BLOCK=${STEP_PER_BLOCK:-${BLOCK_LENGTH}}
TEMPERATURE=${TEMPERATURE:-0}

MODEL_NAME=$(basename "$(dirname "$CKPT")")-$(basename "$CKPT")
BASE_DIR="TEST"
OUTPUT_DIR="${BASE_DIR}/tok${MAX_NEW_TOKENS}_blk${BLOCK_LENGTH}_step${STEP_PER_BLOCK}_t${TEMPERATURE}/${MODEL_NAME}"

run_eval() {
    local bs=$1
    local out_dir=$2
    accelerate launch --num_processes=2 \
        --num_machines=1 \
        -m lmms_eval \
        --model llava_llada \
        --model_args pretrained=$CKPT,conv_template=llada,model_name=llava_llada \
        --tasks $TASKS \
        --batch_size $bs \
        --limit $LIMIT \
        --gen_kwargs prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},temperature=${TEMPERATURE} \
        --log_samples \
        --log_samples_suffix llava_llada \
        --output_path "$out_dir" --verbosity=DEBUG \
        --wandb_args=project=lmms-eval,job_type=eval \
        "${@:3}"
}

if [ "${DEBUG:-0}" = "1" ]; then
    OUT_BS1="${OUTPUT_DIR}_bs1"
    OUT_BS8="${OUTPUT_DIR}_bs8"
    rm -rf "$OUT_BS1" "$OUT_BS8"
    run_eval 1 "$OUT_BS1" "${@:2}"
    run_eval 8 "$OUT_BS8" "${@:2}"

    python - <<PY
import glob, json, os, sys
def load_resps(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*_samples_*.jsonl"), recursive=True))
    by_task = {}
    for f in files:
        task = os.path.basename(f).split("_samples_", 1)[1].rsplit(".jsonl", 1)[0]
        rows = [json.loads(l) for l in open(f)]
        rows.sort(key=lambda r: r.get("doc_id"))
        by_task[task] = [(r.get("doc_id"), r.get("resps"), r.get("filtered_resps")) for r in rows]
    return by_task

a = load_resps("$OUT_BS1")
b = load_resps("$OUT_BS8")
print(f"[PARITY] tasks bs1={list(a)} bs8={list(b)}")
all_match = True
for task in sorted(set(a) | set(b)):
    ra, rb = a.get(task, []), b.get(task, [])
    n = min(len(ra), len(rb))
    mism = []
    for i in range(n):
        if ra[i] != rb[i]:
            mism.append((ra[i][0], ra[i][1], rb[i][1]))
    ok = (len(ra) == len(rb)) and not mism
    all_match &= ok
    print(f"[PARITY] task={task} bs1={len(ra)} bs8={len(rb)} match={ok} mismatches={len(mism)}")
    for doc_id, ra_resp, rb_resp in mism[:5]:
        print(f"  doc_id={doc_id}\n    bs1={ra_resp}\n    bs8={rb_resp}")
print(f"[PARITY] OVERALL {'PASS' if all_match else 'FAIL'}")
sys.exit(0 if all_match else 1)
PY
else
    run_eval 1 "$OUTPUT_DIR" "${@:2}"
fi
