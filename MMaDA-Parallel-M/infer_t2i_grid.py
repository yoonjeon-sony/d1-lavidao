"""Grid-search t2i_generate across (guidance_scale, timesteps, temperature).

Loads model / VQ / tokenizer once, reuses the 16 ThinkMorph gen samples, and
seeds 1% of the masked image tokens with the input image tokens via
UniversalPrompting. Saves one subdir of PNGs per grid config plus a
manifest jsonl.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

REPO_ROOT = "/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M"
DIFFU_GRPO_ROOT = "/music-home-shared-disk/user/yoonjeon.kim/d1/diffu-grpo"
for p in (REPO_ROOT, DIFFU_GRPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from transformers import AutoTokenizer

from data_utils import get_thinkmorph_interleave_questions
from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting

from infer_all import (
    MAX_TEXT_LEN,
    MODEL_PATH,
    NUM_SAMPLES,
    NUM_VQ_TOKENS,
    T2I_CHUNK,
    VQ_MODEL_NAME,
    build_config,
    run_t2i,
)


GRID_GUIDANCE = [0.0]
GRID_TIMESTEPS = [20, 40]
GRID_TEMPERATURE = [0.0, 0.5, 1.0]
GRID_SEED_RATIO = [0.1, 0.2, 0.3, 0.4, 0.5]
OUT_DIR = os.path.join(REPO_ROOT, "logs", "t2i_grid_seed_sweep")


def safe_id(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config()

    print(f"[load] tokenizer from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=MAX_TEXT_LEN,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=cfg.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    print(f"[load] vq_model {VQ_MODEL_NAME}", flush=True)
    vq_model = MAGVITv2.from_pretrained(VQ_MODEL_NAME, low_cpu_mem_usage=False).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    print(f"[load] model {MODEL_PATH}", flush=True)
    model = MMadaModelLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print("[data] get_thinkmorph_interleave_questions(region_edit=False)", flush=True)
    tm_gen, _, _ = get_thinkmorph_interleave_questions(region_edit=False)
    assert len(tm_gen) >= NUM_SAMPLES, f"tm_gen has only {len(tm_gen)} samples"
    gen_samples = [tm_gen[i] for i in range(NUM_SAMPLES)]

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.jsonl")
    manifest = open(manifest_path, "w", encoding="utf-8")

    configs = list(itertools.product(
        GRID_GUIDANCE, GRID_TIMESTEPS, GRID_TEMPERATURE, GRID_SEED_RATIO
    ))
    print(f"[grid] {len(configs)} configs", flush=True)

    for ci, (gs, ts, temp, sr) in enumerate(configs):
        tag = f"gs{gs}_ts{ts}_temp{temp}_sr{sr}"
        print(f"[{ci+1}/{len(configs)}] {tag}", flush=True)
        cfg.training.guidance_scale = float(gs)
        cfg.training.generation_timesteps = int(ts)
        cfg.training.generation_temperature = float(temp)

        sub_dir = os.path.join(OUT_DIR, tag)
        os.makedirs(sub_dir, exist_ok=True)

        prompts_all: list[str] = []
        images_all = []
        for i in range(0, NUM_SAMPLES, T2I_CHUNK):
            chunk = gen_samples[i : i + T2I_CHUNK]
            p, img, _ = run_t2i(
                model, vq_model, uni_prompting, cfg, chunk, device,
                seed_ratio=float(sr),
            )
            prompts_all.extend(p)
            images_all.extend(img)
            torch.cuda.empty_cache()

        for s, prompt, img in zip(gen_samples, prompts_all, images_all):
            out_img = os.path.join(sub_dir, f"{safe_id(s['sample_id'])}.png")
            img.save(out_img)
            manifest.write(json.dumps({
                "config": tag,
                "guidance_scale": float(gs),
                "generation_timesteps": int(ts),
                "generation_temperature": float(temp),
                "seed_ratio": float(sr),
                "sample_id": s["sample_id"],
                "prompt": prompt,
                "input_image": s["image"],
                "image_gt": s.get("image_gt"),
                "image": out_img,
            }, ensure_ascii=False) + "\n")
        manifest.flush()
        print(f"[done] {tag}", flush=True)

    manifest.close()
    print(f"[done] grid -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
