"""Batched inference over three generate modes for MMaDA-Parallel-M.

Runs, over the same 16 aligned ThinkMorph samples:
  * t2i_generate   : tm_gen.instruction -> image
  * mmu_generate   : tm_und.image + tm_und.instruction -> answer text
  * interleave_generate : tm_gen.image + tm_gen.instruction -> (image, text)

All three run with batched inputs (padded as needed).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

REPO_ROOT = "/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M"
DIFFU_GRPO_ROOT = "/music-home-shared-disk/user/yoonjeon.kim/d1/diffu-grpo"
for p in (REPO_ROOT, DIFFU_GRPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from models import MAGVITv2, MMadaModelLM, get_mask_schedule
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform_squash
from training.interleave_utils import prepare_inputs_and_labels_for_interleave_data
from data_utils import get_thinkmorph_interleave_questions


NUM_SAMPLES = 16
T2I_CHUNK = 16
MMU_CHUNK = 4
INTERLEAVE_CHUNK = 2
LOGPROB_CHUNK = 2
LOGPROB_MASK_PROB = 0.15
# MODEL_PATH = "tyfeld/MMaDA-Parallel-M"
MODEL_PATH = "/group2/dgm/yoonjeon/ckpts/sft_MMaDA-PM-thinkmorph_zebracot/checkpoint-4000/unwrapped_model"
VQ_MODEL_NAME = "showlab/magvitv2"
RESOLUTION = 512
NUM_VQ_TOKENS = 1024
CODEBOOK_SIZE = 8192
MAX_TEXT_LEN = 256
MAX_SEQ_LENGTH = 256
MASK_TOKEN_ID = 126336

RESERVED_TOKENS = {
    "<|soi|>": 126084,
    "<|eoi|>": 126085,
    "<|sov|>": 126086,
    "<|eov|>": 126087,
    "<|t2i|>": 126088,
    "<|mmu|>": 126089,
    "<|t2v|>": 126090,
    "<|v2v|>": 126091,
    "<|lvg|>": 126092,
    "[iPAD]": 126093,
    "<|r2i|>": 126094,
    "<|interleave|>": 126095,
}


def build_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {"mmada": {"num_vq_tokens": NUM_VQ_TOKENS, "codebook_size": CODEBOOK_SIZE}},
            "dataset": {"preprocessing": {"max_seq_length": MAX_SEQ_LENGTH, "resolution": RESOLUTION}},
            "training": {
                "guidance_scale": 0,
                "generation_timesteps": 20,
                "cond_dropout_prob": 0.1,
                "generation_temperature": 0.2,
                "noise_type": "mask",
            },
            "mask_schedule": {"schedule": "cosine"},
        }
    )


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return image_transform_squash(img, resolution=RESOLUTION).to(device)


def decode_vq(vq_model: MAGVITv2, token_ids: torch.Tensor) -> list[Image.Image]:
    token_ids = torch.clamp(token_ids, 0, CODEBOOK_SIZE - 1).to(torch.long)
    images = vq_model.decode_code(token_ids)
    images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
    images = (images * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(img) for img in images]


def run_t2i(model, vq_model, uni_prompting, cfg, samples, device, seed_ratio: float = 0.0):
    prompts = [s["instruction"] for s in samples]
    batch_size = len(prompts)
    image_tokens = torch.full(
        (batch_size, NUM_VQ_TOKENS), MASK_TOKEN_ID, dtype=torch.long, device=device
    )
    ref_image_ids = None
    if seed_ratio > 0:
        pixel_batch = torch.stack([load_image_tensor(s["image"], device) for s in samples], 0)
        ref_image_ids = vq_model.get_code(pixel_batch) + len(uni_prompting.text_tokenizer)
    input_ids, attention_mask = uni_prompting(
        (prompts, image_tokens, ref_image_ids, seed_ratio), "t2i_gen"
    )

    guidance_scale = cfg.training.guidance_scale
    if guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(
            ([""] * batch_size, image_tokens, ref_image_ids, seed_ratio), "t2i_gen"
        )
    else:
        uncond_input_ids = None
        uncond_attention_mask = None

    schedule = get_mask_schedule(cfg.mask_schedule.schedule)

    with torch.no_grad():
        gen_ids = model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=guidance_scale,
            temperature=cfg.training.generation_temperature,
            timesteps=cfg.training.generation_timesteps,
            noise_schedule=schedule,
            noise_type=cfg.training.noise_type,
            seq_len=NUM_VQ_TOKENS,
            uni_prompting=uni_prompting,
            config=cfg,
        )
    gen_ids = torch.clamp(gen_ids, 0, CODEBOOK_SIZE - 1).to(torch.long)
    return prompts, decode_vq(vq_model, gen_ids), gen_ids


def run_mmu(model, vq_model, uni_prompting, tokenizer, samples, device):
    pixel_batch = torch.stack([load_image_tensor(s["image"], device) for s in samples], 0)
    image_tokens_shifted = vq_model.get_code(pixel_batch) + len(tokenizer)
    image_tokens = image_tokens_shifted

    text_token_lists = []
    for s in samples:
        messages = [
            {"role": m["role"], "content": m["content"].replace("<image>\n", "").replace("<image>", "")}
            for m in s["prompt"]
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        text_token_lists.append(ids)

    max_text = max(len(ids) for ids in text_token_lists)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded = [[pad_id] * (max_text - len(ids)) + ids for ids in text_token_lists]
    text_batch = torch.tensor(padded, dtype=torch.long, device=device)

    B = len(samples)
    mmu_tok = int(uni_prompting.sptids_dict["<|mmu|>"])
    soi = int(uni_prompting.sptids_dict["<|soi|>"])
    eoi = int(uni_prompting.sptids_dict["<|eoi|>"])

    input_ids = torch.cat(
        [
            torch.full((B, 1), mmu_tok, dtype=torch.long, device=device),
            torch.full((B, 1), soi, dtype=torch.long, device=device),
            image_tokens,
            torch.full((B, 1), eoi, dtype=torch.long, device=device),
            text_batch,
        ],
        dim=1,
    )
    prefix_len = 3 + image_tokens.shape[1]

    max_new_tokens = MAX_SEQ_LENGTH
    prefix_mask = torch.ones((B, prefix_len), dtype=torch.long, device=device)
    text_mask = (text_batch != pad_id).long()
    gen_mask = torch.ones((B, max_new_tokens), dtype=torch.long, device=device)
    attention_mask = torch.cat([prefix_mask, text_mask, gen_mask], dim=1)

    steps = max(1, max_new_tokens // 2)
    block_length = max(1, max_new_tokens // 4)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.mmu_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            steps=steps,
            block_length=block_length,
            attention_mask=attention_mask,
            mask_id=MASK_TOKEN_ID,
        )

    gen_ids = output_ids[:, input_ids.shape[1]:]
    responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return responses, image_tokens_shifted


def run_interleave(model, vq_model, uni_prompting, tokenizer, cfg, gen_samples, device):
    pixel_batch = torch.stack([load_image_tensor(s["image"], device) for s in gen_samples], 0)
    image_tokens_shifted = vq_model.get_code(pixel_batch) + len(tokenizer)
    image_tokens = image_tokens_shifted
    uncond_image_tokens = torch.zeros_like(image_tokens)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    cond_lists = []
    uncond_lists = []
    for s in gen_samples:
        ids = tokenizer(s["instruction"])["input_ids"]
        if len(ids) == 0 or ids[0] != bos:
            ids = [bos] + list(ids)
        else:
            ids = list(ids)
        ids = ids + [eos]
        cond_lists.append(ids)

        u = tokenizer("")["input_ids"]
        if len(u) == 0 or u[0] != bos:
            u = [bos] + list(u)
        else:
            u = list(u)
        u = u + [eos]
        uncond_lists.append(u)

    max_len = max(len(ids) for ids in cond_lists)
    for i in range(len(gen_samples)):
        cond_lists[i] = cond_lists[i] + [eos] * (max_len - len(cond_lists[i]))
        uncond_lists[i] = uncond_lists[i] + [eos] * (max_len - len(uncond_lists[i]))

    text_ids = torch.tensor(cond_lists, dtype=torch.long, device=device)
    uncond_text_ids = torch.tensor(uncond_lists, dtype=torch.long, device=device)

    B = text_ids.shape[0]
    interleave_col = torch.full((B, 1), RESERVED_TOKENS["<|interleave|>"], dtype=torch.long, device=device)
    soi_col = torch.full((B, 1), RESERVED_TOKENS["<|soi|>"], dtype=torch.long, device=device)
    eoi_col = torch.full((B, 1), RESERVED_TOKENS["<|eoi|>"], dtype=torch.long, device=device)

    input_ids = torch.cat([interleave_col, soi_col, image_tokens, eoi_col, text_ids], dim=1)
    uncond_input_ids = torch.cat(
        [interleave_col, soi_col, uncond_image_tokens, eoi_col, uncond_text_ids], dim=1
    )

    with torch.no_grad():
        output_image_ids, output_text_ids = model.interleave_generate(
            input_ids,
            uncond_input_ids,
            text_cfg=2.5,
            image_cfg=4.0,
            noise_schedule=get_mask_schedule(cfg.mask_schedule.schedule),
            text_steps=128,
            image_steps=30,
            reserved_token_mapping=RESERVED_TOKENS,
            uni_prompting=uni_prompting,
            config=cfg,
        )

    output_image_ids = torch.clamp(output_image_ids, 0, CODEBOOK_SIZE - 1).to(torch.long)
    pil_images = decode_vq(vq_model, output_image_ids)
    output_texts = tokenizer.batch_decode(output_text_ids, skip_special_tokens=True)
    return pil_images, output_texts, output_image_ids, output_text_ids, image_tokens_shifted


def _make_mask(shape: tuple[int, int], p: float, device: torch.device, skip_first: bool = False) -> torch.Tensor:
    m = torch.rand(shape, device=device) < p
    if skip_first and shape[1] > 0:
        m[:, 0] = False
    return m


def _output_img_slot() -> tuple[int, int]:
    start = 2 + NUM_VQ_TOKENS + MAX_SEQ_LENGTH + 2
    return start, start + NUM_VQ_TOKENS


def _output_text_slot() -> tuple[int, int]:
    img_end = _output_img_slot()[1]
    start = img_end + 1
    return start, start + MAX_SEQ_LENGTH


def compute_mode_logprobs(
    mode: str,
    model,
    tokenizer,
    *,
    input_image_tokens_shifted: torch.Tensor,
    input_texts: list[str],
    output_image_tokens_shifted: torch.Tensor,
    output_texts: list[str],
    device: torch.device,
    chunk_size: int = LOGPROB_CHUNK,
    mask_prob: float = LOGPROB_MASK_PROB,
) -> list[dict]:
    """For each sample, mask ~``mask_prob`` of the output slot(s) for the given
    ``mode`` and return per-token log probs at the masked positions.

    mode: "t2i" (mask image slot only), "mmu" (mask text slot only),
    or "interleave" (mask both).
    """
    if mode not in {"t2i", "mmu", "interleave"}:
        raise ValueError(f"unknown mode {mode!r}")

    B = input_image_tokens_shifted.shape[0]
    mask_img_slot = mode in {"t2i", "interleave"}
    mask_text_slot = mode in {"mmu", "interleave"}

    img_start, img_end = _output_img_slot()
    text_start, text_end = _output_text_slot()

    results: list[dict] = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        sub_input_img = input_image_tokens_shifted[start:end]
        sub_output_img = output_image_tokens_shifted[start:end]
        sub_input_text = list(input_texts[start:end])
        sub_output_text = list(output_texts[start:end])
        b = end - start

        zeros_img = torch.zeros(b, NUM_VQ_TOKENS, dtype=torch.bool, device=device)
        zeros_text = torch.zeros(b, MAX_SEQ_LENGTH, dtype=torch.bool, device=device)

        img_mask = _make_mask((b, NUM_VQ_TOKENS), mask_prob, device) if mask_img_slot else zeros_img
        text_mask = (
            _make_mask((b, MAX_SEQ_LENGTH), mask_prob, device, skip_first=True)
            if mask_text_slot
            else zeros_text
        )

        input_ids, labels, attn_mask, _ = prepare_inputs_and_labels_for_interleave_data(
            input_pixel_values=None,
            input_text=sub_input_text,
            output_pixel_values=None,
            output_text=sub_output_text,
            text_tokenizer=tokenizer,
            mask_id=MASK_TOKEN_ID,
            reserved_token_mapping=RESERVED_TOKENS,
            input_image_tokens=sub_input_img.clone(),
            output_image_tokens=sub_output_img.clone(),
            external_output_image_mask=img_mask,
            external_output_text_mask=text_mask,
            cond_dropout_prob=0.0,
            max_text_len=MAX_SEQ_LENGTH,
        )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward(input_ids=input_ids, attention_mask=attn_mask).logits

        log_probs = F.log_softmax(logits.float(), dim=-1)

        for i in range(b):
            out: dict = {}
            lbl = labels[i]
            lp = log_probs[i]

            if mask_img_slot:
                img_lbl = lbl[img_start:img_end]
                img_lp = lp[img_start:img_end]
                mask_i = img_lbl != -100
                positions = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)
                if positions.numel() > 0:
                    values = img_lp[positions, img_lbl[positions]]
                    out["image_masked_positions"] = positions.cpu().tolist()
                    out["image_token_ids"] = img_lbl[positions].cpu().tolist()
                    out["image_logprobs"] = values.cpu().tolist()
                    out["image_mean_logprob"] = float(values.mean().item())
                else:
                    out["image_masked_positions"] = []
                    out["image_token_ids"] = []
                    out["image_logprobs"] = []
                    out["image_mean_logprob"] = None

            if mask_text_slot:
                text_lbl = lbl[text_start:text_end]
                text_lp = lp[text_start:text_end]
                mask_i = text_lbl != -100
                positions = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)
                if positions.numel() > 0:
                    values = text_lp[positions, text_lbl[positions]]
                    out["text_masked_positions"] = positions.cpu().tolist()
                    out["text_token_ids"] = text_lbl[positions].cpu().tolist()
                    out["text_logprobs"] = values.cpu().tolist()
                    out["text_mean_logprob"] = float(values.mean().item())
                else:
                    out["text_masked_positions"] = []
                    out["text_token_ids"] = []
                    out["text_logprobs"] = []
                    out["text_mean_logprob"] = None

            results.append(out)

        del logits, log_probs
        torch.cuda.empty_cache()

    return results


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
    tm_gen, tm_und, _ = get_thinkmorph_interleave_questions(region_edit=False)
    assert len(tm_gen) >= NUM_SAMPLES, f"tm_gen has only {len(tm_gen)} samples"
    assert len(tm_und) >= NUM_SAMPLES, f"tm_und has only {len(tm_und)} samples"

    gen_samples = [tm_gen[i] for i in range(NUM_SAMPLES)]
    und_samples = [tm_und[i] for i in range(NUM_SAMPLES)]
    for g, u in zip(gen_samples, und_samples):
        assert g["sample_id"] == u["sample_id"], (
            f"sample_id mismatch: {g['sample_id']} vs {u['sample_id']}"
        )

    log_root = os.path.join(REPO_ROOT, "logs")
    t2i_dir = os.path.join(log_root, "t2i")
    il_dir = os.path.join(log_root, "interleave")
    os.makedirs(t2i_dir, exist_ok=True)
    os.makedirs(il_dir, exist_ok=True)

    def safe_id(sid: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)

    print(f"[run] t2i_generate (N={NUM_SAMPLES}, chunk={T2I_CHUNK})", flush=True)
    t2i_prompts: list[str] = []
    t2i_images: list[Image.Image] = []
    t2i_gen_ids_chunks: list[torch.Tensor] = []
    for i in range(0, NUM_SAMPLES, T2I_CHUNK):
        chunk = gen_samples[i : i + T2I_CHUNK]
        p, img, ids = run_t2i(model, vq_model, uni_prompting, cfg, chunk, device)
        t2i_prompts.extend(p)
        t2i_images.extend(img)
        t2i_gen_ids_chunks.append(ids)
        torch.cuda.empty_cache()
    t2i_gen_ids = torch.cat(t2i_gen_ids_chunks, dim=0)
    t2i_output_img_shifted = t2i_gen_ids + len(tokenizer)
    t2i_input_img_zeros = torch.zeros(
        NUM_SAMPLES, NUM_VQ_TOKENS, dtype=torch.long, device=device
    )

    print(f"[logprob] t2i (mask_prob={LOGPROB_MASK_PROB})", flush=True)
    t2i_logprobs = compute_mode_logprobs(
        mode="t2i",
        model=model,
        tokenizer=tokenizer,
        input_image_tokens_shifted=t2i_input_img_zeros,
        input_texts=t2i_prompts,
        output_image_tokens_shifted=t2i_output_img_shifted,
        output_texts=[""] * NUM_SAMPLES,
        device=device,
    )

    with open(os.path.join(log_root, "t2i.jsonl"), "w", encoding="utf-8") as f:
        for s, prompt, img, lp in zip(gen_samples, t2i_prompts, t2i_images, t2i_logprobs):
            img_path = os.path.join(t2i_dir, f"{safe_id(s['sample_id'])}.png")
            img.save(img_path)
            row = {
                "sample_id": s["sample_id"],
                "prompt": prompt,
                "image": img_path,
                **lp,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[done] t2i", flush=True)

    print(f"[run] mmu_generate (N={NUM_SAMPLES}, chunk={MMU_CHUNK})", flush=True)
    mmu_responses: list[str] = []
    mmu_input_img_chunks: list[torch.Tensor] = []
    for i in range(0, NUM_SAMPLES, MMU_CHUNK):
        chunk = und_samples[i : i + MMU_CHUNK]
        resp, img_tokens = run_mmu(model, vq_model, uni_prompting, tokenizer, chunk, device)
        mmu_responses.extend(resp)
        mmu_input_img_chunks.append(img_tokens)
        torch.cuda.empty_cache()
    mmu_input_img_shifted = torch.cat(mmu_input_img_chunks, dim=0)
    mmu_output_img_zeros = torch.zeros(
        NUM_SAMPLES, NUM_VQ_TOKENS, dtype=torch.long, device=device
    )

    print(f"[logprob] mmu (mask_prob={LOGPROB_MASK_PROB})", flush=True)
    mmu_logprobs = compute_mode_logprobs(
        mode="mmu",
        model=model,
        tokenizer=tokenizer,
        input_image_tokens_shifted=mmu_input_img_shifted,
        input_texts=[s["instruction"] for s in und_samples],
        output_image_tokens_shifted=mmu_output_img_zeros,
        output_texts=mmu_responses,
        device=device,
    )

    with open(os.path.join(log_root, "mmu.jsonl"), "w", encoding="utf-8") as f:
        for s, resp, lp in zip(und_samples, mmu_responses, mmu_logprobs):
            row = {
                "sample_id": s["sample_id"],
                "image": s["image"],
                "instruction": s["instruction"],
                "answer": resp,
                "answer_gt": s["answer_gt"],
                **lp,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[done] mmu", flush=True)

    print(f"[run] interleave_generate (N={NUM_SAMPLES}, chunk={INTERLEAVE_CHUNK})", flush=True)
    il_images: list[Image.Image] = []
    il_texts: list[str] = []
    il_output_img_chunks: list[torch.Tensor] = []
    il_input_img_chunks: list[torch.Tensor] = []
    for i in range(0, NUM_SAMPLES, INTERLEAVE_CHUNK):
        chunk = gen_samples[i : i + INTERLEAVE_CHUNK]
        imgs, txts, out_img_ids, _, in_img_shifted = run_interleave(
            model, vq_model, uni_prompting, tokenizer, cfg, chunk, device
        )
        il_images.extend(imgs)
        il_texts.extend(txts)
        il_output_img_chunks.append(out_img_ids)
        il_input_img_chunks.append(in_img_shifted)
        torch.cuda.empty_cache()
    il_output_img_shifted = torch.cat(il_output_img_chunks, dim=0) + len(tokenizer)
    il_input_img_shifted = torch.cat(il_input_img_chunks, dim=0)

    print(f"[logprob] interleave (mask_prob={LOGPROB_MASK_PROB})", flush=True)
    il_logprobs = compute_mode_logprobs(
        mode="interleave",
        model=model,
        tokenizer=tokenizer,
        input_image_tokens_shifted=il_input_img_shifted,
        input_texts=[s["instruction"] for s in gen_samples],
        output_image_tokens_shifted=il_output_img_shifted,
        output_texts=il_texts,
        device=device,
    )

    with open(os.path.join(log_root, "interleave.jsonl"), "w", encoding="utf-8") as f:
        for g, u, img, txt, lp in zip(gen_samples, und_samples, il_images, il_texts, il_logprobs):
            img_path = os.path.join(il_dir, f"{safe_id(g['sample_id'])}.png")
            img.save(img_path)
            row = {
                "sample_id": g["sample_id"],
                "src_image": g["image"],
                "gen_instruction": g["instruction"],
                "und_question": u["instruction"],
                "output_image": img_path,
                "output_text": txt,
                "answer_gt": u["answer_gt"],
                **lp,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[done] interleave", flush=True)

    print(f"Saved results to {log_root}", flush=True)


if __name__ == "__main__":
    main()
