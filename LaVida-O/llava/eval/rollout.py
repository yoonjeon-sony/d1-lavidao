# ADOBE CONFIDENTIAL
# Copyright 2025 Adobe
# All Rights Reserved.
"""Standalone rollout helpers for LaVida-O / LLaDA.

This module is a near-verbatim port of ``_rollout_image_edit_latents`` and
``_rollout_multimodal_text_gen`` from ``diffu-grpo/diffu_grpo_trainer.py``.
It lifts the trainer's ``self.args`` / ``self.processing_class`` /
``self.max_prompt_length`` into explicit function parameters so the same
rollout code paths can be reused outside the GRPO trainer (e.g. from
``lmms-eval``'s ``generate_until``).

Exports:
    - run_image_rollout(model, examples, *, tokenizer, image_processor,
                        device, gen_cfg) -> (xt, image_contexts)
    - run_text_rollout(model, examples, *, tokenizer, image_processor,
                       device, generation_kwargs, image_edit_resolution,
                       max_prompt_length=None, conv_version="llada")
                       -> (completion_ids, image_contexts)
    - build_image_edit_gen_cfg(**overrides) -> dict
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import pad_to_square_and_resize, process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import (
    cosine_schedule_2,
    exp_schedule,
    generate as llada_generate,
    get_logits as llada_get_logits,
    get_num_transfer_tokens_sch,
    logit_normal_schedule,
    wte as llada_wte,
)
from llava.model.utils import pad_along_last_dim


__all__ = [
    "run_image_rollout",
    "run_text_rollout",
    "build_image_edit_gen_cfg",
]


# ---------------------------------------------------------------------------
# build_image_edit_gen_cfg — port of Diffu_Grpo_Trainer._get_image_edit_gen_dict
# ---------------------------------------------------------------------------

# Defaults mirror ``diffu-grpo/diffu_grpo_config.py`` image_edit_* fields.
_IMAGE_EDIT_DEFAULTS: dict[str, Any] = {
    "sample_policy": "multinomial",
    "confidence_policy": "halton",
    "guidance_scale": 0.0,
    "guidance_scale_image": 5.0,
    "batch_size": 1,
    "image_resolution": 1024,
    # n_tokens / prompt_n_tokens are resolved from image_resolution below
    # unless explicitly overridden.
    "shift": 5,
    "n_steps": 64,
    "schedule": "shift",
    "alg_temp": 5.0,
    "dynamic_temperature": True,
    "schedule_temp": "cosine2",
    "min_temperature": 0.5,
    "schedule_temp_samp": "linear",
    "dynamic_temperature_samp": False,
    "min_temperature_samp": 1.0,
    "cfg_interval": [0.0, 1.0],
    "order_cutoff": 1.0,
    "edit_mode": 0,
    "micro_cond": "",
    "remask_ratio": 0.01,
}


def build_image_edit_gen_cfg(**overrides: Any) -> dict[str, Any]:
    """Return a gen_cfg dict consumable by ``run_image_rollout``.

    Port of ``Diffu_Grpo_Trainer._get_image_edit_gen_dict`` — defaults match
    ``diffu_grpo_config.py`` image_edit_* fields and ``n_tokens`` /
    ``prompt_n_tokens`` are derived from ``image_resolution`` if absent.
    """
    cfg: dict[str, Any] = dict(_IMAGE_EDIT_DEFAULTS)
    # cfg_interval may be supplied as a pair or as start/end values.
    if "cfg_interval_start" in overrides or "cfg_interval_end" in overrides:
        start = float(overrides.pop("cfg_interval_start", cfg["cfg_interval"][0]))
        end = float(overrides.pop("cfg_interval_end", cfg["cfg_interval"][1]))
        cfg["cfg_interval"] = [start, end]
    cfg.update(overrides)

    latent_token_map = {256: 256, 512: 1024, 1024: 4096}
    prompt_token_map = {256: 64, 512: 256, 1024: 1024}
    res = int(cfg["image_resolution"])
    cfg["image_resolution"] = res
    if "n_tokens" not in cfg or cfg.get("n_tokens") is None:
        cfg["n_tokens"] = latent_token_map.get(res, 4096)
    if "prompt_n_tokens" not in cfg or cfg.get("prompt_n_tokens") is None:
        cfg["prompt_n_tokens"] = prompt_token_map.get(res, max(1, cfg["n_tokens"] // 4))
    # Normalize numeric types (mirrors _get_image_edit_gen_dict int/float casts).
    cfg["n_tokens"] = int(cfg["n_tokens"])
    cfg["prompt_n_tokens"] = int(cfg["prompt_n_tokens"])
    cfg["shift"] = int(cfg["shift"])
    cfg["n_steps"] = int(cfg["n_steps"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["alg_temp"] = float(cfg["alg_temp"])
    cfg["min_temperature"] = float(cfg["min_temperature"])
    cfg["min_temperature_samp"] = float(cfg["min_temperature_samp"])
    cfg["guidance_scale"] = float(cfg["guidance_scale"])
    cfg["guidance_scale_image"] = float(cfg["guidance_scale_image"])
    cfg["order_cutoff"] = float(cfg["order_cutoff"])
    cfg["edit_mode"] = int(cfg["edit_mode"])
    cfg["dynamic_temperature"] = bool(cfg["dynamic_temperature"])
    cfg["dynamic_temperature_samp"] = bool(cfg["dynamic_temperature_samp"])
    cfg["remask_ratio"] = float(cfg["remask_ratio"])
    if isinstance(cfg["cfg_interval"], (tuple, list)):
        cfg["cfg_interval"] = [float(cfg["cfg_interval"][0]), float(cfg["cfg_interval"][1])]
    return cfg


# ---------------------------------------------------------------------------
# stratified_random — port of top-level helper in diffu_grpo_trainer.py
# ---------------------------------------------------------------------------


def stratified_random(
    n: int = 64, seed: Optional[int] = None, shuffle_blocks: bool = True
) -> List[int]:
    """Progressive Multi-Jittered (PMJ) ordering over an n*n grid (n power of 2).

    Returns row-major linear indices ``y*n + x`` for ``x, y`` in ``[0, n)``.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two (e.g., 64)")

    rng = random.Random(seed)
    occupied = [[False] * n for _ in range(n)]
    seq: List[int] = []
    blocks: List[Tuple[int, int, int]] = [(0, 0, n)]

    def block_has_sample(x0: int, y0: int, size: int) -> bool:
        for yy in range(y0, y0 + size):
            row = occupied[yy]
            for xx in range(x0, x0 + size):
                if row[xx]:
                    return True
        return False

    def place_random_in_block(x0: int, y0: int, size: int) -> None:
        x = rng.randrange(x0, x0 + size)
        y = rng.randrange(y0, y0 + size)
        attempts = 0
        while occupied[y][x]:
            x = rng.randrange(x0, x0 + size)
            y = rng.randrange(y0, y0 + size)
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Too many attempts to place a sample; logic error?")
        occupied[y][x] = True
        seq.append(y * n + x)

    size = n
    while size > 1:
        half = size // 2
        children: List[Tuple[int, int, int]] = []
        for (x0, y0, s) in blocks:
            assert s == size
            children.extend([
                (x0, y0, half),
                (x0 + half, y0, half),
                (x0, y0 + half, half),
                (x0 + half, y0 + half, half),
            ])
        if shuffle_blocks:
            rng.shuffle(children)
        for (x0, y0, s) in children:
            if not block_has_sample(x0, y0, s):
                place_random_in_block(x0, y0, s)
        blocks = children
        size = half

    remaining: List[int] = []
    for y in range(n):
        for x in range(n):
            if not occupied[y][x]:
                remaining.append(y * n + x)
    rng.shuffle(remaining)
    seq.extend(remaining)
    assert len(seq) == n * n, (len(seq), n * n)
    return seq


# ---------------------------------------------------------------------------
# Small helpers lifted from diffu_grpo_trainer.py
# ---------------------------------------------------------------------------


def _role_to_conv_role(role: str) -> str:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    return "assistant"


def _build_llada_prompt(
    prompt_messages: Any,
    has_gen_image: bool = False,
    conv_version: str = "llada",
) -> str:
    if conv_version not in conv_templates:
        conv_version = "llada"
    conv = copy.deepcopy(conv_templates[conv_version])

    if isinstance(prompt_messages, str):
        conv.append_message(conv.roles[0], prompt_messages)
    else:
        for turn in prompt_messages:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            mapped_role = _role_to_conv_role(role)
            conv.append_message(mapped_role, content)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if has_gen_image:
        if "<image>\n" not in prompt:
            raise ValueError(
                "has_gen_image=True but the built prompt does not contain "
                "'<image>\\n' to duplicate. Check the conversation template "
                "and the sample's 'prompt' field."
            )
        prompt = prompt.replace("<image>\n", "<image>\n<image>\n", 1)
    return prompt


def _load_image(image_like: Any) -> Any:
    if image_like is None:
        return None
    if isinstance(image_like, str):
        from PIL import Image

        return Image.open(image_like).convert("RGB")
    if isinstance(image_like, list) and len(image_like) > 0:
        return _load_image(image_like[0])
    return image_like


def _left_pad_2d(
    tensors: list[torch.Tensor], pad_value: int, dtype_: torch.dtype
) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.empty(0, 0, dtype=dtype_)
    max_len = max(t.shape[1] for t in tensors)
    out = []
    for t in tensors:
        pad_len = max_len - t.shape[1]
        if pad_len > 0:
            pad_t = torch.full(
                (t.shape[0], pad_len), pad_value, dtype=dtype_, device=t.device
            )
            t = torch.cat([pad_t, t], dim=1)
        out.append(t)
    return torch.cat(out, dim=0)


def _normalize_mm_image_payload(
    image_tensor: Any, *, dtype: torch.dtype, device: torch.device
) -> list[torch.Tensor]:
    if isinstance(image_tensor, list):
        return [_x.to(dtype=dtype, device=device) for _x in image_tensor]
    image_tensor = image_tensor.to(dtype=dtype, device=device)
    return [img for img in image_tensor]


def _image_edit_gumbel_noise(tensor: torch.Tensor) -> torch.Tensor:
    noise = torch.zeros_like(tensor, dtype=torch.float32).uniform_(0, 1)
    return -torch.log(-torch.log(noise))


def _extract_image_edit_instruction(example: dict[str, Any]) -> str:
    instruction = example.get("instruction")
    if instruction is not None:
        return instruction
    prompt_data = example.get("prompt", [{"role": "user", "content": ""}])
    if isinstance(prompt_data, list) and len(prompt_data) > 0:
        last = prompt_data[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return ""


def _normalize_text_rollout(
    generated: torch.Tensor, prompt_ids: torch.Tensor, prefix_lm: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefix_lm:
        completion_ids = generated
        normalized_prompt_ids = prompt_ids
    else:
        prompt_len = prompt_ids.size(1)
        if generated.size(1) < prompt_len:
            completion_ids = generated
            normalized_prompt_ids = prompt_ids
        else:
            completion_ids = generated[:, prompt_len:]
            normalized_prompt_ids = prompt_ids
    return normalized_prompt_ids, completion_ids


def _build_image_edit_temperature_schedule(
    n_steps: int, schedule_name: str, min_temperature: float, shift: int
) -> torch.Tensor:
    temperatures = torch.linspace(0, 1, n_steps, device="cpu").numpy()
    if schedule_name == "linear":
        temperatures = (1 - temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_name == "cosine2":
        temperatures = cosine_schedule_2(1 - temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_name == "shift":
        temperatures = logit_normal_schedule(shift, 1 - temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_name == "exp":
        temperatures = exp_schedule(1 - temperatures) * (1 - min_temperature) + min_temperature
    else:
        raise NotImplementedError(
            f"Unknown image-edit temperature schedule: {schedule_name}"
        )
    return torch.tensor(temperatures, dtype=torch.float32)


def _make_invalid_image_edit_ctx(latent_template: torch.Tensor) -> dict[str, Any]:
    return {
        "valid": False,
        "latent_shape": tuple(latent_template.shape),
        "decoded_image": None,
    }


# ---------------------------------------------------------------------------
# run_image_rollout — port of Diffu_Grpo_Trainer._rollout_image_edit_latents
# ---------------------------------------------------------------------------


def run_image_rollout(
    model,
    examples: Union[dict[str, Any], list[dict[str, Any]]],
    *,
    tokenizer,
    image_processor,
    device: torch.device,
    gen_cfg: dict[str, Any],
    conv_version: str = "llada",
    init_image=None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Block-wise masked diffusion image rollout.

    Near-verbatim port of
    ``Diffu_Grpo_Trainer._rollout_image_edit_latents`` (diffu_grpo_trainer.py:
    768–1163). All trainer-bound state has been lifted into explicit arguments
    (``tokenizer``, ``image_processor``, ``device``, ``gen_cfg``).
    """
    if isinstance(examples, dict):
        examples = [examples]

    batch_size = len(examples)
    reserve_id = 126089
    reserve_id2 = 126090
    img_mask_id = 8193
    reserve_token = "<|reserved_token_5|>"
    reserve_token_2 = "<|reserved_token_6|>"
    base_model = model.get_model() if hasattr(model, "get_model") else model
    if conv_version not in conv_templates:
        conv_version = "llada"
    gen_shape_map = {1024: (64, 64), 512: (32, 32), 256: (16, 16)}
    gen_shape = gen_shape_map.get(gen_cfg["image_resolution"], (32, 32))
    is_unitok = "unitok" in getattr(base_model.config, "mm_vqvae", "")
    latent_shape = (8, gen_cfg["n_tokens"]) if is_unitok else (gen_cfg["n_tokens"],)

    batch_latents = [
        torch.full(latent_shape, img_mask_id, dtype=torch.long, device=device)
        for _ in range(batch_size)
    ]
    image_contexts = [_make_invalid_image_edit_ctx(batch_latents[idx]) for idx in range(batch_size)]

    if (
        image_processor is None
        or not hasattr(model, "encode_image_gen")
        or not hasattr(base_model, "call_gen_embedding")
        or not hasattr(base_model, "image_processor_gen")
    ):
        return torch.stack(batch_latents, dim=0), image_contexts

    all_input_ids = []
    all_edit_images = []
    image_sizes = []
    all_enc_embeddings = []
    vq_latents = []
    prompts = []
    for batch_idx, example in enumerate(examples):
        edit_image = _load_image(example.get("image"))
        assert edit_image is not None, f"Edit image is None for example {example}"
        instruction = _extract_image_edit_instruction(example)

        image_sizes.append(edit_image.size)
        all_edit_images.append(edit_image)

        image_1024 = pad_to_square_and_resize(
            edit_image.convert("RGB"), gen_cfg["image_resolution"]
        )
        vq_latent = base_model.image_processor_gen.preprocess(image_1024).to(
            device, dtype=model.dtype
        )
        vq_latents.append(vq_latent)

        enc_latents, enc_shape = model.encode_image_gen(vq_latent, enc=True)
        enc_embeddings = base_model.call_gen_embedding(enc_latents, enc_shape, enc=True)
        all_enc_embeddings.append(enc_embeddings)

        conv = copy.deepcopy(conv_templates[conv_version])
        conv.append_message(
            conv.roles[0],
            f"<image> {reserve_token_2 * enc_embeddings.shape[1]}\n {instruction} ",
        )
        conv.append_message(conv.roles[1], reserve_token * gen_cfg["prompt_n_tokens"])
        prompt_question = conv.get_prompt()
        prompt_question = prompt_question.removesuffix(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(prompt_question)
        input_ids = tokenizer_image_token(
            prompt_question,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)
        all_input_ids.append(input_ids)

    per_row_prompt_lens = [int(t.shape[1]) for t in all_input_ids]
    all_input_ids = _left_pad_2d(all_input_ids, tokenizer.pad_token_id, torch.long)
    attention_mask = (all_input_ids != tokenizer.pad_token_id).long()
    image_tensor = process_images(all_edit_images, image_processor, model.config)
    image_tensor = _normalize_mm_image_payload(image_tensor, dtype=model.dtype, device=device)

    inputs = model.prepare_inputs_labels_for_multimodal(
        input_ids=all_input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
        modalities=["image"] * all_input_ids.shape[0],
        image_sizes=image_sizes,
        return_inputs=True,
    )
    _, _, attention_mask, _, inputs_embeds, _, raw_input_ids = inputs

    init_latents, _ = model.encode_image_gen(torch.cat(vq_latents, dim=0))
    enc_embeddings = pad_along_last_dim(
        torch.cat(all_enc_embeddings, dim=0), size=inputs_embeds.shape[-1]
    )
    inputs_embeds = inputs_embeds.to(dtype=enc_embeddings.dtype)
    inputs_embeds[raw_input_ids == reserve_id2] = 0
    inputs_embeds[raw_input_ids == reserve_id2] = enc_embeddings.flatten(0, 1)

    is_prompt = torch.zeros_like(raw_input_ids, dtype=torch.bool)
    for row_idx in range(raw_input_ids.shape[0]):
        row_eot = torch.where(raw_input_ids[row_idx] == 126348)[0]
        if row_eot.numel() >= 2:
            prompt_cutoff = int(row_eot[1].item())
            is_prompt[row_idx, : prompt_cutoff + 1] = True
    is_gen = raw_input_ids == reserve_id
    is_gen_enc = raw_input_ids == reserve_id2

    mask_id_for_noise = int(gen_cfg.get("mask_id", 126336))
    noise_embed = base_model.transformer.wte(torch.tensor([mask_id_for_noise], device=device))
    inputs_embeds_uncond = inputs_embeds.clone()
    inputs_embeds_uncond[is_prompt] = noise_embed

    inputs_embeds_uncond_enc = inputs_embeds.clone()
    edit_mode = gen_cfg["edit_mode"]
    if edit_mode == 0:
        inputs_embeds_uncond_enc[~is_gen_enc] = noise_embed
        is_gen_enc_ccc = is_gen_enc
    elif edit_mode == 1:
        inputs_embeds_uncond_enc[is_gen_enc] = noise_embed
        is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
    elif edit_mode == 2:
        inputs_embeds_uncond_enc[is_gen_enc | (raw_input_ids < 0)] = noise_embed
        is_gen_enc_ccc = torch.zeros_like(is_gen_enc, dtype=torch.bool)
    elif edit_mode == 3:
        inputs_embeds_uncond_enc[(~is_gen_enc) & (raw_input_ids > 0)] = noise_embed
        is_gen_enc_ccc = is_gen_enc
    else:
        raise ValueError("Not Supported edit_mode")
    is_gen_enc_null = torch.zeros_like(is_gen_enc, dtype=torch.bool)

    n_tokens = gen_cfg["n_tokens"]
    image_gen_latents_offset = torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device)
    if is_unitok:
        image_gen_latents_offset = image_gen_latents_offset.unsqueeze(1).repeat(1, 8, 1)
    image_gen_latents_offset[:] = img_mask_id

    xt = image_gen_latents_offset.clone()
    if init_image is not None:
        remask_ratio = gen_cfg.get("remask_ratio", 0.01)
        n_mask_remask = max(int(n_tokens * remask_ratio), 1)
        indices = np.arange(n_tokens)
        np.random.shuffle(indices)
        init_mask_indices = indices[:n_mask_remask]
        xt[:, init_mask_indices] = init_latents[:, init_mask_indices]

    use_3d = False
    mask_idx_sched = xt == img_mask_id
    if is_unitok and not use_3d:
        mask_idx_sched = mask_idx_sched[:, 0, :]
    if gen_cfg["schedule"] == "shift":
        num_transfer_tokens = get_num_transfer_tokens_sch(
            mask_idx_sched,
            gen_cfg["n_steps"],
            schedule="shift",
            schedule_kwargs={"shift": gen_cfg["shift"]},
        )
    else:
        num_transfer_tokens = get_num_transfer_tokens_sch(
            mask_idx_sched,
            gen_cfg["n_steps"],
            schedule=gen_cfg["schedule"],
            schedule_kwargs={"shift": gen_cfg["shift"]},
        )

    confidence_policy = gen_cfg["confidence_policy"]
    if confidence_policy == "halton":
        confidence_policy = "mmada"
    unmask_order = None
    if confidence_policy == "stratified":
        unmask_order = stratified_random(
            n=int(np.sqrt(gen_cfg["n_tokens"])), seed=42, shuffle_blocks=True
        )

    sch_temperatures = _build_image_edit_temperature_schedule(
        gen_cfg["n_steps"],
        gen_cfg["schedule_temp"],
        gen_cfg["min_temperature"],
        gen_cfg["shift"],
    ).to(device=device)
    sch_temperatures_samp = _build_image_edit_temperature_schedule(
        gen_cfg["n_steps"],
        gen_cfg["schedule_temp_samp"],
        gen_cfg["min_temperature_samp"],
        gen_cfg["shift"],
    ).to(device=device)
    cfg_start = int(gen_cfg["cfg_interval"][0] * gen_cfg["n_steps"])
    cfg_end = int(gen_cfg["cfg_interval"][1] * gen_cfg["n_steps"])

    x0 = xt.clone()
    active_steps = torch.nonzero(num_transfer_tokens.sum(dim=0) > 0, as_tuple=False).squeeze(-1)
    for step_idx, step_col in enumerate(active_steps, start=1):
        local_temp = sch_temperatures[min(step_idx - 1, sch_temperatures.numel() - 1)]
        local_temp_samp = sch_temperatures_samp[min(step_idx - 1, sch_temperatures_samp.numel() - 1)]
        step_confidence_policy = confidence_policy
        if step_idx / max(gen_cfg["n_steps"], 1) > gen_cfg["order_cutoff"]:
            step_confidence_policy = "mmada"
        mask_idx = xt == img_mask_id
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:, 0, :]
        n_mask_per_sample = mask_idx.sum(dim=1)
        if n_mask_per_sample.max().item() == 0:
            break

        timesteps = n_mask_per_sample.float() / mask_idx.shape[1]
        do_cfg = gen_cfg["guidance_scale"] > 0 and (cfg_start <= step_idx <= cfg_end)

        if do_cfg:
            embed_input = torch.cat(
                [inputs_embeds_uncond, inputs_embeds_uncond_enc, inputs_embeds], dim=0
            )
            xt_input = torch.cat([xt, xt, xt], dim=0)
            new_token_mask = is_gen.repeat(3, 1)
            is_gen_enc_mask = torch.cat([is_gen_enc_null, is_gen_enc_ccc, is_gen_enc], dim=0)
            timesteps_input = timesteps.repeat(3)
        else:
            embed_input = inputs_embeds
            xt_input = xt
            new_token_mask = is_gen
            is_gen_enc_mask = is_gen_enc
            timesteps_input = timesteps

        if getattr(base_model.config, "enc_use_image_branch", False):
            modality_indices = new_token_mask | is_gen_enc_mask
        else:
            modality_indices = new_token_mask

        all_input_embeddings, new_token_mask = llada_wte(
            base_model,
            None,
            True,
            x_gen=xt_input,
            gen_shape=gen_shape,
            inputs_embeds_curr=embed_input.clone(),
            new_token_mask=new_token_mask,
        )
        logits = llada_get_logits(
            base_model,
            all_input_embeddings,
            new_token_mask,
            True,
            gen_shape=gen_shape,
            input_modality_indices=modality_indices,
            timesteps=timesteps_input,
        )
        if is_unitok:
            logits[..., 4096:] = float("-inf")
        if do_cfg:
            _, _, new_token_mask = new_token_mask.chunk(3)
            logits_un, logits_un_enc, logits = logits.chunk(3)
            logits_is_ninf = logits == -np.inf
            if edit_mode in [0, 3]:
                logits_cond = (1 + gen_cfg["guidance_scale_image"]) * logits - (
                    gen_cfg["guidance_scale_image"] * logits_un_enc
                )
            elif edit_mode in [1, 2]:
                logits_cond = (logits + gen_cfg["guidance_scale_image"] * logits_un_enc) / (
                    1 + gen_cfg["guidance_scale_image"]
                )
            else:
                raise ValueError("Not Supported edit_mode")
            logits = (1 + gen_cfg["guidance_scale"]) * logits_cond - gen_cfg["guidance_scale"] * logits_un
            logits[logits_is_ninf] = -np.inf

        safe_logits = torch.nan_to_num(
            logits.to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9
        )
        probs = safe_logits.softmax(-1)
        if gen_cfg["sample_policy"] == "argmax":
            x0 = safe_logits.argmax(-1)
        else:
            temperature = 1.0
            if gen_cfg["dynamic_temperature_samp"]:
                temperature = temperature * float(local_temp_samp)
            if temperature <= 0:
                x0 = safe_logits.argmax(-1)
            else:
                x0 = torch.distributions.Categorical(logits=safe_logits / temperature).sample()
        x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

        if is_unitok:
            x0 = x0.permute(0, 2, 1)
            if use_3d:
                x0 = torch.where(mask_idx, x0, xt)
                x0_p = x0_p.permute(0, 2, 1).max(dim=1)[0]
            else:
                x0 = torch.where(mask_idx.unsqueeze(1).repeat(1, 8, 1), x0, xt)
                x0_p = x0_p.permute(0, 2, 1).max(dim=1)[0]
        else:
            x0 = torch.where(mask_idx, x0, xt)

        for batch_idx in range(len(examples)):
            b_mask = mask_idx[batch_idx]
            b_n_mask = int(b_mask.sum().item())
            if b_n_mask == 0:
                continue
            k = min(int(num_transfer_tokens[batch_idx, step_col].item()), b_n_mask)
            if k <= 0:
                continue

            if step_confidence_policy == "mask_git":
                alg_temp = (
                    gen_cfg["alg_temp"] * float(local_temp)
                    if gen_cfg["dynamic_temperature"]
                    else gen_cfg["alg_temp"]
                )
                confidence = x0_p[batch_idx] / max(alg_temp, 1e-6)
                confidence = torch.where(b_mask, confidence, -np.inf)
                confidence = torch.softmax(confidence, dim=-1)
                select_index = torch.multinomial(confidence, num_samples=k)
            elif step_confidence_policy == "stratified" and unmask_order is not None:
                start = gen_cfg["n_tokens"] - b_n_mask
                select_index = torch.tensor(
                    unmask_order[start : start + k], device=device, dtype=torch.long
                )
            else:
                alg_temp = (
                    gen_cfg["alg_temp"] * float(local_temp)
                    if gen_cfg["dynamic_temperature"]
                    else gen_cfg["alg_temp"]
                )
                confidence = torch.log(x0_p[batch_idx].clamp(1e-20)) + alg_temp * _image_edit_gumbel_noise(
                    x0_p[batch_idx]
                )
                confidence = torch.where(b_mask, confidence, -np.inf)
                _, select_index = torch.topk(confidence, k=k)

            if is_unitok:
                transfer_index[batch_idx, :, select_index] = True
            else:
                transfer_index[batch_idx, select_index] = True
        xt[transfer_index] = x0[transfer_index]

    xt[xt == img_mask_id] = x0[xt == img_mask_id]
    decoded_images = model.decode_image_gen(
        xt,
        gen_cfg["image_resolution"],
        gen_cfg["image_resolution"],
    )
    image_contexts = []
    for batch_idx, (example, decoded_image) in enumerate(zip(examples, decoded_images)):
        prompt = prompts[batch_idx]
        decoded_image_obj = decoded_image
        if torch.is_tensor(decoded_image):
            from PIL import Image

            image_array = decoded_image.detach().cpu()
            if image_array.ndim == 3 and image_array.shape[0] in (1, 3):
                image_array = image_array.permute(1, 2, 0)
            image_array = image_array.clamp(0, 1).mul(255).to(torch.uint8).numpy()
            if image_array.ndim == 3 and image_array.shape[-1] == 1:
                image_array = image_array[..., 0]
            decoded_image_obj = Image.fromarray(image_array)
        elif isinstance(decoded_image, np.ndarray):
            from PIL import Image

            image_array = np.clip(decoded_image, 0, 255).astype(np.uint8)
            if image_array.ndim == 3 and image_array.shape[-1] == 1:
                image_array = image_array[..., 0]
            decoded_image_obj = Image.fromarray(image_array)

        image_contexts.append(
            {
                "valid": True,
                "latent_shape": tuple(xt[batch_idx].shape),
                "decoded_image": decoded_image_obj,
                "prompt": prompt,
                "prompt_len_tokens": per_row_prompt_lens[batch_idx],
                "completion_len_tokens": int(gen_cfg["n_tokens"]),
            }
        )
    return xt, image_contexts


# ---------------------------------------------------------------------------
# run_text_rollout — port of Diffu_Grpo_Trainer._rollout_multimodal_text_gen
# ---------------------------------------------------------------------------


def run_text_rollout(
    model,
    examples: Union[dict[str, Any], list[dict[str, Any]]],
    *,
    tokenizer,
    image_processor,
    device: torch.device,
    generation_kwargs: dict[str, Any],
    image_edit_resolution: int,
    max_prompt_length: Optional[int] = None,
    conv_version: str = "llada",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Text rollout with optional ``gen_image`` second-image injection.

    Near-verbatim port of ``Diffu_Grpo_Trainer._rollout_multimodal_text_gen``
    (diffu_grpo_trainer.py:538–686). When an example carries a ``gen_image``
    key, the prompt's ``<image>\\n`` is duplicated so the LM sees both the
    problem image and the rollout-generated image (in that order).
    """
    if isinstance(examples, dict):
        examples = [examples]

    prompt_texts: list[str] = []
    all_input_ids: list[torch.Tensor] = []
    all_images: list[Any] = []
    image_sizes: list[tuple[int, int]] = []
    resolution = int(image_edit_resolution)
    if image_processor is None:
        raise ValueError("Text rollout requires an image processor for multimodal prompts.")

    for example in examples:
        has_gen_image = example.get("gen_image") is not None
        prompt_text = _build_llada_prompt(
            example["prompt"], has_gen_image=has_gen_image, conv_version=conv_version
        )
        if "<image>" not in prompt_text:
            sample_id = example.get("id", example.get("pid", "unknown"))
            raise ValueError(
                "Text rollout example includes an image but the prompt does not contain '<image>': "
                f"sample_id={sample_id}"
            )
        prompt_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(device)
        if max_prompt_length is not None:
            prompt_ids = prompt_ids[-max_prompt_length:]
        all_input_ids.append(prompt_ids.unsqueeze(0))
        prompt_texts.append(prompt_text)

        image = _load_image(example.get("image"))
        if image is None:
            sample_id = example.get("id", example.get("pid", "unknown"))
            raise ValueError(f"Text rollout example is missing an image: sample_id={sample_id}")
        processed_image = pad_to_square_and_resize(image.convert("RGB"), resolution)
        all_images.append(processed_image)
        image_sizes.append(processed_image.size)

        if has_gen_image:
            gen_image = _load_image(example.get("gen_image"))
            if gen_image is None:
                sample_id = example.get("id", example.get("pid", "unknown"))
                raise ValueError(
                    f"Text rollout example declared gen_image but it could not "
                    f"be loaded: sample_id={sample_id}"
                )
            processed_gen = pad_to_square_and_resize(gen_image.convert("RGB"), resolution)
            all_images.append(processed_gen)
            image_sizes.append(processed_gen.size)

    prompt_ids = _left_pad_2d(all_input_ids, tokenizer.pad_token_id, torch.long).to(device)
    prompt_mask = (prompt_ids != tokenizer.pad_token_id).long()
    position_ids = None
    image_tensor = process_images(all_images, image_processor, model.config)
    image_tensor = _normalize_mm_image_payload(image_tensor, dtype=model.dtype, device=device)

    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=prompt_ids,
        position_ids=position_ids,
        attention_mask=prompt_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
        modalities=["image"] * prompt_ids.shape[0],
        image_sizes=image_sizes,
    )

    generated = llada_generate(
        model.get_model(),
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    if isinstance(generated, tuple):
        generated = generated[0]

    prefix_lm = bool(generation_kwargs.get("prefix_lm", False))
    _, completion_ids = _normalize_text_rollout(generated, prompt_ids, prefix_lm)
    decoded_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    pad_id = tokenizer.pad_token_id
    per_row_prompt_lens = (prompt_ids != pad_id).sum(dim=1).tolist()
    per_row_completion_lens = (completion_ids != pad_id).sum(dim=1).tolist()

    image_contexts = []
    for batch_idx, example in enumerate(examples):
        image_contexts.append(
            {
                "decoded_text": decoded_texts[batch_idx],
                "prompt": prompt_texts[batch_idx],
                "prompt_len_tokens": int(per_row_prompt_lens[batch_idx]),
                "completion_len_tokens": int(per_row_completion_lens[batch_idx]),
            }
        )

    return completion_ids.detach(), image_contexts
