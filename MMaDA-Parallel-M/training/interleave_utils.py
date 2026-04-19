"""Shared helper that builds training/eval-style interleave input sequences.

Extracted from ``train_interleave.py`` so both the training loop and inference
scripts (e.g. ``infer_all.py`` log-prob evaluation) can reuse the same layout.

Sequence layout per sample:

    [task] <|soi|> input_img (num_vq_tokens) <|eoi|> input_text (max_text_len)
        || <|soi|> masked_output_img (num_vq_tokens) <|eoi|> output_text (max_text_len)

When ``external_output_image_mask`` and ``external_output_text_mask`` are
provided, the internal ``t ~ U(eps, 1)``-based mask generation is skipped and
the provided masks are used verbatim. Token overrides (``input_image_tokens``,
``output_image_tokens``) let callers supply pre-encoded VQ token ids so the
``vq_model`` argument is not required.
"""
from __future__ import annotations

import math
from typing import Union

import torch


@torch.no_grad()
def prepare_inputs_and_labels_for_interleave_data(
    input_pixel_values: torch.Tensor | None,
    input_text: Union[str, list[str]],
    output_pixel_values: torch.Tensor | None,
    output_text: Union[str, list[str]],
    *,
    text_tokenizer,
    mask_id: int,
    reserved_token_mapping: dict,
    vq_model=None,
    mask_schedule=None,
    input_image_tokens: torch.Tensor | None = None,
    output_image_tokens: torch.Tensor | None = None,
    external_output_image_mask: torch.Tensor | None = None,
    external_output_text_mask: torch.Tensor | None = None,
    eps: float = 1e-3,
    is_text_only_mask: torch.Tensor | None = None,
    is_text_only_output_mask: torch.Tensor | None = None,
    seed: int | None = None,
    cond_dropout_prob: float = 0.0,
    max_text_len: int | None = None,
):
    if text_tokenizer is None:
        raise ValueError("text_tokenizer is required")
    if mask_id is None:
        raise ValueError("mask_id is required")
    if max_text_len is None:
        raise ValueError("max_text_len is required")

    if input_image_tokens is None:
        if vq_model is None or input_pixel_values is None:
            raise ValueError("Need input_image_tokens or (vq_model + input_pixel_values)")
        input_image_tokens = vq_model.get_code(input_pixel_values) + len(text_tokenizer)
    if output_image_tokens is None:
        if vq_model is None or output_pixel_values is None:
            raise ValueError("Need output_image_tokens or (vq_model + output_pixel_values)")
        output_image_tokens = vq_model.get_code(output_pixel_values) + len(text_tokenizer)

    device = input_image_tokens.device
    batch_size = input_image_tokens.shape[0]

    if is_text_only_mask is None:
        is_text_only_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i in range(batch_size):
        if is_text_only_mask[i]:
            input_image_tokens[i] = torch.zeros_like(input_image_tokens[i])

    input_text_ids = text_tokenizer(input_text)["input_ids"]
    output_text_ids = text_tokenizer(output_text)["input_ids"]

    output_image_seq_len = output_image_tokens.shape[1]

    external_provided = (
        external_output_image_mask is not None and external_output_text_mask is not None
    )
    if external_provided:
        if external_output_image_mask.shape != output_image_tokens.shape:
            raise ValueError(
                f"external_output_image_mask {tuple(external_output_image_mask.shape)} must match "
                f"output image tokens {tuple(output_image_tokens.shape)}"
            )
        if external_output_text_mask.shape != (batch_size, max_text_len):
            raise ValueError(
                f"external_output_text_mask {tuple(external_output_text_mask.shape)} must be "
                f"(batch_size={batch_size}, max_text_len={max_text_len})"
            )
        mask = external_output_image_mask.to(device=device, dtype=torch.bool)
        text_masked_indices = external_output_text_mask.to(device=device, dtype=torch.bool)
        t = torch.ones(batch_size, device=device)
    else:
        if mask_schedule is None:
            raise ValueError("mask_schedule is required when external masks are not provided")
        t = torch.rand(batch_size, device=device)
        t = t * (1 - eps) + eps
        mask_prob = mask_schedule(t).clip(eps)
        mask_prob = torch.cos(mask_prob * math.pi * 0.5)
        num_token_masked = (output_image_seq_len * mask_prob).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, output_image_seq_len, device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        text_masked_indices = (
            torch.rand(batch_size, max_text_len, device=device) < mask_prob.unsqueeze(1)
        )
        text_masked_indices[:, 0] = False

    masked_output_image_ids = torch.where(mask, mask_id, output_image_tokens)
    output_image_labels = torch.where(mask, output_image_tokens, -100)

    if is_text_only_output_mask is not None:
        # Rows where the output has no GT image: mask the whole image slot
        # and zero the image CE contribution (labels = -100 everywhere).
        tmo = is_text_only_output_mask.to(device=device, dtype=torch.bool)
        if tmo.any():
            row_sel = tmo.view(-1, 1)
            masked_output_image_ids = torch.where(
                row_sel, torch.full_like(masked_output_image_ids, mask_id), masked_output_image_ids
            )
            output_image_labels = torch.where(
                row_sel, torch.full_like(output_image_labels, -100), output_image_labels
            )

    dropout_text_probs = torch.rand(batch_size)
    dropout_image_probs = torch.rand(batch_size)

    output_sequences_ids = []
    output_labels_ids = []
    output_attention_masks = []

    interleave_token_id = reserved_token_mapping.get(
        "<|interleave|>", reserved_token_mapping.get("<|t2it|>", 126095)
    )
    text_only_token_id = reserved_token_mapping.get(
        "<|t2it|>", reserved_token_mapping.get("<t2it>", interleave_token_id)
    )
    soi_id = reserved_token_mapping["<|soi|>"]
    eoi_id = reserved_token_mapping["<|eoi|>"]

    for i in range(batch_size):
        task_token_id = text_only_token_id if is_text_only_mask[i] else interleave_token_id

        # --- Input side ---
        if len(input_text_ids[i]) == 0 or input_text_ids[i][0] != text_tokenizer.bos_token_id:
            input_text_ids[i] = [text_tokenizer.bos_token_id] + list(input_text_ids[i])
        if input_text_ids[i][-1] != text_tokenizer.eos_token_id:
            input_text_ids[i] = list(input_text_ids[i]) + [text_tokenizer.eos_token_id]

        if dropout_text_probs[i] < cond_dropout_prob:
            input_text_ids[i] = [text_tokenizer.bos_token_id, text_tokenizer.eos_token_id]
        if dropout_image_probs[i] < cond_dropout_prob:
            input_image_tokens[i] = torch.zeros_like(input_image_tokens[i])

        input_len = len(input_text_ids[i])
        if max_text_len >= input_len:
            input_text_padding_masks = [1] * (input_len + 3 + input_image_tokens.shape[-1]) + [0] * (
                max_text_len - input_len
            )
            input_text_padding_masks = torch.tensor(input_text_padding_masks, device=device)
            input_text_ids[i] = list(input_text_ids[i]) + [text_tokenizer.eos_token_id] * (
                max_text_len - input_len
            )
        else:
            input_text_padding_masks = torch.tensor(
                [1] * (max_text_len + 3 + input_image_tokens.shape[-1]), device=device
            )
            input_text_ids[i] = list(input_text_ids[i])[: max_text_len - 1] + [
                text_tokenizer.eos_token_id
            ]

        input_interleave_ids = torch.cat(
            [
                torch.tensor([task_token_id], device=device),
                torch.tensor([soi_id], device=device),
                input_image_tokens[i],
                torch.tensor([eoi_id], device=device),
                torch.tensor(input_text_ids[i], device=device),
            ]
        )
        input_interleave_labels = torch.full(
            (input_interleave_ids.shape[0],), -100, dtype=torch.long, device=device
        )

        # --- Output side ---
        if len(output_text_ids[i]) == 0 or output_text_ids[i][0] != text_tokenizer.bos_token_id:
            output_text_ids[i] = [text_tokenizer.bos_token_id] + list(output_text_ids[i])
        if output_text_ids[i][-1] != text_tokenizer.eos_token_id:
            output_text_ids[i] = list(output_text_ids[i]) + [text_tokenizer.eos_token_id]

        out_len = len(output_text_ids[i])
        if max_text_len >= out_len:
            output_text_padding_masks = torch.tensor(
                [1] * (out_len + 2 + output_image_tokens.shape[-1])
                + [0] * (max_text_len - out_len),
                device=device,
            )
            output_text_ids[i] = list(output_text_ids[i]) + [text_tokenizer.eos_token_id] * (
                max_text_len - out_len
            )
        else:
            output_text_padding_masks = torch.tensor(
                [1] * (max_text_len + 2 + output_image_tokens.shape[-1]), device=device
            )
            output_text_ids[i] = list(output_text_ids[i])[: max_text_len - 1] + [
                text_tokenizer.eos_token_id
            ]

        output_text_tensor = torch.tensor(output_text_ids[i], device=device)
        output_noisy_text_ids = torch.where(
            text_masked_indices[i], torch.tensor(mask_id, device=device), output_text_tensor
        )
        output_text_labels = torch.where(
            text_masked_indices[i], output_text_tensor, torch.tensor(-100, device=device)
        )

        output_interleave_ids = torch.cat(
            [
                torch.tensor([soi_id], device=device),
                masked_output_image_ids[i],
                torch.tensor([eoi_id], device=device),
                output_noisy_text_ids,
            ]
        )
        output_interleave_labels = torch.cat(
            [
                torch.tensor([-100], device=device),
                output_image_labels[i],
                torch.tensor([-100], device=device),
                output_text_labels,
            ]
        )

        sequence_ids = torch.cat([input_interleave_ids, output_interleave_ids], dim=0)
        label_ids = torch.cat([input_interleave_labels, output_interleave_labels], dim=0)
        all_mask = torch.cat([input_text_padding_masks, output_text_padding_masks], dim=0)

        output_sequences_ids.append(sequence_ids.unsqueeze(0))
        output_labels_ids.append(label_ids.unsqueeze(0))
        output_attention_masks.append(all_mask.unsqueeze(0))

    return (
        torch.cat(output_sequences_ids, dim=0),
        torch.cat(output_labels_ids, dim=0),
        torch.cat(output_attention_masks, dim=0),
        t,
    )
