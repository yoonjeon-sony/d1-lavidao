import copy
import inspect
import os
import random
import re
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from types import SimpleNamespace
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import DistributedType, gather, gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import math

import wandb
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.integrations import WandbCallback
from transformers.utils import is_peft_available
from trl.extras.profiling import profiling_context, profiling_decorator
# from trl.import_utils import is_rich_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer, nanstd, nanmin, nanmax
from diffu_grpo_trainer import DiffuGRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample, selective_log_softmax
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
)
if is_peft_available():
    from peft import PeftConfig
from reward_func import perceptual_score_reward_func, strict_format_reward_func, correctness_reward_func

@contextmanager
def _timer(timings: dict, key: str):
    """Accumulate wall-clock seconds into *timings[key]* and print a status line."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"[time_profile] {key} started")
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    timings[key] = timings.get(key, 0.0) + elapsed
    print(f"[time_profile] {key} finished in {elapsed:.2f}s")


class MMaDAGRPOTrainer(DiffuGRPOTrainer):
    """GRPO trainer adapted for MMaDA, inheriting LaVida-O rollout infrastructure from DiffuGRPOTrainer."""

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        train_dataset_und: Optional[Union[Dataset, IterableDataset]] = None,
        train_dataset_ground: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
        modality: str = "gen",
    ):
        # ``modality`` controls how the main train_dataset is interpreted:
        #   - "gen":  train_dataset is the gen side (image-edit rollout driver);
        #             train_dataset_und (optional) supplies paired und rows via
        #             sample_id lookup. This is the standard interleave flow.
        #   - "und":  train_dataset is the und side (text-only rollout driver);
        #             train_dataset_und MUST be None; the image rollout is
        #             bypassed entirely. Used by thinkmorph_answer.
        # For thinkmorph_edit (gen-only) the default "gen" modality with
        # train_dataset_und=None is exactly the right behavior — the text
        # rollout is already bypassed when self._und_by_sample_id is None.
        if modality not in ("gen", "und"):
            raise ValueError(
                f"modality must be 'gen' or 'und', got {modality!r}"
            )
        if modality == "und" and train_dataset_und is not None:
            raise ValueError(
                "modality='und' requires train_dataset_und=None: pass the "
                "und rows as train_dataset instead. Got a non-None "
                "train_dataset_und."
            )
        self.modality = modality

        _register_lavida_architectures()
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        args.use_fast_dlm = False
        
        if self.accelerator.is_main_process and wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "huggingface"),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=args.run_name,
                config=args.to_dict(),
            )

        grad_accum_steps = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        if getattr(self, "_buffered_inputs", None) is None:
            self._buffered_inputs = [None] * grad_accum_steps
        # Buffered micro-batches for gradient accumulation.
        # Filled by _prepare_inputs on generation steps, consumed on subsequent steps.
        self._buffered_inputs = None
        
        # ---------- und side: sample-id pairing (no second dataloader) ----------
        # Rationale: running a second DataLoader with its own RepeatSampler drifts
        # from the gen-side sampler in distributed training, because each loader's
        # ``accelerator.prepare`` dispatches batches to ranks independently.  The
        # observed failure is
        #     gen sample_id='arxivqa:q-bio-5057' vs und sample_id='arxivqa:cs-30504'
        # at row 0 — same step, different ArxivQA rows on the two sides.
        #
        # Fix: build a ``{sample_id -> und_row}`` map once at init, then in
        # ``_prepare_inputs`` resolve und rows from the gen batch by sample_id.
        # This makes the pairing bulletproof regardless of sampler / shard /
        # worker ordering — a gen row and its paired und row always share
        # sample_id by construction (they were jointly permuted upstream in
        # diffu_grpo_train.py).
        self._und_by_sample_id = None
        if train_dataset_und is not None:
            if "sample_id" not in train_dataset_und.column_names:
                raise ValueError(
                    "train_dataset_und must carry a 'sample_id' column for "
                    "gen/und pairing. Got columns: "
                    f"{train_dataset_und.column_names}"
                )
            # Materialize to a dict for O(1) lookup. The und rows are small
            # (text prompt + answer_gt + image path), so the memory cost is
            # negligible compared to the model.
            self._und_by_sample_id = {
                row["sample_id"]: row for row in train_dataset_und
            }
            if len(self._und_by_sample_id) != len(train_dataset_und):
                raise ValueError(
                    f"train_dataset_und has duplicate sample_ids: "
                    f"{len(train_dataset_und)} rows but only "
                    f"{len(self._und_by_sample_id)} unique sample_ids."
                )

        # Grounding side: paired with gen rows by sample_id (same pattern as
        # und). Populated only when ``region_edit`` is enabled upstream and a
        # ground_ds was constructed by the data loader.
        self._ground_by_sample_id = None
        if train_dataset_ground is not None:
            if "sample_id" not in train_dataset_ground.column_names:
                raise ValueError(
                    "train_dataset_ground must carry a 'sample_id' column for "
                    "gen/ground pairing. Got columns: "
                    f"{train_dataset_ground.column_names}"
                )
            self._ground_by_sample_id = {
                row["sample_id"]: row for row in train_dataset_ground
            }
            if len(self._ground_by_sample_id) != len(train_dataset_ground):
                raise ValueError(
                    f"train_dataset_ground has duplicate sample_ids: "
                    f"{len(train_dataset_ground)} rows but only "
                    f"{len(self._ground_by_sample_id)} unique sample_ids."
                )


    @staticmethod
    def _make_generator(device: torch.device, seed: int) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    
    def _rollout_multimodal_text_gen(
        self,
        model,
        examples: Union[dict[str, Any], list[dict[str, Any]]],
        image_processor,
        generation_kwargs: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        pass


    def _rollout_image_edit_latents(
        self,
        model,
        examples: list[dict[str, Any]],
        init_image=None,
        predicted_bbox: Optional[list[tuple[float, float, float, float]]] = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        pass

    
    def _get_per_token_logps(
        self,
        model,
        gen_scoring=None,
        und_scoring=None,
        mask_seeds: list[int] = None,
        cached_gen_samples: list[dict] = None,
        cached_und_samples: list[dict] = None,
        force_gen_masks: list[list[dict]] = None,
        force_und_masks: list[list[dict]] = None,
        gen_slice: tuple = None,
        und_slice: tuple = None,
    ) -> tuple:
        pass

    @profiling_decorator
    def _compute_loss(
        self, model, inputs: dict[str, Any],
    ) -> torch.Tensor:
        """Compute GRPO clipped loss for gen and/or und modalities.

        Args:
            model: the (possibly wrapped) model.
            inputs: dict with keys ``gen_batch``, ``und_batch``, ``mask_seeds``.
                Each batch is ``[scoring, advantages, old_logps, ref_logps]`` or None.

        Returns:
            Scalar loss (gen_loss + und_loss).  Either component is 0 when its
            batch is None.
        """
        beta = float(getattr(self.args, "beta", 0.0))
        epsilon = float(getattr(self.args, "epsilon", 0.0))
        num_iterations = int(getattr(self.args, "num_iterations", 1))

        gen_batch = inputs.get("gen_batch")  # [scoring, adv_sub, old_sub, ref_sub, (s,e)] or None
        und_batch = inputs.get("und_batch")  # [scoring, adv_sub, old_sub, ref_sub, (s,e)] or None
        mask_seeds = inputs["mask_seeds"]    # list[int], length = num_iterations
        # Per-modality raw-sample caches (no longer a unified list — gen_slice /
        # und_slice index directly into each).
        cached_gen_samples = inputs.get("cached_gen_samples")
        cached_und_samples = inputs.get("cached_und_samples")
        # Per-modality per-(seed, sample) mask caches captured during the
        # old-policy forward.  Shape: [num_iterations][N_full] of dicts with
        # "text_mask" / "gen_mask" CPU tensors.
        cached_gen_masks_full = inputs.get("cached_gen_masks")
        cached_und_masks_full = inputs.get("cached_und_masks")

        # Map ``_step`` to the PPO-style inner iteration index.  We want the
        # first pass over all ``steps_per_generation`` micro-chunks (one full
        # optimizer step) to use iteration 0 across every sample, then the
        # second pass to use iteration 1 across every sample, and so on.
        # That is a **step function** of _step at the optimizer-step boundary,
        # not per-micro-chunk — i.e. floor-divide by steps_per_generation
        # before taking mod num_iterations.  The previous formulation
        # (``self._step % num_iterations``) cycled on the micro-step scale,
        # which silently partitioned samples into ``num_iterations`` disjoint
        # subsets (each chunk permanently bound to one mask seed) and left
        # half of the precomputed ``(num_iter, N, D)`` old/ref logps unused.
        this_itr_idx = (self._step // self.args.steps_per_generation) % num_iterations

        current_mask_seed = mask_seeds[this_itr_idx]
        if torch.is_tensor(current_mask_seed):
            current_mask_seed = int(current_mask_seed.item())

        _t0_curr = time.perf_counter()

        # ---- Unpack gen / und buffers ----
        # Each batch is [full_scoring, adv_sub, old_sub, ref_sub, (chunk_start, chunk_end)].
        # full_scoring is the complete (unsplit) dataset; (s, e) marks this chunk's slice.
        if gen_batch is not None:
            gen_scoring, gen_advantages, old_gen_logps, ref_gen_logps, gen_chunk_range = gen_batch
        else:
            gen_scoring, gen_advantages, old_gen_logps, ref_gen_logps, gen_chunk_range = None, None, None, None, None
        if und_batch is not None:
            und_scoring, und_advantages, old_und_logps, ref_und_logps, und_chunk_range = und_batch
        else:
            und_scoring, und_advantages, old_und_logps, ref_und_logps, und_chunk_range = None, None, None, None, None

        # ---- Current log-probs via model forward ----
        # Reuse cached per-sample dicts AND the per-sample masks captured
        # during the old-policy pass.  This gives us two guarantees:
        #   (1) the cached samples ensure identical tokenization / padding
        #       (the non-determinism of LazySupervisedDataset's random crops
        #       and retries is sidestepped);
        #   (2) the cached masks, passed back as force_*_masks, make the
        #       model bypass its random-draw masking entirely so the current
        #       forward sees bitwise-identical final_masked_indices and
        #       masked_indices_gen to what the old forward used.
        # Together these guarantee that on step 0 (identical weights) the
        # current per_token_logps equals old_ptl exactly → coef_1 = 1.
        #
        # scoring_batch_size MUST be 1 in this path: the captured masks are
        # stored per sample with sample-specific seq_len, so they can't be
        # trivially batch-stacked (different samples would need different
        # padding).  The force-mask path in _run_modality asserts this.
        #
        # Slice the full [num_iterations][N_full] mask caches down to just
        # the current iteration and the current chunk.  We call
        # _get_per_token_logps with mask_seeds=[current_mask_seed] (length 1),
        # so the outer list of force_*_masks has length 1.
        def _slice_masks(masks_full, chunk_range):
            if masks_full is None or chunk_range is None:
                return None
            s, e = chunk_range
            return [masks_full[this_itr_idx][s:e]]

        force_gen_masks = _slice_masks(cached_gen_masks_full, gen_chunk_range)
        force_und_masks = _slice_masks(cached_und_masks_full, und_chunk_range)

        # _get_per_token_logps now hard-codes scoring_batch_size=1 internally
        # (required by the force-mask plumbing), so no need to override
        # self._train_batch_size here.
        gen_logps, und_logps, _, _, _, _ = self._get_per_token_logps(
            model, gen_scoring=gen_scoring, und_scoring=und_scoring,
            mask_seeds=[current_mask_seed],
            cached_gen_samples=cached_gen_samples,
            cached_und_samples=cached_und_samples,
            force_gen_masks=force_gen_masks,
            force_und_masks=force_und_masks,
            gen_slice=gen_chunk_range,
            und_slice=und_chunk_range,
        )
        # Squeeze the seed dimension (we only have one seed here).
        if gen_logps is not None:
            gen_logps = gen_logps[0]   # (chunk_size, latent_total)
        if und_logps is not None:
            und_logps = und_logps[0]   # (chunk_size, L_text)

        mode = "train" if self.model.training else "eval"
        epsilon_low = self.epsilon_low if hasattr(self, "epsilon_low") else epsilon
        epsilon_high = self.epsilon_high if hasattr(self, "epsilon_high") else epsilon

        # ---- Helper: GRPO clipped loss for one modality ----
        def _grpo_loss(per_token_logps, old_logps_all, ref_logps_all, advantages, prefix):
            """Returns scalar loss for a single modality, or None."""
            if per_token_logps is None:
                return None

            old_ptl = old_logps_all[this_itr_idx]  # (N, D_old)
            ref_ptl = ref_logps_all[this_itr_idx] if (beta != 0.0 and ref_logps_all is not None) else None

            # old/ref logps may live on CPU (offloaded in _generate_and_score_completions
            # to free GPU memory during backward across gradient accumulation steps).
            # Bring only the current iteration's slice back to the device, non-blocking
            # so the copy overlaps with the current-policy forward.
            target_device = per_token_logps.device
            if old_ptl.device != target_device:
                old_ptl = old_ptl.to(target_device, non_blocking=True)
            if ref_ptl is not None and ref_ptl.device != target_device:
                ref_ptl = ref_ptl.to(target_device, non_blocking=True)

            # Align all log-prob tensors to a common sequence length.
            # Old/ref logps were collated from the full dataset (longer max seq);
            # current logps from just this chunk (shorter max seq).  The
            # underlying samples are identical (cached), so zero-padding is
            # safe — completion_mask zeroes out all padded positions.
            target_D = max(
                per_token_logps.shape[-1],
                old_ptl.shape[-1],
                ref_ptl.shape[-1] if ref_ptl is not None else 0,
            )
            def _pad_to(t, D):
                d = D - t.shape[-1]
                return torch.nn.functional.pad(t, (0, d)) if d > 0 else t

            per_token_logps = _pad_to(per_token_logps, target_D)
            old_ptl = _pad_to(old_ptl, target_D)
            if ref_ptl is not None:
                ref_ptl = _pad_to(ref_ptl, target_D)

            coef_1 = torch.exp(per_token_logps - old_ptl)
            coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # -------- DEBUG: coef_1 magnitude ----------
            def _dbg_coef1():
                mask = (old_ptl.detach() != 0)
                if not mask.any():
                    return
                diff = (per_token_logps.detach() - old_ptl.detach())[mask]
                c1 = coef_1.detach()[mask]
                _debug_log(
                    f"{prefix}coef_1 | min={float(c1.min()):.3e} "
                    f"max={float(c1.max()):.3e} mean={float(c1.mean()):.3e} "
                    f"| logdiff min={float(diff.min()):.3e} "
                    f"max={float(diff.max()):.3e} "
                    f"| adv min={float(advantages.min()):.3e} "
                    f"max={float(advantages.max()):.3e}"
                )
                # Early warning for any catastrophic entry.
                if float(c1.max()) > 1e4:
                    topk = diff.abs().topk(min(5, diff.numel()))
                    _debug_log(
                        f"{prefix}coef_1 EXPLOSION | top |logdiff|="
                        f"{[float(v) for v in topk.values.tolist()]}"
                    )
            _debug_run(_dbg_coef1)

            if ref_ptl is not None:
                log_ratio = ref_ptl - per_token_logps
                per_token_kl = torch.exp(log_ratio) - log_ratio - 1
                per_token_loss = per_token_loss + beta * per_token_kl

            else:
                per_token_kl = None

            # Use non-zero (non-padding) positions as the completion mask.
            # For gen: masked latent positions have non-zero loss from gen_loss_none_reduction.
            # For und: masked text positions have non-zero loss from und_loss_none_reduction.
            # Positions that were not masked during the forward pass have zero log-prob,
            # so they contribute nothing and can be safely included in the denominator.
            completion_mask = (old_ptl != 0).float()
            denom = completion_mask.sum().clamp(min=1.0)
            loss = (per_token_loss * completion_mask).sum() / denom

            # ---- Metrics ----
            if per_token_kl is not None:
                mean_kl = ((per_token_kl * completion_mask).sum() / denom)
                self._metrics[mode][f"{prefix}kl"].append(
                    self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
                )

            is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = ((is_low_clipped.float() * completion_mask).sum() / denom).detach()
            high_clip = ((is_high_clipped.float() * completion_mask).sum() / denom).detach()
            clip_ratio = ((is_region_clipped.float() * completion_mask).sum() / denom).detach()

            self._metrics[mode][f"{prefix}clip_ratio/low_mean"].append(
                self.accelerator.gather_for_metrics(low_clip).nanmean().item()
            )
            self._metrics[mode][f"{prefix}clip_ratio/high_mean"].append(
                self.accelerator.gather_for_metrics(high_clip).nanmean().item()
            )
            self._metrics[mode][f"{prefix}clip_ratio/region_mean"].append(
                self.accelerator.gather_for_metrics(clip_ratio).nanmean().item()
            )
            return loss

        # ---- Compute losses ----
        # Unified mode (text_rollout_use_gen_image=True at rollout time):
        # ``gen_logps`` carries the concatenated gen+und per-token logps for
        # the unified payload (set by `_run_modality(unified=True)`),
        # ``und_logps`` is None (und_batch was None in the batch dict
        # returned by _generate_and_score_completions). Run a single
        # _grpo_loss with the ``unified/`` prefix; skip the und call.
        unified_mode_loss = (gen_logps is not None and und_logps is None
                             and getattr(self.args, "text_rollout_use_gen_image", False))
        if unified_mode_loss:
            unified_loss = _grpo_loss(
                gen_logps, old_gen_logps, ref_gen_logps, gen_advantages, "unified/",
            )
            gen_loss = None
            und_loss = None
        else:
            gen_loss = _grpo_loss(gen_logps, old_gen_logps, ref_gen_logps, gen_advantages, "gen/")
            und_loss = _grpo_loss(und_logps, old_und_logps, ref_und_logps, und_advantages, "und/")
            unified_loss = None

        # ---- Combine ----
        if unified_loss is not None:
            loss = unified_loss
        elif gen_loss is not None and und_loss is not None:
            loss = gen_loss + und_loss
        elif gen_loss is not None:
            loss = gen_loss
        elif und_loss is not None:
            loss = und_loss
        else:
            raise ValueError("Both gen_batch and und_batch are None — nothing to train on.")

        if unified_loss is not None:
            self._metrics[mode]["unified/loss"].append(unified_loss.detach().item())
        if gen_loss is not None:
            self._metrics[mode]["gen/loss"].append(gen_loss.detach().item())
        if und_loss is not None:
            self._metrics[mode]["und/loss"].append(und_loss.detach().item())

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        _elapsed_curr = time.perf_counter() - _t0_curr
        self._metrics[mode]["time_profile/curr_logprobs_and_update"].append(_elapsed_curr)
        return loss

    def _generate_and_score_completions(
        self,
        gen_inputs: dict[str, Union[torch.Tensor, Any]] = None,
        und_inputs: dict[str, Union[torch.Tensor, Any]] = None,
        ground_inputs: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Union[torch.Tensor, Any]]:

        pass