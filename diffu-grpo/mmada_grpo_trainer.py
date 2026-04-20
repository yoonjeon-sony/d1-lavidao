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
from diffu_grpo_trainer import (
    DiffuGRPOTrainer,
    RewardFunc,
    _debug_log,
    _debug_run,
    _register_lavida_architectures,
    DIFFU_GRPO_DEBUG,
)
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


# ---------------------------------------------------------------------------
# MMaDA-Parallel constants (mirroring infer_all.py layout).
#
# These fix the interleave sequence layout used by
# ``prepare_inputs_and_labels_for_interleave_data``. Every scoring/rollout
# call in this trainer assumes this exact layout so the (img_start, img_end)
# / (text_start, text_end) slot computation in ``_mmada_score_modality``
# remains correct.
# ---------------------------------------------------------------------------
_MMADA_NUM_VQ_TOKENS = 1024
_MMADA_CODEBOOK_SIZE = 8192
_MMADA_RESOLUTION = 512
_MMADA_MAX_SEQ_LENGTH = 256
_MMADA_MAX_TEXT_LEN = 256
_MMADA_MASK_TOKEN_ID = 126336
_MMADA_VQ_MODEL_NAME = "showlab/magvitv2"

_MMADA_RESERVED_TOKENS = {
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


def _mmada_output_img_slot(num_vq: int = _MMADA_NUM_VQ_TOKENS, max_text_len: int = _MMADA_MAX_SEQ_LENGTH):
    start = 2 + num_vq + max_text_len + 2
    return start, start + num_vq


def _mmada_output_text_slot(num_vq: int = _MMADA_NUM_VQ_TOKENS, max_text_len: int = _MMADA_MAX_SEQ_LENGTH):
    img_end = _mmada_output_img_slot(num_vq, max_text_len)[1]
    start = img_end + 1
    return start, start + max_text_len


class _NullCtx:
    """No-op context manager used as a fallback when ``torch.autocast`` is
    unavailable (CPU-only runs during unit tests)."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _register_mmada_architectures() -> None:
    """Expose MMaDA model classes on the ``transformers`` namespace.

    TRL's ``GRPOTrainer.__init__`` builds the ref-model via
    ``getattr(transformers, config.architectures[0]).from_pretrained(...)``.
    The MMaDA-8B checkpoint sets ``architectures=["LLaDAModelLM"]`` so that
    attribute must exist on ``transformers`` before ``super().__init__``
    runs — otherwise TRL raises ``AttributeError: module transformers has
    no attribute LLaDAModelLM``. Registering ``MMadaModelLM`` as well keeps
    us forward-compatible with checkpoints that set the subclass name.
    Lazy-imports the classes so this module still imports on rigs where
    MMaDA-Parallel isn't on sys.path yet.
    """
    here = Path(__file__).resolve()
    candidates = [
        Path("/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M"),
        here.parents[1] / "MMaDA-Parallel-M",
        here.parents[1] / "MMaDA",
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from models.modeling_llada import LLaDAModelLM  # noqa: E402
        from models.modeling_mmada import MMadaModelLM, MMadaConfig  # noqa: E402
    except ImportError:
        return
    # LaVida-O's import machinery swaps ``sys.modules["transformers"]`` during
    # its load, so the module reference captured at the top of this file can
    # go stale. Resolve the live module fresh, so TRL (which does its own
    # ``import transformers`` inside GRPOTrainer) sees our setattr.
    import transformers as _transformers_live  # noqa: E402
    for name, cls in (
        ("LLaDAModelLM", LLaDAModelLM),
        ("MMadaModelLM", MMadaModelLM),
        ("MMadaConfig", MMadaConfig),
    ):
        if not hasattr(_transformers_live, name):
            setattr(_transformers_live, name, cls)


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

        # Register BOTH LaVida and MMaDA classes on the transformers namespace
        # before ``super().__init__`` runs. TRL's GRPOTrainer builds the
        # ref-model via ``getattr(transformers, config.architectures[0])``;
        # LaVida registration is kept for mixed-flow safety, MMaDA
        # registration is what this trainer actually needs.
        _register_lavida_architectures()
        _register_mmada_architectures()
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            train_dataset_und=train_dataset_und,
            train_dataset_ground=train_dataset_ground,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            modality=modality,
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

    # ------------------------------------------------------------------
    # MMaDA helpers (lazy resource init, image I/O, mask draw)
    # ------------------------------------------------------------------
    def _ensure_mmada_resources(self, device: torch.device):
        """Lazily build ``self._uni_prompting`` / ``self._vq_model``.

        MMaDA-Parallel modules (``training.prompting_utils``, ``models``) live
        outside ``diffu-grpo`` and are typically added to ``sys.path`` by the
        training entry point before importing the trainer. This helper is
        tolerant to the path not being configured yet — it probes a few
        plausible locations and errors cleanly if the import still fails.
        """
        if getattr(self, "_mmada_resources_ready", False):
            return self._uni_prompting, self._vq_model

        here = Path(__file__).resolve()
        candidates = [
            Path("/music-home-shared-disk/user/yoonjeon.kim/d1/MMaDA-Parallel-M"),
            here.parents[1] / "MMaDA-Parallel-M",
            here.parents[1] / "MMaDA",
        ]
        for p in candidates:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

        try:
            from training.prompting_utils import UniversalPrompting  # noqa: E402
            from models import MAGVITv2  # noqa: E402
        except ImportError as e:
            raise RuntimeError(
                "MMaDA-Parallel 'training' and 'models' packages are not on "
                "sys.path. Add the MMaDA-Parallel-M repo to sys.path before "
                "using MMaDAGRPOTrainer."
            ) from e

        max_text_len = int(getattr(self.args, "mmada_max_text_len", _MMADA_MAX_TEXT_LEN))
        self._uni_prompting = UniversalPrompting(
            self.processing_class,
            max_text_len=max_text_len,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.1,
            use_reserved_token=True,
        )
        vq_name = getattr(self.args, "mmada_vq_model_name", _MMADA_VQ_MODEL_NAME)
        vq_model = MAGVITv2.from_pretrained(vq_name, low_cpu_mem_usage=False).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        self._vq_model = vq_model
        self._mmada_resources_ready = True
        return self._uni_prompting, self._vq_model

    @staticmethod
    def _mmada_load_image_tensor(path_or_img, device: torch.device, resolution: int) -> torch.Tensor:
        from training.utils import image_transform_squash  # lazy
        from PIL import Image

        if hasattr(path_or_img, "convert"):
            img = path_or_img
        elif isinstance(path_or_img, str):
            img = Image.open(path_or_img)
        else:
            raise ValueError(f"Unsupported image source: {type(path_or_img)!r}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        return image_transform_squash(img, resolution=resolution).to(device)

    @staticmethod
    def _mmada_decode_vq(vq_model, token_ids: torch.Tensor):
        from PIL import Image as _PILImage

        token_ids = torch.clamp(token_ids, 0, _MMADA_CODEBOOK_SIZE - 1).to(torch.long)
        images = vq_model.decode_code(token_ids)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        images = (images * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        return [_PILImage.fromarray(img) for img in images]

    @staticmethod
    def _mmada_seeded_mask(
        shape: Tuple[int, ...],
        p: float,
        device: torch.device,
        seed: int,
        sample_idx: int,
        skip_first: bool = False,
    ) -> torch.Tensor:
        """Deterministic per-(seed, sample) mask draw used when no force-mask
        was provided. The combination seed*P + sample_idx produces a unique
        generator per scoring sample so that two samples in the same chunk
        don't end up with identical masks."""
        gen = torch.Generator(device=device)
        combined = (int(seed) * 1000003 + int(sample_idx)) & 0x7FFFFFFF
        gen.manual_seed(combined)
        m = torch.rand(shape, device=device, generator=gen) < p
        if skip_first and shape[-1] > 0:
            m[..., 0] = False
        return m

    @staticmethod
    def _mmada_gather_logp(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Gather per-token logp at ``labels`` positions; zero out ``-100``.

        Matches DiffuGRPO's convention where non-masked positions contribute
        zero to the logp tensor — the _compute_loss helper then uses
        ``completion_mask = (old_ptl != 0)`` to select scored positions.
        """
        safe = labels.clamp(min=0)
        gathered = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # (B, L)
        mask = (labels != -100).float()
        return gathered * mask

    def _build_mmada_gen_config(self):
        from omegaconf import OmegaConf

        return OmegaConf.create({
            "model": {"mmada": {
                "num_vq_tokens": _MMADA_NUM_VQ_TOKENS,
                "codebook_size": _MMADA_CODEBOOK_SIZE,
            }},
            "dataset": {"preprocessing": {
                "max_seq_length": _MMADA_MAX_SEQ_LENGTH,
                "resolution": _MMADA_RESOLUTION,
            }},
            "training": {
                "guidance_scale": float(getattr(self.args, "mmada_guidance_scale", 3.5)),
                "generation_timesteps": int(getattr(self.args, "mmada_generation_timesteps", 50)),
                "cond_dropout_prob": 0.1,
                "generation_temperature": 1.0,
                "noise_type": "mask",
            },
            "mask_schedule": {"schedule": getattr(self.args, "mmada_mask_schedule", "cosine")},
        })

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------
    def _rollout_multimodal_text_gen(
        self,
        model,
        examples: Union[dict[str, Any], list[dict[str, Any]]],
        image_processor,
        generation_kwargs: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Text rollout (image + question -> answer) via ``model.mmu_generate``.

        Mirrors ``run_mmu`` in MMaDA-Parallel-M/infer_all.py. In unified mode
        (``text_rollout_use_gen_image=True``), each example carries ``gen_image``
        (the PIL image produced by the image rollout); we use that as the MMU
        input image so the text answer is conditioned on the model's own
        reasoning image, matching the DiffuGRPOTrainer semantics adapted for a
        single-image MMU path.
        """
        if isinstance(examples, dict):
            examples = [examples]
        uni_prompting, vq_model = self._ensure_mmada_resources(device)
        tokenizer = self.processing_class
        resolution = int(getattr(self.args, "mmada_resolution", _MMADA_RESOLUTION))
        max_seq_length = int(getattr(self.args, "mmada_max_seq_length", _MMADA_MAX_SEQ_LENGTH))
        mask_id = int(getattr(self.args, "mask_id", _MMADA_MASK_TOKEN_ID))

        # Input image: gen_image (unified mode) takes precedence over the
        # problem image, so the text rollout sees the rollout-generated
        # reasoning image when available.
        pixels = []
        for ex in examples:
            src = ex.get("gen_image") if ex.get("gen_image") is not None else ex.get("image")
            if src is None:
                raise ValueError(
                    f"Text rollout example missing an input image "
                    f"(sample_id={ex.get('sample_id', 'unknown')})"
                )
            pixels.append(self._mmada_load_image_tensor(src, device, resolution))
        pixel_batch = torch.stack(pixels, dim=0)
        image_tokens_shifted = vq_model.get_code(pixel_batch) + len(tokenizer)

        # Build the MMU text prefix via chat template (matches infer_all.run_mmu).
        text_token_lists: list[list[int]] = []
        for ex in examples:
            prompt_data = ex.get("prompt")
            if isinstance(prompt_data, list):
                messages = [
                    {
                        "role": m.get("role", m.get("from", "user")),
                        "content": str(m.get("content", m.get("value", "")))
                        .replace("<image>\n", "")
                        .replace("<image>", ""),
                    }
                    for m in prompt_data
                ]
            else:
                messages = [{"role": "user", "content": str(prompt_data or ex.get("instruction", ""))}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            text_token_lists.append(list(ids))

        max_text = max(len(ids) for ids in text_token_lists)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padded = [[pad_id] * (max_text - len(ids)) + ids for ids in text_token_lists]
        text_batch = torch.tensor(padded, dtype=torch.long, device=device)

        B = len(examples)
        mmu_tok = int(uni_prompting.sptids_dict["<|mmu|>"])
        soi = int(uni_prompting.sptids_dict["<|soi|>"])
        eoi = int(uni_prompting.sptids_dict["<|eoi|>"])
        input_ids = torch.cat(
            [
                torch.full((B, 1), mmu_tok, dtype=torch.long, device=device),
                torch.full((B, 1), soi, dtype=torch.long, device=device),
                image_tokens_shifted,
                torch.full((B, 1), eoi, dtype=torch.long, device=device),
                text_batch,
            ],
            dim=1,
        )

        prefix_len = 3 + image_tokens_shifted.shape[1]
        max_new_tokens = max_seq_length
        prefix_mask = torch.ones((B, prefix_len), dtype=torch.long, device=device)
        text_mask = (text_batch != pad_id).long()
        gen_mask = torch.ones((B, max_new_tokens), dtype=torch.long, device=device)
        attention_mask = torch.cat([prefix_mask, text_mask, gen_mask], dim=1)

        steps = max(1, max_new_tokens // 2)
        block_length = max(1, max_new_tokens // 4)

        ctx_mgr = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else _NullCtx()
        )
        with torch.no_grad(), ctx_mgr:
            output_ids = model.mmu_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_length=block_length,
                attention_mask=attention_mask,
                mask_id=mask_id,
            )

        gen_ids = output_ids[:, input_ids.shape[1]:]
        decoded_texts = tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        answer_contexts: list[dict[str, Any]] = []
        for i, ex in enumerate(examples):
            instruction = ex.get("instruction", "")
            answer_contexts.append({
                "decoded_text": decoded_texts[i],
                "decoded_image": ex.get("gen_image"),
                "instruction": instruction,
                "prompt": instruction,
                "prompt_len_tokens": int((text_batch[i] != pad_id).sum().item()),
                "completion_len_tokens": int((gen_ids[i] != pad_id).sum().item()),
                "input_image_tokens": image_tokens_shifted[i].detach(),
                "sample_id": ex.get("sample_id"),
                "payload": {
                    "sample_id": ex.get("sample_id"),
                    "instruction": instruction,
                    "image": ex.get("image"),
                },
            })
        return gen_ids.detach(), answer_contexts

    def _rollout_image_edit_latents(
        self,
        model,
        examples: list[dict[str, Any]],
        init_image=None,
        predicted_bbox: Optional[list[tuple[float, float, float, float]]] = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Image-edit rollout (input image + instruction -> output image) via
        ``model.interleave_generate``. Mirrors ``run_interleave`` in infer_all.py.

        MMaDA-Parallel has no separate ``image_edit`` path — the interleave
        generator jointly samples image + text tokens, and we return both. The
        decoded text is carried in the context so the unified scoring payload
        can reuse it alongside the text rollout's output if needed.

        ``predicted_bbox`` / ``init_image`` are accepted for signature
        compatibility with the parent but ignored — region-edit is explicitly
        disabled for MMaDA-Parallel.
        """
        if isinstance(examples, dict):
            examples = [examples]
        device = self.accelerator.device
        uni_prompting, vq_model = self._ensure_mmada_resources(device)
        tokenizer = self.processing_class
        cfg = self._build_mmada_gen_config()
        resolution = int(getattr(self.args, "mmada_resolution", _MMADA_RESOLUTION))
        num_vq = _MMADA_NUM_VQ_TOKENS
        codebook_size = _MMADA_CODEBOOK_SIZE

        # Encode input images.
        pixel_batch = torch.stack(
            [self._mmada_load_image_tensor(ex["image"], device, resolution) for ex in examples],
            dim=0,
        )
        image_tokens_shifted = vq_model.get_code(pixel_batch) + len(tokenizer)  # (B, N_vq)
        uncond_image_tokens = torch.zeros_like(image_tokens_shifted)

        # Build instruction / uncond text token sequences with BOS/EOS padding.
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        cond_lists: list[list[int]] = []
        uncond_lists: list[list[int]] = []
        for ex in examples:
            instruction = ex.get("instruction", "")
            ids = list(tokenizer(instruction)["input_ids"])
            if not ids or ids[0] != bos:
                ids = [bos] + ids
            ids = ids + [eos]
            cond_lists.append(ids)

            u_ids = list(tokenizer("")["input_ids"])
            if not u_ids or u_ids[0] != bos:
                u_ids = [bos] + u_ids
            u_ids = u_ids + [eos]
            uncond_lists.append(u_ids)
        max_len = max(len(ids) for ids in cond_lists)
        for i in range(len(examples)):
            cond_lists[i] = cond_lists[i] + [eos] * (max_len - len(cond_lists[i]))
            uncond_lists[i] = uncond_lists[i] + [eos] * (max_len - len(uncond_lists[i]))
        text_ids = torch.tensor(cond_lists, dtype=torch.long, device=device)
        uncond_text_ids = torch.tensor(uncond_lists, dtype=torch.long, device=device)

        B = text_ids.shape[0]
        interleave_col = torch.full(
            (B, 1), _MMADA_RESERVED_TOKENS["<|interleave|>"], dtype=torch.long, device=device
        )
        soi_col = torch.full((B, 1), _MMADA_RESERVED_TOKENS["<|soi|>"], dtype=torch.long, device=device)
        eoi_col = torch.full((B, 1), _MMADA_RESERVED_TOKENS["<|eoi|>"], dtype=torch.long, device=device)
        input_ids = torch.cat([interleave_col, soi_col, image_tokens_shifted, eoi_col, text_ids], dim=1)
        uncond_input_ids = torch.cat(
            [interleave_col, soi_col, uncond_image_tokens, eoi_col, uncond_text_ids], dim=1
        )

        from models import get_mask_schedule  # lazy

        with torch.no_grad():
            output_image_ids, output_text_ids = model.interleave_generate(
                input_ids,
                uncond_input_ids,
                text_cfg=float(getattr(self.args, "mmada_text_cfg", 2.5)),
                image_cfg=float(getattr(self.args, "mmada_image_cfg", 4.0)),
                noise_schedule=get_mask_schedule(getattr(self.args, "mmada_mask_schedule", "cosine")),
                text_steps=int(getattr(self.args, "mmada_text_steps", 128)),
                image_steps=int(getattr(self.args, "mmada_image_steps", 30)),
                reserved_token_mapping=_MMADA_RESERVED_TOKENS,
                uni_prompting=uni_prompting,
                config=cfg,
            )

        output_image_ids = torch.clamp(output_image_ids, 0, codebook_size - 1).to(torch.long)
        output_image_tokens_shifted = output_image_ids + len(tokenizer)
        decoded_texts = tokenizer.batch_decode(output_text_ids, skip_special_tokens=True)
        decoded_images = self._mmada_decode_vq(vq_model, output_image_ids)

        rollout_dir = Path("/tmp/mmada_grpo_rollouts")
        rollout_dir.mkdir(parents=True, exist_ok=True)

        image_contexts: list[dict[str, Any]] = []
        for i, (ex, pil_img, decoded_text) in enumerate(zip(examples, decoded_images, decoded_texts)):
            sample_id = str(ex.get("sample_id", ex.get("id", i)))
            safe_sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
            img_path = rollout_dir / f"{safe_sid}_{os.getpid()}_{i}.png"
            try:
                pil_img.save(img_path)
            except Exception:
                pass
            image_contexts.append({
                "valid": True,
                "decoded_image": pil_img,
                "decoded_image_path": str(img_path),
                "decoded_text": decoded_text,
                "sample_id": sample_id,
                "prompt": ex.get("instruction", ""),
                "prompt_len_tokens": int((text_ids[i] != eos).sum().item()),
                "completion_len_tokens": int(num_vq),
                "input_image_tokens": image_tokens_shifted[i].detach(),
                "output_image_tokens": output_image_tokens_shifted[i].detach(),
                "payload": {
                    "sample_id": sample_id,
                    "instruction": ex.get("instruction", ""),
                    "image": ex.get("image"),
                    "image_gen": str(img_path),
                },
            })
        return output_image_ids.detach(), image_contexts

    # ------------------------------------------------------------------
    # Per-token logp scoring
    # ------------------------------------------------------------------
    def _encode_scoring_sample(self, sample: dict, device: torch.device) -> dict:
        """Convert a scoring sample dict into device tensors + strings.

        The caller-provided dicts (built in ``_generate_and_score_completions``)
        already carry pre-encoded ``input_image_tokens`` / ``output_image_tokens``
        (shifted by ``len(tokenizer)``), so there's no re-encoding here. For
        und-only scoring (no image rollout on that sample) ``output_image_tokens``
        is None — we substitute zeros so the image slot is all-mask (see
        infer_all.compute_mode_logprobs for the analogous pattern).
        """
        num_vq = _MMADA_NUM_VQ_TOKENS
        in_img = sample.get("input_image_tokens")
        if in_img is None:
            in_img = torch.zeros(num_vq, dtype=torch.long, device=device)
        else:
            in_img = in_img.to(device=device, dtype=torch.long)

        out_img = sample.get("output_image_tokens")
        if out_img is None:
            out_img = torch.zeros(num_vq, dtype=torch.long, device=device)
        else:
            out_img = out_img.to(device=device, dtype=torch.long)

        return {
            "input_image_tokens": in_img,
            "output_image_tokens": out_img,
            "input_text": sample.get("input_text", ""),
            "output_text": sample.get("output_text", ""),
            "is_unified": bool(sample.get("is_unified", False)),
            "sample_id": sample.get("sample_id"),
        }

    def _mmada_score_modality(
        self,
        model,
        selected: list[dict],
        mode: str,
        mask_seeds: list,
        force_masks: Optional[list],
        device: torch.device,
        num_vq: int,
        max_text_len: int,
        mask_id: int,
        p_mask: float,
    ) -> Tuple[Optional[list[torch.Tensor]], Optional[list[list[dict]]]]:
        """Run ``prepare_inputs_and_labels_for_interleave_data`` + forward per
        sample, returning per-seed per-token logps and the masks used.

        ``mode``:
          - ``"gen"``: score output image slot only (text slot fully -100).
          - ``"und"``: score output text slot only.
          - ``"unified"``: score both; returned logp is ``concat([gen, und], -1)``.

        Masks come from ``force_masks[seed_idx][sample_idx]`` when provided;
        otherwise drawn deterministically from ``(seed, sample_idx)``.
        """
        if not selected:
            return None, None

        from training.interleave_utils import prepare_inputs_and_labels_for_interleave_data  # lazy

        img_start, img_end = _mmada_output_img_slot(num_vq, max_text_len)
        text_start, text_end = _mmada_output_text_slot(num_vq, max_text_len)
        zeros_img = torch.zeros(1, num_vq, dtype=torch.bool, device=device)
        zeros_text = torch.zeros(1, max_text_len, dtype=torch.bool, device=device)

        per_seed_outputs: list[torch.Tensor] = []
        captured_masks_per_seed: list[list[dict]] = []
        for seed_idx, seed in enumerate(mask_seeds):
            seed_int = int(seed.item()) if torch.is_tensor(seed) else int(seed)
            per_sample_logps: list[torch.Tensor] = []
            captured_masks: list[dict] = []
            seed_force = force_masks[seed_idx] if force_masks is not None else None

            for sample_local_idx, sample in enumerate(selected):
                input_img = sample["input_image_tokens"].to(device).unsqueeze(0)
                output_img = sample["output_image_tokens"].to(device).unsqueeze(0)
                input_text = sample["input_text"]
                output_text = sample["output_text"]

                slot = seed_force[sample_local_idx] if seed_force is not None else None

                # Resolve image-slot mask.
                if slot is not None and slot.get("gen_mask") is not None:
                    img_mask = slot["gen_mask"].to(device=device, dtype=torch.bool)
                    if img_mask.dim() == 1:
                        img_mask = img_mask.unsqueeze(0)
                elif mode in ("gen", "unified"):
                    img_mask = self._mmada_seeded_mask(
                        (1, num_vq), p_mask, device, seed_int, sample_local_idx,
                        skip_first=False,
                    )
                else:
                    img_mask = zeros_img.clone()

                # Resolve text-slot mask.
                if slot is not None and slot.get("text_mask") is not None:
                    text_mask = slot["text_mask"].to(device=device, dtype=torch.bool)
                    if text_mask.dim() == 1:
                        text_mask = text_mask.unsqueeze(0)
                elif mode in ("und", "unified"):
                    text_mask = self._mmada_seeded_mask(
                        (1, max_text_len), p_mask, device,
                        seed_int, sample_local_idx + 1_000_003,
                        skip_first=True,
                    )
                else:
                    text_mask = zeros_text.clone()

                input_ids, labels, attn_mask, _ = prepare_inputs_and_labels_for_interleave_data(
                    input_pixel_values=None,
                    input_text=[input_text],
                    output_pixel_values=None,
                    output_text=[output_text],
                    text_tokenizer=self.processing_class,
                    mask_id=mask_id,
                    reserved_token_mapping=_MMADA_RESERVED_TOKENS,
                    input_image_tokens=input_img.clone(),
                    output_image_tokens=output_img.clone(),
                    external_output_image_mask=img_mask,
                    external_output_text_mask=text_mask,
                    cond_dropout_prob=0.0,
                    max_text_len=max_text_len,
                )

                output = model.forward(input_ids=input_ids, attention_mask=attn_mask)
                logits = output.logits

                # Avoid materializing full (1, L, V) log_probs: slice to the
                # scored slot, then compute logp[label] = logits[label] -
                # logsumexp(logits) so only (1, L_slot) tensors are allocated.
                def _slot_logp(sl_start: int, sl_end: int) -> torch.Tensor:
                    sl_logits = logits[:, sl_start:sl_end]
                    sl_labels = labels[:, sl_start:sl_end]
                    safe = sl_labels.clamp(min=0)
                    gathered = sl_logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
                    lse = torch.logsumexp(sl_logits.float(), dim=-1)
                    mask = (sl_labels != -100).float()
                    return (gathered.float() - lse) * mask

                parts: list[torch.Tensor] = []
                if mode in ("gen", "unified"):
                    parts.append(_slot_logp(img_start, img_end))  # (1, N_vq)
                if mode in ("und", "unified"):
                    parts.append(_slot_logp(text_start, text_end))  # (1, max_text_len)
                sample_logp = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
                per_sample_logps.append(sample_logp)

                captured_masks.append({
                    "gen_mask": img_mask.detach().to("cpu") if mode in ("gen", "unified") else None,
                    "text_mask": text_mask.detach().to("cpu") if mode in ("und", "unified") else None,
                })

                del logits, output
            per_seed_outputs.append(torch.cat(per_sample_logps, dim=0))  # (N_sel, D)
            captured_masks_per_seed.append(captured_masks)
        return per_seed_outputs, captured_masks_per_seed

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
        """Compute per-token logps for gen and und scoring lists.

        Unlike the LaVida DiffuGRPOTrainer that drives scoring through
        ``LazySupervisedDataset`` + ``MaskDataCollator``, the MMaDA scoring path
        runs one forward per sample directly on the fixed interleave layout
        (``prepare_inputs_and_labels_for_interleave_data``). The masks are
        captured on the old-policy pass and replayed via ``force_*_masks`` on
        ref + current passes so Old / Current / Reference log-probs all score
        bitwise-identical positions.

        Returns the same 6-tuple as the parent:
          (per_gen_logps, per_und_logps, cached_gen_samples, cached_und_samples,
           captured_gen_masks, captured_und_masks)
        """
        device = self.accelerator.device
        self._ensure_mmada_resources(device)
        num_vq = _MMADA_NUM_VQ_TOKENS
        max_text_len = int(getattr(self.args, "mmada_max_seq_length", _MMADA_MAX_SEQ_LENGTH))
        mask_id = int(getattr(self.args, "mask_id", _MMADA_MASK_TOKEN_ID))
        p_mask = float(getattr(self.args, "p_mask_prompt", 0.15))

        N_gen_full = len(gen_scoring) if gen_scoring is not None else 0
        N_und_full = len(und_scoring) if und_scoring is not None else 0
        assert gen_scoring is not None or und_scoring is not None, (
            "At least one of gen_scoring or und_scoring must be provided."
        )

        if gen_scoring is not None and cached_gen_samples is None:
            cached_gen_samples = [
                self._encode_scoring_sample(gen_scoring[i], device) for i in range(N_gen_full)
            ]
        if und_scoring is not None and cached_und_samples is None:
            cached_und_samples = [
                self._encode_scoring_sample(und_scoring[i], device) for i in range(N_und_full)
            ]

        gs, ge = gen_slice if gen_slice is not None else (0, N_gen_full)
        us, ue = und_slice if und_slice is not None else (0, N_und_full)
        gen_selected = cached_gen_samples[gs:ge] if cached_gen_samples is not None else []
        und_selected = cached_und_samples[us:ue] if cached_und_samples is not None else []

        # Unified detection: the gen-side sample was built with both output
        # image AND output text (``is_unified=True``) and there is no und
        # scoring list. In this mode the "gen" branch emits concatenated
        # [image, text] per-token logps — the und branch is a no-op.
        unified_mode = bool(gen_selected) and bool(gen_selected[0].get("is_unified", False))

        per_gen_logps, captured_gen_masks = (None, None)
        if gen_selected:
            per_gen_logps, captured_gen_masks = self._mmada_score_modality(
                model, gen_selected,
                mode=("unified" if unified_mode else "gen"),
                mask_seeds=mask_seeds, force_masks=force_gen_masks,
                device=device, num_vq=num_vq, max_text_len=max_text_len,
                mask_id=mask_id, p_mask=p_mask,
            )
        per_und_logps, captured_und_masks = (None, None)
        if und_selected:
            per_und_logps, captured_und_masks = self._mmada_score_modality(
                model, und_selected,
                mode="und",
                mask_seeds=mask_seeds, force_masks=force_und_masks,
                device=device, num_vq=num_vq, max_text_len=max_text_len,
                mask_id=mask_id, p_mask=p_mask,
            )

        def _stack(per_seed):
            if per_seed is None:
                return None
            max_D = max(t.shape[-1] for t in per_seed)
            padded = []
            for t in per_seed:
                if t.shape[-1] < max_D:
                    pad_len = max_D - t.shape[-1]
                    pad = torch.zeros(*t.shape[:-1], pad_len, device=t.device, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=-1)
                padded.append(t)
            return torch.stack(padded, dim=0)  # (N_seeds, N_sel, D)

        return (
            _stack(per_gen_logps),
            _stack(per_und_logps),
            cached_gen_samples,
            cached_und_samples,
            captured_gen_masks,
            captured_und_masks,
        )

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
        """MMaDA variant of the DiffuGRPOTrainer rollout + scoring pipeline.

        Pipeline (mirrors the parent; region-edit / grounding removed):
          1. Image rollout  — ``_rollout_image_edit_latents`` (interleave_generate)
          2. Optional: inject rollout image into und inputs as ``gen_image``
          3. Text rollout   — ``_rollout_multimodal_text_gen`` (mmu_generate)
          4. Build per-modality scoring sample dicts carrying pre-encoded
             input/output VQ tokens so ``_get_per_token_logps`` can run the
             forward without re-encoding
          5. Old / ref per-token logps with shared ``mask_seeds``; captured
             masks returned so ``_compute_loss`` replays them for the current
             policy forward (bitwise-identical mask positions across passes)
          6. Rewards (perceptual for gen; format+correctness for und), grouped
             advantages per modality, with ``unified_mode`` summing perceptual
             and format+correctness signals when text_rollout_use_gen_image
             is active

        Supported modes (matching DiffuGRPOTrainer):
          - gen-only  (und_inputs=None)            — thinkmorph_edit
          - und-only  (gen_inputs=None)            — thinkmorph_answer
          - gen + und, text_rollout_use_gen_image=False  — independent rollouts
          - gen + und, text_rollout_use_gen_image=True   — unified payload
            (single forward scores both image + text slots)
        """
        device = self.accelerator.device
        beta = float(getattr(self.args, "beta", 0.0))
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        mode_str = "eval" if self.control.should_evaluate else "train"
        _timings: dict[str, float] = {}

        # Sanity-check gen/und alignment by sample_id (same contract as parent).
        if gen_inputs is not None and und_inputs is not None:
            for i, (g, u) in enumerate(zip(gen_inputs, und_inputs)):
                g_id = g.get("sample_id") if hasattr(g, "get") else None
                u_id = u.get("sample_id") if hasattr(u, "get") else None
                if g_id is not None and u_id is not None and g_id != u_id:
                    raise ValueError(
                        f"gen/und alignment broken at row {i}: "
                        f"gen sample_id={g_id!r} vs und sample_id={u_id!r}"
                    )

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            # ---- Image rollout ----
            image_contexts: Optional[list[Optional[dict]]] = None
            if gen_inputs is not None:
                image_contexts = [None] * len(gen_inputs)
                img_bs = max(1, int(getattr(self.args, "image_edit_batch_size", 2)))
                with _timer(_timings, "image_rollout"):
                    for s in trange(0, len(gen_inputs), img_bs, desc="MMaDA Image Rollout"):
                        chunk = gen_inputs[s : s + img_bs]
                        _, ctxs = self._rollout_image_edit_latents(unwrapped_model, chunk)
                        for off, ctx in enumerate(ctxs):
                            image_contexts[s + off] = ctx

            # ---- Inject gen_image into und_inputs (unified mode trigger) ----
            if (
                getattr(self.args, "text_rollout_use_gen_image", False)
                and und_inputs is not None
                and image_contexts is not None
            ):
                if len(und_inputs) != len(image_contexts):
                    raise ValueError(
                        f"text_rollout_use_gen_image: length mismatch "
                        f"{len(und_inputs)} vs {len(image_contexts)}"
                    )
                paired: list[dict] = []
                for sample, ctx in zip(und_inputs, image_contexts):
                    if ctx is None or ctx.get("decoded_image") is None:
                        raise ValueError(
                            "text_rollout_use_gen_image: image rollout produced "
                            "no decoded_image for a sample."
                        )
                    merged = dict(sample)
                    merged["gen_image"] = ctx["decoded_image"]
                    paired.append(merged)
                und_inputs = paired

            # ---- Text rollout ----
            answer_contexts: Optional[list[Optional[dict]]] = None
            if und_inputs is not None:
                answer_contexts = [None] * len(und_inputs)
                txt_bs = max(1, int(getattr(self.args, "text_rollout_batch_size", 2)))
                with _timer(_timings, "text_rollout"):
                    for s in trange(0, len(und_inputs), txt_bs, desc="MMaDA Text Rollout"):
                        chunk = und_inputs[s : s + txt_bs]
                        _, ctxs = self._rollout_multimodal_text_gen(
                            unwrapped_model, chunk, None, {}, device,
                        )
                        for off, ctx in enumerate(ctxs):
                            answer_contexts[s + off] = ctx

            # ---- Length metrics ----
            def _log_len(prefix: str, vals: list[int]) -> None:
                if not vals:
                    return
                local = torch.tensor(vals, dtype=torch.float32, device=device)
                gathered = self.accelerator.gather(local)
                self._metrics[mode_str][f"{prefix}/mean_length"].append(gathered.mean().item())
                self._metrics[mode_str][f"{prefix}/min_length"].append(gathered.min().item())
                self._metrics[mode_str][f"{prefix}/max_length"].append(gathered.max().item())

            if image_contexts is not None:
                _log_len(
                    "gen/prompt",
                    [c.get("prompt_len_tokens", 0) for c in image_contexts if c is not None],
                )
                _log_len(
                    "gen/completion",
                    [c.get("completion_len_tokens", 0) for c in image_contexts if c is not None],
                )
            if answer_contexts is not None:
                _log_len(
                    "und/prompt",
                    [c.get("prompt_len_tokens", 0) for c in answer_contexts if c is not None],
                )
                _log_len(
                    "und/completion",
                    [c.get("completion_len_tokens", 0) for c in answer_contexts if c is not None],
                )

            # ---- Build scoring sample lists ----
            # Unified mode: fold paired (gen, und) rollouts into ONE scoring
            # sample whose gpt turn carries both rollout-decoded image AND
            # rollout-decoded text. Fed through the "gen" scoring slot;
            # ``_mmada_score_modality`` in mode="unified" emits concatenated
            # [image_logps, text_logps] for the unified GRPO loss.
            unified_mode = bool(
                getattr(self.args, "text_rollout_use_gen_image", False)
                and gen_inputs is not None
                and und_inputs is not None
                and image_contexts is not None
                and answer_contexts is not None
            )

            gen_scoring: Optional[list[dict]] = None
            und_scoring: Optional[list[dict]] = None
            if unified_mode:
                gen_scoring = []
                for gen_ex, img_ctx, ans_ctx in zip(gen_inputs, image_contexts, answer_contexts):
                    gen_scoring.append({
                        "sample_id": gen_ex.get("sample_id"),
                        "input_text": ans_ctx.get("instruction", gen_ex.get("instruction", "")),
                        "output_text": ans_ctx["decoded_text"],
                        "input_image_tokens": img_ctx["input_image_tokens"],
                        "output_image_tokens": img_ctx["output_image_tokens"],
                        "is_unified": True,
                    })
            else:
                if gen_inputs is not None:
                    gen_scoring = []
                    for gen_ex, img_ctx in zip(gen_inputs, image_contexts):
                        gen_scoring.append({
                            "sample_id": gen_ex.get("sample_id"),
                            "input_text": gen_ex.get("instruction", ""),
                            "output_text": "",
                            "input_image_tokens": img_ctx["input_image_tokens"],
                            "output_image_tokens": img_ctx["output_image_tokens"],
                            "is_unified": False,
                        })
                if und_inputs is not None:
                    und_scoring = []
                    for und_ex, ans_ctx in zip(und_inputs, answer_contexts):
                        und_scoring.append({
                            "sample_id": und_ex.get("sample_id"),
                            "input_text": und_ex.get("instruction", ""),
                            "output_text": ans_ctx["decoded_text"],
                            "input_image_tokens": ans_ctx["input_image_tokens"],
                            "output_image_tokens": None,  # zeros — und slot isn't masked
                            "is_unified": False,
                        })
            assert gen_scoring is not None or und_scoring is not None, (
                "No scoring examples were built."
            )

        # ---- Shared mask seeds across modalities and ranks ----
        mask_seeds = torch.randint(0, 2**12, (num_iterations,), device=device)
        mask_seed_list = mask_seeds.detach().cpu().tolist()

        with torch.no_grad():
            with _timer(_timings, "old_logps"):
                (
                    old_gen_logps,
                    old_und_logps,
                    cached_gen_samples,
                    cached_und_samples,
                    cached_gen_masks,
                    cached_und_masks,
                ) = self._get_per_token_logps(
                    self.model,
                    gen_scoring=gen_scoring,
                    und_scoring=und_scoring,
                    mask_seeds=mask_seed_list,
                )

            with _timer(_timings, "ref_logps"):
                ref_gen_logps, ref_und_logps = None, None
                if beta != 0.0:
                    if getattr(self, "ref_model", None) is not None:
                        ref_gen_logps, ref_und_logps, _, _, _, _ = self._get_per_token_logps(
                            self.ref_model,
                            gen_scoring=gen_scoring, und_scoring=und_scoring,
                            mask_seeds=mask_seed_list,
                            cached_gen_samples=cached_gen_samples,
                            cached_und_samples=cached_und_samples,
                            force_gen_masks=cached_gen_masks,
                            force_und_masks=cached_und_masks,
                        )
                    else:
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        if hasattr(unwrapped, "disable_adapter"):
                            with unwrapped.disable_adapter():
                                ref_gen_logps, ref_und_logps, _, _, _, _ = self._get_per_token_logps(
                                    self.model,
                                    gen_scoring=gen_scoring, und_scoring=und_scoring,
                                    mask_seeds=mask_seed_list,
                                    cached_gen_samples=cached_gen_samples,
                                    cached_und_samples=cached_und_samples,
                                    force_gen_masks=cached_gen_masks,
                                    force_und_masks=cached_und_masks,
                                )
                        else:
                            ref_gen_logps, ref_und_logps, _, _, _, _ = self._get_per_token_logps(
                                self.model,
                                gen_scoring=gen_scoring, und_scoring=und_scoring,
                                mask_seeds=mask_seed_list,
                                cached_gen_samples=cached_gen_samples,
                                cached_und_samples=cached_und_samples,
                                force_gen_masks=cached_gen_masks,
                                force_und_masks=cached_und_masks,
                            )

            def _offload_cpu_pinned(t):
                if t is None:
                    return None
                cpu = t.detach().to("cpu")
                try:
                    cpu = cpu.pin_memory()
                except RuntimeError:
                    pass
                return cpu

            old_gen_logps = _offload_cpu_pinned(old_gen_logps)
            old_und_logps = _offload_cpu_pinned(old_und_logps)
            ref_gen_logps = _offload_cpu_pinned(ref_gen_logps)
            ref_und_logps = _offload_cpu_pinned(ref_und_logps)

        # ---- Rewards per modality ----
        with _timer(_timings, "reward"):
            gen_local_rewards = None
            und_local_rewards = None
            if gen_inputs is not None:
                gen_reward_inputs = list(gen_inputs)
                gen_prompts = [c.get("prompt", "") for c in image_contexts]
                gen_completions = [c.get("decoded_image") for c in image_contexts]
                gen_reward_fns = [perceptual_score_reward_func]

                # Rollout-only rows (no image_gt) contribute a nan perceptual
                # score; nansum collapses them to 0 and grouped advantages
                # neutralize the GRPO loss on those rows.
                gen_has_gt = [ex.get("image_gt") is not None for ex in gen_reward_inputs]
                sub_idx = [i for i, ok in enumerate(gen_has_gt) if ok]

                gen_rewards_per_func = torch.zeros(
                    len(gen_completions), len(gen_reward_fns), device=device
                )
                for i, reward_func in enumerate(gen_reward_fns):
                    rname = reward_func.__name__
                    with profiling_context(self, rname):
                        if sub_idx:
                            sub_inputs = [gen_reward_inputs[j] for j in sub_idx]
                            sub_prompts = [gen_prompts[j] for j in sub_idx]
                            sub_completions = [gen_completions[j] for j in sub_idx]
                            keys = [k for k in sub_inputs[0] if k not in ["prompt", "completion"]]
                            reward_kwargs = {k: [ex.get(k) for ex in sub_inputs] for k in keys}
                            try:
                                out = reward_func(
                                    prompts=sub_prompts, completions=sub_completions,
                                    step=self._step, run_name=self.args.output_dir, **reward_kwargs,
                                )
                            except Exception:
                                out = [float("nan")] * len(sub_completions)
                            out = [r if r is not None else float("nan") for r in out]
                            gen_rewards_per_func[sub_idx, i] = torch.tensor(
                                out, dtype=torch.float32, device=device
                            )
                    self._metrics[mode_str][f"rewards/{rname}/mean"].append(
                        torch.nanmean(gen_rewards_per_func[:, i]).item()
                    )
                    self._metrics[mode_str][f"rewards/{rname}/std"].append(
                        nanstd(gen_rewards_per_func[:, i]).item()
                    )
                gen_local_rewards = gen_rewards_per_func.nansum(dim=1)

            if und_inputs is not None:
                und_reward_inputs = list(und_inputs)
                und_prompts = [c.get("prompt", "") for c in answer_contexts]
                und_completions = [
                    [{"role": "assistant", "content": c["decoded_text"]}]
                    for c in answer_contexts
                ]
                und_reward_fns = [strict_format_reward_func, correctness_reward_func]

                und_rewards_per_func = torch.zeros(
                    len(und_completions), len(und_reward_fns), device=device
                )
                for i, reward_func in enumerate(und_reward_fns):
                    rname = reward_func.__name__
                    with profiling_context(self, rname):
                        keys = [k for k in und_reward_inputs[0] if k not in ["prompt", "completion"]]
                        reward_kwargs = {k: [ex.get(k) for ex in und_reward_inputs] for k in keys}
                        try:
                            out = reward_func(
                                prompts=und_prompts, completions=und_completions,
                                step=self._step, run_name=self.args.output_dir, **reward_kwargs,
                            )
                        except Exception:
                            out = [float("nan")] * len(und_completions)
                        out = [r if r is not None else float("nan") for r in out]
                        und_rewards_per_func[:, i] = torch.tensor(
                            out, dtype=torch.float32, device=device
                        )
                    self._metrics[mode_str][f"rewards/{rname}/mean"].append(
                        torch.nanmean(und_rewards_per_func[:, i]).item()
                    )
                    self._metrics[mode_str][f"rewards/{rname}/std"].append(
                        nanstd(und_rewards_per_func[:, i]).item()
                    )
                und_local_rewards = und_rewards_per_func.nansum(dim=1)

        # ---- Grouped advantages per modality ----
        def _compute_advantages(local_rewards: torch.Tensor, tag: str) -> torch.Tensor:
            local_n = local_rewards.size(0)
            rewards = gather(local_rewards)
            mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
            std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)
            process_slice = slice(
                self.accelerator.process_index * local_n,
                (self.accelerator.process_index + 1) * local_n,
            )
            centered = rewards - mean_grouped
            advantages = centered[process_slice]
            is_std_zero = std_grouped < 1e-6
            self._metrics[mode_str][f"{tag}/reward"].append(rewards.mean().item())
            self._metrics[mode_str][f"{tag}/reward_std"].append(rewards.std().item())
            self._metrics[mode_str][f"{tag}/frac_reward_zero_std"].append(
                is_std_zero.float().mean().item()
            )
            self._metrics[mode_str][f"{tag}/advantages_mean"].append(centered.mean().item())
            self._metrics[mode_str][f"{tag}/advantages_std"].append(centered.std().item())
            self._metrics[mode_str][f"{tag}/advantages_abs_mean"].append(centered.abs().mean().item())
            self._metrics[mode_str][f"{tag}/advantages_min"].append(centered.min().item())
            self._metrics[mode_str][f"{tag}/advantages_max"].append(centered.max().item())
            return advantages

        if unified_mode:
            # Unified advantage = (perceptual) + (format + correctness). NaN→0
            # so rollout-only rows (image_gt=None) still contribute the text
            # signal while their perceptual term is 0.
            assert gen_local_rewards is not None and und_local_rewards is not None
            assert gen_local_rewards.size(0) == und_local_rewards.size(0)
            unified_local_rewards = (
                torch.nan_to_num(gen_local_rewards, nan=0.0)
                + torch.nan_to_num(und_local_rewards, nan=0.0)
            )
            gen_advantages = _compute_advantages(unified_local_rewards, "unified")
            und_advantages = None
        else:
            gen_advantages = (
                _compute_advantages(gen_local_rewards, "gen")
                if gen_inputs is not None
                else None
            )
            und_advantages = (
                _compute_advantages(und_local_rewards, "und")
                if und_inputs is not None
                else None
            )

        for k, v in _timings.items():
            self._metrics[mode_str][f"time_profile/{k}"].append(v)

        return {
            "gen": [gen_scoring, gen_advantages, old_gen_logps, ref_gen_logps]
                if (gen_inputs is not None or unified_mode) else None,
            "und": [und_scoring, und_advantages, old_und_logps, ref_und_logps]
                if (und_inputs is not None and not unified_mode) else None,
            "mask_seeds": mask_seed_list,
            "cached_gen_samples": cached_gen_samples,
            "cached_und_samples": cached_und_samples,
            "cached_gen_masks": cached_gen_masks,
            "cached_und_masks": cached_und_masks,
        }