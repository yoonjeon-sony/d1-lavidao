import copy
import inspect
import os
import random
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

# Required by LaVida-O sampling path.
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")
# Disable complementary masking (do_inv) during GRPO forward passes.
# do_inv doubles the batch size (B → 2B) which causes shape mismatches between
# per_token_logps (2B, L) and the stored old_per_token_logps (B, L).
os.environ.setdefault("SKIP_COMPLEMENTARY_MASKING", "1")

# ---------------------------------------------------------------------------
# Loss-explosion debug instrumentation
# ---------------------------------------------------------------------------
# Gated by the DIFFU_GRPO_DEBUG env var.  When enabled, the GRPO loss helper
# runs a battery of shape / magnitude checks around ``coef_1 = exp(curr - old)``
# and the KL term so we can localize the source of the astronomical loss /
# grad_norm values observed in training.  Kept out of the hot path in release
# mode so there's zero overhead when DIFFU_GRPO_DEBUG is off.
DIFFU_GRPO_DEBUG = os.environ.get("DIFFU_GRPO_DEBUG", "0") == "1"
# When debug is on, also promote the step-0 determinism check to strict (raise
# instead of warn) so any non-determinism aborts loudly on the first step.
if DIFFU_GRPO_DEBUG:
    os.environ.setdefault("DIFFU_GRPO_STEP0_STRICT", "1")


def _debug_log(msg: str) -> None:
    """Tagged print gated by DIFFU_GRPO_DEBUG. No-op when debug is off."""
    if DIFFU_GRPO_DEBUG:
        print(f"[DIFFU_GRPO_DEBUG] {msg}", flush=True)


def _debug_run(fn):
    """Run ``fn`` only when DIFFU_GRPO_DEBUG is on.  Used as a cheap guard so
    callers can write ``_debug_run(lambda: ...)`` without an explicit ``if``
    everywhere.  Any exception inside is printed (not raised) so the debug
    hooks can't take down a real training run when left on by accident.
    """
    if not DIFFU_GRPO_DEBUG:
        return None
    try:
        return fn()
    except Exception as e:
        print(f"[DIFFU_GRPO_DEBUG] hook raised: {type(e).__name__}: {e}", flush=True)
        return None

REPO_ROOT = Path(__file__).resolve().parents[1]
LAVIDA_ROOT = REPO_ROOT / "LaVida-O"
if str(LAVIDA_ROOT) not in sys.path:
    sys.path.insert(0, str(LAVIDA_ROOT))

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.train.data.datasets import LazySupervisedDataset

from llava.mm_utils import pad_to_square_and_resize, process_images, tokenizer_image_token
from llava.model.language_model.llava_llada import (
    LlavaLladaConfig,
    LlavaLladaForMaskedDiffusion,
)
from llava.model.language_model.llada.modeling_llada import LLaDAModelLM
from llava.model.language_model.llada.generate import (
    add_gumbel_noise,
    cosine_schedule_2,
    generate as llada_generate,
    exp_schedule,
    get_logits as llada_get_logits,
    get_num_transfer_tokens_sch,
    logit_normal_schedule,
    wte as llada_wte,
)
from llava.model.utils import maybe_truncate_last_dim, pad_along_last_dim

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



def stratified_random(n: int = 64, seed: Optional[int] = None, shuffle_blocks: bool = True) -> List[int]:
    """
    Progressive Multi‑Jittered (PMJ) ordering over an n×n integer grid, n must be a power of two.

    The algorithm recursively subdivides the full grid into 2×2 blocks. At each level, it ensures
    every sub‑block contains exactly one sample by placing a new integer‑grid point uniformly at
    random in each sub‑block that doesn't already contain a sample from a previous level. The
    resulting sequence is progressive: the first 4^k samples are 1‑per‑cell stratified over a
    (n/2^k)×(n/2^k) tiling of the domain.

    Returns
    -------
    List[int]
        Row‑major linear indices y*n + x for x,y in [0, n).
    """
    # Validate power of two
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two (e.g., 64)")

    rng = random.Random(seed)

    # Occupancy grid: False means empty, True means already sampled
    occupied = [[False] * n for _ in range(n)]

    # The progressive sequence (row‑major linear indices)
    seq: List[int] = []

    # A block is represented as (x0, y0, size)
    blocks: List[Tuple[int, int, int]] = [(0, 0, n)]

    def block_has_sample(x0: int, y0: int, size: int) -> bool:
        for yy in range(y0, y0 + size):
            row = occupied[yy]
            for xx in range(x0, x0 + size):
                if row[xx]:
                    return True
        return False

    def place_random_in_block(x0: int, y0: int, size: int):
        # Because we only call this on blocks known to be empty, a single draw suffices.
        x = rng.randrange(x0, x0 + size)
        y = rng.randrange(y0, y0 + size)
        # Safety: loop until we hit an empty cell (should be first try for empty block)
        attempts = 0
        while occupied[y][x]:
            x = rng.randrange(x0, x0 + size)
            y = rng.randrange(y0, y0 + size)
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Too many attempts to place a sample; logic error?")
        occupied[y][x] = True
        seq.append(y * n + x)

    # Iterate levels until block size == 1
    size = n
    while size > 1:
        # Subdivide each block into 4 children
        half = size // 2
        children: List[Tuple[int, int, int]] = []
        for (x0, y0, s) in blocks:
            assert s == size
            children.extend([
                (x0, y0, half),                # NW
                (x0 + half, y0, half),         # NE
                (x0, y0 + half, half),         # SW
                (x0 + half, y0 + half, half),  # SE
            ])
        # Optionally randomize visitation order to reduce directional bias in sequence order
        if shuffle_blocks:
            rng.shuffle(children)
        # For each child, if empty, place a random sample inside it
        for (x0, y0, s) in children:
            if not block_has_sample(x0, y0, s):
                place_random_in_block(x0, y0, s)
        # Next level
        blocks = children
        size = half

    # At this point, every 1×1 cell is a block; any still‑empty cells need to be appended
    # (these are exactly those not yet selected at previous levels).
    # To preserve the progressive property, all remaining cells are appended in a random order.
    # (Any order works because they are all 1×1; using random makes ties less structured.)
    remaining: List[int] = []
    for y in range(n):
        for x in range(n):
            if not occupied[y][x]:
                remaining.append(y * n + x)
    rng.shuffle(remaining)
    seq.extend(remaining)

    assert len(seq) == n * n, (len(seq), n * n)

    return seq

def _register_lavida_architectures() -> None:
    # TRL resolves ref-model classes via `getattr(transformers, config.architectures[0])`.
    if not hasattr(transformers, "LlavaLladaForMaskedDiffusion"):
        setattr(transformers, "LlavaLladaForMaskedDiffusion", LlavaLladaForMaskedDiffusion)
    if not hasattr(transformers, "LlavaLladaConfig"):
        setattr(transformers, "LlavaLladaConfig", LlavaLladaConfig)


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


class MaskDataCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def pad_sequence(self, input_ids, batch_first, padding_value,extra_pad=-1):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        # breakpoint()
        input_ids = list(input_ids)
        max_k = max(range(len(input_ids)),key=lambda x:input_ids[x].shape[-1])
       
        # extra_pad = -1
        if extra_pad > 0 :
            extra_pad_seq = torch.tensor([padding_value]*extra_pad)
            input_ids[max_k] = torch.cat([input_ids[max_k],extra_pad_seq])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        images_gen = list([instance["image_gen"] for instance in instances if instance["image_gen"] is not None])
        image_gen_enc = list([instance["image_gen_enc"] for instance in instances if instance["image_gen_enc"] is not None])
        image_gen_weight = list([instance["image_gen_weight"] for instance in instances if instance["image_gen_weight"] is not None])

        if len(images_gen) > 0:
            batch['images_gen'] = images_gen
        else:
            batch['images_gen']  = None
        if len(image_gen_enc) > 0:
            batch['images_gen_enc'] = image_gen_enc
        else:
            batch['images_gen_enc']  = None
        if len(image_gen_weight) > 0:
            batch['image_gen_weight'] = image_gen_weight
        else:
            batch['image_gen_weight'] = None

        if 'name' in instances[0]:
            batch['dataset_name'] = instances[0]['name']

        # Per-sample flag: True if the sample has image generation targets (image_edit),
        # False otherwise (text/multimodal-understanding).  Used in _get_per_token_logps
        # to route each sample to gen_loss_none_reduction vs und_loss_none_reduction.
        batch['sample_is_image_edit'] = torch.tensor(
            [instance.get('image_gen') is not None for instance in instances],
            dtype=torch.bool,
        )

        batch['do_not_mask_text'] = [x['do_not_mask_text'] for x in instances]

        return batch

class DiffuGRPOTrainer(GRPOTrainer):
    """GRPO trainer adapted for LaVida-O text + image rollouts with unified token-level scoring."""

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        train_dataset_und: Optional[Union[Dataset, IterableDataset]] = None,
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

        # Hard-disable every source of dropout in the policy model (and ref
        # model if present).  GRPO requires that the old-policy forward and
        # the current-policy forward produce bitwise-identical logps for the
        # same (sample, mask_seed) — otherwise
        #     coef_1 = exp(per_token_logps - old_ptl)
        # blows up at masked positions and the loss becomes enormous.  We
        # cannot use model.eval() on the current-policy pass because it must
        # build a grad graph for backward, so we instead hard-zero every
        # dropout source so train-mode forwards become deterministic too.
        # This is safe because GRPO derives its training signal from the
        # clipped-ratio loss, not from the underlying masked-LM dropout
        # regularization.
        self._hard_disable_dropout(self.model)
        if getattr(self, "ref_model", None) is not None:
            self._hard_disable_dropout(self.ref_model)

        # Initialize wandb on the main process only.
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


    @staticmethod
    def _make_generator(device: torch.device, seed: int) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    @staticmethod
    def _hard_disable_dropout(module: nn.Module) -> None:
        """Walk ``module`` and set every dropout source to 0 in-place.

        LLaDA has three dropout paths that all need to be killed for GRPO
        forwards to be deterministic between the old-policy and
        current-policy passes:

          1. ``nn.Dropout`` submodules — ``residual_dropout`` and
             ``embedding_dropout`` are instantiated as ``nn.Dropout``
             modules (see ``modeling_llada.py`` ``self.dropout`` and
             ``emb_drop``).  Setting ``.p = 0.0`` makes their forward the
             identity even when ``self.training`` is True.
          2. ``attention_dropout`` — LLaDABlock reads this dynamically on
             every forward via ``dropout_p=0.0 if not self.training else
             self.config.attention_dropout`` (see modeling_llada.py lines
             ~822 and ~838).  To defeat it we have to walk every nested
             ``config`` object and zero the attribute.
          3. Defensive catch-all: any ``*_dropout`` on any nested config,
             in case a future LLaDA variant adds more.

        This runs once at trainer ``__init__`` time — the model hasn't been
        wrapped by DeepSpeed yet, but even after wrapping ``module.modules()``
        still yields the same underlying submodules because the wrappers
        just hold references.
        """
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
            cfg = getattr(m, "config", None)
            if cfg is None:
                continue
            for attr in ("attention_dropout", "residual_dropout", "embedding_dropout"):
                if hasattr(cfg, attr):
                    try:
                        setattr(cfg, attr, 0.0)
                    except Exception:
                        # Some HF configs are frozen / use @property — ignore.
                        pass

    def _get_image_edit_gen_dict(self) -> dict[str, Any]:
        latent_token_map = {256: 256, 512: 1024, 1024: 4096}
        prompt_token_map = {256: 64, 512: 256, 1024: 1024}
        res = int(self.args.image_edit_resolution)
        n_tokens = latent_token_map.get(res, int(self.args.image_edit_n_tokens))
        prompt_n_tokens = prompt_token_map.get(res, max(1, n_tokens // 4))
        return {
            "sample_policy": self.args.image_edit_sample_policy,
            "confidence_policy": self.args.image_edit_confidence_policy,
            "guidance_scale": float(self.args.image_edit_guidance_scale),
            "guidance_scale_image": float(getattr(self.args, "image_edit_guidance_scale_image", 0.0)),
            "batch_size": int(self.args.image_edit_batch_size),
            "image_resolution": res,
            "n_tokens": n_tokens,
            "prompt_n_tokens": prompt_n_tokens,
            "shift": int(self.args.image_edit_shift),
            "n_steps": int(self.args.image_edit_n_steps),
            "schedule": self.args.image_edit_schedule,
            "alg_temp": float(self.args.image_edit_alg_temp),
            "dynamic_temperature": bool(self.args.image_edit_dynamic_temperature),
            "schedule_temp": self.args.image_edit_schedule_temp,
            "min_temperature": float(self.args.image_edit_min_temperature),
            "schedule_temp_samp": self.args.image_edit_schedule_temp_samp,
            "dynamic_temperature_samp": bool(self.args.image_edit_dynamic_temperature_samp),
            "min_temperature_samp": float(self.args.image_edit_min_temperature_samp),
            "cfg_interval": [
                float(self.args.image_edit_cfg_interval_start),
                float(self.args.image_edit_cfg_interval_end),
            ],
            "order_cutoff": float(self.args.image_edit_order_cutoff),
            "edit_mode": int(self.args.image_edit_edit_mode),
            "micro_cond": self.args.image_edit_micro_cond,
        }

    @staticmethod
    def _role_to_conv_role(role: str) -> str:
        role = role.lower()
        if role in {"human", "user"}:
            return "user"
        return "assistant"

    def _build_llada_prompt(self, prompt_messages: Any, has_gen_image: bool = False) -> str:
        conv_version = getattr(self.args, "version", "llada")
        if conv_version not in conv_templates:
            conv_version = "llada"
        conv = copy.deepcopy(conv_templates[conv_version])

        if isinstance(prompt_messages, str):
            conv.append_message(conv.roles[0], prompt_messages)
        else:
            for turn in prompt_messages:
                role = turn.get("role", turn.get("from", "user"))
                content = turn.get("content", turn.get("value", ""))
                mapped_role = self._role_to_conv_role(role)
                conv.append_message(mapped_role, content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if has_gen_image:
            # Inject a second <image> token so the model sees BOTH the
            # problem image and the image-rollout-generated image.
            # ``get_image_answer_placeholder_questions`` produces user content
            # of the form "<image>\n{instruction}", so replacing the first
            # occurrence of "<image>\n" with "<image>\n<image>\n" yields
            # "<image>\n<image>\n{instruction}" and leaves any trailing
            # template tokens intact.
            if "<image>\n" not in prompt:
                raise ValueError(
                    "has_gen_image=True but the built prompt does not contain "
                    "'<image>\\n' to duplicate. Check the conversation template "
                    "and the sample's 'prompt' field."
                )
            prompt = prompt.replace("<image>\n", "<image>\n<image>\n", 1)
        return prompt

    def _build_lazy_supervised_data_args(self, model, image_processor):
        base_model = model.get_model() if hasattr(model, "get_model") else model
        data_args = vars(self.args).copy()
        data_args.update(
            {
                "is_multimodal": True,
                "early_mix_text": data_args.get("early_mix_text", False),
                "image_folder": data_args.get("image_folder"),
                "image_processor": image_processor,
                "image_processor_gen": data_args.get(
                    "image_processor_gen", getattr(base_model, "image_processor_gen", None)
                ),
                "image_aspect_ratio": data_args.get("image_aspect_ratio", "square"),
                "image_grid_pinpoints": data_args.get("image_grid_pinpoints"),
                "image_crop_resolution": data_args.get("image_crop_resolution"),
                "image_split_resolution": data_args.get("image_split_resolution"),
                "video_folder": data_args.get("video_folder"),
                "video_fps": data_args.get("video_fps", 1),
                "frames_upbound": data_args.get("frames_upbound", 0),
                "add_time_instruction": data_args.get("add_time_instruction", False),
                "force_sample": data_args.get("force_sample", False),
                "image_gen_size": data_args.get("image_gen_size", 1024),
                "num_gen_image_tokens": data_args.get("num_gen_image_tokens", 1024),
                "num_gen_image_tokens_enc_ds": data_args.get("num_gen_image_tokens_enc_ds", 1),
                "mm_edit_area_weight": data_args.get("mm_edit_area_weight", 5.0),
            }
        )
        return SimpleNamespace(**data_args)

    def _normalize_text_rollout(
        self,
        generated: torch.Tensor,
        prompt_ids: torch.Tensor,
        prefix_lm: bool,
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

    def _load_image(self, image_like):
        if image_like is None:
            return None
        if isinstance(image_like, str):
            from PIL import Image

            return Image.open(image_like).convert("RGB")
        if isinstance(image_like, list) and len(image_like) > 0:
            return self._load_image(image_like[0])
        return image_like

    def _rollout_multimodal_text_gen(
        self,
        model,
        examples: Union[dict[str, Any], list[dict[str, Any]]],
        image_processor,
        generation_kwargs: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if isinstance(examples, dict):
            examples = [examples]

        prompt_texts = []
        all_input_ids = []
        all_images = []
        image_sizes = []
        source_images_per_example = []
        resolution = int(self.args.image_edit_resolution)
        if image_processor is None:
            raise ValueError("Text rollout requires an image processor for multimodal prompts.")

        for example in examples:
            # When the example carries a "gen_image" key (injected by
            # _generate_and_score_completions in the text_rollout_use_gen_image
            # mode), the text rollout sees BOTH the problem image and the
            # rollout-generated image.  _build_llada_prompt then duplicates
            # the "<image>\n" token so the prompt has two <image> slots.
            has_gen_image = example.get("gen_image") is not None
            prompt_text = self._build_llada_prompt(
                example["prompt"], has_gen_image=has_gen_image
            )
            if "<image>" not in prompt_text:
                sample_id = example.get("id", example.get("pid", "unknown"))
                raise ValueError(
                    "Text rollout example includes an image but the prompt does not contain '<image>': "
                    f"sample_id={sample_id}"
                )
            prompt_ids = tokenizer_image_token(
                prompt_text,
                self.processing_class,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).to(device)
            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[-self.max_prompt_length :]
            all_input_ids.append(prompt_ids.unsqueeze(0))
            prompt_texts.append(prompt_text)

            # Problem image — always required.
            image = self._load_image(example.get("image"))
            if image is None:
                sample_id = example.get("id", example.get("pid", "unknown"))
                raise ValueError(f"Text rollout example is missing an image: sample_id={sample_id}")
            processed_image = pad_to_square_and_resize(image.convert("RGB"), resolution)
            all_images.append(processed_image)
            image_sizes.append(processed_image.size)

            # Optional rollout-generated "gen_image" — appended AFTER the
            # problem image so the flat images list order matches the order
            # of <image> tokens in the prompt (problem first, gen second).
            if has_gen_image:
                gen_image = self._load_image(example.get("gen_image"))
                if gen_image is None:
                    sample_id = example.get("id", example.get("pid", "unknown"))
                    raise ValueError(
                        f"Text rollout example declared gen_image but it could not "
                        f"be loaded: sample_id={sample_id}"
                    )
                processed_gen = pad_to_square_and_resize(
                    gen_image.convert("RGB"), resolution
                )
                all_images.append(processed_gen)
                image_sizes.append(processed_gen.size)

            source_images = example.get("image")
            if source_images is None:
                source_images = []
            elif not isinstance(source_images, list):
                source_images = [source_images]
            source_images_per_example.append(source_images)

        prompt_ids = self._left_pad_2d(
            all_input_ids,
            self.processing_class.pad_token_id,
            torch.long,
        ).to(device)
        prompt_mask = (prompt_ids != self.processing_class.pad_token_id).long()
        position_ids = None
        image_tensor = process_images(all_images, image_processor, model.config)
        image_tensor = self._normalize_mm_image_payload(
            image_tensor,
            dtype=model.dtype,
            device=device,
        )
        
        inputs, position_ids, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids=prompt_ids,
            position_ids=position_ids,
            attention_mask=prompt_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
            modalities=["image"] * prompt_ids.shape[0],
            image_sizes=image_sizes,
        )
        del inputs

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
        _, completion_ids = self._normalize_text_rollout(generated, prompt_ids, prefix_lm)
        decoded_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Per-row prompt / completion token counts (non-pad). Reported as
        # und/prompt_length and und/completion_length metrics.
        pad_id = self.processing_class.pad_token_id
        per_row_prompt_lens = (prompt_ids != pad_id).sum(dim=1).tolist()
        per_row_completion_lens = (completion_ids != pad_id).sum(dim=1).tolist()

        image_contexts = []
        for batch_idx, example in enumerate(examples):
            prompt_data = example.get("prompt", "")
            human_value = prompt_data[0]["content"] if isinstance(prompt_data, list) and len(prompt_data) > 0 else prompt_data
            image_contexts.append(
                {
                    "decoded_text": decoded_texts[batch_idx],
                    "prompt": prompt_texts[batch_idx],
                    "prompt_len_tokens": int(per_row_prompt_lens[batch_idx]),
                    "completion_len_tokens": int(per_row_completion_lens[batch_idx]),
                    "payload": {
                        "id": str(example.get("sample_id", example.get("pid", example.get("id", "")))),
                        "image": source_images_per_example[batch_idx],
                        "conversations": [
                            {"from": "human", "value": human_value},
                            {"from": "gpt", "value": decoded_texts[batch_idx]},
                        ],
                    },
                }
            )

        return completion_ids.detach(), image_contexts


    def _normalize_mm_image_payload(self, image_tensor: Any, *, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        if isinstance(image_tensor, list):
            return [_x.to(dtype=dtype, device=device) for _x in image_tensor]
        image_tensor = image_tensor.to(dtype=dtype, device=device)
        return [img for img in image_tensor]



    def _get_image_processor(self, model):
        if hasattr(model, "get_vision_tower"):
            vt = model.get_vision_tower()
            if vt is not None and hasattr(vt, "image_processor"):
                return vt.image_processor
        if hasattr(model, "get_model"):
            base = model.get_model()
            if hasattr(base, "get_vision_tower"):
                vt = base.get_vision_tower()
                if vt is not None and hasattr(vt, "image_processor"):
                    return vt.image_processor
        return None

    @staticmethod
    def _left_pad_2d(tensors: list[torch.Tensor], pad_value: int, dtype_: torch.dtype) -> torch.Tensor:
        if len(tensors) == 0:
            return torch.empty(0, 0, dtype=dtype_)
        max_len = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                pad_t = torch.full((t.shape[0], pad_len), pad_value, dtype=dtype_, device=t.device)
                t = torch.cat([pad_t, t], dim=1)
            out.append(t)
        return torch.cat(out, dim=0)

    @staticmethod
    def _image_edit_gumbel_noise(tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.zeros_like(tensor, dtype=torch.float32).uniform_(0, 1)
        return -torch.log(-torch.log(noise))

    def _extract_image_edit_instruction(self, example: dict[str, Any]) -> str:
        instruction = example.get("instruction")
        if instruction is not None:
            return instruction
        prompt_data = example.get("prompt", [{"role": "user", "content": ""}])
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            return prompt_data[-1].get("content", "")
        return ""

    def _make_invalid_image_edit_ctx(
        self,
        latent_template: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "valid": False,
            "latent_shape": tuple(latent_template.shape),
            "decoded_image": None,
        }

    def _build_image_edit_temperature_schedule(
        self,
        n_steps: int,
        schedule_name: str,
        min_temperature: float,
        shift: int,
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
            raise NotImplementedError(f"Unknown image-edit temperature schedule: {schedule_name}")
        return torch.tensor(temperatures, dtype=torch.float32)

    def _rollout_image_edit_latents(
        self,
        model,
        examples: list[dict[str, Any]],
        init_image=None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if isinstance(examples, dict):
            examples = [examples]
        gen_cfg = self._get_image_edit_gen_dict()
        device = model.device
        batch_size = len(examples)
        reserve_id = 126089
        reserve_id2 = 126090
        img_mask_id = 8193
        reserve_token = "<|reserved_token_5|>"
        reserve_token_2 = "<|reserved_token_6|>"
        image_processor = self._get_image_processor(model)
        base_model = model.get_model() if hasattr(model, "get_model") else model
        conv_version = getattr(self.args, "version", "llada")
        if conv_version not in conv_templates:
            conv_version = "llada"
        gen_shape_map = {1024: (64, 64), 512: (32, 32), 256: (16, 16)}
        gen_shape = gen_shape_map.get(gen_cfg["image_resolution"], (32, 32))
        is_unitok = "unitok" in getattr(base_model.config, "mm_vqvae", "")
        latent_shape = (8, gen_cfg["n_tokens"]) if is_unitok else (gen_cfg["n_tokens"],)

        batch_latents = [
            torch.full(latent_shape, img_mask_id, dtype=torch.long, device=device) for _ in range(batch_size)
        ]
        image_contexts = [self._make_invalid_image_edit_ctx(batch_latents[idx]) for idx in range(batch_size)]

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
            edit_image = self._load_image(example.get("image"))
            
            assert edit_image is not None, f"Edit image is None for example {example}"
            instruction = self._extract_image_edit_instruction(example)
            
            image_sizes.append(edit_image.size)
            all_edit_images.append(edit_image)

            image_1024 = pad_to_square_and_resize(edit_image.convert("RGB"), gen_cfg["image_resolution"])
            vq_latent = base_model.image_processor_gen.preprocess(image_1024).to(device, dtype=model.dtype)
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
                self.processing_class,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            all_input_ids.append(input_ids)

        # Per-row prompt token lengths (un-padded), captured before left-padding
        # so we can log them later as gen/prompt_length metrics.
        per_row_prompt_lens = [int(t.shape[1]) for t in all_input_ids]
        all_input_ids = self._left_pad_2d(all_input_ids, self.processing_class.pad_token_id, torch.long)
        attention_mask = (all_input_ids != self.processing_class.pad_token_id).long()
        image_tensor = process_images(all_edit_images, image_processor, model.config)
        image_tensor = self._normalize_mm_image_payload(image_tensor, dtype=model.dtype, device=device)

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
        enc_embeddings = pad_along_last_dim(torch.cat(all_enc_embeddings, dim=0), size=inputs_embeds.shape[-1])
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

        noise_embed = base_model.transformer.wte(torch.tensor([self.args.mask_id], device=device))
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
            unmask_order = stratified_random(n=int(np.sqrt(gen_cfg["n_tokens"])), seed=42, shuffle_blocks=True)

        sch_temperatures = self._build_image_edit_temperature_schedule(
            gen_cfg["n_steps"],
            gen_cfg["schedule_temp"],
            gen_cfg["min_temperature"],
            gen_cfg["shift"],
        ).to(device=device)
        sch_temperatures_samp = self._build_image_edit_temperature_schedule(
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
            do_cfg = gen_cfg["guidance_scale"] > 0 and (
                cfg_start <= step_idx <= cfg_end
            )

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

            safe_logits = torch.nan_to_num(logits.to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9)
            probs = safe_logits.softmax(-1)
            if gen_cfg["sample_policy"] == "argmax":
                x0 = safe_logits.argmax(-1)
            else:
                temperature = gen_cfg.get("temperature", 0.8)
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
                    alg_temp = gen_cfg["alg_temp"] * float(local_temp) if gen_cfg["dynamic_temperature"] else gen_cfg["alg_temp"]
                    confidence = x0_p[batch_idx] / max(alg_temp, 1e-6)
                    confidence = torch.where(b_mask, confidence, -np.inf)
                    confidence = torch.softmax(confidence, dim=-1)
                    select_index = torch.multinomial(confidence, num_samples=k)
                elif step_confidence_policy == "stratified" and unmask_order is not None:
                    start = gen_cfg["n_tokens"] - b_n_mask
                    select_index = torch.tensor(unmask_order[start : start + k], device=device, dtype=torch.long)
                else:
                    alg_temp = gen_cfg["alg_temp"] * float(local_temp) if gen_cfg["dynamic_temperature"] else gen_cfg["alg_temp"]
                    confidence = torch.log(x0_p[batch_idx].clamp(1e-20)) + alg_temp * self._image_edit_gumbel_noise(x0_p[batch_idx])
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
        rollout_dir = Path("/tmp/diffu_grpo_rollouts")
        rollout_dir.mkdir(parents=True, exist_ok=True)
        image_contexts = []
        for batch_idx, (example, decoded_image) in enumerate(zip(examples, decoded_images)):
            prompt = prompts[batch_idx]
            sample_id = str(example.get("sample_id", example.get("id", example.get("pid", batch_idx))))
            safe_sample_id = sample_id.replace("/", "_").replace(os.sep, "_").replace(" ", "_")
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

            decoded_image_path = rollout_dir / f"{safe_sample_id}_{os.getpid()}_{batch_idx}.png"
            if hasattr(decoded_image_obj, "save"):
                decoded_image_obj.save(decoded_image_path)

            source_images = example.get("image")
            if source_images is None:
                source_images = []
            elif not isinstance(source_images, list):
                source_images = [source_images]

            source_enc_images = example.get("image_gen_enc", example.get("image"))
            if source_enc_images is None:
                source_enc_images = []
            elif not isinstance(source_enc_images, list):
                source_enc_images = [source_enc_images]

            instruction = self._extract_image_edit_instruction(example)
            # Number of generated image tokens for this row (constant per run,
            # determined by the gen config; reported per-row for symmetry with
            # the und/text path).
            row_completion_len = int(gen_cfg["n_tokens"])
            image_contexts.append(
                {
                    "valid": True,
                    "latent_shape": tuple(xt[batch_idx].shape),
                    "decoded_image": decoded_image_obj,
                    "prompt": prompt,
                    "prompt_len_tokens": per_row_prompt_lens[batch_idx],
                    "completion_len_tokens": row_completion_len,
                    "payload": {
                        "id": sample_id,
                        "image_gen": str(decoded_image_path),
                        "image": source_images,
                        "image_gen_enc": source_enc_images,
                        "pad_image_gen": True,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image> <image_gen_enc>\n{instruction}",
                            },
                            {
                                "from": "gpt",
                                "value": "<image_gen>",
                            },
                        ],
                    },
                }
            )
        return xt, image_contexts

    @staticmethod
    def _resolve_llada_forward_model(model: PreTrainedModel) -> LLaDAModelLM:
        candidates = [model]
        seen = set()
        while candidates:
            candidate = candidates.pop(0)
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            if isinstance(candidate, LLaDAModelLM):
                return candidate
            if hasattr(candidate, "get_base_model"):
                try:
                    base_candidate = candidate.get_base_model()
                except TypeError:
                    base_candidate = None
                if base_candidate is not None:
                    candidates.append(base_candidate)
            for attr in ("module", "base_model", "model"):
                if hasattr(candidate, attr):
                    nested = getattr(candidate, attr)
                    if nested is not None:
                        candidates.append(nested)
        raise TypeError(f"Could not resolve an LLaDA model from {type(model)!r}")


    def _build_scored_batch_collator(
        self,
        base_collate_fn: Callable[[Sequence[Dict]], Dict[str, torch.Tensor]],
        advantages: torch.Tensor,
        completion_masks: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        ref_per_token_logps: Optional[torch.Tensor],
        num_iterations: int,
    ) -> Callable[[Sequence[Dict[str, Any]]], Dict[str, Any]]:
        detached_advantages = advantages.detach()
        detached_completion_masks = completion_masks.detach()
        detached_old_per_token_logps = old_per_token_logps.detach()
        detached_ref_per_token_logps = None if ref_per_token_logps is None else ref_per_token_logps.detach()

        def collate_scored_instances(instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            sample_indices = [instance["_sample_idx"] for instance in instances]
            base_instances = [
                {key: value for key, value in instance.items() if key != "_sample_idx"}
                for instance in instances
            ]
            batch = self._prepare_input(base_collate_fn(base_instances))
            batch["scoring_instances"] = base_instances
            batch["advantages"] = detached_advantages[sample_indices]
            batch["completion_mask"] = detached_completion_masks[:, sample_indices, :]
            if num_iterations > 1:
                batch["old_per_token_logps"] = detached_old_per_token_logps[:, sample_indices, :]
            if detached_ref_per_token_logps is not None:
                batch["ref_per_token_logps"] = detached_ref_per_token_logps[:, sample_indices, :]
            return batch

        return collate_scored_instances

    @staticmethod
    def _select_model_forward_kwargs(model: PreTrainedModel, inputs: dict[str, Any]) -> dict[str, Any]:
        forward_signature = inspect.signature(model.forward)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in forward_signature.parameters.values()
        ):
            return dict(inputs)
        return {
            key: value
            for key, value in inputs.items()
            if key in forward_signature.parameters
        }

    @staticmethod
    def _pad_and_concat_logps(logps_per_batch: list[torch.Tensor]) -> torch.Tensor:
        if not logps_per_batch:
            return torch.empty(0)
        max_keep = max(logps.shape[-1] for logps in logps_per_batch)
        concat_dim = logps_per_batch[0].ndim - 2
        padded = []
        for logps in logps_per_batch:
            if logps.shape[-1] == max_keep:
                padded.append(logps)
                continue
            pad_shape = (*logps.shape[:-1], max_keep - logps.shape[-1])
            pad = torch.zeros(pad_shape, dtype=logps.dtype, device=logps.device)
            padded.append(torch.cat([logps, pad], dim=-1))
        return torch.cat(padded, dim=concat_dim)

    @staticmethod
    def _pad_and_stack_logps(logps_per_iteration: list[torch.Tensor]) -> torch.Tensor:
        if not logps_per_iteration:
            return torch.empty(0)
        max_keep = max(logps.shape[-1] for logps in logps_per_iteration)
        padded = []
        for logps in logps_per_iteration:
            if logps.shape[-1] == max_keep:
                padded.append(logps)
                continue
            pad_shape = (*logps.shape[:-1], max_keep - logps.shape[-1])
            pad = torch.zeros(pad_shape, dtype=logps.dtype, device=logps.device)
            padded.append(torch.cat([logps, pad], dim=-1))
        return torch.stack(padded, dim=0)

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
        """Compute per-token log-probs for gen and und scoring datasets.

        Gen (image-edit) and und (text) samples are processed in **separate**
        model.forward passes rather than a single concatenated batch.  This
        avoids cross-modality padding in MaskDataCollator (gen latent lengths
        and und text lengths differ substantially) and keeps activation memory
        proportional to the longest sample within a single modality instead of
        the global max.  It also avoids a subtle MaskDataCollator bug where
        ``batch['dataset_name']`` was taken from ``instances[0]`` only —
        homogeneous per-modality batches make that assignment always correct.

        The two modalities' cached raw samples are kept in **separate** lists
        (``cached_gen_samples`` and ``cached_und_samples``) so that gen_slice
        and und_slice index directly into their respective cache without any
        offset arithmetic.

        Force-mask plumbing:
          On the first (old-policy) call, ``force_gen_masks`` /
          ``force_und_masks`` are None.  The function captures
          ``final_masked_indices`` / ``masked_indices_gen`` returned by
          ``model.forward`` for each (seed, sample) pair, and returns them
          alongside the logps.  The caller persists these across ranks /
          grad-accum steps.
          On subsequent calls (ref-policy pass and the current-policy pass
          inside ``_compute_loss``), the caller passes the previously captured
          masks back via ``force_gen_masks`` / ``force_und_masks``.  They are
          forwarded into ``LlavaLladaForMaskedDiffusion.forward`` as
          ``force_text_mask`` / ``force_gen_mask``, which bypass the
          random-draw-based masking entirely.  This guarantees that every
          forward at the same (sample, mask_seed) sees bitwise-identical
          masks, so ``coef_1 = exp(curr - old)`` can never blow up.

        Index alignment (critical — this is what the advantages / old_logps /
        ref_logps buffers rely on):
          - Row ``i`` of the returned ``per_gen_logps`` corresponds to
            ``gen_scoring[gs + i]``, where ``(gs, ge) = gen_slice``.  This
            pairs with ``gen_advantages[gs + i]``, ``old_gen_logps[:, gs + i]``,
            ``ref_gen_logps[:, gs + i]`` — all indexed off ``gen_scoring``.
          - Row ``i`` of the returned ``per_und_logps`` corresponds to
            ``und_scoring[us + i]``, where ``(us, ue) = und_slice``.  Same
            pairing with ``und_advantages`` / ``old_und_logps`` / ``ref_und_logps``.

        Args:
            cached_gen_samples: If provided, reuse these pre-fetched per-sample
                dicts for the gen modality instead of calling ``gen_scoring[i]``
                again.  Avoids non-determinism from LazySupervisedDataset
                (random crops, retry logic) that would otherwise produce
                different sequence lengths across calls.  Length = len(gen_scoring).
            cached_und_samples: Same idea for und modality. Length = len(und_scoring).
            force_gen_masks: Optional list[list[dict]] indexed by
                [seed_index][sample_local_idx_within_slice].  Each entry is
                ``{"text_mask": Tensor, "gen_mask": Tensor}`` or None.  When
                provided, bypasses the random-draw mask inside model.forward.
                Shape of the outer list must match ``len(mask_seeds)``; shape
                of each inner list must match ``ge - gs`` samples covered.
            force_und_masks: Same, but for und modality (gen_mask is unused /
                always None for und samples).
            gen_slice: (start, end) — if provided, only forward gen samples in
                this index range of ``gen_scoring`` / ``cached_gen_samples``.
            und_slice: (start, end) — same, but for und.

        Returns:
            per_gen_logps: (len(mask_seeds), N_gen, latent_total) or None
            per_und_logps: (len(mask_seeds), N_und, L_text) or None
            cached_gen_samples: list[dict] — gen per-sample dicts (for caching)
            cached_und_samples: list[dict] — und per-sample dicts (for caching)
            captured_gen_masks: list[list[dict]] — [seed_index][sample_local_idx]
                of {"text_mask": CPU Tensor, "gen_mask": CPU Tensor}.  Either
                reflects the masks drawn on this call (if force_gen_masks was
                None) or is a straight passthrough of force_gen_masks (so
                callers always have a single handle to the masks in play).
            captured_und_masks: same for und.
        """
        N_gen_full = len(gen_scoring) if gen_scoring is not None else 0
        N_und_full = len(und_scoring) if und_scoring is not None else 0

        # --- Fetch raw samples (once) or reuse cache, independently per modality ---
        if gen_scoring is not None and cached_gen_samples is None:
            cached_gen_samples = [gen_scoring[i] for i in range(N_gen_full)]
        if und_scoring is not None and cached_und_samples is None:
            cached_und_samples = [und_scoring[i] for i in range(N_und_full)]
        assert (gen_scoring is not None) or (und_scoring is not None), (
            "At least one of gen_scoring or und_scoring must be provided."
        )

        # --- Per-modality slices (no offsetting — each cache is self-indexed) ---
        gs, ge = gen_slice if gen_slice is not None else (0, N_gen_full)
        us, ue = und_slice if und_slice is not None else (0, N_und_full)
        gen_selected = cached_gen_samples[gs:ge] if cached_gen_samples is not None else []
        und_selected = cached_und_samples[us:ue] if cached_und_samples is not None else []

        collate_fn = MaskDataCollator(self.processing_class, self.args)
        # Hard-coded to 1 regardless of self._train_batch_size: the force-mask
        # capture/replay path stores one mask per sample with sample-specific
        # seq_len (different samples have different multimodal token counts),
        # so it can't be trivially batch-stacked.  Forcing batch=1 here keeps
        # _run_modality's per-sample lookup correct on every call site
        # (old-pass, ref-pass, and current-policy pass inside _compute_loss),
        # regardless of whether HF Trainer / TRL set ``self._train_batch_size``
        # to something larger for their own reasons.  The throughput cost is
        # small: LLaDA's forward is bottlenecked by activation memory, not
        # launch overhead, and chunking is already per-sample on the call
        # sites that matter.
        scoring_batch_size = 1
        p_mask_prompt = float(getattr(self.args, "p_mask_prompt", 0.15))
        device = self.accelerator.device
        mdtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

        def _prepare_batch(batch: dict) -> dict:
            """Move tensors to device, cast floats to model dtype, drop non-forward keys."""
            prepared: dict = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(dtype=mdtype)
                    prepared[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    prepared[k] = [
                        t.to(device=device, dtype=mdtype) if t.is_floating_point() else t.to(device)
                        for t in v
                    ]
                else:
                    prepared[k] = v
            # Pop keys that are not model.forward parameters. In the split
            # scheme sample_is_image_edit is homogeneous within a batch
            # (all-True for gen, all-False for und) and no longer needed for
            # row filtering — each modality's forward directly returns the
            # loss tensor covering every row of that batch.
            prepared.pop("sample_is_image_edit", None)
            prepared.pop("prompts", None)
            # Force do_not_mask_text to always be absent (= None) during GRPO
            # scoring forwards.  LlavaLladaForMaskedDiffusion.forward has a
            # non-deterministic branch at llava_llada.py:353 that applies a
            # 80% ``torch.rand_like`` gate to the per-sample do_not_mask_text
            # flags using the global default RNG, not the seeded generator.
            # Across the old-policy and current-policy passes this produces
            # different final_masked_indices for the same (sample, mask_seed),
            # which makes coef_1 = exp(curr - old) explode at masked positions.
            # Popping the key here means the model's forward receives None,
            # short-circuiting that branch and yielding deterministic masks.
            prepared.pop("do_not_mask_text", None)
            return prepared

        def _run_modality(selected: list, modality: str, force_masks: list = None):
            """Run model.forward over ``selected`` (all one modality, in order).

            Row i of ``selected`` → row i of the concatenated per-seed output,
            so index alignment with ``gen_scoring[gs:ge]`` / ``und_scoring[us:ue]``
            is preserved by construction.

            modality: "gen" (reads gen_loss_none_reduction) or
                      "und" (reads und_loss_none_reduction).

            force_masks: Optional[list[list[dict]]] indexed by
                [seed_index_in_mask_seeds][sample_local_idx].  When provided,
                the corresponding (text_mask, gen_mask) pair is forwarded to
                ``model.forward`` as ``force_text_mask`` / ``force_gen_mask``
                and bypasses the random-draw masking.

            Returns:
                per_seed_outputs: list[Tensor] — one tensor of shape
                    (len(selected), D) per seed, or None if ``selected`` is
                    empty.
                captured_masks: list[list[dict]] — [seed_idx][sample_local_idx]
                    of {"text_mask": CPU Tensor, "gen_mask": CPU Tensor | None},
                    or None if ``selected`` is empty.

            Assumption: ``scoring_batch_size == 1``.  The force-mask path
            looks up one sample at a time, so a batch must contain exactly
            one sample for per-sample force masks to be addressable.  This
            assumption matches every call site in the trainer (old/ref pass
            under no_grad, current pass under ``_compute_loss``).
            """
            if not selected:
                return None, None

            assert scoring_batch_size == 1, (
                f"Force-mask capture/replay requires scoring_batch_size=1, "
                f"got {scoring_batch_size}. Increase per-sample indexing to "
                f"support batched masks if this changes."
            )

            # Micro-batch in order; with scoring_batch_size=1, batch_idx maps
            # one-to-one to sample_local_idx inside ``selected``.
            batches = []
            for start in range(0, len(selected), scoring_batch_size):
                batches.append(collate_fn(selected[start : start + scoring_batch_size]))

            per_seed_outputs: list[torch.Tensor] = []
            captured_masks_per_seed: list[list[dict]] = []
            for seed_idx, seed in enumerate(mask_seeds):
                per_batch_logps: list[torch.Tensor] = []
                captured_masks: list[dict] = []
                seed_force = force_masks[seed_idx] if force_masks is not None else None
                for sample_local_idx, batch in enumerate(batches):
                    prepared = _prepare_batch(batch)

                    # Look up force masks for this (seed, sample).
                    force_text_mask = None
                    force_gen_mask = None
                    if seed_force is not None:
                        slot = seed_force[sample_local_idx]
                        if slot is not None:
                            t = slot.get("text_mask")
                            g = slot.get("gen_mask")
                            if t is not None:
                                force_text_mask = t.to(device, non_blocking=True)
                            if g is not None:
                                force_gen_mask = g.to(device, non_blocking=True)

                    output = model.forward(
                        **prepared,
                        mask_seeds=[int(seed)],
                        p_mask_prompt=p_mask_prompt,
                        temperature=self.temperature,
                        force_text_mask=force_text_mask,
                        force_gen_mask=force_gen_mask,
                    )
                    if modality == "gen":
                        # (batch_size, latent_total) — covers every (all-gen) row.
                        loss_none = output.get("gen_loss_none_reduction")
                        assert loss_none is not None, (
                            "gen-only forward did not return gen_loss_none_reduction"
                        )
                    else:
                        # (batch_size, L) — covers every (all-und) row.
                        loss_none = output.get("und_loss_none_reduction")
                        assert loss_none is not None, (
                            "und-only forward did not return und_loss_none_reduction"
                        )
                    per_batch_logps.append(-loss_none)

                    # Capture the mask actually used in this forward so the
                    # caller can replay it.  If force_masks was provided, the
                    # model round-trips it unchanged, so the capture is still
                    # a safe single source of truth for downstream calls.
                    captured_text = output.get("final_masked_indices")
                    captured_gen = output.get("masked_indices_gen") if modality == "gen" else None
                    captured_masks.append({
                        "text_mask": (
                            captured_text.detach().to("cpu")
                            if captured_text is not None else None
                        ),
                        "gen_mask": (
                            captured_gen.detach().to("cpu")
                            if captured_gen is not None else None
                        ),
                    })
                # Concatenate batches along the sample dim, padding D to the
                # per-seed max. Ordering within per_batch_logps matches
                # ``selected`` → so is the row order after concatenation.
                per_seed_outputs.append(self._pad_and_concat_logps(per_batch_logps))
                captured_masks_per_seed.append(captured_masks)
            return per_seed_outputs, captured_masks_per_seed

        # Gen fully first, then und — each in its own forward loop.
        all_gen_logps, captured_gen_masks = _run_modality(
            gen_selected, "gen", force_masks=force_gen_masks,
        )
        all_und_logps, captured_und_masks = _run_modality(
            und_selected, "und", force_masks=force_und_masks,
        )

        per_gen_logps = self._pad_and_stack_logps(all_gen_logps) if all_gen_logps else None
        per_und_logps = self._pad_and_stack_logps(all_und_logps) if all_und_logps else None
        # per_gen_logps: (len(mask_seeds), N_gen, latent_total)
        # per_und_logps: (len(mask_seeds), N_und, L_text)
        return (
            per_gen_logps,
            per_und_logps,
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

            # ---- Fix E: Step-0 determinism assertion ----
            # On step 0, model weights are identical between the old-policy
            # pass and the current-policy pass, so per_token_logps and old_ptl
            # MUST be bitwise-identical at every position where old masked
            # (i.e. wherever old_ptl != 0).  Any divergence means one of the
            # non-determinism sources we tried to kill is still leaking
            # through: dropout, do_not_mask_text RNG, scoring_batch_size
            # mismatch between old and current, or force-mask plumbing bug.
            #
            # We run this only on step 0 (cheap, one-shot) so the assertion
            # has no steady-state cost.  Gated by an env var so it can be
            # turned off in production if it ever becomes a nuisance.
            if (
                os.environ.get("DIFFU_GRPO_STEP0_ASSERT", "1") == "1"
                and int(getattr(self.state, "global_step", 0)) == 0
                and self._step % num_iterations == this_itr_idx  # first iter of step 0
            ):
                with torch.no_grad():
                    masked = (old_ptl != 0)
                    if masked.any():
                        diff_at_masked = (
                            per_token_logps.detach() - old_ptl.detach()
                        )[masked].abs()
                        max_diff = float(diff_at_masked.max().item())
                        mean_diff = float(diff_at_masked.mean().item())
                        # bf16 has ~8 bits of mantissa → eps ~3e-3. Allow a
                        # generous tolerance so we only fire on real drift.
                        tol = 5e-2
                        if max_diff > tol:
                            msg = (
                                f"[{prefix}step-0 determinism check] "
                                f"|current - old|@masked: max={max_diff:.4e}, "
                                f"mean={mean_diff:.4e} (tol={tol:.0e}). "
                                f"Non-determinism is leaking through: check "
                                f"dropout, do_not_mask_text, force-mask "
                                f"plumbing, or scoring_batch_size."
                            )
                            if os.environ.get("DIFFU_GRPO_STEP0_STRICT", "0") == "1":
                                raise AssertionError(msg)
                            else:
                                print("[WARN]", msg, flush=True)

            # -------- DEBUG: pre-exp shape / zero-pattern checks --------
            # These catch the "current-pass logps are zero-padded at positions
            # where old_ptl is non-zero" footgun, which makes
            # coef_1 = exp(0 - old_ptl) blow up into the 1e8+ range and, via
            # the clipped-PPO asymmetry on negative advantages, produces the
            # observed 1e7-1e13 loss values.
            def _dbg_preexp():
                cur_mask = (per_token_logps.detach() != 0)
                old_mask = (old_ptl.detach() != 0)
                # Positions where old thinks the token is real but current is
                # zero-padded (the dangerous class).
                only_old = (old_mask & ~cur_mask).sum().item()
                only_cur = (cur_mask & ~old_mask).sum().item()
                shape_ok = per_token_logps.shape == old_ptl.shape
                msg = (
                    f"{prefix}shape check | per_token_logps={tuple(per_token_logps.shape)} "
                    f"old_ptl={tuple(old_ptl.shape)} shape_match={shape_ok} "
                    f"only_old_nonzero={only_old} only_cur_nonzero={only_cur}"
                )
                _debug_log(msg)
                if only_old > 0:
                    # This is the primary suspected cause: sample a handful of
                    # offending positions and print old_ptl values so we can
                    # see the magnitude of exp(-old_ptl) it will produce.
                    bad = (old_mask & ~cur_mask).nonzero(as_tuple=False)[:5]
                    for idx in bad:
                        i, j = int(idx[0]), int(idx[1])
                        old_v = float(old_ptl[i, j].detach())
                        _debug_log(
                            f"{prefix}pad-mismatch @[{i},{j}]: "
                            f"old_ptl={old_v:.3e} → coef_1 would be exp({-old_v:.3e})"
                        )
            _debug_run(_dbg_preexp)

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

                # -------- DEBUG: KL term magnitude ----------
                def _dbg_kl():
                    mask = (old_ptl.detach() != 0)
                    if not mask.any():
                        return
                    lr = log_ratio.detach()[mask]
                    kl = per_token_kl.detach()[mask]
                    _debug_log(
                        f"{prefix}kl | log_ratio min={float(lr.min()):.3e} "
                        f"max={float(lr.max()):.3e} abs_max={float(lr.abs().max()):.3e} "
                        f"| per_token_kl min={float(kl.min()):.3e} "
                        f"max={float(kl.max()):.3e} mean={float(kl.mean()):.3e} "
                        f"| beta={beta}"
                    )
                    # Flag ref-side zero-padding mismatch (the KL analogue of
                    # the coef_1 mismatch above: ref padded with 0 where curr
                    # is very negative → exp(-curr) blows up).
                    ref_mask = (ref_ptl.detach() != 0)
                    only_old = (mask & ~ref_mask).sum().item()
                    if only_old > 0:
                        _debug_log(
                            f"{prefix}kl | ref zero-padded at {only_old} positions "
                            f"where old_ptl is real — KL term is unsafe here"
                        )
                _debug_run(_dbg_kl)
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

            # -------- DEBUG: per-chunk reduced loss ----------
            def _dbg_loss():
                l = float(loss.detach())
                ptl_masked = (per_token_loss.detach() * completion_mask)
                per_pos_max = float(ptl_masked.abs().max()) if completion_mask.any() else 0.0
                _debug_log(
                    f"{prefix}reduced loss={l:.3e} | denom={int(denom.item())} "
                    f"| per_pos |loss| max={per_pos_max:.3e} "
                    f"| global_step={int(getattr(self.state, 'global_step', 0))} "
                    f"inner_step={int(self._step)}"
                )
                if not math.isfinite(l):
                    _debug_log(f"{prefix}LOSS NON-FINITE — aborting hook chain")
            _debug_run(_dbg_loss)

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

        # ---- Compute gen / und losses independently ----
        gen_loss = _grpo_loss(gen_logps, old_gen_logps, ref_gen_logps, gen_advantages, "gen/")
        und_loss = _grpo_loss(und_logps, old_und_logps, ref_und_logps, und_advantages, "und/")

        # ---- Combine ----
        if gen_loss is not None and und_loss is not None:
            loss = gen_loss + und_loss
        elif gen_loss is not None:
            loss = gen_loss
        elif und_loss is not None:
            loss = und_loss
        else:
            raise ValueError("Both gen_batch and und_batch are None — nothing to train on.")

        if gen_loss is not None:
            self._metrics[mode]["gen/loss"].append(gen_loss.detach().item())
        if und_loss is not None:
            self._metrics[mode]["und/loss"].append(und_loss.detach().item())

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        _elapsed_curr = time.perf_counter() - _t0_curr
        self._metrics[mode]["time_profile/curr_logprobs_and_update"].append(_elapsed_curr)
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Average accumulated self._metrics and merge into the logs dict.

        Called by transformers' Trainer on the ``logging_steps`` cadence.
        This mirrors TRL's GRPOTrainer.log pattern: we accumulate metrics
        across ALL gradient-accumulation micro-steps of a logging window,
        then average + clear here. ``grad_norm``, ``loss``, and
        ``learning_rate`` are already injected into ``logs`` by the parent
        Trainer's ``_maybe_log_save_evaluate``.
        """
        mode = "train" if self.model.training else "eval"
        raw = self._metrics.get(mode, {})

        averaged: dict[str, float] = {}
        for key, vals in raw.items():
            valid = [v for v in vals if not math.isnan(v)]
            if valid:
                averaged[key] = sum(valid) / len(valid)

        if mode == "eval":
            averaged = {f"eval_{k}": v for k, v in averaged.items()}

        # Attach our custom step counter so wandb users can x-axis against
        # generation steps rather than only global optimizer steps.
        if self.accelerator.is_main_process:
            averaged["train/gen_step"] = float(self._step)

        logs = {**logs, **averaged}
        try:
            super().log(logs, start_time)
        except TypeError:
            # Older transformers versions don't accept start_time.
            super().log(logs)

        raw.clear()

    @staticmethod
    def _split_modality_batch(batch, num_chunks):
        """Split [scoring, advantages, old_logps, ref_logps] into ``num_chunks`` micro-batches.

        scoring is a LazySupervisedDataset — NOT split (kept as-is in every chunk)
        so that _compute_loss always re-collates from the full dataset using the
        same cached batches, ensuring identical padding/sequence lengths.
        advantages is (N,) → chunk along dim 0.
        old_logps is (num_iterations, N, D) → chunk along dim 1.
        ref_logps is (num_iterations, N, D) or None → same.

        Returns a list of ``num_chunks`` lists, each
        [scoring, adv_sub, old_sub, ref_sub, (chunk_start, chunk_end)].
        """
        if batch is None:
            return [None] * num_chunks
        scoring, advantages, old_logps, ref_logps = batch
        N = len(scoring)
        chunk_size = N // num_chunks
        assert chunk_size * num_chunks == N, (
            f"Cannot evenly split {N} samples into {num_chunks} micro-batches"
        )
        chunks = []
        for i in range(num_chunks):
            s, e = i * chunk_size, (i + 1) * chunk_size
            adv_sub = advantages[s:e]
            old_sub = old_logps[:, s:e] if old_logps is not None else None
            ref_sub = ref_logps[:, s:e] if ref_logps is not None else None
            chunks.append([scoring, adv_sub, old_sub, ref_sub, (s, e)])
        return chunks

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # `inputs` is the full generation batch from the dataloader
                # (per_device_train_batch_size * steps_per_generation samples).
                # Dispatch based on modality:
                #   - "gen": inputs drive the gen side; und side (if any) is
                #     resolved by sample_id lookup.
                #   - "und": inputs drive the und side; gen side is None and
                #     the image rollout is bypassed entirely.
                if self.modality == "und":
                    gen_inputs = None
                    und_inputs = inputs
                else:
                    gen_inputs = inputs
                    # Resolve the paired und batch by sample_id (NOT by a
                    # second dataloader — see __init__ for why).  We look up
                    # one und row per gen row, in order, so ``und_inputs[i]``
                    # always refers to the same source sample as
                    # ``gen_inputs[i]``.
                    und_inputs = None
                    if self._und_by_sample_id is not None:
                        und_inputs = []
                        for i, g in enumerate(gen_inputs):
                            sid = g.get("sample_id") if hasattr(g, "get") else None
                            if sid is None:
                                raise ValueError(
                                    f"gen row {i} is missing 'sample_id' — "
                                    f"cannot pair with und row. Check that "
                                    f"sample_id is not being stripped by "
                                    f"_remove_unused_columns."
                                )
                            u = self._und_by_sample_id.get(sid)
                            if u is None:
                                raise KeyError(
                                    f"gen row {i} has sample_id={sid!r} but "
                                    f"no matching und row exists. Was "
                                    f"train_dataset built from a different "
                                    f"source than train_dataset_und?"
                                )
                            und_inputs.append(u)

                gen_and_score = self._generate_and_score_completions(
                    gen_inputs=gen_inputs, und_inputs=und_inputs,
                )
                # gen_and_score = {
                #   "gen": [gen_scoring, gen_advantages, old_gen_logps, ref_gen_logps] or None,
                #   "und": [und_scoring, und_advantages, old_und_logps, ref_und_logps] or None,
                #   "mask_seeds": list[int],
                # }

                # Split into steps_per_generation micro-batches for gradient accumulation.
                n_chunks = self.args.steps_per_generation
                gen_chunks = self._split_modality_batch(gen_and_score["gen"], n_chunks)
                und_chunks = self._split_modality_batch(gen_and_score["und"], n_chunks)
                mask_seeds = gen_and_score["mask_seeds"]
                cached_gen_samples = gen_and_score["cached_gen_samples"]
                cached_und_samples = gen_and_score["cached_und_samples"]
                cached_gen_masks = gen_and_score["cached_gen_masks"]
                cached_und_masks = gen_and_score["cached_und_masks"]

                self._buffered_inputs = [
                    {
                        "gen_batch": gen_chunks[i],
                        "und_batch": und_chunks[i],
                        "mask_seeds": mask_seeds,
                        "cached_gen_samples": cached_gen_samples,
                        "cached_und_samples": cached_und_samples,
                        # Full [num_iterations][N_full] per-sample mask caches.
                        # Sliced inside _compute_loss to the chunk's samples
                        # and current iteration before being passed back into
                        # the model as force masks.
                        "cached_gen_masks": cached_gen_masks,
                        "cached_und_masks": cached_und_masks,
                    }
                    for i in range(n_chunks)
                ]

            result = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
            return result
        else:
            # Eval path: route ``inputs`` to whichever side ``modality`` points to.
            if self.modality == "und":
                gen_and_score = self._generate_and_score_completions(und_inputs=inputs)
            else:
                gen_and_score = self._generate_and_score_completions(gen_inputs=inputs)
            # Append None chunk_range to match the 5-element format from _split_modality_batch.
            gen_b = gen_and_score.get("gen")
            und_b = gen_and_score.get("und")
            if gen_b is not None:
                gen_b = gen_b + [None]  # [scoring, adv, old, ref, None]
            if und_b is not None:
                und_b = und_b + [None]
            return {
                "gen_batch": gen_b,
                "und_batch": und_b,
                "mask_seeds": gen_and_score["mask_seeds"],
                "cached_gen_samples": gen_and_score["cached_gen_samples"],
                "cached_und_samples": gen_and_score["cached_und_samples"],
                "cached_gen_masks": gen_and_score["cached_gen_masks"],
                "cached_und_masks": gen_and_score["cached_und_masks"],
            }

    def _generate_and_score_completions(
        self, gen_inputs: dict[str, Union[torch.Tensor, Any]] = None, und_inputs: dict[str, Union[torch.Tensor, Any]] = None
    ) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        beta = float(getattr(self.args, "beta", 0.0))
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        mode = "eval" if self.control.should_evaluate else "train"
        _timings: dict[str, float] = {}
        grad_accum_steps = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)

        # Sanity-check gen/und alignment by sample_id when both sides carry one
        # (e.g. thinkmorph_interleave). The two dataloaders are expected to
        # iterate in lockstep so row i refers to the same source sample on both
        # sides; this assert catches sampler/shuffle drift before any rollouts
        # are wasted on a misaligned batch.
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
            image_processor = self._get_image_processor(unwrapped_model)

            if gen_inputs is not None:
                image_contexts: list[Optional[dict]] = [None] * len(gen_inputs)
                image_edit_batch_size = max(1, int(getattr(self.args, "image_edit_batch_size", 1)))
                with _timer(_timings, "image_rollout"):
                    for start_idx in trange(0, len(gen_inputs), image_edit_batch_size, desc="Image Rollout"):
                        batch_examples = gen_inputs[start_idx : start_idx + image_edit_batch_size]
                        batch_init_images = [ex.get("image") for ex in batch_examples]
                        _, batch_contexts = self._rollout_image_edit_latents(unwrapped_model, batch_examples, init_image=batch_init_images)
                        for offset, ctx in enumerate(batch_contexts):
                            image_contexts[start_idx + offset] = ctx
            else:
                image_contexts = None

            # ---- Inject rollout-generated images into und_inputs ----
            # When ``text_rollout_use_gen_image`` is enabled, the image rollout
            # output for each sample is fed as a second image into the
            # corresponding text-rollout sample (as ``gen_image``).  This
            # implements a ThinkMorph-style pattern where the model first
            # generates a reasoning image and then answers a question
            # conditioned on BOTH the problem image and the generated image.
            #
            # Requirements:
            #   - gen_inputs and und_inputs must have the same length (1-to-1
            #     pairing, sample-by-sample, in the order produced by their
            #     respective dataloaders).
            #   - Each image_contexts[i] is expected to carry "decoded_image"
            #     (a PIL image produced by _rollout_image_edit_latents).
            # We build a fresh list of shallow-copied sample dicts rather than
            # mutating ``und_inputs`` in place, since the dataloader may hand
            # us objects that should not be mutated.
            if (
                getattr(self.args, "text_rollout_use_gen_image", False)
                and und_inputs is not None
                and image_contexts is not None
            ):
                if len(und_inputs) != len(image_contexts):
                    raise ValueError(
                        "text_rollout_use_gen_image requires gen_inputs and "
                        f"und_inputs to have the same length, got "
                        f"{len(image_contexts)} vs {len(und_inputs)}. Check "
                        f"that both dataloaders iterate in lockstep."
                    )
                paired_und_inputs = []
                for sample, ctx in zip(und_inputs, image_contexts):
                    if ctx is None or ctx.get("decoded_image") is None:
                        raise ValueError(
                            "text_rollout_use_gen_image: image rollout produced "
                            "no decoded_image for one of the gen_inputs samples."
                        )
                    merged = dict(sample)  # shallow copy — avoid mutating the dataloader's dict
                    merged["gen_image"] = ctx["decoded_image"]
                    paired_und_inputs.append(merged)
                und_inputs = paired_und_inputs

            # ---- Text (und) rollouts ----
            if und_inputs is not None:
                answer_contexts: list[Optional[dict]] = [None] * len(und_inputs)
                text_batch_size = 4
                prefix_lm = bool(getattr(self.args, "prefix_lm", True))
                generation_kwargs = {
                    "max_new_tokens": int(self.args.max_completion_length),
                    "block_length": int(min(self.args.block_length, self.args.max_completion_length)),
                    "step_per_block": self.args.text_rollout_step_per_block
                    if self.args.text_rollout_step_per_block is not None
                    else int(min(self.args.block_length, self.args.max_completion_length)),
                    "temperature": float(self.args.temperature),
                    "do_sample": bool(self.args.text_rollout_do_sample),
                    "prefix_lm": prefix_lm,
                    "use_fast_dlm": False,
                    "remasking": self.args.remasking,
                    "cfg_scale": self.args.cfg_scale,
                }
                generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
                with _timer(_timings, "text_rollout"):
                    for start_idx in trange(0, len(und_inputs), text_batch_size, desc="Text Rollout"):
                        batch_examples = und_inputs[start_idx : start_idx + text_batch_size]
                        _, batch_contexts = self._rollout_multimodal_text_gen(
                            unwrapped_model, batch_examples, image_processor,
                            generation_kwargs, device,
                        )
                        for offset, ctx in enumerate(batch_contexts):
                            answer_contexts[start_idx + offset] = ctx
            else:
                answer_contexts = None

            # ---- Length metrics (gen / und prompt + completion tokens) ----
            # Aggregate per-rank then gather across ranks so the logged values
            # reflect the global step batch (matching the style of the standard
            # GRPOTrainer's completions/* metrics).
            def _log_length_metric(prefix: str, lengths_list: list[int]) -> None:
                if not lengths_list:
                    return
                local = torch.tensor(lengths_list, dtype=torch.float32, device=device)
                gathered = self.accelerator.gather(local)
                self._metrics[mode][f"{prefix}/mean_length"].append(gathered.mean().item())
                self._metrics[mode][f"{prefix}/min_length"].append(gathered.min().item())
                self._metrics[mode][f"{prefix}/max_length"].append(gathered.max().item())

            if image_contexts is not None:
                _log_length_metric(
                    "gen/prompt",
                    [ctx.get("prompt_len_tokens", 0) for ctx in image_contexts if ctx is not None],
                )
                _log_length_metric(
                    "gen/completion",
                    [ctx.get("completion_len_tokens", 0) for ctx in image_contexts if ctx is not None],
                )
            if answer_contexts is not None:
                _log_length_metric(
                    "und/prompt",
                    [ctx.get("prompt_len_tokens", 0) for ctx in answer_contexts if ctx is not None],
                )
                _log_length_metric(
                    "und/completion",
                    [ctx.get("completion_len_tokens", 0) for ctx in answer_contexts if ctx is not None],
                )

            # ---- Build scoring examples per modality ----
            collate_fn = MaskDataCollator(self.processing_class, self.args)
            dataset_args = self._build_lazy_supervised_data_args(unwrapped_model, image_processor)
            gen_scoring = None
            und_scoring = None
            if gen_inputs is not None:
                gen_scoring = LazySupervisedDataset(
                    tokenizer=self.processing_class,
                    data_args=dataset_args,
                    list_data=[context["payload"] for context in image_contexts],
                )
            if und_inputs is not None:
                und_scoring = LazySupervisedDataset(
                    tokenizer=self.processing_class,
                    data_args=dataset_args,
                    list_data=[context["payload"] for context in answer_contexts],
                )
            assert gen_scoring is not None or und_scoring is not None, "No scoring examples were built."
        
        # ---- Shared mask seeds across modalities and ranks ----
        mask_seeds = torch.randint(0, 2**12, (num_iterations,), device=device)
        mask_seed_list = mask_seeds.detach().cpu().tolist()

        with torch.no_grad():
            with _timer(_timings, "old_logps"):
                # Populates cached_gen_samples / cached_und_samples from the
                # scoring datasets AND captures the masks that model.forward
                # drew for each (seed, sample) pair.  Both are reused by
                # (a) the ref-model pass below and (b) the backward-path
                # forward in _compute_loss.  The cached samples guarantee
                # identical tokenization / padding; the captured masks
                # guarantee bitwise-identical final_masked_indices /
                # masked_indices_gen, so coef_1 = exp(curr - old) in the
                # GRPO loss is identically 1 on step 0.
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
                    # Ref pass: reuse the old pass's samples AND masks.  Since
                    # the current-policy backward pass will also replay those
                    # same masks, this keeps the KL term well-defined (ref
                    # and current score the same exact masked positions).
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

            # ---- Offload old/ref logps to (pinned) CPU ----
            # These tensors have shape (num_iterations, N, D) and are buffered in
            # self._buffered_inputs across every gradient accumulation micro-step.
            # During backward on the current-policy forward, keeping them on GPU
            # wastes VRAM since only a single (iteration, chunk) slice is needed
            # at a time.  Move them to pinned host memory here; _grpo_loss copies
            # the active slice back to the device non-blocking per chunk.
            def _offload_cpu_pinned(t):
                if t is None:
                    return None
                cpu = t.detach().to("cpu")
                try:
                    cpu = cpu.pin_memory()
                except RuntimeError:
                    # Pinning can fail on some platforms / dtypes; fall back to
                    # regular pageable CPU memory (copy will be blocking).
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
                # Gen rewards (image-edit → perceptual)
                gen_reward_inputs = [ex for ex, ctx in zip(gen_inputs, image_contexts)]
                gen_prompts = [ctx["prompt"] for ctx in image_contexts]
                gen_completions = [ctx["decoded_image"] for ctx in image_contexts]
                gen_reward_fns = [perceptual_score_reward_func]

                # Rows whose source dataset has no perceptual ground truth
                # (image_gt is None, e.g. ArxivQA in thinkmorph_interleave) are
                # rollout-only: we still ran the image rollout (the generated
                # image is forwarded into the und text rollout via
                # text_rollout_use_gen_image), but we must NOT compute a
                # perceptual reward, advantage, or loss for them. Compute the
                # reward on the image_gt subset and scatter back into a dense
                # tensor with zeros at the no-gt positions; the num_generations
                # grouping in _compute_advantages then yields zero advantages
                # for those rows, so the GRPO per-token loss is identically
                # zero on them.
                gen_has_image_gt = [ex.get("image_gt") is not None for ex in gen_reward_inputs]
                sub_idx = [i for i, ok in enumerate(gen_has_image_gt) if ok]

                gen_rewards_per_func = torch.zeros(len(gen_completions), len(gen_reward_fns), device=device)
                for i, reward_func in enumerate(gen_reward_fns):
                    reward_func_name = reward_func.__name__
                    with profiling_context(self, reward_func_name):
                        if sub_idx:
                            sub_inputs = [gen_reward_inputs[j] for j in sub_idx]
                            sub_prompts = [gen_prompts[j] for j in sub_idx]
                            sub_completions = [gen_completions[j] for j in sub_idx]
                            keys = [k for k in sub_inputs[0] if k not in ["prompt", "completion"]]
                            reward_kwargs = {k: [ex.get(k) for ex in sub_inputs] for k in keys}
                            try:
                                output = reward_func(
                                    prompts=sub_prompts, completions=sub_completions,
                                    step=self._step, run_name=self.args.output_dir, **reward_kwargs,
                                )
                            except Exception:
                                output = [torch.nan for _ in sub_completions]
                            output = [r if r is not None else torch.nan for r in output]
                            gen_rewards_per_func[sub_idx, i] = torch.tensor(
                                output, dtype=torch.float32, device=device
                            )
                    self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(torch.nanmean(gen_rewards_per_func[:, i]).item())
                    self._metrics[mode][f"rewards/{reward_func_name}/std"].append(nanstd(gen_rewards_per_func[:, i]).item())
                gen_local_rewards = gen_rewards_per_func.nansum(dim=1)

                # ---- DEBUG: save first sample (with image_gt) ----
                if DIFFU_GRPO_DEBUG and len(gen_completions) > 0 and sub_idx:
                    try:
                        from PIL import Image, ImageDraw, ImageFont
                        import textwrap

                        # Use sub_idx[0]: first sample that has image_gt
                        dbg_idx = sub_idx[0]
                        debug_dir = Path("./debug")
                        debug_dir.mkdir(parents=True, exist_ok=True)

                        comp_img = gen_completions[dbg_idx]
                        if not isinstance(comp_img, Image.Image):
                            comp_img = Image.open(comp_img).convert("RGB") if isinstance(comp_img, str) else None

                        gt_path = gen_reward_inputs[dbg_idx].get("image_gt")
                        gt_img = None
                        if gt_path is not None:
                            gt_img = Image.open(gt_path).convert("RGB") if isinstance(gt_path, str) else gt_path
                            if not isinstance(gt_img, Image.Image):
                                gt_img = None

                        if comp_img is not None and gt_img is not None:
                            # Resize both to same height
                            h = max(comp_img.height, gt_img.height)
                            comp_resized = comp_img.resize((int(comp_img.width * h / comp_img.height), h))
                            gt_resized = gt_img.resize((int(gt_img.width * h / gt_img.height), h))
                            concat_w = comp_resized.width + gt_resized.width

                            # Prepare prompt text
                            prompt_text = gen_prompts[dbg_idx]
                            prompt_text = prompt_text.replace("<|reserved_token_5|>", "*").replace("<|reserved_token_6|>", "-")

                            font_size = 18
                            try:
                                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                            except (IOError, OSError):
                                font = ImageFont.load_default()

                            # Wrap text to fit concat image width
                            chars_per_line = max(1, concat_w // (font_size * 2 // 3))
                            wrapped_lines = textwrap.wrap(prompt_text, width=chars_per_line)

                            # Reward text
                            reward_val = float(gen_rewards_per_func[dbg_idx, 0].item())
                            reward_text = f"Perceptual Reward: {reward_val:.4f}"

                            line_height = font_size + 4
                            text_height = (len(wrapped_lines) + 2) * line_height  # +2 for gap and reward line
                            total_h = h + text_height

                            canvas = Image.new("RGB", (concat_w, total_h), (255, 255, 255))
                            canvas.paste(comp_resized, (0, 0))
                            canvas.paste(gt_resized, (comp_resized.width, 0))

                            draw = ImageDraw.Draw(canvas)
                            y_pos = h + 4
                            for line in wrapped_lines:
                                draw.text((4, y_pos), line, fill=(0, 0, 0), font=font)
                                y_pos += line_height

                            # Reward centered at bottom
                            try:
                                rw_bbox = draw.textbbox((0, 0), reward_text, font=font)
                                rw = rw_bbox[2] - rw_bbox[0]
                            except AttributeError:
                                rw = len(reward_text) * (font_size * 2 // 3)
                            draw.text(((concat_w - rw) // 2, y_pos + line_height // 2), reward_text, fill=(0, 0, 200), font=font)

                            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
                            save_path = debug_dir / f"step{self._step}_rank{rank}.png"
                            canvas.save(save_path)
                            _debug_log(f"Saved debug image to {save_path}")
                        else:
                            _debug_log(f"Debug image skip: comp_img={comp_img is not None}, gt_img={gt_img is not None}, gt_path={gt_path!r}")
                    except Exception as e:
                        import traceback
                        _debug_log(f"Debug image save failed: {e}\n{traceback.format_exc()}")

            if und_inputs is not None:
                # Und rewards (text → format + correctness)
                und_reward_inputs = [ex for ex, ctx in zip(und_inputs, answer_contexts)]
                und_prompts = [ctx["prompt"] for ctx in answer_contexts]
                und_completions = [
                    [{"role": "assistant", "content": ctx["decoded_text"]}]
                    for ctx in answer_contexts
                ]
                und_reward_fns = [strict_format_reward_func, correctness_reward_func]

                und_rewards_per_func = torch.zeros(len(und_completions), len(und_reward_fns), device=device)
                for i, reward_func in enumerate(und_reward_fns):
                    reward_func_name = reward_func.__name__
                    with profiling_context(self, reward_func_name):
                        keys = [k for k in und_reward_inputs[0] if k not in ["prompt", "completion"]]
                        reward_kwargs = {k: [ex.get(k) for ex in und_reward_inputs] for k in keys}
                        try:
                            output = reward_func(
                                prompts=und_prompts, completions=und_completions,
                                step=self._step, run_name=self.args.output_dir, **reward_kwargs,
                            )
                        except Exception:
                            output = [torch.nan for _ in und_completions]
                        output = [r if r is not None else torch.nan for r in output]
                        und_rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
                    self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(torch.nanmean(und_rewards_per_func[:, i]).item())
                    self._metrics[mode][f"rewards/{reward_func_name}/std"].append(nanstd(und_rewards_per_func[:, i]).item())
                und_local_rewards = und_rewards_per_func.nansum(dim=1)

        # ---- Advantages per modality ----
        def _compute_advantages(local_rewards, tag):
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
            centered = rewards - mean_grouped  # global advantages (pre-slice)
            advantages = centered[process_slice]
            # Metrics
            is_std_zero = std_grouped < 1e-6
            self._metrics[mode][f"{tag}/reward"].append(rewards.mean().item())
            self._metrics[mode][f"{tag}/reward_std"].append(rewards.std().item())
            self._metrics[mode][f"{tag}/frac_reward_zero_std"].append(is_std_zero.float().mean().item())
            # Advantages distribution (global — computed on the full gathered
            # tensor before process slicing). Tracks whether GRPO advantages
            # are informative: mean should be ~0 (by construction), std and
            # abs_mean indicate signal strength.
            self._metrics[mode][f"{tag}/advantages_mean"].append(centered.mean().item())
            self._metrics[mode][f"{tag}/advantages_std"].append(centered.std().item())
            self._metrics[mode][f"{tag}/advantages_abs_mean"].append(centered.abs().mean().item())
            self._metrics[mode][f"{tag}/advantages_min"].append(centered.min().item())
            self._metrics[mode][f"{tag}/advantages_max"].append(centered.max().item())
            return advantages

        if gen_inputs is not None:
            gen_advantages = _compute_advantages(gen_local_rewards, "gen")
        else:
            gen_advantages = None
        if und_inputs is not None:
            und_advantages = _compute_advantages(und_local_rewards, "und")
        else:
            und_advantages = None

        for k, v in _timings.items():
            self._metrics[mode][f"time_profile/{k}"].append(v)
        return {
            "gen": [gen_scoring, gen_advantages, old_gen_logps, ref_gen_logps] if gen_inputs is not None else None,
            "und": [und_scoring, und_advantages, old_und_logps, ref_und_logps] if und_inputs is not None else None,
            "mask_seeds": mask_seed_list,
            "cached_gen_samples": cached_gen_samples,
            "cached_und_samples": cached_und_samples,
            # [num_iterations][N_gen_full] of {"text_mask": CPU Tensor, "gen_mask": CPU Tensor}
            # Sliced by iteration index in _compute_loss, then by chunk range,
            # before being passed back into the current-policy forward.
            "cached_gen_masks": cached_gen_masks,
            "cached_und_masks": cached_und_masks,
        }