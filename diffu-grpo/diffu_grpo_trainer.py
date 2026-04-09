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
from accelerate.utils import gather, gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
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
        
        if len(images_gen) > 0:
            batch['images_gen'] = images_gen
        else:
            batch['images_gen']  = None
        if len(image_gen_enc)>0:
            batch['images_gen_enc'] = image_gen_enc
        else:
            batch['images_gen_enc']  = None
        batch['image_gen_weight'] = None

        # Per-sample flag: True if the sample has image generation targets (image_edit),
        # False otherwise (text/multimodal-understanding).  Used in _get_per_token_logps
        # to route each sample to gen_loss_none_reduction vs und_loss_none_reduction.
        batch['sample_is_image_edit'] = torch.tensor(
            [instance.get('image_gen') is not None for instance in instances],
            dtype=torch.bool,
        )

        batch['do_not_mask_text'] = [x['do_not_mask_text'] for x in instances]

        return batch

class PairedRepeatSampler(torch.utils.data.Sampler):
    """RepeatSampler variant that keeps consecutive index pairs together.

    The thinkmorph_interleave dataset emits rows in strict pairs:
      index 2k   → image_edit
      index 2k+1 → text
    Shuffling must happen at the *pair* level so that every sampler chunk
    always contains both task types.  This guarantees that after the
    DistributedSampler strides across the yielded indices, every rank
    receives at least one image-edit and one text sample — preventing
    DeepSpeed gradient-reduction deadlocks from unused parameters.

    The yielded index order is identical to ``RepeatSampler`` except that
    the two indices in each chunk are always a paired (2k, 2k+1) rather
    than two arbitrary indices drawn from the shuffled pool.
    """

    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 2,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_samples = len(data_source)
        if self.num_samples % 2 != 0:
            raise ValueError(
                f"PairedRepeatSampler requires an even dataset size, got {self.num_samples}"
            )
        if batch_size % 2 != 0:
            raise ValueError(
                f"PairedRepeatSampler requires an even batch_size, got {batch_size}"
            )
        self.num_pairs = self.num_samples // 2
        self.pairs_per_chunk = batch_size // 2  # unique pairs per chunk
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        if shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            pair_order = torch.randperm(self.num_pairs, generator=self.generator).tolist()
        else:
            pair_order = list(range(self.num_pairs))

        # Build chunks of `pairs_per_chunk` pairs each, drop incomplete tail
        chunks = [
            pair_order[i : i + self.pairs_per_chunk]
            for i in range(0, len(pair_order), self.pairs_per_chunk)
        ]
        chunks = [c for c in chunks if len(c) == self.pairs_per_chunk]

        for chunk in chunks:
            for _ in range(self.repeat_count):
                for pair_idx in chunk:
                    idx_a = 2 * pair_idx      # image_edit
                    idx_b = 2 * pair_idx + 1   # text
                    for _ in range(self.mini_repeat_count):
                        yield idx_a
                    for _ in range(self.mini_repeat_count):
                        yield idx_b

    def __len__(self) -> int:
        usable_pairs = (self.num_pairs // self.pairs_per_chunk) * self.pairs_per_chunk
        return usable_pairs * 2 * self.mini_repeat_count * self.repeat_count


class DiffuGRPOTrainer(GRPOTrainer):
    """GRPO trainer adapted for LaVida-O text + image rollouts with unified token-level scoring."""

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
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
    ):
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
        grad_accum_steps = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        if getattr(self, "_buffered_inputs", None) is None:
            self._buffered_inputs = [None] * grad_accum_steps

    def _get_train_sampler(self, dataset=None):
        """Use PairedRepeatSampler for interleave datasets to guarantee type balance per rank."""
        if dataset is None:
            dataset = self.train_dataset
        is_paired = (
            hasattr(dataset, "__len__")
            and len(dataset) >= 2
            and len(dataset) % 2 == 0
            and hasattr(dataset, "__getitem__")
            and dataset[0].get("task_type") != dataset[1].get("task_type")
        )
        if is_paired:
            return PairedRepeatSampler(
                data_source=dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
        return super()._get_train_sampler(dataset)

    @staticmethod
    def _make_generator(device: torch.device, seed: int) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

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
            "guidance_scale_image": float(getattr(self.args, "image_edit_guidance_scale_image", 5.0)),
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

    def _build_llada_prompt(self, prompt_messages: Any) -> str:
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
        return conv.get_prompt()

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
            prompt_text = self._build_llada_prompt(example["prompt"])
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

            image = self._load_image(example.get("image"))
            if image is None:
                sample_id = example.get("id", example.get("pid", "unknown"))
                raise ValueError(f"Text rollout example is missing an image: sample_id={sample_id}")
            processed_image = pad_to_square_and_resize(image.convert("RGB"), resolution)
            all_images.append(processed_image)
            image_sizes.append(processed_image.size)

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

        image_contexts = []
        for batch_idx, example in enumerate(examples):
            prompt_data = example.get("prompt", "")
            human_value = prompt_data[0]["content"] if isinstance(prompt_data, list) and len(prompt_data) > 0 else prompt_data
            image_contexts.append(
                {
                    "decoded_text": decoded_texts[batch_idx],
                    "prompt": prompt_texts[batch_idx],
                    "payload": {
                        "id": str(example.get("pid", example.get("id", ""))),
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

        xt = init_latents.clone()
        xt[:] = img_mask_id
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
            sample_id = str(example.get("id", example.get("pid", batch_idx)))
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
            image_contexts.append(
                {
                    "valid": True,
                    "latent_shape": tuple(xt[batch_idx].shape),
                    "decoded_image": decoded_image_obj,
                    "prompt": prompt,
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
        data_loader: Union[DataLoader, dict[str, Any]],
        mask_seeds: list[int],
    ) -> torch.Tensor:
        single_batch_call = isinstance(data_loader, dict) and "scoring_data_loader" in data_loader
        if isinstance(data_loader, dict):
            if "scoring_data_loader" in data_loader:
                data_loader = data_loader["scoring_data_loader"]
            else:
                data_loader = [data_loader]

        llada_model = self._resolve_llada_forward_model(model)
        collate_fn = MaskDataCollator(self.processing_class, self.args)
        if isinstance(data_loader, DataLoader):
            scoring_loader = data_loader
        elif data_loader and torch.is_tensor(data_loader[0].get("input_ids")) and data_loader[0]["input_ids"].ndim == 1:
            batch_size = len(data_loader) if single_batch_call else min(4, len(data_loader))
            scoring_loader = DataLoader(
                data_loader,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        else:
            scoring_loader = data_loader

        all_logps: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        p_mask_prompt = float(getattr(self.args, "p_mask_prompt", 0.15))
        for seed in mask_seeds:
            seed_logps: list[torch.Tensor] = []
            seed_masks: list[torch.Tensor] = []
            for inputs in scoring_loader:
                inputs = self._prepare_input(inputs)
                labels = inputs["labels"]

                forward_inputs = self._select_model_forward_kwargs(llada_model, inputs)
                output = model.forward(
                    **forward_inputs,
                    mask_seeds=[int(seed)],
                    p_mask_prompt=p_mask_prompt,
                    temperature=self.temperature,
                )

                # Shape reference (SKIP_COMPLEMENTARY_MASKING=1, so no batch doubling):
                #   gen_loss_none_reduction: (N_edit, latent_total)
                #     N_edit = number of image-edit samples in this batch, in batch order.
                #   und_loss_none_reduction: (B, L_text)
                #     B = full batch size (edit + text samples).
                #
                # Edit-sample rows in und_loss_none are NOT all-zero: the
                # <|reserved_token_126095|>/<|reserved_token_126096|> sentinels keep
                # their token IDs as labels.  Use sample_is_image_edit for explicit routing.
                gen_loss_none = output.get("gen_loss_none_reduction")  # (N_edit, latent_total) or None
                und_loss_none = output.get("und_loss_none_reduction")  # (B, L_text)
                is_edit: torch.Tensor = inputs.get("sample_is_image_edit")  # (B,) bool

                batch_logps = None
                batch_mask = None
                if und_loss_none is None:
                    pass  # no labels passed; skip
                elif gen_loss_none is None or is_edit is None or not is_edit.any():
                    # Text-only batch
                    batch_logps = -und_loss_none  # (B, L_text)
                    # The model's forward_process masked random completion positions;
                    # F.cross_entropy returns >0 only at those positions (ignore_index=-100
                    # yields exactly 0).  Track them explicitly so the mask survives
                    # _pad_and_concat_logps zero-padding.
                    batch_mask = (und_loss_none != 0).float()  # (B, L_text)
                elif is_edit.all():
                    # Image-edit-only batch
                    batch_logps = -gen_loss_none  # (N_edit=B, latent_total)
                    batch_mask = (gen_loss_none != 0).float()  # (N_edit, latent_total)
                else:
                    # Mixed batch: route each sample to its own loss source.
                    # Zero-pad to a common length so the output is a rectangular tensor.
                    B = und_loss_none.shape[0]
                    L_text = und_loss_none.shape[-1]
                    latent_total = gen_loss_none.shape[-1]
                    max_len = max(L_text, latent_total)
                    batch_logps = torch.zeros(
                        B, max_len, dtype=und_loss_none.dtype, device=und_loss_none.device
                    )
                    batch_mask = torch.zeros(
                        B, max_len, dtype=und_loss_none.dtype, device=und_loss_none.device
                    )
                    # Text samples
                    batch_logps[~is_edit, :L_text] = -und_loss_none[~is_edit]
                    batch_mask[~is_edit, :L_text] = (und_loss_none[~is_edit] != 0).float()
                    # Edit samples
                    batch_logps[is_edit, :latent_total] = -gen_loss_none
                    batch_mask[is_edit, :latent_total] = (gen_loss_none != 0).float()

                if batch_logps is not None:
                    seed_logps.append(batch_logps)
                    seed_masks.append(batch_mask)

            # _pad_and_concat_logps zero-pads all tensors to the same last dim before
            # concatenating, so edit batches (latent_total) and text batches (L_text)
            # can be combined when both types appear across DataLoader iterations.
            # The mask is padded identically — padded positions stay 0 (non-completion).
            all_logps.append(self._pad_and_concat_logps(seed_logps))  # (total_samples, max_seq_len)
            all_masks.append(self._pad_and_concat_logps(seed_masks))

        num_iterations = len(mask_seeds)
        # Stack across seeds; pad seeds to the same last dim in case different seeds
        # happen to produce different max-sequence-length batches.
        completion_log_probs = self._pad_and_stack_logps(all_logps)  # (num_iterations, total_samples, max_seq_len)
        completion_masks = self._pad_and_stack_logps(all_masks)
        batch_size = completion_log_probs.shape[1]
        return completion_log_probs, completion_masks  # both (num_iterations, batch_size, seq_len)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        beta = float(getattr(self.args, "beta", 0.0))
        epsilon = float(getattr(self.args, "epsilon", 0.0))
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        assert num_iterations > 1, "num_iterations must be greater than 1"

        # `inputs` is a self-contained per-mini-batch dict produced by _prepare_inputs.
        # `scoring_data_loader` holds only model-input keys; training tensors are top-level.
        mask_seeds = inputs["mask_seeds"]
        # Use global_step (optimizer-step counter) to select the GRPO iteration index so
        # that all N gradient-accumulation micro-steps within one optimizer step use the
        # same iteration's old logprobs / completion masks.
        this_itr_idx = self.state.global_step % num_iterations

        current_mask_seed = mask_seeds[this_itr_idx]
        if torch.is_tensor(current_mask_seed):
            current_mask_seed = int(current_mask_seed.item())

        # Each call is a single micro-batch forward; no inner DataLoader loop.
        # _get_per_token_logps sees a dict with "scoring_data_loader" and uses it directly.
        _t0_curr = time.perf_counter()
        per_token_logps, per_token_mask = self._get_per_token_logps(model, inputs, [current_mask_seed])
        per_token_logps = per_token_logps.squeeze(0)
        per_token_mask = per_token_mask.squeeze(0)
        if per_token_logps.ndim == 1:
            per_token_logps = per_token_logps.unsqueeze(0)
            per_token_mask = per_token_mask.unsqueeze(0)

        old_per_token_logps = inputs["old_per_token_logps"][this_itr_idx]
        ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx] if beta != 0.0 else None
        advantages = inputs["advantages"]
        completion_mask = inputs.get("completion_mask", inputs.get("completion_masks"))
        if completion_mask.ndim == 3:
            completion_mask = completion_mask[this_itr_idx]
        if completion_mask.ndim == 1:
            completion_mask = completion_mask.unsqueeze(0)
        completion_mask = completion_mask.float()

        # old_per_token_logps / ref / completion_mask were padded to the global max completion
        # length across all mini-batches.  per_token_logps is computed for this mini-batch only
        # and may be shorter.  Trailing positions are zero-padding with completion_mask==0, so
        # truncating is safe.
        seq_len = per_token_logps.shape[-1]
        old_per_token_logps = old_per_token_logps[..., :seq_len]
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[..., :seq_len]
        completion_mask = completion_mask[..., :seq_len]

        # Intersect the stored completion_mask (from the old forward) with
        # per_token_mask (from the current forward) so only positions that are
        # valid in BOTH forwards contribute to the loss.  Then apply this
        # canonical mask to ALL logps tensors — including old and ref — so that
        # no tensor carries stale values at positions the other tensors have as 0.
        # Without this, e.g. ref=-90 at a position where curr=0 gives
        # log_ratio=90, exp(90)=inf, and inf*mask(=0)=NaN in IEEE 754.
        per_token_mask = per_token_mask[..., :seq_len]
        completion_mask = completion_mask * per_token_mask
        per_token_logps = per_token_logps * completion_mask
        old_per_token_logps = old_per_token_logps * completion_mask
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps * completion_mask

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        if beta != 0.0:
            log_ratio = ref_per_token_logps - per_token_logps
            per_token_kl = torch.exp(log_ratio) - log_ratio - 1

            # --- KL NaN diagnostics (rank-aware) ---
            _rank = self.accelerator.process_index
            _has_bad_kl = per_token_kl.isinf().any() or per_token_kl.isnan().any()
            _has_bad_loss_inputs = (
                per_token_logps.isnan().any() or per_token_logps.isinf().any()
                or ref_per_token_logps.isnan().any() or ref_per_token_logps.isinf().any()
                or old_per_token_logps.isnan().any() or old_per_token_logps.isinf().any()
            )
            _log_kl_diag = (self.state.global_step % 50 == 0) or _has_bad_kl or _has_bad_loss_inputs
            if _log_kl_diag:
                _cm = completion_mask.bool()
                print(
                    f"[KL-diag rank={_rank} step={self.state.global_step}] "
                    f"shapes: curr={tuple(per_token_logps.shape)} old={tuple(old_per_token_logps.shape)} "
                    f"ref={tuple(ref_per_token_logps.shape)} mask={tuple(completion_mask.shape)} | "
                    f"mask_sum={completion_mask.sum().item():.0f} total_elems={completion_mask.numel()} | "
                    f"kl_inf={per_token_kl.isinf().sum().item()} kl_nan={per_token_kl.isnan().sum().item()} | "
                    f"curr_inf={per_token_logps.isinf().sum().item()} ref_inf={ref_per_token_logps.isinf().sum().item()} "
                    f"old_inf={old_per_token_logps.isinf().sum().item()}"
                )
            if _has_bad_kl or _has_bad_loss_inputs:
                _bad_pos = (per_token_kl.isinf() | per_token_kl.isnan()).nonzero(as_tuple=False)
                _n_show = min(10, _bad_pos.shape[0])
                for _i in range(_n_show):
                    _idx = tuple(_bad_pos[_i].tolist())
                    print(
                        f"  [BAD KL rank={_rank} pos={_idx}] "
                        f"kl={per_token_kl[_idx].item():.6g} "
                        f"log_ratio={log_ratio[_idx].item():.6g} "
                        f"curr={per_token_logps[_idx].item():.6g} "
                        f"ref={ref_per_token_logps[_idx].item():.6g} "
                        f"old={old_per_token_logps[_idx].item():.6g} "
                        f"mask={completion_mask[_idx].item():.0f} "
                        f"per_token_mask={per_token_mask[_idx].item():.0f}"
                    )
                _outside_mask = ~_cm
                print(
                    f"  [LEAK CHECK rank={_rank}] nonzero outside mask: "
                    f"curr={( per_token_logps[_outside_mask] != 0).sum().item()} "
                    f"ref={( ref_per_token_logps[_outside_mask] != 0).sum().item()} "
                    f"old={( old_per_token_logps[_outside_mask] != 0).sum().item()}"
                )
            # --- end KL NaN diagnostics ---
            per_token_loss = per_token_loss + beta * per_token_kl

        denom = completion_mask.sum().clamp_min(1.0)
        loss = (per_token_loss * completion_mask).sum() / denom
        mode = "train" if self.model.training else "eval"

        if beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum() / denom).detach()
            gathered_kl = self.accelerator.gather_for_metrics(mean_kl)
            # Check which ranks contributed NaN to the gathered KL
            if gathered_kl.isnan().any():
                _nan_ranks = gathered_kl.isnan().nonzero(as_tuple=False).flatten().tolist()
                print(
                    f"[KL-NaN GATHERED rank={_rank} step={self.state.global_step}] "
                    f"mean_kl={mean_kl.item():.6g} gathered={gathered_kl.tolist()} "
                    f"nan_from_ranks={_nan_ranks} loss={loss.item():.6g}"
                )
            self._metrics[mode]["kl"].append(gathered_kl.mean().item())

        # Compute the clipped probability ratios
        epsilon_low = self.epsilon_low if hasattr(self, "epsilon_low") else epsilon
        epsilon_high = self.epsilon_high if hasattr(self, "epsilon_high") else epsilon
        is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = ((is_low_clipped.float() * completion_mask).sum() / denom).detach()
        high_clip = ((is_high_clipped.float() * completion_mask).sum() / denom).detach()
        clip_ratio = ((is_region_clipped.float() * completion_mask).sum() / denom).detach()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        # Log curr_logprobs time (covers the forward pass inside compute_loss)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        _elapsed_curr = time.perf_counter() - _t0_curr
        self._metrics[mode]["time_profile/curr_logprobs_and_update"].append(_elapsed_curr)

        _dummy = torch.zeros(1, device=loss.device, dtype=loss.dtype)
        for p in model.parameters():
            if p.requires_grad:
                _dummy = _dummy + p.view(-1)[0]  # single scalar per param — cheap
        loss = loss + _dummy * 0.0

        return loss

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        if mode == "train":
            generate_every = self.args.steps_per_generation * num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # Each element is a self-contained per-mini-batch dict.
                self._buffered_inputs = self._generate_and_score_completions(inputs)
            slot = self._step % len(self._buffered_inputs)
            result = self._buffered_inputs[slot]
            self._step += 1
            return result
        else:
            batches = self._generate_and_score_completions(inputs)
            return batches[0] if batches else inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        beta = float(getattr(self.args, "beta", 0.0))
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        prompts = [x["prompt"] for x in inputs]
        sample_modes = [x.get("task_type", "text") for x in inputs]
        image_contexts = [None] * len(inputs)
        answer_contexts = [None] * len(inputs)
        mode = "eval" if self.control.should_evaluate else "train"
        _timings: dict[str, float] = {}

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            image_edit_indices = [idx for idx, mode in enumerate(sample_modes) if mode == "image_edit"]
            image_edit_batch_size = max(1, int(getattr(self.args, "image_edit_batch_size", 1)))
            with _timer(_timings, "image_rollout"):
                for start_idx in trange(0, len(image_edit_indices), image_edit_batch_size, desc="Image Rollout"):
                    batch_indices = image_edit_indices[start_idx : start_idx + image_edit_batch_size]
                    batch_examples = [inputs[idx] for idx in batch_indices]
                    _, batch_contexts = self._rollout_image_edit_latents(unwrapped_model, batch_examples)
                    for batch_offset, sample_idx in enumerate(batch_indices):
                        image_contexts[sample_idx] = batch_contexts[batch_offset]
            image_processor = self._get_image_processor(unwrapped_model)
            text_indices = [idx for idx, sample_mode in enumerate(sample_modes) if sample_mode != "image_edit"]
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
                for start_idx in trange(0, len(text_indices), text_batch_size, desc="Text Rollout"):
                    batch_indices = text_indices[start_idx : start_idx + text_batch_size]
                    batch_examples = [inputs[idx] for idx in batch_indices]
                    _, batch_contexts = self._rollout_multimodal_text_gen(
                        unwrapped_model,
                        batch_examples,
                        image_processor,
                        generation_kwargs,
                        device,
                    )
                    for batch_offset, sample_idx in enumerate(batch_indices):
                        answer_contexts[sample_idx] = batch_contexts[batch_offset]

            scoring_examples = []
            image_data_list = [
                image_context["payload"]
                for image_context in image_contexts
                if image_context is not None
            ]
            text_data_list = [
                answer_context["payload"]
                for answer_context in answer_contexts
                if answer_context is not None
            ]
            collate_fn = MaskDataCollator(self.processing_class, self.args)
            dataset_args = self._build_lazy_supervised_data_args(unwrapped_model, image_processor)
            # Build typed item lists separately so we can interleave them.
            typed_items: list[list[dict]] = []
            global_idx = 0
            for data_list in (image_data_list, text_data_list):
                if not data_list:
                    typed_items.append([])
                    continue
                dataset = LazySupervisedDataset(
                    tokenizer=self.processing_class,
                    data_args=dataset_args,
                    list_data=data_list,
                )
                items = []
                for idx in range(len(dataset)):
                    items.append({**dataset[idx], "_sample_idx": global_idx})
                    global_idx += 1
                typed_items.append(items)
            # Interleave image-edit and text items round-robin so that every
            # DataLoader mini-batch (and especially the *last* one used for
            # gradient accumulation) activates both generation and understanding
            # parameters.  This prevents DeepSpeed ZeRO-2 reduce-scatter
            # deadlocks caused by unused-parameter backward hooks.
            image_items, text_items = typed_items
            interleaved: list[dict] = []
            ii, ti = 0, 0
            while ii < len(image_items) or ti < len(text_items):
                if ii < len(image_items):
                    interleaved.append(image_items[ii]); ii += 1
                if ti < len(text_items):
                    interleaved.append(text_items[ti]); ti += 1
            scoring_examples = interleaved
            if not scoring_examples:
                raise ValueError("No scored batches were built from generated completions.")
        mask_seeds = torch.randint(0, 2**12, (num_iterations,), device=device)

        mask_seed_list = mask_seeds.detach().cpu().tolist() if torch.is_tensor(mask_seeds) else list(mask_seeds)
        with torch.no_grad():
            with _timer(_timings, "old_logprobs"):
                old_per_token_logps, completion_masks = self._get_per_token_logps(
                    self.model,
                    scoring_examples,
                    mask_seed_list,
                )
                # Zero out non-completion positions so all downstream math (ratio, KL,
                # loss) only ever sees real log-probs at completion positions and exact
                # zeros elsewhere.  completion_masks is built alongside the logps inside
                # _get_per_token_logps, so it tracks the same routing and padding.
                old_per_token_logps = old_per_token_logps * completion_masks
            ref_per_token_logps = None
            if beta != 0.0:
                with _timer(_timings, "ref_logprobs"):
                    if getattr(self, "ref_model", None) is not None:
                        ref_per_token_logps, _ = self._get_per_token_logps(
                            self.ref_model, scoring_examples, mask_seed_list
                        )
                    else:
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        if hasattr(unwrapped, "disable_adapter"):
                            with unwrapped.disable_adapter():
                                ref_per_token_logps, _ = self._get_per_token_logps(
                                    self.model, scoring_examples, mask_seed_list
                                )
                        else:
                            ref_per_token_logps, _ = self._get_per_token_logps(
                                self.model, scoring_examples, mask_seed_list
                            )
                    ref_per_token_logps = ref_per_token_logps * completion_masks

        with _timer(_timings, "reward"):
            reward_specs = [
                (
                    [example for example, image_context in zip(inputs, image_contexts) if image_context is not None],
                    [image_context["prompt"] for image_context in image_contexts if image_context is not None],
                    [image_context["decoded_image"] for image_context in image_contexts if image_context is not None],
                    [perceptual_score_reward_func],
                ),
                (
                    [example for example, answer_context in zip(inputs, answer_contexts) if answer_context is not None],
                    [answer_context["prompt"] for answer_context in answer_contexts if answer_context is not None],
                    [
                        [{"role": "assistant", "content": answer_context["decoded_text"]}]
                        for answer_context in answer_contexts
                        if answer_context is not None
                    ],
                    [strict_format_reward_func, correctness_reward_func],
                ),
            ]
            local_rewards = []
            for reward_inputs, branch_prompts, branch_completions, reward_fns in reward_specs:
                n_branch = len(branch_completions) if branch_completions else 0
                rewards_per_func = torch.zeros(n_branch, len(reward_fns), device=device)
                if branch_completions:
                    for i, reward_func in enumerate(reward_fns):
                        if isinstance(reward_func, nn.Module):
                            reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                        else:
                            reward_func_name = reward_func.__name__
                        with profiling_context(self, reward_func_name):
                            keys = [key for key in reward_inputs[0] if key not in ["prompt", "completion"]]
                            reward_kwargs = {key: [example.get(key) for example in reward_inputs] for key in keys}
                            if reward_func_name == "coding_reward_func":
                                reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                            try:
                                output_reward_func = reward_func(
                                    prompts=branch_prompts,
                                    completions=branch_completions,
                                    step=self._step,
                                    run_name=self.args.output_dir,
                                    **reward_kwargs,
                                )
                            except Exception:
                                output_reward_func = [torch.nan for _ in branch_completions]
                            output_reward_func = [
                                reward if reward is not None else torch.nan for reward in output_reward_func
                            ]
                            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

                    
                    
                    for i, reward_func in enumerate(reward_fns):
                        if isinstance(reward_func, nn.Module):
                            reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                        else:
                            reward_func_name = reward_func.__name__
                        
                        mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                        std_rewards = nanstd(rewards_per_func[:, i]).item()
                        self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
                        self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
                    local_rewards.append(rewards_per_func.nansum(dim=1))

        if not local_rewards:
            raise ValueError("No rewards were produced for generated completions.")

        rewards = torch.cat(local_rewards, dim=0)
        local_n = rewards.size(0)
        rewards = gather(rewards)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0
        is_std_zero = std_grouped_rewards < 1e-6
        process_slice = slice(
            self.accelerator.process_index * local_n,
            (self.accelerator.process_index + 1) * local_n,
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_masks.float().sum(-1).mean(dim=0)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        prompts_text = []
        completions_text = []
        for sample_mode, prompt, image_context, answer_context in zip(
            sample_modes, prompts, image_contexts, answer_contexts
        ):
            if sample_mode == "image_edit":
                prompts_text.append(image_context["prompt"] if image_context is not None else prompt)
                completions_text.append("")
            else:
                prompts_text.append(answer_context["prompt"] if answer_context is not None else prompt)
                completions_text.append("" if answer_context is None else answer_context["decoded_text"])

        prepared_batches = DataLoader(
            scoring_examples,
            batch_size=2,
            shuffle=False,
            collate_fn=self._build_scored_batch_collator(
                collate_fn,
                advantages,
                completion_masks,
                old_per_token_logps,
                ref_per_token_logps,
                num_iterations,
            ),
        )

        # Log time profile through self._metrics so it goes through the HF callback system
        for k, v in _timings.items():
            self._metrics[mode][f"time_profile/{k}"].append(v)

        _TRAINING_KEYS = frozenset({"advantages", "completion_mask", "completion_masks",
                                    "old_per_token_logps", "ref_per_token_logps"})
        mask_seeds_val = mask_seeds.detach() if torch.is_tensor(mask_seeds) else mask_seeds
        return [
            {
                "scoring_data_loader": batch["scoring_instances"],
                **{k: v for k, v in batch.items() if k in _TRAINING_KEYS},
                "mask_seeds": mask_seeds_val,
            }
            for batch in prepared_batches  # DataLoader is iterable; no need to materialise separately
        ]
