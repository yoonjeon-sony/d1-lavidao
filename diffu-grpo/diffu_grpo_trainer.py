import copy
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
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
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample, selective_log_softmax
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    ,
)
if is_peft_available():
    from peft import PeftConfig

# Required by LaVida-O sampling path.
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")

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

    @staticmethod
    def _make_generator(device: torch.device, seed: int) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    @staticmethod
    def _detach_structure(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {k: DiffuGRPOTrainer._detach_structure(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DiffuGRPOTrainer._detach_structure(v) for v in value]
        if isinstance(value, tuple):
            return tuple(DiffuGRPOTrainer._detach_structure(v) for v in value)
        return value

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

    def _build_text_completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        if completion_ids.numel() == 0:
            return torch.zeros_like(completion_ids, dtype=torch.int)
        eos_id = self.processing_class.eos_token_id
        if eos_id is None:
            return torch.ones_like(completion_ids, dtype=torch.int)
        is_eos = completion_ids == eos_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).unsqueeze(0)
        return (seq_idx <= eos_idx.unsqueeze(1)).int()

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
        example: dict[str, Any],
        image_processor,
        generation_kwargs: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        prompt_text = self._build_llada_prompt(example["prompt"])
        if example.get("image") is not None and "<image>" not in prompt_text:
            raise ValueError(
                "Text rollout example includes an image but the prompt does not contain '<image>': "
                f"sample_index={idx}"
            )
        prompt_ids = tokenizer_image_token(
            prompt_text,
            self.processing_class,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).to(device)
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[-self.max_prompt_length :]
        prompt_ids = prompt_ids.unsqueeze(0)
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
        image = self._load_image(example.get("image"))
        if image is None:
            return None, None
        if image_processor is None:
            return None, None

        resolution = int(self.args.image_edit_resolution)
        processed_image = pad_to_square_and_resize(image.convert("RGB"), resolution)
        image_tensor = process_images([processed_image], image_processor, model.config)
        image_tensor = self._normalize_mm_image_payload(
            image_tensor,
            dtype=model.dtype,
            device=device,
        )
        self._check_mm_payload_before_prepare_inputs(image_tensor)
        image_sizes = [processed_image.size]

        position_ids = None
        attention_mask = prompt_mask
        if image_tensor is not None:
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
        else:
            inputs_embeds = model.get_model().embed_tokens(prompt_ids)

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
        norm_prompt_ids, completion_ids = self._normalize_text_rollout(generated, prompt_ids, prefix_lm)
        decoded_text = self.processing_class.decode(
            completion_ids.squeeze(0), skip_special_tokens=True
        )
        source_images = example.get("image")
        if source_images is None:
            source_images = []
        elif not isinstance(source_images, list):
            source_images = [source_images]

        return completion_ids.detach(), {
            "id": str(example.get("pid", example.get("id", ""))),
            "image": source_images,
            "conversations": [
                {"from": "human", "value": example["prompt"]["content"]},
                {"from": "gpt", "value": decoded_text},
            ],
        }
        

    @staticmethod
    def _debug_mm_image_shapes_enabled() -> bool:
        flag = os.environ.get("DEBUG_MM_IMAGE_SHAPES", "")
        return flag.lower() not in {"", "0", "false", "no"}

    @staticmethod
    def _describe_mm_image_payload(images: Any) -> str:
        if isinstance(images, list):
            shapes = [tuple(img.shape) if torch.is_tensor(img) else type(img).__name__ for img in images]
            ranks = [img.ndim if torch.is_tensor(img) else None for img in images]
            return f"type=list len={len(images)} ranks={ranks} shapes={shapes}"
        if torch.is_tensor(images):
            return f"type=tensor rank={images.ndim} shape={tuple(images.shape)}"
        return f"type={type(images).__name__}"


    def _normalize_mm_image_payload(self, image_tensor: Any, *, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        if isinstance(image_tensor, list):
            return [_x.to(dtype=dtype, device=device) for _x in image_tensor]
        image_tensor = image_tensor.to(dtype=dtype, device=device)
        return [img for img in image_tensor]

    def _check_mm_payload_before_prepare_inputs(self, images: list[torch.Tensor]) -> None:
        normalized_images = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
        predicted_shapes = [tuple(image.shape) for image in normalized_images]
        concat_images = torch.cat(normalized_images, dim=0)
        # self._log_mm_image_shapes(
        #     "rollout_image_edit_latents.predicted_prepare_inputs",
        #     normalized_shapes=predicted_shapes,
        #     concat_shape=tuple(concat_images.shape),
        # )
        if concat_images.ndim != 4:
            raise ValueError(
                "Image payload would become non-4D before encode_images: "
                f"input={self._describe_mm_image_payload(images)}, "
                f"normalized_shapes={predicted_shapes}, "
                f"concat_shape={tuple(concat_images.shape)}"
            )

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
        self._check_mm_payload_before_prepare_inputs(image_tensor)

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

    @staticmethod
    def _repeat_batch_value(value: Any, repeat_count: int) -> Any:
        if repeat_count == 1 or value is None:
            return value
        if torch.is_tensor(value):
            if value.ndim == 0:
                return value
            return torch.cat([value] * repeat_count, dim=0)
        if isinstance(value, list):
            repeated = []
            for _ in range(repeat_count):
                repeated.extend(copy.deepcopy(value))
            return repeated
        if isinstance(value, tuple):
            repeated = []
            for _ in range(repeat_count):
                repeated.extend(copy.deepcopy(list(value)))
            return tuple(repeated)
        return value

    def _repeat_batch_inputs(self, inputs: dict[str, Any], repeat_count: int) -> dict[str, Any]:
        return {
            key: self._repeat_batch_value(value, repeat_count)
            for key, value in inputs.items()
        }

    @staticmethod
    def _pad_and_concat_logps(logps_per_batch: list[torch.Tensor]) -> torch.Tensor:
        if not logps_per_batch:
            return torch.empty(0)
        max_keep = max(logps.shape[-1] for logps in logps_per_batch)
        padded = []
        for logps in logps_per_batch:
            if logps.shape[-1] == max_keep:
                padded.append(logps)
                continue
            pad_shape = (*logps.shape[:-1], max_keep - logps.shape[-1])
            pad = torch.zeros(pad_shape, dtype=logps.dtype, device=logps.device)
            padded.append(torch.cat([logps, pad], dim=-1))
        return torch.cat(padded, dim=1)

    def _build_masked_indices_gen(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        mask_seeds: list[int],
        *,
        is_unitok: bool,
        is_unitok_submask: bool,
    ) -> torch.Tensor:
        masked_indices_gen = []
        for mask_seed in mask_seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(mask_seed))
            if is_unitok_submask:
                random_matrix = torch.rand((batch_size, seq_len * 8), device=device, generator=generator)
                is_mask_gen = random_matrix < float(self.args.p_mask_image)
                is_mask_gen = is_mask_gen.view(batch_size, 8, seq_len)
            else:
                random_matrix = torch.rand((batch_size, seq_len), device=device, generator=generator)
                is_mask_gen = random_matrix < float(self.args.p_mask_image)
                if is_unitok:
                    is_mask_gen = is_mask_gen.unsqueeze(1).repeat(1, 8, 1)
            masked_indices_gen.append(is_mask_gen)
        return torch.cat(masked_indices_gen, dim=0)

    def _get_per_token_logps(
        self,
        model,
        data_loader: Union[DataLoader, dict[str, Any]],
        mask_seeds: list[int],
    ) -> torch.Tensor:
        if isinstance(data_loader, dict):
            if "scoring_data_loader" in data_loader:
                data_loader = data_loader["scoring_data_loader"]
            else:
                data_loader = [data_loader]

        llada_model = self._resolve_llada_forward_model(model)
        llada_config = getattr(llada_model.get_model(), "config", llada_model.config)
        is_unitok = "unitok" in getattr(llada_config, "mm_vqvae", "")
        is_unitok_submask = bool(getattr(llada_config, "mm_submask", False))
        num_image_tokens = int(getattr(self.args, "num_gen_image_tokens", 0))
        repeat_count = len(mask_seeds)
        all_logps: list[torch.Tensor] = []

        for inputs in data_loader:
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            attention_mask = inputs["attention_mask"].bool()
            prompt_index = attention_mask & labels.eq(IGNORE_INDEX)
            logits_to_keep = int(labels.ne(IGNORE_INDEX).sum(dim=1).max().item())

            masked_indices = []
            for mask_seed in mask_seeds:
                generator = torch.Generator(device=input_ids.device)
                generator.manual_seed(int(mask_seed))
                random_matrix = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
                is_mask = (~prompt_index) & attention_mask & (random_matrix < float(self.args.p_mask_prompt))
                masked_indices.append(is_mask)
            masked_indices = torch.cat(masked_indices, dim=0)

            masked_indices_gen = None
            gen_latents = inputs.get("gen_latents")
            if inputs.get("images_gen") is not None and num_image_tokens > 0:
                masked_indices_gen = self._build_masked_indices_gen(
                    batch_size=gen_latents.shape[0],
                    seq_len=num_image_tokens,
                    device=input_ids.device,
                    mask_seeds=mask_seeds,
                    is_unitok=is_unitok,
                    is_unitok_submask=is_unitok_submask,
                )

            repeated_inputs = self._repeat_batch_inputs(inputs, repeat_count)
            output = model.forward(
                **repeated_inputs,
                masked_indices=masked_indices,
                masked_indices_gen=masked_indices_gen,
            )

            batch_logps = None
            answer_logits = output.get("und_logits")
            if answer_logits is not None and logits_to_keep > 0:
                answer_logits = answer_logits[:, -logits_to_keep:, :].div(self.temperature)
                completion_ids = repeated_inputs["input_ids"][:, -logits_to_keep:]
                batch_logps = selective_log_softmax(answer_logits, completion_ids).view(
                    repeat_count, input_ids.shape[0], logits_to_keep
                )

            image_gen_logits = output.get("gen_logits")
            image_mask = output.get("gen_x_mask")
            image_targets = output.get("gen_x0_gt")
            if image_gen_logits is not None and image_mask is not None and image_targets is not None:
                image_mask = image_mask.reshape(image_mask.shape[0], -1)
                image_targets = image_targets.reshape(image_targets.shape[0], -1)
                flat_targets = image_targets[image_mask]
                image_logps_flat = selective_log_softmax(
                    image_gen_logits.div(self.temperature),
                    flat_targets,
                )
                counts = image_mask.sum(dim=-1).tolist()
                max_keep = max(counts) if counts else 0
                padded_logps = image_gen_logits.new_zeros((image_mask.shape[0], max_keep))
                cursor = 0
                for row_idx, keep_count in enumerate(counts):
                    keep_count = int(keep_count)
                    if keep_count == 0:
                        continue
                    padded_logps[row_idx, :keep_count] = image_logps_flat[cursor : cursor + keep_count]
                    cursor += keep_count
                image_logps = padded_logps.view(repeat_count, gen_latents.shape[0], max_keep)
                if batch_logps is not None:
                    raise NotImplementedError("Mixed text and image logprob batches are not supported yet.")
                batch_logps = image_logps

            if batch_logps is not None:
                all_logps.append(batch_logps)

        return self._pad_and_concat_logps(all_logps)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        beta = float(getattr(self.args, "beta", 0.0))
        epsilon = float(getattr(self.args, "epsilon", 0.0))
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        completion_mask = inputs["completion_mask"].float()
        mask_seeds = inputs["mask_seeds"]
        this_itr_idx = self._step % num_iterations
        per_token_logps = self._get_per_token_logps(model, inputs, [mask_seeds[this_itr_idx]])
        if beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        if num_iterations > 1:
            old_per_token_logps = inputs["old_per_token_logps"][this_itr_idx]
        else:
            old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if beta != 0.0:
            per_token_loss = per_token_loss + beta * per_token_kl

        denom = completion_mask.sum().clamp_min(1.0)
        loss = (per_token_loss * completion_mask).sum() / denom
        mode = "eval" if self.control.should_evaluate else "train"

        if beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / denom
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / denom
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        num_iterations = int(getattr(self.args, "num_iterations", 1))
        if mode == "train":
            grad_accum_steps = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
            if getattr(self, "_buffered_inputs", None) is None:
                self._buffered_inputs = [None] * grad_accum_steps
            if self.state.global_step % num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % grad_accum_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % grad_accum_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        sample_modes = [x.get("task_type", "text") for x in inputs]

        # text_completion_ids = [None] * len(inputs)
        image_contexts = [None] * len(inputs)
        answer_contexts = [None] * len(inputs)

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            image_edit_indices = [idx for idx, mode in enumerate(sample_modes) if mode == "image_edit"]
            image_edit_batch_size = max(1, int(getattr(self.args, "image_edit_batch_size", 1)))
            for start_idx in trange(0, len(image_edit_indices), image_edit_batch_size, desc="Image Rollout"):
                batch_indices = image_edit_indices[start_idx : start_idx + image_edit_batch_size]
                batch_examples = [inputs[idx] for idx in batch_indices]
                _, batch_contexts = self._rollout_image_edit_latents(unwrapped_model, batch_examples)
                for batch_offset, sample_idx in enumerate(batch_indices):
                    # completion_tokens_per_sample[sample_idx] = image_completion_ids[batch_offset].detach()
                    image_contexts[sample_idx] = batch_contexts[batch_offset]
            image_processor = self._get_image_processor(unwrapped_model)

            for idx, example in tqdm(enumerate(inputs), desc=f"Text Rollout"):
                mode = sample_modes[idx]
                if mode == "image_edit":
                    continue
                else:
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
                    _, text_context = self._rollout_multimodal_text_gen(
                        unwrapped_model,
                        example,
                        image_processor,
                        generation_kwargs,
                        device,
                    )
                
                # text_completion_ids[idx] = sample_text_completion_ids.squeeze(0).detach()
                answer_contexts[idx] = text_context

            image_data_list = [
                image_context["payload"]
                for image_context in image_contexts
            ] # len(image_contexts)
            text_data_list = [
                answer_context["payload"] 
                for answer_context in answer_contexts
            ] # len(answer_contexts)
            image_dataset = LazySupervisedDataset(
                tokenizer=self.processing_class,
                data_args=self.args,
                list_data=image_data_list,
            )
            text_dataset = LazySupervisedDataset(
                tokenizer=self.processing_class,
                data_args=self.args,
                list_data=text_data_list,
            )
            dataset = ConcatDataset(image_dataset, text_dataset)
            collate_fn = MaskDataCollator(self.processing_class, self.args)
            data_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            mask_seeds = torch.randint(0, 2**12, (self.args.num_iterations,), device=device)

        
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(
                self.model, data_loader, mask_seeds
            )
            if beta != 0.0:
                if getattr(self, "ref_model", None) is not None:
                    all_ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, data_loader, mask_seeds
                    )
                else:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped, "disable_adapter"):
                        with unwrapped.disable_adapter():
                            all_ref_per_token_logps = self._get_per_token_logps(
                                self.model, data_loader, mask_seeds
                            )
                    else:
                        all_ref_per_token_logps = self._get_per_token_logps(
                            self.model, data_loader, mask_seeds
                        )

        completions = []
        for idx, (mode, completion) in enumerate(zip(sample_modes, completions_text)):
            if mode == "image_edit":
                image_ctx = image_contexts[idx]
                completions.append(None if image_ctx is None else image_ctx.get("decoded_image"))
            else:
                completions.append([{"role": "assistant", "content": completion}])

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example.get(key) for example in inputs] for key in keys}
                if reward_func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                try:
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        step=self._step,
                        run_name=self.args.output_dir,
                        **reward_kwargs,
                    )
                except Exception:
                    output_reward_func = [torch.nan for _ in completions]
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for kwargs: {row_reward_kwargs}. "
                "Ensure at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        reward_weights = self.reward_weights
        if not torch.is_tensor(reward_weights):
            reward_weights = torch.tensor(reward_weights, dtype=torch.float32, device=device)
        else:
            reward_weights = reward_weights.to(device)
        rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            wandb_payloads = []
            for mode, completion_text, image_ctx in zip(sample_modes, completions_text, image_contexts):
                payload = {
                    "completion": completion_text,
                    "completion_image": None,
                }
                if mode == "image_edit" and image_ctx is not None:
                    payload["completion_image"] = image_ctx.get("decoded_image")
                wandb_payloads.append(payload)
            wandb_payloads_to_log = gather_object(wandb_payloads)
            if self.accelerator.is_main_process:
                # if is_rich_available():
                print_prompt_completions_sample(
                    prompts_to_log,
                    completions_to_log,
                    rewards_to_log,
                    self.state.global_step,
                )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    table = wandb.Table(
                        columns=["step", "prompt", "completion", "completion_image", "reward"]
                    )
                    for prompt, reward_value, payload in zip(
                        prompts_to_log, rewards_to_log, wandb_payloads_to_log
                    ):
                        completion_image = payload.get("completion_image")
                        if completion_image is not None:
                            completion_image = wandb.Image(completion_image)
                        table.add_data(
                            str(self.state.global_step),
                            prompt,
                            payload.get("completion"),
                            completion_image,
                            reward_value,
                        )
                    wandb.log({"completions": table})

        return {
            "sample_modes": sample_modes,
            "text_prompt_ids": text_prompt_ids,
            "text_completion_ids": text_completion_ids,
            "image_contexts": image_contexts,
            "score_payloads": score_payloads,
            "score_mask_seeds": mask_seed_list,
            "completion_ids": completion_ids.detach(),
            "completion_mask": completion_mask.float().detach(),
            "old_per_token_logps": None if all_old_per_token_logps is None else all_old_per_token_logps.detach(),
            "ref_per_token_logps": None if all_ref_per_token_logps is None else all_ref_per_token_logps.detach(),
            "advantages": advantages.detach(),
            "mask_seeds": mask_seeds.detach() if torch.is_tensor(mask_seeds) else mask_seeds,
        }
