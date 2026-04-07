import copy
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union, List

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from accelerate.utils import gather, gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers.utils import is_peft_available
from trl.extras.profiling import profiling_context, profiling_decorator
# from trl.import_utils import is_rich_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig

# Required by LaVida-O sampling path.
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
LAVIDA_ROOT = REPO_ROOT / "LaVida-O"
if str(LAVIDA_ROOT) not in sys.path:
    sys.path.insert(0, str(LAVIDA_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from llava.mm_utils import pad_to_square_and_resize, process_images, tokenizer_image_token
from llava.model.language_model.llava_llada import (
    LlavaLladaConfig,
    LlavaLladaForMaskedDiffusion,
)
from llava.model.language_model.llada.modeling_llada import LLaDAModelLM
from llava.model.language_model.llada.generate import (
    add_gumbel_noise,
    cosine_schedule_2,
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
    def _pad_sequence(tokenizer, seqs, padding_value: int):
        if len(seqs) == 0:
            return torch.empty(0, 0, dtype=torch.long)
        if tokenizer.padding_side == "left":
            seqs = [torch.flip(x, [0]) for x in seqs]
        out = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)
        if tokenizer.padding_side == "left":
            out = torch.flip(out, [1])
        return out

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

    # def _log_mm_image_shapes(self, message: str, **payload: Any) -> None:
        # if not self._debug_mm_image_shapes_enabled():
        #     return
        # details = ", ".join(f"{key}={value}" for key, value in payload.items())
        # print(f"[DEBUG_MM_IMAGE_SHAPES] {message}: {details}")

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

        valid_indices = []
        all_input_ids = []
        all_edit_images = []
        image_sizes = []
        all_enc_embeddings = []
        vq_latents = []
        for batch_idx, example in enumerate(examples):
            edit_image = self._load_image(example.get("image"))
            if edit_image is None:
                continue
            instruction = self._extract_image_edit_instruction(example)
            valid_indices.append(batch_idx)
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

        if not valid_indices:
            return torch.stack(batch_latents, dim=0), image_contexts

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

            for batch_idx in range(len(valid_indices)):
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
        decoded_images = None
        if hasattr(model, "decode_image_gen"):
            decoded_images = model.decode_image_gen(
                xt,
                gen_cfg["image_resolution"],
                gen_cfg["image_resolution"],
            )

        modality_indices = is_gen | is_gen_enc if getattr(base_model.config, "enc_use_image_branch", False) else is_gen
        for local_idx, batch_idx in enumerate(valid_indices):
            batch_latents[batch_idx] = xt[local_idx].detach()
            image_contexts[batch_idx] = {
                "valid": True,
                "inputs_embeds": inputs_embeds[local_idx].detach(),
                "inputs_embeds_uncond": inputs_embeds_uncond[local_idx].detach(),
                "is_gen": is_gen[local_idx].detach(),
                "is_gen_enc": is_gen_enc[local_idx].detach(),
                "base_inputs_embeds": inputs_embeds[local_idx].detach(),
                "modality_indices": modality_indices[local_idx].detach(),
                "gen_shape": gen_shape,
                "guidance_scale": gen_cfg["guidance_scale"],
                "latent_shape": tuple(xt[local_idx].shape),
                "decoded_image": None if decoded_images is None else decoded_images[local_idx],
            }
        return torch.stack(batch_latents, dim=0), image_contexts

    def _build_text_score_payload(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        mask_seeds: list[int],
    ) -> dict[str, Any]:
        valid_len = int(completion_mask.sum().item())
        completion_ids = completion_ids[:valid_len]
        sequence_ids = torch.cat([prompt_ids, completion_ids], dim=0)
        seq_len = sequence_ids.size(0)
        prompt_len = int(prompt_ids.size(0))
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=sequence_ids.device)
        prompt_index[:prompt_len] = True

        masked_indices = torch.zeros(len(mask_seeds), seq_len, dtype=torch.bool, device=sequence_ids.device)
        for iter_idx, mask_seed in enumerate(mask_seeds):
            random_matrix = torch.rand(
                seq_len,
                device=sequence_ids.device,
                generator=self._make_generator(sequence_ids.device, mask_seed),
            )
            masked_indices[iter_idx] = (~prompt_index) | (
                prompt_index & (random_matrix < float(self.args.p_mask_prompt))
            )

        text_target_ids = torch.full_like(sequence_ids, -100)
        if valid_len > 0:
            text_target_ids[prompt_len:] = completion_ids

        return {
            "valid": valid_len > 0,
            "sequence_ids": sequence_ids,
            # Scoring intentionally disables prefix-LM even when rollout uses it.
            # This avoids the flex-attention `create_block_mask` dependency here.
            "prompt_len": None,
            "modality_indices": torch.zeros(seq_len, dtype=torch.bool, device=sequence_ids.device),
            "new_token_mask": torch.zeros(seq_len, dtype=torch.bool, device=sequence_ids.device),
            "masked_indices": masked_indices,
            "text_target_ids": text_target_ids,
            "gen_targets": None,
            "gen_latents_masked": None,
            "gen_shape": None,
            "completion_axis_text": torch.arange(valid_len, device=sequence_ids.device, dtype=torch.long),
            "completion_axis_image": torch.empty(0, device=sequence_ids.device, dtype=torch.long),
        }

    def _build_image_score_payload(
        self,
        image_ctx: dict[str, Any],
        completion_flat: torch.Tensor,
        valid_len: int,
        mask_seeds: list[int],
        device: torch.device,
    ) -> dict[str, Any]:
        if image_ctx is None or not image_ctx.get("valid", False):
            return {
                "valid": False,
                "base_inputs_embeds": None,
                "prompt_len": None,
                "modality_indices": None,
                "new_token_mask": None,
                "masked_indices": None,
                "text_target_ids": None,
                "gen_targets": None,
                "gen_latents_masked": None,
                "gen_shape": None,
                "completion_axis_text": torch.empty(0, device=device, dtype=torch.long),
                "completion_axis_image": torch.empty(0, device=device, dtype=torch.long),
            }

        latent_shape = tuple(image_ctx["latent_shape"])
        if len(latent_shape) > 1 and latent_shape[0] == 1:
            latent_shape = latent_shape[1:]
        gen_targets = completion_flat[:valid_len].view(*latent_shape).to(device)
        gen_latents_masked = []
        for mask_seed in mask_seeds:
            masked = gen_targets.clone()
            random_values = torch.rand(
                masked.shape,
                device=device,
                generator=self._make_generator(device, mask_seed),
            )
            masked[random_values < 0.5] = 8193
            gen_latents_masked.append(masked)

        new_token_mask = image_ctx["is_gen"].to(device=device, dtype=torch.bool).squeeze(0)
        modality_indices = image_ctx["modality_indices"].to(device=device, dtype=torch.bool)

        seq_len = new_token_mask.size(0)
        return {
            "valid": True,
            "base_inputs_embeds": image_ctx["base_inputs_embeds"].to(device=device),
            "prompt_len": None,
            "modality_indices": modality_indices,
            "new_token_mask": new_token_mask,
            "masked_indices": torch.zeros(len(mask_seeds), seq_len, dtype=torch.bool, device=device),
            "text_target_ids": torch.full((seq_len,), -100, dtype=torch.long, device=device),
            "gen_targets": gen_targets,
            "gen_latents_masked": torch.stack(gen_latents_masked, dim=0),
            "gen_shape": image_ctx["gen_shape"],
            "completion_axis_text": torch.empty(0, device=device, dtype=torch.long),
            "completion_axis_image": torch.arange(valid_len, device=device, dtype=torch.long),
        }

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

    def _compute_unified_per_token_logps(
        self,
        model,
        inputs: dict[str, Any],
        mask_seeds: list[int],
    ) -> torch.Tensor:
        completion_mask = inputs["completion_mask"]
        batch_size, token_axis = completion_mask.shape
        num_iterations = len(mask_seeds)
        out = torch.zeros(num_iterations, batch_size, token_axis, device=completion_mask.device)
        unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        llada_model = self._resolve_llada_forward_model(unwrapped_model)
        multimodal_model = llada_model.get_model()
        noise_embeddings = multimodal_model.transformer.wte(
            torch.tensor([self.args.mask_id], device=completion_mask.device)
        ).view(1, 1, -1)
        precomputed_mask_seeds = [int(x) for x in inputs.get("score_mask_seeds", mask_seeds)]

        for sample_idx in range(batch_size):
            payload = inputs["score_payloads"][sample_idx]
            if not payload.get("valid", False):
                continue

            masked_indices = payload["masked_indices"]
            gen_latents_masked = payload["gen_latents_masked"]
            if masked_indices.size(0) != num_iterations:
                seed_positions = []
                for mask_seed in mask_seeds:
                    try:
                        seed_positions.append(precomputed_mask_seeds.index(int(mask_seed)))
                    except ValueError as exc:
                        raise ValueError("Requested seed was not precomputed for scoring payloads") from exc
                masked_indices = masked_indices[seed_positions]
                if gen_latents_masked is not None:
                    gen_latents_masked = gen_latents_masked[seed_positions]

            if payload.get("sequence_ids") is not None:
                base_inputs_embeds = multimodal_model.transformer.wte(payload["sequence_ids"].unsqueeze(0))
            else:
                base_inputs_embeds = payload["base_inputs_embeds"].unsqueeze(0)
            base_inputs_embeds = base_inputs_embeds.to(device=completion_mask.device, dtype=noise_embeddings.dtype)
            base_inputs_embeds = base_inputs_embeds.expand(num_iterations, -1, -1).clone()
            inputs_embeds = torch.where(
                masked_indices.unsqueeze(-1),
                noise_embeddings.to(dtype=base_inputs_embeds.dtype),
                base_inputs_embeds,
            )

            new_token_mask = payload["new_token_mask"].unsqueeze(0).expand(num_iterations, -1)
            modality_indices = payload["modality_indices"].unsqueeze(0).expand(num_iterations, -1)
            if gen_latents_masked is not None:
                gen_latents_comp_embeds = multimodal_model.call_gen_embedding(
                    gen_latents_masked,
                    gen_shape=payload["gen_shape"],
                )
                gen_latents_comp_embeds = pad_along_last_dim(gen_latents_comp_embeds, llada_model.config.d_model)
                inputs_embeds = inputs_embeds.masked_scatter(
                    new_token_mask.unsqueeze(-1),
                    gen_latents_comp_embeds.reshape(-1, llada_model.config.d_model),
                )

            prompt_len = payload["prompt_len"]
            if prompt_len is not None:
                prompt_len = prompt_len.view(1).repeat(num_iterations)

            output = LLaDAModelLM.forward(
                llada_model,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                position_ids=None,
                labels=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                prompt_len=prompt_len,
                num_items_in_batch=None,
                modality_indices=modality_indices,
            )

            text_targets = payload["text_target_ids"].unsqueeze(0).expand(num_iterations, -1)
            text_mask = text_targets.ne(-100)
            if text_mask.any():
                text_logps = -F.cross_entropy(
                    output.logits[text_mask].float(),
                    text_targets[text_mask],
                    reduction="none",
                ).view(num_iterations, -1)
                text_axis = payload["completion_axis_text"]
                out[:, sample_idx, text_axis] = text_logps.to(torch.float32)

            gen_targets = payload["gen_targets"]
            if gen_targets is not None:
                hidden_states = output.hidden_states[-1]
                gen_hidden_states = hidden_states[new_token_mask]
                gen_hidden_states = maybe_truncate_last_dim(gen_hidden_states, llada_model.config.d_model_gen)

                gen_mask = gen_latents_masked == 8193
                timesteps = gen_mask.float().sum(-1) / gen_mask.shape[-1]
                expanded_gen_targets = gen_targets.unsqueeze(0).expand(num_iterations, *gen_targets.shape)
                gen_logits = multimodal_model.call_gen_predictor(
                    gen_hidden_states,
                    payload["gen_shape"],
                    timesteps=timesteps,
                )

                if expanded_gen_targets.dim() == 3:
                    batch_dim, depth_dim, seq_dim = expanded_gen_targets.shape
                    gen_logits = gen_logits.view(batch_dim, seq_dim, *gen_logits.shape[-2:]).permute(0, 2, 1, 3)
                    image_logps = -F.cross_entropy(
                        gen_logits.reshape(-1, gen_logits.shape[-1]).float(),
                        expanded_gen_targets.reshape(-1),
                        reduction="none",
                    ).view(batch_dim, depth_dim * seq_dim)
                else:
                    image_logps = -F.cross_entropy(
                        gen_logits.float(),
                        expanded_gen_targets.reshape(-1),
                        reduction="none",
                    ).view(num_iterations, -1)

                image_axis = payload["completion_axis_image"]
                out[:, sample_idx, image_axis] = image_logps.to(torch.float32)
        return out

    def _compute_unified_per_token_logps_batched_grad(
        self,
        model,
        inputs: dict[str, Any],
        mask_seed: int,
    ) -> torch.Tensor:
        completion_mask = inputs["completion_mask"]
        batch_size, token_axis = completion_mask.shape
        out = torch.zeros(batch_size, token_axis, device=completion_mask.device)
        unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        llada_model = self._resolve_llada_forward_model(unwrapped_model)
        multimodal_model = llada_model.get_model()
        dtype = multimodal_model.transformer.wte.weight.dtype
        noise_embedding = multimodal_model.transformer.wte(
            torch.tensor([self.args.mask_id], device=completion_mask.device)
        ).view(1, -1)
        precomputed_mask_seeds = [int(x) for x in inputs.get("score_mask_seeds", [mask_seed])]
        try:
            seed_position = precomputed_mask_seeds.index(int(mask_seed))
        except ValueError as exc:
            raise ValueError("Requested seed was not precomputed for scoring payloads") from exc

        valid_entries = []
        for sample_idx, payload in enumerate(inputs["score_payloads"]):
            if not payload.get("valid", False):
                continue
            seq_len = (
                int(payload["sequence_ids"].size(0))
                if payload.get("sequence_ids") is not None
                else int(payload["base_inputs_embeds"].size(0))
            )
            valid_entries.append((sample_idx, payload, seq_len))

        if not valid_entries:
            return out

        num_valid = len(valid_entries)
        max_seq_len = max(seq_len for _, _, seq_len in valid_entries)
        batched_inputs_embeds = torch.zeros(
            num_valid,
            max_seq_len,
            llada_model.config.d_model,
            dtype=dtype,
            device=completion_mask.device,
        )
        batched_attention_mask = torch.zeros(num_valid, max_seq_len, dtype=torch.long, device=completion_mask.device)
        batched_masked_indices = torch.zeros(num_valid, max_seq_len, dtype=torch.bool, device=completion_mask.device)
        batched_modality_indices = torch.zeros(num_valid, max_seq_len, dtype=torch.bool, device=completion_mask.device)
        batched_new_token_mask = torch.zeros(num_valid, max_seq_len, dtype=torch.bool, device=completion_mask.device)
        batched_text_targets = torch.full(
            (num_valid, max_seq_len),
            -100,
            dtype=torch.long,
            device=completion_mask.device,
        )
        batched_sequence_ids = torch.zeros(num_valid, max_seq_len, dtype=torch.long, device=completion_mask.device)
        text_rows = []
        row_to_sample_idx = []

        for row_idx, (sample_idx, payload, seq_len) in enumerate(valid_entries):
            row_to_sample_idx.append(sample_idx)
            batched_attention_mask[row_idx, :seq_len] = 1
            batched_masked_indices[row_idx, :seq_len] = payload["masked_indices"][seed_position]
            batched_modality_indices[row_idx, :seq_len] = payload["modality_indices"]
            batched_new_token_mask[row_idx, :seq_len] = payload["new_token_mask"]
            batched_text_targets[row_idx, :seq_len] = payload["text_target_ids"]
            if payload.get("sequence_ids") is not None:
                batched_sequence_ids[row_idx, :seq_len] = payload["sequence_ids"]
                text_rows.append(row_idx)
            else:
                batched_inputs_embeds[row_idx, :seq_len] = payload["base_inputs_embeds"].to(
                    device=completion_mask.device, dtype=dtype
                )

        if text_rows:
            text_rows_tensor = torch.tensor(text_rows, dtype=torch.long, device=completion_mask.device)
            text_base_embeds = multimodal_model.transformer.wte(batched_sequence_ids[text_rows_tensor])
            batched_inputs_embeds[text_rows_tensor] = text_base_embeds.to(dtype=dtype)

        batched_inputs_embeds = torch.where(
            batched_masked_indices.unsqueeze(-1),
            noise_embedding.to(dtype=dtype).view(1, 1, -1),
            batched_inputs_embeds,
        )

        image_groups: dict[tuple[int, int], list[tuple[int, dict[str, Any], int]]] = {}
        for row_idx, (sample_idx, payload, _) in enumerate(valid_entries):
            if payload["gen_latents_masked"] is None:
                continue
            gen_shape = tuple(payload["gen_shape"])
            image_groups.setdefault(gen_shape, []).append((row_idx, payload, sample_idx))

        for gen_shape, group_entries in image_groups.items():
            group_latents = []
            for _, payload, _ in group_entries:
                group_latents.append(payload["gen_latents_masked"][seed_position])
            group_latents_tensor = torch.stack(group_latents, dim=0)
            group_embeds = multimodal_model.call_gen_embedding(group_latents_tensor, gen_shape=gen_shape)
            group_embeds = pad_along_last_dim(group_embeds, llada_model.config.d_model)
            for local_idx, (row_idx, payload, _) in enumerate(group_entries):
                row_mask = batched_new_token_mask[row_idx]
                row_embed = group_embeds[local_idx].reshape(-1, llada_model.config.d_model)
                batched_inputs_embeds[row_idx] = batched_inputs_embeds[row_idx].masked_scatter(
                    row_mask.unsqueeze(-1),
                    row_embed,
                )

        # One trainable forward for the whole batch avoids ZeRO duplicate gradient reduction.
        output = LLaDAModelLM.forward(
            llada_model,
            input_ids=None,
            inputs_embeds=batched_inputs_embeds,
            attention_mask=batched_attention_mask,
            position_ids=None,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            prompt_len=None,
            num_items_in_batch=None,
            modality_indices=batched_modality_indices,
        )

        for row_idx, (sample_idx, payload, seq_len) in enumerate(valid_entries):
            text_axis = payload["completion_axis_text"]
            if text_axis.numel() > 0:
                target_mask = batched_text_targets[row_idx, :seq_len].ne(-100)
                row_text_targets = batched_text_targets[row_idx, :seq_len][target_mask]
                if row_text_targets.numel() > 0:
                    row_text_logps = -F.cross_entropy(
                        output.logits[row_idx, :seq_len][target_mask].float(),
                        row_text_targets,
                        reduction="none",
                    )
                    out[sample_idx, text_axis] = row_text_logps.to(torch.float32)

        hidden_states = output.hidden_states[-1]
        for gen_shape, group_entries in image_groups.items():
            group_targets = []
            group_latents = []
            group_hidden_states = []
            sample_indices = []
            for row_idx, payload, sample_idx in group_entries:
                row_mask = batched_new_token_mask[row_idx]
                group_hidden_states.append(hidden_states[row_idx][row_mask])
                group_targets.append(payload["gen_targets"])
                group_latents.append(payload["gen_latents_masked"][seed_position])
                sample_indices.append(sample_idx)

            group_hidden_states_tensor = torch.stack(group_hidden_states, dim=0)
            group_hidden_states_tensor = maybe_truncate_last_dim(
                group_hidden_states_tensor,
                llada_model.config.d_model_gen,
            )
            flat_group_hidden_states = group_hidden_states_tensor.reshape(-1, group_hidden_states_tensor.size(-1))
            group_targets_tensor = torch.stack(group_targets, dim=0)
            group_latents_tensor = torch.stack(group_latents, dim=0)
            timesteps = group_latents_tensor.eq(8193).float().sum(-1) / group_latents_tensor.shape[-1]
            gen_logits = multimodal_model.call_gen_predictor(
                flat_group_hidden_states,
                gen_shape,
                timesteps=timesteps,
            )

            if group_targets_tensor.dim() == 3:
                batch_dim, depth_dim, seq_dim = group_targets_tensor.shape
                gen_logits = gen_logits.view(batch_dim, seq_dim, *gen_logits.shape[-2:]).permute(0, 2, 1, 3)
                image_logps = -F.cross_entropy(
                    gen_logits.reshape(-1, gen_logits.shape[-1]).float(),
                    group_targets_tensor.reshape(-1),
                    reduction="none",
                ).view(batch_dim, depth_dim * seq_dim)
            else:
                image_logps = -F.cross_entropy(
                    gen_logits.float(),
                    group_targets_tensor.reshape(-1),
                    reduction="none",
                ).view(group_targets_tensor.size(0), -1)

            for local_idx, (_, payload, sample_idx) in enumerate(group_entries):
                image_axis = payload["completion_axis_image"]
                out[sample_idx, image_axis] = image_logps[local_idx, : image_axis.numel()].to(torch.float32)

        return out

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
        if torch.is_tensor(mask_seeds):
            this_itr_seed = int(mask_seeds[this_itr_idx].item())
        else:
            this_itr_seed = int(mask_seeds[this_itr_idx])

        per_token_logps = self._compute_unified_per_token_logps_batched_grad(model, inputs, this_itr_seed)
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

        completion_tokens_per_sample = [None] * len(inputs)
        completion_masks_per_sample = [None] * len(inputs)
        text_prompt_ids = [None] * len(inputs)
        text_completion_ids = [None] * len(inputs)
        image_contexts = [None] * len(inputs)
        completions_text = [""] * len(inputs)
        prompts_text = [""] * len(inputs)

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            image_edit_indices = [idx for idx, mode in enumerate(sample_modes) if mode == "image_edit"]
            image_edit_batch_size = max(1, int(getattr(self.args, "image_edit_batch_size", 1)))
            for start_idx in trange(0, len(image_edit_indices), image_edit_batch_size, desc="Image Rollout"):
                batch_indices = image_edit_indices[start_idx : start_idx + image_edit_batch_size]
                batch_examples = [inputs[idx] for idx in batch_indices]
                batch_latents, batch_contexts = self._rollout_image_edit_latents(unwrapped_model, batch_examples)
                for local_idx, sample_idx in enumerate(batch_indices):
                    completion_flat = batch_latents[local_idx].flatten()
                    completion_tokens_per_sample[sample_idx] = completion_flat
                    completion_masks_per_sample[sample_idx] = torch.ones_like(
                        completion_flat, dtype=torch.int, device=completion_flat.device
                    )
                    image_contexts[sample_idx] = batch_contexts[local_idx]
                    completions_text[sample_idx] = "<image_latents>"
                    prompts_text[sample_idx] = self._build_llada_prompt(inputs[sample_idx].get("prompt", []))

            for idx, example in tqdm(enumerate(inputs), desc=f"Text Rollout"):
                mode = sample_modes[idx]
                if mode == "image_edit":
                    continue
                else:
                    prompt_text = self._build_llada_prompt(example["prompt"])
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

                    generated = unwrapped_model.generate(
                        inputs=prompt_ids,
                        attention_mask=prompt_mask,
                        pad_token_id=self.processing_class.pad_token_id,
                        use_cache=True,
                        **generation_kwargs,
                    )
                    if hasattr(generated, "sequences"):
                        generated = generated.sequences
                    if isinstance(generated, tuple):
                        generated = generated[0]

                    norm_prompt_ids, completion_ids = self._normalize_text_rollout(
                        generated, prompt_ids, prefix_lm
                    )
                    completion_mask = self._build_text_completion_mask(completion_ids)

                    completion_tokens_per_sample[idx] = completion_ids.squeeze(0)
                    completion_masks_per_sample[idx] = completion_mask.squeeze(0).to(device)
                    text_prompt_ids[idx] = norm_prompt_ids.squeeze(0)
                    text_completion_ids[idx] = completion_ids.squeeze(0)
                    completions_text[idx] = self.processing_class.decode(
                        completion_ids.squeeze(0), skip_special_tokens=True
                    )
                    prompts_text[idx] = prompt_text

        if any(x is None for x in completion_tokens_per_sample):
            raise ValueError("Missing completion tokens for one or more samples during rollout")
        max_completion_len = max(x.numel() for x in completion_tokens_per_sample) if completion_tokens_per_sample else 0
        completion_ids = torch.zeros(
            len(inputs),
            max_completion_len,
            dtype=torch.long,
            device=device,
        )
        completion_mask = torch.zeros(
            len(inputs),
            max_completion_len,
            dtype=torch.int,
            device=device,
        )
        for i, (ids, mask) in enumerate(zip(completion_tokens_per_sample, completion_masks_per_sample)):
            l = ids.numel()
            completion_ids[i, :l] = ids.to(device)
            completion_mask[i, :l] = mask.to(device)

        num_iterations = int(getattr(self.args, "num_iterations", 1))
        beta = float(getattr(self.args, "beta", 0.0))
        if self.args.random_masking:
            mask_seeds = torch.randint(0, 2**12, (num_iterations,), device=device)
        else:
            mask_seeds = torch.tensor([42] * num_iterations, device=device)
        mask_seed_list = [int(x) for x in mask_seeds.tolist()]
        score_payloads = []
        for idx, mode in enumerate(sample_modes):
            valid_len = int(completion_mask[idx].sum().item())
            if mode == "image_edit":
                payload = self._build_image_score_payload(
                    image_contexts[idx],
                    completion_ids[idx],
                    valid_len,
                    mask_seed_list,
                    device,
                )
            else:
                payload = self._build_text_score_payload(
                    text_prompt_ids[idx],
                    text_completion_ids[idx],
                    completion_mask[idx],
                    mask_seed_list,
                )
            score_payloads.append(self._detach_structure(payload))

        scoring_inputs = {
            "sample_modes": sample_modes,
            "text_prompt_ids": text_prompt_ids,
            "text_completion_ids": text_completion_ids,
            "image_contexts": image_contexts,
            "score_payloads": score_payloads,
            "score_mask_seeds": mask_seed_list,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }

        all_old_per_token_logps = None
        all_ref_per_token_logps = None
        with torch.no_grad():
            if num_iterations > 1:
                all_old_per_token_logps = self._compute_unified_per_token_logps(
                    self.model, scoring_inputs, mask_seed_list
                )
            if beta != 0.0:
                if getattr(self, "ref_model", None) is not None:
                    all_ref_per_token_logps = self._compute_unified_per_token_logps(
                        self.ref_model, scoring_inputs, mask_seed_list
                    )
                else:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped, "disable_adapter"):
                        with unwrapped.disable_adapter():
                            all_ref_per_token_logps = self._compute_unified_per_token_logps(
                                self.model, scoring_inputs, mask_seed_list
                            )
                    else:
                        all_ref_per_token_logps = self._compute_unified_per_token_logps(
                            self.model, scoring_inputs, mask_seed_list
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
