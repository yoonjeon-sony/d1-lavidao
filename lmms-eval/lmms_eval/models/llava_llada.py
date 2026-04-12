# ADOBE CONFIDENTIAL
# Copyright 2025 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains
# the property of Adobe and its suppliers, if any. The intellectual
# and technical concepts contained herein are proprietary to Adobe
# and its suppliers and are protected by all applicable intellectual
# property laws, including trade secret and copyright laws.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Adobe.

import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import os
import PIL
import torch
import accelerate
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
import time
# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)

# Import LLaVA modules
DEBUG_LOAD_TRAINER = os.environ.get('DEBUG_LOAD_TRAINER',False)
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    pad_to_square_and_resize,
)
from llava.model.builder import load_pretrained_model
from llava.model.utils import pad_along_last_dim
from llava.eval.rollout import (
    build_image_edit_gen_cfg,
    run_image_rollout,
    run_text_rollout,
)

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_llada")
class Llava_Llada(lmms):
    """
    Llava Model
    """
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "llava_llada",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        mc_num=16,
        chat_mode: Optional[str] = None,
        use_bbox: Optional[bool] = True,
        img_gen_guidance_scale: float = 1.2,
        img_gen_guidance_scale_image: float = 1.4,
        img_gen_conf_policy: str = "stratified",
        img_gen_edit_mode: int = 1,
        img_gen_n_steps: int = 64,
        img_gen_temperature: float = 0.8,
        img_gen_enable_stratified: bool = False,
        img_gen_resolution: int = 512,
        gen_img_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.gen_img_dir = gen_img_dir
        # Validate and store chat_mode
        VALID_CHAT_MODES = (None, "text_gen", "image_gen")
        if chat_mode not in VALID_CHAT_MODES:
            raise ValueError(f"Invalid chat_mode={chat_mode!r}. Must be one of {VALID_CHAT_MODES}")
        self.chat_mode = "text_gen" if chat_mode is None else chat_mode
        self.use_bbox = use_bbox
        # Store image generation parameters with explicit type casts
        self.img_gen_guidance_scale = float(img_gen_guidance_scale)
        self.img_gen_guidance_scale_image = float(img_gen_guidance_scale_image)
        self.img_gen_conf_policy = str(img_gen_conf_policy)
        self.img_gen_edit_mode = int(img_gen_edit_mode)
        self.img_gen_n_steps = int(img_gen_n_steps)
        self.img_gen_temperature = float(img_gen_temperature)
        self.img_gen_enable_stratified = bool(img_gen_enable_stratified)
        self.img_gen_resolution = int(img_gen_resolution)
        self.datetime_str = None  # set by evaluator after model creation

        if kwargs:
            eval_logger.warning(f"Unexpected kwargs (ignored): {kwargs}")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        self.mc_num = mc_num
        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = 'llava_llada'# if model_name is not None else get_model_name_from_path(pretrained)
        self.overwrite_image_aspect = os.environ.get("LLAVA_OVERWRITE_IMAGE_ASPECT", None)
        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        llava_model_args["overwrite_config"] = overwrite_config

        vision_kwargs = dict(
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_resampler_type=None,
            mm_projector_type='mlp2x_gelu',
            mm_hidden_size=1152,
            use_mm_proj=True
        )

        resize_embeddings = True # default behavior
            
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args,vision_kwargs=vision_kwargs,resize_embeddings=resize_embeddings)

        assert self._tokenizer is not None

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.model.model.set_activation_checkpointing(None)
        self.model.requires_grad_(False)
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # Image generation modes now support batched inference via text_to_image_batch

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            self.model.to(self._device).to(torch.bfloat16)
            self._model.model.transformer = accelerator.prepare(self.model.model.transformer)
        
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device).to(torch.bfloat16)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        pass

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def _pad_image_for_gen(self, pil_image):
        """Pad image to square and resize to configured generation resolution."""
        return self.model.pad_image(pil_image, image_resolution=self.img_gen_resolution)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            # breakpoint()
            return -len(toks), x[0]

        metadata = requests[0].metadata
        if DEBUG_PRINT_OUTPUT:
            # do not sort by length, instead using lambda x:x[-3]
            re_ords = utils.Collator([reg.args for reg in requests], lambda x:x[-3], grouping=True)
        else:
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)
        if DEBUG_LOAD_TRAINER:
            ckpt1 = torch.load(DEBUG_LOAD_TRAINER, map_location='cpu')
            ckpt1 = {k.replace('module.model','model'):v for k,v in ckpt1.items()}
            _res = self.model.load_state_dict(ckpt1,strict=False)
            print(f"DEBUG_LOAD_TRAINER:{DEBUG_LOAD_TRAINER} {_res}")
            print("Something is broken if above line does not show all keys matched!!!")
            del ckpt1
        delta_t = 0
        num_generated = 0

        # Resolve vision-tower image processor once up front (needed by both
        # rollout paths). Mirrors Diffu_Grpo_Trainer._get_image_processor with
        # a final fallback to the processor load_pretrained_model produced.
        # Must be non-None: if it is, run_image_rollout silently short-circuits
        # to invalid contexts (see rollout.py:390-396).
        vt = self.model.get_vision_tower() if hasattr(self.model, "get_vision_tower") else None
        image_processor = getattr(vt, "image_processor", None)
        if image_processor is None and hasattr(self.model, "get_model"):
            base = self.model.get_model()
            if hasattr(base, "get_vision_tower"):
                bvt = base.get_vision_tower()
                if bvt is not None and hasattr(bvt, "image_processor"):
                    image_processor = bvt.image_processor
        if image_processor is None:
            image_processor = self._image_processor
        if image_processor is None:
            raise RuntimeError(
                "Llava_Llada.generate_until: could not resolve an image_processor from "
                "the vision tower or load_pretrained_model. Both rollout paths require one."
            )
        device = self.model.get_model().device

        # Build image-gen config once — shared across all batches in this
        # generate_until call. Overrides come from the img_gen_* kwargs that
        # run_lmms-eval.sh / model_args already expose on self.
        image_gen_cfg = build_image_edit_gen_cfg(
            image_resolution=self.img_gen_resolution,
            n_steps=self.img_gen_n_steps,
            guidance_scale=self.img_gen_guidance_scale,
            guidance_scale_image=self.img_gen_guidance_scale_image,
            confidence_policy=self.img_gen_conf_policy,
            edit_mode=self.img_gen_edit_mode,
        )

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            batch_size = len(batched_contexts)
            batch_pil_images = [
                doc_to_visual(self.task_dict[task_name][split_name][doc_id])  # List[PIL.Image.Image]
                for doc_to_visual, task_name, split_name, doc_id in zip(
                    batched_doc_to_visual, batched_task, batched_split, batched_doc_id
                )
            ]  # List[List[PIL.Image.Image]]

            # Drop any image-gen-only hint that may be on gen_kwargs — not used
            # by the rollout path, kept for backwards-compat with prior scripts.
            gen_kwargs.pop("image_gen_post_prompt", None)

            t0 = time.time()
            needs_image_gen = self.chat_mode == "image_gen"

            # --- Normalize text-generation kwargs (unchanged contract with run_lmms-eval.sh) ---
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "block_length" not in gen_kwargs:
                gen_kwargs["block_length"] = min(128, gen_kwargs["max_new_tokens"])
            if "step_per_block" not in gen_kwargs and "step_ratio" not in gen_kwargs:
                gen_kwargs["step_per_block"] = gen_kwargs["block_length"]

            text_generation_kwargs = {
                "max_new_tokens": int(gen_kwargs["max_new_tokens"]),
                "block_length": int(gen_kwargs["block_length"]),
                "step_per_block": int(gen_kwargs["step_per_block"]),
                "temperature": float(gen_kwargs.get("temperature", 0.0)),
                "prefix_lm": bool(gen_kwargs.get("prefix_lm", True)),
                "remasking": gen_kwargs.get("remasking", "low_confidence"),
                "cfg_scale": float(gen_kwargs.get("cfg_scale", 0.0)),
            }

            # Build the per-sample "example" dicts that the trainer-style
            # rollout functions expect. The trainer rollout takes a single
            # primary image per sample (not a list); lmms-eval hands us a
            # bare question string + a separate images list, so we
            # reconstruct the same prompt format both rollout paths expect:
            #   - text rollout (run_text_rollout): ``example["prompt"]`` is the
            #     user-turn string wrapped in the conv template by
            #     _build_llada_prompt. It must contain DEFAULT_IMAGE_TOKEN so
            #     _build_llada_prompt's ``has_gen_image=True`` branch (used in
            #     image_gen mode) can duplicate it into two <image>\n slots.
            #     Mirrors the pre-refactor _build_text_prompt helper:
            #     ``"<image>\n {COT_PROMPT} {ctx}"``.
            #   - image rollout (run_image_rollout): ``example["instruction"]``
            #     is read by _extract_image_edit_instruction and appended into
            #     the edit-prompt template. Mirrors the pre-refactor
            #     _build_edit_prompt helper: ``f"{EDIT_PROMPT} {ctx}"``. The
            #     image rollout builds its own ``<image>`` slot internally, so
            #     the instruction must NOT carry DEFAULT_IMAGE_TOKEN.
            # ``example["image"]`` is the primary PIL image, consumed by both
            # paths: the edit path feeds it to the VQ-VAE via
            # image_processor_gen.preprocess, and the text path feeds it to the
            # vision tower via prepare_inputs_labels_for_multimodal.
            from data_utils import COT_PROMPT, EDIT_PROMPT

            examples = []
            for ctx, images, task_name, split_name, doc_id in zip(
                batched_contexts, batch_pil_images, batched_task, batched_split, batched_doc_id
            ):
                primary_image = images[0] if isinstance(images, (list, tuple)) and len(images) > 0 else images
                if primary_image is not None and DEFAULT_IMAGE_TOKEN not in (ctx or ""):
                    user_content = f"{DEFAULT_IMAGE_TOKEN}\n {COT_PROMPT} {ctx}"
                else:
                    user_content = f"{COT_PROMPT} {ctx}"
                edit_instruction = f"{EDIT_PROMPT} {ctx}"
                examples.append({
                    "prompt": user_content,        # wrapped in conv template by _build_llada_prompt
                    "image": primary_image,        # PIL.Image — consumed by both rollout paths
                    "instruction": edit_instruction,  # read by _extract_image_edit_instruction
                    "sample_id": f"{task_name}_{split_name}_{doc_id}",
                })

            # --- Image rollout (image_gen only) ---
            image_contexts = None
            if needs_image_gen:
                _, image_contexts = run_image_rollout(
                    self.model, examples,
                    tokenizer=self.tokenizer,
                    image_processor=image_processor,
                    device=device,
                    gen_cfg=image_gen_cfg,
                    conv_version=self.conv_template,
                    init_image=[ex.get("image") for ex in examples],
                )
                # Inject the decoded PIL image as gen_image on each example so
                # run_text_rollout's _build_llada_prompt duplicates <image>\n.
                # Mirrors diffu_grpo_trainer.py:2028-2038 (text_rollout_use_gen_image=True).
                for ex, ictx in zip(examples, image_contexts):
                    if ictx is None or ictx.get("decoded_image") is None:
                        raise ValueError(
                            "image_gen mode: image rollout produced no decoded_image for one of the samples."
                        )
                    ex["gen_image"] = ictx["decoded_image"]

            # --- Text rollout (always runs; uses gen_image if injected above) ---
            _, text_contexts = run_text_rollout(
                self.model, examples,
                tokenizer=self.tokenizer,
                image_processor=image_processor,
                device=device,
                generation_kwargs=text_generation_kwargs,
                image_edit_resolution=self.img_gen_resolution,
                conv_version=self.conv_template,
            )
            text_outputs = [tc["decoded_text"].lstrip("!").strip() for tc in text_contexts]

            t1 = time.time()
            delta_t += t1 - t0
            num_generated += batch_size
            print(f"Avg Latency (of {num_generated}): {delta_t/num_generated}")

            # --- Save generated images and build per-sample results ---
            if needs_image_gen:
                os.makedirs(self.gen_img_dir, exist_ok=True)
                for b_idx, (ictx, task_name, split_name, doc_id) in enumerate(
                    zip(image_contexts, batched_task, batched_split, batched_doc_id)
                ):
                    img_save_path = os.path.join(self.gen_img_dir, f"{task_name}_{doc_id}.png")
                    ictx["decoded_image"].save(img_save_path)
                    self.task_dict[task_name][split_name][doc_id]["gen_img_path"] = img_save_path
                    res.append({
                        "image_gen_input": examples[b_idx]["instruction"],
                        "text_gen_input": text_contexts[b_idx]["prompt"],
                        "text_gen_output": text_outputs[b_idx],
                        "image_gen_output_path": img_save_path,
                    })
            else:
                res.extend(text_outputs)

            for b_ctx, b_output in zip(batched_contexts, text_outputs):
                self.cache_hook.add_partial("generate_until", (b_ctx, gen_kwargs), b_output)
            pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LlavaLLaDA")
