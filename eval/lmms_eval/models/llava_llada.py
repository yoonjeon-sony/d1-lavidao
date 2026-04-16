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
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import os
import PIL
from PIL import Image
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
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llada.generate import (
        generate as llada_generate,
        get_logits as llada_get_logits,
        get_num_transfer_tokens_sch,
        wte as llada_wte,
        cosine_schedule_2,
        exp_schedule,
        logit_normal_schedule,
    )
    from llava.model.utils import pad_along_last_dim
    from llava.mm_utils import pad_to_square_and_resize
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


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
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

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
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained, trust_remote_code=True)

        llava_model_args["overwrite_config"] = overwrite_config
        # try:
            # Try to load the model with the multimodal argument
            
        if os.path.exists('/data1/jacklishufan/siglip-so400m-patch14-384'):
            vision_tower_path = "/data1/jacklishufan/siglip-so400m-patch14-384"
        else:
            vision_tower_path="/data0/jacklishufan/siglip-so400m-patch14-384"
        print(vision_tower_path)
        vision_tower_path = "google/siglip-so400m-patch14-384"
        # vision_kwargs = dict(
        #     mm_vision_tower=os.environ.get('LLADA_VISION_ENCODER',vision_tower_path),
        #     mm_resampler_type=None,
        #     mm_projector_type=os.environ.get('LLADA_VISION_PROJECTOR','mlp2x_gelu'),
        #     mm_hidden_size=int(os.environ.get('LLADA_VISION_ENCODER_HIDDEN_SIZE',1152)),
        #     mm_pooler_ratio=int(os.environ.get('LLADA_MM_POOLER_RATIO',2)),
        #     use_mm_proj=True,
        #     mm_patch_merge_type='spatial_unpad',            
        # )
        vision_kwargs = None
        resize_embeddings = True # default behavior
        if DEBUG_LOAD_TRAINER:
            resize_embeddings = False
            
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args,vision_kwargs=vision_kwargs,resize_embeddings=resize_embeddings)
        # breakpoint()
        assert self._tokenizer is not None
        # except TypeError:
        #     # for older versions of LLaVA that don't have multimodal argument
        #     llava_model_args.pop("multimodal", None)
        #     self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self._config = self._model.config
        # Force left-padding at inference. prepare_inputs_labels_for_multimodal re-pads
        # per-sample sequences using config.tokenizer_padding_side after inserting image
        # embeddings; with right-padding, gen tokens for short-prefix samples end up
        # separated from their valid prefix by zero padding, which shifts RoPE relative
        # positions and makes batched (bs>1) decoding diverge from bs=1.
        self._config.tokenizer_padding_side = "left"
        self.model.eval()
        self.model.model.set_activation_checkpointing(None)
        self.model.requires_grad_(False)
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # Stores generated image paths keyed by (task, doc_id) for log output.
        self._image_resps = {}

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            # if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
            #     self._model = accelerator.prepare(self.model)
            # else:
            #     self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
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
        #self.model.model.transformer = accelerate.cpu_offload(self.model.model.transformer)

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
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                    task_type = "video"

                # elif type(visual[0]) == PIL.Image.Image:
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts


            if "llama_3" in self.conv_template or 'llada' in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            # conv.messages[-1][1] = continuation
            # full_prompt = conv.get_prompt()
            # full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            # labels = full_input_ids.clone()
            # labels[0, : input_ids.shape[1]] = -100
            input_prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            answers = continuation
            answer_ids = self.tokenizer(continuation)['input_ids']
            answer_ids = torch.tensor(continuation).to(input_ids.device).unsqueeze(0) 


            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            torch.cuda.empty_cache()
            # with torch.inference_mode():
                #outputs = self.model(input_ids=full_input_ids, labels=labels, images=image_tensor, use_cache=True, **kwargs)
            likelyhoods = self.model.log_likelyhood_inference(
                input_ids,
                images=image_tensor.to(torch.bfloat16),
                image_sizes=None,
                verbose=True,
                answer=answer_ids,
                mc_num=self.mc_num,
            ) 

            # loss = outputs["loss"]
            # logits = outputs["logits"]
            # greedy_tokens = logits.argmax(dim=-1)
            # cont_toks = full_input_ids[:, input_ids.shape[1] :]
            # greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            # max_equal = (greedy_tokens == cont_toks).all()
            # lmms eval return loss
            res.append((float(-likelyhoods.item()), False))
            pbar.update(1)

        pbar.close()
        return res

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

    # ------------------------------------------------------------------
    # Image rollout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_edit_gumbel_noise(tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.zeros_like(tensor, dtype=torch.float32).uniform_(0, 1)
        return -torch.log(-torch.log(noise))

    @staticmethod
    def _build_image_edit_temperature_schedule(
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

    @staticmethod
    def _left_pad_2d(
        tensors: List[torch.Tensor],
        pad_value: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        max_len = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                pad_t = torch.full((1, pad_len), pad_value, dtype=dtype, device=t.device)
                t = torch.cat([pad_t, t], dim=1)
            padded.append(t)
        return torch.cat(padded, dim=0)

    def _get_default_image_edit_config(self, gen_kwargs: dict) -> dict:
        """Build image-edit generation config from gen_kwargs with defaults."""
        resolution = int(gen_kwargs.get("image_edit_resolution", 512))
        latent_token_map = {256: 256, 512: 1024, 1024: 4096}
        prompt_token_map = {256: 64, 512: 256, 1024: 1024}
        n_tokens = latent_token_map.get(resolution, 1024)
        prompt_n_tokens = prompt_token_map.get(resolution, max(1, n_tokens // 4))
        return {
            "image_resolution": resolution,
            "n_tokens": n_tokens,
            "prompt_n_tokens": prompt_n_tokens,
            "n_steps": int(gen_kwargs.get("image_edit_n_steps", 64)),
            "guidance_scale": float(gen_kwargs.get("image_edit_guidance_scale", 0.0)),
            "guidance_scale_image": float(gen_kwargs.get("image_edit_guidance_scale_image", 0.0)),
            "schedule": gen_kwargs.get("image_edit_schedule", "shift"),
            "shift": int(gen_kwargs.get("image_edit_shift", 3)),
            "alg_temp": float(gen_kwargs.get("image_edit_alg_temp", 0.1)),
            "dynamic_temperature": bool(gen_kwargs.get("image_edit_dynamic_temperature", True)),
            "dynamic_temperature_samp": bool(gen_kwargs.get("image_edit_dynamic_temperature_samp", False)),
            "sample_policy": gen_kwargs.get("image_edit_sample_policy", "argmax"),
            "confidence_policy": gen_kwargs.get("image_edit_confidence_policy", "mmada"),
            "temperature": float(gen_kwargs.get("image_edit_temperature", 0.8)),
            "schedule_temp": gen_kwargs.get("image_edit_schedule_temp", "shift"),
            "min_temperature": float(gen_kwargs.get("image_edit_min_temperature", 0.0)),
            "schedule_temp_samp": gen_kwargs.get("image_edit_schedule_temp_samp", "shift"),
            "min_temperature_samp": float(gen_kwargs.get("image_edit_min_temperature_samp", 0.0)),
            "cfg_interval": [
                float(gen_kwargs.get("image_edit_cfg_interval_start", 0.0)),
                float(gen_kwargs.get("image_edit_cfg_interval_end", 1.0)),
            ],
            "edit_mode": int(gen_kwargs.get("image_edit_edit_mode", 1)),
            "order_cutoff": float(gen_kwargs.get("image_edit_order_cutoff", 0.5)),
            "remask_ratio": float(gen_kwargs.get("image_edit_remask_ratio", 0.01)),
        }

    def _rollout_image_edit(
        self,
        visuals: List[PIL.Image.Image],
        instructions: List[str],
        gen_kwargs: dict,
    ) -> List[PIL.Image.Image]:
        """Run image-edit diffusion rollout and return generated PIL images.

        Mirrors the core logic of ``_rollout_image_edit_latents`` from the
        GRPO trainer but is self-contained for eval.
        """
        model = self.model
        device = self.device
        gen_cfg = self._get_default_image_edit_config(gen_kwargs)
        batch_size = len(visuals)

        reserve_id = 126089
        reserve_id2 = 126090
        img_mask_id = 8193
        reserve_token = "<|reserved_token_5|>"
        reserve_token_2 = "<|reserved_token_6|>"

        base_model = model.get_model() if hasattr(model, "get_model") else model

        gen_shape_map = {1024: (64, 64), 512: (32, 32), 256: (16, 16)}
        gen_shape = gen_shape_map.get(gen_cfg["image_resolution"], (32, 32))
        is_unitok = "unitok" in getattr(base_model.config, "mm_vqvae", "")
        latent_shape = (8, gen_cfg["n_tokens"]) if is_unitok else (gen_cfg["n_tokens"],)

        # Bail out if model lacks image-gen capabilities
        if (
            not hasattr(model, "encode_image_gen")
            or not hasattr(base_model, "call_gen_embedding")
            or not hasattr(base_model, "image_processor_gen")
        ):
            eval_logger.warning("Model lacks image generation capabilities; skipping image rollout.")
            return [None] * batch_size

        conv_version = self.conv_template
        if conv_version not in conv_templates:
            conv_version = "llada"

        all_input_ids = []
        all_edit_images = []
        image_sizes = []
        all_enc_embeddings = []
        vq_latents = []

        for batch_idx in range(batch_size):
            edit_image = visuals[batch_idx]
            instruction = instructions[batch_idx]
            image_sizes.append(edit_image.size)
            all_edit_images.append(edit_image)

            image_resized = pad_to_square_and_resize(edit_image.convert("RGB"), gen_cfg["image_resolution"])
            vq_latent = base_model.image_processor_gen.preprocess(image_resized).to(device, dtype=model.dtype)
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
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            all_input_ids.append(input_ids)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        all_input_ids = self._left_pad_2d(all_input_ids, pad_token_id, torch.long)
        attention_mask = (all_input_ids != pad_token_id).long()
        image_tensor = process_images(all_edit_images, self._image_processor, self._config)
        if type(image_tensor) is list:
            image_tensor = [t.to(dtype=torch.bfloat16, device=device) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)

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

        mask_id = getattr(self._config, "mask_id", 126336)
        noise_embed = base_model.transformer.wte(torch.tensor([mask_id], device=device))
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
            raise ValueError(f"Unsupported edit_mode: {edit_mode}")
        is_gen_enc_null = torch.zeros_like(is_gen_enc, dtype=torch.bool)

        n_tokens = gen_cfg["n_tokens"]
        n_steps_total = int(gen_cfg["n_steps"])

        # Initialize xt with mask tokens, optionally seeded from init_latents
        if is_unitok:
            xt = torch.full((batch_size, 8, n_tokens), img_mask_id, dtype=torch.long, device=device)
        else:
            xt = torch.full((batch_size, n_tokens), img_mask_id, dtype=torch.long, device=device)

        remask_ratio = gen_cfg.get("remask_ratio", 0.01)
        n_mask_remask = max(int(n_tokens * remask_ratio), 1)
        indices = np.arange(n_tokens)
        np.random.shuffle(indices)
        init_mask_indices = indices[:n_mask_remask]
        if is_unitok:
            xt[:, :, init_mask_indices] = init_latents[:, :, init_mask_indices]
        else:
            xt[:, init_mask_indices] = init_latents[:, init_mask_indices]

        mask_idx_sched = xt == img_mask_id
        if is_unitok:
            mask_idx_sched = mask_idx_sched[:, 0, :]
        n_mask_per_sample = mask_idx_sched.sum(dim=1)
        max_n_mask = int(n_mask_per_sample.max().item())
        max_step = max(1, int(n_steps_total * max_n_mask / n_tokens))
        n_steps_per_sample = (
            (n_mask_per_sample.float() * n_steps_total / float(n_tokens))
            .to(torch.int64)
            .clamp(min=0, max=max_step)
        )
        has_mask = n_mask_per_sample > 0
        n_steps_per_sample = torch.where(
            has_mask & (n_steps_per_sample == 0),
            torch.ones_like(n_steps_per_sample),
            n_steps_per_sample,
        )

        schedule_name = gen_cfg["schedule"]
        num_transfer_tokens = torch.zeros(batch_size, max_step, dtype=torch.int64, device=device)
        for b in range(batch_size):
            n_s = int(n_steps_per_sample[b].item())
            n_m = int(n_mask_per_sample[b].item())
            if n_s == 0 or n_m == 0:
                continue
            ntt = get_num_transfer_tokens_sch(
                mask_idx_sched[b:b+1], n_s,
                schedule=schedule_name,
                schedule_kwargs={"shift": gen_cfg["shift"]},
            )
            ntt_len = min(int(ntt.shape[1]), max_step)
            num_transfer_tokens[b, :ntt_len] = ntt[0, :ntt_len]

        confidence_policy = gen_cfg["confidence_policy"]

        # Temperature schedules
        sch_temperatures = torch.zeros(batch_size, max_step, device=device, dtype=torch.float32)
        sch_temperatures_samp = torch.zeros(batch_size, max_step, device=device, dtype=torch.float32)
        for b in range(batch_size):
            n_s = int(n_steps_per_sample[b].item())
            if n_s == 0:
                continue
            t_b = self._build_image_edit_temperature_schedule(
                n_s, gen_cfg["schedule_temp"], gen_cfg["min_temperature"], gen_cfg["shift"],
            ).to(device=device)
            t_b_samp = self._build_image_edit_temperature_schedule(
                n_s, gen_cfg["schedule_temp_samp"], gen_cfg["min_temperature_samp"], gen_cfg["shift"],
            ).to(device=device)
            sch_temperatures[b, :n_s] = t_b
            sch_temperatures_samp[b, :n_s] = t_b_samp

        cfg_start = int(gen_cfg["cfg_interval"][0] * max_step)
        cfg_end = int(gen_cfg["cfg_interval"][1] * max_step)

        x0 = xt.clone()
        active_steps = torch.nonzero(num_transfer_tokens.sum(dim=0) > 0, as_tuple=False).squeeze(-1)
        for step_idx, step_col in enumerate(active_steps, start=1):
            col_idx = min(step_idx - 1, max_step - 1)
            local_temp = sch_temperatures[:, col_idx]
            local_temp_samp = sch_temperatures_samp[:, col_idx]
            step_confidence_policy = confidence_policy
            if step_idx / max(max_step, 1) > gen_cfg["order_cutoff"]:
                step_confidence_policy = "mmada"

            mask_idx = xt == img_mask_id
            if is_unitok:
                mask_idx = mask_idx[:, 0, :]
            n_mask_per_sample = mask_idx.sum(dim=1)
            if n_mask_per_sample.max().item() == 0:
                break

            timesteps = n_mask_per_sample.float() / mask_idx.shape[1]
            do_cfg = gen_cfg["guidance_scale"] > 0 and (cfg_start <= step_idx <= cfg_end)

            if do_cfg:
                embed_input = torch.cat([inputs_embeds_uncond, inputs_embeds_uncond_enc, inputs_embeds], dim=0)
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
                base_model, None, True,
                x_gen=xt_input, gen_shape=gen_shape,
                inputs_embeds_curr=embed_input.clone(),
                new_token_mask=new_token_mask,
            )
            logits = llada_get_logits(
                base_model, all_input_embeddings, new_token_mask, True,
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
                    raise ValueError(f"Unsupported edit_mode: {edit_mode}")
                logits = (1 + gen_cfg["guidance_scale"]) * logits_cond - gen_cfg["guidance_scale"] * logits_un
                logits[logits_is_ninf] = -np.inf

            safe_logits = torch.nan_to_num(logits.to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9)
            probs = safe_logits.softmax(-1)
            if gen_cfg["sample_policy"] == "argmax":
                x0 = safe_logits.argmax(-1)
            else:
                base_temperature = float(gen_cfg.get("temperature", 0.8))
                if gen_cfg["dynamic_temperature_samp"]:
                    temperature_vec = base_temperature * local_temp_samp
                    temperature_vec = torch.clamp(temperature_vec, min=1e-6)
                    logits_for_sample = safe_logits / temperature_vec.view(-1, 1, 1)
                    x0 = torch.distributions.Categorical(logits=logits_for_sample).sample()
                else:
                    if base_temperature <= 0:
                        x0 = safe_logits.argmax(-1)
                    else:
                        x0 = torch.distributions.Categorical(logits=safe_logits / base_temperature).sample()
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            if is_unitok:
                x0 = x0.permute(0, 2, 1)
                x0 = torch.where(mask_idx.unsqueeze(1).repeat(1, 8, 1), x0, xt)
                x0_p = x0_p.permute(0, 2, 1).max(dim=1)[0]
            else:
                x0 = torch.where(mask_idx, x0, xt)

            for b_idx in range(batch_size):
                b_mask = mask_idx[b_idx]
                b_n_mask = int(b_mask.sum().item())
                if b_n_mask == 0:
                    continue
                k = min(int(num_transfer_tokens[b_idx, step_col].item()), b_n_mask)
                if k <= 0:
                    continue
                local_temp_b = float(local_temp[b_idx].item())
                if step_confidence_policy == "mask_git":
                    alg_t = gen_cfg["alg_temp"] * local_temp_b if gen_cfg["dynamic_temperature"] else gen_cfg["alg_temp"]
                    confidence = x0_p[b_idx] / max(alg_t, 1e-6)
                    confidence = torch.where(b_mask, confidence, -np.inf)
                    confidence = torch.softmax(confidence, dim=-1)
                    select_index = torch.multinomial(confidence, num_samples=k)
                else:
                    alg_t = gen_cfg["alg_temp"] * local_temp_b if gen_cfg["dynamic_temperature"] else gen_cfg["alg_temp"]
                    confidence = torch.log(x0_p[b_idx].clamp(1e-20)) + alg_t * self._image_edit_gumbel_noise(x0_p[b_idx])
                    confidence = torch.where(b_mask, confidence, -np.inf)
                    _, select_index = torch.topk(confidence, k=k)

                if is_unitok:
                    transfer_index[b_idx, :, select_index] = True
                else:
                    transfer_index[b_idx, select_index] = True
            xt[transfer_index] = x0[transfer_index]

        xt[xt == img_mask_id] = x0[xt == img_mask_id]

        # Decode VQ latents to images
        decoded_images = model.decode_image_gen(
            xt, gen_cfg["image_resolution"], gen_cfg["image_resolution"],
        )
        result = []
        for batch_idx, decoded_image in enumerate(decoded_images):
            if torch.is_tensor(decoded_image):
                image_array = decoded_image.detach().cpu()
                if image_array.ndim == 3 and image_array.shape[0] in (1, 3):
                    image_array = image_array.permute(1, 2, 0)
                image_array = image_array.clamp(0, 1).mul(255).to(torch.uint8).numpy()
                if image_array.ndim == 3 and image_array.shape[-1] == 1:
                    image_array = image_array[..., 0]
                result.append(Image.fromarray(image_array))
            elif isinstance(decoded_image, np.ndarray):
                image_array = np.clip(decoded_image, 0, 255).astype(np.uint8)
                if image_array.ndim == 3 and image_array.shape[-1] == 1:
                    image_array = image_array[..., 0]
                result.append(Image.fromarray(image_array))
            else:
                result.append(decoded_image)
        return result

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            # breakpoint()
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
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
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            # assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []

            do_image_rollout = gen_kwargs.get("do_image_rollout", False)

            # Accumulate per-sample visual data so batched generation has one entry per
            # <image> token across the entire batch. Previously image_tensor was overwritten
            # each iteration, which only worked when batch_size == 1.
            # A single sample can itself carry multiple images; each contributes one entry
            # (possibly a multi-patch tensor) to flat_images.
            flat_images = []          # list of per-image tensors, flattened across batch
            flat_image_sizes = []     # [(w, h), ...] aligned with flat_images (image task only)
            task_type = "text"

            # ------------------------------------------------------------------
            # Image rollout: generate edited images before text generation.
            # When do_image_rollout is True, we run _rollout_image_edit on
            # each sample's first image + context (as edit instruction), then
            # inject the generated image as a second <image> for text gen.
            # ------------------------------------------------------------------
            generated_images = [None] * len(batched_visuals)  # per-sample generated PIL images
            if do_image_rollout:
                rollout_visuals = []
                rollout_instructions = []
                rollout_indices = []  # which batch indices have images to roll out
                for sample_idx, (visual, context) in enumerate(zip(batched_visuals, batched_contexts)):
                    if visual is not None and len(visual) > 0 and isinstance(visual[0], PIL.Image.Image):
                        rollout_visuals.append(visual[0])
                        rollout_instructions.append(context)
                        rollout_indices.append(sample_idx)
                if rollout_visuals:
                    with torch.inference_mode():
                        gen_images = self._rollout_image_edit(
                            rollout_visuals, rollout_instructions, gen_kwargs,
                        )
                    for ri, gi in zip(rollout_indices, gen_images):
                        generated_images[ri] = gi

            for sample_idx, (visual, context) in enumerate(zip(batched_visuals, batched_contexts)):
                t0 = time.time()
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")
                if self.overwrite_image_aspect:
                    self._config.image_aspect_ratio = self.overwrite_image_aspect
                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        # If image rollout produced a generated image for this
                        # sample, append it to the visual list so that all
                        # images are processed together by process_images
                        # (guaranteeing consistent tensor dimensions).
                        gen_img = generated_images[sample_idx] if do_image_rollout else None
                        if gen_img is not None:
                            visual = list(visual) + [gen_img]
                            # With >1 image, force "pad" aspect ratio so all
                            # images produce tensors of the same shape (avoids
                            # anyres grid-shape mismatches in
                            # prepare_inputs_labels_for_multimodal).
                            if len(visual) > 1:
                                self._config.image_aspect_ratio = "pad"

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                # Flatten this sample's image_tensor into per-image entries so the batched
                # prepare_inputs_labels_for_multimodal sees exactly one entry per <image> token.
                # The number of <image> tokens this sample will emit equals placeholder_count.
                if image_tensor is None or (isinstance(image_tensor, list) and len(image_tensor) == 0):
                    sample_entries = []
                elif isinstance(image_tensor, list):
                    sample_entries = list(image_tensor)
                elif torch.is_tensor(image_tensor):
                    if image_tensor.ndim == 5:
                        sample_entries = [image_tensor[i] for i in range(image_tensor.shape[0])]
                    elif image_tensor.ndim == 4:
                        if task_type == "video" and placeholder_count == 1:
                            # All frames collapse into a single <image> token
                            sample_entries = [image_tensor]
                        else:
                            sample_entries = [image_tensor[i:i+1] for i in range(image_tensor.shape[0])]
                    else:
                        sample_entries = [image_tensor]
                else:
                    sample_entries = [image_tensor]
                flat_images.extend(sample_entries)

                if task_type == "image" and visual is not None:
                    flat_image_sizes.extend([v.size for v in visual])

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                # print(self.conv_template)
                if "llama_3" in self.conv_template or 'llada' in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    # breakpoint()
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            schedule_kwargs = {}
            for key in list(gen_kwargs.keys()):
                if key.startswith('schedule__'):
                    value = gen_kwargs.pop(key)
                    schedule_kwargs[key.replace('schedule__','')] = value
            if len(schedule_kwargs) > 0:
                gen_kwargs['schedule_kwargs'] = schedule_kwargs
            
            if 'block_length' not in gen_kwargs:
                gen_kwargs['block_length'] = min(128,gen_kwargs["max_new_tokens"])
            if 'step_per_block' not in gen_kwargs and 'step_ratio' not in gen_kwargs:
                gen_kwargs['step_per_block'] = gen_kwargs['block_length']

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            # prepare_inputs_labels_for_multimodal zips embeds with modalities, so the
            # modalities list must have one entry per batch sample (not per <image> token),
            # otherwise zip silently truncates the batch.
            if task_type == "image":
                gen_kwargs["image_sizes"] = flat_image_sizes
                gen_kwargs["modalities"] = ["image"] * len(batched_contexts)
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"] * len(batched_contexts)
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            # breakpoint()
            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            # Strip image-edit rollout keys — they are not valid model.generate args.
            gen_kwargs.pop("do_image_rollout", None)
            for key in list(gen_kwargs.keys()):
                if key.startswith("image_edit_"):
                    gen_kwargs.pop(key)
            # Pass the batch-flattened image list (None if no visuals at all) so that
            # len(images) == total <image> tokens across the batch.
            images_arg = flat_images if len(flat_images) > 0 else None
            # image_sizes is a named param on model.generate(), not **kwargs,
            # so extract it to pass explicitly.
            image_sizes_arg = gen_kwargs.pop("image_sizes", None)
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=images_arg, image_sizes=image_sizes_arg, use_cache=self.use_cache, **gen_kwargs)
                    # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=images_arg, use_cache=self.use_cache, **gen_kwargs)
                if gen_kwargs.get('use_fast_dlm'):
                    # generate_with_dual_cache returns (x, nfe); also it hardcodes bsz=1
                    # internally, so batched inference is not supported on this path.
                    assert cont[0].shape[0] == len(batched_contexts), (
                        f"use_fast_dlm is hardcoded to bsz=1 but got batch={len(batched_contexts)}. "
                        f"Either disable use_fast_dlm or patch generate_with_dual_cache to use bsz."
                    )
                    cont = cont[0]
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
                assert len(text_outputs) == len(batched_contexts), (
                    f"Decoded {len(text_outputs)} outputs for batch of {len(batched_contexts)}. "
                    f"cont shape={tuple(cont.shape) if torch.is_tensor(cont) else type(cont)}."
                )
            except Exception as e:
                raise e
            # with open('/data1/jacklishufan/lmms-eval.pt', 'wb') as f:
            #     torch.save(self.model.state_dict(), f)
            #     print('saved')
            #     print(1/0)
            t1 = time.time()
            delta_t += t1-t0
            num_generated +=1
            print(f"Avg Latency (of {num_generated}): {delta_t/num_generated}")
            if DEBUG_PRINT_OUTPUT:
                print(f'\n--------Start of Sample {batched_doc_id[0]}---------')
                print("Question: ",prompt_question)
                print("Answer: ",text_outputs)
                print("Answer: ",gen_kwargs)
                print('--------End---------')

            text_outputs = [response.strip() for response in text_outputs]

            # Save generated images to disk and record paths for logging.
            if do_image_rollout:
                rollout_save_dir = Path(os.environ.get(
                    "IMAGE_ROLLOUT_SAVE_DIR", "/tmp/image_rollout_outputs"
                ))
                rollout_save_dir.mkdir(parents=True, exist_ok=True)
                for s_idx, (doc_id, gen_img) in enumerate(zip(batched_doc_id, generated_images)):
                    if gen_img is not None and hasattr(gen_img, "save"):
                        safe_id = str(doc_id).replace("/", "_").replace(os.sep, "_").replace(" ", "_")
                        img_path = rollout_save_dir / f"{task}_{safe_id}_{os.getpid()}.png"
                        gen_img.save(img_path)
                        self._image_resps[(task, doc_id)] = str(img_path)

            res.extend(text_outputs)
            # Cache per-sample: the previous single add_partial used the last sample's
            # context for the whole batch output, which is incorrect when batch_size > 1.
            for _ctx, _out in zip(batched_contexts, text_outputs):
                self.cache_hook.add_partial("generate_until", (_ctx, gen_kwargs), _out)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []
        raise NotImplementedError()

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_to_text, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # multi round inference: terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                question_input = []

                if round_idx != 0:  # get current round visual and context from doc_to_text function
                    batched_visuals, batched_contexts, batched_terminal_singal, batched_round_res, batched_previous_round_info = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    # import ipdb; ipdb.set_trace()
                    batched_round_res = list(zip(*batched_round_res))  # [(r1_1, r1_2), (r2_1, r2_2), ...]
                    if batched_terminal_singal[0]:  # terminal signal from doc_to_text function
                        break

                for visual, context in zip(batched_visuals, batched_contexts):
                    if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                        self._config.image_aspect_ratio = origin_image_aspect_ratio
                        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                    if visual is None or visual == []:  # for text-only tasks.
                        visual = None
                        task_type = "text"
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                            self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                            eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                        if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                            assert type(visual) == list, "sample_frames must be specified for video task"
                            sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata["sample_frames"]

                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                            task_type = "video"
                            placeholder_count = 1

                        elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                            task_type = "image"
                            placeholder_count = len(visual) if isinstance(visual, list) else 1

                        elif type(visual[0]) == str:  # For video task
                            image_tensor = []
                            try:
                                if self.video_decode_backend == "decord":
                                    frames = self.load_video(visual, self.max_frames_num)
                                elif self.video_decode_backend == "pyav":
                                    frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                                image_tensor.append(frames)
                            except Exception as e:
                                eval_logger.error(f"Error {e} in loading video")
                                image_tensor = None

                            task_type = "video"
                            placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                        4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                        """
                        # if task_type == "image": # indeed in multi-image case, not the video in frames.
                        #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        # elif task_type == "video":
                        # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                    else:
                        conv = conv_templates[self.conv_template].copy()

                    if utils.is_json(question):  # conversational question input
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item["value"]
                            conv.append_message(role, message)

                        assert len(conv.messages) % 2 == 1
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:  # only simple string for question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # preconfigure gen_kwargs with defaults
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                if task_type == "image":
                    gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                elif task_type == "video":
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    gen_kwargs["modalities"] = ["video"]
                    gen_kwargs["stopping_criteria"] = [stopping_criteria]
                    self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                    self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: attention to this major generation step...
                if "image_aspect_ratio" in gen_kwargs.keys():
                    gen_kwargs.pop("image_aspect_ratio")
                try:
                    with torch.inference_mode():
                        cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e

                text_outputs = [response.strip() for response in text_outputs]
                batched_round_res.append(text_outputs)

                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial("generate_until_multi_round", (context, gen_kwargs), batched_round_res)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
