# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class DiffuGRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation

        temperature (`float`, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `50`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str` or `None`, *optional*, defaults to `None`):
            Implementation of the cache method for faster generation when use_vllm is set to False.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training. This assumes that
            training has not already occupied all available GPUs. If only one device is available, the device will be
            shared between both training and vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        vllm_enable_prefix_caching (`bool`, *optional*, defaults to `True`):
            Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and the hardware
            support this feature.
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training
            speed, but may be numerically unstable for long training runs.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    cast_lm_head_to_fp32: bool = field(
        default=False,
        metadata={
            "help": "Whether to cast the LM head of the policy/reference models to float32."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    model_path: Optional[str] = field(
        default="",
        metadata={
            "help": "Model checkpoint path used by LaVida-O loading. Falls back to `ModelConfig.model_name_or_path` "
            "if unset."
        },
    )

    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    num_generations_eval: Optional[int] = field(
        default=None,
        metadata={"help": "Optional number of generations to sample during evaluation."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "Optional multiple used when padding prompts and completions."},
    )

    # Parameters that control generation
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={"help": "Number of optimization steps covered by one generation batch."},
    )
    temperature: float = field(
        default=0.1,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    use_fast_dlm: bool = field(
        default=False,
        metadata={"help": "Whether to use fast DLM generation. Forced to False for LaVida-O rollout parity."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "Extra generation keyword arguments passed through to TRL/transformers."},
    )
    chat_template_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to `apply_chat_template` during generation."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    use_transformers_paged: bool = field(
        default=False,
        metadata={"help": "Whether to use transformers paged generation when not using vLLM."},
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Implementation of the cache method for faster generation when use_vllm is set to False."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={"help": "TRL vLLM integration mode: `server` or `colocate`."},
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={"help": "Model implementation for vLLM: `vllm` or `transformers`."},
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Enable vLLM sleep mode to offload weights/cache during optimizer steps."},
    )
    vllm_server_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the vLLM server when using server mode."},
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host for the vLLM server when `vllm_server_base_url` is unset."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port for the vLLM server when `vllm_server_base_url` is unset."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Timeout in seconds while waiting for a vLLM server."},
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={"help": "Port used for the vLLM weight update group."},
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for colocated vLLM execution."},
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    vllm_max_model_length: Optional[int] = field(
        default=None,
        metadata={"help": "Alias for `vllm_max_model_len` used by newer TRL versions."},
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and "
            "the hardware support this feature."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )
    vllm_structured_outputs_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM structured outputs in newer TRL versions."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: Optional[float] = field(
        default=None,
        metadata={"help": "Optional upper clipping bound for two-sided GRPO."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={"help": "Optional upper clipping epsilon; defaults to `epsilon` when unset."},
    )
    sapo_temperature_neg: float = field(
        default=1.05,
        metadata={"help": "Temperature for tokens with non-positive advantages in SAPO."},
    )
    sapo_temperature_pos: float = field(
        default=1.0,
        metadata={"help": "Temperature for tokens with positive advantages in SAPO."},
    )
    vespo_k_pos: float = field(
        default=2.0,
        metadata={"help": "VESPO exponent for positive advantages."},
    )
    vespo_lambda_pos: float = field(
        default=3.0,
        metadata={"help": "VESPO decay for positive advantages."},
    )
    vespo_k_neg: float = field(
        default=3.0,
        metadata={"help": "VESPO exponent for negative advantages."},
    )
    vespo_lambda_neg: float = field(
        default=2.0,
        metadata={"help": "VESPO decay for negative advantages."},
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={"help": "Importance sampling granularity: `token` or `sequence`."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={"help": "Whether to normalize rewards by their standard deviation."},
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        metadata={"help": "Aggregation strategy for multiple reward functions."},
    )
    loss_type: str = field(
        default="bnpo",
        metadata={"help": "TRL GRPO loss variant."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Exclude truncated completions from the loss when enabled."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={"help": "Keep only the top entropy quantile of tokens in the policy loss."},
    )
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use Liger fused GRPO loss."},
    )
    max_tool_calling_iterations: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tool-calling turns during generation."},
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={"help": "Whether to correct vLLM logprob mismatch with importance sampling."},
    )
    vllm_importance_sampling_mode: str = field(
        default="sequence_mask",
        metadata={"help": "Importance sampling mode used for vLLM logprob correction."},
    )
    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={"help": "Cap used by vLLM importance sampling correction."},
    )
    off_policy_mask_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Optional threshold for off-policy sequence masking."},
    )
    use_bias_correction_kl: bool = field(
        default=False,
        metadata={"help": "Whether to use bias-corrected KL with importance sampling."},
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log the completions during training."},
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of logged completions."},
    )
    wandb_log_unique_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to log only unique prompts to Weights & Biases."},
    )
    log_unique_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to log only unique prompts."},
    )
    log_completions_hub_repo: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hub repo used to store logged completions."},
    )

    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size for generation. If not set, the batch size will be equal to the number of generations."
        },
    )

    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "diffusion block length"},
    )
    diffusion_steps: Optional[int] = field(
        default=64,
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
        metadata={
            "help": "Training dataset name. Supported values include gsm8k, countdown, sudoku, math, code, thinkmorph_edit, and mixed_placeholder."
        },
    )
    p_mask_prompt: float = field(
        default=0.3,
        metadata={"help": "Probability of masking the prompt."},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )
    random_masking: bool = field(
        default=True,
        metadata={"help": "Whether to randomly mask tokens."},
    )
    # Image editing rollout defaults (LaVida-O eval_img baseline)
    image_edit_sample_policy: str = field(default="multinomial")
    image_edit_confidence_policy: str = field(default="halton")
    image_edit_guidance_scale: float = field(default=0.0)
    image_edit_batch_size: int = field(default=1)
    image_edit_resolution: int = field(default=1024)
    image_edit_n_tokens: int = field(default=4096)
    image_edit_shift: int = field(default=5)
    image_edit_n_steps: int = field(default=64)
    image_edit_schedule: str = field(default="shift")
    image_edit_alg_temp: float = field(default=5.0)
    image_edit_dynamic_temperature: bool = field(default=True)
    image_edit_schedule_temp: str = field(default="cosine2")
    image_edit_min_temperature: float = field(default=0.5)
    image_edit_micro_cond: str = field(default="")
    image_edit_schedule_temp_samp: str = field(default="linear")
    image_edit_dynamic_temperature_samp: bool = field(default=False)
    image_edit_min_temperature_samp: float = field(default=1.0)
    image_edit_cfg_interval_start: float = field(default=0.0)
    image_edit_cfg_interval_end: float = field(default=1.0)
    image_edit_guidance_scale_image: float = field(default=5.0)
    image_edit_edit_mode: int = field(default=0)
    image_edit_order_cutoff: float = field(default=1.0)

    # Text rollout normalization.
    text_rollout_force_prefix_lm: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Override prefix_lm during rollout. If None, uses prefix_lm field.",
        },
    )
    text_rollout_step_per_block: Optional[int] = field(
        default=None,
        metadata={"help": "Optional step_per_block override for text rollout."},
    )
    text_rollout_do_sample: bool = field(default=False)

    # LaVida model/data arguments.
    version: str = field(default="llada")
    load_vlm: bool = field(default=True)
    prefix_lm: bool = field(default=True)
    unified_gen: bool = field(default=True)
    dual_tower: bool = field(default=False)
    vqvae: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default="siglip-so400m-patch14-384")
    mm_tunable_parts: Optional[str] = field(
        default="mm_vision_tower,mm_mlp_adapter,mm_vision_resampler,mm_language_model,mm_language_model_vision_parms"
    )
    peft_task_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Legacy alias for TRL ModelConfig.lora_task_type (e.g. CAUSAL_LM).",
        },
    )
    mm_vision_tower_lr: Optional[float] = field(default=2e-6)
    mm_projector_lr: Optional[float] = field(default=None)
    dual_tower_layers: Optional[int] = field(default=16)
    enc_use_image_branch: Optional[bool] = field(default=True)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: str = field(default="spatial_unpad")
    mm_resampler_type: Optional[str] = field(default="spatial_pool")
    mm_spatial_pool_mode: str = field(default="conv")
    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_out_channels: Optional[int] = field(default=1152)
    image_aspect_ratio: str = field(default="anyres")
    image_grid_pinpoints: Optional[str] = field(
        default="[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]"
    )
    image_gen_size: int = field(default=1024)
    num_gen_image_tokens: int = field(default=1024)
    group_by_modality_length: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
        if self.steps_per_generation is None or self.steps_per_generation < 1:
            raise ValueError("`steps_per_generation` must be a positive integer.")

        if self.vllm_max_model_length is not None and self.vllm_max_model_len is None:
            self.vllm_max_model_len = self.vllm_max_model_length
        elif self.vllm_max_model_len is not None and self.vllm_max_model_length is None:
            self.vllm_max_model_length = self.vllm_max_model_len

        if self.vllm_structured_outputs_regex is not None and self.vllm_guided_decoding_regex is None:
            self.vllm_guided_decoding_regex = self.vllm_structured_outputs_regex
        elif self.vllm_guided_decoding_regex is not None and self.vllm_structured_outputs_regex is None:
            self.vllm_structured_outputs_regex = self.vllm_guided_decoding_regex

        if self.log_unique_prompts != self.wandb_log_unique_prompts:
            if self.log_unique_prompts:
                self.wandb_log_unique_prompts = True
            else:
                self.log_unique_prompts = self.wandb_log_unique_prompts
