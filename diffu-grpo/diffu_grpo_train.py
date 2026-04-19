import ast
import inspect
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer, Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from llava.model.builder import load_pretrained_model
from trl import ModelConfig, TrlParser

# Required by LaVida-O sampling paths.
# IMPORTANT: these must be set BEFORE any llava imports, because llava_llada.py
# reads them at module-load time into module-level constants.
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")
os.environ.setdefault("SKIP_COMPLEMENTARY_MASKING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
LAVIDA_ROOT = REPO_ROOT / "LaVida-O"
if str(LAVIDA_ROOT) not in sys.path:
    sys.path.insert(0, str(LAVIDA_ROOT))

from llava import conversation as conversation_lib
from llava.model.language_model.llava_llada import LlavaLladaForMaskedDiffusion

# Custom imports
import numpy as np
from datasets import concatenate_datasets

from data_utils import (
    get_arxivqa_interleave_questions,
    get_code_questions,
    get_countdown_questions,
    get_image_answer_placeholder_questions,
    get_gsm8k_questions,
    get_image_edit_placeholder_questions,
    get_math_questions,
    get_mixed_placeholder_questions,
    get_sudoku_questions,
    get_thinkmorph_interleave_questions,
    set_random_seed,
)
from diffu_grpo_config import DiffuGRPOConfig
from diffu_grpo_trainer import DiffuGRPOTrainer
from mmada_grpo_trainer import MMaDAGRPOTrainer
from reward_func import (
    boxed_and_answer_tags_format_reward,
    coding_reward_func,
    correctness_reward_func,
    correctness_reward_func_math,
    countdown_reward_func,
    int_reward_func,
    perceptual_score_reward_func,
    strict_format_reward_func,
    sudoku_reward_func,
    xmlcount_reward_func,
)


def _resolve_model_type(args: DiffuGRPOConfig, model_config: ModelConfig) -> str:
    """Return 'lavida' or 'mmada' based on args.model_type or auto-detection from model path."""
    mt = args.model_type.lower()
    if mt not in ("lavida", "mmada", "auto"):
        raise ValueError(f"model_type must be 'lavida', 'mmada', or 'auto', got {mt!r}")
    if mt != "auto":
        return mt
    # Auto-detect from model path
    path = (model_config.model_name_or_path or args.model_path or "").lower()
    if "mmada" in path:
        return "mmada"
    return "lavida"


def _resolve_model_name_or_path(args: DiffuGRPOConfig, model_config: ModelConfig) -> str:
    return model_config.model_name_or_path or args.model_path


def _maybe_literal_eval(value: Optional[str]):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _build_overwrite_config(args: DiffuGRPOConfig, model_name_or_path: str) -> AutoConfig:
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    overwrite = {
        "enc_use_image_branch": args.enc_use_image_branch,
        "dual_tower_layers": args.dual_tower_layers,
        "mm_vision_select_layer": args.mm_vision_select_layer,
        "mm_use_im_start_end": args.mm_use_im_start_end,
        "mm_use_im_patch_token": args.mm_use_im_patch_token,
        "mm_patch_merge_type": args.mm_patch_merge_type,
        "mm_resampler_type": args.mm_resampler_type,
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_spatial_pool_out_channels": args.mm_spatial_pool_out_channels,
        "image_aspect_ratio": args.image_aspect_ratio,
        "image_grid_pinpoints": _maybe_literal_eval(args.image_grid_pinpoints),
        "unified_gen": args.unified_gen,
        "num_gen_image_tokens": args.num_gen_image_tokens,
        "image_gen_size": args.image_gen_size,
        "mm_tunable_parts": args.mm_tunable_parts,
        "mm_vision_tower_lr": args.mm_vision_tower_lr,
        "mm_projector_lr": args.mm_projector_lr,
    }
    for key, value in overwrite.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def init_lavida_model_and_tokenizer(args: DiffuGRPOConfig, model_config: ModelConfig):
    model_name_or_path = _resolve_model_name_or_path(args, model_config)
    if not model_name_or_path:
        raise ValueError("model_name_or_path or model_path is required for LaVida-O loading.")
    lower = model_name_or_path.lower()
    if "llada" not in lower and "lavida" not in lower:
        raise ValueError(
            f"Expected llada/lavida checkpoint path, got: {model_name_or_path}"
        )

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=args.mm_resampler_type,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True,
        mm_patch_merge_type=args.mm_patch_merge_type,
    )
    resize_embeddings = True
    tokenizer, model, _, _ = load_pretrained_model(
            model_name_or_path,
            None,
            "llava_llada",
            attn_implementation="sdpa",
            device_map="cpu",
            vision_kwargs=vision_kwargs,
            resize_embeddings=resize_embeddings,
            torch_dtype=('bfloat16' if args.bf16 else None),
        )
    model.config.use_cache = False
    tokenizer.padding_side = "left"

    if args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llada"]

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_tunable_parts = args.mm_tunable_parts
    model.config.mm_vision_tower_lr = args.mm_vision_tower_lr
    model.config.mm_projector_lr = args.mm_projector_lr
    return model, tokenizer


def init_mmada_model_and_tokenizer(args: DiffuGRPOConfig, model_config: ModelConfig):
    """Initialize MMaDA model and tokenizer."""
    MMADA_ROOT = REPO_ROOT / "MMaDA-Parallel-M"
    if str(MMADA_ROOT) not in sys.path:
        sys.path.insert(0, str(MMADA_ROOT))

    from models.modeling_mmada import MMadaModelLM
    from training.prompting_utils import UniversalPrompting

    model_name_or_path = _resolve_model_name_or_path(args, model_config)
    if not model_name_or_path:
        raise ValueError("model_name_or_path or model_path is required for MMaDA loading.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=int(getattr(args, "max_prompt_length", 256)),
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=True,
    )

    model = MMadaModelLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    model.config.use_cache = False

    # MMaDA's mask token id is fixed (see infer_all.py: MASK_TOKEN_ID=126336).
    # We stash it on args / model.config for downstream use.
    mask_id = 126336
    args.mask_id = mask_id
    model.config.mask_id = mask_id

    tokenizer.padding_side = "left"
    return model, tokenizer, uni_prompting


def _build_mmada_optimizer(model, args: DiffuGRPOConfig):
    """Build optimizer for MMaDA with no-decay groups."""
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


def _set_trainable_parameters(model, mm_tunable_parts: Optional[str]):
    if not mm_tunable_parts:
        model.requires_grad_(True)
        return

    model.requires_grad_(False)
    base = model.get_model() if hasattr(model, "get_model") else model
    tunable_parts = [part.strip() for part in mm_tunable_parts.split(",") if part.strip()]

    if "mm_mlp_adapter" in tunable_parts and hasattr(base, "mm_projector"):
        base.mm_projector.requires_grad_(True)
    if "mm_vision_resampler" in tunable_parts and hasattr(base, "vision_resampler"):
        base.vision_resampler.requires_grad_(True)

    for name, param in model.named_parameters():
        if "mm_vision_tower" in tunable_parts and "vision_tower" in name:
            param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                param.requires_grad_(True)
        if "mm_language_model_vision_parms" in tunable_parts:
            if "_vision" in name or "gen_embedding" in name or "gen_predictor" in name or "_gen" in name:
                param.requires_grad_(True)
        if "mm_updown_layers" in tunable_parts:
            if (
                "upsample_gen" in name
                or "downsample_gen" in name
                or "gen_predictor_2" in name
                or "gen_embedding_2" in name
                or "downsample_gen_enc" in name
            ):
                param.requires_grad_(True)
        if "mm_gen_input_output_layers" in tunable_parts:
            if (
                "upsample_gen" in name
                or "downsample_gen" in name
                or "gen_predictor" in name
                or "gen_embedding" in name
                or "downsample_gen_enc" in name
            ):
                param.requires_grad_(True)
        if "extra_gen_dit" in tunable_parts and "extra_gen_dit" in name:
            param.requires_grad_(True)
        if "gen_predictor" in tunable_parts and "gen_predictor" in name:
            param.requires_grad_(True)

        if "vqvae" in name or "vq_model" in name or "vqvae_model" in name:
            param.requires_grad_(False)


def _log_trainable_ratio(model):
    total_params = sum(
        p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.ds_numel if hasattr(p, "ds_numel") else p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    ratio = (trainable_params / total_params) if total_params else 0.0
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio (trainable/total): {ratio:.6f}")


def _build_optimizer(model, args: DiffuGRPOConfig):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    lr_mapper = {}
    if args.mm_projector_lr is not None:
        lr_mapper["mm_projector"] = args.mm_projector_lr
    if args.mm_vision_tower_lr is not None:
        lr_mapper["vision_tower"] = args.mm_vision_tower_lr

    if lr_mapper:
        special_lr_parameters = [
            name
            for name, _ in model.named_parameters()
            if any(module_keyword in name for module_keyword in lr_mapper)
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_parameters and n not in special_lr_parameters and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_parameters and n not in special_lr_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        for module_keyword, lr in lr_mapper.items():
            module_parameters = [
                name for name, _ in model.named_parameters() if module_keyword in name
            ]
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n in decay_parameters and n in module_parameters and p.requires_grad
                        ],
                        "weight_decay": args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in decay_parameters and n in module_parameters and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
            )
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


def neutral_reward_func(prompts, completions, **kwargs):
    return [0.0 for _ in completions]


REWARD_REQUIRED_COLUMNS = {
    "boxed_and_answer_tags_format_reward": {"answer"},
    "coding_reward_func": {"answer"},
    "correctness_reward_func": {"answer_gt"},
    "correctness_reward_func_math": {"answer"},
    "countdown_reward_func": {"target", "numbers"},
    "correct_grounding_reward_func": {"ground_gt"},
    "perceptual_score_reward_func": {"image_gt"},
    "sudoku_reward_func": {"puzzle", "solution"},
}

REWARD_RESERVED_ARGS = {"self", "prompts", "completions", "step", "run_name", "rank"}

def _get_reward_required_columns(reward_func) -> set[str]:
    if isinstance(reward_func, torch.nn.Module):
        return set()

    reward_name = reward_func.__name__
    required = set(REWARD_REQUIRED_COLUMNS.get(reward_name, set()))
    signature = inspect.signature(reward_func)
    for name, parameter in signature.parameters.items():
        if name in REWARD_RESERVED_ARGS:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect.Parameter.empty:
            required.add(name)
    return required


def _validate_reward_dataset_compatibility(dataset_name: str, dataset, reward_functions) -> None:
    if len(dataset) == 0:
        raise ValueError(f"Dataset '{dataset_name}' is empty; cannot validate reward compatibility.")

    dataset_columns = set(dataset.column_names)
    mismatches = []
    for reward_func in reward_functions:
        required_columns = _get_reward_required_columns(reward_func)
        missing_columns = sorted(required_columns - dataset_columns)
        if missing_columns:
            reward_name = reward_func.__name__ if not isinstance(reward_func, torch.nn.Module) else type(reward_func).__name__
            mismatches.append(f"{reward_name}: missing columns {missing_columns}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"Dataset '{dataset_name}' is incompatible with the configured reward functions. "
            f"Available columns: {sorted(dataset_columns)}. {mismatch_text}"
        )


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    # ``modality`` controls whether train_dataset drives the gen-side
    # (image-edit rollout) or the und-side (text rollout). Default "gen"
    # matches every legacy dataset path. ``thinkmorph_answer`` is text-only
    # and flips this to "und" so the image rollout is bypassed entirely.
    modality = "gen"

    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func,
            xmlcount_reward_func,
        ]
    elif grpo_config.dataset == "code":
        dataset = get_code_questions()
        reward_functions = [xmlcount_reward_func, coding_reward_func]
    elif grpo_config.dataset == "thinkmorph_edit":
        # Gen-only flow: train_dataset carries the image-edit rows and
        # drives the image rollout. und_dataset is None so the trainer
        # automatically skips the text rollout / text loss.
        dataset, ground_dataset = get_image_edit_placeholder_questions(
            region_edit=bool(getattr(grpo_config, "region_edit", False)),
        )
        reward_functions = [perceptual_score_reward_func]
        modality = "gen"
    elif grpo_config.dataset == "thinkmorph_answer":
        # Und-only flow: train_dataset carries the text-QA rows and drives
        # the text rollout. modality="und" tells the trainer to treat
        # train_dataset as und-side inputs and bypass the image rollout
        # entirely. und_dataset MUST stay None.
        dataset = get_image_answer_placeholder_questions()
        reward_functions = [
            strict_format_reward_func,
            correctness_reward_func,
        ]
        modality = "und"
    elif grpo_config.dataset == "thinkmorph_interleave":
        region_edit = bool(getattr(grpo_config, "region_edit", False))
        tm_gen, tm_und, tm_ground = get_thinkmorph_interleave_questions(region_edit=region_edit)
        ax_gen, ax_und, ax_ground = get_arxivqa_interleave_questions(region_edit=region_edit)

        # Concatenate gen and und sides. Each loader emits its gen and und rows
        # in the same order, so within each sub-dataset row i refers to the same
        # source sample on both sides; concatenation preserves this index alignment.
        dataset = concatenate_datasets([tm_gen, ax_gen])
        und_dataset = concatenate_datasets([tm_und, ax_und])
        if len(dataset) != len(und_dataset):
            raise ValueError(
                f"thinkmorph_interleave gen/und length mismatch after concat: "
                f"{len(dataset)} vs {len(und_dataset)}"
            )
        ground_dataset = None
        if region_edit:
            ground_dataset = concatenate_datasets([tm_ground, ax_ground])
            if len(dataset) != len(ground_dataset):
                raise ValueError(
                    f"thinkmorph_interleave gen/ground length mismatch after concat: "
                    f"{len(dataset)} vs {len(ground_dataset)}"
                )

        # Joint shuffle: draw one permutation and apply it to both datasets so
        # that row i continues to refer to the same sample_id on both sides.
        # The trainer enforces this alignment with a runtime sample_id assert.
        rng = np.random.default_rng(grpo_config.seed)
        perm = rng.permutation(len(dataset)).tolist()
        dataset = dataset.select(perm)
        und_dataset = und_dataset.select(perm)
        if ground_dataset is not None:
            ground_dataset = ground_dataset.select(perm)

        reward_functions = [
            perceptual_score_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ]
    else:
        raise ValueError(f"Unsupported dataset: {grpo_config.dataset}")

    # _validate_reward_dataset_compatibility(grpo_config.dataset, dataset, reward_functions)

    # thinkmorph_interleave already applied a joint permutation above; re-shuffling
    # here independently would break the gen/und sample_id alignment.
    if grpo_config.dataset != "thinkmorph_interleave":
        dataset = dataset.shuffle(seed=grpo_config.seed)
    und_train_set = None
    if "und_dataset" in locals():
        if grpo_config.dataset != "thinkmorph_interleave":
            und_dataset = und_dataset.shuffle(seed=grpo_config.seed)
        und_train_set = und_dataset
    ground_train_set = None
    if "ground_dataset" in locals() and ground_dataset is not None:
        # Ground is paired to gen by sample_id, so the trainer resolves it via
        # a {sample_id -> row} map (no second dataloader). Shuffling here is
        # safe because lookup is by key, not by index.
        ground_train_set = ground_dataset
    if grpo_config.dataset in ["countdown", "sudoku"] and len(dataset) > 500:
        train_set = dataset.select(range(0, len(dataset) - 500))
    else:
        train_set = dataset

    resolved_model_type = _resolve_model_type(grpo_config, model_config)

    if resolved_model_type == "mmada":
        model, tokenizer, uni_prompting = init_mmada_model_and_tokenizer(grpo_config, model_config)
        _log_trainable_ratio(model)
        optimizer = _build_mmada_optimizer(model, grpo_config)

        trainer = MMaDAGRPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            train_dataset_und=und_train_set,
            train_dataset_ground=ground_train_set,
            optimizers=(optimizer, None),
            modality=modality,
        )
    else:
        model, tokenizer = init_lavida_model_and_tokenizer(grpo_config, model_config)
        _set_trainable_parameters(model, grpo_config.mm_tunable_parts)
        _log_trainable_ratio(model)
        optimizer = _build_optimizer(model, grpo_config)

        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            train_dataset_und=und_train_set,
            train_dataset_ground=ground_train_set,
            optimizers=(optimizer, None),
            modality=modality,
        )

    if grpo_config.save_steps % grpo_config.num_iterations != 0:
        warnings.warn(
            f"save_steps ({grpo_config.save_steps}) is not divisible by num_iterations ({grpo_config.num_iterations}). "
            f"If resuming training from a checkpoint, use a checkpoint aligned with num_iterations={grpo_config.num_iterations}."
        )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    if grpo_config.peft_task_type:
        if getattr(model_config, "lora_task_type", None) in (None, ""):
            model_config.lora_task_type = grpo_config.peft_task_type
        elif model_config.lora_task_type != grpo_config.peft_task_type:
            warnings.warn(
                "Both peft_task_type and lora_task_type are set and differ. "
                f"Using lora_task_type={model_config.lora_task_type!r}."
            )
    main(grpo_config=grpo_config, model_config=model_config)
