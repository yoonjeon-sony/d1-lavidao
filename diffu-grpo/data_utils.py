from datasets import load_dataset, Dataset, Features, Value, concatenate_datasets
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from reward_func import extract_hash_answer
from tqdm import tqdm


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "task_type": "text",
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "task_type": "text",
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "task_type": "text",
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "task_type": "text",
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in <answer> </answer>. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore


def get_code_questions(split="train"):
    data = load_dataset("KodCode/KodCode-Light-RL-10K", split=split)
    data = data.train_test_split(test_size=0.1, seed=42)[
        "train"
    ]  # NOTE: 10% of the data was used for a different experiment
    data = data.map(
        lambda x: {
            "task_type": "text",
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a coding expert. You will be given a coding problem to solve. Solve it step by step. \n\n{x['question']}",
                }
            ],
            "answer": {"solution": x["solution"], "tests": x["test"]},
        }
    )
    return data


def get_image_edit_placeholder_questions() -> Dataset:
    data_root = DATA_ROOT
    image_root = DATA_ROOT
    train_data_paths = [os.path.join(data_root, name) for name in THINKMORPH_LOCAL_JSONL_FILES]

    missing_paths = [path for path in train_data_paths if not os.path.isfile(path)]
    if missing_paths:
        missing_str = ", ".join(missing_paths)
        raise FileNotFoundError(f"ThinkMorph jsonl file(s) not found: {missing_str}")

    rows: list[dict] = []
    for data_path in train_data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(
                tqdm(f, desc=f"Loading {os.path.basename(data_path)}", unit="line")
            ):
                if not line.strip():
                    continue

                example = json.loads(line)
                sample_id = str(example.get("pid", f"{os.path.basename(data_path)}:{idx}"))

                question = example.get("question")
                image_input_rel = example.get("problem_image_0")
                image_gt_rel = example.get("reasoning_image_0")
                answer = example.get("answer")

                if not isinstance(question, str) or not question.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid question: {question!r}"
                    )
                if not isinstance(image_input_rel, str) or not image_input_rel.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid problem_image_0: "
                        f"{image_input_rel!r}"
                    )
                if not isinstance(image_gt_rel, str) or not image_gt_rel.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid reasoning_image_0: "
                        f"{image_gt_rel!r}"
                    )

                instruction = f"{EDIT_PROMPT} {question.strip()}"
                image_input = os.path.join(image_root, image_input_rel)
                image_gt = os.path.join(image_root, image_gt_rel)

                rows.append(
                    {
                        "task_type": "image_edit",
                        "prompt": [
                            {
                                "role": "user",
                                "content": f"<image>\n{instruction}",
                            }
                        ],
                        "instruction": instruction,
                        "image": image_input,
                        "image_gen_enc": image_input,
                        "image_gen": image_gt,
                        "image_gt": image_gt,
                        "answer": answer,
                    }
                )

    return Dataset.from_list(rows)


def get_image_answer_placeholder_questions() -> Dataset:
    data_root = DATA_ROOT
    image_root = DATA_ROOT
    train_data_paths = [os.path.join(data_root, name) for name in THINKMORPH_LOCAL_JSONL_FILES]

    missing_paths = [path for path in train_data_paths if not os.path.isfile(path)]
    if missing_paths:
        missing_str = ", ".join(missing_paths)
        raise FileNotFoundError(f"ThinkMorph jsonl file(s) not found: {missing_str}")

    rows: list[dict] = []
    for data_path in train_data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(
                tqdm(f, desc=f"Loading {os.path.basename(data_path)}", unit="line")
            ):
                if not line.strip():
                    continue

                example = json.loads(line)
                sample_id = str(example.get("pid", f"{os.path.basename(data_path)}:{idx}"))

                question = example.get("question")
                image_input_rel = example.get("problem_image_0")
                answer = example.get("answer")

                if not isinstance(question, str) or not question.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid question: {question!r}"
                    )
                if not isinstance(image_input_rel, str) or not image_input_rel.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid problem_image_0: "
                        f"{image_input_rel!r}"
                    )
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid answer: {answer!r}"
                    )

                instruction = f"{COT_PROMPT} {question.strip()}"
                image_input = os.path.join(image_root, image_input_rel)

                rows.append(
                    {
                        "task_type": "text",
                        "prompt": [
                            {
                                "role": "user",
                                "content": f"<image>\n{instruction}",
                            }
                        ],
                        "instruction": instruction,
                        "image": image_input,
                        "answer_gt": answer.strip(),
                    }
                )

    return Dataset.from_list(rows)


# Shared schemas for the paired gen / und datasets used by `thinkmorph_interleave`.
# Both ThinkMorph and ArxivQA loaders emit rows with these exact features so the
# datasets can be concatenated and jointly shuffled. `image_gt` is nullable on the
# gen side: a non-null path means the row is eligible for perceptual reward, a
# null value (e.g. ArxivQA) means the gen rollout is rollout-only and must not
# contribute to gen-side reward / advantage / loss.
INTERLEAVE_GEN_FEATURES = Features(
    {
        "task_type": Value("string"),
        "dataset_type": Value("string"),
        "sample_id": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "instruction": Value("string"),
        "image": Value("string"),
        "image_gen_enc": Value("string"),
        "image_gt": Value("string"),  # nullable: None for answer_only rows
        "answer_gt": Value("string"),  # nullable: unused on gen side
    }
)
INTERLEAVE_UND_FEATURES = Features(
    {
        "task_type": Value("string"),
        "dataset_type": Value("string"),
        "sample_id": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "instruction": Value("string"),
        "image": Value("string"),
        "answer_gt": Value("string"),
    }
)


def get_thinkmorph_interleave_questions(split: str = "train") -> tuple[Dataset, Dataset]:
    """Build paired (gen_ds, und_ds) ThinkMorph datasets for `thinkmorph_interleave`.

    Each source jsonl sample contributes one row to gen_ds and one row to und_ds
    at the same index, sharing a `sample_id` so the trainer can assert alignment.
    Both rows are tagged ``dataset_type="image_answer"``.
    """
    if split != "train":
        raise ValueError(f"Unsupported split '{split}' for ThinkMorph interleave data. Use 'train'.")

    data_root = DATA_ROOT
    image_root = DATA_ROOT
    train_data_paths = [os.path.join(data_root, name) for name in THINKMORPH_LOCAL_JSONL_FILES]

    missing_paths = [path for path in train_data_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f"ThinkMorph jsonl file(s) not found: {', '.join(missing_paths)}")

    gen_rows: list[dict] = []
    und_rows: list[dict] = []
    for data_path in train_data_paths:
        basename = os.path.basename(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(
                tqdm(f, desc=f"Loading {basename}", unit="line")
            ):
                if not line.strip():
                    continue

                example = json.loads(line)
                sample_id = str(example.get("pid", f"{basename}:{idx}"))

                question = example.get("question")
                image_input_rel = example.get("problem_image_0")
                image_gt_rel = example.get("reasoning_image_0")
                answer = example.get("answer")

                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"ThinkMorph sample '{sample_id}' has invalid question: {question!r}")
                if not isinstance(image_input_rel, str) or not image_input_rel.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid problem_image_0: {image_input_rel!r}"
                    )
                if not isinstance(image_gt_rel, str) or not image_gt_rel.strip():
                    raise ValueError(
                        f"ThinkMorph sample '{sample_id}' has invalid reasoning_image_0: {image_gt_rel!r}"
                    )
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"ThinkMorph sample '{sample_id}' has invalid answer: {answer!r}")

                image_input = os.path.join(image_root, image_input_rel)
                image_gt = os.path.join(image_root, image_gt_rel)
                edit_instruction = f"{EDIT_PROMPT} {question.strip()}"
                cot_instruction = f"{COT_PROMPT} {question.strip()}"

                gen_rows.append(
                    {
                        "task_type": "image_edit",
                        "dataset_type": "image_answer",
                        "sample_id": sample_id,
                        "prompt": [{"role": "user", "content": f"<image>\n{edit_instruction}"}],
                        "instruction": edit_instruction,
                        "image": image_input,
                        "image_gen_enc": image_input,
                        "image_gt": image_gt,
                        "answer_gt": None,
                    }
                )
                und_rows.append(
                    {
                        "task_type": "text",
                        "dataset_type": "image_answer",
                        "sample_id": sample_id,
                        "prompt": [{"role": "user", "content": f"<image>\n{cot_instruction}"}],
                        "instruction": cot_instruction,
                        "image": image_input,
                        "answer_gt": answer.strip(),
                    }
                )

    gen_ds = Dataset.from_list(gen_rows, features=INTERLEAVE_GEN_FEATURES)
    und_ds = Dataset.from_list(und_rows, features=INTERLEAVE_UND_FEATURES)
    return gen_ds, und_ds


def get_mixed_placeholder_questions(split: str = "train") -> Dataset:
    """Placeholder mixed schema for future text+image integration tests."""
    rows = [
        {
            "task_type": "text",
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nWhat is 2 + 2?",
                }
            ],
            "answer": "4",
        },
        {
            "task_type": "image_edit",
            "prompt": [
                {
                    "role": "user",
                    "content": "<image>\nAdd a rainbow over the mountains.",
                }
            ],
            "instruction": "Add a rainbow over the mountains.",
            "image": None,
            "image_gen_enc": None,
            "image_gen": None,
            "answer": "",
        },
    ]
    return Dataset.from_list(rows)


THINKMORPH_LOCAL_JSONL_FILES = (
    "ThinkMorph-Spatial_Navigation_loc_val.jsonl",
    "ThinkMorph-Visual_Search_loc_val.jsonl",
    "ThinkMorph-Chart_Refocus_loc_val.jsonl",
    "ThinkMorph-Jigsaw_Assembly_loc_val.jsonl",
)
DATA_ROOT = "/scratch2/yoonjeon.kim/data/"



COT_PROMPT = (
        "Let's think step-by-step to solve the question."
        "Put your final answer in <answer> </answer> tags. "
    )
GROUNDING_PROMPT = (
    "Your job is to identify the region where auxiliary line, box, or editing could help solve the following problem. Give bounding boxes in LOC format."
)
EDIT_PROMPT = (
    "Edit the region where auxiliary line, box, or drawing could help solve the following problem."
)

def _build_question_prompt(question):
    return (
    "<|startoftext|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"<image>\n {COT_PROMPT} {question}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

def _build_grounding_prompt(question):
    """Build the grounding prompt for a given question."""
    return f'''<|startoftext|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n {GROUNDING_PROMPT} {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<LOC_BEGIN><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><LOC_END><|eot_id|>'''


def _strip_option_prefix(option_text: str) -> str:
    stripped = option_text.strip()
    stripped = re.sub(r"^\(?\s*[A-Za-z]\s*\)?\s*[\.\):\-]?\s*", "", stripped)
    return stripped.strip()


def _normalize_arxivqa_options(options: list[str]) -> list[str]:
    if not isinstance(options, list) or len(options) == 0:
        raise ValueError("ArxivQA sample has invalid or empty 'options'.")

    normalized_options: list[str] = []
    for idx, option in enumerate(options):
        if not isinstance(option, str):
            raise ValueError(f"ArxivQA option at index {idx} is not a string: {type(option)!r}")
        option_char = chr(ord("A") + idx)
        option_text = _strip_option_prefix(option)
        normalized_options.append(f"{option_char}) {option_text}")
    return normalized_options

def _parse_bbox(raw_bbox):
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, str):
        raw_bbox = raw_bbox.strip()
        if not raw_bbox:
            return None
        try:
            raw_bbox = json.loads(raw_bbox)
        except json.JSONDecodeError:
            return None
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            return [int(coord) for coord in raw_bbox]
        except (TypeError, ValueError):
            return None
    return None



def _resolve_arxivqa_image_path(raw_image, image_root: str, sample_id: str) -> str:
    if not isinstance(raw_image, str) or not raw_image.strip():
        raise ValueError(f"ArxivQA sample '{sample_id}' has invalid image: {raw_image!r}")
    rel = raw_image.strip()
    if os.path.isabs(rel):
        return rel
    return os.path.join(image_root, rel)


def _normalize_arxivqa_label(raw_label) -> str:
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError(f"ArxivQA sample has invalid label: {raw_label!r}")
    return raw_label.strip()


def download_process_arxivqa(data_root: str = "./data"):
    dataset = load_dataset("MMInstruction/ArxivQA", split="train")
    dataset.select(range(10000)).save_to_disk(os.path.join(data_root, "arxivqa_select_10k.jsonl"))
    new_dataset = Dataset.from_list([])
    for data in dataset:
        question = data["question"]
        options = data["options"]
        label = data["label"]
        image = data["image"]
        image_path = os.path.join(data_root, f"{data['id']}.png")
        image.save(image_path)
        new_dataset.append(
            {
                "question": question,
                "options": options,
                "image": image_path,
                "label": label,
            }
        )
    return new_dataset



def get_arxivqa_interleave_questions(
    split: str = "train",
    data_root: str | None = None,
    image_root: str | None = None,
) -> tuple[Dataset, Dataset]:
    """Build paired (gen_ds, und_ds) ArxivQA datasets for `thinkmorph_interleave`.

    The gen row has ``image_gt=None`` so the trainer skips perceptual reward,
    advantage, and loss computation for it (rollout-only). The und row carries
    the multiple-choice answer label and is rewarded by ``correctness_reward_func``.
    Both rows are tagged ``dataset_type="answer_only"`` and share a sample_id
    prefixed with ``arxivqa:`` to avoid collisions with ThinkMorph pids.
    """
    if split != "train":
        raise ValueError(f"Unsupported split '{split}' for ArxivQA. Use 'train'.")

    data_root = DATA_ROOT
    image_root = os.path.join(data_root, "arxivqa")

    data_path = os.path.join(data_root, "arxivqa_select_10k.jsonl")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"ArxivQA file not found: {data_path}")

    gen_rows: list[dict] = []
    und_rows: list[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading arxivqa.jsonl", unit="line")):
            if not line.strip():
                continue
            sample = json.loads(line)
            raw_id = sample.get("id", f"row-{idx}")
            sample_id = f"arxivqa:{raw_id}"

            question = sample.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"ArxivQA sample '{sample_id}' has invalid question: {question!r}")

            normalized_options = _normalize_arxivqa_options(sample.get("options"))
            question_text = (
                f"{question.strip()}\n"
                f"{chr(10).join(normalized_options)}\n"
                "choose one of the options (char ord by len(options))"
            )
            image_path = _resolve_arxivqa_image_path(sample.get("image"), image_root, sample_id)
            answer_gt = _normalize_arxivqa_label(sample.get("label"))

            edit_instruction = f"{EDIT_PROMPT} {question_text}"
            cot_instruction = f"{COT_PROMPT} {question_text}"

            gen_rows.append(
                {
                    "task_type": "image_edit",
                    "dataset_type": "answer_only",
                    "sample_id": sample_id,
                    "prompt": [{"role": "user", "content": f"<image>\n{edit_instruction}"}],
                    "instruction": edit_instruction,
                    "image": image_path,
                    "image_gen_enc": image_path,
                    "image_gt": None,  # gating signal: no perceptual reward / advantage / loss
                    "answer_gt": None,
                }
            )
            und_rows.append(
                {
                    "task_type": "text",
                    "dataset_type": "answer_only",
                    "sample_id": sample_id,
                    "prompt": [{"role": "user", "content": f"<image>\n{cot_instruction}"}],
                    "instruction": cot_instruction,
                    "image": image_path,
                    "answer_gt": answer_gt,
                }
            )

    gen_ds = Dataset.from_list(gen_rows, features=INTERLEAVE_GEN_FEATURES)
    und_ds = Dataset.from_list(und_rows, features=INTERLEAVE_UND_FEATURES)
    return gen_ds, und_ds
