"""Preprocess functions that map raw ThinkMorph / ZebraCoT jsonl rows to the
MMaDA interleave training schema.

Input schema (per row, already loaded from jsonl):
  - ThinkMorph: pid, question, answer, problem_image_0, reasoning_image_0,
    full_text_only_thought, (optional) loc_string
  - ZebraCoT  : id,  question, answer, problem_image,   reasoning_image,
    full_text_only_thought

Output schema (per row):
  {
    "id":                    sample id,
    "input_text":            str,   # prompt + question
    "output_text":           str,   # reasoning + answer
    "input_image":           str,   # absolute path (always present)
    "output_image":          str | None,   # None for text_gen mode
    "is_text_only":          bool,  # no input image — always False here
    "is_text_only_output":   bool,  # True for text_gen mode (skip image CE)
    "mode":                  "text_gen" | "image_gen",
  }

``grounding`` mode is intentionally not mapped; callers should skip those rows.
"""
from __future__ import annotations

import os


COT_PROMPT = (
    "Let's think step-by-step to solve the question."
    "Put your final answer in <answer> </answer> tags. "
)
EDIT_PROMPT = (
    "Edit the region where auxiliary line, box, or drawing could help solve the following problem."
)


def _resolve(prefix: str, path: str) -> str:
    if not path:
        return path
    return os.path.join(prefix, path) if prefix else path


def preprocess_thinkmorph_mmada(
    row: dict,
    s3_prefix: str = "",
    s3_prefix_gen: str = "",
    mode: str = "text_gen",
) -> dict:
    q = row["question"]
    a = row["answer"]
    thought = row["full_text_only_thought"]
    problem_img = _resolve(s3_prefix, row["problem_image_0"])
    reasoning_img = _resolve(s3_prefix_gen, row.get("reasoning_image_0") or "")
    out_txt = f"{thought} Therefore the answer is {a}. <answer> {a} </answer>"

    if mode == "image_gen":
        return {
            "id": row["pid"],
            "input_text": f"{EDIT_PROMPT} {q}",
            "output_text": out_txt,
            "input_image": problem_img,
            "output_image": reasoning_img or None,
            "is_text_only": False,
            "is_text_only_output": False,
            "mode": "image_gen",
        }
    elif mode == "text_gen":
        return {
            "id": row["pid"],
            "input_text": f"{COT_PROMPT} {q}",
            "output_text": out_txt,
            "input_image": problem_img,
            "output_image": None,
            "is_text_only": False,
            "is_text_only_output": True,
            "mode": "text_gen",
        }
    else:
        raise ValueError(
            f"preprocess_thinkmorph_mmada: unsupported mode {mode!r} (expected text_gen or image_gen)"
        )


def preprocess_zebracot_mmada(
    row: dict,
    s3_prefix: str = "",
    s3_prefix_gen: str = "",
    mode: str = "text_gen",
) -> dict:
    q = row["question"]
    a = row["answer"]
    thought = row["full_text_only_thought"]
    problem_img = _resolve(s3_prefix, row["problem_image"])
    reasoning_img = _resolve(s3_prefix_gen, row.get("reasoning_image") or "")
    out_txt = f"{thought} <answer> {a} </answer>"

    if mode == "image_gen":
        return {
            "id": row["id"],
            "input_text": f"{EDIT_PROMPT} {q}",
            "output_text": out_txt,
            "input_image": problem_img,
            "output_image": reasoning_img or None,
            "is_text_only": False,
            "is_text_only_output": False,
            "mode": "image_gen",
        }
    elif mode == "text_gen":
        return {
            "id": row["id"],
            "input_text": f"{COT_PROMPT} {q}",
            "output_text": out_txt,
            "input_image": problem_img,
            "output_image": None,
            "is_text_only": False,
            "is_text_only_output": True,
            "mode": "text_gen",
        }
    else:
        raise ValueError(
            f"preprocess_zebracot_mmada: unsupported mode {mode!r} (expected text_gen or image_gen)"
        )


PROCESS_FUNCTIONs: dict = {
    "preprocess_thinkmorph": preprocess_thinkmorph_mmada,
    "preprocess_zebracot": preprocess_zebracot_mmada,
}
