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

def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if lmms_eval_specific_kwargs["format"] == "default":
        if context:
            context = f"Context: {context}\n"

        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        return f"{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["format"] == "qwen_vl":
        prompt = "Context: {}\nQuestion: {}\nOptions: {}\nAnswer:"
        context = context if context else "N/A"
        prompt = prompt.format(context, question, choices_str)
        return prompt
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs}")


def sqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


from lmms_eval.tasks._robust_extract import extract_mc_letter


def sqa_process_results(doc, results):
    """Robust ScienceQA-image scorer.

    Original scorer only accepted `<letter>.` prefix or exact-lowercase match. This
    extended version tolerates chain-of-thought prose, <answer> tags, \\boxed answers,
    \"The answer is X\" phrasing, and trailing/last letters.
    """
    target = sqa_doc_to_target(doc).strip().lower()
    pred_raw = results[0] if results else ""
    # Fast path: exact match (unchanged behaviour)
    if pred_raw.strip().lower() == target:
        return {"exact_match": 1.0}
    # Robust extractor
    n = len(doc["choices"])
    valid = "".join(chr(ord("A") + i) for i in range(max(2, min(n, 26))))
    letter = extract_mc_letter(pred_raw or "", valid_letters=valid).lower()
    if letter and letter == target:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
