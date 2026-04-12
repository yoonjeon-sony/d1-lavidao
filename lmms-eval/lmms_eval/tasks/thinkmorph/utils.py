import random
import re

def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def thinkmorph_doc_to_visual(doc):
    return [doc["problem_image_0"]]


def thinkmorph_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"].strip()
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") + question + lmms_eval_specific_kwargs.get("post_prompt", "")
    return prompt

def thinkmorph_process_result(doc, results):
    raw_pred = results[0].strip()
    extracted_pred = extract_xml_answer(raw_pred)

    target = doc["answer"]
    if extracted_pred.lower() == target.lower():
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
