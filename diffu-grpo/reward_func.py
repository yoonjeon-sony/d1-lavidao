import io
import os
import multiprocessing
import random
import numpy as np
import re
import resource
import time
import string
import shutil

import torch
import torch.nn.functional as F
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer_gt, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] if isinstance(completion, list) else completion for completion in completions]
    q = prompts[0][-1]["content"] if isinstance(prompts[0], list) else prompts[0]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer_gt[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer_gt)]

def correct_grounding_reward_func(prompts, completions, ground_gt, step=None, run_name=None, **kwargs) -> list[float]:
    def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise IoU between two tensors of shape (4,) in xyxy format.
        """
        # Intersection
        inter_x1 = torch.max(boxes1[0], boxes2[0])
        inter_y1 = torch.max(boxes1[1], boxes2[1])
        inter_x2 = torch.min(boxes1[2], boxes2[2])
        inter_y2 = torch.min(boxes1[3], boxes2[3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Areas
        area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
        area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        # Union
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou


    box = [[int(y) for y in re.compile('<LOC_([0-9]+)>').findall(x)] for x in completions]
    rewards = []
    for pred_box, gt_box in zip(box, ground_gt):
        if len(pred_box) == 4:
            rewards.append(_pairwise_iou(
                torch.tensor(pred_box, dtype=torch.float32),
                torch.tensor(gt_box,   dtype=torch.float32),
            ).item())
        else:
            rewards.append(0.0)
            
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{prompts[0]}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{ground_gt[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{box[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Reward:{RESET}\n{rewards[0]}\n",
    )
    return rewards

_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _get_clip(device: torch.device = None):
    """Return a cached frozen CLIP image encoder + processor.

    Lazily loads on first use, freezes parameters, and migrates to ``device``
    on demand. Matches the LPIPS-cache pattern that lived here before. Model
    id is read from ``CLIP_REWARD_MODEL`` (default ``openai/clip-vit-base-patch32``).
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor

        model_id = os.environ.get("CLIP_REWARD_MODEL", "openai/clip-vit-base-patch32")
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        for p in _CLIP_MODEL.parameters():
            p.requires_grad_(False)
    if device is not None:
        cur = next(_CLIP_MODEL.parameters()).device
        if cur != device:
            _CLIP_MODEL = _CLIP_MODEL.to(device)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def perceptual_score_reward_func(
    prompts,
    completions,
    image_gt,
    step=None,
    run_name=None,
    **kwargs,
) -> list[float]:
    """CLIP-image cosine similarity between generated image and ground truth.

    Inputs:
      * ``completions[i]``: PIL.Image — the rollout-decoded generated image.
      * ``image_gt[i]``:    string path or PIL.Image — the ground-truth image.

    For each pair we encode both images with a frozen CLIP image tower,
    L2-normalize, and take cosine similarity in [-1, 1]. The similarity is
    then linearly remapped to a reward in [0, 1] via::

        reward = max(0, (sim - threshold) / (1 - threshold))

    With the default ``CLIP_REWARD_THRESHOLD=0.5``:
      * sim = 1.0 (identical images)        -> reward = 1.0
      * sim = 0.75 (visually similar)       -> reward = 0.5
      * sim <= threshold (unrelated/poor)   -> reward = 0.0

    Lower the threshold (env var) for a more forgiving curve, raise it to
    only reward near-perfect matches. Set ``DIFFU_GRPO_DEBUG=1`` to log the
    raw similarity for sample 0.
    """
    from PIL import Image as _PILImage

    threshold = float(os.environ.get("CLIP_REWARD_THRESHOLD", "0.5"))
    debug = os.environ.get("DIFFU_GRPO_DEBUG") == "1"

    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    try:
        clip_model, clip_proc = _get_clip(device)
    except Exception as e:
        print(f"[clip_reward] CLIP load failed: {type(e).__name__}: {e}", flush=True)
        return [float("nan") for _ in completions]

    def _to_pil(x):
        if x is None:
            return None
        if isinstance(x, _PILImage.Image):
            return x.convert("RGB")
        if isinstance(x, str):
            return _PILImage.open(x).convert("RGB")
        # numpy / torch fall-through
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return _PILImage.fromarray(x).convert("RGB")
        except Exception:
            pass
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.permute(1, 2, 0)
            arr = arr.clamp(0, 1).mul(255).to(torch.uint8).numpy()
            return _PILImage.fromarray(arr).convert("RGB")
        return None

    rewards: list[float] = []
    sims: list[float] = []
    for i, (pred_img, gt_img) in enumerate(zip(completions, image_gt)):
        try:
            pred_pil = _to_pil(pred_img)
            gt_pil = _to_pil(gt_img)
            if pred_pil is None or gt_pil is None:
                if debug:
                    print(
                        f"[clip_reward] sample {i} NaN (missing image): "
                        f"pred={pred_pil is not None} gt={gt_pil is not None}",
                        flush=True,
                    )
                rewards.append(float("nan"))
                sims.append(float("nan"))
                continue

            inputs = clip_proc(images=[pred_pil, gt_pil], return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = clip_model.get_image_features(**inputs)  # (2, D)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            sim = float((feats[0] * feats[1]).sum().item())
            r = max(0.0, (sim - threshold) / max(1.0 - threshold, 1e-8))
            if not np.isfinite(r):
                r = float("nan")
            rewards.append(r)
            sims.append(sim)
        except Exception as e:
            import traceback
            print(
                f"[clip_reward] sample {i} failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            rewards.append(float("nan"))
            sims.append(float("nan"))

    # ANSI colors for the per-batch banner
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    if prompts and rewards:
        prompt0 = prompts[0] if prompts[0] is not None else ""
        prompt0 = prompt0.replace("<|reserved_token_5|>", "*").replace(
            "<|reserved_token_6|>", "-"
        )
        sim0 = sims[0] if sims else float("nan")
        print(
            "-" * 20,
            f"\n{RED}Prompt:{RESET}\n{prompt0}\n",
            "-" * 20,
            f"\n{YELLOW}CLIP Similarity:{RESET} {sim0:.4f}  "
            f"(threshold={threshold:.3f})  reward={rewards[0]:.4f}\n",
            flush=True,
        )
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # <answer>...</answer> can appear anywhere in the response, possibly
    # spanning multiple lines. Use re.search + re.DOTALL so the pattern is
    # not anchored to the start of the string.
    pattern = r"<answer>(.*?)</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]



def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


def run_test(test_func_name, code_str, result_dict, cwd_path, rank):
    cwd_path = cwd_path + "/" + str(rank)
    os.makedirs(cwd_path, exist_ok=True)

    def target():
        try:
            # Set memory limit to 1 GB (in bytes)
            soft, hard = 1_000_000_000, 1_000_000_000
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

            # Change working directory
            os.chdir(cwd_path)

            # Create a new namespace and execute code
            local_ns = {}
            exec(code_str, local_ns)
            local_ns[test_func_name]()
            result_dict[test_func_name] = True
        except Exception:
            result_dict[test_func_name] = False

    proc = multiprocessing.Process(target=target)
    proc.start()
    proc.join(timeout=1.0)

    if proc.is_alive():
        proc.terminate()
        result_dict[test_func_name] = False


def split_test_function(test_code: str, base_name: str = "test_case"):
    lines = test_code.strip().splitlines()
    result = []
    counter = 1

    for line in lines:
        line = line.strip()
        if line.startswith("assert"):
            fn = f"def {base_name}_{counter}():\n    {line}"
            result.append(fn)
            counter += 1

    return "\n\n".join(result)


BLOCKED_MODULES = [
    " os",
    " sys",
    " subprocess",
    " shutil",
    " socket",
    " psutil",
    " ctypes",
    " pathlib",
    " builtins",
    "__import__",
]


def is_safe_code(code_str):
    """Reject code that contains any blocked modules or keywords."""
    for blocked in BLOCKED_MODULES:
        if blocked in code_str:
            return False
    return True


def time_based_random_string(length=10):
    seed = int(time.time() * 1e6)
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def coding_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    execution_cwd = kwargs.get("cwd_path")
    if not os.path.exists(execution_cwd):
        os.makedirs(execution_cwd, exist_ok=True)
    programs = []
    for group in completions:
        for message in group:
            content = message["content"]

            # Look for the first python code block
            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            code = code_match.group(1) if code_match else ""

            # Check if the code is within <answer>...</answer>
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            is_in_answer = False
            if answer_match and code:
                answer_content = answer_match.group(1)
                is_in_answer = f"```python\n{code}```" in answer_content

            programs.append((code, is_in_answer))

    unit_tests = [entry["tests"] for entry in answer]
    rewards = []
    for i, (solution, tests) in enumerate(zip(programs, unit_tests)):
        solution, is_in_answer = solution
        is_in_answer_reward = 0.5 if is_in_answer else 0
        # Step 1: Extract imported function name
        import_match = re.search(r"from solution import (\w+)", tests)
        if not import_match:
            assert_match = re.search(r"assert\s+(\w+)\s*\(", tests)
            if assert_match:
                imported_func = assert_match.group(1)
            else:
                print("No import or assert-based function name found in test code.")
                print("=" * 10, "tests", "=" * 10)
                print(tests)
                print("=" * 30)
                rewards.append(0)
                continue
        else:
            imported_func = import_match.group(1)

        # Step 2: Extract defined function name from solution
        solution_match = re.search(r"def (\w+)\(", solution)
        if not solution_match:
            print("No function definition in generation")
            print("=" * 10, "model output", "=" * 10)
            print(solution)
            print("=" * 10)
            rewards.append(0 + is_in_answer_reward)
            continue
        defined_func = solution_match.group(1)

        # Step 3: Rename if function names differ
        if defined_func != imported_func:
            solution = re.sub(rf"\bdef {defined_func}\b", f"def {imported_func}", solution)

        # Step 3.5: Check iff code is safe to run
        if not is_safe_code(solution):
            rewards.append(0 + is_in_answer_reward)
            print(f"Potentially Unsafe Generation:\n{solution}")
            continue
        # Step 4: Extract test function names
        test_funcs = re.findall(r"def (\w+)\(\):", tests)

        if len(test_funcs) <= 1:
            tests = split_test_function(tests)
            print(f"Fixed Test functions\n{tests}")
            test_funcs = re.findall(r"def (\w+)\(\):", tests)

        # Step 5: Replace import with the solution code
        if import_match:
            test_code = re.sub(r"from solution import \w+", lambda _: solution, tests)
        elif assert_match:
            test_code = solution + tests

        # Step 6: Build complete executable test code
        # Dict to hold test results
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        # Run all test functions in parallel
        jobs = []
        for rank, fn in enumerate(test_funcs):
            p = multiprocessing.Process(
                target=run_test, args=(fn, test_code, result_dict, execution_cwd, rank)
            )
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        # Compute reward
        passed = sum(result_dict.get(fn, False) for fn in test_funcs)
        total = len(test_funcs)
        reward = passed / total if total > 0 else 0.0
        rewards.append(reward + is_in_answer_reward)
    return rewards
