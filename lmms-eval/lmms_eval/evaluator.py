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

import collections
import inspect
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import Image, Sequence
from loguru import logger as eval_logger
from PIL import Image as PILImage
from tqdm import tqdm

import lmms_eval.api
import lmms_eval.api.metrics
import lmms_eval.api.registry
from lmms_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.models import get_model
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import (
    create_iterator,
    get_datetime_str,
    get_git_commit_hash,
    handle_non_serializable,
    hash_string,
    make_table,
    positional_deprecated,
    run_task_tests,
    simple_parse_args_string,
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = get_datetime_str(),
    cli_args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.

    :return
        Dictionary of results
    """
    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(f"generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""

    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=model)

    task_dict = get_task_dict(tasks, task_manager)

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lmms_eval.models.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    elif isinstance(model, lmms_eval.api.model.lmms):
        lm = model

    # Expose datetime_str to the model so it can use it for output paths
    lm.datetime_str = datetime_str

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if cli_args is not None and hasattr(cli_args, "lmms_eval_specific_kwargs") and cli_args.lmms_eval_specific_kwargs:
                    model_key, _, kwargs_str = cli_args.lmms_eval_specific_kwargs.partition(":")
                    if model_key == task_obj.model_name or model_key == "default":
                        cli_specific_kwargs = simple_parse_args_string(kwargs_str)
                        # Preserve task-level kwargs (e.g. prompt templates) and let CLI
                        # values override only the provided keys.
                        task_specific_kwargs = task_obj.lmms_eval_specific_kwargs if isinstance(task_obj.lmms_eval_specific_kwargs, dict) else {}
                        task_obj.lmms_eval_specific_kwargs = {**task_specific_kwargs, **cli_specific_kwargs}

                if predict_only:
                    eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                    else:
                        eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        cli_args=cli_args,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available TODO: add model info
        # if isinstance(lm, lm_eval.models.huggingface.HFLM):
        #     results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = datetime_str
        # add_env_info(results)  # additional environment info to results
        # add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm: "LM",
    task_dict,
    limit: Optional[int] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    cli_args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :return
        Dictionary of results
    """

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store the ordering of tasks and groups
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    save_parity_cases = bool(getattr(cli_args, "save_parity_cases", False)) if cli_args is not None else False
    parity_cases_root = Path(getattr(cli_args, "parity_cases_root", "DEBUG/parity_text_gen")) if cli_args is not None else Path("DEBUG/parity_text_gen")
    parity_cases_max_per_task = int(getattr(cli_args, "parity_cases_max_per_task", 4)) if cli_args is not None else 4
    parity_cases_dir = parity_cases_root / "cases"
    if save_parity_cases:
        parity_cases_dir.mkdir(parents=True, exist_ok=True)

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    name_to_task = {}
    results_dict = {}
    if not log_samples:
        if not all("bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys() for task_output in eval_tasks):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    for task_output in eval_tasks:
        task: Task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args

        name_to_task[task_name] = task

        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"
        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        if ("group_alias" in configs[task_name]) and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,  # later we will add them
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
        eval_logger.debug(f"Task: {task_output.task_name}; number of requests on this rank: {len(task._instances)}")
        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = "loglikelihood" if task.OUTPUT_TYPE == "multiple_choice" else task.OUTPUT_TYPE
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    # Parity-case export mode: save built requests only, bypass model inference.
    if save_parity_cases:
        RANK = lm.rank
        WORLD_SIZE = lm.world_size

        for task_output in eval_tasks:
            task = task_output.task
            if task is None:
                continue

            instances_by_doc_id = collections.defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            for reqs in instances_by_doc_id.values():
                reqs.sort(key=lambda x: x.idx)

            selected_doc_ids = sorted(instances_by_doc_id.keys())[:parity_cases_max_per_task]
            for doc_id in selected_doc_ids:
                reqs = instances_by_doc_id[doc_id]
                req0 = reqs[0]
                req_args = req0.arguments if req0.arguments else ()

                input_text = req_args[0] if len(req_args) > 0 else None
                doc_to_visual = req_args[2] if len(req_args) > 2 else task.doc_to_visual
                req_task_name = req_args[4] if len(req_args) > 4 and req_args[4] is not None else task_output.task_name
                req_split_name = req_args[5] if len(req_args) > 5 else None

                if req_split_name is not None and hasattr(task, "dataset") and req_split_name in task.dataset:
                    doc = task.dataset[req_split_name][doc_id]
                else:
                    doc = task.eval_docs_no_media[doc_id]

                target = task.doc_to_target(doc)
                saved_doc = {}
                for key, value in doc.items():
                    if "image" not in key:
                        if isinstance(value, dict) and "array" in value:
                            continue
                        saved_doc[key] = value

                image_paths = []
                try:
                    visuals = doc_to_visual(doc)
                    if visuals is None:
                        visuals = []
                    task_image_dir = parity_cases_dir / str(req_task_name)
                    task_image_dir.mkdir(parents=True, exist_ok=True)
                    for img_idx, visual in enumerate(visuals):
                        out_path = task_image_dir / f"{doc_id}_{img_idx}.png"
                        if isinstance(visual, PILImage.Image):
                            visual.convert("RGB").save(out_path)
                        elif isinstance(visual, np.ndarray):
                            PILImage.fromarray(visual).convert("RGB").save(out_path)
                        elif isinstance(visual, (str, Path)):
                            PILImage.open(visual).convert("RGB").save(out_path)
                        elif isinstance(visual, dict):
                            if visual.get("path"):
                                PILImage.open(visual["path"]).convert("RGB").save(out_path)
                            elif visual.get("bytes") is not None:
                                from io import BytesIO

                                PILImage.open(BytesIO(visual["bytes"])).convert("RGB").save(out_path)
                            else:
                                continue
                        else:
                            continue
                        image_paths.append(str(out_path))
                except Exception as e:
                    eval_logger.debug(f"Failed to save parity images for task={req_task_name}, doc_id={doc_id}: {e}")

                task_output.logged_samples.append(
                    {
                        "doc_id": doc_id,
                        "task": req_task_name,
                        "split": req_split_name,
                        "input_text": input_text,
                        "image_paths": image_paths,
                        "doc": saved_doc,
                        "target": target,
                        "resps": [],
                    }
                )

        if WORLD_SIZE > 1:
            for task_output in eval_tasks:
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = list(task_output.logged_samples)
                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )
                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))
            dist.barrier()

        if RANK == 0:
            samples = collections.defaultdict(list)
            configs = collections.defaultdict(dict)
            for task_output in eval_tasks:
                samples[task_output.task_name] = task_output.logged_samples
                configs[task_output.task_name] = task_output.task_config

            results_dict["samples"] = dict(samples)
            results_dict["configs"] = dict(sorted(configs.items()))

            parity_cases_root.mkdir(parents=True, exist_ok=True)
            test_cases = {}
            for task_name_key, task_samples in samples.items():
                ordered = sorted(task_samples, key=lambda x: x.get("doc_id", 0))
                task_cases = []
                for sample in ordered[:parity_cases_max_per_task]:
                    task_cases.append(
                        {
                            "doc_id": sample.get("doc_id"),
                            "task": sample.get("task", task_name_key),
                            "split": sample.get("split"),
                            "input_text": sample.get("input_text"),
                            "image_paths": sample.get("image_paths", []),
                            "doc": sample.get("doc", {}),
                            "target": sample.get("target"),
                        }
                    )
                test_cases[task_name_key] = task_cases

            test_cases_path = parity_cases_root / "test_cases.json"
            with open(test_cases_path, "w", encoding="utf-8") as f:
                json.dump(test_cases, f, indent=2, ensure_ascii=False, default=handle_non_serializable)
            eval_logger.info(f"Saved parity test cases from built requests: {test_cases_path}")
        else:
            results_dict = None

        if hasattr(lm, "accelerator"):
            lm.accelerator.wait_for_everyone()
        return results_dict

    ### Run LMM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info("Running {} requests".format(reqtype))
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)  # Choiszt run generate until

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            if not cli_args.process_with_media:
                doc_iterator = create_iterator(enumerate(task.eval_docs_no_media), rank=RANK, limit=int(limit) if limit else None, world_size=WORLD_SIZE)
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)
            doc_iterator_for_counting = itertools.islice(range(len(task.test_docs())), RANK, limit, WORLD_SIZE) if task.has_test_docs() else itertools.islice(range(len(task.validation_docs())), RANK, limit, WORLD_SIZE)
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=total_docs, desc=f"Postprocessing", disable=(RANK != 0))
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                # metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])
                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        # If image is not in key
                        if "image" not in key:
                            # If audio is also not the value
                            if isinstance(value, dict) and "array" in value:
                                continue
                            else:
                                saved_doc[key] = value
                    input_text = None
                    req_task_name = task_name
                    req_split_name = None
                    image_paths = []
                    if save_parity_cases and requests and requests[0].arguments:
                        req_args = requests[0].arguments
                        if len(req_args) > 0:
                            input_text = req_args[0]
                        if len(req_args) > 4 and req_args[4] is not None:
                            req_task_name = req_args[4]
                        if len(req_args) > 5:
                            req_split_name = req_args[5]
                        try:
                            visuals = task.doc_to_visual(doc)
                            if visuals is None:
                                visuals = []
                            task_image_dir = parity_cases_dir / str(req_task_name)
                            task_image_dir.mkdir(parents=True, exist_ok=True)
                            for img_idx, visual in enumerate(visuals):
                                out_path = task_image_dir / f"{doc_id}_{img_idx}.png"
                                if isinstance(visual, PILImage.Image):
                                    visual.convert("RGB").save(out_path)
                                elif isinstance(visual, np.ndarray):
                                    PILImage.fromarray(visual).convert("RGB").save(out_path)
                                elif isinstance(visual, (str, Path)):
                                    PILImage.open(visual).convert("RGB").save(out_path)
                                elif isinstance(visual, dict):
                                    if visual.get("path"):
                                        PILImage.open(visual["path"]).convert("RGB").save(out_path)
                                    elif visual.get("bytes") is not None:
                                        from io import BytesIO

                                        PILImage.open(BytesIO(visual["bytes"])).convert("RGB").save(out_path)
                                    else:
                                        continue
                                else:
                                    continue
                                image_paths.append(str(out_path))
                        except Exception as e:
                            eval_logger.debug(f"Failed to save sample images for task={req_task_name}, doc_id={doc_id}: {e}")

                    example = {
                        "doc_id": doc_id,
                        "task": req_task_name,
                        "split": req_split_name,
                        "input_text": input_text,
                        "image_paths": image_paths,
                        "doc": saved_doc,
                        "target": target,
                        # "arguments": filtered_arguments,
                        "resps": [req.resps for req in requests],
                        # "filtered_resps": [req.filtered_resps[filter_key] for req in requests],
                        # "doc_hash": hash_string(
                        #     json.dumps(
                        #         requests[0].doc,
                        #         indent=2,
                        #         default=handle_non_serializable,
                        #         ensure_ascii=False,
                        #     )
                        # ),
                        # "prompt_hash": hash_string(requests[0].arguments[0]),
                        # "target_hash": hash_string(str(target)),
                    }
                    # example.update(metrics)
                    task_output.logged_samples.append(example)
                pbar.update(1)

            pbar.close()

    if hasattr(lm, "_model"):
        del lm._model
        torch.cuda.empty_cache()

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            # for task_name, task_samples in list(samples.items()):
            full_samples = [None] * WORLD_SIZE if RANK == 0 else None
            per_rank_samples = []
            for sample in task_output.logged_samples:
                per_rank_samples.append(sample)

            torch.distributed.gather_object(
                obj=per_rank_samples,
                object_gather_list=full_samples,
                dst=0,
            )

            if RANK == 0:
                task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))
        dist.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        samples = collections.defaultdict(list)
        configs = collections.defaultdict(dict)
        for task_output in eval_tasks: 
            samples[task_output.task_name] = task_output.logged_samples
            configs[task_output.task_name] = task_output.task_config

        results_dict["samples"] = dict(samples)
        results_dict["configs"] = dict(sorted(configs.items()))

        if save_parity_cases:
            # Save compact parity cases (up to N samples per task) for downstream debugging.
            parity_cases_root.mkdir(parents=True, exist_ok=True)
            test_cases = {}
            for task_name_key, task_samples in samples.items():
                ordered = sorted(task_samples, key=lambda x: x.get("doc_id", 0))
                task_cases = []
                for sample in ordered[:parity_cases_max_per_task]:
                    task_cases.append(
                        {
                            "doc_id": sample.get("doc_id"),
                            "task": sample.get("task", task_name_key),
                            "split": sample.get("split"),
                            "input_text": sample.get("input_text"),
                            "image_paths": sample.get("image_paths", []),
                            "doc": sample.get("doc", {}),
                            "target": sample.get("target"),
                        }
                    )
                test_cases[task_name_key] = task_cases

            test_cases_path = parity_cases_root / "test_cases.json"
            with open(test_cases_path, "w", encoding="utf-8") as f:
                json.dump(test_cases, f, indent=2, ensure_ascii=False, default=handle_non_serializable)
            eval_logger.info(f"Saved parity test cases: {test_cases_path}")
    else:
        results_dict = None

    if hasattr(lm, "accelerator"):
        lm.accelerator.wait_for_everyone()

    return results_dict


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
