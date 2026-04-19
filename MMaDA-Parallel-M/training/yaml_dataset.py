"""YAML-driven interleave dataset loader for MMaDA-Parallel-M.

Ports the yaml branch of dLLM-RL's ``build_dataset_lazy`` but returns a
``torch.utils.data.Dataset`` that yields samples in the MMaDA interleave
schema (see ``dataset_adapter.py``).

Grounding-mode yaml entries are skipped. Weights/length_group/batch_sizes in
the yaml are ignored (uniform concatenation).

Source paths (``json_path`` and ``s3_prefix`` / ``s3_prefix_gen``) are
transparently remapped to ``DATA_ROOT`` (``/scratch2/yoonjeon.kim/data/``) so
the yaml written against ``/group2/dgm/yoonjeon/data/`` works out of the box.
"""
from __future__ import annotations

import json
import math
import os
import random
from functools import partial
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

try:  # yaml dependency check (fail loudly at import time)
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required; install with `pip install pyyaml`") from exc

from .dataset_adapter import PROCESS_FUNCTIONs


DATA_ROOT = "/scratch2/yoonjeon.kim/data/"
_LEGACY_PREFIXES = (
    "/group2/dgm/yoonjeon/data/",
    "/group2/dgm/yoonjeon/data",
    "./data/",
    "./data",
    "data/",
    "data",
)


def _remap_path(path: str) -> str:
    """Rewrite paths that point at the legacy group2 / relative-data roots
    onto the actual on-disk location."""
    if not path:
        return path
    path = path.strip()
    if os.path.isabs(path) and not path.startswith("/group2/dgm/yoonjeon/data"):
        return path
    for prefix in _LEGACY_PREFIXES:
        if path.startswith(prefix):
            tail = path[len(prefix):].lstrip("/")
            return os.path.join(DATA_ROOT, tail)
    # bare filename → treat as relative to DATA_ROOT
    return os.path.join(DATA_ROOT, path)


def _load_jsonl(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file type for json_path={path}")


def _parse_sampling(sampling_strategy: str, total: int) -> tuple[str, int | None]:
    if ":" not in sampling_strategy:
        return sampling_strategy, None
    name, n = sampling_strategy.split(":")
    if "%" in n:
        n = math.ceil(int(n.split("%")[0]) * total / 100)
    else:
        n = int(n)
    return name, n


def _make_image_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class ThinkMorphZebraCoTDataset(Dataset):
    """One subset defined by one yaml entry.

    Owns a list of raw jsonl rows + a bound preprocess_fn. ``__getitem__``
    applies the preprocess_fn, opens the image paths, runs the transform, and
    returns the MMaDA interleave training sample.
    """

    def __init__(
        self,
        rows: list[dict],
        preprocess_fn,
        resolution: int,
        name: str = "",
    ) -> None:
        self.rows = rows
        self.preprocess_fn = preprocess_fn
        self.resolution = resolution
        self.name = name
        self._transform = _make_image_transform(resolution)

    def __len__(self) -> int:
        return len(self.rows)

    def _placeholder_tensor(self) -> torch.Tensor:
        placeholder = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        return self._transform(placeholder)

    def __getitem__(self, idx: int) -> dict:
        raw = self.rows[idx]
        sample = self.preprocess_fn(raw)

        in_img_path = _remap_path(sample["input_image"]) if sample.get("input_image") else None
        out_img_path = _remap_path(sample["output_image"]) if sample.get("output_image") else None

        if in_img_path and os.path.isfile(in_img_path):
            input_image = self._transform(_open_rgb(in_img_path))
        else:
            input_image = self._placeholder_tensor()

        if out_img_path and os.path.isfile(out_img_path):
            output_image = self._transform(_open_rgb(out_img_path))
        else:
            output_image = self._placeholder_tensor()

        return {
            "id": sample.get("id", idx),
            "input_text": sample["input_text"],
            "output_text": sample["output_text"],
            "input_image": input_image,
            "output_image": output_image,
            "is_text_only": bool(sample.get("is_text_only", False)),
            "is_text_only_output": bool(sample.get("is_text_only_output", False)),
            "mode": sample.get("mode", "image_gen"),
            "dataset": self.name,
        }


def build_yaml_dataset(
    yaml_path: str,
    resolution: int = 512,
    *,
    val_split: float = 0.1,
    seed: int = 42,
    skip_modes: Iterable[str] = ("grounding",),
    data_root: str = DATA_ROOT,
) -> tuple[Dataset, Dataset]:
    """Build (train, val) datasets from a thinkmorph/zebracot-style yaml.

    Arguments mirror the relevant parts of dLLM-RL's ``build_dataset_lazy`` --
    only the yaml branch, sampling via ``Random(seed)``, and no weights.
    """
    global DATA_ROOT  # allow override from caller
    if data_root != DATA_ROOT:
        DATA_ROOT = data_root  # noqa: F841 — used by _remap_path via module global

    skip_modes = set(skip_modes)

    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    entries = spec.get("datasets") or []
    if not entries:
        raise ValueError(f"No `datasets:` entries in {yaml_path}")

    subsets: list[ThinkMorphZebraCoTDataset] = []
    for entry in entries:
        kwargs = dict(entry.get("process_fn_kwargs") or {})
        mode = kwargs.get("mode", "text_gen")
        if mode in skip_modes:
            continue

        fn_name = entry.get("preprocess_fn")
        fn = PROCESS_FUNCTIONs.get(fn_name)
        if fn is None:
            raise KeyError(
                f"Unknown preprocess_fn {fn_name!r} — add it to dataset_adapter.PROCESS_FUNCTIONs"
            )
        preprocess = partial(fn, **kwargs)

        raw_json = entry.get("json_path")
        if not raw_json:
            raise ValueError(f"Entry missing json_path: {entry!r}")
        json_path = _remap_path(raw_json)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Dataset file not found: {json_path} (from {raw_json!r})")

        rows = _load_jsonl(json_path)
        raw_n = len(rows)

        strategy, n = _parse_sampling(entry.get("sampling_strategy", "all"), raw_n)
        if strategy == "first" and n is not None:
            rows = rows[:n]
        elif strategy == "end" and n is not None:
            rows = rows[-n:]
        elif strategy == "random" and n is not None:
            k = min(n, raw_n)
            rng = random.Random(seed)
            rows = rng.sample(rows, k)
        # strategy == "all" → keep everything

        name = entry.get("name", os.path.basename(json_path))
        subsets.append(
            ThinkMorphZebraCoTDataset(
                rows=rows,
                preprocess_fn=preprocess,
                resolution=resolution,
                name=f"{name}::{mode}",
            )
        )
        print(f"[yaml_dataset] {name} [{mode}]: {len(rows)} / {raw_n} rows from {json_path}")

    if not subsets:
        raise ValueError(
            f"No subsets after filtering (skip_modes={sorted(skip_modes)}) in {yaml_path}"
        )

    full = ConcatDataset(subsets)

    # Deterministic train/val split
    total = len(full)
    n_val = int(total * val_split)
    n_train = total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val], generator=gen)
    print(f"[yaml_dataset] total={total}  train={n_train}  val={n_val}")
    return train_ds, val_ds


def interleave_collate(batch: list[dict]) -> dict:
    """Collate function matching ``train_interleave.py`` expectations:
    stacked image tensors, list[str] for text, list[bool] for flags.
    """
    out = {
        "input_image": torch.stack([b["input_image"] for b in batch], dim=0),
        "output_image": torch.stack([b["output_image"] for b in batch], dim=0),
        "input_text": [b["input_text"] for b in batch],
        "output_text": [b["output_text"] for b in batch],
        "is_text_only": [b["is_text_only"] for b in batch],
        "is_text_only_output": [b["is_text_only_output"] for b in batch],
        "id": [b["id"] for b in batch],
        "mode": [b["mode"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }
    return out
