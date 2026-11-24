"""Shared helpers used by every step of the pipeline.

Keeping these in one place means training, evaluation and the demo all use
exactly the same system prompt and chat format. A mismatch between how a model
is trained and how it is served ("train/serve skew") is one of the most common
reasons a fine-tuned model looks worse in production than in evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

# The same system prompt is used for training, the baseline evaluation, the
# tuned evaluation and the demo app. app/app.py keeps a copy (the Space only
# ships the app/ folder); tests/test_common.py checks that the two match.
SYSTEM_PROMPT = (
    "You are a friendly, concise customer support assistant for an online store. "
    "Help customers with orders, cancellations, refunds, shipping, payments, "
    "invoices and their account. When you need a store-specific detail you do not "
    "know (an order number, phone number, website URL, opening hours and so on), "
    "write a placeholder in double curly braces, such as {{Order Number}} or "
    "{{Customer Support Phone Number}}, instead of making one up."
)

# Matches placeholders such as {{Order Number}} in the dataset's responses.
PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

# evaluate.py --compare writes results/REPORT.md and, next to it, this record of what the
# report was built from. export.py reads it to label the model card, to spot a report that
# is older than the latest evaluation, and to refuse to publish an adapter the report does
# not describe.
REPORT_INFO_FILE = "report.json"


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Read a YAML experiment config."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def build_prompt_messages(user_text: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """The conversation the model sees before it answers: system + user turn."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def file_digest(path: str | os.PathLike) -> str:
    """A short hash of a file's bytes: it changes whenever the file does."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def adapter_path(cfg: dict[str, Any], folder: str | os.PathLike | None = None) -> Path:
    """The LoRA adapter folder: the one given (evaluate.py --adapter), or the one step 3 saves."""
    return Path(folder) if folder else Path(cfg["train"]["output_dir"]) / "final_adapter"


def adapter_digest(folder: str | os.PathLike) -> str | None:
    """A short hash of a LoRA adapter's weights, or None when the folder has none.

    Training again overwrites the adapter in the same folder, so the hash, not the
    folder name, tells which adapter a set of results describes.
    """
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (Path(folder) / name).is_file():
            return file_digest(Path(folder) / name)
    return None


def set_seed(seed: int) -> None:
    """Make runs repeatable: same seed, same data order, same LoRA init.

    numpy and torch are seeded only when they are installed (the unit tests run without torch).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def use_one_gpu() -> None:
    """Show only the first GPU to this process, unless a distributed launcher is in charge.

    Kaggle's "GPU T4 x2" has two GPUs. Without a launcher such as torchrun, the
    Hugging Face Trainer would spread one process over both with DataParallel,
    which TRL's SFTTrainer refuses. One T4 is all a 4B QLoRA run needs. Call this
    before torch is imported: CUDA reads the variable when it starts.
    """
    if "WORLD_SIZE" not in os.environ and "LOCAL_RANK" not in os.environ:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
