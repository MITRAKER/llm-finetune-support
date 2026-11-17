"""Step 1: data preparation.

Turns the raw Bitext customer-support dataset into clean, deduplicated,
stratified train / validation / test splits in chat format.

Usage:
    python src/prepare_data.py --config configs/qwen3_4b_qlora.yaml
    # offline / tests: read a local CSV, JSON or JSONL file instead of the Hub
    python src/prepare_data.py --config configs/qwen3_4b_qlora.yaml --source my_rows.csv

Output (in data.out_dir from the config):
    train.jsonl, validation.jsonl, test.jsonl   one example per line
    stats.json                                  row counts after every step
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from common import build_prompt_messages, load_config, write_json, write_jsonl

REQUIRED_COLUMNS = ["instruction", "response", "intent"]
OPTIONAL_COLUMNS = ["category", "flags"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(source: str) -> pd.DataFrame:
    """Load the raw rows from a local file or from the Hugging Face Hub."""
    path = Path(source)
    if path.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"} and path.exists():
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        from datasets import load_dataset  # imported lazily: not needed for local files

        df = load_dataset(source, split="train").to_pandas()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source is missing required columns: {missing}")
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def normalize(text: Any) -> str:
    """Unicode-normalize and tidy whitespace without changing the wording."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    clean = unicodedata.normalize("NFKC", str(text))  # full-width letters, odd spaces -> standard
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = re.sub(r"[ \t]+", " ", clean)  # runs of spaces/tabs -> one space
    clean = re.sub(r" *\n *", "\n", clean)  # no spaces around line breaks
    clean = re.sub(r"\n{3,}", "\n\n", clean)  # at most one blank line
    return clean.strip()


def dedupe_key(text: str) -> str:
    """Key under which near-identical questions collide.

    "Cancel my order!" and "cancel   my order" both become "cancel my order".
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize, filter by length and drop duplicate questions.

    Returns the cleaned frame and the row count after each step, so the data
    card in the README can show exactly what was removed and why.
    """
    stats: dict[str, int] = {"raw": len(df)}

    df = df.copy()
    for col in ["instruction", "response", "intent", "category", "flags"]:
        df[col] = df[col].map(normalize)

    keep = (
        (df["instruction"].str.len() >= cfg["min_instruction_chars"])
        & (df["response"].str.len() >= cfg["min_response_chars"])
        & (df["response"].str.len() <= cfg["max_response_chars"])
        & (df["intent"] != "")
    )
    df = df[keep]
    stats["after_length_filter"] = len(df)

    # Deduplicate on the normalized question. Done BEFORE splitting, so the same
    # question can never end up in both train and test (that would be leakage).
    df = df.assign(_key=df["instruction"].map(dedupe_key))
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    stats["after_dedupe"] = len(df)

    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split by intent.

    Stratified means each intent keeps the same share in every split, so rare
    intents still show up in validation and test.
    """
    holdout = val_fraction + test_fraction

    def _split(frame: pd.DataFrame, size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        counts = frame["intent"].value_counts()
        n_holdout = math.ceil(len(frame) * size)
        # Stratifying needs at least 2 rows per intent and at least one holdout
        # row per intent; tiny datasets (like the unit tests) fall back to random.
        can_stratify = counts.min() >= 2 and n_holdout >= counts.size and len(frame) - n_holdout >= counts.size
        kept, held_out = train_test_split(
            frame,
            test_size=size,
            random_state=seed,
            stratify=frame["intent"] if can_stratify else None,
        )
        return kept, held_out

    train, rest = _split(df, holdout)
    val, test = _split(rest, test_fraction / holdout)
    return {
        "train": train.reset_index(drop=True),
        "validation": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def to_record(row: pd.Series, split_name: str, index: int) -> dict[str, Any]:
    """One training example in TRL's conversational prompt-completion format.

    "prompt" is what the model reads (system + user turn); "completion" is what
    it must learn to write. The loss is computed on the completion only, so the
    model is not trained to repeat the system prompt or the customer's message.
    """
    return {
        "id": f"{split_name}-{index:05d}",
        "intent": row["intent"],
        "category": row["category"],
        "flags": row["flags"],
        "prompt": build_prompt_messages(row["instruction"]),
        "completion": [{"role": "assistant", "content": row["response"]}],
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--source", help="override data.source (Hub id or local file)")
    parser.add_argument("--out-dir", help="override data.out_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    source = args.source or dcfg["source"]
    out_dir = Path(args.out_dir or dcfg["out_dir"])

    raw = load_raw(source)
    df, stats = clean(raw, dcfg)
    splits = split(df, dcfg["val_fraction"], dcfg["test_fraction"], cfg["seed"])

    for name, frame in splits.items():
        stats[name] = write_jsonl(
            out_dir / f"{name}.jsonl",
            (to_record(row, name, i) for i, row in frame.iterrows()),
        )
    stats["n_intents"] = int(df["intent"].nunique())
    write_json(out_dir / "stats.json", stats)

    print("Data preparation summary")
    for key, value in stats.items():
        print(f"  {key:>22}: {value}")
    return stats


if __name__ == "__main__":
    main()
