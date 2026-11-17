"""Unit tests for step 1 (data preparation). They run offline in a second."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import prepare_data as pdp
from common import SYSTEM_PROMPT

CFG = {"min_instruction_chars": 5, "min_response_chars": 20, "max_response_chars": 3000}


def make_frame(n_per_intent: int = 30, intents=("cancel_order", "get_refund", "track_order")) -> pd.DataFrame:
    rows = [
        {
            "instruction": f"question {i} about {intent.replace('_', ' ')}",
            "response": f"Here is a helpful answer number {i} for {intent}, see {{{{Order Number}}}}.",
            "intent": intent,
            "category": "ORDER",
            "flags": "B",
        }
        for intent in intents
        for i in range(n_per_intent)
    ]
    return pd.DataFrame(rows)


def test_normalize_tidies_whitespace_and_unicode():
    assert pdp.normalize("  Hello  world \t!  ") == "Hello world !"
    assert pdp.normalize("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
    assert pdp.normalize("“quoted”") == "“quoted”"  # NFKC keeps curly quotes
    assert pdp.normalize("ＡＢＣ") == "ABC"  # full-width letters -> ASCII
    assert pdp.normalize(None) == ""
    assert pdp.normalize(float("nan")) == ""


def test_dedupe_key_ignores_case_and_punctuation():
    assert pdp.dedupe_key("Cancel my order!") == pdp.dedupe_key("cancel   my ORDER")
    assert pdp.dedupe_key("cancel my order") != pdp.dedupe_key("track my order")


def test_clean_filters_and_dedupes():
    df = make_frame(5, intents=("cancel_order",))
    extra = pd.DataFrame(
        [
            {
                "instruction": "hi",
                "response": "x" * 50,
                "intent": "cancel_order",
                "category": "",
                "flags": "",
            },  # too short
            {
                "instruction": "valid question",
                "response": "short",
                "intent": "cancel_order",
                "category": "",
                "flags": "",
            },  # short answer
            {
                "instruction": "valid question",
                "response": "y" * 5000,
                "intent": "cancel_order",
                "category": "",
                "flags": "",
            },  # too long
            {
                "instruction": "QUESTION 0 about cancel order!!",
                "response": "z" * 40,
                "intent": "cancel_order",
                "category": "",
                "flags": "",
            },  # duplicate
        ]
    )
    cleaned, stats = pdp.clean(pd.concat([df, extra], ignore_index=True), CFG)
    assert stats == {"raw": 9, "after_length_filter": 6, "after_dedupe": 5}
    assert len(cleaned) == 5
    # The first occurrence wins, so the original answer is kept.
    assert cleaned.loc[0, "response"].startswith("Here is a helpful answer number 0")


def test_split_is_stratified_disjoint_and_reproducible():
    df, _ = pdp.clean(make_frame(30), CFG)
    s1 = pdp.split(df, 0.1, 0.1, seed=42)
    s2 = pdp.split(df, 0.1, 0.1, seed=42)
    assert {k: len(v) for k, v in s1.items()} == {"train": 72, "validation": 9, "test": 9}
    for name in s1:
        pd.testing.assert_frame_equal(s1[name], s2[name])
    # Every intent appears in every split (stratification).
    for frame in s1.values():
        assert set(frame["intent"]) == {"cancel_order", "get_refund", "track_order"}
    # No question appears in two splits (no leakage).
    keys = {name: set(frame["instruction"].map(pdp.dedupe_key)) for name, frame in s1.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["validation"]
    assert not keys["validation"] & keys["test"]


def test_split_falls_back_to_random_for_tiny_data():
    df, _ = pdp.clean(make_frame(2), CFG)
    splits = pdp.split(df, 0.2, 0.2, seed=0)
    assert sum(len(v) for v in splits.values()) == len(df)


def test_record_format_matches_trl_prompt_completion():
    df, _ = pdp.clean(make_frame(2, intents=("cancel_order",)), CFG)
    rec = pdp.to_record(df.iloc[0], "train", 7)
    assert rec["id"] == "train-00007"
    assert rec["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": df.iloc[0]["instruction"]},
    ]
    assert rec["completion"] == [{"role": "assistant", "content": df.iloc[0]["response"]}]
    assert "messages" not in rec  # TRL would treat a "messages" column as a different format


def test_main_end_to_end_on_local_csv(tmp_path):
    src = tmp_path / "raw.csv"
    make_frame(30).to_csv(src, index=False)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "seed: 1\n"
        "data:\n"
        f"  source: {src}\n"
        f"  out_dir: {tmp_path / 'out'}\n"
        "  val_fraction: 0.1\n  test_fraction: 0.1\n"
        "  min_instruction_chars: 5\n  min_response_chars: 20\n  max_response_chars: 3000\n"
    )
    stats = pdp.main(["--config", str(cfg_path)])
    assert stats["train"] + stats["validation"] + stats["test"] == stats["after_dedupe"] == 90
    lines = (tmp_path / "out" / "test.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert set(first) == {"id", "intent", "category", "flags", "prompt", "completion"}
    assert json.loads((tmp_path / "out" / "stats.json").read_text())["n_intents"] == 3


def test_missing_columns_raise(tmp_path):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"instruction": ["a"]}).to_csv(src, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        pdp.load_raw(str(src))
