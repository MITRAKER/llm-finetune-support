"""Tests for shared helpers and for train/serve consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

import common
from app import app


def test_app_uses_the_training_system_prompt():
    """Train/serve skew guard: the demo must send the exact prompt the model was trained with."""
    assert app.SYSTEM_PROMPT == common.SYSTEM_PROMPT


def test_build_prompt_messages():
    assert common.build_prompt_messages("hi") == [
        {"role": "system", "content": common.SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
    ]


def test_jsonl_round_trip(tmp_path: Path):
    rows = [{"a": 1, "text": "héllo"}, {"a": 2, "text": "{{Order Number}}"}]
    assert common.write_jsonl(tmp_path / "x" / "rows.jsonl", rows) == 2
    assert common.read_jsonl(tmp_path / "x" / "rows.jsonl") == rows


def test_load_config_and_write_json(tmp_path: Path):
    cfg = common.load_config("configs/qwen3_4b_qlora.yaml")
    assert cfg["lora"]["r"] > 0
    assert cfg["data"]["val_fraction"] < 0.5
    common.write_json(tmp_path / "c.json", {"ok": True})
    assert (tmp_path / "c.json").read_text(encoding="utf-8").strip() == '{\n  "ok": true\n}'


def test_set_seed_is_repeatable():
    import random

    common.set_seed(7)
    first = random.random()
    common.set_seed(7)
    assert random.random() == first


def test_fill_placeholders():
    text = "Call {{Customer Support Phone Number}} about {{ Order Number }}."
    assert app.fill_placeholders(text) == "Call +1 (555) 010-0199 about [Order Number]."


def test_build_messages_keeps_recent_text_turns_only():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    history.append({"role": "assistant", "content": {"path": "image.png"}})  # non-text content is skipped
    msgs = app.build_messages("new question", history)
    assert msgs[0] == {"role": "system", "content": app.SYSTEM_PROMPT}
    assert msgs[-1] == {"role": "user", "content": "new question"}
    assert len(msgs) == 1 + (app.MAX_HISTORY_MESSAGES - 1) + 1
    with pytest.raises(ValueError, match="empty"):
        app.build_messages("   ", [])
