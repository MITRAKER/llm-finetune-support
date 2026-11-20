"""Tests for shared helpers and for train/serve consistency."""

from __future__ import annotations

import os
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
    with pytest.raises(ValueError, match="under 1,500 characters"):
        app.build_messages("x" * (app.MAX_MESSAGE_CHARS + 1), [])


def test_build_messages_reads_the_gradio_6_history_format():
    """Gradio 6 sends earlier messages as lists of parts; the second turn must still see the first."""
    reply = "Sure. Call +1 (555) 010-0199 about [Order Number]."  # as the app showed it
    history = [
        {"role": "user", "metadata": None, "content": [{"text": "I want to cancel my order", "type": "text"}]},
        {"role": "assistant", "content": [{"text": reply, "type": "text"}]},
        {"role": "user", "content": [{"type": "file", "file": {"path": "photo.png"}}]},  # no text: skipped
    ]
    msgs = app.build_messages("It's 12345", history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "I want to cancel my order"
    # The model sees its own reply with the placeholders it wrote, not the values the app filled in.
    assert msgs[2]["content"] == "Sure. Call {{Customer Support Phone Number}} about {{Order Number}}."


def test_history_is_bounded_by_characters_too():
    long_turn = "x" * (app.HISTORY_CHAR_BUDGET - 10)
    history = [{"role": "user", "content": "old question"}, {"role": "assistant", "content": long_turn}]
    msgs = app.build_messages("next", history)
    assert [m["content"] for m in msgs[1:]] == [long_turn, "next"]  # the older turn no longer fits


def test_history_is_trimmed_to_the_context_window_in_real_tokens():
    """Characters are not enough: Chinese or a list of order numbers takes far more tokens per character."""

    def one_token_per_character(text: str) -> int:
        return len(text)

    system = {"role": "system", "content": "s" * 100}
    older = [{"role": "user", "content": "a" * 900}, {"role": "assistant", "content": "b" * 400}]
    latest = {"role": "user", "content": "c" * 300}
    messages = [system, *older, latest]
    # The oldest turn no longer fits next to a full reply; the newer one still does.
    assert app.fit_to_context(messages, one_token_per_character) == [system, older[1], latest]
    assert app.fit_to_context(messages, one_token_per_character, budget=10_000) == messages
    too_long = {"role": "user", "content": "c" * app.PROMPT_TOKEN_BUDGET}
    with pytest.raises(ValueError, match="shorter message"):
        app.fit_to_context([system, too_long], one_token_per_character)
    assert app.PROMPT_TOKEN_BUDGET + app.MAX_NEW_TOKENS < app.CONTEXT_TOKENS  # room for the whole reply


def test_restore_placeholders_undoes_fill_placeholders():
    text = "Call {{Customer Support Phone Number}} about {{Order Number}}."
    assert app.restore_placeholders(app.fill_placeholders(text)) == text


def test_use_one_gpu_respects_an_existing_choice_and_distributed_launchers(monkeypatch: pytest.MonkeyPatch):
    for name in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    common.use_one_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    common.use_one_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("WORLD_SIZE", "2")  # torchrun decides which GPU each process uses
    common.use_one_gpu()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
