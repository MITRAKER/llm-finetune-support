"""Unit tests for the evaluation gates/report, the training config and the export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import evaluate
import export
import train
from common import load_config

GATES = {
    "min_intent_accuracy": 0.85,
    "min_intent_accuracy_gain": 0.05,
    "max_invented_detail_rate": 0.05,
    "min_rouge_l_gain": 0.05,
}


def report(intent: float, rouge: float, invented: float, judge: dict[str, float] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metrics": {
            "n": 300,
            "intent_accuracy": intent,
            "rouge_l": rouge,
            "invented_detail_rate": invented,
            "placeholder_rate": 0.5,
            "avg_words": 80.0,
        },
        "intent_classifier_ceiling": 0.97,
    }
    if judge:
        out["judge"] = {"in_domain": judge}
    return out


def test_gates_pass_for_a_clear_improvement():
    assert evaluate.check_gates(report(0.70, 0.20, 0.30), report(0.93, 0.45, 0.01), GATES) == []


def test_gates_report_every_failure():
    failures = evaluate.check_gates(report(0.90, 0.40, 0.0), report(0.80, 0.41, 0.10), GATES)
    assert len(failures) == 4
    assert failures[0].startswith("intent accuracy 0.800")


def test_compare_writes_report_and_returns_exit_code(tmp_path: Path):
    cfg = {"eval": {"results_dir": str(tmp_path), "gates": GATES}}
    judge_scores = {"judge_correctness": 3.1, "judge_helpfulness": 3.0, "judge_tone": 4.0, "judge_acceptable_rate": 0.4}
    better = {**judge_scores, "judge_correctness": 4.5}
    (tmp_path / "base_metrics.json").write_text(json.dumps(report(0.7, 0.2, 0.3, judge_scores)), encoding="utf-8")
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(report(0.93, 0.45, 0.01, better)), encoding="utf-8")
    assert evaluate.compare(cfg) == 0
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "PASS" in text
    assert "| Judge correctness (LLM judge) | 3.100 | 4.500 | +1.400 |" in text

    (tmp_path / "tuned_metrics.json").write_text(json.dumps(report(0.5, 0.1, 0.5)), encoding="utf-8")
    assert evaluate.compare(cfg) == evaluate.EXIT_GATES_FAILED
    assert "FAIL" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_load_eval_examples_validates_n(tmp_path: Path):
    (tmp_path / "test.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    assert [r["id"] for r in evaluate.load_eval_examples(tmp_path, 1)] == ["a"]
    with pytest.raises(ValueError, match="positive"):
        evaluate.load_eval_examples(tmp_path, 0)


def test_load_for_inference_checks_arguments(tmp_path: Path):
    cfg = {"train": {"output_dir": str(tmp_path)}, "model": {"name": "x", "load_in_4bit": False}}
    with pytest.raises(ValueError, match="'base' or 'tuned'"):
        evaluate.load_for_inference(cfg, "other", "hf", None)
    with pytest.raises(FileNotFoundError, match="Run step 3 first"):
        evaluate.load_for_inference(cfg, "tuned", "hf", None)


def test_sft_config_kwargs_from_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(train, "precision_flags", lambda: {"bf16": False, "fp16": True})  # a T4
    cfg = load_config("configs/qwen3_4b_qlora.yaml")
    kwargs = train.build_sft_config_kwargs(cfg, "unsloth", max_steps=30)
    assert kwargs["max_steps"] == 30
    assert kwargs["completion_only_loss"] is True
    assert kwargs["gradient_checkpointing"] is False  # Unsloth handles it
    assert kwargs["report_to"] == "none"
    assert kwargs["max_length"] == cfg["model"]["max_seq_length"]
    assert (kwargs["fp16"], kwargs["bf16"]) == (True, False)
    assert kwargs["save_steps"] % kwargs["eval_steps"] == 0  # needed by load_best_model_at_end
    assert train.build_sft_config_kwargs(cfg, "hf")["gradient_checkpointing"] is True


def test_report_to_auto(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_API_KEY", "x")
    assert train.resolve_report_to("auto") == "wandb"
    assert train.resolve_report_to("none") == "none"


def test_load_model_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        train.load_model({}, "tensorflow")


def test_load_splits_requires_step_one(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Run step 1 first"):
        train.load_splits(tmp_path, None, None, 0)


def test_find_gguf_looks_next_to_and_inside_the_merge_folder(tmp_path: Path):
    merged = tmp_path / "merged"
    (tmp_path / "merged_gguf").mkdir()
    (tmp_path / "merged_gguf" / "Qwen3-4B.Q4_K_M.gguf").write_bytes(b"x")  # Unsloth 2026.x layout
    merged.mkdir()
    (merged / "model.Q8_0.gguf").write_bytes(b"x")  # older layout
    dirs = export.gguf_search_dirs(merged)
    assert export.find_gguf(dirs, "q4_k_m").name == "Qwen3-4B.Q4_K_M.gguf"
    assert export.find_gguf(dirs, "q8_0").name == "model.Q8_0.gguf"
    with pytest.raises(FileNotFoundError, match="Found"):
        export.find_gguf([*dirs, tmp_path / "missing"], "q5_k_m")


def test_hub_gguf_name_matches_the_app_pattern():
    import fnmatch

    from app import app

    name = export.hub_gguf_name("someone/support-bot", "q4_k_m")
    assert name == "support-bot.Q4_K_M.gguf"
    assert fnmatch.fnmatchcase(name, app.MODEL_FILE)
    assert export.hub_gguf_name(None, "q8_0") == "model.Q8_0.gguf"


def test_model_card_is_filled(tmp_path: Path):
    cfg = load_config("configs/qwen3_4b_qlora.yaml")
    cfg["data"]["out_dir"] = str(tmp_path)
    cfg["export"]["hub_model_id"] = "someone/support-bot"
    (tmp_path / "stats.json").write_text('{"train": 123}', encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("# Evaluation report\n\n| a | b |", encoding="utf-8")
    card = export.render_model_card(cfg, tmp_path)
    assert "someone/support-bot" in card
    assert "123" in card
    assert "| a | b |" in card
    assert "{{base_model}}" not in card
    assert "{{gguf_file}}" not in card
    assert "support-bot.Q4_K_M.gguf" in card
    assert "{{Order Number}}" in card  # the store placeholders in the card text are left alone
