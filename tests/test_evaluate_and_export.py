"""Unit tests for the evaluation gates/report, the training config and the export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import evaluate
import export
import judge as judge_mod
import train
from common import SYSTEM_PROMPT, load_config, write_jsonl

GATES = {
    "min_intent_accuracy": 0.85,
    "min_intent_accuracy_gain": 0.05,
    "max_invented_detail_rate": 0.05,
    "min_rouge_l_gain": 0.05,
    "max_oos_invented_detail_increase": 0.05,
    "max_oos_judge_acceptable_drop": 0.10,
}


def report(
    intent: float,
    rouge: float,
    invented: float,
    judge: dict[str, float] | None = None,
    oos_invented: float = 0.0417,
    oos_judge: dict[str, float] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_fingerprint": "abc123",
        "metrics": {
            "n": 300,
            "intent_accuracy": intent,
            "rouge_l": rouge,
            "invented_detail_rate": invented,
            "placeholder_rate": 0.5,
            "avg_words": 80.0,
        },
        "intent_classifier_reference_accuracy": 0.97,
        "out_of_scope": {"n": 24, "invented_detail_rate": oos_invented},
    }
    if judge or oos_judge:
        out["judge"] = {"in_domain": judge or {}, "out_of_scope": oos_judge or {}}
    return out


def test_gates_pass_for_a_clear_improvement():
    assert evaluate.check_gates(report(0.70, 0.20, 0.30), report(0.93, 0.45, 0.01), GATES) == []


def test_gates_report_every_failure():
    failures = evaluate.check_gates(report(0.90, 0.40, 0.0), report(0.80, 0.41, 0.10), GATES)
    assert len(failures) == 4
    assert failures[0].startswith("intent accuracy 0.800")


def test_out_of_scope_gates_catch_a_safety_regression():
    base = report(0.70, 0.20, 0.30, oos_invented=0.0417, oos_judge={"judge_acceptable_rate": 0.9})
    one_more = report(0.93, 0.45, 0.01, oos_invented=0.0833, oos_judge={"judge_acceptable_rate": 0.85})
    assert evaluate.check_gates(base, one_more, GATES) == []  # one more of 24 is within the allowance
    worse = report(0.93, 0.45, 0.01, oos_invented=0.2083, oos_judge={"judge_acceptable_rate": 0.6})
    failures = evaluate.check_gates(base, worse, GATES)
    assert len(failures) == 2
    assert failures[0].startswith("out-of-scope invented detail rate 0.208 vs 0.042")
    assert failures[1].startswith("out-of-scope replies the judge accepts 0.600 vs 0.900")
    # Without the judge on both runs, only the automatic out-of-scope gate applies.
    assert len(evaluate.check_gates(report(0.70, 0.20, 0.30), worse, GATES)) == 1


def test_compare_refuses_results_from_different_test_examples(tmp_path: Path):
    cfg = {"eval": {"results_dir": str(tmp_path), "gates": GATES}}
    other = {**report(0.93, 0.45, 0.01), "sample_fingerprint": "def456"}
    (tmp_path / "base_metrics.json").write_text(json.dumps(report(0.7, 0.2, 0.3)), encoding="utf-8")
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="different prompts"):
        evaluate.compare(cfg)


def test_exit_code_for_failed_gates_differs_from_usage_errors():
    assert evaluate.EXIT_GATES_FAILED not in (0, 1, 2)  # 1: crash, 2: argparse usage error
    with pytest.raises(SystemExit) as usage_error:
        evaluate.main(["--model", "base"])  # --config is missing
    assert usage_error.value.code == 2


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
    assert "| Out-of-scope prompts: invented-detail rate (lower is better) | 0.042 | 0.042 | +0.000 |" in text
    assert "24 out-of-scope prompts" in text

    (tmp_path / "tuned_metrics.json").write_text(json.dumps(report(0.5, 0.1, 0.5)), encoding="utf-8")
    assert evaluate.compare(cfg) == evaluate.EXIT_GATES_FAILED
    assert "FAIL" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def tiny_generation(tmp_path: Path) -> tuple[dict[str, Any], evaluate.Generated]:
    """A config, training answers for the intent classifier, and the replies of one evaluation run."""
    answers = {
        "cancel_order": ["I have cancelled order {{Order Number}}", "Your cancellation is confirmed", "We cancel it"],
        "get_refund": ["Your refund has been issued", "Refunds take five days", "I will process the refund"],
    }
    rows = [
        {"id": f"{intent}-{i}", "intent": intent, "completion": [{"role": "assistant", "content": text}]}
        for intent, texts in answers.items()
        for i, text in enumerate(texts * 2)
    ]
    write_jsonl(tmp_path / "data" / "train.jsonl", rows)
    examples = [
        {
            "id": "t-1",
            "intent": "get_refund",
            "prompt": [{"role": "user", "content": "refund please"}],
            "completion": [{"role": "assistant", "content": "Your refund has been issued"}],
        }
    ]
    oos = [{"id": "oos-1", "type": "off_topic", "prompt": "write a poem", "expected": "decline politely"}]
    cfg = {
        "data": {"out_dir": str(tmp_path / "data")},
        "seed": 1,
        "eval": {"results_dir": str(tmp_path / "results"), "judge": {"model": "m", "effort": "low", "max_workers": 2}},
    }
    return cfg, evaluate.Generated(examples, ["Your refund was issued"], oos, ["Sorry, I can only help with orders."])


def test_replies_are_saved_before_the_judge_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, gen = tiny_generation(tmp_path)
    monkeypatch.setattr(evaluate, "generate_all", lambda *_: gen)

    def judge_down(*_: Any) -> None:
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(evaluate, "run_judge", judge_down)
    with pytest.raises(RuntimeError, match="judge unavailable"):
        evaluate.evaluate_model(cfg, "tuned", "hf", None, use_judge=True)
    saved = json.loads((tmp_path / "results" / "tuned_metrics.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["n"] == 1  # the slow generation is not lost
    assert saved["out_of_scope"]["n"] == 1
    assert len((tmp_path / "results" / "tuned_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_run_judge_reports_in_domain_and_out_of_scope_separately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, gen = tiny_generation(tmp_path)
    good = {"correctness": 5, "helpfulness": 5, "tone": 5, "invented_details": False, "acceptable": True}

    def fake_judge_all(items: list[Any], model: str, *_: Any) -> tuple[list[Any], Any]:
        results = [judge_mod.JudgeResult(it.item_id, good if it.reference else None, None, 100, 10) for it in items]
        return results, judge_mod.summarize(results, model)

    monkeypatch.setattr(evaluate.judge_mod, "judge_all", fake_judge_all)
    report_, rows = evaluate.run_judge(cfg, gen)
    assert report_["in_domain"]["judge_acceptable_rate"] == 1.0
    assert report_["out_of_scope"] == {}  # its only verdict failed
    assert report_["errors"] == 1
    assert [r["id"] for r in rows] == ["t-1", "oos-1"]


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


def test_count_parameters_uses_the_peft_counter_for_4bit_models():
    class FakePeftModel:  # PEFT counts 4-bit weights correctly (each byte holds two)
        def get_nb_trainable_parameters(self) -> tuple[int, int]:
            return 33_030_144, 4_055_498_240

    params = train.count_parameters(FakePeftModel())
    assert params["trainable"] == 33_030_144
    assert params["trainable_pct"] == pytest.approx(0.8145, abs=1e-4)

    class Param:
        def __init__(self, n: int, trainable: bool) -> None:
            self.n, self.requires_grad = n, trainable

        def numel(self) -> int:
            return self.n

    class PlainModel:
        def parameters(self) -> list[Param]:
            return [Param(90, False), Param(10, True)]

    assert train.count_parameters(PlainModel()) == {"trainable": 10, "total": 100, "trainable_pct": 10.0}


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
    with pytest.raises(FileNotFoundError, match="No q5_k_m GGUF file in"):
        export.find_gguf([*dirs, tmp_path / "missing"], "q5_k_m")


def test_select_gguf_picks_the_quantized_file_from_the_converter_output():
    files = ["out/m_gguf/Qwen3-4B.F16.gguf", "out/m_gguf/Qwen3-4B.Q4_K_M.gguf", "out/m_gguf/Modelfile"]
    assert export.select_gguf(files, "q4_k_m") == Path("out/m_gguf/Qwen3-4B.Q4_K_M.gguf")
    with pytest.raises(FileNotFoundError, match=r"Qwen3-4B\.F16\.gguf"):
        export.select_gguf(files, "q8_0")


def test_gguf_file_name_is_stable_and_matches_the_app_pattern():
    import fnmatch

    from app import app

    cfg = load_config("configs/qwen3_4b_qlora.yaml")
    assert export.gguf_file_name(cfg) == "qwen3-4b-support.Q4_K_M.gguf"  # named after the run, not the base model
    cfg["export"]["hub_model_id"] = "someone/support-bot"
    name = export.gguf_file_name(cfg)
    assert name == "support-bot.Q4_K_M.gguf"
    assert fnmatch.fnmatchcase(name, app.MODEL_FILE)
    assert export.exported_gguf(cfg) == Path(cfg["export"]["output_dir"]) / name


def test_push_settings_are_checked_before_any_slow_work(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config("configs/qwen3_4b_qlora.yaml")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with pytest.raises(ValueError, match="hub_model_id"):
        export.push_settings(cfg)  # the shipped config has no Hub repo
    cfg["export"]["hub_model_id"] = "someone/support-bot"
    assert export.push_settings(cfg) == ("someone/support-bot", "hf_test")
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        export.push_settings(cfg)


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
    assert SYSTEM_PROMPT in card  # users need the exact training prompt
    assert "{{system_prompt}}" not in card
