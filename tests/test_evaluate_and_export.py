"""Unit tests for the evaluation gates/report, the training config and the export helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import evaluate
import export
import judge as judge_mod
import train
from common import SYSTEM_PROMPT, adapter_digest, adapter_path, load_config, write_jsonl

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


def test_judge_scores_are_compared_only_when_the_same_judge_graded_both_runs(tmp_path: Path):
    scores = {"judge_correctness": 4.0, "judge_helpfulness": 4.0, "judge_tone": 4.0, "judge_acceptable_rate": 0.9}
    base = report(0.7, 0.2, 0.3, scores, oos_judge={"judge_acceptable_rate": 0.95})
    tuned = report(0.93, 0.45, 0.01, scores, oos_judge={"judge_acceptable_rate": 0.80})
    base["judge"].update(model="claude-opus-5-5", effort="low")
    tuned["judge"].update(model="claude-haiku-5-5", effort="low")  # a cheaper judge for the second run
    assert evaluate.check_gates(base, tuned, GATES) == []  # no judge gate across two judges
    (tmp_path / "base_metrics.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(tuned), encoding="utf-8")
    assert evaluate.compare({"eval": {"results_dir": str(tmp_path), "gates": GATES}}) == 0
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "Judge correctness" not in text
    assert "base model was graded by claude-opus-5-5, effort low and the fine-tuned model by claude-haiku-5-5" in text
    tuned["judge"]["model"] = "claude-opus-5-5"  # the same judge for both runs: the gate applies
    assert evaluate.check_gates(base, tuned, GATES) == [
        "out-of-scope replies the judge accepts 0.800 vs 0.950 for the base model (allowed drop 0.1)"
    ]
    tuned["judge"]["effort"] = "high"  # the same model at another effort is another judge
    assert evaluate.check_gates(base, tuned, GATES) == []
    del base["judge"]["effort"]  # results written before the judge's effort was recorded
    note = evaluate.judge_note(base, tuned)
    assert note is not None
    assert note.endswith("The base model's results do not record the judge's effort: evaluate that model again.")


def test_the_report_says_when_the_judge_graded_one_run_only_or_failed(tmp_path: Path):
    """Without this note, the judge rows and the out-of-scope judge gate would vanish in silence."""
    scores = {"judge_correctness": 4.0, "judge_acceptable_rate": 0.9}
    base = report(0.7, 0.2, 0.3, scores, oos_judge={"judge_acceptable_rate": 0.95})
    base["judge"].update(model="m", effort="low", errors=0)
    tuned = report(0.93, 0.45, 0.01)  # step 4 ran without the judge
    assert evaluate.judge_note(base, tuned) == (
        "The LLM judge scores are not compared: only the base model was graded by the judge. Evaluate the "
        "fine-tuned model with the judge too (it runs when ANTHROPIC_API_KEY is set) to compare them."
    )
    # Every judge request for the fine-tuned model failed, for example because the API credit ran out.
    tuned["judge"] = {"model": "m", "effort": "low", "errors": 324, "in_domain": {}, "out_of_scope": {}}
    assert evaluate.check_gates(base, tuned, GATES) == []  # the judge gate cannot be applied
    note = (
        "The judge could not grade 324 of the fine-tuned model's 324 replies, so the out-of-scope judge gate "
        "was not applied. Evaluate the fine-tuned model again to grade them."
    )
    assert evaluate.judge_note(base, tuned) == note
    (tmp_path / "base_metrics.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(tuned), encoding="utf-8")
    assert evaluate.compare({"eval": {"results_dir": str(tmp_path), "gates": GATES}}) == 0
    assert note in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    tuned["judge"] = {**base["judge"], "errors": 3}  # a few failures: the scores leave them out
    assert evaluate.judge_note(base, tuned) == (
        "The judge could not grade 3 of the fine-tuned model's 324 replies, and its scores leave them out. "
        "Evaluate the fine-tuned model again to grade them."
    )
    assert evaluate.judge_note(base, base) is None
    assert evaluate.judge_note(report(0.7, 0.2, 0.3), report(0.93, 0.45, 0.01)) is None  # no judge at all


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
        "train": {"output_dir": str(tmp_path / "lora")},
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


def fake_adapter(cfg: dict[str, Any], weights: bytes) -> str:
    """Save LoRA weights where step 3 saves them, and return their digest."""
    folder = adapter_path(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_model.safetensors").write_bytes(weights)
    digest = adapter_digest(folder)
    assert digest is not None
    return digest


def test_the_fine_tuned_results_record_which_adapter_they_describe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, gen = tiny_generation(tmp_path)
    monkeypatch.setattr(evaluate, "generate_all", lambda *_: gen)
    digest = fake_adapter(cfg, b"weights v1")
    assert evaluate.evaluate_model(cfg, "tuned", "hf", None, use_judge=False)["adapter_digest"] == digest
    assert "adapter_digest" not in evaluate.evaluate_model(cfg, "base", "hf", None, use_judge=False)
    other = tmp_path / "other_adapter"  # evaluate.py --adapter names another folder
    other.mkdir()
    (other / "adapter_model.safetensors").write_bytes(b"weights v2")
    report_ = evaluate.evaluate_model(cfg, "tuned", "hf", str(other), use_judge=False)
    assert report_["adapter_digest"] == adapter_digest(other) != digest
    saved = json.loads((tmp_path / "results" / "tuned_metrics.json").read_text(encoding="utf-8"))
    assert saved["adapter_digest"] == report_["adapter_digest"]


def test_run_judge_reports_in_domain_and_out_of_scope_separately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, gen = tiny_generation(tmp_path)
    good = {"correctness": 5, "helpfulness": 5, "tone": 5, "invented_details": False, "acceptable": True}

    def fake_judge_all(items: list[Any], model: str, *_: Any) -> tuple[list[Any], Any]:
        results = [judge_mod.JudgeResult(it.item_id, good if it.reference else None, None, 100, 10) for it in items]
        return results, judge_mod.summarize(results, model)

    monkeypatch.setattr(evaluate.judge_mod, "judge_all", fake_judge_all)
    report_, rows = evaluate.run_judge(cfg, gen)
    assert (report_["model"], report_["effort"]) == ("m", "low")  # compare checks that both runs had one judge
    assert report_["in_domain"]["judge_acceptable_rate"] == 1.0
    assert report_["out_of_scope"] == {}  # its only verdict failed
    assert report_["errors"] == 1
    assert [r["id"] for r in rows] == ["t-1", "oos-1"]


def test_fingerprint_changes_when_a_question_changes_under_the_same_id(tmp_path: Path):
    """Ids only number the rows: after a data change the same ids hold other questions."""
    _, gen = tiny_generation(tmp_path)
    original = evaluate.sample_fingerprint(gen)
    assert evaluate.sample_fingerprint(copy.deepcopy(gen)) == original  # stable
    changes = [
        lambda g: g.examples[0]["prompt"][0].update(content="where is my refund"),
        lambda g: g.examples[0]["completion"][0].update(content="Refunds take a week"),
        lambda g: g.examples[0].update(intent="cancel_order"),
        lambda g: g.oos[0].update(prompt="write a song"),
        lambda g: g.oos[0].update(expected="answer it"),
    ]
    for change in changes:
        changed = copy.deepcopy(gen)
        change(changed)
        assert evaluate.sample_fingerprint(changed) != original


def test_score_records_what_compare_checks(tmp_path: Path):
    cfg, gen = tiny_generation(tmp_path)
    cfg["eval"]["max_new_tokens"] = 64
    first = evaluate.score(cfg, "base", gen).report
    assert first["sample_fingerprint"] == evaluate.sample_fingerprint(gen)
    assert first["max_new_tokens"] == 64
    # The intent classifier learns from the training split, so new training data changes
    # how intent accuracy is measured, even on the same test questions.
    train_file = tmp_path / "data" / "train.jsonl"
    train_file.write_text("".join(train_file.read_text(encoding="utf-8").splitlines(keepends=True)[1:]), "utf-8")
    second = evaluate.score(cfg, "base", gen).report
    assert second["sample_fingerprint"] == first["sample_fingerprint"]
    assert second["classifier_fingerprint"] != first["classifier_fingerprint"]


def test_compare_refuses_a_baseline_scored_before_the_data_or_settings_changed(tmp_path: Path):
    cfg = {"eval": {"results_dir": str(tmp_path), "gates": GATES}}
    base = {**report(0.7, 0.2, 0.3), "classifier_fingerprint": "old", "max_new_tokens": 256}
    tuned = {**report(0.93, 0.45, 0.01), "classifier_fingerprint": "new", "max_new_tokens": 512}
    (tmp_path / "base_metrics.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(tuned), encoding="utf-8")
    with pytest.raises(ValueError, match="different training split") as refused:
        evaluate.compare(cfg)
    assert "eval.max_new_tokens: base 256, tuned 512" in str(refused.value)
    assert "re-run step 2, the baseline" in str(refused.value)
    assert not (tmp_path / "REPORT.md").exists()


def test_load_eval_examples_validates_n(tmp_path: Path):
    (tmp_path / "test.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    assert [r["id"] for r in evaluate.load_eval_examples(tmp_path, 1)] == ["a"]
    with pytest.raises(ValueError, match="positive"):
        evaluate.load_eval_examples(tmp_path, 0)


def test_eval_examples_come_from_the_chosen_split_and_never_from_train(tmp_path: Path):
    for split in ("train", "validation", "test"):
        write_jsonl(tmp_path / f"{split}.jsonl", [{"id": f"{split}-00000"}])
    assert evaluate.load_eval_examples(tmp_path, 5, "validation") == [{"id": "validation-00000"}]
    assert evaluate.load_eval_examples(tmp_path, 5) == [{"id": "test-00000"}]
    with pytest.raises(ValueError, match=r"eval\.split must be one of validation, test"):
        evaluate.load_eval_examples(tmp_path, 5, "train")
    assert evaluate.eval_split({"eval": {}}) == "test"
    assert evaluate.eval_split({"eval": {"split": "validation"}}) == "validation"


def test_split_option_overrides_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(evaluate, "compare", lambda cfg: seen.update(cfg["eval"]) or 0)
    config = tmp_path / "c.yaml"
    config.write_text("eval:\n  split: test\n  n_samples: 300\n", encoding="utf-8")
    assert evaluate.main(["--config", str(config), "--compare", "--split", "validation", "--n-samples", "50"]) == 0
    assert seen["split"] == "validation"
    assert seen["n_samples"] == 50
    with pytest.raises(SystemExit):
        evaluate.main(["--config", str(config), "--compare", "--split", "train"])


def test_report_names_the_split_and_compare_refuses_to_mix_splits(tmp_path: Path):
    cfg = {"eval": {"results_dir": str(tmp_path), "gates": GATES}}
    base = {**report(0.7, 0.2, 0.3), "split": "validation"}
    tuned = {**report(0.93, 0.45, 0.01), "split": "validation"}
    (tmp_path / "base_metrics.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(tuned), encoding="utf-8")
    assert evaluate.compare(cfg) == 0
    assert "Validation examples: 300" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    on_test = {**tuned, "split": "test", "sample_fingerprint": "def456"}
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(on_test), encoding="utf-8")
    with pytest.raises(ValueError, match="300 validation examples, tuned: 300 test examples"):
        evaluate.compare(cfg)


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


def test_hub_repos_are_created_and_checked_for_write_access():
    """create_repo(exist_ok=True) succeeds for a read-only token when the repo exists, so check explicitly."""
    calls: list[tuple[str, str, bool]] = []

    class FakeApi:
        def create_repo(self, repo_id: str, exist_ok: bool) -> None:
            calls.append(("create", repo_id, exist_ok))

        def auth_check(self, repo_id: str, write: bool) -> None:
            calls.append(("auth_check", repo_id, write))
            if repo_id.endswith("-GGUF"):
                raise PermissionError("403: the token cannot write here")

    with pytest.raises(PermissionError, match="403"):
        export.create_hub_repos(FakeApi(), "someone/support-bot")
    assert calls == [
        ("create", "someone/support-bot-lora", True),
        ("auth_check", "someone/support-bot-lora", True),
        ("create", "someone/support-bot-GGUF", True),
        ("auth_check", "someone/support-bot-GGUF", True),
    ]


def save_and_compare(results: Path, split: str, adapter: str | None = None) -> None:
    """Step 2 and step 4 on one split: save both metrics files, then compare them.

    adapter is the digest of the adapter the fine-tuned model was evaluated with
    (None: the results come from a version of evaluate.py that did not record it).
    """
    for name, intent in (("base", 0.7), ("tuned", 0.93)):
        result = {**report(intent, 0.45, 0.01), "split": split}
        if name == "tuned" and adapter:
            result["adapter_digest"] = adapter
        (results / f"{name}_metrics.json").write_text(json.dumps(result), encoding="utf-8")
    evaluate.compare({"eval": {"results_dir": str(results), "gates": GATES}})


def export_config(tmp_path: Path) -> dict[str, Any]:
    """The real config, with the results and the adapter in a temporary folder."""
    cfg = load_config("configs/qwen3_4b_qlora.yaml")
    cfg["eval"]["results_dir"] = str(tmp_path)
    cfg["train"]["output_dir"] = str(tmp_path / "lora")
    return cfg


def test_model_card_names_the_split_of_its_report_and_push_warns_until_it_is_final(tmp_path: Path):
    """The card's numbers come from REPORT.md, so its split must too: --split overrides the
    config, and a model evaluated again without a new comparison makes the metrics newer."""
    cfg = export_config(tmp_path)
    adapter = fake_adapter(cfg, b"weights v1")
    assert cfg["eval"]["split"] == "test"
    assert export.results_split(cfg, tmp_path) == "test"  # nothing compared yet: the config's split
    assert export.push_warnings(cfg) == [
        "Warning: there is no evaluation report yet, so the model card will have no numbers. Run step 4 first."
    ]
    save_and_compare(tmp_path, "validation", adapter)
    assert "held-out validation questions" in export.render_model_card(cfg, tmp_path)
    assert not export.report_is_stale(tmp_path)
    assert ["validation split" in w for w in export.push_warnings(cfg)] == [True]
    # The tuned model is evaluated on the test split, but the comparison is not run again.
    # REPORT.md still holds the validation numbers, so the card must still say validation.
    on_test = {**report(0.95, 0.5, 0.0), "split": "test", "adapter_digest": adapter}
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(on_test), encoding="utf-8")
    assert "held-out validation questions" in export.render_model_card(cfg, tmp_path)
    assert export.report_is_stale(tmp_path)
    assert "evaluated again after REPORT.md was written" in export.push_warnings(cfg)[0]
    save_and_compare(tmp_path, "test", adapter)
    assert "held-out test questions" in export.render_model_card(cfg, tmp_path)
    assert export.push_warnings(cfg) == []
    export.check_report(cfg)  # the report describes the adapter that --push would upload


def test_push_refuses_an_adapter_trained_again_after_the_report(tmp_path: Path):
    """Training again overwrites the adapter in place. The card would then show the previous
    adapter's numbers and verdict, so --push stops until the new one is evaluated and compared."""
    cfg = export_config(tmp_path)
    first = fake_adapter(cfg, b"weights v1")
    save_and_compare(tmp_path, "test", first)
    second = fake_adapter(cfg, b"weights v2")  # step 3 again, without step 4
    with pytest.raises(ValueError, match=r"changed after REPORT\.md was written"):
        export.check_report(cfg)
    tuned = {**report(0.6, 0.45, 0.01), "split": "test", "adapter_digest": second}
    (tmp_path / "tuned_metrics.json").write_text(json.dumps(tuned), encoding="utf-8")  # evaluated, not compared
    with pytest.raises(ValueError, match=r"evaluate\.py --model tuned, then evaluate\.py --compare"):
        export.check_report(cfg)
    save_and_compare(tmp_path, "test", second)
    export.check_report(cfg)
    assert export.push_warnings(cfg) == []


def test_a_report_is_stale_after_either_model_is_evaluated_again_until_a_comparison_succeeds(tmp_path: Path):
    save_and_compare(tmp_path, "test", "a1b2c3")
    assert not export.report_is_stale(tmp_path)
    # Step 2 again after a data change: the baseline was scored on other prompts.
    base = {**report(0.7, 0.45, 0.01), "split": "test", "sample_fingerprint": "def456"}
    (tmp_path / "base_metrics.json").write_text(json.dumps(base), encoding="utf-8")
    assert export.report_is_stale(tmp_path)
    info = (tmp_path / "report.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="different prompts"):
        evaluate.compare({"eval": {"results_dir": str(tmp_path), "gates": GATES}})
    assert (tmp_path / "report.json").read_text(encoding="utf-8") == info  # a refused comparison writes nothing
    assert export.report_is_stale(tmp_path)


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


def test_model_card_shows_the_data_steps_and_the_examples_training_really_used(tmp_path: Path):
    cfg = load_config("configs/qwen3_4b_qlora.yaml")
    cfg["data"]["out_dir"] = str(tmp_path / "data")
    cfg["train"]["output_dir"] = str(tmp_path / "lora")
    stats = {"raw": 26872, "after_length_filter": 26872, "after_dedupe": 24072, "train": 19257}
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "stats.json").write_text(json.dumps({**stats, "n_intents": 27}), encoding="utf-8")
    card = export.render_model_card(cfg, tmp_path)
    assert "| Bitext dataset | 26,872 |" in card
    assert "| After removing repeated questions | 24,072 |" in card
    assert "| Train / validation / test | 19,257 / n/a / n/a |" in card  # unknown counts say so
    assert "| Training examples | 19,257 |" in card
    assert "held-out test questions" in card
    # train.max_train_samples capped the run: the card must say what training really used.
    (tmp_path / "lora").mkdir()
    (tmp_path / "lora" / "training_summary.json").write_text('{"train_examples": 5000}', encoding="utf-8")
    cfg["eval"]["split"] = "validation"
    card = export.render_model_card(cfg, tmp_path)
    assert "| Training examples | 5,000 |" in card
    assert "held-out validation questions" in card
    assert "{{rows_" not in card
    assert "{{train_examples}}" not in card
    assert "{{eval_split}}" not in card
