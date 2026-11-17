"""Steps 2 and 4: evaluate a model, then compare the tuned model with the base model.

Usage:
    # Step 2: baseline = the base model with the same system prompt (no fine-tuning)
    python src/evaluate.py --config configs/qwen3_4b_qlora.yaml --model base
    # Step 4: the fine-tuned model (base + LoRA adapter)
    python src/evaluate.py --config configs/qwen3_4b_qlora.yaml --model tuned
    # Compare the two and check the quality gates (exit code 2 = retrain)
    python src/evaluate.py --config configs/qwen3_4b_qlora.yaml --compare

Each evaluation writes results/<name>_predictions.jsonl (every prompt, reply and
score, for error analysis) and results/<name>_metrics.json (the averages).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import judge as judge_mod
import metrics
from common import build_prompt_messages, load_config, read_jsonl, set_seed, write_json, write_jsonl

EXIT_GATES_FAILED = 2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_eval_examples(data_dir: str | Path, n_samples: int) -> list[dict[str, Any]]:
    """The first n test examples (the test split is already shuffled)."""
    rows = read_jsonl(Path(data_dir) / "test.jsonl")
    if n_samples <= 0:
        raise ValueError("eval.n_samples must be positive")
    return rows[:n_samples]


def user_text(example: dict[str, Any]) -> str:
    return str(next(m["content"] for m in example["prompt"] if m["role"] == "user"))


def reference_text(example: dict[str, Any]) -> str:
    return str(example["completion"][0]["content"])


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------
def load_for_inference(cfg: dict[str, Any], which: str, backend: str, adapter_dir: str | None) -> tuple[Any, Any]:
    """Load the base model, or the base model plus the trained LoRA adapter."""
    if which not in ("base", "tuned"):
        raise ValueError(f"--model must be 'base' or 'tuned', got {which!r}")
    adapter = adapter_dir or str(Path(cfg["train"]["output_dir"]) / "final_adapter")
    if which == "tuned" and not Path(adapter).exists():
        raise FileNotFoundError(f"No adapter at {adapter}. Run step 3 first: python src/train.py --config ...")
    if backend == "unsloth":  # pragma: no cover - needs a CUDA GPU
        return _load_unsloth(cfg, adapter if which == "tuned" else None)
    return _load_hf(cfg, adapter if which == "tuned" else None)


def _load_unsloth(cfg: dict[str, Any], adapter: str | None) -> tuple[Any, Any]:  # pragma: no cover
    from unsloth import FastLanguageModel

    # Given an adapter folder, Unsloth reads the base model name from adapter_config.json
    # and loads base + adapter in one call.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter or cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    FastLanguageModel.for_inference(model)  # enables Unsloth's faster generation path
    return model, tokenizer


def _load_hf(cfg: dict[str, Any], adapter: str | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(adapter or name)
    kwargs: dict[str, Any] = {}
    if cfg["model"]["load_in_4bit"] and torch.cuda.is_available():  # pragma: no cover - GPU only
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_replies(
    model: Any,
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Greedy (deterministic) generation in batches.

    Prompts are sorted by length so each batch needs little padding, and
    padding goes on the LEFT so every prompt ends right where generation starts.
    """
    import torch
    from transformers import GenerationConfig

    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for conv in conversations
    ]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )
    replies: list[str] = [""] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_config)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for i, text in zip(idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True), strict=True):
            replies[i] = metrics.strip_reasoning(text)
        print(f"  generated {min(start + batch_size, len(order))}/{len(order)}")
    return replies


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def build_classifier(data_dir: str | Path) -> tuple[metrics.IntentClassifier, float]:
    """Train the intent classifier on training ANSWERS; report its accuracy on test references."""
    train_rows = read_jsonl(Path(data_dir) / "train.jsonl")
    clf = metrics.IntentClassifier().fit([reference_text(r) for r in train_rows], [r["intent"] for r in train_rows])
    test_rows = read_jsonl(Path(data_dir) / "test.jsonl")
    ceiling = clf.accuracy([reference_text(r) for r in test_rows], [r["intent"] for r in test_rows])
    return clf, round(ceiling, 4)


@dataclass
class Generated:
    """Everything one evaluation run generated, kept together for scoring and saving."""

    examples: list[dict[str, Any]]  # in-domain test examples
    replies: list[str]
    oos: list[dict[str, Any]]  # out-of-scope prompts
    oos_replies: list[str]


def run_judge(cfg: dict[str, Any], gen: Generated) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grade in-domain and out-of-scope replies with the LLM judge."""
    jcfg = cfg["eval"]["judge"]
    items = [
        judge_mod.JudgeItem(ex["id"], user_text(ex), rep, reference=reference_text(ex))
        for ex, rep in zip(gen.examples, gen.replies, strict=True)
    ]
    items += [
        judge_mod.JudgeItem(o["id"], o["prompt"], rep, expected=o["expected"])
        for o, rep in zip(gen.oos, gen.oos_replies, strict=True)
    ]
    results, summary = judge_mod.judge_all(items, jcfg["model"], jcfg["effort"], jcfg["max_workers"])
    by_id = {r.item_id: r for r in results}
    in_domain = [by_id[ex["id"]] for ex in gen.examples]
    out_scope = [by_id[o["id"]] for o in gen.oos]
    report = {
        "model": jcfg["model"],
        "errors": summary.n_errors,
        "est_cost_usd": summary.est_cost_usd,
        "in_domain": judge_mod.summarize(in_domain, jcfg["model"]).scores,
        "out_of_scope": judge_mod.summarize(out_scope, jcfg["model"]).scores,
    }
    return report, [{"id": r.item_id, "verdict": r.verdict, "error": r.error} for r in results]


def generate_all(cfg: dict[str, Any], which: str, backend: str, adapter_dir: str | None) -> Generated:
    """Load the model and generate replies for the test examples and the out-of-scope prompts."""
    ecfg = cfg["eval"]
    examples = load_eval_examples(cfg["data"]["out_dir"], ecfg["n_samples"])
    oos = read_jsonl(ecfg["out_of_scope_file"])
    model, tokenizer = load_for_inference(cfg, which, backend, adapter_dir)
    conversations = [ex["prompt"] for ex in examples] + [build_prompt_messages(o["prompt"]) for o in oos]
    replies = generate_replies(model, tokenizer, conversations, ecfg["max_new_tokens"], ecfg["batch_size"])
    return Generated(examples, replies[: len(examples)], oos, replies[len(examples) :])


def score(cfg: dict[str, Any], which: str, gen: Generated, use_judge: bool) -> tuple[dict[str, Any], list[dict]]:
    """Automatic metrics (+ the optional judge). Returns the report and one row per prediction."""
    clf, ceiling = build_classifier(cfg["data"]["out_dir"])
    summary, rows = metrics.summarize(
        gen.replies,
        [reference_text(e) for e in gen.examples],
        [user_text(e) for e in gen.examples],
        [e["intent"] for e in gen.examples],
        clf,
    )
    oos_details = [metrics.invented_details(r, o["prompt"]) for o, r in zip(gen.oos, gen.oos_replies, strict=True)]
    report: dict[str, Any] = {
        "model": which,
        "metrics": summary.as_dict(),
        "intent_classifier_ceiling": ceiling,
        "out_of_scope": {
            "n": len(gen.oos),
            "invented_detail_rate": round(sum(bool(d) for d in oos_details) / max(len(gen.oos), 1), 4),
        },
    }
    judge_rows: list[dict[str, Any]] = []
    if use_judge:
        report["judge"], judge_rows = run_judge(cfg, gen)
    return report, prediction_rows(gen, rows, oos_details, judge_rows)


def prediction_rows(
    gen: Generated, rows: list[dict[str, Any]], oos_details: list[dict[str, list[str]]], judge_rows: list[dict]
) -> list[dict[str, Any]]:
    """One JSON row per reply, for error analysis: what was asked, what the model said, how it scored."""
    verdicts = {j["id"]: j for j in judge_rows}
    preds = [
        {
            "id": ex["id"],
            "intent": ex["intent"],
            "user": user_text(ex),
            "reference": reference_text(ex),
            "reply": rep,
            **row,
            "judge": verdicts.get(ex["id"]),
        }
        for ex, rep, row in zip(gen.examples, gen.replies, rows, strict=True)
    ]
    preds += [
        {
            "id": o["id"],
            "type": o["type"],
            "user": o["prompt"],
            "expected": o["expected"],
            "reply": rep,
            "invented_details": details,
            "judge": verdicts.get(o["id"]),
        }
        for o, rep, details in zip(gen.oos, gen.oos_replies, oos_details, strict=True)
    ]
    return preds


def evaluate_model(
    cfg: dict[str, Any], which: str, backend: str, adapter_dir: str | None, use_judge: bool
) -> dict[str, Any]:
    """Generate, score and save one model's results."""
    set_seed(cfg["seed"])
    gen = generate_all(cfg, which, backend, adapter_dir)
    report, preds = score(cfg, which, gen, use_judge)
    out = Path(cfg["eval"]["results_dir"])
    write_jsonl(out / f"{which}_predictions.jsonl", preds)
    write_json(out / f"{which}_metrics.json", report)
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Comparison and quality gates
# ---------------------------------------------------------------------------
def check_gates(base: dict[str, Any], tuned: dict[str, Any], gates: dict[str, float]) -> list[str]:
    """Return the list of failed gates (empty list = pass)."""
    b, t = base["metrics"], tuned["metrics"]
    failures = []
    if t["intent_accuracy"] < gates["min_intent_accuracy"]:
        failures.append(f"intent accuracy {t['intent_accuracy']:.3f} < {gates['min_intent_accuracy']}")
    if t["intent_accuracy"] - b["intent_accuracy"] < gates["min_intent_accuracy_gain"]:
        gain = t["intent_accuracy"] - b["intent_accuracy"]
        failures.append(f"intent accuracy gain {gain:+.3f} < {gates['min_intent_accuracy_gain']}")
    if t["invented_detail_rate"] > gates["max_invented_detail_rate"]:
        failures.append(f"invented detail rate {t['invented_detail_rate']:.3f} > {gates['max_invented_detail_rate']}")
    if t["rouge_l"] - b["rouge_l"] < gates["min_rouge_l_gain"]:
        failures.append(f"ROUGE-L gain {t['rouge_l'] - b['rouge_l']:+.3f} < {gates['min_rouge_l_gain']}")
    return failures


def comparison_table(base: dict[str, Any], tuned: dict[str, Any]) -> str:
    """Markdown table for the README and the model card."""
    rows = [
        ("Intent accuracy (higher is better)", "intent_accuracy"),
        ("ROUGE-L (higher is better)", "rouge_l"),
        ("Invented-detail rate (lower is better)", "invented_detail_rate"),
        ("Uses placeholders", "placeholder_rate"),
        ("Average words", "avg_words"),
    ]
    lines = ["| Metric | Base model | Fine-tuned | Change |", "|---|---|---|---|"]
    for label, key in rows:
        b, t = base["metrics"][key], tuned["metrics"][key]
        lines.append(f"| {label} | {b:.3f} | {t:.3f} | {t - b:+.3f} |")
    for key in ("judge_correctness", "judge_helpfulness", "judge_tone", "judge_acceptable_rate"):
        b_j = base.get("judge", {}).get("in_domain", {}).get(key)
        t_j = tuned.get("judge", {}).get("in_domain", {}).get(key)
        if b_j is not None and t_j is not None:
            lines.append(
                f"| {key.replace('_', ' ').capitalize()} (LLM judge) | {b_j:.3f} | {t_j:.3f} | {t_j - b_j:+.3f} |"
            )
    return "\n".join(lines)


def compare(cfg: dict[str, Any]) -> int:
    """Write results/REPORT.md and return the process exit code (0 pass, 2 retrain)."""
    out = Path(cfg["eval"]["results_dir"])
    base = json.loads((out / "base_metrics.json").read_text(encoding="utf-8"))
    tuned = json.loads((out / "tuned_metrics.json").read_text(encoding="utf-8"))
    failures = check_gates(base, tuned, cfg["eval"]["gates"])
    verdict = "PASS: ready to export" if not failures else "FAIL: retrain needed"
    lines = [
        "# Evaluation report",
        "",
        f"Test examples: {tuned['metrics']['n']}. Intent-classifier ceiling (accuracy on the reference answers): "
        f"{tuned['intent_classifier_ceiling']:.3f}.",
        "",
        comparison_table(base, tuned),
        "",
        f"**Quality gates: {verdict}**",
    ]
    lines += [f"- {f}" for f in failures]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if not failures else EXIT_GATES_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["base", "tuned"], help="which model to evaluate")
    group.add_argument("--compare", action="store_true", help="compare saved base and tuned results")
    parser.add_argument("--backend", choices=["unsloth", "hf"], help="override train.backend")
    parser.add_argument("--adapter", help="adapter folder (default: <train.output_dir>/final_adapter)")
    parser.add_argument("--n-samples", type=int, help="override eval.n_samples")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge even if a key is set")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.n_samples:
        cfg["eval"]["n_samples"] = args.n_samples
    if args.compare:
        return compare(cfg)
    use_judge = judge_mod.judge_available() and not args.no_judge
    if not use_judge:
        print("LLM judge skipped (set ANTHROPIC_API_KEY to enable it).")
    evaluate_model(cfg, args.model, args.backend or cfg["train"]["backend"], args.adapter, use_judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
