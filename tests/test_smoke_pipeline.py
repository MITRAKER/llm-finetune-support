"""End-to-end smoke test: data prep -> baseline eval -> training -> tuned eval -> compare.

Runs the real scripts with the plain Hugging Face backend on a tiny random
model, on CPU, in well under a minute. It catches API breakage in TRL /
transformers / PEFT before you spend GPU hours on Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
# TRL's training loss is a Triton kernel, and Triton only has Linux wheels.
pytest.importorskip("triton", reason="TRL's training loss needs Triton, which only has Linux wheels")

import evaluate
import export
import prepare_data
import train
from common import SYSTEM_PROMPT, adapter_digest, load_config
from tests.tiny_model import build_tiny_model

pytestmark = pytest.mark.slow

INTENTS = ("cancel_order", "get_refund", "track_order")


def synthetic_rows(n_per_intent: int = 20) -> pd.DataFrame:
    rows = [
        {
            "instruction": f"hello, I need help to {intent.replace('_', ' ')} number {i}",
            "response": f"Sure, I can help you {intent.replace('_', ' ')}. Please share {{{{Order Number}}}} ({i}).",
            "intent": intent,
            "category": "ORDER",
            "flags": "B",
        }
        for intent in INTENTS
        for i in range(n_per_intent)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("smoke")
    raw = synthetic_rows()
    raw_path = root / "raw.csv"
    raw.to_csv(raw_path, index=False)
    # Include the system prompt so the tokenizer learns compact tokens for it; otherwise
    # the prompt alone fills max_length and every answer is truncated away.
    corpus = list(raw["instruction"]) + list(raw["response"]) + [SYSTEM_PROMPT] * 20 + ["system user assistant"]
    model_dir = build_tiny_model(root / "tiny-model", corpus)

    base_cfg = yaml.safe_load(Path("configs/qwen3_4b_qlora.yaml").read_text(encoding="utf-8"))
    base_cfg["data"].update(source=str(raw_path), out_dir=str(root / "data"))
    base_cfg["model"].update(name=str(model_dir), max_seq_length=512, load_in_4bit=False)
    base_cfg["lora"].update(r=4, alpha=8)
    base_cfg["train"].update(
        backend="hf",
        output_dir=str(root / "outputs"),
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        max_eval_samples=4,
        report_to="none",
    )
    base_cfg["eval"].update(n_samples=4, max_new_tokens=8, batch_size=2, results_dir=str(root / "results"))
    cfg_path = root / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    return {"root": root, "cfg": cfg_path}


def test_full_pipeline(workspace: dict[str, Path]) -> None:
    cfg, root = str(workspace["cfg"]), workspace["root"]

    stats = prepare_data.main(["--config", cfg])
    assert stats["train"] > 0
    assert stats["test"] > 0

    assert evaluate.main(["--config", cfg, "--model", "base", "--no-judge"]) == 0
    summary = train.main(["--config", cfg])
    adapter = Path(summary["adapter_dir"])
    assert (adapter / "adapter_config.json").exists()
    assert (adapter / "adapter_model.safetensors").exists()
    assert summary["parameters"]["trainable"] < summary["parameters"]["total"]
    assert summary["eval_loss"] is not None

    assert evaluate.main(["--config", cfg, "--model", "tuned", "--no-judge"]) == 0
    tuned = json.loads((root / "results" / "tuned_metrics.json").read_text(encoding="utf-8"))
    assert tuned["metrics"]["n"] == 4
    assert tuned["adapter_digest"] == adapter_digest(adapter)  # which adapter these numbers describe
    preds = (root / "results" / "tuned_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(preds) == 4 + 24  # test examples + out-of-scope prompts

    code = evaluate.main(["--config", cfg, "--compare"])
    assert code in (0, evaluate.EXIT_GATES_FAILED)  # a random tiny model may fail the gates
    assert "Quality gates" in (root / "results" / "REPORT.md").read_text(encoding="utf-8")
    # Checked on real files: REPORT.md is up to date and describes the adapter --push would upload.
    export.check_report(load_config(cfg))
    assert export.push_warnings(load_config(cfg)) == []


def test_loss_is_computed_on_the_answer_only(workspace: dict[str, Path]) -> None:
    """The completion mask must be 0 for system/user tokens and 1 for the assistant answer."""
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(workspace["cfg"].read_text(encoding="utf-8"))
    prepare_data.main(["--config", str(workspace["cfg"])])
    train_ds, val_ds = train.load_splits(cfg["data"]["out_dir"], 4, 2, cfg["seed"])
    model, tokenizer, peft_config = train.load_model(cfg, "hf")
    kwargs = train.build_sft_config_kwargs(cfg, "hf", max_steps=1)
    kwargs["output_dir"] = str(workspace["root"] / "mask-check")
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    example = trainer.train_dataset[0]
    ids = example["input_ids"]
    # TRL turns the completion mask into labels: -100 means "no loss on this token".
    mask = [label != -100 for label in example["labels"]] if "labels" in example else example["completion_mask"]
    answer = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m])
    prompt = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if not m])
    assert "Sure, I can help you" in answer
    assert "<|im_start|>system" in prompt
    assert "hello, I need help" in prompt
    assert "hello, I need help" not in answer
