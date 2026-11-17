"""Step 5: export the fine-tuned model for deployment.

1. Merge the LoRA adapter into the base model (16-bit) so it is one normal model
   (written to export.merged_dir; about 8 GB for a 4B model, so keep it on scratch disk).
2. Convert it to GGUF and quantize it to 4-bit (q4_k_m) for llama.cpp, so it
   runs on a laptop or a free CPU Space (about 2.5 GB).
3. Optionally upload the adapter and the GGUF file to the Hugging Face Hub,
   with a model card that reports the evaluation results.

Needs a GPU with Unsloth (run it in the Kaggle notebook after training).

Usage:
    python src/export.py --config configs/qwen3_4b_qlora.yaml
    HF_TOKEN=... python src/export.py --config configs/qwen3_4b_qlora.yaml --push
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common import load_config

MODEL_CARD_TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "MODEL_CARD.md"


def gguf_search_dirs(save_dir: str | Path) -> list[Path]:
    """Folders where the converter may have written the GGUF files.

    Unsloth 2026.x writes the 16-bit merge into save_dir and the GGUF files into
    a sibling folder named "<save_dir>_gguf". Older versions wrote both into
    save_dir. We look in both, so the script works with either.
    """
    save_dir = Path(save_dir)
    return [save_dir.with_name(save_dir.name + "_gguf"), save_dir]


def find_gguf(directories: Sequence[str | Path], quantization: str) -> Path:
    """Locate the GGUF file for this quantization (its exact name depends on the Unsloth version)."""
    found: list[Path] = []
    for directory in directories:
        if Path(directory).is_dir():
            found.extend(sorted(Path(directory).rglob("*.gguf")))
    matches = [p for p in found if quantization.lower() in p.name.lower()]
    if not matches:
        searched = [str(d) for d in directories]
        raise FileNotFoundError(f"No {quantization} GGUF file in {searched}. Found: {[p.name for p in found]}")
    return matches[0]


def hub_gguf_name(hub_model_id: str | None, quantization: str) -> str:
    """A stable file name for the GGUF on the Hub, e.g. "qwen3-4b-support.Q4_K_M.gguf".

    The demo app finds the file with the pattern "*Q4_K_M.gguf", so the name
    must not depend on what the converter happened to call it.
    """
    stem = hub_model_id.split("/")[-1] if hub_model_id else "model"
    return f"{stem}.{quantization.upper()}.gguf"


def render_model_card(cfg: dict[str, Any], results_dir: str | Path, template: Path = MODEL_CARD_TEMPLATE) -> str:
    """Fill the model card template with the config and the evaluation report."""
    report_path = Path(results_dir) / "REPORT.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "_Evaluation not run yet._"
    stats_path = Path(cfg["data"]["out_dir"]) / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    values = {
        "base_model": cfg["model"]["name"],
        "hub_model_id": cfg["export"]["hub_model_id"] or "local",
        "dataset": cfg["data"]["source"],
        "train_examples": str(stats.get("train", "n/a")),
        "lora_r": str(cfg["lora"]["r"]),
        "lora_alpha": str(cfg["lora"]["alpha"]),
        "learning_rate": str(cfg["train"]["learning_rate"]),
        "epochs": str(cfg["train"]["num_train_epochs"]),
        "quantization": cfg["export"]["gguf_quantization"],
        "gguf_file": hub_gguf_name(cfg["export"]["hub_model_id"], cfg["export"]["gguf_quantization"]),
        "evaluation_report": report.replace("# Evaluation report", "").strip(),
    }
    text = template.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def export(cfg: dict[str, Any], push: bool) -> dict[str, str]:  # pragma: no cover - needs a CUDA GPU
    import unsloth  # noqa: F401  # import first so it can patch transformers
    from unsloth import FastLanguageModel

    ecfg = cfg["export"]
    adapter_dir = str(Path(cfg["train"]["output_dir"]) / "final_adapter")
    # Loading the adapter folder (not the base model) makes Unsloth attach the adapter properly,
    # so save_pretrained_gguf merges it into the base weights before converting.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=cfg["model"]["max_seq_length"], load_in_4bit=True
    )
    model.save_pretrained_gguf(ecfg["merged_dir"], tokenizer, quantization_method=ecfg["gguf_quantization"])
    gguf = find_gguf(gguf_search_dirs(ecfg["merged_dir"]), ecfg["gguf_quantization"])
    outputs = {"adapter": adapter_dir, "gguf": str(gguf)}
    print(f"GGUF model: {gguf} ({gguf.stat().st_size / 1e9:.2f} GB)")
    if push:
        outputs.update(push_to_hub(cfg, model, tokenizer, gguf))
    return outputs


def push_to_hub(cfg: dict[str, Any], model: Any, tokenizer: Any, gguf: Path) -> dict[str, str]:  # pragma: no cover
    from huggingface_hub import HfApi

    repo = cfg["export"]["hub_model_id"]
    token = os.environ.get("HF_TOKEN")
    if not repo or not token:
        raise ValueError("Set export.hub_model_id in the config and the HF_TOKEN environment variable to push.")
    api = HfApi(token=token)
    model.push_to_hub(f"{repo}-lora", token=token)  # the small adapter, for people who use transformers + peft
    tokenizer.push_to_hub(f"{repo}-lora", token=token)
    gguf_repo = f"{repo}-GGUF"
    api.create_repo(gguf_repo, exist_ok=True)
    gguf_name = hub_gguf_name(repo, cfg["export"]["gguf_quantization"])
    api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf_name, repo_id=gguf_repo)
    card = render_model_card(cfg, cfg["eval"]["results_dir"])
    for target in (gguf_repo, f"{repo}-lora"):
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=target)
    print(f"Pushed https://huggingface.co/{gguf_repo} and https://huggingface.co/{repo}-lora")
    return {"gguf_repo": gguf_repo, "adapter_repo": f"{repo}-lora"}


def main(argv: list[str] | None = None) -> dict[str, str]:  # pragma: no cover - needs a CUDA GPU
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub")
    parser.add_argument("--merged-dir", help="override export.merged_dir (put it on scratch disk on Kaggle)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.merged_dir:
        cfg["export"]["merged_dir"] = args.merged_dir
    return export(cfg, args.push)


if __name__ == "__main__":  # pragma: no cover
    main()
