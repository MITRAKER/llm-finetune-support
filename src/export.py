"""Step 5: export the fine-tuned model for deployment.

1. Merge the LoRA adapter into the base model (16-bit) so it is one normal model
   (written to export.merged_dir; about 8 GB for a 4B model, so keep it on scratch disk).
2. Convert it to GGUF and quantize it to 4-bit (q4_k_m) for llama.cpp, so it
   runs on a laptop or a free CPU Space (about 2.5 GB). The file is copied to
   export.output_dir under a stable name, such as qwen3-4b-support.Q4_K_M.gguf.
3. Optionally upload the adapter and the GGUF file to the Hugging Face Hub,
   with a model card that reports the evaluation results.

Needs a GPU with Unsloth for steps 1 and 2 (run it in the Kaggle notebook after training).

Usage:
    python src/export.py --config configs/qwen3_4b_qlora.yaml
    HF_TOKEN=... python src/export.py --config configs/qwen3_4b_qlora.yaml --push
    # if the upload failed: upload the files exported before, without exporting again
    HF_TOKEN=... python src/export.py --config configs/qwen3_4b_qlora.yaml --push-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common import SYSTEM_PROMPT, load_config

MODEL_CARD_TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "MODEL_CARD.md"


def gguf_search_dirs(save_dir: str | Path) -> list[Path]:
    """Folders where the converter may have written the GGUF files.

    Unsloth 2026.x writes the 16-bit merge into save_dir and the GGUF files into
    a sibling folder named "<save_dir>_gguf". Older versions wrote both into
    save_dir. We look in both, so the script works with either.
    """
    save_dir = Path(save_dir)
    return [save_dir.with_name(save_dir.name + "_gguf"), save_dir]


def select_gguf(files: Sequence[str | Path], quantization: str) -> Path:
    """The file for this quantization among the converter's output files."""
    paths = [Path(f) for f in files if str(f).lower().endswith(".gguf")]
    matches = [p for p in paths if quantization.lower() in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No {quantization} GGUF file among {[p.name for p in paths]}")
    return matches[0]


def find_gguf(directories: Sequence[str | Path], quantization: str) -> Path:
    """Search folders for the GGUF file (for converters that do not say where they wrote it)."""
    found: list[Path] = []
    for directory in directories:
        if Path(directory).is_dir():
            found.extend(sorted(Path(directory).rglob("*.gguf")))
    try:
        return select_gguf(found, quantization)
    except FileNotFoundError:
        raise FileNotFoundError(f"No {quantization} GGUF file in {[str(d) for d in directories]}") from None


def gguf_file_name(cfg: dict[str, Any]) -> str:
    """A stable file name for the GGUF, on disk and on the Hub, e.g. "qwen3-4b-support.Q4_K_M.gguf".

    The demo app finds the file with the pattern "*Q4_K_M.gguf", so the name must
    not depend on what the converter called it, and it should not look like the
    stock base model (the converter names it after the base model).
    """
    ecfg = cfg["export"]
    if ecfg["hub_model_id"]:
        stem = ecfg["hub_model_id"].split("/")[-1]
    else:
        stem = Path(cfg["train"]["output_dir"]).name.removesuffix("-lora")
    return f"{stem}.{ecfg['gguf_quantization'].upper()}.gguf"


def adapter_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["train"]["output_dir"]) / "final_adapter"


def exported_gguf(cfg: dict[str, Any]) -> Path:
    """Where export() puts the final GGUF file."""
    return Path(cfg["export"]["output_dir"]) / gguf_file_name(cfg)


def push_settings(cfg: dict[str, Any]) -> tuple[str, str]:
    """The Hub repo and token for an upload. Checked before any slow work starts."""
    repo = cfg["export"]["hub_model_id"]
    token = os.environ.get("HF_TOKEN")
    if not repo or "/" not in repo:
        raise ValueError("Set export.hub_model_id in the config (like 'your-name/qwen3-4b-support') to upload.")
    if not token:
        raise ValueError("Set the HF_TOKEN environment variable (a token with write access) to upload.")
    return repo, token


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
        "gguf_file": gguf_file_name(cfg),
        "system_prompt": SYSTEM_PROMPT,
        "evaluation_report": report.replace("# Evaluation report", "").strip(),
    }
    text = template.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def export(cfg: dict[str, Any]) -> Path:  # pragma: no cover - needs a CUDA GPU
    """Merge, convert and quantize; return the GGUF file, copied to export.output_dir."""
    import unsloth  # noqa: F401  # import first so it can patch transformers
    from unsloth import FastLanguageModel

    ecfg = cfg["export"]
    # Loading the adapter folder (not the base model) makes Unsloth attach the adapter properly,
    # so save_pretrained_gguf merges it into the base weights before converting.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir(cfg)), max_seq_length=cfg["model"]["max_seq_length"], load_in_4bit=True
    )
    result = model.save_pretrained_gguf(ecfg["merged_dir"], tokenizer, quantization_method=ecfg["gguf_quantization"])
    # Unsloth returns the files it wrote (it may move the output to scratch disk when space is short).
    files = result.get("gguf_files") if isinstance(result, dict) else None
    if files:
        produced = select_gguf(files, ecfg["gguf_quantization"])
    else:
        produced = find_gguf(gguf_search_dirs(ecfg["merged_dir"]), ecfg["gguf_quantization"])
    target = exported_gguf(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, target)
    print(f"GGUF model: {target} ({target.stat().st_size / 1e9:.2f} GB)")
    return target


def create_hub_repos(api: Any, repo: str) -> tuple[str, str]:  # pragma: no cover - talks to the Hub
    """Create the adapter and GGUF repos (if needed). Run first: it fails fast on a read-only token."""
    adapter_repo, gguf_repo = f"{repo}-lora", f"{repo}-GGUF"
    for target in (adapter_repo, gguf_repo):
        api.create_repo(target, exist_ok=True)
    return adapter_repo, gguf_repo


def push_to_hub(cfg: dict[str, Any], gguf: Path) -> dict[str, str]:  # pragma: no cover - talks to the Hub
    """Upload the adapter folder, the GGUF file and the model card."""
    from huggingface_hub import HfApi

    repo, token = push_settings(cfg)
    if not gguf.is_file() or not adapter_dir(cfg).is_dir():
        raise FileNotFoundError(f"Nothing to upload: export first (missing {gguf} or {adapter_dir(cfg)}).")
    api = HfApi(token=token)
    adapter_repo, gguf_repo = create_hub_repos(api, repo)
    # The small adapter, for people who use transformers + peft (not the pickled training arguments).
    api.upload_folder(
        repo_id=adapter_repo,
        folder_path=str(adapter_dir(cfg)),
        ignore_patterns=["training_args.bin"],
        commit_message="Upload the LoRA adapter",
    )
    api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf.name, repo_id=gguf_repo)
    card = render_model_card(cfg, cfg["eval"]["results_dir"])
    for target in (gguf_repo, adapter_repo):
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=target)
    print(f"Pushed https://huggingface.co/{gguf_repo} and https://huggingface.co/{adapter_repo}")
    return {"gguf_repo": gguf_repo, "adapter_repo": adapter_repo}


def main(argv: list[str] | None = None) -> dict[str, str]:  # pragma: no cover - needs a CUDA GPU
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--push", action="store_true", help="export, then upload to the Hugging Face Hub")
    mode.add_argument("--push-only", action="store_true", help="upload the files a previous export wrote")
    parser.add_argument("--merged-dir", help="override export.merged_dir (put it on scratch disk on Kaggle)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.merged_dir:
        cfg["export"]["merged_dir"] = args.merged_dir
    if args.push or args.push_only:
        from huggingface_hub import HfApi

        repo, token = push_settings(cfg)  # before the slow export: a missing setting fails in seconds
        create_hub_repos(HfApi(token=token), repo)
    gguf = exported_gguf(cfg) if args.push_only else export(cfg)
    outputs = {"adapter": str(adapter_dir(cfg)), "gguf": str(gguf)}
    if args.push or args.push_only:
        outputs.update(push_to_hub(cfg, gguf))
    return outputs


if __name__ == "__main__":  # pragma: no cover
    main()
