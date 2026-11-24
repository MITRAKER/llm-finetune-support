"""Step 3: QLoRA supervised fine-tuning (SFT).

Loads the base model in 4-bit, adds small trainable LoRA adapters, and trains
them on the prompt-completion data from step 1. Only the adapters are saved
(about 100 MB), not a new copy of the whole model.

Usage:
    python src/train.py --config configs/qwen3_4b_qlora.yaml
    python src/train.py --config configs/qwen3_4b_qlora.yaml --max-steps 30   # quick test run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess  # only used to read the git commit hash
import time
from pathlib import Path
from typing import Any

from common import adapter_path, load_config, set_seed, use_one_gpu, write_json

BACKENDS = ("unsloth", "hf")
TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "unsloth", "accelerate")


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------
def precision_flags() -> dict[str, bool]:
    """Pick the training precision for the GPU we are on.

    bf16 needs an Ampere-or-newer GPU (compute capability 8.0+: A100, L4,
    RTX 30xx/40xx). The free T4 is 7.5, so it trains in fp16. Note that
    torch.cuda.is_bf16_supported() returns True on a T4 because bf16 can be
    *emulated*, which is slow, so we check the capability directly. On CPU
    (the smoke test) we use plain fp32.
    """
    import torch

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    major, _minor = torch.cuda.get_device_capability()
    use_bf16 = major >= 8
    return {"bf16": use_bf16, "fp16": not use_bf16}


def resolve_report_to(setting: str) -> str:
    """ "auto" logs to Weights & Biases only when an API key is configured."""
    if setting != "auto":
        return setting
    return "wandb" if os.environ.get("WANDB_API_KEY") else "none"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_splits(data_dir: str | Path, max_train: int | None, max_eval: int | None, seed: int) -> tuple[Any, Any]:
    """Load train/validation JSONL as Hugging Face datasets with only the columns TRL needs."""
    data_dir = Path(data_dir)
    files = {"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "validation.jsonl")}
    for path in files.values():
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run step 1 first: python src/prepare_data.py --config ...")

    from datasets import load_dataset  # imported after the check, so a missing file fails fast

    ds = load_dataset("json", data_files=files)
    train = ds["train"].select_columns(["prompt", "completion"]).shuffle(seed=seed)
    val = ds["validation"].select_columns(["prompt", "completion"])
    if max_train:
        train = train.select(range(min(max_train, len(train))))
    if max_eval:
        val = val.select(range(min(max_eval, len(val))))
    return train, val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(cfg: dict[str, Any], backend: str) -> tuple[Any, Any, Any]:
    """Return (model, tokenizer, peft_config).

    With Unsloth the LoRA adapters are added here and peft_config is None.
    With plain Hugging Face we hand a LoraConfig to the trainer instead.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "unsloth":
        return _load_unsloth(cfg)
    return _load_hf(cfg)


def _load_unsloth(cfg: dict[str, Any]) -> tuple[Any, Any, None]:  # pragma: no cover - needs a CUDA GPU
    from unsloth import FastLanguageModel

    mcfg, lcfg = cfg["model"], cfg["lora"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg["max_seq_length"],
        load_in_4bit=mcfg["load_in_4bit"],
        dtype=None,  # auto: bf16 on Ampere+, fp16 on T4
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # recomputes activations to save memory
        random_state=cfg["seed"],
    )
    return model, tokenizer, None


def _load_hf(cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    mcfg, lcfg = cfg["model"], cfg["lora"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {}
    if mcfg["load_in_4bit"]:  # pragma: no cover - bitsandbytes 4-bit needs a CUDA GPU
        compute_dtype = torch.bfloat16 if precision_flags()["bf16"] else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4: the QLoRA paper's data type
            bnb_4bit_use_double_quant=True,  # also quantize the quantization constants
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], **kwargs)
    peft_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, peft_config


def count_parameters(model: Any) -> dict[str, Any]:
    """How many weights we actually train vs. the total model size.

    A 4-bit weight matrix packs two weights into each byte, so its tensor reports
    only half of them. PEFT's own counter corrects for that, so we use it when the
    model has one (every LoRA model does).
    """
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / max(total, 1), 4)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def build_sft_config_kwargs(cfg: dict[str, Any], backend: str, max_steps: int | None = None) -> dict[str, Any]:
    """Translate our YAML config into SFTConfig keyword arguments (pure function, unit tested)."""
    tcfg, mcfg = cfg["train"], cfg["model"]
    steps = max_steps if max_steps is not None else tcfg["max_steps"]
    return {
        "output_dir": tcfg["output_dir"],
        "num_train_epochs": tcfg["num_train_epochs"],
        "max_steps": steps,
        "per_device_train_batch_size": tcfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": tcfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": tcfg["gradient_accumulation_steps"],
        "learning_rate": tcfg["learning_rate"],
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "warmup_steps": tcfg["warmup_steps"],  # a float below 1 means a fraction of all steps
        "weight_decay": tcfg["weight_decay"],
        "optim": tcfg["optim"],
        "logging_steps": tcfg["logging_steps"],
        "eval_strategy": "steps",
        "eval_steps": tcfg["eval_steps"],
        "save_strategy": "steps",
        "save_steps": tcfg["save_steps"],
        "save_total_limit": tcfg["save_total_limit"],
        "load_best_model_at_end": True,  # keep the checkpoint with the lowest validation loss
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "max_length": mcfg["max_seq_length"],
        "packing": False,
        "completion_only_loss": True,  # learn the answers, not the prompts
        # Unsloth adds its own (faster) gradient checkpointing in get_peft_model.
        "gradient_checkpointing": backend == "hf",
        "report_to": resolve_report_to(tcfg["report_to"]),
        "run_name": Path(tcfg["output_dir"]).name,
        "seed": cfg["seed"],
        **precision_flags(),
    }


def environment_info() -> dict[str, Any]:
    """Library versions and git commit, saved next to the adapter for reproducibility."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit: str | None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - git is looked up on PATH on purpose
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"packages": versions, "git_commit": commit}


def train(cfg: dict[str, Any], backend: str, max_steps: int | None = None) -> dict[str, Any]:
    """Run one fine-tuning job and return a summary dict."""
    if backend == "unsloth":  # pragma: no cover - needs a CUDA GPU
        import unsloth  # noqa: F401  # must be imported before trl/transformers so it can patch them

    from trl import SFTConfig, SFTTrainer

    set_seed(cfg["seed"])
    train_ds, val_ds = load_splits(
        cfg["data"]["out_dir"], cfg["train"]["max_train_samples"], cfg["train"]["max_eval_samples"], cfg["seed"]
    )
    model, tokenizer, peft_config = load_model(cfg, backend)
    sft_args = SFTConfig(**build_sft_config_kwargs(cfg, backend, max_steps))
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    params = count_parameters(trainer.model)
    print(f"Trainable parameters: {params['trainable']:,} of {params['total']:,} ({params['trainable_pct']}%)")

    start = time.time()
    result = trainer.train()
    eval_metrics = trainer.evaluate()
    adapter_dir = adapter_path(cfg)
    trainer.save_model(str(adapter_dir))  # saves only the LoRA adapter weights + config
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "backend": backend,
        "base_model": cfg["model"]["name"],
        "adapter_dir": str(adapter_dir),
        "train_examples": len(train_ds),
        "parameters": params,
        "train_loss": result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss"),
        "global_steps": result.global_step,
        "runtime_minutes": round((time.time() - start) / 60, 2),
        "log_history": trainer.state.log_history,
        "environment": environment_info(),
        "config": cfg,
    }
    write_json(Path(cfg["train"]["output_dir"]) / "training_summary.json", summary)
    print(f"Saved adapter to {adapter_dir}. Final validation loss: {summary['eval_loss']}")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=BACKENDS, help="override train.backend")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return train(cfg, args.backend or cfg["train"]["backend"], args.max_steps)


if __name__ == "__main__":
    use_one_gpu()  # before anything imports torch
    main()
