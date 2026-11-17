# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): MAJOR for changes that break how the
pipeline is used, MINOR for new features, PATCH for fixes.

## [Unreleased]

## [0.1.0] - 2026-11-17

### Added

- Step 1, data preparation: cleans, deduplicates and splits the Bitext customer-support
  dataset (stratified by intent) into prompt-completion chat examples.
- Step 2, baseline evaluation: intent accuracy, ROUGE-L, invented-detail rate and
  placeholder rate, plus 24 out-of-scope prompts and an optional Claude LLM judge.
- Step 3, QLoRA fine-tuning of Qwen3-4B-Instruct-2507 with Unsloth and TRL, with the
  loss on the answers only.
- Step 4, tuned evaluation with quality gates: `evaluate.py --compare` writes
  `results/REPORT.md` and exits with code 2 when the model must be retrained.
- Step 5, export to GGUF (q4_k_m), upload to the Hugging Face Hub with a model card,
  and a Gradio demo for a free CPU Space.
- Kaggle notebook that runs the whole pipeline on a free GPU.
- CI on every pull request: ruff, mypy, the NASA/JPL Power of Ten checker, unit tests on
  Python 3.11 to 3.13, an end-to-end CPU smoke test with coverage, and pip-audit.
- CodeQL scanning, Dependabot, pre-commit hooks (including secret scanning), issue and
  pull request templates, release workflow and continuous deployment of the demo.
- LEARNING.md: a step-by-step lesson for every part of the project.
