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
- Step 2, baseline evaluation: intent accuracy, ROUGE-L, invented-detail rate (phone
  numbers, emails, links, long numbers and amounts the customer did not give) and
  placeholder rate, plus 24 out-of-scope prompts and an optional Claude LLM judge.
- Step 3, QLoRA fine-tuning of Qwen3-4B-Instruct-2507 with Unsloth and TRL, with the
  loss on the answers only.
- Step 4, tuned evaluation with quality gates: `evaluate.py --compare` writes
  `results/REPORT.md` and exits with code 3 when the model must be retrained. Two of
  the gates check that the out-of-scope prompts did not get worse than with the base
  model. The comparison refuses a baseline that was scored on other questions or
  reference answers, with an intent classifier trained on other data, or with another
  reply length (`eval.max_new_tokens`). `eval.split` (or `--split`) evaluates on the
  validation split while you iterate, so the test set is used once, for the final
  report.
- Step 5, export to GGUF (q4_k_m) under a stable file name, upload to the Hugging Face
  Hub with a model card that includes the data counts, the system prompt, the
  evaluation report and the split it was measured on (`--push`, which checks that the
  token can write to the repos before the slow export, or `--push-only` to retry an
  upload), and a Gradio demo for a free CPU Space that trims the chat history to the
  model's context window, counted in tokens.
- Kaggle notebook that runs the whole pipeline on a free GPU, and `make install-train`
  for any other CUDA machine (it keeps a torch build that matches the NVIDIA driver).
- CI on every pull request: ruff, mypy, the NASA/JPL Power of Ten checker (including
  indirect recursion), the pre-commit hooks, unit tests on Python 3.11 to 3.13, an
  end-to-end CPU smoke test with coverage (Linux only; `make check-fast` runs the rest
  on macOS), and pip-audit of every requirements file and of the Space's Gradio.
- CodeQL scanning, Dependabot (it leaves the GPU stack that Unsloth caps to manual
  upgrades), pre-commit hooks (including secret scanning), issue and pull request
  templates, a release workflow that checks the tag against `pyproject.toml`,
  `CHANGELOG.md` and `CITATION.cff`, and continuous deployment of the demo (after CI
  on `main`, or by hand on `main` only).
- LEARNING.md: a step-by-step lesson for every part of the project.
