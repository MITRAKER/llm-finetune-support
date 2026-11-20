# Shortcuts for the commands CI runs. Usage: make help

PYTHON ?= python
CONFIG ?= configs/qwen3_4b_qlora.yaml

.PHONY: help install install-train lint format typecheck power-of-ten test smoke smoke-stack check check-fast data baseline train evaluate compare export

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install the dev tools and pre-commit hooks (no GPU needed)
	$(PYTHON) -m pip install -r requirements/dev.txt
	pre-commit install

install-train: ## Install the GPU training stack, with the torch build for your NVIDIA driver
	$(PYTHON) -m pip install uv
	$(PYTHON) -m uv pip install --python $(PYTHON) -r requirements/train.txt --torch-backend=auto

lint: ## Ruff lint
	ruff check .

format: ## Format the code with ruff
	ruff format .
	ruff check --fix .

typecheck: ## Mypy type check
	mypy

power-of-ten: ## NASA/JPL Power of Ten rules, adapted to Python
	$(PYTHON) tools/check_power_of_ten.py src tools app tests

test: ## Fast unit tests
	$(PYTHON) -m pytest -m "not slow"

smoke: smoke-stack ## All tests including the CPU training run, with coverage
	TRITON_INTERPRET=1 $(PYTHON) -m pytest --cov --cov-report=term --cov-fail-under=85

check: lint typecheck power-of-ten smoke-stack ## Lint, format, types, rules and every test with coverage (Linux)
	ruff format --check .
	TRITON_INTERPRET=1 $(PYTHON) -m pytest --cov --cov-report=term --cov-fail-under=85

check-fast: lint typecheck power-of-ten ## The same without the CPU training run (works on macOS too)
	ruff format --check .
	$(PYTHON) -m pytest -m "not slow"

smoke-stack: # the end-to-end test trains a tiny model: without torch it is skipped and coverage fails
	@$(PYTHON) -c "import torch, transformers, trl, peft, datasets" 2>/dev/null || \
		{ echo "This needs the CPU training stack: pip install -r requirements/smoke.txt"; exit 1; }
	@$(PYTHON) -c "import triton" 2>/dev/null || \
		{ echo "The end-to-end test needs Linux (or WSL2, or a Linux container): TRL's training loss"; \
		  echo "uses Triton, which only has Linux wheels. Run 'make check-fast' here; CI runs the full check."; \
		  exit 1; }

data: ## Step 1: prepare the data
	$(PYTHON) src/prepare_data.py --config $(CONFIG)

baseline: ## Step 2: evaluate the base model
	$(PYTHON) src/evaluate.py --config $(CONFIG) --model base

train: ## Step 3: fine-tune (GPU)
	$(PYTHON) src/train.py --config $(CONFIG)

evaluate: ## Step 4: evaluate the fine-tuned model
	$(PYTHON) src/evaluate.py --config $(CONFIG) --model tuned

compare: ## Step 4: compare base vs tuned and apply the quality gates
	$(PYTHON) src/evaluate.py --config $(CONFIG) --compare

export: ## Step 5: merge, convert to GGUF (GPU)
	$(PYTHON) src/export.py --config $(CONFIG)
