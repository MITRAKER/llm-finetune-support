# Shortcuts for the commands CI runs. Usage: make help

PYTHON ?= python
CONFIG ?= configs/qwen3_4b_qlora.yaml

.PHONY: help install install-train lint format typecheck power-of-ten test smoke check data baseline train evaluate compare export

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install the dev tools and pre-commit hooks (no GPU needed)
	$(PYTHON) -m pip install -r requirements/dev.txt
	pre-commit install

install-train: ## Install the GPU training stack (Kaggle, Colab or a CUDA machine)
	$(PYTHON) -m pip install -r requirements/train.txt

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

smoke: ## All tests including the CPU training run, with coverage (needs requirements/smoke.txt)
	TRITON_INTERPRET=1 $(PYTHON) -m pytest --cov --cov-report=term --cov-fail-under=85

check: lint typecheck power-of-ten ## Everything CI checks
	ruff format --check .
	TRITON_INTERPRET=1 $(PYTHON) -m pytest --cov --cov-report=term --cov-fail-under=85

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
