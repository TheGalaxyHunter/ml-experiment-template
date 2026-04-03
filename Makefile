.PHONY: train test lint format typecheck clean docker-build docker-run install help

PROJECT := src
SCRIPTS := scripts
TESTS := tests

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (including dev)
	uv sync

train: ## Run training with default config
	uv run python $(SCRIPTS)/train.py

evaluate: ## Run evaluation
	uv run python $(SCRIPTS)/evaluate.py

test: ## Run test suite
	uv run pytest $(TESTS) -v --tb=short

test-cov: ## Run tests with coverage
	uv run pytest $(TESTS) -v --tb=short --cov=$(PROJECT) --cov-report=term-missing

lint: ## Lint with ruff
	uv run ruff check $(PROJECT) $(SCRIPTS) $(TESTS)

format: ## Auto-format with ruff
	uv run ruff format $(PROJECT) $(SCRIPTS) $(TESTS)
	uv run ruff check --fix $(PROJECT) $(SCRIPTS) $(TESTS)

typecheck: ## Type check with mypy
	uv run mypy $(PROJECT)

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf outputs/ multirun/ checkpoints/ wandb/ mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-build: ## Build Docker image
	docker build -t ml-experiment-template .

docker-run: ## Run training in Docker
	docker run --gpus all -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs \
		ml-experiment-template

docker-run-interactive: ## Run interactive shell in Docker
	docker run --gpus all -it -v $(PWD):/app ml-experiment-template /bin/bash
