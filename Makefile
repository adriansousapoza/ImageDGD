# Makefile for ImageDGD

.PHONY: help setup train train-weak-gpu test-weak-gpu optimize evaluate clean mlflow-ui install lint test

# Default target
help:
	@echo "Available targets:"
	@echo "  setup          - Setup environment and dependencies"
	@echo "  install        - Install dependencies"
	@echo "  train          - Train model with default config"
	@echo "  train-weak-gpu - Train with weak GPU optimized settings (NEW!)"
	@echo "  test-weak-gpu  - Test weak GPU setup and show available commands (NEW!)"
	@echo "  optimize       - Run hyperparameter optimization"
	@echo "  evaluate       - Evaluate trained model"
	@echo "  mlflow-ui      - Launch MLflow UI"
	@echo "  clean          - Clean generated files"
	@echo "  lint           - Run code linting"
	@echo "  test           - Run tests"

# Setup environment
setup:
	@echo "Setting up ImageDGD environment..."
	./scripts/setup.sh

# Install dependencies
install:
	pip install -r requirements.txt

# Train model
train:
	python scripts/cli.py train

# Weak GPU training (NEW!)
train-weak-gpu:
	@echo "Starting weak GPU training (10% data, 50 epochs)..."
	python scripts/cli.py train-weak-gpu

# Test weak GPU setup
test-weak-gpu:
	python scripts/test_weak_gpu.py

# Run optimization
optimize:
	python scripts/cli.py optimize

# Evaluate model (requires MODEL_PATH)
evaluate:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Please specify MODEL_PATH: make evaluate MODEL_PATH=path/to/model.pth"; \
		exit 1; \
	fi
	python scripts/cli.py evaluate -m $(MODEL_PATH)

# Launch MLflow UI
mlflow-ui:
	python scripts/cli.py mlflow-ui

# Clean generated files
clean:
	rm -rf outputs/
	rm -rf logs/
	rm -rf figures/
	rm -rf .hydra/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Lint code
lint:
	flake8 src/ scripts/
	black --check src/ scripts/
	isort --check-only src/ scripts/

# Format code
format:
	black src/ scripts/
	isort src/ scripts/

# Run tests
test:
	pytest tests/ -v

# Quick development cycle
dev-train:
	python scripts/cli.py train -o training.epochs=10 data.use_subset=true data.subset_fraction=0.1

# Examples
example-train:
	python scripts/cli.py train -o training.epochs=50 model.representation.n_features=8

example-optimize:
	python scripts/cli.py optimize -o optimization.n_trials=20
