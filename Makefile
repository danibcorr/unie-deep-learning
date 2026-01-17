.PHONY: setup \
		clean-cache-temp-files \
		lint \
		doc  \
		pipeline pre-commit all

.DEFAULT_GOAL := all

SOURCE_PATH ?= docs/course

setup:
	@echo "Installing dependencies..."
	@uv sync --all-extras
	@uv run pre-commit install
	@echo "✅ Dependencies installed."

clean-cache-temp-files:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@echo "✅ Clean complete."

lint:
	@echo "Running lint checks..."
	@uv run isort $(SOURCE_PATH)
	@uv run ruff check --fix $(SOURCE_PATH)
	@uv run ruff format $(SOURCE_PATH)
	@echo "✅ Linting complete."

doc:
	@echo "Serving documentation..."
	@uv run mkdocs serve

pipeline: clean-cache-temp-files lint
	@echo "✅ Pipeline complete."

pre-commit: clean-cache-temp-files lint
	@echo "✅ Pipeline pre-commit complete."

all: setup pipeline doc
	@echo "✅ All tasks complete."
