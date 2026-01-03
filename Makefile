# Declare all phony targets
.PHONY: install clean notebooks-clean lint doc pipeline all

# Default target
.DEFAULT_GOAL := all

# Variables
SRC_ALL ?= .

# Install project dependencies
install:
	@echo "Installing dependencies..."
	@uv sync --all-extras
	@echo "✅ Dependencies installed."

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@echo "✅ Clean complete."

# Clean notebooks output
notebooks-clean:
	@echo "Cleaning notebooks output..."
	@find . -name '*.ipynb' -exec uv run nbstripout {} +
	@echo "✅ Clean complete."

# Check code formatting and linting
lint:
	@echo "Running lint checks..."
	@uv run black $(SRC_ALL)
	@uv run isort $(SRC_ALL)
	@uv run nbqa isort $(SRC_ALL)
	@echo "✅ Linting complete."

# Serve documentation locally
doc:
	@echo "Serving documentation..."
	@uv run mkdocs serve

# Run code checks
pipeline: clean notebooks-clean lint
	@echo "✅ Pipeline complete."

# Run full workflow including install and docs
all: install pipeline doc
	@echo "✅ All tasks complete."