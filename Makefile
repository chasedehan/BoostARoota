# BoostARoota Makefile

.PHONY: test test-verbose test-quick conda-env install clean example

# Conda environment name
CONDA_ENV ?= boostaroota

# Create / update conda environment
conda-env:
	conda env update -n $(CONDA_ENV) -f environment.yml || conda env create -n $(CONDA_ENV) -f environment.yml

# Install via pip (fallback if not using conda)
install:
	pip install -r requirements.txt
	pip install -e .

# Run test suite with coverage (default CI target)
test:
	conda run -n $(CONDA_ENV) pytest tests/test_boostaroota.py -v --cov=boostaroota --cov-report=xml --cov-report=term

# Alias for test
test-verbose: test

# Fast test run without coverage
test-quick:
	conda run -n $(CONDA_ENV) pytest tests/test_boostaroota.py -q

# Run example validation script
example:
	conda run -n $(CONDA_ENV) python examples/run_example.py

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage*" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
