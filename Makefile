# BoostARoota Makefile

.PHONY: test test-verbose install clean example

# Default Python interpreter
PYTHON ?= python3

# Install dependencies from requirements.txt
install:
	$(PYTHON) -m pip install -r requirements.txt

# Run test suite
test:
	$(PYTHON) -m pytest tests/test_boostaroota.py -q

# Run test suite with verbose output
test-verbose:
	$(PYTHON) -m pytest tests/test_boostaroota.py -v

# Run example validation script
example:
	$(PYTHON) examples/run_example.py

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
