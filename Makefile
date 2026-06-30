# BoostARoota Makefile

.PHONY: test test-verbose test-quick install clean example

# Install via pip
install:
	pip install -r requirements.txt
	pip install -e .

# Run test suite with coverage (default CI target)
test:
	pytest tests/test_boostaroota.py -v --cov=boostaroota --cov-report=xml --cov-report=term

# Alias for test
test-verbose: test

# Fast test run without coverage
test-quick:
	pytest tests/test_boostaroota.py -q

# Run example validation script
example:
	python examples/run_example.py

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage*" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
