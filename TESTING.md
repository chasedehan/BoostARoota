# Testing BoostARoota

## Install dependencies (conda – recommended)

```bash
conda env create -n boostaroota -f environment.yml
conda activate boostaroota
```

Or update an existing environment:
```bash
conda env update -n boostaroota -f environment.yml
```

### pip fallback

```bash
pip install -r requirements.txt
```

Requirements: numpy>=1.21,<3.0, pandas>=1.5,<3.0, scikit-learn>=1.3,<2.0, xgboost>=1.7,<3.0, pytest>=7.0,<9.0, pytest-cov>=4.0

## Run test suite

With conda / Makefile:
```bash
make test          # pytest with coverage, CI target
make test-quick    # fast, no coverage
make test-verbose  # alias for make test
```

Direct:
```bash
conda run -n boostaroota pytest tests/test_boostaroota.py -q
```

The test suite covers:
- Binary classification (XGBoost, logloss)
- Regression (XGBoost, rmse)
- Multiclass classification (mlogloss)
- sklearn tree classifiers (ExtraTreesClassifier)
- sklearn regressors (ExtraTreesRegressor)
- fit / transform / fit_transform API
- Parameter validation (cutoff, iters, delta)
- Error handling (transform before fit, missing metric/clf)
- Warnings (metric+clf conflict, low delta, low max_rounds)
- Silent mode output suppression
- keep_vars_ attribute correctness
- Task auto-detection (regression vs classification)
- Cutoff aggressiveness
- max_rounds stopping criteria

## Example validation

```bash
conda activate boostaroota
make example
# or: python examples/run_example.py
```

This runs BoostARoota end-to-end on synthetic data for:
1. Classification with XGBoost
2. Regression with XGBoost
3. Classification with sklearn ExtraTreesClassifier
4. Regression with sklearn ExtraTreesRegressor

All four scenarios print kept features and transformed shapes, confirming the algorithm works correctly.

A simpler regression-only example is at `examples/regression_simple.py`.

## CI

GitHub Actions runs on push to `master`, Python 3.9–3.12 matrix, via conda (`environment.yml`), with coverage reporting to Codecov. CI runs the pytest assertion suite only.
