# Testing BoostARoota

## Install dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, xgboost, scikit-learn, pytest

## Run test suite

```bash
pytest tests/test_boostaroota.py -q
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

## Run example validation

```bash
python examples/run_example.py
```

This runs BoostARoota end-to-end on synthetic data for:
1. Classification with XGBoost
2. Regression with XGBoost
3. Classification with sklearn ExtraTreesClassifier
4. Regression with sklearn ExtraTreesRegressor

All four scenarios print kept features and transformed shapes, confirming the algorithm works correctly.
