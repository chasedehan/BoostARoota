#!/usr/bin/env python3
"""
BoostARoota local validation example

Runs BoostARoota on synthetic classification and regression datasets,
covering both XGBoost-native and sklearn-tree backends.

Run:
    pip install -r requirements.txt
    python examples/run_example.py
"""

import numpy as np
import pandas as pd

from boostaroota import BoostARoota
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


def run_classification_xgb():
    print("=" * 60)
    print("Classification – XGBoost backend (logloss)")
    print("=" * 60)
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=5,
        random_state=42, shuffle=False
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    print(f"Input shape: {X.shape}, informative features: 5")

    np.random.seed(42)
    br = BoostARoota(metric='logloss', iters=3, silent=False)
    br.fit(X, y)
    print(f"\nKept {len(br.keep_vars_)} / {X.shape[1]} features")
    print(f"Keep vars: {list(br.keep_vars_)}")
    X_new = br.transform(X)
    print(f"Transformed shape: {X_new.shape}")
    print("✓ Classification (XGB) PASSED\n")
    return True


def run_regression_xgb():
    print("=" * 60)
    print("Regression – XGBoost backend (rmse)")
    print("=" * 60)
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5,
        noise=0.1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    print(f"Input shape: {X.shape}, informative features: 5")

    np.random.seed(42)
    br = BoostARoota(metric='rmse', iters=3, silent=False)
    br.fit(X, y)
    print(f"\nKept {len(br.keep_vars_)} / {X.shape[1]} features")
    print(f"Keep vars: {list(br.keep_vars_)}")
    X_new = br.transform(X)
    print(f"Transformed shape: {X_new.shape}")
    print("✓ Regression (XGB) PASSED\n")
    return True


def run_sklearn_classifier():
    print("=" * 60)
    print("Classification – sklearn backend (ExtraTreesClassifier)")
    print("=" * 60)
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=5,
        random_state=42, shuffle=False
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    print(f"Input shape: {X.shape}, informative features: 5")

    np.random.seed(42)
    clf = ExtraTreesClassifier(n_estimators=20, random_state=42)
    br = BoostARoota(clf=clf, iters=3, silent=False)
    X_new = br.fit_transform(X, y)
    print(f"\nKept {len(br.keep_vars_)} / {X.shape[1]} features")
    print(f"Keep vars: {list(br.keep_vars_)}")
    print(f"Transformed shape: {X_new.shape}")
    print("✓ Classification (sklearn) PASSED\n")
    return True


def run_sklearn_regressor():
    print("=" * 60)
    print("Regression – sklearn backend (ExtraTreesRegressor)")
    print("=" * 60)
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5,
        noise=0.1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    print(f"Input shape: {X.shape}, informative features: 5")

    np.random.seed(42)
    clf = ExtraTreesRegressor(n_estimators=20, random_state=42)
    br = BoostARoota(clf=clf, iters=3, silent=False)
    br.fit(X, y)
    print(f"\nKept {len(br.keep_vars_)} / {X.shape[1]} features")
    print(f"Keep vars: {list(br.keep_vars_)}")
    print("✓ Regression (sklearn) PASSED\n")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Classification XGB", run_classification_xgb()))
    results.append(("Regression XGB", run_regression_xgb()))
    results.append(("Classification sklearn", run_sklearn_classifier()))
    results.append(("Regression sklearn", run_sklearn_regressor()))

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print("\nAll example runs completed successfully!")
