import os
import sys

# Ensure we import the local repo version, not a pip-installed copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from boostaroota.boostaroota import BoostARoota
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate regression data with some noise features
X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=0.1, random_state=42)
X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Testing BoostARoota with regression (rmse metric)...")
br = BoostARoota(metric='rmse', silent=False)
br.fit(X_train, y_train)
print(f"Keep vars: {list(br.keep_vars_)}")
print(f"Number of features kept: {len(br.keep_vars_)} out of {X.shape[1]}")

X_train_transformed = br.transform(X_train)
print(f"Transformed shape: {X_train_transformed.shape}")

print("\nRegression test PASSED!")
