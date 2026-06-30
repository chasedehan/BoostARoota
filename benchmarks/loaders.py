import os
import pandas as pd


def load_csv(path, target, sep=","):
    if not (path.startswith("http://") or path.startswith("https://")):
        if not os.path.isabs(path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_path = os.path.join(repo_root, path)
        else:
            abs_path = path
        if not os.path.exists(abs_path):
            raise RuntimeError(
                f"Dataset CSV not found at '{abs_path}'. "
                f"Place CSV in benchmarks/data/ and register in benchmarks/datasets.yaml. "
                f"No synthetic data will be created."
            )
        path = abs_path

    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV from '{path}': {e}") from e

    if df.empty:
        raise RuntimeError(f"CSV at '{path}' is empty")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV '{path}'. Available: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])

    if y.isnull().any():
        raise RuntimeError(f"Target column '{target}' contains null values")

    return X, y


def prepare_features(X):
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype(str).str.strip()
    X = pd.get_dummies(X, drop_first=False)
    X.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(',', '_').replace(' ', '_') for c in X.columns]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X


def encode_target(y):
    y = y.copy()
    if y.dtype == object:
        y = y.astype(str).str.strip()
        y = y.str.replace(".", "", regex=False)
    if y.dtype == object or pd.api.types.is_categorical_dtype(y):
        codes, uniques = pd.factorize(y)
        return pd.Series(codes, index=y.index)
    return y
