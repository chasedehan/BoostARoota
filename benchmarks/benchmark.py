import time
import subprocess
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, mean_squared_error

from boostaroota import BoostARoota


def _ensure_boruta():
    try:
        from boruta import BorutaPy
        return BorutaPy
    except ImportError:
        print("Boruta not found, attempting pip install...", file=sys.stderr)
        for pkg in ("Boruta", "boruta_py", "boruta"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
                from boruta import BorutaPy
                return BorutaPy
            except Exception:
                continue
        raise ImportError(
            "Boruta is required for benchmarks but could not be installed automatically. "
            "Install with: pip install Boruta"
        )


def train_get_preds(X_train, y_train, X_test, metric, n_classes=None):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    param = {"eval_metric": metric, "verbosity": 0}
    if metric == "mlogloss":
        param["objective"] = "multi:softprob"
        if n_classes is None:
            n_classes = len(np.unique(y_train))
        param["num_class"] = n_classes
    elif metric in ("logloss", "auc", "aucpr", "error", "error_rate"):
        param["objective"] = "binary:logistic"
    else:
        param["objective"] = "reg:squarederror"
    bst = xgb.train(param, dtrain, num_boost_round=10)
    return bst.predict(dtest)


def prep_logloss(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        preds = np.vstack([1 - y_pred, y_pred]).T
        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        return log_loss(y_true, preds)
    else:
        preds = np.clip(y_pred, 1e-15, 1 - 1e-15)
        preds = preds / preds.sum(axis=1, keepdims=True)
        n_classes = preds.shape[1]
        labels = list(range(n_classes))
        return log_loss(y_true, preds, labels=labels)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def evaluate_classification(y_true, y_pred_br, y_pred_boruta, y_pred_all):
    return {
        "bar_logloss": prep_logloss(y_true, y_pred_br),
        "boruta_logloss": prep_logloss(y_true, y_pred_boruta),
        "all_logloss": prep_logloss(y_true, y_pred_all),
    }


def evaluate_regression(y_true, y_pred_br, y_pred_boruta, y_pred_all):
    return {
        "bar_rmse": rmse(y_true, y_pred_br),
        "boruta_rmse": rmse(y_true, y_pred_boruta),
        "all_rmse": rmse(y_true, y_pred_all),
    }


def run_fold(X_train, X_test, y_train, y_test, metric, task, n_classes=None):
    from sklearn.ensemble import RandomForestClassifier

    BorutaPy = _ensure_boruta()

    y_pred_all = train_get_preds(X_train, y_train, X_test, metric, n_classes=n_classes)

    t0 = time.time()
    br = BoostARoota(metric=metric, silent=True, task=task)
    br.fit(X_train, y_train)
    bar_time = time.time() - t0
    X_train_br = br.transform(X_train)
    X_test_br = br.transform(X_test)
    if X_train_br.shape[1] == 0:
        raise RuntimeError("BoostARoota selected zero features")
    y_pred_br = train_get_preds(X_train_br, y_train, X_test_br, metric, n_classes=n_classes)

    t0 = time.time()
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=1)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=1)
    feat_selector.fit(X_train.values, y_train.values)
    boruta_time = time.time() - t0
    X_train_boruta = feat_selector.transform(X_train.values)
    X_test_boruta = feat_selector.transform(X_test.values)
    if X_train_boruta.shape[1] == 0:
        X_train_boruta = X_train.values
        X_test_boruta = X_test.values
    y_pred_boruta = train_get_preds(X_train_boruta, y_train, X_test_boruta, metric, n_classes=n_classes)

    return {
        "bar_time": bar_time,
        "boruta_time": boruta_time,
        "y_pred_br": y_pred_br,
        "y_pred_boruta": y_pred_boruta,
        "y_pred_all": y_pred_all,
    }


def benchmark_dataset(X, y, metric="logloss", task="auto", folds=5, repeats=1):
    if task == "auto":
        y_unique = np.unique(y)
        if np.issubdtype(np.array(y).dtype, np.floating) or len(y_unique) > 20:
            task = "regression"
        else:
            task = "classification"

    is_classification = task != "regression"

    if is_classification:
        y = pd.Series(pd.factorize(y)[0], index=y.index)

    n_classes = len(np.unique(y)) if is_classification and metric == "mlogloss" else None

    results = []
    for rep in range(repeats):
        if is_classification:
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42 + rep)
            split_iter = kf.split(X, y)
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=42 + rep)
            split_iter = kf.split(X)
        for train_idx, test_idx in split_iter:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_res = run_fold(X_train, X_test, y_train, y_test, metric, task, n_classes=n_classes)

            if is_classification:
                metrics = evaluate_classification(
                    y_test.values,
                    fold_res["y_pred_br"],
                    fold_res["y_pred_boruta"],
                    fold_res["y_pred_all"],
                )
            else:
                metrics = evaluate_regression(
                    y_test.values,
                    fold_res["y_pred_br"],
                    fold_res["y_pred_boruta"],
                    fold_res["y_pred_all"],
                )

            results.append({
                "bar_time": fold_res["bar_time"],
                "boruta_time": fold_res["boruta_time"],
                **metrics,
            })

    df = pd.DataFrame(results)
    summary = df.mean().to_dict()
    summary["folds"] = folds
    summary["repeats"] = repeats
    summary["task"] = task
    return summary
