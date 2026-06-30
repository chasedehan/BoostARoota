# BoostARoota

A fast, practical feature selection algorithm built on XGBoost — with support for other scikit-learn tree-based models too.

Boruta was a great step forward for automated feature selection with Random Forests, but it can be slow on high-dimensional data and doesn't always transfer well to boosting models or other modern algorithms. Regularized linear methods like LASSO, Ridge, and Elastic Net have the opposite problem: they work well for linear models but not so much for trees and ensembles.

BoostARoota takes the core idea from Boruta — compare real features against randomized "shadow" features — and adapts it for XGBoost. In practice this means much faster runtimes and better feature sets for gradient boosting, while keeping the API familiar if you've used scikit-learn before.

## Installation

```bash
pip install boostaroota
```

Requires Python 3.9+, pandas, numpy, scikit-learn, and xgboost. See `requirements.txt` for tested version ranges.

## Quick start

BoostARoota expects a pandas DataFrame with numeric columns. If you have categoricals, one-hot encode them first (e.g. with `pd.get_dummies`). This is important — the shadow feature logic assumes numeric input, and string columns that get expanded can blow up your feature space.

```python
from boostaroota import BoostARoota
import pandas as pd

# One-hot encode categoricals
X = pd.get_dummies(X)

# Pick an XGBoost metric you like. For multiclass, use "mlogloss".
br = BoostARoota(metric="logloss")

br.fit(X, y)

# Selected features
br.keep_vars_

# Filter down to just the useful columns
X_selected = br.transform(X)
```

That's the basic flow: `fit`, inspect `keep_vars_`, then `transform`.

A couple of gotchas I've run into:
- If a numeric column is read in as object/string, `get_dummies` will explode it into lots of dummy columns. Cast to numeric first if that's not what you want.
- For multiclass problems, BoostARoota currently only supports `mlogloss` as the eval metric.

You can see a more complete walkthrough in [`odsc_west/demo.py`](odsc_west/demo.py).

## Using other tree models

You aren't limited to XGBoost. Any scikit-learn tree-based estimator with `feature_importances_` will work, though you may need to tune `cutoff`, `iters`, etc. a bit since the defaults were chosen with XGBoost in mind.

```python
from sklearn.ensemble import ExtraTreesClassifier
from boostaroota import BoostARoota

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
br = BoostARoota(clf=clf)

X_new = br.fit_transform(X, y)
```

If you pass both `metric` and `clf`, the classifier takes precedence and the metric is ignored (you'll get a warning).

## Parameters

Defaults work well for most tabular datasets, but here's what you can tweak:

- **metric** (str, default=None) – XGBoost eval metric like `"logloss"`, `"auc"`, `"rmse"`, `"mlogloss"`, etc. Required if you aren't passing your own `clf`. For multiclass, use `"mlogloss"`.
- **clf** (estimator, default=None) – A scikit-learn tree model. Leave as None to use XGBoost internally.
- **cutoff** (float > 0, default=4) – Shadow importance is averaged and divided by this value to set the removal threshold. Higher = more conservative (fewer features removed). Lower = more aggressive.
- **iters** (int > 0, default=10) – How many times to retrain per round to smooth out importance estimates. Don't use 1 — there's too much variance. Runtime scales linearly with this.
- **max_rounds** (int > 0, default=100) – Hard cap on elimination rounds. The default is intentionally high; you'll rarely hit it unless the data is pathological or `delta` is very small.
- **delta** (float, 0 < delta <= 1, default=0.1) – Minimum fraction of features that must be removed to continue to the next round. `0.1` means at least 10% need to go. Set to 1.0 to force a single round. Very small values can over-prune.
- **silent** (bool, default=False) – Suppress per-iteration progress output. Warnings and errors still show.
- **task** ({"auto", "classification", "regression"}, default="auto") – How to configure XGBoost. Auto-detects based on `y`, but you can override.

## How it works

The intuition is straightforward:

1. Start with a one-hot encoded feature matrix.
2. Make a copy of every column and randomly shuffle each copy. These are the "shadow" features — they have the same distribution as the real ones but no relationship to the target.
3. Train XGBoost (or your chosen tree model) on the combined real + shadow matrix. Repeat `iters` times with different shuffles to get stable importance estimates.
4. For each feature, average its importance across iterations. Do the same for shadows.
5. Compute a cutoff: mean shadow importance divided by the `cutoff` parameter (default 4). This makes the bar higher than just beating random noise.
6. Drop any real feature whose mean importance is below that cutoff.
7. Repeat from step 2 with the reduced feature set until fewer than `delta` fraction of features are removed in a round, or `max_rounds` is hit.

What you get back is the set of features that consistently beat the shuffled versions — a simple but effective signal that they're actually useful to the model.

## Testing and examples

Install deps and run the suite:

```bash
pip install -r requirements.txt
pytest tests/test_boostaroota.py -q
# or
make test
```

For a quick end-to-end check across classification, regression, and sklearn backends:

```bash
make example
# or
python examples/run_example.py
```

See [TESTING.md](TESTING.md) for full details on what's covered.

## Notes

- Input must be a pandas DataFrame. Numpy arrays will need to be wrapped first.
- One-hot encoding is on you — BoostARoota doesn't do it automatically so you stay in control of how categoricals are handled.
- If you hit weird results, double-check dtypes after `get_dummies` and make sure the target `y` is in the expected format for your chosen metric.

## License

MIT — see [LICENSE](LICENSE).
