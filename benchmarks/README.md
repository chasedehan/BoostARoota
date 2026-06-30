# Benchmarks

Executable benchmarks comparing BoostARoota vs Boruta vs All Features.

## Quick start

Open and run the Jupyter notebook:

```bash
jupyter notebook benchmarks/benchmarks.ipynb
```

Each cell is independent for debugging. The final cell renders the benchmark table inline as markdown and DataFrame.

## Data

CSVs in `benchmarks/data/` downloaded directly from UCI:
- `wine_quality.csv` – Wine Quality (UCI 186), 6497 rows
- `adult.csv` – Adult/Income (UCI 2), 48842 rows
- `spambase.csv` – Spambase (UCI 94), 4601 rows
- `lsvt.csv` – LSVT Voice Rehabilitation (UCI 282), 126 rows, 310 features

If a CSV is missing, the run fails loudly – no synthetic data.

## Adding a dataset

1. Place CSV at `benchmarks/data/{name}.csv`
2. Register in `benchmarks/datasets.yaml`:

```yaml
datasets:
  - name: my_dataset
    display_name: "My Dataset"
    target: target_column
    task: classification
    metric: logloss
```

Path defaults to `benchmarks/data/{name}.csv`. Override with `source.path` if needed.

All datasets are one-hot encoded automatically. Categorical targets are factorized.

## Running manually

```bash
# Via notebook (recommended)
jupyter notebook benchmarks/benchmarks.ipynb
# run each cell sequentially – final cell renders table inline
```

## Output

Notebook renders a table matching README format:

|Data Set | Target | Boruta Time| BoostARoota Time |BoostARoota LogLoss|Boruta LogLoss|All Features LogLoss| BAR >= All |
...

Results are displayed inline as markdown table and pandas DataFrame (no CSV file written).

## Dependencies

- `pyyaml` for config parsing
- `boruta_py` is installed at runtime if missing (pip install Boruta)
- `pandas`, `scikit-learn`, `xgboost`
