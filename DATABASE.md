# Data and Database Specification

## Database status

**No database is used.** The active proposed system intentionally has no patient store, audit store, ORM, SQLite file, or external database connection.

## Input dataset

| Item | Value |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) / WDBC |
| Local path | `cdss/data/raw/wdbc.data` |
| Source format | Headerless comma-separated file |
| Records | 569 |
| Raw columns | ID, diagnosis, and 30 numeric measurements |
| Target mapping | `B` = 0 (Benign), `M` = 1 (Malignant) |

## Data handling

1. The record `id` column is excluded because it is not a predictive tumour measurement.
2. Exact duplicate rows are counted and removed before training; the current source has zero duplicate rows.
3. All 30 predictor columns are converted to numeric values. Invalid numeric values become missing values.
4. Median imputation is fitted only within model training pipelines, preventing leakage from test data.
5. Standard scaling is also fitted within each model pipeline.

## Generated artefacts

Generated, non-database outputs live in `cdss/artifacts/breast/`:

- `best_model.joblib`
- `model_metadata.json`
- `training_summary.json`
- `model_comparison.csv` and `model_comparison.png`
- `confusion_matrices.png`
- `reports/feature_summary.csv`
- `reports/feature_correlations.csv`
- EDA PNG reports

These files may be regenerated at any time with `cdss/train_pipeline.py`; they do not contain patient identities or user-submitted predictions.
