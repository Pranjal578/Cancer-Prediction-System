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
6. The data is split stratified into 60% training, 20% validation, and 20% held-out test partitions. The validation partition compares all models and selects the winner by recall; the selected pipeline is then refit on the combined 80% development partition and evaluated once on the held-out test partition.
7. A user upload is not training data and is never written to disk. The dashboard processes at most one 10 MB CSV, UTF-8 text, PDF, or image in memory, extracts only explicitly labelled WDBC values, and leaves all values subject to user review. Batch CSVs and patient-history storage are not supported.

## Generated artefacts

### Research Framework Artifacts (`research/artifacts/`)
- `models/tabpfn.joblib`, `models/catboost.joblib`, `models/ft_transformer.joblib`, `models/xgboost.joblib`
- `results/experiment_results.json`: Full 12-metric evaluation, ECE calibration comparisons, 5-fold CV summaries, timing benchmarks, and metadata.
- `reports/roc_curves_comparison.png`: Multi-model ROC curves with AUC annotations.
- `reports/pr_curves_comparison.png`: Precision-Recall curves with Average Precision.
- `reports/confusion_matrices_grid.png`: 2x2 grid of test confusion matrices.
- `reports/metrics_comparison_bars.png`: Comparative bar chart across primary metrics.
- `reports/calibration_comparison.png`: Reliability diagrams pre/post Platt & Isotonic calibration.
- `reports/feature_importance_top10.png`: Side-by-side Top 10 feature importance.
- `reports/*_shap_beeswarm.png`, `reports/*_shap_bar.png`: SHAP beeswarm and summary bar charts.
- `reports/model_comparison_table.csv`, `reports/model_rankings.csv`.

### Educational CDSS Artefacts (`cdss/artifacts/breast/`)

- `best_model.joblib`
- `model_pipelines.joblib` (all five pipelines refit on the development partition for optional educational selection)
- `model_metadata.json` (validation comparison, selected-model held-out-test benchmark, and selection rule)
- `training_summary.json`
- `model_comparison.csv` and `model_comparison.png`
- `confusion_matrices.png`
- `reports/feature_summary.csv`
- `reports/feature_correlations.csv`
- EDA PNG reports

These files may be regenerated at any time with `cdss/train_pipeline.py`; they do not contain patient identities or user-submitted predictions.

## Inactive recovered legacy materials

At the user's request, the following ignored files are retained only as an
archive recovered from the local Git object store:

- `cdss/data/raw/risk_factors_cervical_cancer.csv`
- `cdss/artifacts/cervical/metrics.json`
- `cdss/artifacts/cervical/threshold.json`

They are not read by the breast-only training pipeline or Streamlit dashboard,
do not reinstate a cervical workflow, and must not be used for prediction or
evaluation without an explicit scope change to the project documents.
