# Testing and Verification Guide

## Prerequisites

```bash
cd cdss
source .venv/bin/activate
```

Install dependencies when necessary:

```bash
pip install -r requirements.txt
```

## Automated tests

Run:

```bash
.venv/bin/pytest -q
```

Current suite: **5 tests**.

| Test area | Checks |
|---|---|
| Dataset loader | WDBC shape, labels, feature count, duplicate handling, and breast-only scope. |
| Preprocessing | Median imputation, scaling, and absence of transformed missing values. |
| Model set | Presence of exactly the five required classifiers. |
| Training workflow | Metrics, selection, saved pipeline, metadata, comparison chart, and confusion-matrix artefacts. |

## End-to-end verification

1. Regenerate all outputs:

   ```bash
   cd cdss
   .venv/bin/python train_pipeline.py
   ```

   Confirm it prints five rows of comparison metrics and writes `best_model.joblib`.

2. Start the interface:

   ```bash
   .venv/bin/streamlit run dashboard/app.py --server.headless=true --server.port 8502
   ```

3. Check the server:

   ```bash
   curl --fail http://127.0.0.1:8502/_stcore/health
   curl --fail http://127.0.0.1:8502/
   ```

   Expected: `ok` from health and HTTP 200 from the dashboard route.

4. In the dashboard, submit the default 30 values in the Prediction tab. Confirm one class result, two probability metrics, and no displayed error.

5. Inspect the Model comparison and Exploratory data analysis tabs for loaded charts and reports.

## Latest verification

On 2026-11-17: training completed, all five tests passed, the saved model made a prediction, the Streamlit form submitted successfully, and the health/homepage routes returned successfully.

## Notes on warnings

- A missing Streamlit script context warning can appear when using Streamlit's bare test harness. It is not emitted as a dashboard runtime failure.
- A Matplotlib cache warning may appear in restricted environments where the default user cache directory is not writable. It does not affect generated charts or model behavior.
