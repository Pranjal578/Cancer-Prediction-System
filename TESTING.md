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

PNG/JPEG/TIFF uploads and scanned-PDF uploads also require the local
[Tesseract OCR engine](https://tesseract-ocr.github.io/tessdoc/Installation.html).
CSV, text, and text-based PDF extraction do not require that executable.

## Research Framework Automated Tests

Run research tests:
```bash
cd research
../cdss/.venv/bin/pytest tests/ -v
```

| Research Test Suite | Scope |
|---|---|
| `test_data.py` | WDBC loading, validation, label encoding, stratified train/test split, 5-fold CV integrity |
| `test_models.py` | Model instantiation, fit, predict, and predict_proba contracts for TabPFN, CatBoost, FT-Transformer, XGBoost |
| `test_evaluation.py` | All 12 evaluation metrics, specificity, NPV, ECE calculation (10 equal-width bins), Platt and Isotonic calibration |
| Streamlit Workbench | Headless execution test of `research/app.py` across all 10 modules on port 8501 |
| React Frontend Build | Production build and bundle verification via `cd frontend && npm run build` |

## Legacy Automated tests

Run:

```bash
.venv/bin/pytest -q
```

Current suite: **10 tests**.

| Test area | Checks |
|---|---|
| Dataset loader | WDBC shape, labels, feature count, duplicate handling, and breast-only scope. |
| Preprocessing | Median imputation, scaling, and absence of transformed missing values. |
| Model set | Presence of exactly the five required classifiers. |
| Training workflow | Validation metrics, recall-first selection, selected-model held-out-test metrics, selected pipeline, five-pipeline bundle, metadata, comparison chart, and confusion-matrix artefacts. |
| Report ingestion | Single-row CSV extraction, explicit text-label extraction, text-PDF extraction, image-OCR routing, incomplete-field reporting, and batch-CSV rejection. |

## End-to-end verification

1. Regenerate all outputs:

   ```bash
   cd cdss
   .venv/bin/python train_pipeline.py
   ```

   Confirm it prints five validation-comparison rows, the selected model's
   held-out-test metrics, and writes both `best_model.joblib` and
   `model_pipelines.joblib`.

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

4. In the dashboard, confirm the Signal Lab masthead, model-register sidebar, numbered prediction workflow, and three workbench tabs render without layout errors. Confirm the algorithm selector defaults to the recall-selected model, select another classifier, and check the five-model validation graph.

5. Upload a one-row CSV with explicitly labelled WDBC fields. Confirm values are prefilled, missing values (if any) are named, and a prediction is blocked until the review checkbox is selected. Do not use a real patient report for verification.

6. Inspect the Model comparison and Exploratory data analysis tabs for loaded charts and reports.

## Latest verification

On 2026-11-18: training completed with Support Vector Machine selected by validation recall and regenerated both the selected model and five-model bundle; all 10 tests passed; the dashboard harness confirmed 30 inputs, one uploader, one model selector, one review gate, and a successful Random Forest prediction; and the health/homepage routes returned `ok` and HTTP 200. CSV, text, and text-PDF extraction paths were verified. Image and scanned-PDF OCR is dependency-gated in this environment because the native Tesseract executable is unavailable.

On 2026-11-19: the Signal Lab visual redesign compiled successfully; all 10 tests passed; the Streamlit harness confirmed three workbench tabs, 30 inputs, uploader, selector, review gate, and alternate-model prediction; and the live health/homepage routes returned `ok` and HTTP 200.

## Notes on warnings

- A missing Streamlit script context warning can appear when using Streamlit's bare test harness. It is not emitted as a dashboard runtime failure.
- A Matplotlib cache warning may appear in restricted environments where the default user cache directory is not writable. It does not affect generated charts or model behavior.
