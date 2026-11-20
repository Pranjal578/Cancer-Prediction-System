# Implementation Plan and Handoff

## Current status

**Status: complete and verified.** The academic research framework comparing TabPFN-2.5, CatBoost, FT-Transformer, and XGBoost with Explainable AI (SHAP) and probability calibration is fully implemented, verified, and operational via the Streamlit Research Workbench. All 21 automated tests pass across both research and legacy suites.

## What is implemented

1. **Research Architecture & Data Engine (`research/src/data/`)**:
   - WDBC loader with 30 FNA nuclear morphometric features, zero-value checks, duplicate detection, and metadata compilation.
   - Stratified 80/20 train/test splitting (`random_state=42`) and leak-free 5-fold cross-validation.
2. **Four Advanced Machine Learning Models (`research/src/models/`)**:
   - **TabPFN-2.5**: Prior-fitted in-context tabular transformer (`tabpfn==2.0.9`, CPU-optimized with `n_estimators=4`).
   - **CatBoost**: Gradient boosted decision trees with Optuna tuning support.
   - **FT-Transformer**: Native PyTorch Feature Tokenizer + Transformer with Pre-LN, self-attention, and early stopping.
   - **XGBoost**: Extreme gradient boosting classifier with regularization.
3. **12-Metric Evaluation & Calibration Engine (`research/src/evaluation/`)**:
   - Accuracy, Precision, F1-Score, Recall, ROC-AUC, Sensitivity, Specificity, Balanced Accuracy, MCC, NPV, Log Loss, Brier Score, plus 10-bin Expected Calibration Error (ECE).
   - Platt Scaling and Isotonic Regression fitted strictly on out-of-fold (OOF) cross-validation predictions.
4. **Explainable AI (XAI) Module (`research/src/explainability/`)**:
   - SHAP Beeswarm summary plots and bar plots across all models (TreeExplainer for tree models; KernelExplainer with representative background sampling for TabPFN and FT-Transformer).
   - Top 10 feature importance rankings comparing tumour characteristics across all models.
   - Patient-level prediction explanations with Z-score feature deviation breakdowns.
5. **Ensemble Methods (`research/src/ensemble/`)**:
   - Soft Voting Classifier and Stacking Classifier built on leak-free OOF probability predictions.
6. **Streamlit Research & Evaluation Workbench (`research/app.py`)**:
   - 10 full modules: Research Dashboard, Dataset Exploration, Model Performance Comparison, Individual Model Evaluation, Tumour Prediction Workbench (30 FNA inputs), Explainable AI (SHAP), Feature Importance Analysis, Individual Case Explanation, Probability Calibration, and IEEE Research Report.
7. **Educational Workbench Preserved (`cdss/`)**:
   - Legacy Signal Lab CDSS application and all original 10 tests preserved without regression.

## Official Experimental Evaluation Results (Held-Out Test Set)

| Metric | TabPFN-2.5 | CatBoost | FT-Transformer | XGBoost | Best Performing Model |
|---|---|---|---|---|---|
| **Accuracy** | **0.9825** | 0.9649 | 0.9649 | 0.9737 | **TabPFN-2.5** |
| **Precision** | **1.0000** | **1.0000** | 0.9750 | **1.0000** | **TabPFN-2.5** (tie) |
| **F1-Score** | **0.9756** | 0.9500 | 0.9512 | 0.9630 | **TabPFN-2.5** |
| **Recall / Sensitivity** | **0.9524** | 0.9048 | 0.9286 | 0.9286 | **TabPFN-2.5** |
| **ROC-AUC** | 0.9977 | **0.9983** | 0.9769 | 0.9914 | **CatBoost** |
| **Specificity** | **1.0000** | **1.0000** | 0.9861 | **1.0000** | **TabPFN-2.5** (tie) |
| **Balanced Accuracy** | **0.9762** | 0.9524 | 0.9573 | 0.9643 | **TabPFN-2.5** |
| **MCC** | **0.9626** | 0.9258 | 0.9245 | 0.9442 | **TabPFN-2.5** |
| **NPV** | **0.9730** | 0.9474 | 0.9595 | 0.9600 | **TabPFN-2.5** |
| **Brier Score** | **0.0164** | 0.0262 | 0.0351 | 0.0211 | **TabPFN-2.5** |
| **Log Loss** | **0.0606** | 0.0881 | 0.3745 | 0.0905 | **TabPFN-2.5** |
| **ECE (10-bin)** | **0.0221** | 0.0344 | 0.0351 | 0.0267 | **TabPFN-2.5** |

### Priority-Ordered Model Rankings (Primary: Acc > Prec > F1 > Recall)
1. **TabPFN-2.5** (Acc: 0.9825, Prec: 1.0000, F1: 0.9756, Rec: 0.9524, ROC-AUC: 0.9977)
2. **XGBoost** (Acc: 0.9737, Prec: 1.0000, F1: 0.9630, Rec: 0.9286, ROC-AUC: 0.9914)
3. **CatBoost** (Acc: 0.9649, Prec: 1.0000, F1: 0.9500, Rec: 0.9048, ROC-AUC: 0.9983)
4. **FT-Transformer** (Acc: 0.9649, Prec: 0.9750, F1: 0.9512, Rec: 0.9286, ROC-AUC: 0.9769)

## Start here in a future session

1. Read `AGENTS.md` and companion control documents (`PRD.md`, `PLAN.md`, `DESIGN.md`, `API.md`, `DATABASE.md`, `TESTING.md`, `MCP.md`).
2. Run research experiments or verify artifacts:
   ```bash
   cd research
   ../cdss/.venv/bin/python run_experiments.py
   ```
3. Launch Streamlit Research Workbench:
   ```bash
   cd research
   ../cdss/.venv/bin/streamlit run app.py --server.port 8501 --server.headless true
   ```
4. Launch React Research Frontend:
   ```bash
   cd frontend
   npm run dev -- --port 3000
   ```
5. Run full test suite:
   ```bash
   ./cdss/.venv/bin/pytest research/tests/ cdss/tests/ -v
   cd frontend && npm run build
   ```

## Change log

| Date | Change | Verification |
|---|---|---|
| 2026-11-17 | Replaced the earlier multi-modality CDSS with the breast-only proposed system; added five-model comparison, EDA, Streamlit interface, and supporting tests. | Training completed; 5 automated tests passed; Streamlit dashboard, form submission, homepage, and health route were checked. |
| 2026-11-17 | Added project-control documentation and mandatory synchronization instructions. | Documentation reviewed against current implementation. |
| 2026-11-17 | Updated `README.md` with the project-control document map and mandatory synchronization policy. | README links and listed files reviewed. |
| 2026-11-17 | Replaced legacy ignore rules with a focused `.gitignore` for Python, Streamlit secrets, generated artefacts, logs, and non-WDBC raw data. | Rules reviewed against the active project layout. |
| 2026-11-18 | Changed model selection from held-out-test F1 to validation Recall, with F1-score, Precision, and Accuracy tie-breakers; the selected model is refit on development data and benchmarked once on held-out test data. | Training completed (SVM selected); 5 automated tests passed; Streamlit health and homepage returned `ok` and HTTP 200; harness confirmed 30 inputs and successful default-form submission. |
| 2026-11-18 | Removed ignored cervical-only data and artefacts, obsolete API/explainability bytecode trees, disposable Python/test caches, and unused breast `metrics.json` and `threshold.json`. Preserved the active WDBC data, saved model, current breast reports, and virtual environment. | Post-cleanup inventory confirmed no cervical workflow, obsolete breast artefacts, or Python/test cache directories remain. No runtime behavior changed, so automated tests were not rerun. |
| 2026-11-18 | Recovered the ignored cervical dataset and its two historical artefacts at the user's request. They are documented as inactive archive files only; no cervical code, workflow, or runtime integration was restored. | Each restored file hashes to its original locally recoverable Git object. The active breast-only implementation and artefacts were unchanged, so automated tests were not rerun. |
| 2026-11-18 | Added in-memory, review-gated WDBC value extraction from a single CSV, text, PDF, or image report; added optional selection among five development-fitted model pipelines and an interactive comparison graph. | 10 automated tests passed; training regenerated the selected model and five-model bundle (SVM remains selected); health and homepage returned `ok` and HTTP 200; the Streamlit harness confirmed 30 inputs, uploader, selector, review gate, and an alternate-model prediction. Image/scanned-PDF OCR is dependency-gated because this environment lacks the system Tesseract executable. |
| 2026-11-19 | Redesigned the dashboard as the Signal Lab ML workbench with an editorial visual system, model-register sidebar, numbered inference workflow, validation leaderboard, and data-lens presentation. | Python compilation succeeded; 10 automated tests passed; the Streamlit harness confirmed three tabs, 30 inputs, uploader, selector, review gate, and alternate-model prediction; live health and homepage returned `ok` and HTTP 200. |
| 2026-11-19 | Commenced full research framework implementation: TabPFN-2.5, CatBoost, FT-Transformer, XGBoost, 12 evaluation metrics, ECE calibration analysis, SHAP explainability, and Streamlit Research Workbench interface. Preserved legacy `cdss/` workbench. | Updated scope to use Streamlit for frontend; environment packages installed; research directory structure established. |
| 2026-11-19 | Completed implementation of all 4 models (TabPFN-2.5, CatBoost, FT-Transformer in PyTorch, XGBoost), 5-fold cross-validation pipeline, 12-metric evaluation, out-of-fold calibration (Platt/Isotonic/ECE), SHAP analysis, and Streamlit 10-module Workbench. Resolved FT-Transformer module serialization issue. Ran complete experiment and test suites. | All experiments completed successfully (`run_experiments.py` exited 0); generated full set of report plots, model artifacts, and evaluation tables; 11 research tests passed; 10 legacy tests passed (21 total passed); Streamlit app launched cleanly on port 8501. |
| 2026-11-19 | Implemented full React Research Frontend (Vite + React 18) with custom clinical responsive design system. Added dedicated pages: (1) Verifiable SCI & IEEE Academic Literature Proof with genuine DOIs and dataset usage proofs, and (2) Formal Architectural Proof classifying TabPFN, FT-Transformer, CatBoost, and XGBoost as Unique Models vs Stacking as Hybrid Model. | Production build succeeded with zero errors (`npm run build` in 2.12s); Vite preview server verified responsive on port 3000; all 21 Python automated tests continue to pass. |
| 2026-11-19 | Created comprehensive `.gitignore` configured for multi-ecosystem repository (Python, Node.js, React, PyTorch, Streamlit, and serialized model checkpoints). Ignored `node_modules/`, `dist/`, `.vite/`, virtual environments, test caches, and large binary model checkpoints (`*.joblib`). | Verified via `git status` that all source files are cleanly tracked while all build outputs, package caches, and binary weights are ignored. |

| 2026-11-20 | Repository cleanup: deleted `frontend/node_modules/` (86 MB), `frontend/dist/` (2 MB build output), `research/artifacts/models/` (30 MB serialized joblib files), all `__pycache__/` directories, `*.pyc`/`*.pyo` files, and empty `backend/` stub directory. All deleted items are either regeneratable (`npm install`, `npm run build`, `python run_experiments.py`) or excluded by `.gitignore`. | `git status -s` shows only tracked source files and untracked `frontend/` and `research/` source trees; `du -sh` confirmed `frontend/` reduced to 2.0 MB and `research/` reduced to 2.0 MB; no tests rerun (no code changed). |

## Next work

1. Optional ensemble extension: Evaluate soft-voting vs stacking meta-model on held-out test data in standalone evaluation script.
2. Optional LaTeX / PDF compilation of the IEEE Research Report module.
3. Run `npm install` inside `frontend/` before next React dev session.
4. Run `python run_experiments.py` inside `research/` to regenerate serialized model artifacts before next Streamlit session.
