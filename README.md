# Breast Cancer Prediction System

An interactive machine-learning application that classifies breast tumours as **Benign** or **Malignant** from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. It implements the proposed breast-only system: preprocessing, exploratory analysis, comparison of five classifiers, selection of the best model, and a Streamlit prediction interface.

> This project is for education and decision-support demonstrations only. It is not a medical device and must not replace clinical examination, imaging, pathology, or professional medical judgement.

## Scope

| Item | Implementation |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository |
| Records and inputs | 569 records and 30 numerical tumour-characteristic features |
| Target | Benign (`B`) or Malignant (`M`) |
| Models compared | Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors |
| Evaluation | Accuracy, Precision, Recall, F1-score, and a confusion matrix for every model |
| Model selection | Highest held-out-test F1-score; ties use Recall, Precision, then Accuracy |
| User interface | Streamlit form for the 30 WDBC parameters, prediction result, model metrics, and visual reports |

The application intentionally supports **breast cancer only**. Cervical prediction, XGBoost, SHAP, FastAPI services, batch triage, document parsing, and audit-log workflows are outside this proposed-system scope and are not used by the active training or dashboard flow.

## Modules

1. **Dataset collection and management** — loads the local WDBC file, removes the record ID, checks/removes exact duplicate rows, and maps `M`/`B` labels to `1`/`0`.
2. **Data preprocessing and EDA** — converts inputs to numeric values, uses median imputation and feature scaling within model pipelines, and produces class-distribution, feature-distribution, correlation, and descriptive-statistics reports.
3. **Model training** — trains the five specified classifiers with a stratified 80/20 train/test split.
4. **Model evaluation and selection** — saves `model_comparison.csv`, a comparison chart, and confusion matrices; then serializes the selected classifier.
5. **Cancer prediction** — applies the saved pipeline to a single 30-feature input and returns a Benign/Malignant result with probabilities.
6. **Streamlit frontend** — provides prediction, comparison, and EDA tabs.

## Project layout

```text
Breast_Cancer_Pridiction/
├── wdbc.data                         # Original WDBC source data
├── README.md
├── AGENTS.md                         # Required workflow and scope guardrails
├── PRD.md                             # Product requirements and exclusions
├── PLAN.md                            # Current state and change log
├── DESIGN.md                          # Streamlit design specification
├── API.md                             # Application route and input/output contract
├── DATABASE.md                        # Dataset and artefact policy
├── TESTING.md                         # Verification commands and expectations
├── MCP.md                             # MCP boundary and future-integration policy
└── cdss/
    ├── data/raw/wdbc.data             # Application copy of WDBC data
    ├── train_pipeline.py              # End-to-end training command
    ├── dashboard/app.py               # Streamlit application
    ├── artifacts/breast/              # Generated model and reports
    ├── src/analysis/eda.py            # EDA report generation
    ├── src/pipeline/                  # Data loading and preprocessing
    ├── src/training/trainer.py        # Five-model comparison and selection
    └── tests/                         # Automated checks
```

## Project control documents

The root-level control documents preserve the current implementation context for future work. Before changing the project, read them in this order:

1. [AGENTS.md](AGENTS.md)
2. [PRD.md](PRD.md)
3. [PLAN.md](PLAN.md)
4. [DESIGN.md](DESIGN.md)
5. [API.md](API.md)
6. [DATABASE.md](DATABASE.md)
7. [TESTING.md](TESTING.md)
8. [MCP.md](MCP.md)

`AGENTS.md` requires every implementation, configuration, dependency, data-handling, test, or runtime change to immediately update the affected control documents and add a verified entry to `PLAN.md`.

## Run locally

```bash
cd cdss
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_pipeline.py
streamlit run dashboard/app.py
```

Run the automated checks with:

```bash
cd cdss
.venv/bin/pytest -q
```

## Generated outputs

Training writes the following to `cdss/artifacts/breast/`:

- `best_model.joblib` — selected model and its preprocessing steps.
- `model_metadata.json` — selected-model name, metrics, data summary, and selection rule.
- `model_comparison.csv` and `model_comparison.png` — comparison of all five classifiers.
- `confusion_matrices.png` — confusion matrix for each classifier.
- `reports/` — class distribution, all feature distributions, correlation heatmap, feature summary, and correlation matrix.
