# Breast Cancer Prediction & XAI Research Framework

A research-oriented machine-learning and Explainable AI (XAI) framework comparing **TabPFN-2.5**, **CatBoost**, **FT-Transformer**, and **XGBoost** on the Breast Cancer Wisconsin (Diagnostic) dataset (WDBC).

> **Safety Notice:** This system is an academic research and educational platform only. It is not a diagnostic medical device and must never substitute for biopsy pathology, clinical imaging, or certified medical judgment.

## Research Scope & Architecture

| Item | Research Implementation |
|---|---|
| **Dataset** | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository (569 cases, 30 continuous features) |
| **Primary Model** | **TabPFN-2.5** (Prior-Fitted Network in-context transformer) |
| **Deep Learning Benchmark** | **FT-Transformer** (Feature Tokenizer + Transformer in PyTorch with Pre-LN & early stopping) |
| **Gradient Boosting Benchmarks** | **CatBoost** & **XGBoost** (with Optuna hyperparameter optimization) |
| **Evaluation Metrics** | 12 Metrics: Accuracy, Precision, F1-Score, Recall, ROC-AUC, Sensitivity, Specificity, Balanced Accuracy, MCC, NPV, Log Loss, Brier Score, plus Expected Calibration Error (ECE) |
| **Calibration Analysis** | Reliability diagrams, Platt Scaling, Isotonic Regression, 10-bin ECE pre/post comparison |
| **Explainable AI (XAI)** | TreeExplainer & KernelExplainer SHAP (Beeswarm, Summary, Bar, Waterfall plots), Permutation & Global Feature Importance |
| **User Interface** | Dedicated **Streamlit Research Workbench** (`research/app.py`) + Legacy Educational Workbench (`cdss/`) |

## Repository Layout

```text
Breast_Cancer_Pridiction/
├── AGENTS.md                         # Workflow rules and active scope
├── PRD.md                            # Product and research requirements
├── PLAN.md                           # Implementation plan and audit change log
├── DESIGN.md                         # UI/UX design specifications
├── API.md                            # Interface contracts
├── DATABASE.md                       # Data policy and artifact storage
├── TESTING.md                        # Verification and test instructions
├── MCP.md                            # Customization and tools reference
├── README.md                         # Project documentation
│
├── research/                         # Research Framework
│   ├── app.py                        # Streamlit Research Workbench (10 modules)
│   ├── config.py                     # Central configuration & parameters
│   ├── requirements.txt              # Research dependencies
│   ├── run_experiments.py           # Master 5-fold CV & evaluation pipeline
│   ├── src/
│   │   ├── data/                     # WDBC data loader & stratified splitter
│   │   ├── models/                   # TabPFN, CatBoost, FT-Transformer, XGBoost
│   │   ├── training/                 # 5-fold CV & master trainer
│   │   ├── evaluation/               # 12 metrics, ECE, calibration, comparison
│   │   ├── explainability/           # SHAP studio & multi-angle feature importance
│   │   ├── visualization/           # Publication-quality ROC, PR, CM, calibration plots
│   │   └── ensemble/                 # Soft voting & stacking ensembles
│   ├── artifacts/
│   │   ├── models/                   # Serialized model pipelines
│   │   ├── reports/                  # Generated comparison plots & CSV tables
│   │   └── results/                  # Complete experiment JSON results
│   └── tests/                        # Research unit and integration tests
│
├── frontend/                         # Modern React Research Frontend
│   ├── src/                          # React 18 components, views & data
│   │   ├── components/               # 9 dedicated research views
│   │   │   ├── SciIeeeProofView.jsx  # Verifiable SCI & IEEE literature proofs
│   │   │   └── ModelParadigmProofView.jsx # Unique vs Hybrid architecture proofs
│   │   └── data/researchData.js      # Consolidated research metrics & citations
│   ├── public/reports/               # High-resolution benchmark figures
│   └── package.json                  # React + Vite dependencies
│
└── cdss/                             # Legacy Educational Workbench (Streamlit)
    ├── data/raw/wdbc.data            # Raw WDBC data
    ├── train_pipeline.py             # 5-model classical pipeline
    └── dashboard/app.py              # Educational dashboard
```

## Quick Start Commands

```bash
# 1. Run the Full Comparative Research Experiments
cd research
../cdss/.venv/bin/python run_experiments.py

# 2. Launch the React Research Frontend
cd frontend
npm run dev -- --port 3000

# 3. Launch the Streamlit Research Workbench
cd ../research
../cdss/.venv/bin/streamlit run app.py --server.port 8501

# 4. Run the Research Test Suite
cd ../research
../cdss/.venv/bin/pytest tests/ -v

# 4. (Optional) Run the Legacy Educational Workbench
cd ../cdss
.venv/bin/streamlit run dashboard/app.py --server.port 8502
```

## Mandatory Reading Order

1. `AGENTS.md`
2. `PRD.md`
3. `PLAN.md`
4. `DESIGN.md`
5. `API.md`
6. `DATABASE.md`
7. `TESTING.md`
8. `MCP.md`
