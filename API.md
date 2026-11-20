# Application Interface Contract

## Interface types

The system provides multiple interface layers:
1. **React Research Frontend (`frontend/`)**: Modern React application (Vite + React 18) featuring 9 full modules, including 30-feature prediction workbench, 12-metric evaluation comparisons, calibration analysis, SHAP explainability, verifiable SCI & IEEE academic literature proofs, architectural unique vs hybrid proofs, and IEEE research paper viewer.
2. **Research Streamlit Workbench (`research/app.py`)**: 10-module research platform providing multi-model comparative predictions (TabPFN-2.5, CatBoost, FT-Transformer, XGBoost), 12-metric evaluation comparisons, calibration analysis, and SHAP explainability.
3. **Educational Streamlit Workbench (`cdss/dashboard/app.py`)**: Legacy Signal Lab ML workbench for 5-model educational exploration.

## React Research Web Application Modules (`frontend/src/`)

| Module View | Purpose | Features & Visualizations |
|---|---|---|
| **1. Research Dashboard** | System overview & high-level comparative insights | Executive summary, key benchmark KPIs, priority rankings leaderboard, metric bar charts |
| **2. Dataset Exploration** | WDBC exploratory data analysis | 569 instances, 30 continuous features, class balance bars, feature tier descriptions |
| **3. 12-Metric Benchmark** | Comparative evaluation matrix | Consolidated 12-metric table, ROC & PR curves, confusion matrix multi-model grid |
| **4. Tumour Prediction** | Interactive multi-model inference workbench | Accepts 30 FNA continuous nuclear measurements; presets for Typical Benign & Malignant; instant probabilities |
| **5. Explainable AI (SHAP)** | Model interpretability & feature attributions | Model selector (TabPFN, CatBoost, FT-Transformer, XGBoost), SHAP beeswarm & bar plots, top 10 features table |
| **6. Probability Calibration** | Predictive reliability & risk quantification | Reliability diagrams, 10-bin ECE table, Brier score comparison before & after Platt/Isotonic scaling |
| **7. SCI & IEEE Proof** | Bibliographic & literature verification | Searchable catalog of 8 landmark SCI and IEEE publications using WDBC with DOIs, impact factors & dataset proofs |
| **8. Unique vs Hybrid Proof** | Structural & mathematical taxonomy | Formal proof classifying TabPFN, FT-Transformer, CatBoost, and XGBoost as Unique Models vs Stacking as Hybrid Model |
| **9. IEEE Research Paper** | Publication-ready academic manuscript | Abstract, literature review, methodology, mathematical formulations, print-to-PDF support |

## Research Streamlit Workbench Modules (`research/app.py`)

| Module | Purpose | Features & Visualizations |
|---|---|---|
| **1. Research Dashboard** | System overview & high-level comparative insights | Model comparison bar chart, key architecture summaries, quick metrics |
| **2. Dataset Exploration** | WDBC exploratory data analysis | 569 instances, 30 features, class balance, distribution histograms, correlation heatmap |
| **3. Model Performance Comparison** | Comparative benchmark across 12 metrics | Consolidated table, priority-ordered rankings (Acc > Prec > F1 > Recall), metric bar charts |
| **4. Individual Model Evaluation** | Deep diagnostic profile for each model | Confusion matrices, per-fold CV summaries, training & inference times |
| **5. Tumour Prediction Workbench** | Real-time comparative inference | Accepts 30 FNA continuous nuclear measurements; evaluates across all 4 models simultaneously |
| **6. Explainable AI (SHAP)** | Global SHAP feature explanations | SHAP beeswarm plots, global SHAP mean bar plots, game-theoretic mathematical formulation |
| **7. Feature Importance Analysis** | Multi-model feature ranking comparison | Side-by-side Top 10 tumour characteristics across all models; association vs causation notes |
| **8. Individual Case Explanation** | Local patient prediction transparency | Local Z-score feature deviation breakdowns, individual sample probability risk breakdown |
| **9. Probability Calibration** | Predictive reliability & risk quantification | Reliability diagrams, 10-bin ECE table, Brier score comparison before & after Platt/Isotonic scaling |
| **10. IEEE Research Paper** | Publication-ready academic manuscript | Abstract, literature review, methodology, mathematical formulations, references |

## Streamlit Runtime routes (`cdss/`)

| Route | Purpose | Expected response |
|---|---|---|
| `/` | Streamlit dashboard shell; the browser establishes the Streamlit session to run the UI. | HTTP 200 |
| `/_stcore/health` | Streamlit server health check. | `ok` |

The health route verifies server availability only; it is not a cancer-prediction API.

## Prediction input contract

The dashboard accepts one numeric value for each WDBC feature in this order:

1. `radius_mean` through `fractal_dimension_mean`
2. `radius_se` through `fractal_dimension_se`
3. `radius_worst` through `fractal_dimension_worst`

The exact canonical list is `WDBC_FEATURE_COLUMNS` in `cdss/src/pipeline/data_loader.py`. Inputs are assembled into a one-row DataFrame and passed directly to the selected saved sklearn pipeline.

### Optional upload prefill

The UI may process one local file up to 10 MB in the active Streamlit session:

- a single-row CSV;
- a UTF-8 `.txt` report;
- a text-based or scanned `.pdf`; or
- a `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff` image.

The parser maps only explicitly labelled WDBC feature/value pairs (for example,
`radius mean: 12.3`). It rejects multi-row CSV files, does not infer values
from narrative medical text, and does not persist the file or its extracted
text. Text-based PDFs use `pypdf`; images and scanned PDFs require a locally
installed Tesseract OCR engine. Any missing values must be entered manually,
and the user must confirm review of all 30 values before prediction.

### Algorithm selection

The dashboard defaults to the model selected by validation recall, but permits
the user to choose any of the five pipelines saved in `model_pipelines.joblib`.
Each is refit on the same development partition. The validation comparison is
for educational comparison only; it does not make any model clinically valid.

## Prediction output contract

The UI displays:

- A class label: `Benign` or `Malignant`.
- The model's malignant probability.
- The complementary benign probability.
- The chosen algorithm name.

The saved pipeline is selected by validation recall (with F1-score, precision,
and accuracy as deterministic tie-breakers), refit on all development data, and
evaluated once on a separate held-out test set before use by the dashboard.

There are no response payloads, authentication rules, database records, or external service contracts.

## Future API work

Adding a REST or MCP interface is out of scope. It requires explicit user approval and updates to `PRD.md`, `DATABASE.md`, `TESTING.md`, `PLAN.md`, and this file before implementation.
