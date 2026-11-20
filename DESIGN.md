# Design Specification

## Design goals

- Make the workflow understandable to students and evaluators.
- Keep tumour input collection manageable despite 30 required values.
- Make the selected prediction and model-comparison evidence easy to distinguish.
- Allow a single local report to prefill, but never replace review of, the 30 WDBC measurements.
- Preserve a clear educational-use disclaimer.
- Present the application as a composed ML workbench with a clear visual hierarchy, rather than a generic dashboard-card layout.

## Information architecture

The Streamlit dashboard has exactly three tabs:

1. **Prediction**
   - A dark-green editorial masthead and persistent model-register sidebar establish the active model, reference cohort, selection protocol, and safety boundary.
   - The screen follows a numbered two-step workflow: first select an input source and inference strategy, then review the full measurement set.
   - An optional uploader accepts one CSV, UTF-8 text, PDF, or image up to 10 MB. It extracts only explicitly labelled WDBC values and never saves the upload.
   - A form groups all 30 WDBC fields as Mean, Standard-error, and Worst measurements. Uploaded values prefill matching fields; all remaining values stay editable.
   - Dataset medians populate initial values.
   - A classifier selector defaults to the recall-selected model and permits any of the five development-fitted pipelines.
   - Submitting is disabled by a required review confirmation and then runs the selected saved model pipeline.
   - The result uses a clear Benign or Malignant status and displays malignant/benign probabilities with a bar chart.

2. **Model comparison**
   - Opens with the recall-selected model's four validation metrics and a ranked validation leaderboard that marks the default pipeline.
   - Shows an interactive validation comparison graph and table: Accuracy, Precision, Recall, and F1-score for all five models.
   - Shows the validation performance chart and all five validation confusion matrices.
   - Explains the recall-first deterministic selection rule and shows the selected model's one-time held-out-test benchmark.

3. **Exploratory data analysis**
   - Presents class distribution, feature distributions, and correlation heatmap as an editorial data lens generated during training.

## Research Framework Architecture (React + FastAPI)

In addition to the legacy educational Streamlit workbench (`cdss/`), the research system comprises:

1. **Research ML Engine (`research/`)**
   - Independent model wrappers: TabPFN-2.5, CatBoost, FT-Transformer (PyTorch), XGBoost.
   - Stratified 80/20 train/test partition + stratified 5-fold cross-validation on training data.
   - Comprehensive 12-metric evaluation + ECE calculation across 10 equal-width bins.
   - Probability calibration suite (Platt Scaling, Isotonic Regression, reliability diagrams).
   - Multi-model Explainable AI module (TreeExplainer, KernelExplainer, PyTorch gradient integration) producing summary beeswarm, bar, and individual prediction waterfall plots.
   - Automated publication-quality figure generation (ROC, PR, calibration curves, confusion matrix grids, timing comparisons).

2. **FastAPI Backend (`backend/`)**
   - Serves high-performance endpoints for 4-model concurrent inference, metrics inspection, calibration curves, and SHAP explanations.
   - Strict CORS configuration and OpenAPI 3.0 interactive documentation.

3. **React Research Web Application (`frontend/`)**
   - Built with Vite, React 18, Lucide React icons, and custom responsive clinical CSS system.
   - Dedicated modules:
     - Research Dashboard (system overview, KPIs, leaderboard, multi-model bars)
     - Dataset Exploration (WDBC statistics, 30 continuous features, class balance)
     - Model Performance Comparison (consolidated 12-metric table, ROC/PR curves, confusion matrices)
     - 30-Feature Prediction Workbench (sample presets, multi-model inferences, real-time risk bars)
     - Explainable AI Studio (SHAP beeswarm, bar plots, cooperative game theory math)
     - Calibration Analysis (reliability diagrams, 10-bin ECE benchmarks, Platt vs Isotonic)
     - Academic Literature Proof (verifiable SCI & IEEE peer-reviewed papers with DOIs & dataset usage proofs)
     - Architectural Proof (rigorous classification of TabPFN/FT-Transformer/GBDT as Unique Models vs Stacking as Hybrid Model)
     - Research Report Viewer (IEEE-style formatted academic report)

## Interaction rules

- Each input is non-negative and accepts a decimal value.
- Upload extraction accepts a single 10 MB-or-smaller file. CSV uploads must have exactly one data row; batch triage remains unavailable.
- The parser only recognizes explicit WDBC labels and never infers FNA measurements from free-form clinical language. Image and scanned-PDF OCR requires a local Tesseract installation.
- Prediction is only available after valid trained artefacts exist.
- Prediction requires a user confirmation that all 30 values have been reviewed and are WDBC-compatible, whether they were entered manually or prefilled from an upload.
- Missing artefacts produce a clear command to train the system; the UI does not fall back to an untrained model.
- The UI does not collect patient identity, persist uploaded files or user input, or provide clinical recommendations.

## Visual behavior

- Benign results use a success state; Malignant results use an alert state.
- The visual language uses deep green for trusted workflow controls, amber for decision-stage markers, a paper-toned background, and red only for alert outcomes.
- Panels use top rules, typography, and spacing to structure information; they avoid a repeated rounded-card treatment.
- Metric tables use percentage formatting for easy comparison.
- The chosen algorithm's validation metrics are visible before prediction, and the comparison graph contains all five algorithms.
- Images stretch to their containing tab area for readability.
- The layout is responsive through a masthead, Streamlit columns, tabs, forms, and expanders; the masthead condenses to one column on narrow screens.

## Accessibility and safety

- Labels are derived from the actual WDBC feature names and remain visible with inputs.
- Visual charts supplement rather than replace textual metrics.
- The educational/non-diagnostic warning appears above all interactive controls.
- The uploader warns that hospital reports can contain unrelated measurements and that every extracted value requires human verification.
