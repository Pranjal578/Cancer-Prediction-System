# Product Requirements Document

## Product

**Name:** Breast Cancer Prediction System  
**Purpose:** Provide an educational Streamlit application that compares five machine-learning classifiers on WDBC data and predicts a breast-tumour class from 30 FNA measurements.

## Users

- Students and project evaluators reviewing an end-to-end ML workflow.
- Users entering known WDBC-compatible measurements for demonstration purposes.

The product is not a diagnostic device and is not for clinical decision-making.

## Functional requirements

| ID | Requirement | Current implementation |
|---|---|---|
| FR-01 | Load the Breast Cancer Wisconsin (Diagnostic) dataset. | Loads the local WDBC file, removes the record ID, validates labels, and checks/removes exact duplicate rows. |
| FR-02 | Use 569 records and 30 numeric tumour features. | Implemented and covered by tests. |
| FR-03 | Preprocess model inputs consistently. | Median imputation and standard scaling are inside each model pipeline. |
| FR-04 | Perform EDA. | Generates descriptive statistics, class distribution, feature distributions, and feature correlations. |
| FR-05 | Train Logistic Regression, Decision Tree, Random Forest, SVM, and KNN. | Implemented as the complete model set. |
| FR-06 | Evaluate models with Accuracy, Precision, Recall, F1-score, and confusion matrices. | Implemented on a stratified 80/20 held-out test split. |
| FR-07 | Select and save a suitable model. | Selects highest F1-score; ties use Recall, Precision, then Accuracy. Saves `best_model.joblib`. |
| FR-08 | Provide an interactive Streamlit prediction interface. | User enters all 30 features and receives Benign/Malignant output with probabilities. |
| FR-09 | Visualize model performance and analysis. | Comparison chart, confusion matrices, and EDA plots are available in the dashboard. |

## Explicit exclusions

- Other cancer modalities, including cervical cancer.
- XGBoost, SMOTE, SHAP, and clinical-risk thresholds.
- FastAPI/REST prediction services.
- Databases, audit logs, patient history, document ingestion, and batch processing.

## Acceptance criteria

1. Training creates the selected model, metadata, comparison CSV/chart, confusion matrices, and EDA reports.
2. The dashboard renders three tabs and 30 feature inputs without exceptions.
3. The dashboard submits a prediction through the saved model pipeline.
4. Automated tests pass.
5. The application server responds at `/` and `/_stcore/health`.

## Safety statement

The predicted class and probability are educational ML outputs only. They must not replace clinical evaluation, imaging, biopsy, pathology, or professional medical judgement.
