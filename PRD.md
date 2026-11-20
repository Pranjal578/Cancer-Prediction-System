# Product Requirements Document

## Product

**Name:** Breast Cancer Prediction System  
**Purpose:** Provide an educational Streamlit application that compares five machine-learning classifiers on WDBC data and predicts a breast-tumour class from 30 reviewed FNA measurements.

## Users

- Students and project evaluators reviewing an end-to-end ML workflow.
- Users entering known WDBC-compatible measurements for demonstration purposes, including values explicitly labelled in a single uploaded report.

The product is not a diagnostic device and is not for clinical decision-making.

## Functional requirements

| ID | Requirement | Current implementation |
|---|---|---|
| FR-01 | Load the Breast Cancer Wisconsin (Diagnostic) dataset. | Loads the local WDBC file, removes the record ID, validates labels, and checks/removes exact duplicate rows. |
| FR-02 | Use 569 records and 30 numeric tumour features. | Implemented and covered by tests. |
| FR-03 | Preprocess model inputs consistently. | Median imputation and standard scaling are inside each model pipeline. |
| FR-04 | Perform EDA. | Generates descriptive statistics, class distribution, feature distributions, and feature correlations. |
| FR-05 | Train Logistic Regression, Decision Tree, Random Forest, SVM, and KNN. | Implemented as the complete model set. |
| FR-06 | Evaluate models with Accuracy, Precision, Recall, F1-score, and confusion matrices. | Compares all five on a stratified validation set; evaluates the selected model once on a separate held-out test set. |
| FR-07 | Select and save a suitable model. | Selects highest validation Recall; ties use F1-score, Precision, then Accuracy. The winner is refit on all development data and saved as `best_model.joblib`. |
| FR-08 | Provide an interactive Streamlit prediction interface. | A professional ML workbench lets the user enter and confirm all 30 features, select any of the five fitted classifiers, inspect validation evidence, and receive Benign/Malignant output with probabilities. |
| FR-09 | Visualize model performance and analysis. | Interactive comparison graph, comparison chart, confusion matrices, and EDA plots are available in the dashboard. |
| FR-10 | Accept one optional local report upload to prefill inputs. | Accepts a 10 MB-or-smaller CSV, UTF-8 text, PDF, or image; extracts only explicitly labelled WDBC values in memory, rejects batch CSVs, and requires review before prediction. |
| FR-11 | Implement TabPFN-2.5 as primary research candidate. | Wraps TabPFN with in-context learning and probability prediction. |
| FR-12 | Implement CatBoost with Optuna hyperparameter optimization. | CatBoostClassifier with cross-validated hyperparameter search. |
| FR-13 | Implement FT-Transformer in PyTorch. | Feature Tokenizer + Transformer with numerical embeddings, attention, and early stopping. |
| FR-14 | Implement XGBoost with Optuna hyperparameter optimization. | XGBClassifier with regularized gradient boosting and probability prediction. |
| FR-15 | 12-Metric Comparative Evaluation + ECE. | Accuracy, Precision, F1-Score, Recall, ROC-AUC, Sensitivity, Specificity, Balanced Accuracy, MCC, NPV, Log Loss, Brier Score, and Expected Calibration Error. |
| FR-16 | Probability Calibration Analysis. | Reliability diagrams, Platt Scaling, Isotonic Regression, ECE calculation with 10 equal-width bins. |
| FR-17 | Explainable AI (XAI) with SHAP. | Summary, beeswarm, waterfall, and bar plots using model-appropriate explainers. |
| FR-18 | Multi-perspective Feature Importance. | Native model importance, permutation importance, and global SHAP importance comparing top 10 tumour features. |
| FR-19 | Research Evaluation Engine & Serialization. | Orchestrates training, 5-fold CV, metrics computation, and artifact serialization. |
| FR-20 | Streamlit Research Workbench Application. | Comprehensive multi-view dashboard: System Overview, Dataset Explorer, 12-Metric Comparison, ROC/PR Curves, 30-Feature Multi-Model Predictor, SHAP Studio, Feature Importance Rankings, Calibration Analysis, and IEEE-format Research Report. |
| FR-21 | React Research Frontend Application. | Modern Vite + React frontend in `frontend/` featuring 9 dedicated research modules, interactive 30-feature prediction workbench with instant probability calculation and agreement consensus. |
| FR-22 | Academic Literature Proof (SCI & IEEE Verification). | Dedicated module with verifiable peer-reviewed citations, DOIs, impact factors, and exact dataset usage proofs confirming WDBC usage across SCI Q1 journals and IEEE publications. |
| FR-23 | Architectural Proof (Unique Model vs. Hybrid Model Analysis). | Comprehensive formal and mathematical proof categorizing TabPFN-2.5, FT-Transformer, CatBoost, and XGBoost as Unique Single-Paradigm Models, and Stacking/Voting Ensembles as Hybrid Multi-Paradigm Ensembles. |

## Explicit exclusions

- Other cancer modalities, including cervical cancer.
- Patient history tracking, database persistence, and automatic clinical diagnosis from prose.
- Fabricating evaluation results or clinical claims.

## Acceptance criteria

1. All four models (TabPFN, CatBoost, FT-Transformer, XGBoost) train successfully on WDBC data without leakage.
2. 5-fold cross-validation reports mean and standard deviation for all evaluation metrics.
3. Untouched held-out test set benchmark evaluates all four models on all 12 metrics + ECE.
4. SHAP analysis generates beeswarm, bar, and individual waterfall plots.
5. Reliability diagrams and calibration curves compare probability calibration before and after Platt/Isotonic scaling.
6. Streamlit and React Research Frontends provide interactive user interfaces for multi-model prediction, 12-metric comparison, calibration, and SHAP explainability.
7. Dedicated SCI & IEEE Literature Proof page provides verifiable DOIs, citations, and excerpts confirming WDBC usage in top peer-reviewed literature.
8. Architectural Analysis page provides formal proofs and structural taxonomy distinguishing Unique Single-Paradigm Models from Hybrid Multi-Paradigm Ensembles.
9. Automated tests pass covering data integrity, model contracts, metric calculations, and frontend production builds.
10. Publication-ready research reports and visualizations generated in IEEE manuscript format.

## Safety statement

The predicted class and probability are educational ML outputs only. They must not replace clinical evaluation, imaging, biopsy, pathology, or professional medical judgement. Uploading a hospital report does not make it WDBC-compatible or clinically validated; users must verify every extracted value before prediction.
