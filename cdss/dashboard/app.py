"""Streamlit interface for the proposed breast-cancer prediction system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, MODEL_NAMES, MODEL_PIPELINES_PATH, REPORTS_DIR
from dashboard.design_system import (
    inject_design_system,
    render_masthead,
    render_panel,
    render_result_band,
    render_workflow_heading,
)
from src.pipeline.data_loader import load_breast_dataset
from src.pipeline.report_ingestion import ReportExtractionError, extract_report_values

MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="Signal Lab | Breast ML",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_design_system()


@st.cache_resource
def load_trained_models() -> dict:
    """Load the five deployment pipelines produced by the training script."""
    trained_models = joblib.load(MODEL_PIPELINES_PATH)
    if set(trained_models) != set(MODEL_NAMES):
        raise ValueError("The saved model bundle does not contain the five required classifiers.")
    return trained_models


@st.cache_data
def load_reference_data() -> tuple[list[str], dict[str, float], dict]:
    """Load feature names, median defaults, and dataset metadata."""
    X, _, feature_names, metadata = load_breast_dataset()
    return feature_names, {name: float(X[name].median()) for name in feature_names}, metadata


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def display_feature_inputs(feature_names: list[str], defaults: dict[str, float]) -> dict[str, float]:
    """Render the 30 WDBC tumour-characteristic inputs in readable groups."""
    groups = {
        "Mean measurements": [name for name in feature_names if name.endswith("_mean")],
        "Standard-error measurements": [name for name in feature_names if name.endswith("_se")],
        "Worst measurements": [name for name in feature_names if name.endswith("_worst")],
    }
    values: dict[str, float] = {}
    for group_name, group_features in groups.items():
        with st.expander(group_name, expanded=group_name == "Mean measurements"):
            columns = st.columns(3, gap="medium")
            for index, feature in enumerate(group_features):
                label = feature.replace("_", " ").title()
                key = f"feature_{feature}"
                if key not in st.session_state:
                    st.session_state[key] = defaults[feature]
                with columns[index % 2]:
                    values[feature] = st.number_input(
                        label,
                        min_value=0.0,
                        format="%.6f",
                        key=key,
                    )
    return values


if (
    not MODEL_PATH.exists()
    or not MODEL_PIPELINES_PATH.exists()
    or not METADATA_PATH.exists()
    or not COMPARISON_PATH.exists()
):
    st.error("Training artefacts are not available yet.")
    st.code("cd cdss && .venv/bin/python train_pipeline.py", language="bash")
    st.stop()

feature_names, default_values, dataset_metadata = load_reference_data()
metadata = load_metadata()
trained_models = load_trained_models()
comparison = pd.read_csv(COMPARISON_PATH)
available_model_names = [
    model_name for model_name in comparison["model"].tolist() if model_name in trained_models
]
if set(available_model_names) != set(MODEL_NAMES):
    st.error("The trained model bundle and comparison results do not match. Retrain the system.")
    st.stop()

render_masthead(
    metadata["selected_model"], dataset_metadata["records_after_duplicate_removal"]
)
st.warning(
    "Educational ML demonstration only. A hospital report is not automatically WDBC-compatible; verify every extracted value before inference."
)

with st.sidebar:
    st.markdown("<p class='sidebar-label'>Model register</p>", unsafe_allow_html=True)
    st.header("Run context")
    st.write(f"**Dataset**  \\n+{dataset_metadata['dataset']}")
    st.write(f"**Cohort**  \\n+{dataset_metadata['records_after_duplicate_removal']} records · {dataset_metadata['feature_count']} fields")
    st.write(f"**Default**  \\n+{metadata['selected_model']}")
    st.divider()
    st.markdown("<p class='sidebar-label'>Selection protocol</p>", unsafe_allow_html=True)
    st.caption("Validation Recall → F1-score → Precision → Accuracy")
    st.divider()
    st.markdown("<p class='sidebar-label'>Use boundary</p>", unsafe_allow_html=True)
    st.caption(
        "Educational output only. It is not a diagnostic device and must not replace clinical assessment, imaging, pathology, or medical judgement."
    )

tab_predict, tab_comparison, tab_eda = st.tabs(
    ["01 / PREDICT", "02 / BENCHMARK", "03 / DATA LENS"]
)

with tab_predict:
    render_workflow_heading(
        "01",
        "Assemble one reviewed case",
        "Start with a local report or enter the 30 WDBC fine-needle aspirate measurements directly.",
    )
    source_column, model_column = st.columns([1.08, 0.92], gap="large")
    with source_column:
        render_panel(
            "Report-to-field assist",
            "Optional only. The parser fills explicit WDBC labels; it never interprets narrative clinical text or stores your file.",
            "Input source",
        )
        uploaded_report = st.file_uploader(
            "Upload one WDBC-compatible report",
            type=["csv", "txt", "pdf", "png", "jpg", "jpeg", "tif", "tiff"],
            help="Maximum 10 MB. CSV files must contain one row. Files are processed in memory and are not saved.",
            label_visibility="collapsed",
        )
        if uploaded_report and st.button("Extract labelled values", type="secondary"):
            try:
                extraction = extract_report_values(
                    uploaded_report.getvalue(), uploaded_report.name, feature_names
                )
            except ReportExtractionError as error:
                st.error(str(error))
            else:
                for feature_name, value in extraction.values.items():
                    st.session_state[f"feature_{feature_name}"] = value
                st.session_state["report_extraction_summary"] = {
                    "source": extraction.source,
                    "values_found": len(extraction.values),
                    "missing_features": extraction.missing_features,
                }
                st.session_state["report_value_reviewed"] = False
                st.rerun()

        extraction_summary = st.session_state.get("report_extraction_summary")
        if extraction_summary:
            st.success(
                f"{extraction_summary['source'].title()} extraction populated "
                f"{extraction_summary['values_found']} of {len(feature_names)} fields."
            )
            if extraction_summary["missing_features"]:
                with st.expander("Fields still requiring manual entry"):
                    st.write(", ".join(extraction_summary["missing_features"]))

    with model_column:
        render_panel(
            "Choose a fitted pipeline",
            "The recall-selected model is the reference choice. Every option is refit on the same development partition.",
            "Inference strategy",
            warm=True,
        )
        selected_index = available_model_names.index(metadata["selected_model"])
        prediction_model_name = st.selectbox(
            "Prediction algorithm",
            available_model_names,
            index=selected_index,
            help="The recall-selected model is the default. Other choices are for educational comparison.",
        )
        selected_metrics = comparison.loc[
            comparison["model"] == prediction_model_name,
            ["accuracy", "precision", "recall", "f1_score"],
        ].iloc[0]
        metric_columns = st.columns(2, gap="medium")
        for index, (metric, value) in enumerate(selected_metrics.items()):
            metric_columns[index % 2].metric(metric.replace("_", " "), f"{value:.2%}")
        with st.expander("Five-model validation comparison"):
            st.bar_chart(
                comparison.set_index("model")[["accuracy", "precision", "recall", "f1_score"]],
                height=300,
            )

    render_workflow_heading(
        "02",
        "Review measurement set",
        "A prediction stays locked until all 30 fields have been checked for WDBC compatibility.",
    )
    with st.form("prediction_form"):
        feature_values = display_feature_inputs(feature_names, default_values)
        reviewed_values = st.checkbox(
            "I have reviewed all 30 values and confirm they are WDBC-compatible measurements.",
            key="report_value_reviewed",
        )
        submitted = st.form_submit_button("Run model inference →", type="primary", width="stretch")

    if submitted:
        if not reviewed_values:
            st.error("Review and confirm all 30 WDBC values before running a prediction.")
        else:
            input_frame = pd.DataFrame([feature_values], columns=feature_names)
            prediction_model = trained_models[prediction_model_name]
            predicted_class = int(prediction_model.predict(input_frame)[0])
            predicted_probability = float(prediction_model.predict_proba(input_frame)[0, 1])
            diagnosis = "Malignant" if predicted_class == 1 else "Benign"
            render_result_band(diagnosis)
            result_columns = st.columns([0.85, 0.85, 1.3], gap="large")
            result_columns[0].metric("Malignant probability", f"{predicted_probability:.2%}")
            result_columns[1].metric("Benign probability", f"{1 - predicted_probability:.2%}")
            result_columns[2].markdown(
                f"<p class='eyebrow'>Inference pipeline</p><h3>{prediction_model_name}</h3>",
                unsafe_allow_html=True,
            )
            st.bar_chart(
                pd.DataFrame(
                    {"Probability": [1 - predicted_probability, predicted_probability]},
                    index=["Benign", "Malignant"],
                ),
                height=190,
            )

with tab_comparison:
    render_workflow_heading(
        "02",
        "Benchmark the candidate models",
        "Validation recall ranks the models; F1-score, precision, and accuracy resolve ties deterministically.",
    )
    default_metrics = comparison.loc[
        comparison["model"] == metadata["selected_model"],
        ["accuracy", "precision", "recall", "f1_score"],
    ].iloc[0]
    summary_columns = st.columns(4, gap="medium")
    for column, (metric, value) in zip(summary_columns, default_metrics.items()):
        column.metric(f"Default {metric.replace('_', ' ')}", f"{value:.2%}")
    leaderboard = comparison.copy()
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard["decision"] = leaderboard["model"].eq(metadata["selected_model"]).map(
        {True: "Default pipeline", False: "Available for comparison"}
    )
    st.markdown("<p class='eyebrow'>Validation leaderboard</p>", unsafe_allow_html=True)
    st.dataframe(
        leaderboard.style.format(
            {
                "accuracy": "{:.2%}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1_score": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    chart_column, matrix_column = st.columns([1.04, 0.96], gap="large")
    with chart_column:
        st.markdown("<p class='eyebrow'>Metric distribution</p>", unsafe_allow_html=True)
        st.image(
            str(ARTIFACTS_DIR / "model_comparison.png"),
            caption="Validation metrics across all five models",
        )
    with matrix_column:
        st.markdown("<p class='eyebrow'>Error anatomy</p>", unsafe_allow_html=True)
        st.image(
            str(ARTIFACTS_DIR / "confusion_matrices.png"),
            caption="Validation confusion matrices",
        )
    st.markdown(
        "<p class='data-note'>The selected default is refit on the development partition and evaluated once against untouched test data. The table and charts above are validation evidence, not clinical validation.</p>",
        unsafe_allow_html=True,
    )
    test_metrics = metadata["selected_model_test_metrics"]
    st.markdown("<p class='eyebrow'>Default-model held-out test benchmark</p>", unsafe_allow_html=True)
    test_columns = st.columns(4, gap="medium")
    for column, (metric, value) in zip(test_columns, test_metrics.items()):
        column.metric(metric.replace("_", " ").title(), f"{value:.2%}")

with tab_eda:
    render_workflow_heading(
        "03",
        "Inspect the reference cohort",
        "These views describe the fixed WDBC educational dataset used to train and benchmark the model set.",
    )
    st.markdown(
        "<p class='data-note'>Exploration informs model understanding only. The dataset contains 569 WDBC records and does not represent a clinical validation cohort.</p>",
        unsafe_allow_html=True,
    )
    eda_images = [
        (
            "Class balance",
            "Target distribution across the reference cohort",
            REPORTS_DIR / "class_distribution.png",
        ),
        (
            "Feature landscape",
            "Distribution of all 30 FNA-derived measurements",
            REPORTS_DIR / "feature_distributions.png",
        ),
        (
            "Correlation structure",
            "Relationships between the measured characteristics",
            REPORTS_DIR / "correlation_heatmap.png",
        ),
    ]
    for title, description, path in eda_images:
        if path.exists():
            st.markdown(
                f"<p class='eyebrow'>{title}</p><h3>{description}</h3>",
                unsafe_allow_html=True,
            )
            st.image(str(path), width="stretch")
