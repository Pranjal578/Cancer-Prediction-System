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

from config import ARTIFACTS_DIR, REPORTS_DIR
from src.pipeline.data_loader import load_breast_dataset

MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_trained_model():
    """Load the selected model produced by the training script."""
    return joblib.load(MODEL_PATH)


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
            columns = st.columns(2)
            for index, feature in enumerate(group_features):
                label = feature.replace("_", " ").title()
                with columns[index % 2]:
                    values[feature] = st.number_input(
                        label,
                        min_value=0.0,
                        value=defaults[feature],
                        format="%.6f",
                        key=f"feature_{feature}",
                    )
    return values


st.title("Breast Cancer Prediction System")
st.caption("WDBC-based benign/malignant classification using five compared machine-learning models.")
st.warning(
    "Educational decision-support only. This application is not a substitute for clinical assessment, imaging, pathology, or medical advice."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists() or not COMPARISON_PATH.exists():
    st.error("Training artefacts are not available yet.")
    st.code("cd cdss && .venv/bin/python train_pipeline.py", language="bash")
    st.stop()

feature_names, default_values, dataset_metadata = load_reference_data()
metadata = load_metadata()
model = load_trained_model()

with st.sidebar:
    st.header("System summary")
    st.write(f"**Dataset:** {dataset_metadata['dataset']}")
    st.write(f"**Records:** {dataset_metadata['records_after_duplicate_removal']}")
    st.write(f"**Features:** {dataset_metadata['feature_count']}")
    st.write(f"**Selected model:** {metadata['selected_model']}")
    st.write(f"**Selection rule:** {metadata['selection_rule']}")

tab_predict, tab_comparison, tab_eda = st.tabs(
    ["Prediction", "Model comparison", "Exploratory data analysis"]
)

with tab_predict:
    st.subheader("Enter tumour characteristics")
    st.caption("All 30 values correspond to the WDBC fine-needle aspirate measurements.")
    with st.form("prediction_form"):
        feature_values = display_feature_inputs(feature_names, default_values)
        submitted = st.form_submit_button("Predict diagnosis", type="primary", width="stretch")

    if submitted:
        input_frame = pd.DataFrame([feature_values], columns=feature_names)
        predicted_class = int(model.predict(input_frame)[0])
        predicted_probability = float(model.predict_proba(input_frame)[0, 1])
        diagnosis = "Malignant" if predicted_class == 1 else "Benign"
        probability_label = "Malignant probability"

        if predicted_class == 1:
            st.error(f"Prediction: {diagnosis}")
        else:
            st.success(f"Prediction: {diagnosis}")
        left, right = st.columns(2)
        left.metric(probability_label, f"{predicted_probability:.2%}")
        right.metric("Benign probability", f"{1 - predicted_probability:.2%}")
        st.bar_chart(
            pd.DataFrame(
                {"Probability": [1 - predicted_probability, predicted_probability]},
                index=["Benign", "Malignant"],
            )
        )

with tab_comparison:
    st.subheader("Classifier evaluation and selection")
    st.write("The selected model is the highest-ranked classifier by held-out-test F1-score, then recall, precision, and accuracy.")
    comparison = pd.read_csv(COMPARISON_PATH)
    st.dataframe(
        comparison.style.format({
            "accuracy": "{:.2%}",
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1_score": "{:.2%}",
        }),
        width="stretch",
        hide_index=True,
    )
    chart_column, matrix_column = st.columns(2)
    with chart_column:
        st.image(str(ARTIFACTS_DIR / "model_comparison.png"), caption="Metric comparison")
    with matrix_column:
        st.image(str(ARTIFACTS_DIR / "confusion_matrices.png"), caption="Confusion matrices")

with tab_eda:
    st.subheader("Exploratory data analysis")
    st.write(
        "Training generates data-quality summaries, class balance, feature distributions, and feature correlations from the WDBC dataset."
    )
    eda_images = [
        ("Class distribution", REPORTS_DIR / "class_distribution.png"),
        ("Feature distributions", REPORTS_DIR / "feature_distributions.png"),
        ("Correlation heatmap", REPORTS_DIR / "correlation_heatmap.png"),
    ]
    for caption, path in eda_images:
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
