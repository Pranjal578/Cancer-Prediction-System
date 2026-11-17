# Design Specification

## Design goals

- Make the workflow understandable to students and evaluators.
- Keep tumour input collection manageable despite 30 required values.
- Make the selected prediction and model-comparison evidence easy to distinguish.
- Preserve a clear educational-use disclaimer.

## Information architecture

The Streamlit dashboard has exactly three tabs:

1. **Prediction**
   - A form groups all 30 WDBC fields as Mean, Standard-error, and Worst measurements.
   - Dataset medians populate initial values.
   - Submitting runs the saved model pipeline.
   - The result uses a clear Benign or Malignant status and displays malignant/benign probabilities with a bar chart.

2. **Model comparison**
   - Shows the five-model comparison table: Accuracy, Precision, Recall, and F1-score.
   - Shows the performance comparison chart and all five confusion matrices.
   - Explains the deterministic selection rule.

3. **Exploratory data analysis**
   - Shows class distribution, feature distributions, and correlation heatmap generated during training.

## Interaction rules

- Each input is non-negative and accepts a decimal value.
- Prediction is only available after valid trained artefacts exist.
- Missing artefacts produce a clear command to train the system; the UI does not fall back to an untrained model.
- The UI does not collect patient identity, persist user input, or provide clinical recommendations.

## Visual behavior

- Benign results use a success state; Malignant results use an alert state.
- Metric tables use percentage formatting for easy comparison.
- Images stretch to their containing tab area for readability.
- The layout is responsive through Streamlit columns, tabs, forms, and expanders.

## Accessibility and safety

- Labels are derived from the actual WDBC feature names and remain visible with inputs.
- Visual charts supplement rather than replace textual metrics.
- The educational/non-diagnostic warning appears above all interactive controls.
