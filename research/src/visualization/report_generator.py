"""Research report generator.

Assembles all evaluation results, SHAP analyses, calibration data, and
visualization paths into a structured Markdown and HTML research report.
The report is saved to REPORTS_DIR and includes:
- Executive summary with best model recommendation
- Comparative metrics table
- CV results table
- Calibration comparison
- SHAP feature importance rankings
- Ensemble experiment result
- Methodology notes and reproducibility information
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

import pandas as pd


def generate_markdown_report(
    comparison_df: pd.DataFrame,
    cv_results: dict,
    calibration_df: pd.DataFrame,
    shap_importance: dict[str, pd.Series],
    ensemble_result: dict | None,
    report_path: Path,
    metadata: dict | None = None,
) -> str:
    """Generate a Markdown research report and save it to report_path.

    Parameters
    ----------
    comparison_df:
        Output of evaluate_all_models (test-set metrics table).
    cv_results:
        Output of run_cv_all_models (per-fold CV metrics).
    calibration_df:
        Output of compare_calibration.
    shap_importance:
        dict mapping model_name -> global SHAP importance pd.Series.
    ensemble_result:
        dict from ensemble experiment (may be None if not run).
    report_path:
        Path where the .md file is written.
    metadata:
        Optional dataset metadata dict (from load_wdbc).

    Returns
    -------
    str — the full Markdown content.

    Raises
    ------
    NotImplementedError
        Until this stub is implemented.
    """
    raise NotImplementedError("TODO")


def generate_html_report(
    markdown_content: str,
    html_path: Path,
    title: str = "Breast Cancer ML Research Report",
) -> None:
    """Convert Markdown to HTML and save to html_path.

    Uses the ``markdown`` package or a simple template if unavailable.

    Parameters
    ----------
    markdown_content:
        Markdown string from generate_markdown_report.
    html_path:
        Destination .html file path.
    title:
        HTML <title> element content.

    Raises
    ------
    NotImplementedError
        Until this stub is implemented.
    """
    raise NotImplementedError("TODO")
