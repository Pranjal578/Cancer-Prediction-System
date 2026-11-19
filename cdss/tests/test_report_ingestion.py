"""Tests for safe, in-memory extraction of explicitly labelled WDBC values."""

from __future__ import annotations

import pandas as pd
import pytest
import fitz

from src.pipeline.data_loader import WDBC_FEATURE_COLUMNS
from src.pipeline import report_ingestion
from src.pipeline.report_ingestion import ReportExtractionError, extract_report_values


def test_single_row_wdbc_csv_extracts_every_feature():
    """A one-row CSV can prefill all 30 canonical WDBC feature columns."""
    expected = {feature: index + 0.5 for index, feature in enumerate(WDBC_FEATURE_COLUMNS)}
    payload = pd.DataFrame([expected]).to_csv(index=False).encode("utf-8")

    result = extract_report_values(payload, "sample.csv")

    assert result.source == "CSV"
    assert result.is_complete
    assert result.values == expected


def test_text_report_extracts_only_explicitly_labelled_aliases():
    """Text parsing maps labels but does not infer unlabelled clinical prose."""
    payload = b"""Clinical note: findings require clinical review.
Mean radius: 12.5
texture mean = 23.1
radius standard error: 0.2
worst fractal dimension: 0.09
"""

    result = extract_report_values(payload, "report.txt")

    assert result.values == {
        "radius_mean": 12.5,
        "texture_mean": 23.1,
        "radius_se": 0.2,
        "fractal_dimension_worst": 0.09,
    }
    assert not result.is_complete
    assert len(result.missing_features) == 26


def test_batch_csv_is_rejected():
    """Uploading multiple records cannot trigger unsupported batch triage."""
    payload = b"radius_mean\n12.1\n13.2\n"

    with pytest.raises(ReportExtractionError, match="exactly one data row"):
        extract_report_values(payload, "batch.csv")


def test_text_pdf_extracts_explicitly_labelled_values():
    """Text-based PDFs are parsed locally through the PDF extraction path."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Radius mean: 12.7\nWorst texture: 29.2")
    payload = document.tobytes()
    document.close()

    result = extract_report_values(payload, "report.pdf")

    assert result.values == {"radius_mean": 12.7, "texture_worst": 29.2}


def test_image_route_passes_ocr_text_to_the_same_label_parser(monkeypatch):
    """Image OCR output receives the same strict labelled-value treatment."""
    monkeypatch.setattr(
        report_ingestion,
        "_ocr_image",
        lambda _: "Radius mean: 14.2\nWorst perimeter: 101.7",
    )

    result = report_ingestion.extract_report_values(b"test image", "report.png")

    assert result.values == {"radius_mean": 14.2, "perimeter_worst": 101.7}
