"""Extract explicitly labelled WDBC values from one uploaded local report.

The module deliberately does not interpret clinical language. It only maps
values whose labels match the 30 WDBC feature names (or their plain-language
aliases), and leaves a person to review every value before prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import re

import pandas as pd

from src.pipeline.data_loader import WDBC_FEATURE_COLUMNS


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 10
SUPPORTED_EXTENSIONS = frozenset(
    {".csv", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_LABELLED_VALUE = re.compile(
    rf"^\s*(?P<label>[A-Za-z_()\- ]+?)\s*(?::|=)\s*(?P<value>{_NUMBER.pattern})"
)


class ReportExtractionError(ValueError):
    """Raised when an uploaded report cannot safely provide WDBC values."""


@dataclass(frozen=True)
class ExtractionResult:
    """Values found in one upload and the fields still requiring review."""

    source: str
    values: dict[str, float]
    missing_features: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_features


def _normalise_identifier(value: object) -> str:
    """Normalise a report label without guessing a clinical measurement."""
    normalised = str(value).strip().lower()
    normalised = normalised.replace("µ", "u")
    normalised = re.sub(r"\bstandard[\s_-]*(?:error|err)\b", "se", normalised)
    normalised = re.sub(r"\bstd(?:\.|\s)*(?:error|err)\b", "se", normalised)
    normalised = re.sub(r"\b(millimet(?:er|re)s?|centimet(?:er|re)s?|microns?|um)\b", " ", normalised)
    normalised = re.sub(r"[^a-z0-9]+", " ", normalised)
    return " ".join(normalised.split())


def _feature_aliases(feature_name: str) -> set[str]:
    """Return unambiguous labels that can represent one WDBC feature."""
    base_name, statistic = feature_name.rsplit("_", maxsplit=1)
    base_label = base_name.replace("_", " ")
    statistic_aliases = {
        "mean": ("mean",),
        "se": ("se", "standard error", "standard err", "stderr"),
        "worst": ("worst",),
    }[statistic]
    aliases = {feature_name, f"{base_label} {statistic}"}
    for statistic_alias in statistic_aliases:
        aliases.add(f"{base_label} {statistic_alias}")
        aliases.add(f"{statistic_alias} {base_label}")
    return {_normalise_identifier(alias) for alias in aliases}


def _alias_lookup(feature_names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for feature_name in feature_names:
        for alias in _feature_aliases(feature_name):
            existing = lookup.get(alias)
            if existing and existing != feature_name:
                raise RuntimeError(f"Ambiguous WDBC report alias: {alias}")
            lookup[alias] = feature_name
    return lookup


def _coerce_value(value: object, feature_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportExtractionError(
            f"{feature_name.replace('_', ' ')} does not contain a numeric value."
        ) from error
    if not math.isfinite(result) or result < 0:
        raise ReportExtractionError(
            f"{feature_name.replace('_', ' ')} must be a finite, non-negative value."
        )
    return result


def _record_value(values: dict[str, float], feature_name: str, value: object) -> None:
    numeric_value = _coerce_value(value, feature_name)
    existing = values.get(feature_name)
    if existing is not None and not math.isclose(existing, numeric_value, rel_tol=1e-9):
        raise ReportExtractionError(
            f"The report has conflicting values for {feature_name.replace('_', ' ')}."
        )
    values[feature_name] = numeric_value


def _result(source: str, values: dict[str, float], feature_names: list[str]) -> ExtractionResult:
    if not values:
        raise ReportExtractionError(
            "No explicitly labelled WDBC measurements were found. Upload a single-row "
            "CSV or a report containing labels such as 'radius mean: 12.3'."
        )
    missing = tuple(feature for feature in feature_names if feature not in values)
    return ExtractionResult(source=source, values=values, missing_features=missing)


def extract_values_from_csv(payload: bytes, feature_names: list[str]) -> ExtractionResult:
    """Extract one WDBC-compatible CSV row; batch uploads remain unsupported."""
    try:
        table = pd.read_csv(BytesIO(payload))
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ReportExtractionError("The CSV file could not be read.") from error
    if len(table) != 1:
        raise ReportExtractionError(
            "Upload a CSV with exactly one data row. Batch prediction is not supported."
        )

    aliases = _alias_lookup(feature_names)
    values: dict[str, float] = {}
    matched_columns: set[str] = set()
    for column in table.columns:
        feature_name = aliases.get(_normalise_identifier(column))
        if not feature_name:
            continue
        if feature_name in matched_columns:
            raise ReportExtractionError(
                f"The CSV includes more than one column for {feature_name.replace('_', ' ')}."
            )
        matched_columns.add(feature_name)
        cell = table.iloc[0][column]
        if pd.isna(cell) or str(cell).strip().lower() in {"", "?", "na", "n/a"}:
            continue
        _record_value(values, feature_name, cell)
    return _result("CSV", values, feature_names)


def extract_values_from_text(
    text: str, feature_names: list[str], source: str = "text"
) -> ExtractionResult:
    """Extract ``label: value`` pairs without inferring values from prose."""
    aliases = _alias_lookup(feature_names)
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _LABELLED_VALUE.match(line)
        if not match:
            continue
        feature_name = aliases.get(_normalise_identifier(match.group("label")))
        if feature_name:
            _record_value(values, feature_name, match.group("value"))
    return _result(source, values, feature_names)


def _ocr_image(payload: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as error:
        raise ReportExtractionError(
            "Image extraction requires the Pillow and pytesseract dependencies."
        ) from error
    try:
        with Image.open(BytesIO(payload)) as image:
            return str(pytesseract.image_to_string(image))
    except Exception as error:  # Tesseract also raises a package-specific error when absent.
        raise ReportExtractionError(
            "Image text could not be read. Install the local Tesseract OCR engine or use "
            "a text, PDF, or CSV report."
        ) from error


def _extract_pdf_text(payload: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ReportExtractionError("PDF extraction requires the pypdf dependency.") from error
    try:
        reader = PdfReader(BytesIO(payload))
        if reader.is_encrypted:
            raise ReportExtractionError("Password-protected PDFs cannot be processed.")
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]
        ).strip()
    except ReportExtractionError:
        raise
    except Exception as error:
        raise ReportExtractionError("The PDF could not be read.") from error
    if text:
        return text
    return _ocr_pdf_pages(payload)


def _ocr_pdf_pages(payload: bytes) -> str:
    try:
        import fitz
    except ImportError as error:
        raise ReportExtractionError(
            "Scanned-PDF extraction requires the PyMuPDF dependency."
        ) from error
    try:
        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as error:
        raise ReportExtractionError("The scanned PDF could not be opened for OCR.") from error
    try:
        page_images = [
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
            for page in document[:MAX_PDF_PAGES]
        ]
    except Exception as error:
        raise ReportExtractionError("The scanned PDF could not be rendered for OCR.") from error
    finally:
        document.close()
    return "\n".join(_ocr_image(image) for image in page_images)


def extract_report_values(
    payload: bytes,
    filename: str,
    feature_names: list[str] | None = None,
) -> ExtractionResult:
    """Extract reviewable values from one CSV, text, PDF, or image upload.

    Upload bytes are accepted in memory and are never written to disk.
    """
    if not payload:
        raise ReportExtractionError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ReportExtractionError("Upload a file no larger than 10 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ReportExtractionError(f"Unsupported file type. Use one of: {supported}.")

    expected_features = feature_names or list(WDBC_FEATURE_COLUMNS)
    if suffix == ".csv":
        return extract_values_from_csv(payload, expected_features)
    if suffix == ".txt":
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ReportExtractionError("Text reports must use UTF-8 encoding.") from error
        return extract_values_from_text(text, expected_features, source="text")
    if suffix == ".pdf":
        return extract_values_from_text(_extract_pdf_text(payload), expected_features, source="PDF")
    return extract_values_from_text(_ocr_image(payload), expected_features, source="image OCR")
