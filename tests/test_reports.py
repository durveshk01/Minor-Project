"""Export fidelity, PDF rendering and aggregate report snapshot tests."""

import io

import openpyxl
import pandas as pd
import pytest
from pypdf import PdfReader

from src.data.pipeline import process_dataset
from src.data.schema import AggregationOptions, ColumnMapping
from src.evaluation.model_comparison import evaluate_models
from src.features.time_features import seasonal_decomposition
from src.forecasting.forecast_service import generate_forecast
from src.forecasting.types import ModelConfig
from src.reports.report_generator import (
    ReportContext,
    build_report_context,
    export_csv,
    export_excel,
    generate_excel_report,
    generate_pdf_report,
)


@pytest.fixture
def report_context(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), metadata={"name": "Synthetic QA fixture", "kind": "synthetic", "currency": "sales units"})
    config = ModelConfig("ARIMA", order=(0, 1, 0))
    evaluation = evaluate_models(dataset, [config], 14)
    run = generate_forecast(dataset, config, 7, evaluation.metrics_for("ARIMA"))
    return build_report_context(dataset, evaluation, run, seasonal_decomposition(dataset.series, 7))


def test_csv_export_retains_actual_values_and_negative_numbers():
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "prediction": [12.5, -4.2]})
    exported = pd.read_csv(io.BytesIO(export_csv(frame)))
    pd.testing.assert_frame_equal(exported, frame)


def test_excel_export_does_not_execute_uploaded_formulas():
    frame = pd.DataFrame({"Label": ["=1+1", "@unsafe"], "Sales": [-10, 20]})
    workbook = openpyxl.load_workbook(io.BytesIO(export_excel(frame)))
    sheet = workbook.active
    assert sheet["A2"].data_type != "f"
    assert sheet["A2"].value == "'=1+1"
    assert sheet["B2"].value == -10


def test_excel_report_contains_real_forecasts_and_evaluation(report_context):
    content = generate_excel_report(report_context)
    workbook = pd.ExcelFile(io.BytesIO(content))
    assert {"Forecast", "Model Comparison", "Historical", "Cleaning Audit", "Seasonal Components"}.issubset(workbook.sheet_names)
    forecasts = pd.read_excel(workbook, sheet_name="Forecast")
    pd.testing.assert_series_equal(forecasts["prediction"], report_context.forecast["prediction"], check_names=False)


def test_illustrated_pdf_is_parseable_and_uses_actual_outputs(report_context):
    content = generate_pdf_report(report_context)
    reader = PdfReader(io.BytesIO(content))
    assert len(reader.pages) >= 4
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "SalesVision" in text and "SYNTHETIC TEST DATA" in text
    assert "ARIMA" in text and "SARIMA" in text and "Prophet" in text
    assert f"{report_context.forecast['prediction'].iloc[0]:,.3f}" in text
    assert any("/XObject" in page["/Resources"] for page in reader.pages)


def test_report_snapshot_round_trip_preserves_unknown_periods(report_context):
    restored = ReportContext.from_payload(report_context.to_payload())
    pd.testing.assert_series_equal(restored.history, report_context.history, check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(restored.forecast, report_context.forecast, check_exact=False, rtol=1e-9)


def test_pdf_wraps_fitted_summary_instead_of_discarding_long_line_evidence(report_context):
    marker = "FULL_SUMMARY_EVIDENCE"
    report_context.model_summaries = {"ARIMA": "Recorded evidence " * 20 + marker}
    reader = PdfReader(io.BytesIO(generate_pdf_report(report_context)))
    assert marker in "\n".join(page.extract_text() for page in reader.pages)


def test_mean_target_reports_explain_period_average_sums_with_unambiguous_export_labels(daily_frame):
    frame = pd.concat([daily_frame, daily_frame.assign(Sales=daily_frame["Sales"] * 2)], ignore_index=True)
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(method="mean"), metadata={"name": "Mean-target audit", "kind": "synthetic"})
    report = build_report_context(dataset)
    pdf = PdfReader(io.BytesIO(generate_pdf_report(report)))
    text = " ".join("\n".join(page.extract_text() for page in pdf.pages).split())
    assert "not total transaction revenue" in text
    assert "sum_of_period_averages" in text
    workbook = pd.ExcelFile(io.BytesIO(generate_excel_report(report)))
    statistics = pd.read_excel(workbook, sheet_name="Statistics").set_index("Item")["Value"]
    assert "total" not in statistics.index
    assert float(statistics["sum_of_period_averages"]) == pytest.approx(daily_frame["Sales"].sum() * 1.5)
    monthly = pd.read_excel(workbook, sheet_name="Monthly Comparison")
    assert "sum_of_period_averages" in monthly and "sales" not in monthly
    metadata = pd.read_excel(workbook, sheet_name="Dataset").set_index("Item")["Value"]
    assert "period averages" in metadata["target_definition"]
