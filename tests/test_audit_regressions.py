"""Independently reproduced audit defects and numerical/input security boundaries."""

import io
import json
import logging
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import openpyxl
import pandas as pd
import pytest

from config.settings import get_settings
from src.ai.gemini_client import GeminiClient
from src.ai.insight_generator import build_analytics_context, generate_insights
from src.data.loader import load_dataset, merge_walmart_files
from src.data.pipeline import process_dataset
from src.data.schema import AggregationOptions, ColumnMapping
from src.database.db_manager import DatabaseManager
from src.evaluation.backtesting import rolling_origin_backtest
from src.evaluation.metrics import calculate_metrics
from src.evaluation.model_comparison import evaluate_models, select_best_model
from src.features.time_features import (
    holiday_summary,
    seasonal_decomposition,
    weekday_summary,
    yearly_comparison,
)
from src.forecasting.common import validate_config
from src.forecasting.forecast_service import generate_forecast, train_model
from src.forecasting.prophet_model import seasonality_settings, validate_holidays
from src.forecasting.types import ModelConfig
from src.reports.report_generator import build_report_context, export_excel
from src.ui.dashboard import prepare_uploaded_dataset
from src.utils import logging_config
from src.utils.exceptions import DataValidationError, ForecastError, SalesVisionError
from src.utils.serialization import dataframe_from_payload, dataframe_payload, strict_json


@pytest.mark.parametrize("value", ["invalid", "", "0", "-1", "100.5", "10001"])
def test_upload_configuration_is_validated_with_a_safe_error(monkeypatch, value):
    monkeypatch.setenv("SALESVISION_MAX_UPLOAD_MB", value)
    with pytest.raises(SalesVisionError, match="SALESVISION_MAX_UPLOAD_MB"):
        get_settings()


@pytest.mark.parametrize("frequency", ["W-INVALID", "", None, 7])
def test_bad_calendar_frequency_is_a_typed_validation_error(daily_frame, frequency):
    with pytest.raises(DataValidationError):
        process_dataset(daily_frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(frequency=frequency))


@pytest.mark.parametrize("window", [("not-a-date", "2023-02-01"), ("2023-02-01", "not-a-date"), ("2023-02-01",), ("NaT", "2023-02-01")])
def test_invalid_date_windows_do_not_escape_as_pandas_errors(daily_frame, window):
    with pytest.raises(DataValidationError):
        process_dataset(daily_frame, ColumnMapping("Date", "Sales"), date_range=window)


def test_timezone_aware_date_window_uses_the_same_utc_calendar(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), date_range=("2023-01-02T02:00:00+02:00", "2023-01-03T02:00:00+02:00"))
    assert dataset.series.index.tolist() == [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")]


def test_custom_named_sales_cannot_be_selected_as_a_regressor(daily_frame):
    frame = daily_frame.rename(columns={"Sales": "Amount"})
    with pytest.raises(DataValidationError, match="target"):
        process_dataset(frame, ColumnMapping("Date", "Amount"), aggregation=AggregationOptions(regressors=("Amount",)))


def test_incomplete_regressor_period_is_not_a_partial_mean():
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01", "2024-01-02"], "Sales": [10, 20, 30], "Promotion": [1, np.nan, 0]})
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(regressors=("Promotion",)))
    assert pd.isna(dataset.regressors["Promotion"].iloc[0])
    assert dataset.regressors["Promotion"].iloc[1] == 0


def test_entire_unknown_month_remains_unknown_in_ai_evidence():
    frame = pd.DataFrame({"Date": ["2023-01-01", "2023-03-01"], "Sales": [10, 20]})
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(frequency="MS", source_frequency="MS", missing_periods="ffill"))
    context = build_analytics_context(dataset)
    february = context["history"]["recent_monthly_totals"][1]
    assert february["sales"] is None
    assert february["observed_periods"] == 0
    assert generate_insights(context, client=GeminiClient(api_key="")).source == "Local analytics"


def test_all_unknown_year_and_weekday_totals_are_not_zero():
    dates = pd.date_range("2023-12-30", periods=4, freq="D")
    series = pd.Series([10, 20, np.nan, np.nan], index=dates)
    years = yearly_comparison(series).set_index("year")
    assert pd.isna(years.loc["2024", "total"])
    weekdays = weekday_summary(series, "D").set_index("weekday")
    assert pd.isna(weekdays.loc["Monday", "total"])


def test_local_mean_target_is_not_described_as_transaction_revenue(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(method="mean"))
    result = generate_insights(build_analytics_context(dataset), client=GeminiClient(api_key=""))
    assert "period averages" in result.text.lower()
    assert "observed historical sales total" not in result.text.lower()


@pytest.mark.parametrize("changes", [
    {"order": None}, {"order": (True, 1, 0)}, {"seasonal_order": None},
    {"seasonal_order": (0, 1, 0, True)}, {"interval_width": "95%"},
    {"maxiter": 20.5}, {"maxiter": "150"}, {"seed": -1}, {"seed": 2**32},
    {"yearly_seasonality": "auto"}, {"weekly_seasonality": 1},
    {"changepoint_prior_scale": np.nan}, {"regressor_columns": (1,)},
    {"regressor_columns": "Promotion"}, {"seasonality_mode": "unknown"},
])
def test_invalid_model_parameters_are_rejected_before_library_fitting(changes):
    with pytest.raises(ForecastError):
        validate_config(replace(ModelConfig("Prophet"), **changes))


@pytest.mark.parametrize("frequency,count", [("D", 730), ("W-FRI", 104), ("MS", 24)])
def test_prophet_yearly_seasonality_accepts_two_complete_cycles(frequency, count):
    history = pd.Series(np.arange(count) + 100, index=pd.date_range("2020-01-01", periods=count, freq=frequency))
    assert seasonality_settings(history, ModelConfig("Prophet"))[0]
    assert not seasonality_settings(history.iloc[:-1], ModelConfig("Prophet"))[0]


@pytest.mark.parametrize("horizon,folds", [(2.5, 2), (2, True), (True, 2), (367, 1), (2, 2.5)])
def test_backtesting_validates_integer_boundaries_before_partitioning(monthly_series, horizon, folds):
    with pytest.raises(DataValidationError):
        rolling_origin_backtest(monthly_series, [ModelConfig("ARIMA")], horizon=horizon, folds=folds)


def test_zero_stl_period_is_not_silently_replaced(monthly_series):
    with pytest.raises(DataValidationError):
        seasonal_decomposition(monthly_series, 0)


def test_large_finite_errors_have_a_finite_correct_rmse():
    result = calculate_metrics([1e200, 1e200], [0, 0])
    assert result["RMSE"] == pytest.approx(1e200)
    assert result["sMAPE"] == pytest.approx(200)


def test_non_numeric_metric_vectors_raise_a_safe_validation_error():
    with pytest.raises(DataValidationError):
        calculate_metrics(["invalid"], [1])


def test_invalid_nonfinite_and_negative_error_scores_cannot_win():
    table = pd.DataFrame({"Model": ["ARIMA", "SARIMA"], "RMSE": [np.inf, -1], "Status": ["success", "success"]})
    assert select_best_model(table) is None


def test_snapshot_preserves_small_sales_and_submillisecond_dates():
    frame = pd.DataFrame({"date": [pd.Timestamp("2024-01-01T00:00:00.123456789")], "prediction": [1e-20], "other": [np.nan]})
    restored = dataframe_from_payload(dataframe_payload(frame))
    assert restored["prediction"].iloc[0] == frame["prediction"].iloc[0]
    assert restored["date"].iloc[0] == frame["date"].iloc[0]
    assert pd.isna(restored["other"].iloc[0])
    assert strict_json({"date": pd.NaT}) == '{"date": null}'


def test_spreadsheet_headers_and_linebreak_formulas_are_escaped():
    frame = pd.DataFrame({"=1+1": ["\n=2+2", "   =3+3"], "Sales": [-10, 20]})
    sheet = openpyxl.load_workbook(io.BytesIO(export_excel(frame))).active
    assert sheet["A1"].data_type != "f"
    assert sheet["A1"].value == "'=1+1"
    assert sheet["A2"].value.startswith("'") and sheet["A3"].value.startswith("'")
    assert sheet["B2"].value == -10


def test_corrupt_database_produces_a_safe_application_error(tmp_path):
    path = tmp_path / "corrupt.sqlite3"
    content = b"Deliberately corrupt database for isolated audit regression"
    path.write_bytes(content)
    with pytest.raises(SalesVisionError, match="database"):
        DatabaseManager(path)
    assert path.read_bytes() == content


def test_reports_cannot_be_saved_against_a_different_execution(tmp_path, daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"))
    first = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 3)
    second = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 4)
    manager = DatabaseManager(tmp_path / "history.sqlite3")
    with pytest.raises(DataValidationError, match="match"):
        manager.save_forecast(first, build_report_context(dataset, forecast=second))
    assert manager.list_forecasts().empty


def test_report_records_the_actual_full_history_fit(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"))
    config = ModelConfig("ARIMA", order=(0, 1, 0))
    evaluation = evaluate_models(dataset, [config], 14)
    run = generate_forecast(dataset, config, 7, evaluation.metrics_for("ARIMA"))
    report = build_report_context(dataset, evaluation, run)
    assert run.model_summary in report.model_summaries.values()
    assert evaluation.summaries["ARIMA"] in report.model_summaries.values()


def test_closed_gemini_client_cannot_report_a_successful_request():
    provider = SimpleNamespace(models=SimpleNamespace(), close=lambda: None)
    client = GeminiClient(api_key="test-only", client_factory=lambda **kwargs: provider)
    client.close()
    assert not client.check_connectivity().ok
    assert not client.generate("context", "Explain").ok


@pytest.mark.parametrize("configuration", [None, {}, "ARIMA", SimpleNamespace(name="ARIMA")])
def test_invalid_configuration_objects_fail_safely(monthly_series, configuration):
    with pytest.raises(ForecastError):
        train_model(monthly_series, configuration)


def test_holiday_summaries_preserve_incomplete_and_unknown_dates():
    records = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]), "sales": [10, np.nan, np.nan, 50], "Holiday": [True, True, True, False]})
    table = holiday_summary(records, "Holiday").set_index("holiday")
    assert pd.isna(table.loc[True, "total_sales"])
    assert pd.isna(table.loc[True, "mean_period_sales"])
    assert table.loc[True, "observed_dates"] == 0
    assert table.loc[False, "total_sales"] == 50


def test_holiday_calendar_normalizes_timezone_aware_dates():
    result = validate_holidays(pd.DataFrame({"ds": ["2024-01-01T02:00:00+02:00"], "holiday": ["Known event"]}))
    assert result["ds"].iloc[0] == pd.Timestamp("2024-01-01")
    assert result["ds"].dt.tz is None


def test_prophet_rejects_holidays_outside_the_training_calendar(monthly_series):
    calendar = pd.DataFrame({"ds": [pd.Timestamp("2010-01-01")], "holiday": ["Historical event"]})
    with pytest.raises(ForecastError, match="training"):
        train_model(monthly_series, ModelConfig("Prophet", use_holidays=True), holidays=calendar)


def test_duplicate_csv_headers_after_blank_lines_are_rejected():
    with pytest.raises(DataValidationError, match="Duplicate"):
        load_dataset(b"\n   \nDate,Date\n2024-01-01,10\n", "headers.csv")


def test_walmart_join_types_have_a_safe_validation_error():
    train = pd.DataFrame({"Store": [1], "Dept": [1], "Date": ["2024-01-05"], "Weekly_Sales": [10]})
    stores = pd.DataFrame({"Store": ["invalid"], "Size": [100]})
    with pytest.raises(DataValidationError, match="Store"):
        merge_walmart_files(train, stores)


def test_walmart_holiday_agreement_uses_boolean_meaning():
    train = pd.DataFrame({"Store": [1], "Dept": [1], "Date": ["2024-01-05"], "Weekly_Sales": [10], "IsHoliday": [True]})
    features = pd.DataFrame({"Store": [1], "Date": ["2024-01-05"], "IsHoliday": [" yes "]})
    result, operations = merge_walmart_files(train, features=features)
    assert len(result) == 1 and len(operations) == 1
    features["IsHoliday"] = False
    with pytest.raises(DataValidationError, match="disagree"):
        merge_walmart_files(train, features=features)


def test_holdout_horizon_respects_supported_forecast_limit():
    frame = pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=500), "Sales": np.arange(500) + 100})
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"))
    with pytest.raises(DataValidationError, match="holdout"):
        evaluate_models(dataset, [ModelConfig("ARIMA", order=(0, 1, 0))], 400)


def test_finite_sales_that_overflow_aggregation_fail_safely():
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], "Sales": [1e308] * 4})
    with pytest.raises(DataValidationError, match="numeric range"):
        process_dataset(frame, ColumnMapping("Date", "Sales"))


def test_joined_upload_preserves_actual_file_hashes_and_small_driver_changes():
    history = load_dataset(b"Store,Dept,Date,Weekly_Sales\n1,1,2024-01-05,10\n1,1,2024-01-12,20\n", "train.csv")
    first_features = load_dataset(b"Store,Date,Promotion\n1,2024-01-05,1e-20\n1,2024-01-12,0\n", "features.csv")
    second_features = load_dataset(b"Store,Date,Promotion\n1,2024-01-05,2e-20\n1,2024-01-12,0\n", "features.csv")
    first, metadata, operations = prepare_uploaded_dataset(history, features=first_features)
    second, changed_metadata, _ = prepare_uploaded_dataset(history, features=second_features)
    assert metadata["dataset_file_sha256"] == history.sha256
    assert metadata["source_sha256"] == history.sha256
    assert metadata["ancillary_sources"]["features"]["sha256"] == first_features.sha256
    assert first.sha256 != second.sha256
    assert metadata["joined_dataset_sha256"] != changed_metadata["joined_dataset_sha256"]
    assert len(first.frame) == len(history.frame) and len(operations) == 1


def test_future_holidays_must_match_weekly_forecast_anchors():
    frame = pd.DataFrame({"Date": pd.date_range("2023-01-06", periods=40, freq="W-FRI"), "Sales": np.arange(40) + 100, "Holiday": [True] + [False] * 39})
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales", holiday="Holiday"), aggregation=AggregationOptions(frequency="W-FRI"))
    invalid = pd.DataFrame({"ds": [dataset.series.index[-1] + pd.Timedelta(days=3)], "holiday": ["Known event"]})
    with pytest.raises(ForecastError, match="calendar"):
        generate_forecast(dataset, ModelConfig("Prophet", yearly_seasonality=False, use_holidays=True), 4, future_holidays=invalid)


def test_snapshot_restores_mixed_datetime_precision():
    frame = pd.DataFrame({"date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02T00:00:00.123456789"), pd.NaT], "sales": [10.0, 20.0, np.nan]})
    restored = dataframe_from_payload(dataframe_payload(frame))
    pd.testing.assert_frame_equal(restored, frame)


@pytest.mark.parametrize("column,field,value", [("run_json", "metadata", []), ("run_json", "configuration", "invalid"), ("report_json", "metadata", []), ("report_json", "holdout_predictions", [])])
def test_valid_json_with_corrupt_snapshot_shapes_fails_safely(tmp_path, daily_frame, column, field, value):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"))
    run = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 3)
    manager = DatabaseManager(tmp_path / "snapshot.sqlite3")
    manager.save_forecast(run, build_report_context(dataset, forecast=run))
    with sqlite3.connect(manager.path) as connection:
        payload = json.loads(connection.execute(f"SELECT {column} FROM forecasts WHERE run_id=?", (run.run_id,)).fetchone()[0])
        payload[field] = value
        connection.execute(f"UPDATE forecasts SET {column}=? WHERE run_id=?", (json.dumps(payload), run.run_id))
    read = manager.get_forecast if column == "run_json" else manager.get_report
    with pytest.raises(SalesVisionError, match="corrupted"):
        read(run.run_id)


def test_unavailable_file_logging_preserves_server_diagnostics(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("Existing file, not a log directory", encoding="utf-8")
    isolated_logger = logging.Logger("audit-isolated")
    original = logging.getLogger
    with monkeypatch.context() as scoped:
        scoped.setattr(logging_config.logging, "getLogger", lambda name=None: isolated_logger if name == "salesvision" else original(name))
        configured = logging_config.configure_logging(blocked)
    assert configured is isolated_logger
    assert any(isinstance(handler, logging.StreamHandler) for handler in configured.handlers)
