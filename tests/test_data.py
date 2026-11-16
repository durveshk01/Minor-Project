"""Meaningful ingestion, policy, calendar and join regression tests."""

import io

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_dataset
from src.data.loader import load_dataset, merge_walmart_files
from src.data.pipeline import process_dataset
from src.data.schema import AggregationOptions, CleaningOptions, ColumnMapping
from src.data.validator import detect_columns, validate_dataset
from src.utils.exceptions import DataValidationError


def test_csv_upload_and_column_detection():
    loaded = load_dataset(b"Date,Weekly_Sales,Store\n2023-01-06,100,1\n2023-01-13,200,1\n", "sales.csv")
    assert len(loaded.frame) == 2
    assert detect_columns(loaded.frame)["sales"] == "Weekly_Sales"
    assert len(loaded.sha256) == 64


def test_excel_upload_and_sheet_selection():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"Date": ["2024-01-01"], "Sales": [12]}).to_excel(writer, sheet_name="Sales", index=False)
        pd.DataFrame({"Date": ["2024-02-01"], "Sales": [24]}).to_excel(writer, sheet_name="Other", index=False)
    loaded = load_dataset(buffer.getvalue(), "sales.xlsx", sheet_name="Other")
    assert loaded.frame["Sales"].iloc[0] == 24
    assert loaded.sheets == ("Sales", "Other")


@pytest.mark.parametrize("content,name", [(b"", "empty.csv"), (b"a,b\n", "empty.csv"), (b"abc", "data.txt"), (b"invalid workbook", "data.xlsx"), (b"Date,Date\n1,2", "duplicate.csv")])
def test_invalid_files_raise_user_facing_errors(content, name):
    with pytest.raises(DataValidationError):
        load_dataset(content, name)


def test_date_numeric_validation_and_explicit_exclusion():
    raw = pd.DataFrame({"Date": ["2024-01-01", "bad", "2024-01-03", "2024-01-04"], "Sales": ["10", "20", "bad", "-4"]})
    mapping = ColumnMapping("Date", "Sales")
    quality = validate_dataset(raw, mapping)
    assert (quality.invalid_dates, quality.invalid_sales, quality.negative_sales) == (1, 1, 1)
    with pytest.raises(DataValidationError):
        clean_dataset(raw, mapping)
    cleaned = clean_dataset(raw, mapping, CleaningOptions(invalid_rows="drop"))
    assert cleaned.records["sales"].tolist() == [10.0, -4.0]
    assert len(cleaned.rejected_records) == 2
    assert sum(operation["affected_rows"] for operation in cleaned.operations if operation["operation"] == "Invalid-record policy") == 2


def test_duplicates_preserved_by_default_and_explicit_drop():
    frame = pd.DataFrame({"Date": ["2024-01-01"] * 2, "Sales": [10, 10]})
    mapping = ColumnMapping("Date", "Sales")
    assert len(clean_dataset(frame, mapping).records) == 2
    result = clean_dataset(frame, mapping, CleaningOptions(duplicates="drop"))
    assert len(result.records) == 1
    assert result.rejected_records["_rejection_reason"].iloc[0] == "Exact duplicate"


def test_missing_observation_is_not_genuine_zero():
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-03", "2024-01-04"], "Sales": [0, 20, 30]})
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"))
    assert dataset.series.iloc[0] == 0
    assert np.isnan(dataset.series.iloc[1])
    assert dataset.observed_mask.tolist() == [True, False, True, True]
    filled = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(missing_periods="zero"))
    assert filled.series.iloc[1] == 0
    assert not filled.observed_mask.iloc[1]
    assert dataset.fingerprint != filled.fingerprint


def test_partial_missing_transactions_do_not_undercount_period():
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"], "Sales": [10, np.nan, 20, 30]})
    result = process_dataset(frame, ColumnMapping("Date", "Sales"), CleaningOptions(missing_sales="keep"), AggregationOptions(missing_periods="ffill"))
    assert np.isnan(result.observed_series.iloc[0])
    assert not result.model_ready


def test_transaction_aggregation_sorting_and_filters():
    frame = pd.DataFrame({"Date": ["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03"], "Sales": [30, 10, 20, 40], "Store": [1, 1, 2, 1]})
    result = process_dataset(frame, ColumnMapping("Date", "Sales"), filters={"Store": [1]})
    assert result.series.tolist() == [10, 30, 40]
    assert result.series.index.is_monotonic_increasing
    assert any(item["operation"] == "Filter Store" for item in result.operations)


def test_weekly_data_cannot_be_made_daily():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-05", periods=6, freq="W-FRI"), "Sales": range(6)})
    with pytest.raises(DataValidationError, match="cannot be disaggregated"):
        process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(frequency="D"))
    result = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(frequency="W-FRI"))
    assert result.frequency == "W-FRI"


def test_revenue_derivation_and_negative_returns():
    frame = pd.DataFrame({"InvoiceDate": ["2024-01-01", "2024-01-02"], "Quantity": [2, -1], "UnitPrice": [5, 7]})
    result = process_dataset(frame, ColumnMapping("InvoiceDate", quantity="Quantity", unit_price="UnitPrice"))
    assert result.series.tolist() == [10, -7]
    assert result.quality.negative_sales == 1


def test_forward_fill_is_causal():
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-03", "2024-01-04"], "Sales": [10, 900, 1000]})
    result = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(missing_periods="ffill"))
    assert result.series.iloc[1] == 10


def test_walmart_joins_do_not_multiply_rows():
    train = pd.DataFrame({"Store": [1, 1], "Dept": [1, 2], "Date": ["2024-01-05"] * 2, "Weekly_Sales": [10, 20]})
    stores = pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [100]})
    features = pd.DataFrame({"Store": [1], "Date": ["2024-01-05"], "Temperature": [20]})
    result, audit = merge_walmart_files(train, stores, features)
    assert len(result) == 2
    assert len(audit) == 2
    with pytest.raises(DataValidationError):
        merge_walmart_files(train, pd.concat([stores, stores]))
    with pytest.raises(DataValidationError):
        merge_walmart_files(train, features=pd.concat([features, features]))


def test_daily_to_monthly_aggregation(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(frequency="MS"))
    assert dataset.frequency == "MS"
    assert len(dataset.series) == 3
    assert dataset.series.sum() == pytest.approx(daily_frame["Sales"].sum())


def test_invalid_regressor_and_holiday_flags(daily_frame):
    daily_frame["Promotion"] = "bad"
    with pytest.raises(DataValidationError, match="Regressor"):
        process_dataset(daily_frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(regressors=("Promotion",)))
    daily_frame["Holiday"] = "unknown"
    with pytest.raises(DataValidationError, match="Holiday"):
        process_dataset(daily_frame, ColumnMapping("Date", "Sales", holiday="Holiday"))


def test_date_window_and_original_source_rows_are_audited():
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], "Sales": [10, 20, 30, 40], "Store": [2, 1, 1, 1]})
    result = process_dataset(frame, ColumnMapping("Date", "Sales"), filters={"Store": [1]}, date_range=("2024-01-03", "2024-01-04"))
    assert result.series.tolist() == [30, 40]
    assert result.records["_source_row"].tolist() == [3, 4]
    operations = [row["operation"] for row in result.operations]
    assert operations.index("Chronological sorting") < operations.index("Select historical date window")
    assert result.metadata["date_range"] == ("2024-01-03", "2024-01-04")


def test_invalid_dates_are_reviewed_before_date_window():
    frame = pd.DataFrame({"Date": ["bad", "2024-01-02", "2024-01-03"], "Sales": [10, 20, 30]})
    with pytest.raises(DataValidationError, match="invalid dates"):
        process_dataset(frame, ColumnMapping("Date", "Sales"), date_range=("2024-01-02", "2024-01-03"))
