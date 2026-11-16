"""Non-destructive column detection, type conversion and quality diagnostics."""

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from src.data.schema import ColumnMapping, ValidationReport
from src.utils.exceptions import DataValidationError

WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    aliases = {
        "date": ("date", "invoicedate", "orderdate", "salesdate", "transactiondate", "ds", "month", "week"),
        "sales": ("weeklysales", "sales", "salesamount", "revenue", "totalsales", "netsales", "amount", "y"),
        "quantity": ("quantity", "qty", "units"),
        "unit_price": ("unitprice", "price", "sellingprice"),
        "store": ("store", "storeid", "shop", "branch"),
        "department": ("dept", "department", "departmentid"),
        "category": ("category", "productcategory", "product", "stockcode", "productid"),
        "region": ("region", "country", "state", "territory"),
        "holiday": ("isholiday", "holiday", "holidayflag"),
    }
    normalized = {normalize_name(column): column for column in frame.columns}
    detected = {
        role: next((normalized[alias] for alias in options if alias in normalized), None)
        for role, options in aliases.items()
    }
    if detected["date"] is None:
        detected["date"] = next((column for column in frame if is_datetime64_any_dtype(frame[column])), None)
    return detected


def validate_mapping(frame: pd.DataFrame, mapping: ColumnMapping) -> None:
    selected = [mapping.date, mapping.sales, mapping.quantity, mapping.unit_price, mapping.holiday]
    missing = [column for column in selected if column and column not in frame.columns]
    if missing:
        raise DataValidationError(f"Selected columns do not exist: {', '.join(missing)}")
    if not mapping.date:
        raise DataValidationError("Select a date column.")
    if not mapping.sales and not (mapping.quantity and mapping.unit_price):
        raise DataValidationError("Select a sales column or both Quantity and UnitPrice to derive revenue.")
    if mapping.sales == mapping.date or mapping.date in {mapping.quantity, mapping.unit_price}:
        raise DataValidationError("Date and sales inputs must be different columns.")


def convert_dates(values: pd.Series, date_format: str | None = None, day_first: bool = False) -> pd.Series:
    if is_numeric_dtype(values) and not is_datetime64_any_dtype(values) and not date_format:
        raise DataValidationError(
            "A numeric date column needs an explicit format such as %Y%m%d. "
            "Use real Excel date cells for spreadsheet dates."
        )
    source = values.astype("string") if is_numeric_dtype(values) and date_format else values
    converted = pd.to_datetime(source, errors="coerce", format=date_format or "mixed", dayfirst=day_first, utc=True)
    return converted.dt.tz_convert(None)


def convert_sales(frame: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
    validate_mapping(frame, mapping)
    if mapping.sales:
        values = pd.to_numeric(frame[mapping.sales], errors="coerce")
    else:
        values = pd.to_numeric(frame[mapping.quantity], errors="coerce") * pd.to_numeric(frame[mapping.unit_price], errors="coerce")
    return values.astype(float).replace([np.inf, -np.inf], np.nan)


def convert_holiday_flags(values: pd.Series) -> pd.Series:
    """Interpret supported flag spellings consistently while retaining unknown flags."""
    source = values.astype("string").str.strip().str.lower()
    meanings = {"true": True, "1": True, "1.0": True, "yes": True, "false": False, "0": False, "0.0": False, "no": False}
    if (source.notna() & ~source.isin(meanings)).any():
        raise DataValidationError("Holiday flags must be boolean, 0/1 or yes/no.")
    return source.map(meanings).astype("boolean")


def infer_frequency(dates: pd.Series | pd.DatetimeIndex) -> str | None:
    index = pd.DatetimeIndex(dates).dropna().normalize().unique().sort_values()
    if len(index) < 3:
        return None
    inferred = pd.infer_freq(index)
    if inferred:
        if inferred.startswith("W-"):
            return inferred
        if inferred in {"MS", "ME", "M"}:
            return "MS"
        if inferred == "D":
            return "D"
    deltas = np.diff(index.values).astype("timedelta64[D]").astype(int)
    if len(set(index.dayofweek)) == 1 and (deltas % 7 == 0).all():
        return f"W-{WEEKDAYS[index[0].dayofweek]}"
    months = index.to_period("M")
    if months.is_unique and np.median(deltas) >= 27 and (index.is_month_start.all() or index.is_month_end.all() or len(set(index.day)) == 1):
        return "MS"
    if np.median(deltas) <= 3:
        return "D"
    return None


def validate_dataset(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    date_format: str | None = None,
    day_first: bool = False,
) -> ValidationReport:
    validate_mapping(frame, mapping)
    dates = convert_dates(frame[mapping.date], date_format, day_first)
    sales = convert_sales(frame, mapping)
    raw_missing = frame[mapping.sales].isna() if mapping.sales else frame[[mapping.quantity, mapping.unit_price]].isna().any(axis=1)
    return ValidationReport(
        rows=len(frame),
        columns=len(frame.columns),
        missing_by_column={str(key): int(value) for key, value in frame.isna().sum().items()},
        duplicates=int(frame.duplicated().sum()),
        invalid_dates=int((dates.isna() & frame[mapping.date].notna()).sum()),
        missing_dates=int(frame[mapping.date].isna().sum()),
        invalid_sales=int((sales.isna() & ~raw_missing).sum()),
        missing_sales=int(raw_missing.sum()),
        negative_sales=int(sales.lt(0).sum()),
        zero_sales=int(sales.eq(0).sum()),
        suggested_frequency=infer_frequency(dates),
    )
