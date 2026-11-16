"""Policy-controlled cleaning that records every transformation and exclusion."""

import numpy as np
import pandas as pd

from src.data.schema import CleaningOptions, CleaningResult, ColumnMapping
from src.data.validator import convert_dates, convert_sales, validate_dataset
from src.utils.exceptions import DataValidationError


def clean_dataset(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    options: CleaningOptions | None = None,
) -> CleaningResult:
    options = options or CleaningOptions()
    policies = {
        "invalid_rows": (options.invalid_rows, {"reject", "drop"}),
        "duplicates": (options.duplicates, {"keep", "drop"}),
        "missing_sales": (options.missing_sales, {"reject", "drop", "keep"}),
        "negative_sales": (options.negative_sales, {"reject", "drop", "keep"}),
    }
    for name, (value, allowed) in policies.items():
        if value not in allowed:
            raise DataValidationError(f"Unsupported {name} policy: {value}")
    if "_source_row" in frame.columns or "_rejection_reason" in frame.columns:
        raise DataValidationError("Rename reserved _source_row/_rejection_reason columns before upload.")
    quality = validate_dataset(frame, mapping, options.date_format, options.day_first)
    dates = convert_dates(frame[mapping.date], options.date_format, options.day_first)
    sales = convert_sales(frame, mapping)
    raw_missing = frame[mapping.sales].isna() if mapping.sales else frame[[mapping.quantity, mapping.unit_price]].isna().any(axis=1)
    invalid = dates.isna() | (sales.isna() & ~raw_missing)
    if invalid.any() and options.invalid_rows == "reject":
        raise DataValidationError(
            f"{int(invalid.sum())} records have missing/invalid dates or invalid numeric sales. "
            "Correct them or explicitly choose to exclude invalid rows."
        )
    if raw_missing.any() and options.missing_sales == "reject":
        raise DataValidationError(
            f"{int(raw_missing.sum())} sales inputs are missing. Choose a missing-sales policy."
        )
    if sales.lt(0).any() and options.negative_sales == "reject":
        raise DataValidationError("Negative sales are present. Review returns/cancellations and choose a policy.")
    records = frame.copy()
    records["_source_row"] = np.arange(1, len(frame) + 1)
    records["date"] = dates.dt.normalize()
    records["sales"] = sales
    reasons = pd.Series("", index=frame.index, dtype="string")
    operations = [
        {"operation": "Convert dates and normalize to calendar date", "affected_rows": len(frame), "detail": f"Source: {mapping.date}; format: {options.date_format or 'mixed'}; day-first: {options.day_first}; timezone-aware inputs normalized to UTC."},
        {"operation": "Convert sales to numeric", "affected_rows": len(frame), "detail": f"Source: {mapping.sales or (str(mapping.quantity) + ' × ' + str(mapping.unit_price))}; non-finite values are invalid."},
    ]
    actions = [
        (invalid, options.invalid_rows == "drop", "Invalid date or sales", "Invalid-record policy"),
        (raw_missing, options.missing_sales == "drop", "Missing sales", "Missing-sales policy"),
        (sales.lt(0), options.negative_sales == "drop", "Negative sales", "Negative-sales policy"),
        (frame.duplicated(), options.duplicates == "drop", "Exact duplicate", "Duplicate policy"),
    ]
    for mask, exclude, reason, operation in actions:
        new_rejections = mask & reasons.eq("") if exclude else pd.Series(False, index=frame.index)
        reasons.loc[new_rejections] = reason
        operations.append({"operation": operation, "affected_rows": int(new_rejections.sum()), "detail": f"Detected {int(mask.sum())}; {'excluded' if exclude else 'retained'} by explicit policy."})
    rejected = records.loc[reasons.ne("")].copy()
    rejected["_rejection_reason"] = reasons.loc[reasons.ne("")]
    records = records.loc[reasons.eq("")].sort_values("date", kind="stable").reset_index(drop=True)
    if records.empty:
        raise DataValidationError("No usable records remain after the selected cleaning policies.")
    operations.append({"operation": "Chronological sorting", "affected_rows": len(records), "detail": "Stable date sort; source-row identifiers retained."})
    return CleaningResult(records, quality, operations, rejected)


def detect_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.DataFrame:
    values = series.dropna()
    if len(values) < 4:
        return pd.DataFrame(columns=["date", "sales", "lower_threshold", "upper_threshold"])
    q1, q3 = values.quantile([0.25, 0.75])
    spread = q3 - q1
    lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
    selected = values[(values < lower) | (values > upper)]
    return pd.DataFrame({
        "date": selected.index, "sales": selected.values,
        "lower_threshold": lower, "upper_threshold": upper,
    })
