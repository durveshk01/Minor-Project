"""Frequency-aware aggregation without inventing finer-grained observations."""

import numpy as np
import pandas as pd

from src.data.schema import AggregationOptions
from src.data.validator import WEEKDAYS, convert_holiday_flags, infer_frequency
from src.utils.exceptions import DataValidationError


def frequency_family(frequency: str) -> str:
    if not isinstance(frequency, str):
        raise DataValidationError("Choose a daily, anchored weekly or monthly frequency.")
    frequency = frequency.upper()
    if frequency == "D":
        return "daily"
    if frequency == "W" or frequency in {f"W-{day}" for day in WEEKDAYS}:
        return "weekly"
    if frequency in {"MS", "ME", "M"}:
        return "monthly"
    raise DataValidationError("Choose a daily, anchored weekly or monthly frequency.")


def normalize_frequency(frequency: str) -> str:
    family = frequency_family(frequency)
    if family == "monthly":
        return "MS"
    return "W-SUN" if frequency.upper() == "W" else frequency.upper()


def validate_aggregation_frequency(records: pd.DataFrame, options: AggregationOptions) -> None:
    target = normalize_frequency(options.frequency)
    native = infer_frequency(records["date"]) if options.source_frequency == "auto" else normalize_frequency(options.source_frequency)
    if native:
        rank = {"daily": 1, "weekly": 2, "monthly": 3}
        if rank[frequency_family(target)] < rank[frequency_family(native)]:
            raise DataValidationError(
                f"{frequency_family(native).title()} observations cannot be disaggregated into "
                f"{frequency_family(target)} sales. Select the native or a coarser frequency."
            )


def aggregate_records(
    records: pd.DataFrame,
    options: AggregationOptions,
    holiday_column: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, list[dict]]:
    if options.method not in {"sum", "mean"}:
        raise DataValidationError("Aggregation must be sum or mean.")
    if options.missing_periods not in {"keep", "zero", "ffill", "reject"}:
        raise DataValidationError("Choose keep, zero, forward-fill or reject for missing periods.")
    validate_aggregation_frequency(records, options)
    frequency = normalize_frequency(options.frequency)
    indexed = records.set_index("date").sort_index()
    grouped = indexed["sales"].resample(frequency)
    observed = grouped.sum(min_count=1) if options.method == "sum" else grouped.mean()
    incomplete = grouped.size() > grouped.count()
    observed.loc[incomplete] = np.nan
    if not np.isfinite(observed.dropna().to_numpy()).all():
        raise DataValidationError("Aggregated sales exceed the supported numeric range; review source amounts and units.")
    observed.name = "sales"
    observed.index.name = "date"
    missing = observed.isna()
    series = observed.copy()
    if missing.any():
        if options.missing_periods == "reject":
            raise DataValidationError(f"{int(missing.sum())} periods are absent or have incomplete sales.")
        if options.missing_periods == "zero":
            series = series.fillna(0.0)
        elif options.missing_periods == "ffill":
            series = series.ffill()
    operations = [
        {"operation": "Aggregate sales", "affected_rows": len(records), "detail": f"{options.method} into {len(series)} {frequency} periods; periods with any missing sales remain unknown."},
        {"operation": "Build continuous calendar index", "affected_rows": int(missing.sum()), "detail": f"{int((grouped.size() == 0).sum())} absent periods and {int(incomplete.sum())} incomplete periods detected."},
        {"operation": "Missing-period policy", "affected_rows": int(missing.sum()), "detail": f"Policy: {options.missing_periods}. Original observations and imputed periods remain distinguishable; imputed targets are not scored during evaluation."},
    ]
    regressors = pd.DataFrame(index=series.index)
    for column in options.regressors:
        if column not in indexed:
            raise DataValidationError(f"Regressor column does not exist: {column}")
        numeric = pd.to_numeric(indexed[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if (numeric.isna() & indexed[column].notna()).any():
            raise DataValidationError(f"Regressor {column} contains invalid numeric values.")
        regressor_groups = numeric.resample(frequency)
        regressors[column] = regressor_groups.mean().mask(regressor_groups.size() > regressor_groups.count())
    if options.regressors:
        operations.append({"operation": "Aggregate selected regressors", "affected_rows": len(records), "detail": "Numeric period means; no automatic filling or fabricated future values."})
    holidays = pd.DataFrame(columns=["ds", "holiday"])
    if holiday_column:
        flags = convert_holiday_flags(records[holiday_column])
        holidays = pd.DataFrame({
            "ds": records.loc[flags.fillna(False), "date"].drop_duplicates(),
            "holiday": "observed_holiday",
        }).reset_index(drop=True)
        operations.append({"operation": "Extract supplied holiday flags", "affected_rows": len(holidays), "detail": "Historical holiday dates only; no future holidays inferred."})
    return series, observed, regressors, holidays, operations
