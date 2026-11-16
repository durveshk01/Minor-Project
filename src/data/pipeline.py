"""One reusable processing workflow for Streamlit, tests and Colab."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from src.data.aggregator import aggregate_records
from src.data.cleaner import clean_dataset, detect_outliers
from src.data.schema import AggregationOptions, CleaningOptions, ColumnMapping, ProcessedDataset
from src.data.validator import convert_dates
from src.utils.exceptions import DataValidationError


def process_dataset(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    cleaning: CleaningOptions | None = None,
    aggregation: AggregationOptions | None = None,
    metadata: dict[str, Any] | None = None,
    filters: dict[str, list] | None = None,
    prior_operations: list[dict] | None = None,
    date_range: tuple[str, str] | None = None,
) -> ProcessedDataset:
    cleaning = cleaning or CleaningOptions()
    aggregation = aggregation or AggregationOptions()
    target_inputs = {mapping.date, mapping.sales, mapping.quantity, mapping.unit_price, "date", "sales"} - {None}
    if not isinstance(aggregation.regressors, tuple) or any(not isinstance(column, str) or not column for column in aggregation.regressors):
        raise DataValidationError("Regressor columns must be a tuple of unique column names.")
    if len(set(aggregation.regressors)) != len(aggregation.regressors):
        raise DataValidationError("Regressor column names must be unique.")
    if target_inputs.intersection(aggregation.regressors):
        raise DataValidationError("Date, sales target and revenue-derivation inputs cannot be external regressors.")
    frame = frame.reset_index(drop=True)
    selected = frame
    filter_operations = []
    window_operations = []
    for column, values in (filters or {}).items():
        if column not in selected:
            raise DataValidationError(f"Filter column does not exist: {column}")
        if not isinstance(values, (list, tuple)):
            raise DataValidationError("Supply filter values as a list or tuple.")
        before = len(selected)
        selected = selected.loc[selected[column].isin(values)]
        filter_operations.append({"operation": f"Filter {column}", "affected_rows": before - len(selected), "detail": f"Selected {len(values)} values; {len(selected)} records remain."})
    if selected.empty:
        raise DataValidationError("No records match the selected filters.")
    cleaned = clean_dataset(selected.reset_index(drop=True), mapping, cleaning)
    source_positions = selected.index.to_numpy() + 1
    cleaned.records["_source_row"] = source_positions[cleaned.records["_source_row"].to_numpy() - 1]
    if not cleaned.rejected_records.empty:
        cleaned.rejected_records["_source_row"] = source_positions[cleaned.rejected_records["_source_row"].to_numpy() - 1]
    if date_range is not None:
        if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
            raise DataValidationError("Supply both the historical date-window start and end.")
        bounds = convert_dates(pd.Series(date_range)).dt.normalize()
        if bounds.isna().any():
            raise DataValidationError("The historical date window contains missing or invalid dates.")
        start, end = bounds.tolist()
        if start > end:
            raise DataValidationError("The historical date-window start must precede its end.")
        before = len(cleaned.records)
        cleaned.records = cleaned.records.loc[cleaned.records["date"].between(start, end)].copy()
        window_operations.append({"operation": "Select historical date window", "affected_rows": before - len(cleaned.records), "detail": f"{start.date()} to {end.date()}, inclusive; applied after date validation and audited cleaning."})
        if cleaned.records.empty:
            raise DataValidationError("No cleaned records fall within the selected historical window.")
    series, observed, regressors, holidays, operations = aggregate_records(cleaned.records, aggregation, mapping.holiday)
    if observed.notna().sum() < 2:
        raise DataValidationError("At least two observed sales periods must remain after aggregation.")
    outliers = detect_outliers(observed)
    operations.append({"operation": "Flag period outliers using 1.5 × IQR", "affected_rows": len(outliers), "detail": "Diagnostic only. No values clipped, replaced or removed; no full-series thresholds used by models."})
    context = dict(metadata or {})
    context.update({
        "input_rows": len(frame), "filtered_rows": len(selected), "cleaned_rows": len(cleaned.records),
        "frequency": series.index.freqstr, "aggregation": aggregation.method,
        "mapping": asdict(mapping), "cleaning": asdict(cleaning),
        "aggregation_options": asdict(aggregation), "filters": filters or {},
        "date_range": date_range,
    })
    digest = hashlib.sha256(pd.util.hash_pandas_object(series, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(observed, index=True).values.tobytes())
    if not regressors.empty:
        digest.update(pd.util.hash_pandas_object(regressors, index=True).values.tobytes())
    digest.update(holidays.to_json(date_format="iso").encode())
    digest.update(json.dumps(context, sort_keys=True, default=str).encode())
    return ProcessedDataset(
        series=series,
        observed_series=observed,
        observed_mask=observed.notna(),
        records=cleaned.records,
        regressors=regressors,
        holidays=holidays,
        quality=cleaned.validation,
        operations=(prior_operations or []) + filter_operations + cleaned.operations + window_operations + operations,
        rejected_records=cleaned.rejected_records,
        missing_periods=[date.isoformat() for date in observed.index[observed.isna()]],
        outliers=outliers,
        metadata=context,
        fingerprint=digest.hexdigest(),
    )
