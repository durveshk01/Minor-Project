"""Explicit data policies and immutable processing configuration."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColumnMapping:
    date: str
    sales: str | None = None
    quantity: str | None = None
    unit_price: str | None = None
    holiday: str | None = None


@dataclass(frozen=True)
class CleaningOptions:
    invalid_rows: str = "reject"
    duplicates: str = "keep"
    missing_sales: str = "reject"
    negative_sales: str = "keep"
    date_format: str | None = None
    day_first: bool = False


@dataclass(frozen=True)
class AggregationOptions:
    frequency: str = "D"
    method: str = "sum"
    missing_periods: str = "keep"
    source_frequency: str = "auto"
    regressors: tuple[str, ...] = ()


@dataclass
class ValidationReport:
    rows: int
    columns: int
    missing_by_column: dict[str, int]
    duplicates: int
    invalid_dates: int
    missing_dates: int
    invalid_sales: int
    missing_sales: int
    negative_sales: int
    zero_sales: int
    suggested_frequency: str | None


@dataclass
class CleaningResult:
    records: pd.DataFrame
    validation: ValidationReport
    operations: list[dict[str, Any]]
    rejected_records: pd.DataFrame


@dataclass
class ProcessedDataset:
    series: pd.Series
    observed_series: pd.Series
    observed_mask: pd.Series
    records: pd.DataFrame
    regressors: pd.DataFrame
    holidays: pd.DataFrame
    quality: ValidationReport
    operations: list[dict[str, Any]]
    rejected_records: pd.DataFrame
    missing_periods: list[str]
    outliers: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def frequency(self) -> str:
        return self.series.index.freqstr

    @property
    def model_ready(self) -> bool:
        return bool(len(self.series) and self.series.notna().all())
