"""Observed-data summaries and evidence-based seasonal decomposition."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.data.aggregator import frequency_family
from src.data.validator import convert_holiday_flags
from src.utils.exceptions import DataValidationError, InsufficientDataError


@dataclass
class SeasonalResult:
    components: pd.DataFrame
    period: int
    strength: float
    amplitude: float
    residual_std: float
    explanation: str


def default_seasonal_period(frequency: str) -> int:
    return {"daily": 7, "weekly": 52, "monthly": 12}[frequency_family(frequency)]


def statistical_summary(series: pd.Series) -> dict:
    observed = series.dropna().astype(float)
    if observed.empty:
        raise DataValidationError("No observed sales are available for statistical summaries.")
    growth = None if observed.iloc[0] == 0 else float((observed.iloc[-1] - observed.iloc[0]) / abs(observed.iloc[0]) * 100)
    return {
        "total": float(observed.sum()),
        "mean": float(observed.mean()),
        "median": float(observed.median()),
        "std": float(observed.std()) if len(observed) > 1 else None,
        "min": float(observed.min()),
        "max": float(observed.max()),
        "growth_pct": growth,
        "observations": int(len(observed)),
        "calendar_periods": int(len(series)),
        "missing_periods": int(series.isna().sum()),
        "zero_periods": int(observed.eq(0).sum()),
        "negative_periods": int(observed.lt(0).sum()),
        "start": series.index.min().isoformat(),
        "end": series.index.max().isoformat(),
        "growth_definition": "Percentage change from the first to the latest observed period, relative to the absolute first value; undefined when the first value is zero.",
    }


def growth_series(series: pd.Series) -> pd.Series:
    return (series.diff() / series.shift(1).abs().replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan).rename("growth_pct")


def trend_rollups(series: pd.Series, frequency: str) -> dict[str, pd.Series]:
    family = frequency_family(frequency)
    results = {"Historical sales": series}
    if family == "daily":
        results["Daily sales"] = series
        results["Weekly sales"] = series.resample("W-SUN").sum(min_count=1)
    elif family == "weekly":
        results["Weekly sales"] = series
    results["Monthly sales"] = series.resample("MS").sum(min_count=1)
    return results


def monthly_comparison(series: pd.Series) -> pd.DataFrame:
    totals = series.resample("MS").sum(min_count=1)
    frame = pd.DataFrame({"date": totals.index, "sales": totals.values})
    frame["year"] = frame["date"].dt.year.astype(str)
    frame["month"] = frame["date"].dt.month
    frame["month_name"] = frame["date"].dt.strftime("%b")
    return frame


def yearly_comparison(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"sales": series})
    frame["year"] = frame.index.year
    result = frame.groupby("year")["sales"].agg(total=lambda values: values.sum(min_count=1), observed_periods="count").reset_index()
    result["year"] = result["year"].astype(str)
    return result


def weekday_summary(series: pd.Series, frequency: str) -> pd.DataFrame:
    if frequency_family(frequency) != "daily":
        raise DataValidationError("Weekday analysis requires daily observations; weekly data cannot reveal daily behavior.")
    frame = pd.DataFrame({"sales": series, "weekday_number": series.index.dayofweek})
    result = frame.groupby("weekday_number")["sales"].agg(mean="mean", total=lambda values: values.sum(min_count=1), observations="count").reset_index()
    result["weekday"] = result["weekday_number"].map(dict(enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])))
    return result


def dimension_summary(records: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in records:
        raise DataValidationError(f"The requested comparison column is unavailable: {column}")
    grouped = records.groupby(column, dropna=False)["sales"]
    result = pd.DataFrame({"total_sales": grouped.sum(min_count=1), "mean_record_sales": grouped.mean(), "records": grouped.count()}).reset_index()
    return result.sort_values("total_sales", ascending=False)


def holiday_summary(records: pd.DataFrame, holiday_column: str) -> pd.DataFrame:
    flags = convert_holiday_flags(records[holiday_column])
    valid = flags.notna()
    frame = records.loc[valid, ["date", "sales"]].copy()
    frame["holiday"] = flags.loc[valid]
    grouped = frame.groupby(["date", "holiday"])["sales"]
    period = grouped.sum(min_count=1).mask(grouped.size() > grouped.count()).reset_index()
    return period.groupby("holiday")["sales"].agg(mean_period_sales="mean", total_sales=lambda values: values.sum(min_count=1), observed_dates="count").reset_index()


def seasonal_decomposition(series: pd.Series, period: int | None = None) -> SeasonalResult:
    period = default_seasonal_period(series.index.freqstr) if period is None else period
    if type(period) is not int or not 2 <= period <= 366:
        raise DataValidationError("A seasonal period must be an integer of at least 2.")
    if len(series) < 2 * period:
        raise InsufficientDataError(
            f"Decomposition at period {period} needs at least {2 * period} observations; only {len(series)} are available."
        )
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise DataValidationError("Resolve missing periods explicitly before seasonal decomposition.")
    result = STL(series.astype(float), period=period, robust=True).fit()
    components = pd.DataFrame({"observed": series, "trend": result.trend, "seasonal": result.seasonal, "residual": result.resid}, index=series.index)
    denominator = float(np.var(result.resid + result.seasonal))
    strength = float(max(0, 1 - np.var(result.resid) / denominator)) if denominator > 0 else 0.0
    amplitude = float(np.quantile(result.seasonal, 0.95) - np.quantile(result.seasonal, 0.05))
    residual_std = float(np.std(result.resid))
    evidence = "A relatively strong repeating component is present" if strength >= 0.6 else "The estimated repeating component is moderate" if strength >= 0.3 else "The data provides weak evidence for this repeating component"
    explanation = (
        f"{evidence} at a period of {period} observations. "
        f"STL seasonal strength is {strength:.3f}; the 5th–95th percentile seasonal amplitude is {amplitude:,.2f} sales units. "
        f"Residual standard deviation is {residual_std:,.2f}. "
        "This is a descriptive decomposition, not proof of a causal holiday effect or a guarantee of future seasonality."
    )
    return SeasonalResult(components, period, strength, amplitude, residual_std, explanation)


def summarize_forecast(history: pd.Series, forecast: pd.DataFrame) -> dict:
    predictions = forecast["prediction"].astype(float)
    horizon = len(predictions)
    preceding = history.iloc[-horizon:]
    comparable = len(preceding) == horizon and preceding.notna().all() and preceding.sum() != 0
    growth = float((predictions.sum() - preceding.sum()) / abs(preceding.sum()) * 100) if comparable else None
    return {
        "horizon": horizon,
        "forecast_total": float(predictions.sum()),
        "forecast_mean": float(predictions.mean()),
        "forecast_min": float(predictions.min()),
        "forecast_max": float(predictions.max()),
        "negative_predictions": int(predictions.lt(0).sum()),
        "growth_vs_preceding_horizon_pct": growth,
        "preceding_total": float(preceding.sum()) if comparable else None,
        "start": pd.Timestamp(forecast["date"].min()).isoformat(),
        "end": pd.Timestamp(forecast["date"].max()).isoformat(),
        "growth_definition": "Sum of raw future point forecasts versus the immediately preceding equal-length, fully observed historical window. Undefined if history is incomplete, too short or sums to zero.",
        "interval_note": "Intervals are per period. Their endpoints are not summed into a purported joint interval for total sales.",
    }
