"""Seasonal evidence and observed-data summary tests."""

import numpy as np
import pandas as pd
import pytest

from src.features.time_features import (
    seasonal_decomposition,
    statistical_summary,
    summarize_forecast,
    weekday_summary,
)
from src.utils.exceptions import DataValidationError, InsufficientDataError
from src.visualization.charts import decomposition_chart, historical_chart


def test_statistics_ignore_unknown_and_preserve_genuine_zero():
    series = pd.Series([0, 10, np.nan, 20], index=pd.date_range("2024-01-01", periods=4, freq="D"))
    summary = statistical_summary(series)
    assert summary["total"] == 30
    assert summary["mean"] == 10
    assert summary["observations"] == 3
    assert summary["missing_periods"] == 1
    assert summary["growth_pct"] is None


def test_stl_reconstructs_actual_series(monthly_series):
    result = seasonal_decomposition(monthly_series, 12)
    reconstructed = result.components[["trend", "seasonal", "residual"]].sum(axis=1)
    np.testing.assert_allclose(reconstructed, monthly_series)
    assert 0 <= result.strength <= 1
    assert result.strength > 0.8
    assert len(decomposition_chart(result.components).data) == 4


def test_insufficient_seasonal_observations(monthly_series):
    with pytest.raises(InsufficientDataError, match="24"):
        seasonal_decomposition(monthly_series.iloc[:20], 12)


def test_weekday_analysis_rejects_weekly_data(monthly_series):
    with pytest.raises(DataValidationError, match="requires daily"):
        weekday_summary(monthly_series, "W-FRI")


def test_forecast_growth_requires_equal_complete_history():
    history = pd.Series([10, 20, np.nan], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    forecast = pd.DataFrame({"date": pd.date_range("2024-01-04", periods=2), "prediction": [30, -5]})
    summary = summarize_forecast(history, forecast)
    assert summary["forecast_total"] == 25
    assert summary["negative_predictions"] == 1
    assert summary["growth_vs_preceding_horizon_pct"] is None
    assert len(historical_chart(history).data[0].y) == 3
