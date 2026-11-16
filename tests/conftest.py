"""Small reproducible fixtures with explicit synthetic provenance."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    index = np.arange(len(dates))
    return pd.DataFrame({"Date": dates, "Sales": 500 + index * 2 + 35 * np.sin(index * 2 * np.pi / 7)})


@pytest.fixture
def monthly_series() -> pd.Series:
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    index = np.arange(len(dates))
    noise = np.random.default_rng(42).normal(0, 1.5, len(dates))
    return pd.Series(200 + index * 1.2 + 20 * np.sin(index * 2 * np.pi / 12) + noise, index=dates, name="sales")
