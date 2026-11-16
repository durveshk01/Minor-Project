"""Bounded computation caches keyed by complete data and configuration fingerprints."""

import streamlit as st

from src.data.loader import load_dataset
from src.data.pipeline import process_dataset
from src.evaluation.model_comparison import evaluate_models
from src.features.time_features import seasonal_decomposition
from src.forecasting.forecast_service import generate_forecast


@st.cache_data(max_entries=6, ttl=3600, show_spinner=False)
def cached_load(content: bytes, filename: str, sheet_name=0, max_size_mb: int = 100):
    return load_dataset(content, filename, sheet_name=sheet_name, max_size_mb=max_size_mb)


@st.cache_data(max_entries=6, ttl=3600, show_spinner=False)
def cached_process(frame, mapping, cleaning, aggregation, metadata, filters, prior_operations, date_range):
    return process_dataset(frame, mapping, cleaning, aggregation, metadata, filters, prior_operations, date_range)


@st.cache_data(max_entries=12, ttl=3600, show_spinner=False)
def cached_evaluate(_dataset, dataset_fingerprint, configurations, test_size, ranking_metric, backtest_folds, backtest_horizon, known_regressors):
    if _dataset.fingerprint != dataset_fingerprint:
        raise ValueError("Evaluation cache fingerprint mismatch.")
    return evaluate_models(_dataset, list(configurations), test_size, ranking_metric, backtest_folds, backtest_horizon, known_regressors)


@st.cache_data(max_entries=12, ttl=3600, show_spinner=False)
def cached_forecast(_dataset, dataset_fingerprint, config, horizon, evaluation_metrics, future_regressors, future_holidays, business_constraint):
    if _dataset.fingerprint != dataset_fingerprint:
        raise ValueError("Forecast cache fingerprint mismatch.")
    return generate_forecast(_dataset, config, horizon, evaluation_metrics, future_regressors, future_holidays, business_constraint)


@st.cache_data(max_entries=12, ttl=3600, show_spinner=False)
def cached_seasonality(_series, dataset_fingerprint, period):
    return seasonal_decomposition(_series, period)
