"""Official Prophet package with frequency-aware seasonality and genuine intervals."""

import json
import threading
import time
import warnings

import numpy as np
import pandas as pd

from src.data.aggregator import frequency_family
from src.data.validator import convert_dates
from src.forecasting.common import (
    future_dates,
    validate_config,
    validate_regressors,
    validate_series,
)
from src.forecasting.types import FittedModel, ModelConfig
from src.utils.exceptions import DataValidationError, ForecastError, InsufficientDataError

PREDICTION_LOCK = threading.RLock()


def seasonality_settings(history: pd.Series, config: ModelConfig) -> tuple[bool, bool]:
    family = frequency_family(history.index.freqstr)
    span = (history.index[-1] - history.index[0]).days
    yearly_supported = len(history) >= {"daily": 730, "weekly": 104, "monthly": 24}[family]
    weekly_supported = family == "daily" and span >= 13 and len(history) >= 14
    if config.daily_seasonality:
        raise ForecastError("Intraday seasonality is not identifiable in daily, weekly or monthly aggregated sales.")
    if config.yearly_seasonality is True and not yearly_supported:
        raise InsufficientDataError("Prophet yearly seasonality requires at least two years of training history.")
    if config.weekly_seasonality is True and not weekly_supported:
        raise ForecastError("Prophet within-week seasonality needs daily observations spanning at least two weeks.")
    yearly = yearly_supported if config.yearly_seasonality is None else config.yearly_seasonality
    weekly = weekly_supported if config.weekly_seasonality is None else config.weekly_seasonality
    return bool(yearly), bool(weekly)


def validate_holidays(holidays: pd.DataFrame | None) -> pd.DataFrame | None:
    if holidays is None:
        return None
    if not isinstance(holidays, pd.DataFrame) or not holidays.columns.is_unique or not {"ds", "holiday"}.issubset(holidays.columns):
        raise ForecastError("Holiday data requires ds and holiday columns with actual calendar dates.")
    if holidays.empty:
        return None
    result = holidays[["ds", "holiday"]].copy()
    try:
        result["ds"] = convert_dates(result["ds"]).dt.normalize()
    except DataValidationError as exc:
        raise ForecastError("Holiday dates must be valid calendar dates.") from exc
    result["holiday"] = result["holiday"].astype("string").str.strip()
    if result.isna().any().any() or result["holiday"].astype(str).str.strip().eq("").any():
        raise ForecastError("Holiday calendar contains invalid dates or empty holiday names.")
    return result.drop_duplicates().reset_index(drop=True)


def train_prophet(
    series: pd.Series,
    config: ModelConfig,
    holidays: pd.DataFrame | None = None,
    regressors: pd.DataFrame | None = None,
) -> FittedModel:
    validate_config(config)
    history = validate_series(series, 14)
    yearly, weekly = seasonality_settings(history, config)
    if config.seasonality_mode not in {"additive", "multiplicative"}:
        raise ForecastError("Prophet seasonality mode must be additive or multiplicative.")
    if config.seasonality_mode == "multiplicative" and history.le(0).any():
        raise ForecastError("Use additive seasonality when sales include zero or negative values.")
    if not 0.001 <= config.changepoint_prior_scale <= 1:
        raise ForecastError("Prophet changepoint prior scale must be between 0.001 and 1.")
    exog = validate_regressors(regressors, history.index, config.regressor_columns)
    calendar = validate_holidays(holidays) if config.use_holidays else None
    if config.use_holidays and (calendar is None or not calendar["ds"].isin(history.index).any()):
        raise ForecastError("Holiday effects need at least one supplied holiday matching the training calendar; use its daily, weekly anchor or monthly-start dates.")
    frame = pd.DataFrame({"ds": history.index, "y": history.values})
    if exog is not None:
        for column in exog:
            frame[column] = exog[column].to_numpy()
    started = time.perf_counter()
    try:
        from prophet import Prophet

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Prophet(
                yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=False,
                seasonality_mode=config.seasonality_mode,
                changepoint_prior_scale=config.changepoint_prior_scale,
                interval_width=config.interval_width, uncertainty_samples=500,
                holidays=calendar,
            )
            for column in config.regressor_columns:
                model.add_regressor(column)
            model.fit(frame, seed=config.seed)
    except ImportError as exc:
        raise ForecastError("Prophet is not installed. Install the project requirements in Python 3.12.") from exc
    except Exception as exc:
        raise ForecastError(f"Prophet fitting failed ({type(exc).__name__}). Check package installation and selected settings.") from exc
    duration = time.perf_counter() - started
    summary = json.dumps({
        "package": "prophet", "training_observations": len(history),
        "yearly_seasonality": yearly, "weekly_seasonality": weekly,
        "daily_seasonality": False, "seasonalities": model.seasonalities,
        "seasonality_mode": config.seasonality_mode,
        "changepoint_prior_scale": config.changepoint_prior_scale,
        "changepoints": [value.isoformat() for value in model.changepoints],
        "holidays": len(calendar) if calendar is not None else 0,
        "regressors": list(config.regressor_columns), "interval_width": config.interval_width,
        "interval_method": "Prophet predictive simulation, 500 uncertainty draws; intervals are not a coverage guarantee.",
    }, indent=2, default=str)
    notes = list(dict.fromkeys(str(item.message)[:300] for item in caught))
    if config.yearly_seasonality is None and not yearly:
        notes.append("Yearly seasonality automatically disabled: less than two years of training history.")
    return FittedModel("Prophet", config, model, history, duration, summary, notes)


def seeded_prediction(model, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    with PREDICTION_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            return model.predict(frame)
        finally:
            np.random.set_state(state)


def forecast_prophet(fitted: FittedModel, horizon: int, future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
    dates = future_dates(fitted.history, horizon)
    exog = validate_regressors(future_regressors, dates, fitted.config.regressor_columns)
    frame = pd.DataFrame({"ds": dates})
    if exog is not None:
        for column in exog:
            frame[column] = exog[column].to_numpy()
    try:
        result = seeded_prediction(fitted.estimator, frame, fitted.config.seed)
        component_names = [column for column in result if column in {"trend", "weekly", "yearly", "holidays", "additive_terms", "multiplicative_terms"}]
        fitted.components = result[["ds", *component_names]].rename(columns={"ds": "date"})
        table = pd.DataFrame({"date": dates, "prediction": result["yhat"].values, "lower": result["yhat_lower"].values, "upper": result["yhat_upper"].values})
        for column in component_names:
            table[f"component_{column}"] = result[column].to_numpy()
        return table
    except Exception as exc:
        raise ForecastError("Prophet could not generate a valid future forecast.") from exc
