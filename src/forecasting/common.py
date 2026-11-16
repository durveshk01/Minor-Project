"""Input, calendar and exogenous-data validation shared by genuine models."""

import numpy as np
import pandas as pd

from src.data.aggregator import normalize_frequency
from src.forecasting.types import ModelConfig
from src.utils.exceptions import DataValidationError, ForecastError, InsufficientDataError


def validate_config(config: ModelConfig) -> None:
    if not isinstance(config, ModelConfig):
        raise ForecastError("Supply a ModelConfig for ARIMA, SARIMA or Prophet.")
    if not isinstance(config.name, str) or config.name not in {"ARIMA", "SARIMA", "Prophet"}:
        raise ForecastError(f"Unknown forecasting model: {config.name}")
    if not isinstance(config.order, tuple) or len(config.order) != 3 or any(type(value) is not int or not 0 <= value <= 5 for value in config.order):
        raise ForecastError("ARIMA p, d, q must be integer values between 0 and 5.")
    if not isinstance(config.seasonal_order, tuple) or len(config.seasonal_order) != 4 or any(type(value) is not int or value < 0 for value in config.seasonal_order):
        raise ForecastError("Seasonal parameters P, D, Q, s must be non-negative integers.")
    if any(value > 5 for value in config.seasonal_order[:3]):
        raise ForecastError("SARIMA P, D and Q must be between 0 and 5.")
    if config.name == "SARIMA" and (config.seasonal_order[3] < 2 or config.seasonal_order[3] > 366):
        raise ForecastError("SARIMA seasonal period must be between 2 and 366.")
    if not isinstance(config.interval_width, (int, float)) or isinstance(config.interval_width, bool) or not np.isfinite(config.interval_width) or not 0.5 <= config.interval_width < 1:
        raise ForecastError("Forecast interval width must be at least 0.5 and less than 1.")
    if type(config.maxiter) is not int or not 20 <= config.maxiter <= 1000:
        raise ForecastError("Optimizer iterations must be between 20 and 1000.")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ForecastError("The random seed must be an integer between 0 and 4294967295.")
    if any(value is not None and type(value) is not bool for value in (config.yearly_seasonality, config.weekly_seasonality)) or type(config.daily_seasonality) is not bool or type(config.use_holidays) is not bool:
        raise ForecastError("Seasonality and holiday flags must be boolean; auto seasonality uses None.")
    if config.seasonality_mode not in ("additive", "multiplicative"):
        raise ForecastError("Prophet seasonality mode must be additive or multiplicative.")
    if not isinstance(config.changepoint_prior_scale, (int, float)) or isinstance(config.changepoint_prior_scale, bool) or not np.isfinite(config.changepoint_prior_scale) or not 0.001 <= config.changepoint_prior_scale <= 1:
        raise ForecastError("Prophet changepoint prior scale must be between 0.001 and 1.")
    if config.trend is not None and config.trend not in ("n", "c", "t", "ct"):
        raise ForecastError("Trend must be None, n, c, t or ct.")
    if not isinstance(config.regressor_columns, tuple) or any(not isinstance(column, str) or not column.strip() for column in config.regressor_columns):
        raise ForecastError("Regressor columns must be a tuple of unique, non-empty names.")
    forbidden = {"sales", "weeklysales", "salesamount", "revenue", "y", "actual", "prediction"}
    if any(column.lower().replace("_", "").replace(" ", "") in forbidden for column in config.regressor_columns):
        raise ForecastError("Sales targets or predictions cannot be used as external regressors.")
    if len(set(config.regressor_columns)) != len(config.regressor_columns):
        raise ForecastError("Regressor column names must be unique.")


def validate_series(series: pd.Series, minimum: int = 12) -> pd.Series:
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ForecastError("Models require a sales Series indexed by calendar dates.")
    if len(series) < minimum:
        raise InsufficientDataError(f"At least {minimum} observations are required; received {len(series)}.")
    if series.index.tz is not None or not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ForecastError("Training dates must be unique, sorted and timezone-naive.")
    try:
        values = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise ForecastError("Sales must be numeric.") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise ForecastError("Missing or invalid sales remain. Choose an explicit missing-period policy before training.")
    frequency = series.index.freqstr or (pd.infer_freq(series.index) if len(series) >= 3 else None)
    if not frequency:
        raise ForecastError("A regular daily, weekly or monthly calendar is required.")
    try:
        frequency = normalize_frequency(frequency)
    except DataValidationError as exc:
        raise ForecastError(str(exc)) from exc
    expected = pd.date_range(values.index[0], periods=len(values), freq=frequency)
    if not expected.equals(values.index):
        raise ForecastError("Training dates do not form a complete calendar at the selected frequency.")
    values.index = expected.rename("date")
    values.name = "sales"
    return values


def future_dates(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 366:
        raise ForecastError("Forecast horizon must be an integer between 1 and 366 periods.")
    history = validate_series(history, minimum=1)
    return pd.date_range(history.index[-1], periods=horizon + 1, freq=history.index.freq)[1:].rename("date")


def validate_regressors(
    regressors: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    columns: tuple[str, ...],
) -> pd.DataFrame | None:
    if not columns:
        return None
    if not isinstance(regressors, pd.DataFrame) or not regressors.columns.is_unique or not set(columns).issubset(regressors.columns):
        raise ForecastError("Supply all configured external regressor columns for every required date.")
    if not isinstance(regressors.index, pd.DatetimeIndex) or not regressors.index.equals(dates):
        raise ForecastError("External regressors must exactly match the training or forecast dates in chronological order.")
    try:
        numeric = regressors.loc[:, list(columns)].astype(float)
    except (TypeError, ValueError) as exc:
        raise ForecastError("External regressors must be numeric.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ForecastError("External regressors contain missing or non-finite values; no automatic filling is applied.")
    return numeric


def scaled_training_regressors(regressors: pd.DataFrame | None):
    if regressors is None:
        return None, None, None
    mean = regressors.mean()
    scale = regressors.std(ddof=0).replace(0, 1)
    return (regressors - mean) / scale, mean, scale


def validate_prediction_table(table: pd.DataFrame, expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    required = {"date", "prediction", "lower", "upper"}
    if not isinstance(table, pd.DataFrame) or not table.columns.is_unique or not required.issubset(table) or len(table) != len(expected_dates):
        raise ForecastError("The model returned an invalid forecast shape.")
    result = table.copy().reset_index(drop=True)
    try:
        result["date"] = pd.to_datetime(result["date"], format="mixed", errors="raise")
        result[["prediction", "lower", "upper"]] = result[["prediction", "lower", "upper"]].astype(float)
    except (ValueError, TypeError) as exc:
        raise ForecastError("The model returned invalid forecast dates or numeric values.") from exc
    if not pd.DatetimeIndex(result["date"]).equals(expected_dates):
        raise ForecastError("The model returned misaligned forecast dates.")
    if not np.isfinite(result["prediction"].to_numpy()).all():
        raise ForecastError("The model returned non-finite point predictions.")
    for column in ("lower", "upper"):
        if not np.isfinite(result[column].to_numpy()).all():
            raise ForecastError("The model returned invalid forecast intervals.")
    if (result["lower"] > result["upper"]).any():
        raise ForecastError("The model returned inverted forecast intervals.")
    return result
