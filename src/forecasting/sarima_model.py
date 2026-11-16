"""Statsmodels SARIMAX with seasonal orders and explicitly supplied regressors."""

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.forecasting.common import (
    future_dates,
    scaled_training_regressors,
    validate_config,
    validate_regressors,
    validate_series,
)
from src.forecasting.types import FittedModel, ModelConfig
from src.utils.exceptions import ForecastError


def train_sarima(series: pd.Series, config: ModelConfig, regressors: pd.DataFrame | None = None) -> FittedModel:
    validate_config(config)
    seasonal = config.seasonal_order
    minimum = max(20, 2 * seasonal[3]) if any(seasonal[:3]) else 20
    history = validate_series(series, minimum)
    raw_regressors = validate_regressors(regressors, history.index, config.regressor_columns)
    exog, exog_mean, exog_scale = scaled_training_regressors(raw_regressors)
    scale = max(float(history.std(ddof=0)), 1.0)
    started = time.perf_counter()
    optimizer_note = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = SARIMAX(
                history / scale, exog=exog, order=config.order, seasonal_order=seasonal,
                trend=config.trend, enforce_stationarity=True, enforce_invertibility=True,
            )
            result = estimator.fit(disp=False, maxiter=config.maxiter)
            if not result.mle_retvals.get("converged", True):
                result = estimator.fit(method="powell", disp=False, maxiter=config.maxiter)
                optimizer_note.append("LBFGS did not converge; retried the same model using Powell optimization.")
        if not result.mle_retvals.get("converged", True):
            raise ForecastError("SARIMA optimization did not converge after LBFGS and Powell. Simplify the orders or increase optimizer iterations.")
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(f"SARIMA fitting failed ({type(exc).__name__}). Review seasonal period and regressor values.") from exc
    return FittedModel(
        "SARIMA", config, result, history, time.perf_counter() - started,
        f"SARIMAX implementation; sales scaled by {scale:.6g}. Regressor scaling learned only from training data. Forecasts restored to original units.\n{result.summary()}",
        list(dict.fromkeys(str(item.message)[:300] for item in caught)) + optimizer_note,
        scale=scale, regressor_mean=exog_mean, regressor_scale=exog_scale,
    )


def forecast_sarima(fitted: FittedModel, horizon: int, future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
    dates = future_dates(fitted.history, horizon)
    raw = validate_regressors(future_regressors, dates, fitted.config.regressor_columns)
    exog = (raw - fitted.regressor_mean) / fitted.regressor_scale if raw is not None else None
    try:
        result = fitted.estimator.get_forecast(steps=horizon, exog=exog)
        interval = np.asarray(result.conf_int(alpha=1 - fitted.config.interval_width)) * fitted.scale
        return pd.DataFrame({"date": dates, "prediction": np.asarray(result.predicted_mean) * fitted.scale, "lower": interval[:, 0], "upper": interval[:, 1]})
    except Exception as exc:
        raise ForecastError("SARIMA forecasting failed. Check future regressor coverage and model stability.") from exc
