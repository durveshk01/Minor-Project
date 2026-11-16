"""Statsmodels ARIMA fitting and interval forecasts with recorded diagnostics."""

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.forecasting.common import future_dates, validate_config, validate_series
from src.forecasting.types import FittedModel, ModelConfig
from src.utils.exceptions import ForecastError


def train_arima(series: pd.Series, config: ModelConfig) -> FittedModel:
    validate_config(config)
    if config.regressor_columns:
        raise ForecastError("External regressors are supported in SARIMAX; select SARIMA for this configuration.")
    history = validate_series(series, max(12, 3 * (config.order[0] + config.order[2] + 1)))
    scale = max(float(history.std(ddof=0)), 1.0)
    started = time.perf_counter()
    optimizer_note = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = ARIMA(history / scale, order=config.order, trend=config.trend)
            result = estimator.fit(method_kwargs={"maxiter": config.maxiter})
            if not result.mle_retvals.get("converged", True):
                result = estimator.fit(method_kwargs={"method": "powell", "maxiter": config.maxiter})
                optimizer_note.append("LBFGS did not converge; retried the same model using Powell optimization.")
        if not result.mle_retvals.get("converged", True):
            raise ForecastError("ARIMA optimization did not converge after LBFGS and Powell. Simplify p/d/q or increase optimizer iterations.")
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(f"ARIMA fitting failed ({type(exc).__name__}). Review data and p/d/q parameters.") from exc
    return FittedModel(
        "ARIMA", config, result, history, time.perf_counter() - started,
        f"Sales were scaled by {scale:.6g} for optimization; forecasts are restored to original units.\n{result.summary()}",
        list(dict.fromkeys(str(item.message)[:300] for item in caught)) + optimizer_note, scale=scale,
    )


def forecast_arima(fitted: FittedModel, horizon: int) -> pd.DataFrame:
    dates = future_dates(fitted.history, horizon)
    try:
        result = fitted.estimator.get_forecast(steps=horizon)
        interval = np.asarray(result.conf_int(alpha=1 - fitted.config.interval_width)) * fitted.scale
        return pd.DataFrame({"date": dates, "prediction": np.asarray(result.predicted_mean) * fitted.scale, "lower": interval[:, 0], "upper": interval[:, 1]})
    except Exception as exc:
        raise ForecastError("ARIMA could not generate future predictions and intervals.") from exc
