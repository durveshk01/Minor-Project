"""Shared training dispatch and full-history final-forecast orchestration."""

import logging
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.data.schema import ProcessedDataset
from src.forecasting.arima_model import forecast_arima, train_arima
from src.forecasting.common import future_dates, validate_config, validate_prediction_table
from src.forecasting.prophet_model import forecast_prophet, train_prophet, validate_holidays
from src.forecasting.sarima_model import forecast_sarima, train_sarima
from src.forecasting.types import FittedModel, ForecastRun, ModelConfig
from src.utils.exceptions import ForecastError


def train_model(
    series: pd.Series,
    config: ModelConfig,
    regressors: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
) -> FittedModel:
    validate_config(config)
    if config.name == "ARIMA":
        return train_arima(series, config)
    if config.name == "SARIMA":
        return train_sarima(series, config, regressors)
    if config.name == "Prophet":
        return train_prophet(series, config, holidays, regressors)
    raise ForecastError(f"Unknown forecasting model: {config.name}")


def predict_model(fitted: FittedModel, horizon: int, future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
    expected = future_dates(fitted.history, horizon)
    if fitted.name == "ARIMA":
        table = forecast_arima(fitted, horizon)
    elif fitted.name == "SARIMA":
        table = forecast_sarima(fitted, horizon, future_regressors)
    elif fitted.name == "Prophet":
        table = forecast_prophet(fitted, horizon, future_regressors)
    else:
        raise ForecastError("The fitted model type is unsupported.")
    return validate_prediction_table(table, expected)


def generate_forecast(
    dataset: ProcessedDataset,
    config: ModelConfig,
    horizon: int,
    evaluation_metrics: dict | None = None,
    future_regressors: pd.DataFrame | None = None,
    future_holidays: pd.DataFrame | None = None,
    business_constraint: str = "none",
) -> ForecastRun:
    if not dataset.model_ready:
        raise ForecastError("Resolve unknown sales periods before generating a final forecast.")
    if business_constraint not in {"none", "nonnegative"}:
        raise ForecastError("Business constraint must be none or nonnegative.")
    expected = future_dates(dataset.series, horizon)
    future_calendar = validate_holidays(future_holidays) if future_holidays is not None else None
    if future_calendar is not None and not future_calendar["ds"].isin(expected).all():
        raise ForecastError("Future holiday dates must exactly match periods in the forecast calendar.")
    holidays = dataset.holidays
    if future_calendar is not None:
        holidays = pd.concat([holidays, future_calendar], ignore_index=True).drop_duplicates()
    fitted = train_model(dataset.series, config, dataset.regressors, holidays)
    table = predict_model(fitted, horizon, future_regressors)
    notes = list(fitted.warnings)
    if config.use_holidays and future_calendar is None:
        notes.append("No future holiday calendar was supplied; historical flags do not imply holiday dates in the forecast horizon.")
    negative_count = int(table["prediction"].lt(0).sum())
    if negative_count:
        notes.append(f"{negative_count} raw point forecasts are negative. They may reflect net returns or model extrapolation; review business suitability.")
    if business_constraint == "nonnegative":
        for source, destination in [("prediction", "business_prediction"), ("lower", "business_lower"), ("upper", "business_upper")]:
            table[destination] = table[source].clip(lower=0)
        table["business_adjustment"] = table["business_prediction"] - table["prediction"]
        notes.append("Business view applies a floor of zero. Raw forecasts and original model intervals are retained; clipped endpoints do not retain calibrated interval coverage.")
    logging.getLogger("salesvision").info("Forecast completed model=%s fingerprint=%s horizon=%d", config.name, dataset.fingerprint[:12], horizon)
    return ForecastRun(
        run_id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=dataset.fingerprint,
        metadata=dict(dataset.metadata), model=config.name, configuration=config.to_dict(),
        frequency=dataset.frequency, horizon=horizon, predictions=table,
        training_seconds=fitted.training_seconds, training_observations=len(dataset.series),
        training_start=dataset.series.index[0].isoformat(), training_end=dataset.series.index[-1].isoformat(),
        evaluation_metrics=evaluation_metrics or {}, model_summary=fitted.summary,
        warnings=notes, components=fitted.components, business_constraint=business_constraint,
    )
