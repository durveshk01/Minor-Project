"""Real-model training, calendar forecasts, fair evaluation and leakage checks."""

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import process_dataset
from src.data.schema import AggregationOptions, ColumnMapping
from src.evaluation.backtesting import chronological_split, rolling_origin_backtest
from src.evaluation.model_comparison import evaluate_models
from src.forecasting.common import future_dates
from src.forecasting.forecast_service import generate_forecast, predict_model, train_model
from src.forecasting.types import ModelConfig
from src.utils.exceptions import ForecastError


@pytest.mark.integration
@pytest.mark.parametrize("name", ["ARIMA", "SARIMA", "Prophet"])
def test_real_models_generate_monthly_forecasts(name, monthly_series):
    config = ModelConfig(name, order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), yearly_seasonality=False)
    fitted = train_model(monthly_series, config)
    forecast = predict_model(fitted, 6)
    assert len(forecast) == 6
    assert forecast["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert forecast["date"].iloc[-1] == pd.Timestamp("2024-06-01")
    assert np.isfinite(forecast[["prediction", "lower", "upper"]].to_numpy()).all()
    assert (forecast["upper"] >= forecast["lower"]).all()
    assert fitted.training_seconds > 0
    assert fitted.summary


def test_weekly_forecast_dates_preserve_friday():
    history = pd.Series(np.arange(30) + 100, index=pd.date_range("2024-01-05", periods=30, freq="W-FRI"))
    dates = future_dates(history, 4)
    assert (dates.dayofweek == 4).all()
    assert dates[0] - history.index[-1] == pd.Timedelta(days=7)


def test_daily_forecast_dates(daily_frame):
    history = daily_frame.set_index("Date")["Sales"].asfreq("D")
    assert future_dates(history, 7)[0] == history.index[-1] + pd.Timedelta(days=1)


def test_chronological_split_never_randomizes(monthly_series):
    train, test = chronological_split(monthly_series, 12)
    assert len(train) == 60 and len(test) == 12
    assert train.index[-1] < test.index[0]
    assert test.index.equals(monthly_series.index[-12:])


def test_missing_and_short_training_data_are_rejected(monthly_series):
    missing = monthly_series.copy()
    missing.iloc[20] = np.nan
    with pytest.raises(ForecastError, match="Missing"):
        train_model(missing, ModelConfig("ARIMA"))
    with pytest.raises(ForecastError):
        train_model(monthly_series.iloc[:5], ModelConfig("SARIMA"))


def test_prophet_does_not_enable_unidentifiable_weekly_pattern(monthly_series):
    with pytest.raises(ForecastError, match="daily observations"):
        train_model(monthly_series, ModelConfig("Prophet", weekly_seasonality=True))
    with pytest.raises(ForecastError, match="Intraday"):
        train_model(monthly_series, ModelConfig("Prophet", daily_seasonality=True))


@pytest.mark.integration
def test_comparison_uses_same_holdout_and_full_retraining(daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"))
    configs = [ModelConfig("ARIMA", order=(1, 1, 0)), ModelConfig("SARIMA", order=(1, 0, 0), seasonal_order=(0, 1, 0, 7)), ModelConfig("Prophet", yearly_seasonality=False)]
    evaluation = evaluate_models(dataset, configs, 14)
    assert len(evaluation.predictions) == 3, evaluation.comparison.to_string()
    for table in evaluation.predictions.values():
        assert pd.DatetimeIndex(table["date"]).equals(dataset.series.index[-14:])
        np.testing.assert_allclose(table["actual"], dataset.observed_series.iloc[-14:])
    assert evaluation.best_model in evaluation.predictions
    selected = evaluation.configurations[evaluation.best_model]
    run = generate_forecast(dataset, selected, 7, evaluation.metrics_for(evaluation.best_model))
    assert run.training_observations == 90
    assert run.training_end == dataset.series.index[-1].isoformat()
    assert run.predictions["date"].min() > dataset.series.index[-1]


@pytest.mark.integration
def test_imputed_holdout_targets_are_not_scored(daily_frame):
    frame = daily_frame.drop(index=[80, 85]).reset_index(drop=True)
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"), aggregation=AggregationOptions(missing_periods="ffill"))
    result = evaluate_models(dataset, [ModelConfig("ARIMA", order=(1, 1, 0))], 14)
    assert result.comparison["Scored Observations"].iloc[0] == 12
    assert result.predictions["ARIMA"]["actual"].isna().sum() == 2


@pytest.mark.integration
def test_rolling_origin_training_never_includes_test(monthly_series):
    result = rolling_origin_backtest(monthly_series, [ModelConfig("ARIMA", order=(1, 1, 0))], horizon=6, folds=2)
    assert len(result.folds) == 2
    assert (result.folds["Train End"] < result.folds["Test Start"]).all()
    assert result.comparison["Completed Folds"].iloc[0] == 2
    assert result.comparison["Scored Observations"].iloc[0] == 12


@pytest.mark.integration
def test_sarimax_requires_future_regressors(monthly_series):
    rng = np.random.default_rng(8)
    regressors = pd.DataFrame({"Promotion": rng.integers(0, 2, len(monthly_series))}, index=monthly_series.index)
    config = ModelConfig("SARIMA", order=(0, 1, 0), seasonal_order=(0, 1, 0, 12), regressor_columns=("Promotion",))
    fitted = train_model(monthly_series, config, regressors)
    with pytest.raises(ForecastError, match="Supply all"):
        predict_model(fitted, 3)
    future = pd.DataFrame({"Promotion": [0, 1, 0]}, index=future_dates(fitted.history, 3))
    result = predict_model(fitted, 3, future)
    assert len(result) == 3
    assert fitted.regressor_mean["Promotion"] == pytest.approx(regressors["Promotion"].mean())


@pytest.mark.integration
def test_negative_raw_predictions_are_preserved_by_business_constraint():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=40), "Sales": np.linspace(-50, -100, 40)})
    dataset = process_dataset(frame, ColumnMapping("Date", "Sales"))
    run = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 4, business_constraint="nonnegative")
    assert run.predictions["prediction"].lt(0).all()
    assert run.predictions["business_prediction"].eq(0).all()
    assert run.predictions["business_adjustment"].gt(0).all()


@pytest.mark.integration
@pytest.mark.parametrize("name", ["ARIMA", "SARIMA", "Prophet"])
def test_real_models_preserve_weekly_calendar_and_generate_intervals(name):
    dates = pd.date_range("2020-01-03", periods=156, freq="W-FRI")
    rng = np.random.default_rng(47)
    values = 1000 + np.arange(156) * 2 + 100 * np.sin(2 * np.pi * np.arange(156) / 52) + rng.normal(0, 10, 156)
    history = pd.Series(values, index=dates)
    config = ModelConfig(name, order=(1, 1, 0), seasonal_order=(0, 1, 0, 52), yearly_seasonality=name == "Prophet")
    fitted = train_model(history, config)
    table = predict_model(fitted, 12)
    expected = pd.date_range(dates[-1] + pd.Timedelta(weeks=1), periods=12, freq="W-FRI")
    assert pd.DatetimeIndex(table["date"]).equals(expected)
    assert np.isfinite(table[["prediction", "lower", "upper"]].to_numpy()).all()
    assert table["upper"].gt(table["lower"]).all()
    assert fitted.name == name and fitted.training_seconds > 0


@pytest.mark.integration
def test_random_walk_arima_and_seasonal_random_walk_match_analytical_predictions(monthly_series):
    arima = train_model(monthly_series, ModelConfig("ARIMA", order=(0, 1, 0)))
    np.testing.assert_allclose(predict_model(arima, 6)["prediction"], monthly_series.iloc[-1], rtol=1e-9, atol=1e-6)
    sarima = train_model(monthly_series, ModelConfig("SARIMA", order=(0, 0, 0), seasonal_order=(0, 1, 0, 12)))
    np.testing.assert_allclose(predict_model(sarima, 6)["prediction"], monthly_series.iloc[-12:-6], rtol=1e-9, atol=1e-6)
