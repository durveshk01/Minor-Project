"""Expanding-window rolling-origin validation with training-only preprocessing."""

from dataclasses import dataclass, field

import pandas as pd

from src.evaluation.metrics import calculate_metrics, prediction_errors
from src.forecasting.forecast_service import predict_model, train_model
from src.forecasting.types import ModelConfig
from src.utils.exceptions import DataValidationError, SalesVisionError


@dataclass
class BacktestResult:
    comparison: pd.DataFrame
    folds: pd.DataFrame
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)


def chronological_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    if not isinstance(test_size, int) or isinstance(test_size, bool) or test_size < 1 or test_size >= len(series):
        raise DataValidationError("Holdout size must be a positive number smaller than the historical series.")
    return series.iloc[:-test_size].copy(), series.iloc[-test_size:].copy()


def rolling_origin_backtest(
    series: pd.Series,
    configurations: list[ModelConfig],
    horizon: int,
    folds: int = 3,
    observed_series: pd.Series | None = None,
    regressors: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    known_regressors: bool = False,
) -> BacktestResult:
    if type(folds) is not int or type(horizon) is not int or not 1 <= folds <= 5 or not 1 <= horizon <= 366 or len(series) - horizon * folds < 14:
        raise DataValidationError("Rolling backtesting needs 1–5 folds and sufficient initial training history.")
    if not configurations or any(not isinstance(config, ModelConfig) or not isinstance(config.name, str) for config in configurations) or len({config.name for config in configurations}) != len(configurations):
        raise DataValidationError("Select each forecasting model at most once for backtesting.")
    truth = observed_series if observed_series is not None else series
    if not truth.index.equals(series.index):
        raise DataValidationError("Observed targets must align with the modelling calendar.")
    all_predictions = {}
    fold_rows = []
    comparison = []
    origins = [len(series) - folds * horizon + fold * horizon for fold in range(folds)]
    for config in configurations:
        model_tables = []
        error_message = ""
        durations = []
        for fold, origin in enumerate(origins, 1):
            train = series.iloc[:origin]
            test_truth = truth.iloc[origin:origin + horizon]
            try:
                if config.regressor_columns and not known_regressors:
                    raise DataValidationError("Backtesting external regressors requires values known before each forecast origin.")
                historical_exog = regressors.iloc[:origin] if regressors is not None and config.regressor_columns else None
                future_exog = regressors.iloc[origin:origin + horizon] if regressors is not None and config.regressor_columns else None
                calendar = holidays.loc[holidays["ds"].le(train.index[-1])] if holidays is not None and not holidays.empty else None
                fitted = train_model(train, config, historical_exog, calendar)
                table = prediction_errors(test_truth, predict_model(fitted, horizon, future_exog))
                observed = table.loc[table["is_observed"]]
                metrics = calculate_metrics(observed["actual"], observed["prediction"])
                durations.append(fitted.training_seconds)
                table["origin"] = train.index[-1]
                table["fold"] = fold
                model_tables.append(table)
                fold_rows.append({"Model": config.name, "Fold": fold, "Train End": train.index[-1], "Test Start": test_truth.index[0], "Test End": test_truth.index[-1], **metrics, "Training Time": fitted.training_seconds, "Status": "success"})
            except SalesVisionError as exc:
                error_message = str(exc)
                fold_rows.append({"Model": config.name, "Fold": fold, "Train End": train.index[-1], "Test Start": test_truth.index[0], "Test End": test_truth.index[-1], "Status": "failed", "Error": error_message})
                break
        metrics = {}
        if model_tables:
            combined = pd.concat(model_tables, ignore_index=True)
            all_predictions[config.name] = combined
            observed = combined.loc[combined["is_observed"]]
            metrics = calculate_metrics(observed["actual"], observed["prediction"])
        comparison.append({"Model": config.name, **metrics, "Training Time": sum(durations), "Completed Folds": len(model_tables), "Status": "success" if len(model_tables) == folds else "failed", "Error": error_message})
    return BacktestResult(pd.DataFrame(comparison), pd.DataFrame(fold_rows), all_predictions)
