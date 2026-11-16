"""Fair holdout comparisons and explicit, metric-driven model selection."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.schema import ProcessedDataset
from src.evaluation.backtesting import BacktestResult, chronological_split, rolling_origin_backtest
from src.evaluation.metrics import calculate_metrics, prediction_errors
from src.forecasting.forecast_service import predict_model, train_model
from src.forecasting.types import ModelConfig
from src.utils.exceptions import DataValidationError, SalesVisionError


@dataclass
class EvaluationResult:
    dataset_fingerprint: str
    comparison: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    configurations: dict[str, ModelConfig]
    best_model: str | None
    ranking_metric: str
    train_size: int
    test_size: int
    training_end: str
    test_start: str
    test_end: str
    summaries: dict[str, str] = field(default_factory=dict)
    warnings: dict[str, list[str]] = field(default_factory=dict)
    backtest: BacktestResult | None = None

    def metrics_for(self, model: str) -> dict:
        matching = self.comparison.loc[(self.comparison["Model"] == model) & self.comparison["Status"].eq("success")]
        if matching.empty:
            return {}
        return {key: value for key, value in matching.iloc[0].to_dict().items() if key not in {"Model", "Status", "Error"} and pd.notna(value)}


def select_best_model(comparison: pd.DataFrame, metric: str = "RMSE") -> str | None:
    if metric not in {"RMSE", "MAE", "MAPE", "sMAPE"}:
        raise DataValidationError("Choose RMSE, MAE, MAPE or sMAPE as the ranking metric.")
    if metric not in comparison:
        return None
    scores = pd.to_numeric(comparison[metric], errors="coerce")
    candidates = comparison.loc[comparison["Status"].eq("success") & np.isfinite(scores) & scores.ge(0)].copy()
    candidates[metric] = scores.loc[candidates.index]
    if candidates.empty:
        return None
    return str(candidates.sort_values([metric, "Model"]).iloc[0]["Model"])


def evaluate_models(
    dataset: ProcessedDataset,
    configurations: list[ModelConfig],
    test_size: int,
    ranking_metric: str = "RMSE",
    backtest_folds: int = 0,
    backtest_horizon: int | None = None,
    known_regressors: bool = False,
) -> EvaluationResult:
    if not dataset.model_ready:
        raise DataValidationError("Resolve missing periods before model evaluation.")
    if not configurations or any(not isinstance(config, ModelConfig) or not isinstance(config.name, str) for config in configurations) or len({config.name for config in configurations}) != len(configurations):
        raise DataValidationError("Select each forecasting model at most once.")
    if ranking_metric not in {"RMSE", "MAE", "MAPE", "sMAPE"}:
        raise DataValidationError("The comparison metric is unsupported.")
    if type(backtest_folds) is not int or not 0 <= backtest_folds <= 5:
        raise DataValidationError("Backtest folds must be an integer between 0 and 5.")
    if backtest_horizon is not None and (type(backtest_horizon) is not int or not 1 <= backtest_horizon <= 366):
        raise DataValidationError("Backtesting horizon must be an integer between 1 and 366.")
    if type(test_size) is not int or not 2 <= test_size <= 366:
        raise DataValidationError("The chronological holdout must be an integer between 2 and 366 periods.")
    train, test = chronological_split(dataset.series, test_size)
    actual = dataset.observed_series.iloc[-test_size:]
    if actual.notna().sum() < 2:
        raise DataValidationError("The holdout needs at least two genuine observed sales targets; imputed targets are not scored.")
    rows = []
    predictions = {}
    summaries = {}
    notes = {}
    for config in configurations:
        try:
            if config.regressor_columns and not known_regressors:
                raise DataValidationError("External regressors may be evaluated only when their holdout values were known at the forecast origin. Confirm this in Model Training.")
            historical_exog = dataset.regressors.iloc[:-test_size] if config.regressor_columns else None
            future_exog = dataset.regressors.iloc[-test_size:] if config.regressor_columns else None
            calendar = dataset.holidays.loc[dataset.holidays["ds"].le(train.index[-1])] if not dataset.holidays.empty else None
            fitted = train_model(train, config, historical_exog, calendar)
            table = prediction_errors(actual, predict_model(fitted, test_size, future_exog))
            observed = table.loc[table["is_observed"]]
            metrics = calculate_metrics(observed["actual"], observed["prediction"])
            predictions[config.name] = table
            summaries[config.name] = fitted.summary
            notes[config.name] = fitted.warnings
            rows.append({"Model": config.name, **metrics, "Training Time": fitted.training_seconds, "Status": "success", "Error": ""})
        except SalesVisionError as exc:
            rows.append({"Model": config.name, "Training Time": None, "Status": "failed", "Error": str(exc)})
    comparison = pd.DataFrame(rows).reindex(columns=["Model", "MAE", "RMSE", "MAPE", "sMAPE", "Bias", "Training Time", "Scored Observations", "MAPE Observations", "Zero Actuals", "Status", "Error"])
    result = EvaluationResult(
        dataset.fingerprint, comparison, predictions,
        {config.name: config for config in configurations}, select_best_model(comparison, ranking_metric),
        ranking_metric, len(train), len(test), train.index[-1].isoformat(),
        test.index[0].isoformat(), test.index[-1].isoformat(), summaries, notes,
    )
    if backtest_folds:
        result.backtest = rolling_origin_backtest(
            train, configurations, backtest_horizon or test_size, backtest_folds,
            dataset.observed_series.iloc[:-test_size], dataset.regressors.iloc[:-test_size],
            dataset.holidays, known_regressors,
        )
    return result
