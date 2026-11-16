"""Forecast metrics with explicit zero handling and signed forecast bias."""

import numpy as np
import pandas as pd

from src.utils.exceptions import DataValidationError


def calculate_metrics(actual, predicted) -> dict:
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series) and not actual.index.equals(predicted.index):
        raise DataValidationError("Actual and predicted series must have identical date indexes.")
    try:
        truth = np.asarray(actual, dtype=float)
        estimate = np.asarray(predicted, dtype=float)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Metrics require numeric actual and predicted vectors.") from exc
    if truth.ndim != 1 or estimate.ndim != 1 or truth.shape != estimate.shape or not len(truth):
        raise DataValidationError("Actual and predicted values must be non-empty, equally sized vectors.")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise DataValidationError("Metrics require finite observed actuals and finite predictions; missing targets must be explicitly excluded.")
    with np.errstate(over="ignore", invalid="ignore"):
        errors = estimate - truth
    if not np.isfinite(errors).all():
        raise DataValidationError("Forecast errors exceed the supported numeric range.")
    absolute = np.abs(errors)
    nonzero = np.abs(truth) > 0
    scale = max(float(absolute.max()), 1.0)
    pair_scale = np.maximum(np.abs(truth), np.abs(estimate))
    scaled_truth = np.divide(np.abs(truth), pair_scale, out=np.zeros_like(truth), where=pair_scale > 0)
    scaled_estimate = np.divide(np.abs(estimate), pair_scale, out=np.zeros_like(estimate), where=pair_scale > 0)
    scaled_absolute = np.divide(absolute, pair_scale, out=np.zeros_like(absolute), where=pair_scale > 0)
    denominator = scaled_truth + scaled_estimate
    symmetric = np.divide(200 * scaled_absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mape = float(np.mean(absolute[nonzero] / np.abs(truth[nonzero])) * 100) if nonzero.any() else None
    if mape is not None and not np.isfinite(mape):
        raise DataValidationError("Percentage errors exceed the supported numeric range.")
    return {
        "MAE": float(np.mean(absolute / scale) * scale),
        "RMSE": float(np.sqrt(np.mean((errors / scale)**2)) * scale),
        "MAPE": mape,
        "sMAPE": float(np.mean(symmetric)),
        "Bias": float(np.mean(errors / scale) * scale),
        "Scored Observations": int(len(truth)),
        "MAPE Observations": int(nonzero.sum()),
        "Zero Actuals": int((~nonzero).sum()),
    }


def prediction_errors(actual: pd.Series, forecast: pd.DataFrame) -> pd.DataFrame:
    if not pd.DatetimeIndex(forecast["date"]).equals(actual.index):
        raise DataValidationError("Evaluation forecast dates do not match the holdout dates.")
    table = forecast.copy()
    table["actual"] = actual.to_numpy()
    table["is_observed"] = actual.notna().to_numpy()
    table["error"] = table["prediction"] - table["actual"]
    table["absolute_error"] = table["error"].abs()
    table["percentage_error"] = table["absolute_error"] / table["actual"].abs().replace(0, np.nan) * 100
    return table
