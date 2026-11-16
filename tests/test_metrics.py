"""Known-value metric tests, zero-target behavior and alignment protection."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_metrics
from src.evaluation.model_comparison import select_best_model
from src.utils.exceptions import DataValidationError


def test_metrics_match_hand_calculated_errors():
    result = calculate_metrics([100, 200, 300], [110, 180, 330])
    assert result["MAE"] == 20
    assert result["RMSE"] == pytest.approx(np.sqrt(1400 / 3))
    assert result["MAPE"] == pytest.approx(10)
    assert result["Bias"] == pytest.approx(20 / 3)
    expected_symmetric = np.mean([2000 / 210, 4000 / 380, 6000 / 630])
    assert result["sMAPE"] == pytest.approx(expected_symmetric)


def test_zero_actuals_excluded_only_from_mape():
    result = calculate_metrics([0, 0, 10], [0, 5, 15])
    assert result["MAE"] == pytest.approx(10 / 3)
    assert result["MAPE"] == 50
    assert result["sMAPE"] == pytest.approx(80)
    assert result["MAPE Observations"] == 1
    assert result["Scored Observations"] == 3


def test_all_zero_actuals_have_no_mape():
    result = calculate_metrics([0, 0], [0, 0])
    assert result["MAPE"] is None
    assert result["sMAPE"] == 0
    assert result["RMSE"] == 0


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([1, np.nan], [1, 2]), ([1], [np.inf])])
def test_invalid_metric_inputs_are_not_silently_scored(actual, predicted):
    with pytest.raises(DataValidationError):
        calculate_metrics(actual, predicted)


def test_metric_alignment_protection():
    actual = pd.Series([1, 2], index=[1, 2])
    prediction = pd.Series([1, 2], index=[2, 1])
    with pytest.raises(DataValidationError, match="identical date"):
        calculate_metrics(actual, prediction)


def test_model_winner_depends_on_actual_metric_and_failure_status():
    table = pd.DataFrame({"Model": ["ARIMA", "SARIMA", "Prophet"], "RMSE": [4, 2, 1], "MAE": [1, 2, 3], "Status": ["success", "success", "failed"]})
    assert select_best_model(table, "RMSE") == "SARIMA"
    assert select_best_model(table, "MAE") == "ARIMA"
