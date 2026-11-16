"""Transactional history, snapshots, comparisons and confirmed deletion."""

import sqlite3

import pandas as pd
import pytest

from src.data.pipeline import process_dataset
from src.data.schema import ColumnMapping
from src.database.db_manager import DatabaseManager
from src.forecasting.forecast_service import generate_forecast
from src.forecasting.types import ModelConfig
from src.reports.report_generator import build_report_context
from src.utils.exceptions import DataValidationError


def test_database_save_view_idempotency_compare_and_delete(tmp_path, daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), metadata={"name": "Synthetic test", "kind": "synthetic"})
    manager = DatabaseManager(tmp_path / "history.sqlite3")
    first = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 7)
    report = build_report_context(dataset, forecast=first)
    manager.save_forecast(first, report)
    manager.save_forecast(first, report)
    assert len(manager.list_forecasts()) == 1
    restored = manager.get_forecast(first.run_id)
    pd.testing.assert_frame_equal(restored.predictions, first.predictions, check_exact=False, rtol=1e-9)
    assert len(manager.get_report(first.run_id).history) == len(dataset.series)
    second = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 4)
    manager.save_forecast(second, build_report_context(dataset, forecast=second))
    comparison = manager.compare_forecasts([first.run_id, second.run_id])
    assert len(comparison) == 8
    with pytest.raises(DataValidationError, match="Confirm"):
        manager.delete_forecast(first.run_id)
    assert manager.delete_forecast(first.run_id, confirmed=True)
    assert len(manager.list_forecasts()) == 1
    with sqlite3.connect(manager.path) as connection:
        assert connection.execute("SELECT count(*) FROM forecast_points WHERE run_id=?", (first.run_id,)).fetchone()[0] == 0


def test_database_excludes_api_key_fields(tmp_path, daily_frame):
    dataset = process_dataset(daily_frame, ColumnMapping("Date", "Sales"), metadata={"name": "QA", "GEMINI_API_KEY": "test-secret-never-store"})
    run = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 4)
    manager = DatabaseManager(tmp_path / "safe.sqlite3")
    manager.save_forecast(run, build_report_context(dataset, forecast=run))
    assert "test-secret-never-store" not in manager.path.read_bytes().decode("latin1")
    assert "GEMINI_API_KEY" not in manager.get_forecast(run.run_id).metadata


def test_incompatible_history_comparisons_are_rejected(tmp_path, daily_frame):
    manager = DatabaseManager(tmp_path / "comparison.sqlite3")
    ids = []
    for multiplier in (1, 2):
        frame = daily_frame.copy()
        frame["Sales"] *= multiplier
        dataset = process_dataset(frame, ColumnMapping("Date", "Sales"))
        run = generate_forecast(dataset, ModelConfig("ARIMA", order=(0, 1, 0)), 3)
        ids.append(run.run_id)
        manager.save_forecast(run, build_report_context(dataset, forecast=run))
    with pytest.raises(DataValidationError, match="same dataset"):
        manager.compare_forecasts(ids)
    with pytest.raises(DataValidationError, match="no longer exists"):
        manager.get_forecast("unknown-id")
