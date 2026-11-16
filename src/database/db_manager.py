"""Transactional SQLite forecast history, result references and aggregate snapshots."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import pandas as pd

from src.data.aggregator import normalize_frequency
from src.forecasting.common import validate_config, validate_prediction_table
from src.forecasting.types import ForecastRun, ModelConfig
from src.reports.report_generator import ReportContext
from src.utils.exceptions import DatabaseError, DataValidationError, SalesVisionError
from src.utils.serialization import (
    dataframe_from_payload,
    dataframe_payload,
    json_safe,
    strict_json,
)


def run_payload(run: ForecastRun) -> dict:
    return json_safe({
        field.name: dataframe_payload(getattr(run, field.name)) if isinstance(getattr(run, field.name), pd.DataFrame) else getattr(run, field.name)
        for field in fields(ForecastRun)
    })


def run_from_payload(payload: dict) -> ForecastRun:
    if not isinstance(payload, dict) or any(not isinstance(payload.get(key), dict) for key in ("metadata", "configuration", "evaluation_metrics")):
        raise ValueError("Invalid forecast snapshot metadata")
    values = dict(payload)
    for key in ("predictions", "components"):
        values[key] = dataframe_from_payload(values[key])
    run = ForecastRun(**values)
    configuration = dict(run.configuration)
    for key in ("order", "seasonal_order", "regressor_columns"):
        if key in configuration:
            configuration[key] = tuple(configuration[key])
    config = ModelConfig(**configuration)
    validate_config(config)
    if config.name != run.model or type(run.horizon) is not int or not 1 <= run.horizon <= 366:
        raise ValueError("Invalid forecast snapshot model or horizon")
    expected = pd.date_range(pd.Timestamp(run.training_end), periods=run.horizon + 1, freq=normalize_frequency(run.frequency))[1:].rename("date")
    validate_prediction_table(run.predictions, expected)
    return run


class DatabaseManager:
    """Use one short-lived connection per operation for Streamlit thread safety."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError("The forecast history database directory is not writable. Configure another SALESVISION_DATABASE_PATH.") from exc
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    horizon INTEGER NOT NULL CHECK(horizon > 0),
                    metrics_json TEXT NOT NULL,
                    result_reference TEXT NOT NULL,
                    run_json TEXT NOT NULL,
                    report_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_points (
                    run_id TEXT NOT NULL REFERENCES forecasts(run_id) ON DELETE CASCADE,
                    forecast_date TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    lower_value REAL NOT NULL,
                    upper_value REAL NOT NULL,
                    point_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, forecast_date)
                );
                CREATE INDEX IF NOT EXISTS idx_forecast_created ON forecasts(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_forecast_dataset ON forecasts(dataset_fingerprint);
            """)

    @contextmanager
    def _connect(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
            with connection:
                yield connection
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise DatabaseError("The forecast history database is unavailable or corrupted. Back up the existing file and configure a writable SALESVISION_DATABASE_PATH. Existing history was not replaced.") from exc
        finally:
            if connection is not None:
                connection.close()

    def save_forecast(self, run: ForecastRun, report: ReportContext) -> str:
        if run.dataset_fingerprint != report.forecast_metadata.get("dataset_fingerprint") or report.forecast_metadata.get("run_id") != run.run_id or strict_json(dataframe_payload(report.forecast)) != strict_json(dataframe_payload(run.predictions)):
            raise DataValidationError("The forecast and report snapshot do not match.")
        reference = f"sqlite:forecast_points/{run.run_id}"
        with self._connect() as connection:
            existing = connection.execute("SELECT run_json FROM forecasts WHERE run_id=?", (run.run_id,)).fetchone()
            serialized_run = strict_json(run_payload(run))
            if existing:
                if existing["run_json"] != serialized_run:
                    raise DataValidationError("A different forecast already uses this execution identifier.")
                connection.execute("UPDATE forecasts SET report_json=? WHERE run_id=?", (strict_json(report.to_payload()), run.run_id))
                return run.run_id
            connection.execute(
                "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run.run_id, run.created_at, run.dataset_fingerprint, strict_json(run.metadata), run.model, run.frequency, run.horizon, strict_json(run.evaluation_metrics), reference, serialized_run, strict_json(report.to_payload())),
            )
            for record in dataframe_payload(run.predictions)["records"]:
                connection.execute(
                    "INSERT INTO forecast_points VALUES (?, ?, ?, ?, ?, ?)",
                    (run.run_id, record["date"], record["prediction"], record["lower"], record["upper"], strict_json(record)),
                )
        return run.run_id

    def list_forecasts(self, dataset_fingerprint: str | None = None, limit: int = 200) -> pd.DataFrame:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise DataValidationError("History listing limit must be between 1 and 1000.")
        query = "SELECT * FROM forecasts"
        parameters = []
        if dataset_fingerprint:
            query += " WHERE dataset_fingerprint=?"
            parameters.append(dataset_fingerprint)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        records = []
        for row in rows:
            try:
                metadata, metrics = json.loads(row["metadata_json"]), json.loads(row["metrics_json"])
                if not isinstance(metadata, dict) or not isinstance(metrics, dict):
                    raise ValueError("Invalid history metadata")
            except (ValueError, TypeError) as exc:
                raise DatabaseError("A saved database record is corrupted. Back up the database before restoring its history.") from exc
            records.append({
                "Run ID": row["run_id"], "Executed (UTC)": row["created_at"],
                "Dataset": metadata.get("name", metadata.get("filename", "User dataset")),
                "Kind": metadata.get("kind", "user-provided"), "Model": row["model"],
                "Frequency": row["frequency"], "Horizon": row["horizon"],
                "RMSE": metrics.get("RMSE"), "MAE": metrics.get("MAE"),
                "Result Reference": row["result_reference"], "Fingerprint": row["dataset_fingerprint"],
            })
        return pd.DataFrame(records, columns=["Run ID", "Executed (UTC)", "Dataset", "Kind", "Model", "Frequency", "Horizon", "RMSE", "MAE", "Result Reference", "Fingerprint"])

    def _record(self, run_id: str):
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM forecasts WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise DataValidationError("The selected forecast execution no longer exists.")
        return row

    def get_forecast(self, run_id: str) -> ForecastRun:
        payload = self._record(run_id)["run_json"]
        try:
            return run_from_payload(json.loads(payload))
        except (ValueError, TypeError, KeyError, SalesVisionError) as exc:
            raise DatabaseError("The saved forecast database snapshot is corrupted. Restore it from a valid backup.") from exc

    def get_report(self, run_id: str) -> ReportContext:
        payload = self._record(run_id)["report_json"]
        try:
            return ReportContext.from_payload(json.loads(payload))
        except (ValueError, TypeError, KeyError) as exc:
            raise DatabaseError("The saved report database snapshot is corrupted. Restore it from a valid backup.") from exc

    def compare_forecasts(self, run_ids: list[str]) -> pd.DataFrame:
        if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
            raise DataValidationError("Select at least two forecast executions to compare.")
        runs = [self.get_forecast(run_id) for run_id in run_ids]
        if len({run.dataset_fingerprint for run in runs}) > 1 or len({run.frequency for run in runs}) > 1:
            raise DataValidationError("History comparison requires the same dataset, filters, preprocessing and frequency.")
        common_dates = set(runs[0].predictions["date"])
        for run in runs[1:]:
            common_dates &= set(run.predictions["date"])
        if not common_dates:
            raise DataValidationError("Selected forecasts have no overlapping future dates.")
        return pd.concat([
            run.predictions.loc[run.predictions["date"].isin(common_dates), ["date", "prediction"]].assign(execution=f"{run.model} · {run.run_id[:8]}")
            for run in runs
        ], ignore_index=True)

    def delete_forecast(self, run_id: str, confirmed: bool = False) -> bool:
        if confirmed is not True:
            raise DataValidationError("Confirm deletion before removing a forecast from history.")
        with self._connect() as connection:
            result = connection.execute("DELETE FROM forecasts WHERE run_id=?", (run_id,))
            return result.rowcount == 1
