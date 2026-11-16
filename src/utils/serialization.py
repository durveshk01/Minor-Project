"""Strict JSON serialization for aggregate snapshots without secret fields."""

import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

SENSITIVE_KEYS = {"apikey", "geminiapikey", "password", "secret", "accesstoken", "refreshtoken", "credentials"}


def json_safe(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
            if str(key).lower().replace("_", "").replace("-", "") not in SENSITIVE_KEYS
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def strict_json(value) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, sort_keys=True)


def dataframe_payload(frame: pd.DataFrame) -> dict:
    return {
        "columns": list(frame.columns),
        "records": json_safe(frame.to_dict(orient="records")),
        "datetime_columns": [column for column in frame if pd.api.types.is_datetime64_any_dtype(frame[column])],
    }


def dataframe_from_payload(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or not isinstance(payload.get("columns"), list) or not isinstance(payload.get("records"), list) or any(not isinstance(record, dict) for record in payload["records"]):
        raise ValueError("Invalid dataframe snapshot shape")
    if not isinstance(payload.get("datetime_columns", []), list):
        raise ValueError("Invalid datetime snapshot metadata")
    frame = pd.DataFrame(payload["records"], columns=payload["columns"])
    for column in payload.get("datetime_columns", []):
        frame[column] = pd.to_datetime(frame[column], format="mixed", errors="raise")
    return frame


def series_payload(series: pd.Series) -> dict:
    return {"frequency": series.index.freqstr, "data": dataframe_payload(series.rename("sales").rename_axis("date").reset_index())}


def series_from_payload(payload: dict) -> pd.Series:
    frame = dataframe_from_payload(payload["data"])
    result = frame.set_index("date")["sales"].astype(float)
    return result.asfreq(payload["frequency"]) if payload["frequency"] else result
