"""Consistent model configuration, fitted artifacts and forecast result contracts."""

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    name: str
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 7)
    trend: str | None = None
    interval_width: float = 0.95
    maxiter: int = 150
    yearly_seasonality: bool | None = None
    weekly_seasonality: bool | None = None
    daily_seasonality: bool = False
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    use_holidays: bool = False
    regressor_columns: tuple[str, ...] = ()
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FittedModel:
    name: str
    config: ModelConfig
    estimator: Any
    history: pd.Series
    training_seconds: float
    summary: str
    warnings: list[str] = field(default_factory=list)
    scale: float = 1.0
    regressor_mean: pd.Series | None = None
    regressor_scale: pd.Series | None = None
    components: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ForecastRun:
    run_id: str
    created_at: str
    dataset_fingerprint: str
    metadata: dict
    model: str
    configuration: dict
    frequency: str
    horizon: int
    predictions: pd.DataFrame
    training_seconds: float
    training_observations: int
    training_start: str
    training_end: str
    evaluation_metrics: dict
    model_summary: str
    warnings: list[str]
    components: pd.DataFrame = field(default_factory=pd.DataFrame)
    business_constraint: str = "none"
