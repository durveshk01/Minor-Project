"""Prespecified UCI experiments with genuine forecasts and traceable result artifacts."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from src.ai.gemini_client import GeminiClient
from src.ai.insight_generator import InsightResult, build_analytics_context, generate_insights
from src.data.loader import load_dataset
from src.data.pipeline import process_dataset
from src.data.schema import AggregationOptions, CleaningOptions, ColumnMapping, ProcessedDataset
from src.evaluation.backtesting import chronological_split
from src.evaluation.metrics import calculate_metrics
from src.evaluation.model_comparison import EvaluationResult, evaluate_models
from src.features.time_features import SeasonalResult, seasonal_decomposition, statistical_summary
from src.forecasting.forecast_service import generate_forecast
from src.forecasting.types import ForecastRun, ModelConfig
from src.reports.report_generator import (
    build_report_context,
    export_csv,
    generate_excel_report,
    generate_pdf_report,
)
from src.utils.exceptions import ForecastError
from src.utils.serialization import json_safe
from src.visualization.charts import (
    report_comparison_figure,
    report_decomposition_figure,
    report_evaluation_figure,
    report_history_figure,
)


@dataclass
class ExperimentResult:
    dataset: ProcessedDataset
    evaluation: EvaluationResult
    forecasts: dict[str, ForecastRun]
    seasonal: SeasonalResult
    baselines: pd.DataFrame
    insights: InsightResult
    manifest: dict
    output_directory: Path


def prepare_uci_dataset(root: Path = PROJECT_ROOT) -> ProcessedDataset:
    path = root / "data/sample/uci_online_retail_daily.csv"
    loaded = load_dataset(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    metadata.update({"filename": loaded.filename, "dataset_file_sha256": loaded.sha256})
    return process_dataset(
        loaded.frame,
        ColumnMapping("Date", "Sales"),
        CleaningOptions(invalid_rows="reject", duplicates="keep", negative_sales="keep"),
        AggregationOptions(frequency="D", missing_periods="ffill", source_frequency="D"),
        metadata=metadata,
    )


def experiment_configurations() -> list[ModelConfig]:
    return [
        ModelConfig("ARIMA", order=(1, 1, 1), maxiter=150, seed=42),
        ModelConfig("SARIMA", order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), maxiter=150, seed=42),
        ModelConfig("Prophet", yearly_seasonality=False, weekly_seasonality=True, seed=42),
    ]


def benchmark_baselines(dataset: ProcessedDataset, test_size: int) -> pd.DataFrame:
    train, test = chronological_split(dataset.series, test_size)
    actual = dataset.observed_series.reindex(test.index)
    predictions = {
        "Last-value naive": np.repeat(train.iloc[-1], len(test)),
        "Seven-day seasonal naive": np.resize(train.iloc[-7:].to_numpy(), len(test)),
    }
    observed = actual.notna().to_numpy()
    return pd.DataFrame([
        {"Baseline": name, **calculate_metrics(actual.iloc[observed], values[observed])}
        for name, values in predictions.items()
    ])


def experiment_findings(
    dataset: ProcessedDataset,
    evaluation: EvaluationResult,
    baselines: pd.DataFrame,
    forecasts: dict[str, ForecastRun],
) -> str:
    summary = statistical_summary(dataset.observed_series)
    lines = [
        "# Executed UCI Online Retail experiment",
        "",
        f"Real dataset: {dataset.metadata['name']} (CC BY 4.0; GBP net revenue).",
        f"History: {summary['start'][:10]} to {summary['end'][:10]}; "
        f"{summary['observations']} observed dates in {summary['calendar_periods']} calendar days.",
        f"{summary['missing_periods']} unknown days use explicit causal forward-fill for model inputs only. "
        "Observed historical totals and scored targets retain the original unknowns.",
        f"Observed net revenue: GBP {summary['total']:,.2f}.",
        "",
        "## Prespecified evaluation",
        "",
        f"Training: {evaluation.train_size} calendar days, ending {evaluation.training_end[:10]}.",
        f"Common holdout: {evaluation.test_size} calendar days, "
        f"{evaluation.test_start[:10]} to {evaluation.test_end[:10]}.",
        "No random split or holdout-directed parameter search was used. "
        "The winner is selected among ARIMA, SARIMA and Prophet by lowest valid holdout RMSE.",
        "",
        "| Model | MAE (GBP) | RMSE (GBP) | MAPE (%) | sMAPE (%) | Bias (GBP) | Scored dates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation.comparison.to_dict("records"):
        if row["Status"] == "success":
            lines.append(f"| {row['Model']} | {row['MAE']:,.3f} | {row['RMSE']:,.3f} | "
                         f"{row['MAPE']:.3f} | {row['sMAPE']:.3f} | {row['Bias']:,.3f} | "
                         f"{int(row['Scored Observations'])} |")
        else:
            lines.append(f"| {row['Model']} (failed) | — | — | — | — | — | — |")
    best = evaluation.best_model
    if best:
        metrics = evaluation.metrics_for(best)
        lines.extend(["", f"Selected model: **{best}**, holdout RMSE **GBP {metrics['RMSE']:,.3f}**."])
        if best in forecasts:
            run = forecasts[best]
            lines.append(f"Final {best} is refitted on all {run.training_observations} calendar days. "
                         f"Its {run.horizon}-day forecast spans "
                         f"{run.predictions['date'].min().date()} to {run.predictions['date'].max().date()} "
                         f"and has a raw point-forecast total of GBP {run.predictions['prediction'].sum():,.2f}.")
    lines.extend(["", "## Baseline benchmarks", "", "| Baseline | MAE (GBP) | RMSE (GBP) |", "|---|---:|---:|"])
    for row in baselines.to_dict("records"):
        lines.append(f"| {row['Baseline']} | {row['MAE']:,.3f} | {row['RMSE']:,.3f} |")
    if best:
        baseline = baselines.sort_values("RMSE").iloc[0]
        outcome = "beats" if evaluation.metrics_for(best)["RMSE"] < baseline["RMSE"] else "does not beat"
        lines.extend(["", f"The selected model **{outcome}** the stronger naive baseline on this holdout. "
                      "Baselines are benchmarks, not candidates in the application's three-model selection."])
    if evaluation.backtest is not None:
        lines.extend(["", "## Training-partition rolling-origin results", "",
                      "These separate earlier windows do not replace the final holdout selection.", "",
                      "| Model | RMSE (GBP) | Completed folds | Status |", "|---|---:|---:|---|"])
        for row in evaluation.backtest.comparison.to_dict("records"):
            score = f"{row['RMSE']:,.3f}" if pd.notna(row.get("RMSE")) else "—"
            lines.append(f"| {row['Model']} | {score} | {row['Completed Folds']} | {row['Status']} |")
    lines.extend([
        "", "## Interpretation and limits", "",
        "- This archived dataset represents one UK online retailer; these are not Walmart findings.",
        "- Missing dates are unknown; causal forward-fill is a documented modelling assumption. "
        "STL diagnostics also use that filled series and are descriptive, not causal.",
        "- Source returns/cancellations and exact duplicates are retained, so the target is recorded net revenue.",
        "- The final forecast is future relative to the dataset's last date, not a forecast for today's calendar.",
        "- About one year of history does not identify yearly Prophet seasonality; it is explicitly disabled.",
        "- MAPE can be large for small or negative net-sales days; zero targets are excluded only from MAPE.",
        "- Parameters are fixed examples, not proof of globally optimal models; one holdout does not establish general superiority.",
        "- Model-based period intervals are not guaranteed coverage or a joint interval for total revenue.",
        "- Revenue predictions alone cannot determine stock quantities, prices, product mix or lead times.",
        "- No future labelled targets exist for the final forecast; no future accuracy claim is made.",
        "", "Source: https://archive.ics.uci.edu/dataset/352/online+retail", "",
        "Citation: Chen, D. (2015). Online Retail. UCI Machine Learning Repository. DOI: 10.24432/C5BW33.",
    ])
    return "\n".join(lines) + "\n"


def export_experiment(
    dataset: ProcessedDataset,
    evaluation: EvaluationResult,
    forecasts: dict[str, ForecastRun],
    seasonal: SeasonalResult,
    baselines: pd.DataFrame,
    insights: InsightResult,
    output_directory: Path,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def save(name: str, content: bytes) -> None:
        (output_directory / name).write_bytes(content)
        artifacts[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def save_json(name: str, document) -> None:
        save(name, json.dumps(json_safe(document), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    history = pd.DataFrame({"observed_sales": dataset.observed_series, "model_input": dataset.series, "is_observed": dataset.observed_mask}).rename_axis("date").reset_index()
    save("processed_calendar.csv", export_csv(history))
    save("cleaning_audit.csv", export_csv(pd.DataFrame(dataset.operations)))
    save_json("dataset_metadata.json", dataset.metadata)
    save_json("statistics.json", statistical_summary(dataset.observed_series))
    save_json("quality.json", asdict(dataset.quality))
    save("seasonal_components.csv", export_csv(seasonal.components.rename_axis("date").reset_index()))
    save_json("seasonal_diagnostics.json", {"period": seasonal.period, "strength": seasonal.strength, "amplitude": seasonal.amplitude, "residual_std": seasonal.residual_std, "explanation": seasonal.explanation})
    save("model_comparison.csv", export_csv(evaluation.comparison))
    save("baseline_comparison.csv", export_csv(baselines))
    save_json("model_configurations.json", {name: config.to_dict() for name, config in evaluation.configurations.items()})
    for name, table in evaluation.predictions.items():
        save(f"{name.lower()}_holdout.csv", export_csv(table))
        save(f"{name.lower()}_holdout_summary.txt", evaluation.summaries[name].encode("utf-8"))
    if evaluation.backtest is not None:
        save("backtest_comparison.csv", export_csv(evaluation.backtest.comparison))
        save("backtest_folds.csv", export_csv(evaluation.backtest.folds))
        for name, table in evaluation.backtest.predictions.items():
            save(f"{name.lower()}_backtest.csv", export_csv(table))
    for name, run in forecasts.items():
        save(f"{name.lower()}_future_forecast.csv", export_csv(run.predictions))
        save(f"{name.lower()}_full_history_summary.txt", run.model_summary.encode("utf-8"))
    save_json("insights.json", asdict(insights))
    best = evaluation.best_model
    report = build_report_context(dataset, evaluation, forecasts.get(best), seasonal, insights)
    save_json("report_snapshot.json", report.to_payload())
    save("salesvision_report.xlsx", generate_excel_report(report))
    save("salesvision_report.pdf", generate_pdf_report(report))
    save("historical_sales.png", report_history_figure(dataset.observed_series))
    save("seasonal_decomposition.png", report_decomposition_figure(seasonal.components))
    save("model_comparison.png", report_comparison_figure(evaluation.comparison))
    save("holdout_predictions.png", report_evaluation_figure(evaluation.predictions))
    if best in forecasts:
        save("selected_forecast.png", report_history_figure(dataset.observed_series, forecasts[best].predictions))
    save("results.md", experiment_findings(dataset, evaluation, baselines, forecasts).encode("utf-8"))
    manifest = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "status": "success" if len(forecasts) == 3 and evaluation.comparison["Status"].eq("success").all() else "partial",
        "experiment": "UCI daily net revenue; prespecified models, common chronological holdout",
        "dataset_fingerprint": dataset.fingerprint,
        "data_provenance": dataset.metadata,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "prophet", "cmdstanpy", "google-genai")},
        "seed": 42,
        "train_size": evaluation.train_size,
        "holdout_size": evaluation.test_size,
        "training_end": evaluation.training_end,
        "holdout_start": evaluation.test_start,
        "holdout_end": evaluation.test_end,
        "ranking_metric": evaluation.ranking_metric,
        "best_model": best,
        "model_comparison": evaluation.comparison.to_dict("records"),
        "baselines": baselines.to_dict("records"),
        "model_configurations": {name: config.to_dict() for name, config in evaluation.configurations.items()},
        "model_warnings": evaluation.warnings,
        "future_forecasts": {name: {"run_id": run.run_id, "horizon": run.horizon, "training_observations": run.training_observations, "training_seconds": run.training_seconds, "warnings": run.warnings} for name, run in forecasts.items()},
        "insights_source": insights.source,
        "insights_status": insights.status,
        "artifacts": artifacts,
    }
    (output_directory / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest


def run_uci_experiment(
    output_directory: Path | None = None,
    holdout: int = 30,
    horizon: int = 30,
    backtest_folds: int = 3,
    backtest_horizon: int = 14,
    use_gemini: bool = False,
) -> ExperimentResult:
    output_directory = output_directory or PROJECT_ROOT / "outputs/experiments/uci_daily"
    dataset = prepare_uci_dataset()
    seasonal = seasonal_decomposition(dataset.series, 7)
    evaluation = evaluate_models(dataset, experiment_configurations(), holdout, "RMSE", backtest_folds, backtest_horizon)
    if not evaluation.comparison["Status"].eq("success").all():
        errors = evaluation.comparison.loc[evaluation.comparison["Status"].ne("success"), ["Model", "Error"]].to_dict("records")
        raise ForecastError(f"The real experiment did not complete all three models: {errors}")
    forecasts = {
        name: generate_forecast(dataset, config, horizon, evaluation.metrics_for(name))
        for name, config in evaluation.configurations.items()
    }
    baselines = benchmark_baselines(dataset, holdout)
    context = build_analytics_context(dataset, evaluation, forecasts[evaluation.best_model], seasonal)
    with GeminiClient(api_key=None if use_gemini else "") as client:
        insights = generate_insights(context, client=client)
    manifest = export_experiment(dataset, evaluation, forecasts, seasonal, baselines, insights, output_directory)
    return ExperimentResult(dataset, evaluation, forecasts, seasonal, baselines, insights, manifest, output_directory)
