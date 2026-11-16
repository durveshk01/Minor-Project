"""Actual-output CSV, Excel and illustrated PDF reports with reproducible snapshots."""

import io
import json
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    LongTable,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

from src.ai.insight_generator import InsightResult, build_analytics_context, context_fingerprint
from src.data.schema import ProcessedDataset
from src.evaluation.model_comparison import EvaluationResult
from src.features.time_features import SeasonalResult, monthly_comparison, statistical_summary
from src.forecasting.types import ForecastRun
from src.utils.exceptions import ReportError
from src.utils.serialization import (
    dataframe_from_payload,
    dataframe_payload,
    json_safe,
    series_from_payload,
    series_payload,
)
from src.visualization.charts import (
    figure_png,
    report_comparison_figure,
    report_decomposition_figure,
    report_evaluation_figure,
    report_history_figure,
    statistical_figure,
)


@dataclass
class ReportContext:
    generated_at: str
    metadata: dict
    history: pd.Series
    quality: dict
    operations: list[dict]
    rejected_rows: int
    statistics: dict
    seasonal_components: pd.DataFrame = field(default_factory=pd.DataFrame)
    seasonal_explanation: str = "Seasonal decomposition has not been run for this report."
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    model_summaries: dict[str, str] = field(default_factory=dict)
    model_configurations: dict[str, dict] = field(default_factory=dict)
    selected_model: str = ""
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_metadata: dict = field(default_factory=dict)
    insights_text: str = ""
    insights_source: str = ""

    def to_payload(self) -> dict:
        return json_safe({
            "generated_at": self.generated_at, "metadata": self.metadata,
            "history": series_payload(self.history), "quality": self.quality,
            "operations": self.operations, "rejected_rows": self.rejected_rows,
            "statistics": self.statistics,
            "seasonal_components": dataframe_payload(self.seasonal_components),
            "seasonal_explanation": self.seasonal_explanation,
            "comparison": dataframe_payload(self.comparison),
            "holdout_predictions": {name: dataframe_payload(frame) for name, frame in self.holdout_predictions.items()},
            "model_summaries": self.model_summaries, "model_configurations": self.model_configurations,
            "selected_model": self.selected_model, "forecast": dataframe_payload(self.forecast),
            "forecast_metadata": self.forecast_metadata,
            "insights_text": self.insights_text, "insights_source": self.insights_source,
        })

    @classmethod
    def from_payload(cls, payload: dict):
        dictionaries = ("metadata", "quality", "statistics", "holdout_predictions", "model_summaries", "model_configurations", "forecast_metadata")
        if not isinstance(payload, dict) or any(not isinstance(payload.get(key), dict) for key in dictionaries):
            raise ValueError("Invalid report snapshot metadata")
        if not isinstance(payload.get("operations"), list) or any(not isinstance(operation, dict) for operation in payload["operations"]):
            raise ValueError("Invalid report snapshot operations")
        values = dict(payload)
        values["history"] = series_from_payload(payload["history"])
        for key in ("seasonal_components", "comparison", "forecast"):
            values[key] = dataframe_from_payload(payload[key])
        values["holdout_predictions"] = {name: dataframe_from_payload(frame) for name, frame in payload["holdout_predictions"].items()}
        return cls(**values)


def build_report_context(
    dataset: ProcessedDataset,
    evaluation: EvaluationResult | None = None,
    forecast: ForecastRun | None = None,
    seasonal: SeasonalResult | None = None,
    insights: InsightResult | None = None,
) -> ReportContext:
    if evaluation is not None and evaluation.dataset_fingerprint != dataset.fingerprint:
        raise ReportError("Evaluation results belong to a different dataset configuration.")
    if forecast is not None and forecast.dataset_fingerprint != dataset.fingerprint:
        raise ReportError("Forecast results belong to a different dataset configuration.")
    context = ReportContext(
        generated_at=datetime.now(timezone.utc).isoformat(),
        metadata=json_safe(dataset.metadata), history=dataset.observed_series.copy(),
        quality=asdict(dataset.quality), operations=list(dataset.operations),
        rejected_rows=len(dataset.rejected_records), statistics=statistical_summary(dataset.observed_series),
    )
    if seasonal is not None:
        if not seasonal.components.index.equals(dataset.series.index):
            raise ReportError("Seasonal results do not match the current dataset calendar.")
        context.seasonal_components = seasonal.components.rename_axis("date").reset_index()
        context.seasonal_explanation = seasonal.explanation
    if evaluation is not None:
        context.comparison = evaluation.comparison.copy()
        context.holdout_predictions = {name: frame.copy() for name, frame in evaluation.predictions.items()}
        context.model_summaries = dict(evaluation.summaries)
        context.model_configurations = {name: config.to_dict() for name, config in evaluation.configurations.items()}
        context.metadata["evaluation"] = {
            "ranking_metric": evaluation.ranking_metric, "best_model": evaluation.best_model,
            "training_end": evaluation.training_end, "test_start": evaluation.test_start, "test_end": evaluation.test_end,
            "train_size": evaluation.train_size, "test_size": evaluation.test_size,
        }
        if evaluation.backtest is not None:
            context.metadata["backtesting"] = dataframe_payload(evaluation.backtest.comparison)
    if forecast is not None:
        context.selected_model = forecast.model
        context.forecast = forecast.predictions.copy()
        context.forecast_metadata = {
            "run_id": forecast.run_id, "created_at": forecast.created_at,
            "dataset_fingerprint": forecast.dataset_fingerprint,
            "frequency": forecast.frequency, "horizon": forecast.horizon,
            "configuration": forecast.configuration,
            "training_observations": forecast.training_observations,
            "training_start": forecast.training_start, "training_end": forecast.training_end,
            "training_seconds": forecast.training_seconds,
            "evaluation_metrics": forecast.evaluation_metrics,
            "business_constraint": forecast.business_constraint, "warnings": forecast.warnings,
        }
        context.model_summaries[f"{forecast.model} (full history)"] = forecast.model_summary
    if insights is not None:
        current = context_fingerprint(build_analytics_context(dataset, evaluation, forecast, seasonal))
        if insights.context_hash == current:
            context.insights_text, context.insights_source = insights.text, insights.source
    return context


def spreadsheet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    def safe_text(value):
        if isinstance(value, str) and (value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))):
            return "'" + value
        return value

    result = frame.copy()
    result.columns = [safe_text(column) for column in result.columns]
    for column in result:
        if pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column]):
            result[column] = result[column].map(safe_text)
    return result


def export_csv(frame: pd.DataFrame) -> bytes:
    return spreadsheet_safe(frame).to_csv(index=False).encode("utf-8-sig")


def export_excel(frame: pd.DataFrame, sheet_name: str = "Sales") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        spreadsheet_safe(frame).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


def key_value_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame({"Item": list(values), "Value": [json.dumps(json_safe(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for value in values.values()]})


def report_target_definition(context: ReportContext) -> str:
    """Describe the aggregation target consistently across PDF and Excel exports."""
    if context.metadata.get("aggregation") == "mean":
        return (
            "Each target is the mean of retained record sales within its calendar period. "
            "Sums of these period averages are not total transaction revenue. "
            "Monthly comparisons sum observed period averages, and forecasts predict the same mean-aggregated target. "
            "Unknown observations are not treated as zeros."
        )
    return "Each target sums retained record sales within its calendar period. Unknown observations are not treated as zeros."


def report_statistics(context: ReportContext) -> dict:
    """Return statistic labels appropriate to the selected aggregation target."""
    return {
        "sum_of_period_averages" if key == "total" and context.metadata.get("aggregation") == "mean" else key: value
        for key, value in context.statistics.items()
    }


def report_monthly_comparison(context: ReportContext) -> pd.DataFrame:
    """Label sums of monthly period targets according to their aggregation meaning."""
    frame = monthly_comparison(context.history)
    return frame.rename(columns={"sales": "sum_of_period_averages"}) if context.metadata.get("aggregation") == "mean" else frame


def generate_excel_report(context: ReportContext) -> bytes:
    try:
        sheets = {
            "Dataset": key_value_frame({**context.metadata, "target_definition": report_target_definition(context)}),
            "Historical": context.history.rename("sales").rename_axis("date").reset_index(),
            "Data Quality": key_value_frame(context.quality),
            "Cleaning Audit": pd.DataFrame(context.operations),
            "Statistics": key_value_frame(report_statistics(context)),
            "Monthly Comparison": report_monthly_comparison(context),
        }
        if not context.seasonal_components.empty:
            sheets["Seasonal Components"] = context.seasonal_components
        if not context.comparison.empty:
            sheets["Model Comparison"] = context.comparison
        for name, table in context.holdout_predictions.items():
            sheets[f"{name} Holdout"] = table
        if not context.forecast.empty:
            sheets["Forecast"] = context.forecast
            sheets["Forecast Metadata"] = key_value_frame(context.forecast_metadata)
        sheets["Model Parameters"] = key_value_frame(context.model_configurations)
        sheets["Model Summaries"] = key_value_frame(context.model_summaries)
        if context.insights_text:
            sheets["AI Insights"] = pd.DataFrame({"Source": [context.insights_source], "Explanation": [context.insights_text]})
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                spreadsheet_safe(frame).to_excel(writer, sheet_name=name[:31], index=False)
                worksheet = writer.sheets[name[:31]]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions
                for column in worksheet.columns:
                    letter = column[0].column_letter
                    worksheet.column_dimensions[letter].width = min(65, max(15, max(len(str(cell.value or "")) for cell in column[:100]) + 2))
        return buffer.getvalue()
    except Exception as exc:
        raise ReportError("The Excel report could not be generated from the current outputs.") from exc


def readable(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def generate_pdf_report(context: ReportContext) -> bytes:
    try:
        buffer = io.BytesIO()
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="SmallTable", fontName="Helvetica", fontSize=8, leading=11))
        styles["Title"].textColor = colors.HexColor("#172B49")
        styles["Heading1"].textColor = colors.HexColor("#2563EB")
        styles["BodyText"].spaceAfter = 7
        story = []

        def paragraph(text: str, style: str = "BodyText"):
            story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style]))

        def heading(text: str):
            story.append(Spacer(1, 10))
            paragraph(text, "Heading1")

        def table(frame: pd.DataFrame, widths=None):
            if frame.empty:
                paragraph("No results were generated for this section.")
                return
            content = [[Paragraph(escape(str(column)), styles["SmallTable"]) for column in frame.columns]]
            content.extend([[Paragraph(escape(readable(value)), styles["SmallTable"]) for value in row] for row in frame.itertuples(index=False, name=None)])
            item = LongTable(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
            item.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EDFA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#A7BDE0")),
            ]))
            story.append(item)

        def image(content: bytes, height: float = 2.0):
            story.append(Image(io.BytesIO(content), width=6.6 * inch, height=height * inch))

        paragraph("SalesVision", "Title")
        paragraph("Sales Forecasting Using Python", "Heading2")
        paragraph("Intelligent Sales Forecasting & Business Analytics")
        paragraph(f"Generated (UTC): {context.generated_at}")
        heading("1. Project introduction")
        paragraph("SalesVision processes historical sales, investigates trend and seasonality, evaluates ARIMA, SARIMAX and Prophet on a common chronological holdout, retrains a selected model on all available history, and optionally asks Gemini to explain the computed evidence. Domain: Business Analytics; industry use: retail, FMCG and manufacturing.")
        heading("2. Dataset description and source")
        details = {key: context.metadata.get(key, "Not supplied") for key in ["name", "kind", "source_url", "citation", "license", "currency", "input_rows", "cleaned_rows", "frequency", "aggregation"]}
        table(key_value_frame(details), [1.5 * inch, 5.1 * inch])
        paragraph(report_target_definition(context))
        if context.metadata.get("kind") == "synthetic":
            paragraph("SYNTHETIC TEST DATA: these results validate software functionality and are not empirical Walmart or real-industry findings.")
        if context.metadata.get("derivation"):
            paragraph(context.metadata["derivation"])
        heading("3. Data quality and preprocessing summary")
        paragraph(f"Explicitly excluded records: {context.rejected_rows}. Missing observations and genuine zero sales are distinguished. Duplicates and negative sales follow selected user policies. Incomplete periods remain unknown. Outliers are flagged diagnostically and never clipped by full-series rules.")
        table(pd.DataFrame(context.operations), [1.7 * inch, 0.8 * inch, 4.1 * inch])
        heading("4. Exploratory data analysis")
        table(key_value_frame(report_statistics(context)), [2.0 * inch, 4.6 * inch])
        image(report_history_figure(context.history))
        image(figure_png(statistical_figure(context.history)), 2.1)
        monthly_label = "sum_of_period_averages" if context.metadata.get("aggregation") == "mean" else "sales"
        monthly_description = "Monthly sums of observed period averages below are not monthly transaction-revenue totals." if monthly_label == "sum_of_period_averages" else "Monthly totals below are derived from observed periods."
        paragraph(f"{monthly_description} Boundary months and months containing missing observations may be incomplete. Weekly observations remain weekly and do not imply daily behavior.")
        table(report_monthly_comparison(context).tail(24)[["date", monthly_label]], [2.0 * inch, 4.6 * inch])
        heading("5. Seasonal analysis")
        paragraph(context.seasonal_explanation)
        if not context.seasonal_components.empty:
            image(report_decomposition_figure(context.seasonal_components.set_index("date")), 4.0)
        heading("6. Model methodology")
        paragraph("All models use the same time-ordered holdout. There is no random time-series split. Missing targets are not scored. Statistical fitting uses training-only sales scaling and, when configured, training-only regressor scaling. Future covariates must be explicitly known and supplied. Any optimizer retry is recorded. Prophet enables only identifiable seasonalities and uses the official package, not generated numerical predictions.")
        for model, methodology in [
            ("ARIMA", "ARIMA(p,d,q) models autoregressive dependence and moving-average errors after d differences using Statsmodels ARIMA."),
            ("SARIMA", "SARIMAX combines non-seasonal (p,d,q) and seasonal (P,D,Q,s) structure, with optional known external regressors. At least two seasonal cycles are required when seasonal terms are active."),
            ("Prophet", "Prophet models piecewise trend, appropriate additive or multiplicative seasonality, and supplied holiday effects. Intraday seasonality is disabled for aggregated sales; yearly terms require at least two years."),
        ]:
            heading(f"7. {model} results and configuration")
            paragraph(methodology)
            if model in context.model_configurations:
                paragraph(json.dumps(context.model_configurations[model], default=str))
            if not context.comparison.empty:
                row = context.comparison.loc[context.comparison["Model"].eq(model)]
                if not row.empty:
                    table(row[["Model", "MAE", "RMSE", "MAPE", "sMAPE", "Status"]], [0.8 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.8 * inch])
                    if row["Status"].iloc[0] == "failed":
                        paragraph(f"Recorded failure: {row['Error'].iloc[0]}")
                else:
                    paragraph("This model was not evaluated in the current report.")
            else:
                paragraph("This model was not evaluated in the current report.")
        heading("8. Evaluation and model comparison")
        paragraph("MAE = mean absolute error; RMSE = square root of mean squared error; MAPE excludes zero actuals and is unavailable if none remain; sMAPE uses the symmetric absolute denominator, with both-zero pairs contributing zero; bias = mean(prediction - actual), positive for over-forecasting.")
        if not context.comparison.empty:
            metric = context.metadata["evaluation"]["ranking_metric"]
            best = context.metadata["evaluation"]["best_model"]
            paragraph(f"Ranking metric: lowest {metric}. Best successful holdout model: {best or 'unavailable'}. Training end: {context.metadata['evaluation']['training_end']}; holdout: {context.metadata['evaluation']['test_start']} to {context.metadata['evaluation']['test_end']}.")
            table(context.comparison[["Model", "MAE", "RMSE", "MAPE", "sMAPE", "Training Time"]], [0.8 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.8 * inch])
            image(report_comparison_figure(context.comparison))
            image(report_evaluation_figure(context.holdout_predictions))
        heading("9. Selected model and future sales forecast")
        if not context.forecast.empty:
            paragraph(f"Final selected model: {context.selected_model}. Full-history retraining observations: {context.forecast_metadata['training_observations']}. Future periods: {len(context.forecast)}. All point predictions and period intervals originate from the implemented forecasting model.")
            paragraph("Raw predictions are retained. Any business-constrained columns are explicitly named separately. Clipped intervals do not preserve calibrated coverage. Period interval endpoints are not summed into a joint total-sales confidence interval.")
            for warning in context.forecast_metadata.get("warnings", []):
                paragraph(warning)
            image(report_history_figure(context.history, context.forecast))
            table(context.forecast[["date", "prediction", "lower", "upper"]], [1.6 * inch, 1.6 * inch, 1.7 * inch, 1.7 * inch])
        else:
            paragraph("A final future forecast has not been generated for this report.")
        heading("10. Business explanations and Gemini insights")
        if context.insights_text:
            paragraph(f"Explanation source: {context.insights_source}. AI text is not a numerical forecasting model.")
            paragraph(context.insights_text)
        else:
            paragraph("No current Gemini explanation is available. Core historical analytics, evaluation and forecasting remain fully independent of Gemini.")
        heading("11. Conclusion and limitations")
        paragraph("The tables and charts in this report are generated exclusively from the current application outputs. Predictive performance is limited to the displayed holdout, dataset scope and selected parameters. Uncertainty intervals are model-based and not a guarantee. Changes in price, product mix, promotions and market conditions can invalidate patterns. Revenue forecasts require additional inventory-unit and lead-time data for operational stock decisions.")
        heading("Appendix. Recorded fitted-model summaries")
        for name, summary in context.model_summaries.items():
            paragraph(name, "Heading2")
            wrapped = "\n".join(part for line in summary.splitlines() for part in (textwrap.wrap(line, width=112, replace_whitespace=False, drop_whitespace=False) or [""]))
            story.append(Preformatted(wrapped, ParagraphStyle(name=f"Summary{name}", fontName="Courier", fontSize=6.5, leading=9)))

        def footer(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#64748B"))
            canvas.drawString(38, 25, "SalesVision | Actual application outputs | Sales Forecasting Using Python")
            canvas.drawRightString(A4[0] - 38, 25, f"Page {document.page}")
            canvas.restoreState()

        document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=38, rightMargin=38, topMargin=35, bottomMargin=42, title="SalesVision — Sales Forecasting Using Python", author="SalesVision")
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        return buffer.getvalue()
    except ReportError:
        raise
    except Exception as exc:
        raise ReportError("The PDF report could not be generated from the current analytics.") from exc
