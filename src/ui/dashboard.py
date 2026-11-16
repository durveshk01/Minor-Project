"""The twelve connected SalesVision workspaces using actual service outputs."""

import hashlib
import io
import json
import re
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai.gemini_client import GeminiClient
from src.ai.insight_generator import (
    TOPICS,
    build_analytics_context,
    context_fingerprint,
    generate_insights,
)
from src.data.aggregator import frequency_family
from src.data.loader import LoadedDataset, merge_walmart_files
from src.data.schema import AggregationOptions, CleaningOptions, ColumnMapping
from src.data.validator import convert_dates, convert_sales, detect_columns, validate_dataset
from src.features.time_features import (
    default_seasonal_period,
    dimension_summary,
    growth_series,
    holiday_summary,
    monthly_comparison,
    statistical_summary,
    summarize_forecast,
    trend_rollups,
    weekday_summary,
    yearly_comparison,
)
from src.forecasting.common import future_dates, validate_regressors
from src.reports.report_generator import (
    build_report_context,
    export_csv,
    export_excel,
    generate_excel_report,
    generate_pdf_report,
)
from src.ui.cache import (
    cached_evaluate,
    cached_forecast,
    cached_load,
    cached_process,
    cached_seasonality,
)
from src.ui.widgets import (
    compact_number,
    horizon_options,
    kpi_card,
    model_controls,
    navigate,
    page_heading,
    persistent_widget,
    remember_controls,
    render_chart,
    require_dataset,
)
from src.utils.exceptions import DataValidationError, SalesVisionError
from src.visualization.charts import (
    bar_chart,
    comparison_chart,
    decomposition_chart,
    distribution_chart,
    evaluation_chart,
    forecast_chart,
    historical_chart,
    rolling_chart,
    statistical_figure,
)

NAVIGATION = [
    "Overview Dashboard", "Dataset Management", "Data Preprocessing",
    "Exploratory Data Analysis", "Seasonal Analysis", "Model Training",
    "Model Comparison", "Sales Forecasting", "Gemini AI Insights",
    "Reports & Downloads", "Forecast History", "About Project",
]


def clear_results(include_processed: bool = False) -> None:
    keys = ["evaluation", "forecast", "seasonal", "insights", "report_package"]
    if include_processed:
        keys.append("processed")
    for key in keys:
        st.session_state[key] = None


def active_evaluation(dataset):
    result = st.session_state.get("evaluation")
    return result if result is not None and result.dataset_fingerprint == dataset.fingerprint else None


def active_forecast(dataset):
    result = st.session_state.get("forecast")
    return result if result is not None and result.dataset_fingerprint == dataset.fingerprint else None


def overview(settings, database) -> None:
    page_heading("Sales intelligence, in focus", "Intelligent Sales Forecasting & Business Analytics")
    dataset = st.session_state.get("processed")
    if dataset is None:
        st.markdown('<div class="sv-empty"><h2>Turn historical sales into a measurable forecast.</h2><p>Load a real CSV or Excel dataset, review its quality, compare three forecasting models on the same chronological holdout, and forecast future sales with uncertainty intervals.</p></div>', unsafe_allow_html=True)
        st.button("Start with a dataset", type="primary", on_click=navigate, args=("Dataset Management",))
        columns = st.columns(3)
        for column, title, description in zip(columns, ["01 · Prepare", "02 · Compare", "03 · Plan"], ["Inspect dates, numeric sales, duplicates, returns and missing periods.", "Evaluate ARIMA, SARIMA and Prophet using genuine holdout predictions.", "Generate future forecasts, business explanations and downloadable reports."]):
            with column:
                with st.container(border=True):
                    st.subheader(title)
                    st.write(description)
        return
    require_dataset()
    summary = statistical_summary(dataset.observed_series)
    evaluation, run = active_evaluation(dataset), active_forecast(dataset)
    unit = dataset.metadata.get("currency", "")
    forecast_summary = summarize_forecast(dataset.observed_series, run.predictions) if run else None
    best = evaluation.best_model if evaluation else None
    metric = evaluation.ranking_metric if evaluation else "RMSE"
    score = evaluation.metrics_for(best).get(metric) if best else None
    total_label = "Total Sales" if dataset.metadata["aggregation"] == "sum" else "Sum of Period Averages"
    cards = [
        (total_label, compact_number(summary["total"], unit), "Observed historical periods", "Historical"),
        ("Average Sales", compact_number(summary["mean"], unit), "Mean per observed period", "Historical"),
        ("Highest Sales", compact_number(summary["max"], unit), "Maximum observed period", "Historical"),
        ("Lowest Sales", compact_number(summary["min"], unit), "Minimum observed period", "Historical"),
        ("Sales Growth", f"{summary['growth_pct']:+.2f}%" if summary["growth_pct"] is not None else "—", "First to latest observed period", "Historical"),
        ("Forecasted Sales", compact_number(forecast_summary["forecast_total"], unit) if forecast_summary else "—", f"Raw sum over {run.horizon} future periods" if run else "Generate a future forecast", "Prediction"),
        ("Best Forecasting Model", best or "—", f"Lowest holdout {metric}" if best else "Evaluate forecasting models", "Evaluation"),
        ("Model Evaluation Score", compact_number(score), f"Holdout {metric}; lower is better", "Evaluation"),
    ]
    for start in (0, 4):
        for column, values in zip(st.columns(4), cards[start:start + 4]):
            with column:
                kpi_card(*values)
    left, right = st.columns([2, 1])
    with left:
        render_chart(forecast_chart(dataset.observed_series, run.predictions, run.model, run.configuration["interval_width"]) if run else historical_chart(dataset.observed_series), "overview_history")
    with right:
        if evaluation and evaluation.best_model:
            render_chart(comparison_chart(evaluation.comparison, metric), "overview_comparison")
        else:
            with st.container(border=True):
                st.subheader("Workflow status")
                st.write(f"Observed periods: {summary['observations']:,}")
                st.write(f"Unknown/imputed periods: {summary['missing_periods']:,}")
                st.write("Model input ready" if dataset.model_ready else "Missing-period policy needs review")
                st.button("Open Model Training", on_click=navigate, args=("Model Training",), disabled=not dataset.model_ready)
    if evaluation:
        st.subheader("Measured model performance")
        st.dataframe(evaluation.comparison, use_container_width=True, hide_index=True)


def local_catalog(root: Path) -> dict[str, Path]:
    catalog = {}
    for directory in ("data/sample", "data/uploads", "data/source"):
        for path in sorted((root / directory).iterdir()):
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                kind = "SYNTHETIC" if "synthetic" in path.name.lower() else "UCI real retail" if "uci_" in path.name.lower() or path.name == "Online Retail.xlsx" else "Local dataset"
                catalog[f"{kind} · {path.name}"] = path
    return dict(sorted(catalog.items(), key=lambda item: (0 if item[1].name == "train.csv" else 1 if item[1].name == "uci_online_retail_daily.csv" else 3 if "synthetic" in item[1].name else 2, item[0])))


def metadata_for_file(path: Path | None, loaded: LoadedDataset) -> dict:
    metadata = {}
    if path is not None:
        companion = path.with_suffix(".metadata.json")
        if path.name == "Online Retail.xlsx":
            companion = path.parents[1] / "sample/uci_online_retail_daily.metadata.json"
        if companion.is_file():
            metadata = json.loads(companion.read_text(encoding="utf-8"))
            if path.name == "Online Retail.xlsx":
                metadata["name"] = "UCI Online Retail — source invoice-line workbook"
    synthetic = "synthetic" in loaded.filename.lower() or ("Dataset_Type" in loaded.frame and loaded.frame["Dataset_Type"].astype(str).str.contains("synthetic", case=False).any())
    metadata.setdefault("name", loaded.filename)
    metadata.setdefault("kind", "synthetic" if synthetic else "user-provided")
    metadata.setdefault("source_sha256", loaded.sha256)
    metadata.update({"filename": loaded.filename, "dataset_file_sha256": loaded.sha256, "size_bytes": loaded.size_bytes})
    return metadata


def read_candidate(content: bytes, filename: str, prefix: str, maximum: int):
    if len(content) > maximum * 1024**2:
        raise DataValidationError(f"The file exceeds the {maximum} MB limit.")
    sheet = 0
    if Path(filename).suffix.lower() in {".xlsx", ".xls"}:
        try:
            with pd.ExcelFile(io.BytesIO(content)) as workbook:
                sheets = workbook.sheet_names
            sheet = persistent_widget("selectbox", "Worksheet", sheets, key=f"{prefix}_worksheet_{hashlib.sha256(content).hexdigest()[:12]}")
        except Exception as exc:
            raise DataValidationError("The uploaded Excel workbook could not be opened.") from exc
    return cached_load(content, filename, sheet, maximum)


def prepare_uploaded_dataset(candidate: LoadedDataset, stores: LoadedDataset | None = None, features: LoadedDataset | None = None):
    """Preserve byte-level source provenance separately from joined-table cache identity."""
    metadata = metadata_for_file(None, candidate)
    operations = []
    if stores is not None or features is not None:
        joined, operations = merge_walmart_files(candidate.frame, stores.frame if stores is not None else None, features.frame if features is not None else None)
        sources = {role: {"filename": loaded.filename, "sha256": loaded.sha256, "size_bytes": loaded.size_bytes} for role, loaded in (("stores", stores), ("features", features)) if loaded is not None}
        identity = hashlib.sha256(json.dumps({"history": candidate.sha256, "ancillary": sources}, sort_keys=True).encode()).hexdigest()
        candidate = replace(candidate, frame=joined, sha256=identity)
        metadata.update({"joined_dataset_sha256": identity, "ancillary_sources": sources})
    return candidate, metadata, operations


def activate_dataset(loaded, metadata, operations=None) -> None:
    st.session_state["loaded"] = loaded
    st.session_state["loaded_metadata"] = metadata
    st.session_state["load_operations"] = operations or []
    clear_results(include_processed=True)


def dataset_management(settings, database) -> None:
    page_heading("Dataset Management", "Load CSV or Excel sales data, inspect its source and choose an analysis dataset.")
    catalog = local_catalog(settings.root)
    local, upload = st.tabs(["Available datasets", "Upload CSV / Excel"])
    with local:
        if catalog:
            selection = persistent_widget("selectbox", "Available local datasets", list(catalog), key="local_dataset_selection")
            path = catalog[selection]
            if st.button("Load selected dataset", type="primary"):
                with st.spinner("Reading and inspecting the selected dataset…"):
                    if path.stat().st_size > settings.max_upload_mb * 1024**2:
                        raise DataValidationError(f"The file exceeds the {settings.max_upload_mb} MB limit.")
                    loaded = cached_load(path.read_bytes(), path.name, max_size_mb=settings.max_upload_mb)
                    activate_dataset(loaded, metadata_for_file(path, loaded))
                st.success(f"Loaded {len(loaded.frame):,} records from {loaded.filename}.")
        else:
            st.info("No local datasets are available. Upload a CSV/Excel file or run a documented dataset acquisition script.")
    with upload:
        uploaded = st.file_uploader("Historical sales dataset", type=["csv", "xlsx", "xls"], key="historical_uploader")
        if uploaded is not None:
            try:
                candidate = read_candidate(uploaded.getvalue(), uploaded.name, "historical", settings.max_upload_mb)
                st.dataframe(candidate.frame.head(100), use_container_width=True, hide_index=True)
                stores, features = None, None
                operations = []
                if {"Store", "Dept", "Weekly_Sales"}.issubset(candidate.frame):
                    with st.expander("Optional Walmart ancillary files"):
                        stores_file = st.file_uploader("stores.csv", type=["csv", "xlsx"], key="stores_uploader")
                        features_file = st.file_uploader("features.csv", type=["csv", "xlsx"], key="features_uploader")
                        st.caption("stores.csv joins on Store; features.csv joins on Store/Date. Joins are checked for row multiplication. Competition test.csv has no historical sales labels.")
                    if stores_file:
                        stores = cached_load(stores_file.getvalue(), stores_file.name, max_size_mb=settings.max_upload_mb)
                    if features_file:
                        features = cached_load(features_file.getvalue(), features_file.name, max_size_mb=settings.max_upload_mb)
                persist = persistent_widget("checkbox", "Save the uploaded source file in data/uploads", value=False, key="persist_upload")
                if st.button("Use uploaded dataset", type="primary"):
                    candidate, metadata, operations = prepare_uploaded_dataset(candidate, stores, features)
                    activate_dataset(candidate, metadata, operations)
                    if persist:
                        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(uploaded.name).name)
                        destination = settings.root / "data/uploads" / f"{hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]}_{safe_name}"
                        if not destination.exists():
                            with destination.open("xb") as handle:
                                handle.write(uploaded.getvalue())
                    st.success(f"Activated {len(candidate.frame):,} uploaded records.")
            except SalesVisionError as exc:
                st.error(str(exc))
    loaded = st.session_state.get("loaded")
    if loaded is not None:
        st.divider()
        st.subheader("Active source preview")
        metadata = st.session_state["loaded_metadata"]
        if metadata.get("kind") == "synthetic":
            st.warning("SYNTHETIC TEST DATA · This source is explicitly simulated and is not Walmart data.")
        columns = st.columns(4)
        for column, label, value in zip(columns, ["Source records", "Columns", "Exact duplicates", "Missing cells"], [len(loaded.frame), len(loaded.frame.columns), int(loaded.frame.duplicated().sum()), int(loaded.frame.isna().sum().sum())]):
            column.metric(label, f"{value:,}")
        st.dataframe(loaded.frame.head(200), use_container_width=True, hide_index=True)
        missing = loaded.frame.isna().sum()
        st.dataframe(pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Missing (%)": missing.values / len(loaded.frame) * 100}), use_container_width=True, hide_index=True)
        st.caption(f"SHA-256: {loaded.sha256} · Size: {loaded.size_bytes / 1024**2:.2f} MB")
        if metadata.get("joined_dataset_sha256"):
            st.caption(f"The active joined-table fingerprint is separate from the original dataset-file SHA-256: {metadata['dataset_file_sha256']}.")
        st.write("Source:", metadata.get("source_url", "User-provided; specify provenance during preprocessing."))
        if metadata.get("citation"):
            st.caption(metadata["citation"])
        st.button("Continue to Data Preprocessing", on_click=navigate, args=("Data Preprocessing",), type="primary")
    with st.expander("Obtain the preferred Walmart competition dataset"):
        st.markdown("1. Visit [Kaggle Walmart Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data).\n2. Sign in, review the current competition terms and accept them where your academic use is permitted.\n3. Download and extract `train.csv`, `test.csv`, `stores.csv`, and `features.csv`.\n4. Upload `train.csv` as history and optionally join stores/features.\n\nWalmart observations are **weekly**. `test.csv` has no `Weekly_Sales`, so it cannot be scored as a labelled holdout.")
    with st.expander("UCI real-data source and synthetic fallback"):
        st.write("The included real daily revenue file is derived from UCI Online Retail (CC BY 4.0). Customer identifiers are excluded, and returns/cancellations and source duplicates are retained. Dates with no source transactions are absent until you select an explicit missing-period policy.")
        st.code("python scripts/fetch_uci_data.py\npython scripts/generate_synthetic_data.py", language="powershell")


def preprocessing(settings, database) -> None:
    page_heading("Data Preprocessing", "Review data quality and explicitly choose every cleaning and aggregation policy.")
    loaded = st.session_state.get("loaded")
    if loaded is None:
        require_dataset()
        return
    frame = loaded.frame
    detected = detect_columns(frame)
    prefix = loaded.sha256[:12]
    columns = list(frame.columns)
    date_column = persistent_widget("selectbox", "Date column", columns, index=columns.index(detected["date"]) if detected["date"] else 0, key=f"date_column_{prefix}")
    can_derive = detected["quantity"] is not None and detected["unit_price"] is not None
    modes = ["Existing sales column", "Quantity × UnitPrice revenue"] if can_derive else ["Existing sales column"]
    mode = persistent_widget("radio", "Sales definition", modes, index=1 if can_derive and not detected["sales"] else 0, horizontal=True, key=f"sales_definition_{prefix}")
    sales_column, quantity, unit_price = None, None, None
    if mode == "Existing sales column":
        sales_column = persistent_widget("selectbox", "Sales column", columns, index=columns.index(detected["sales"]) if detected["sales"] else min(1, len(columns) - 1), key=f"sales_column_{prefix}")
    else:
        left, right = st.columns(2)
        quantity = persistent_widget("selectbox", "Quantity column", columns, container=left, index=columns.index(detected["quantity"]), key=f"quantity_{prefix}")
        unit_price = persistent_widget("selectbox", "Unit price column", columns, container=right, index=columns.index(detected["unit_price"]), key=f"unitprice_{prefix}")
        st.caption("Revenue derivation is recorded in the cleaning audit; negative quantities may represent returns.")
    holiday_choices = ["None", *columns]
    holiday_column = persistent_widget("selectbox", "Holiday flag column", holiday_choices, index=holiday_choices.index(detected["holiday"]) if detected["holiday"] else 0, key=f"holiday_column_{prefix}")
    mapping = ColumnMapping(date_column, sales_column, quantity, unit_price, None if holiday_column == "None" else holiday_column)
    left, right = st.columns(2)
    date_format = persistent_widget("text_input", "Date format (optional)", container=left, placeholder="e.g. %d/%m/%Y", key=f"date_format_{prefix}") or None
    day_first = persistent_widget("checkbox", "Interpret ambiguous dates day-first", container=right, value=False, key=f"dayfirst_{prefix}")
    try:
        quality = validate_dataset(frame, mapping, date_format, day_first)
        dates = convert_dates(frame[date_column], date_format, day_first)
        sales = convert_sales(frame, mapping)
    except SalesVisionError as exc:
        st.error(str(exc))
        return
    st.subheader("Quality before cleaning")
    metrics = st.columns(4)
    metrics[0].metric("Invalid / missing dates", f"{quality.invalid_dates + quality.missing_dates:,}")
    metrics[1].metric("Invalid / missing sales", f"{quality.invalid_sales + quality.missing_sales:,}")
    metrics[2].metric("Exact duplicates", f"{quality.duplicates:,}")
    metrics[3].metric("Negative / zero sales", f"{quality.negative_sales:,} / {quality.zero_sales:,}")
    invalid = dates.isna() | sales.isna()
    with st.expander("Inspect invalid/missing records and negative sales"):
        st.write("Invalid or missing date/sales records:")
        st.dataframe(frame.loc[invalid].head(200), use_container_width=True, hide_index=True)
        st.write("Negative sales records:")
        st.dataframe(frame.loc[sales.lt(0)].head(200), use_container_width=True, hide_index=True)
    invalid_policy = persistent_widget("selectbox", "Invalid records", ["Stop and review", "Exclude and audit"], key=f"invalid_policy_{prefix}")
    duplicate_policy = persistent_widget("selectbox", "Exact duplicate policy", ["Retain", "Remove exact duplicates"], key=f"duplicates_{prefix}")
    missing_sales_policy = persistent_widget("selectbox", "Missing sales inputs", ["Stop and review", "Retain as unknown", "Exclude and audit"], key=f"missing_sales_{prefix}")
    negative_policy = persistent_widget("selectbox", "Negative sales", ["Retain as net returns", "Stop and review", "Exclude and audit"], key=f"negative_{prefix}")
    native = quality.suggested_frequency
    st.caption(f"Suggested source frequency: {native or 'irregular / insufficient evidence — select the native frequency explicitly'}. Frequency selection never creates finer-grained observed sales.")
    source_frequency = "auto"
    if native is None:
        source_choice = persistent_widget("selectbox", "Native source frequency", ["Select native frequency", "Daily", "Weekly", "Monthly"], key=f"native_frequency_{prefix}")
        source_frequency = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}.get(source_choice)
        native = source_frequency
    family = frequency_family(native) if native else None
    families = ["Daily", "Weekly", "Monthly"]
    allowed = families[families.index(family.title()):] if family else families
    frequency_choice = persistent_widget("selectbox", "Forecasting frequency", allowed, key=f"frequency_{prefix}")
    if frequency_choice == "Weekly":
        anchors = [f"W-{day}" for day in ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]]
        frequency = persistent_widget("selectbox", "Week-ending day", anchors, index=anchors.index(native) if native in anchors else 6, key=f"anchor_{prefix}")
    else:
        frequency = "D" if frequency_choice == "Daily" else "MS"
    method = persistent_widget("selectbox", "Aggregation", ["Total sales (sum)", "Average record sales (mean)"], key=f"aggregation_{prefix}")
    period_policies = {"Keep unknown": "keep", "Forward-fill (past values only)": "ffill", "Zero (explicit business assumption)": "zero", "Stop and review gaps": "reject"}
    missing_period_choice = persistent_widget("selectbox", "Missing time periods", list(period_policies), key=f"missing_periods_{prefix}")
    st.caption("Unknown periods are not genuine zeros. Forward-fill uses past values only. Zero filling requires an explicit business assumption. Imputed targets are excluded from model evaluation.")
    filters = {}
    with st.expander("Store, department, category and region filters"):
        for role in ["store", "department", "category", "region"]:
            column = detected[role]
            if column:
                available = sorted(frame[column].dropna().unique().tolist(), key=str)
                selected = persistent_widget("multiselect", f"{role.title()} · {column}", available, key=f"filter_{role}_{prefix}", help="Leave empty to include all values.")
                if selected:
                    filters[column] = selected
    date_range = None
    valid_dates = dates.dropna()
    if not valid_dates.empty and persistent_widget("checkbox", "Restrict historical date window", key=f"date_window_{prefix}"):
        interval = persistent_widget("date_input", "Historical date window", value=(valid_dates.min().date(), valid_dates.max().date()), min_value=valid_dates.min().date(), max_value=valid_dates.max().date(), key=f"date_window_values_{prefix}")
        if isinstance(interval, tuple) and len(interval) == 2:
            date_range = tuple(value.isoformat() for value in interval)
        else:
            st.info("Select both the start and end of the historical date window.")
            return
    excluded = {date_column, sales_column, quantity, unit_price, "CustomerID", "Customer_ID"}
    numeric_columns = [column for column in frame if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]
    regressors = tuple(persistent_widget("multiselect", "Optional numeric regressors to aggregate", numeric_columns, key=f"regressors_{prefix}", help="Known calendar/planned covariates only. Future values are required for final forecasting."))
    metadata = dict(st.session_state["loaded_metadata"])
    if metadata.get("kind") == "user-provided":
        metadata["source_url"] = persistent_widget("text_input", "Dataset source / description", value=metadata.get("source_url", "User-provided dataset"), key=f"source_{prefix}")
        metadata["currency"] = persistent_widget("selectbox", "Currency or units", ["sales units", "GBP", "USD", "EUR", "INR"], key=f"currency_{prefix}")
    cleaning = CleaningOptions("drop" if invalid_policy == "Exclude and audit" else "reject", "drop" if duplicate_policy == "Remove exact duplicates" else "keep", {"Stop and review": "reject", "Retain as unknown": "keep", "Exclude and audit": "drop"}[missing_sales_policy], {"Retain as net returns": "keep", "Stop and review": "reject", "Exclude and audit": "drop"}[negative_policy], date_format, day_first)
    aggregation = AggregationOptions(frequency, "sum" if method.startswith("Total") else "mean", period_policies[missing_period_choice], source_frequency or "auto", regressors)
    if st.button("Apply preprocessing", type="primary"):
        try:
            if source_frequency is None:
                raise DataValidationError("Select the native source frequency explicitly when it cannot be inferred.")
            with st.spinner("Validating, cleaning and constructing the sales calendar…"):
                processed = cached_process(frame, mapping, cleaning, aggregation, metadata, filters, st.session_state["load_operations"], date_range)
            clear_results()
            st.session_state["processed"] = processed
            st.success(f"Processed {len(processed.records):,} records into {len(processed.series):,} {processed.frequency} periods.")
        except SalesVisionError as exc:
            st.error(str(exc))
    processed = st.session_state.get("processed")
    if processed is not None:
        st.subheader("Applied processing audit")
        st.dataframe(pd.DataFrame(processed.operations), use_container_width=True, hide_index=True)
        preview = pd.DataFrame({"observed_sales": processed.observed_series, "model_input": processed.series, "is_observed": processed.observed_mask}).rename_axis("date").reset_index()
        st.dataframe(preview, use_container_width=True, hide_index=True)
        if processed.missing_periods:
            st.caption(f"{len(processed.missing_periods)} absent or incomplete periods were detected. Original unknown observations remain recorded in observed_sales.")
        if not processed.rejected_records.empty:
            with st.expander("Excluded record audit"):
                st.dataframe(processed.rejected_records, use_container_width=True, hide_index=True)
        with st.expander("Flagged outliers — diagnostic only"):
            st.dataframe(processed.outliers, use_container_width=True, hide_index=True)
        st.download_button("Download processed CSV", export_csv(preview), "salesvision_processed.csv", "text/csv", on_click="ignore")
        st.button("Continue to Exploratory Data Analysis", on_click=navigate, args=("Exploratory Data Analysis",), type="primary")


def eda(settings, database) -> None:
    page_heading("Exploratory Data Analysis", "Explore observed trends, distributions, growth and available business segments.")
    dataset = require_dataset()
    if dataset is None:
        return
    series = dataset.observed_series
    st.dataframe(pd.DataFrame([statistical_summary(series)]).drop(columns=["growth_definition"]), use_container_width=True, hide_index=True)
    trends, distribution, segments = st.tabs(["Sales trends", "Distribution & growth", "Segments & holidays"])
    with trends:
        for index, (title, rolled) in enumerate(trend_rollups(series, dataset.frequency).items()):
            render_chart(historical_chart(rolled, title), f"eda_trend_{index}")
        if frequency_family(dataset.frequency) != "daily":
            st.caption("Daily and weekday views require daily source observations and are unavailable for this frequency.")
        window = int(persistent_widget("number_input", "Trailing average window (periods)", min_value=2, max_value=max(2, len(series)), value=min(default_seasonal_period(dataset.frequency), max(2, len(series))), key=f"rolling_window_{dataset.fingerprint[:12]}"))
        render_chart(rolling_chart(series, window), "eda_rolling")
        monthly = monthly_comparison(series)
        figure = bar_chart(monthly, "month_name", "sales", "Monthly observed-sales comparison", "year")
        figure.update_xaxes(categoryorder="array", categoryarray=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        render_chart(figure, "eda_monthly_comparison")
        years = yearly_comparison(series)
        if len(years) >= 2:
            render_chart(bar_chart(years, "year", "total", "Yearly observed-sales comparison"), "eda_yearly")
        st.caption("Rollups use observed contributions. Boundary periods and periods containing missing observations may be incomplete.")
    with distribution:
        render_chart(distribution_chart(series), "eda_distribution")
        growth = historical_chart(growth_series(series), "Period-on-period observed sales growth")
        growth.update_yaxes(title="Growth (%)")
        render_chart(growth, "eda_growth")
        st.caption("Percentage growth is unavailable after a zero or unknown prior period; negative prior values use the absolute denominator.")
        figure = statistical_figure(series)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        if frequency_family(dataset.frequency) == "daily":
            weekdays = weekday_summary(series, dataset.frequency)
            render_chart(bar_chart(weekdays, "weekday", "mean", "Observed daily sales by weekday"), "eda_weekdays")
            st.dataframe(weekdays, hide_index=True, use_container_width=True)
    with segments:
        detected = detect_columns(dataset.records)
        available = False
        for role in ["store", "department", "category", "region"]:
            column = detected[role]
            if column:
                available = True
                table = dimension_summary(dataset.records, column)
                render_chart(bar_chart(table.head(20), column, "total_sales", f"{role.title()} comparison · top 20 observed-record totals"), f"eda_segment_{role}")
                with st.expander(f"All {role} results"):
                    st.dataframe(table, use_container_width=True, hide_index=True)
        if detected["holiday"]:
            available = True
            holidays = holiday_summary(dataset.records, detected["holiday"])
            render_chart(bar_chart(holidays.astype({"holiday": str}), "holiday", "mean_period_sales", "Supplied holiday flags · observed-date mean sales"), "eda_holidays")
            st.dataframe(holidays, use_container_width=True, hide_index=True)
            st.caption("Holiday comparisons describe observed association, not a proven causal uplift.")
        if not available:
            st.info("This source has no store, department, category, region or holiday fields. Upload transaction-level data to enable segment analysis.")


def seasonal_analysis(settings, database) -> None:
    page_heading("Seasonal Analysis", "Separate observed sales into trend, seasonal and residual components using robust STL.")
    dataset = require_dataset()
    if dataset is None:
        return
    period = int(persistent_widget("number_input", "Seasonal cycle (observations)", min_value=2, max_value=366, value=default_seasonal_period(dataset.frequency), key=f"stl_period_{dataset.fingerprint[:12]}"))
    st.caption(f"At least {2 * period} observations are required. Available: {len(dataset.series)}. Weekly annual seasonality uses an approximate 52-observation cycle; monthly annual seasonality uses 12.")
    if (~dataset.observed_mask).any():
        st.info("This decomposition uses the explicitly filled modelling series. Interpret its seasonal strength alongside the missing-period audit.")
    if st.button("Run seasonal decomposition", type="primary", disabled=not dataset.model_ready):
        try:
            with st.spinner("Computing trend, seasonality and residuals…"):
                result = cached_seasonality(dataset.series, dataset.fingerprint, period)
            st.session_state["seasonal"] = result
            st.session_state["insights"] = None
            st.session_state["report_package"] = None
        except SalesVisionError as exc:
            st.error(str(exc))
    result = st.session_state.get("seasonal")
    if result is not None:
        st.write(result.explanation)
        left, right = st.columns(2)
        left.metric("Estimated seasonal strength", f"{result.strength:.3f}")
        right.metric("Seasonal amplitude (5th–95th percentile)", compact_number(result.amplitude))
        render_chart(decomposition_chart(result.components), "seasonal_decomposition")
        st.dataframe(result.components.rename_axis("date").reset_index(), use_container_width=True, hide_index=True)
    elif not dataset.model_ready:
        st.info("Choose an explicit missing-period policy in preprocessing before decomposition.")


def display_evaluation(evaluation, prefix: str) -> None:
    st.dataframe(evaluation.comparison, use_container_width=True, hide_index=True)
    failed = evaluation.comparison.loc[evaluation.comparison["Status"].eq("failed")]
    for row in failed.itertuples(index=False):
        st.warning(f"{row.Model}: {row.Error}")
    if evaluation.best_model:
        score = evaluation.metrics_for(evaluation.best_model)[evaluation.ranking_metric]
        st.success(f"Best successful model: {evaluation.best_model} · lowest holdout {evaluation.ranking_metric} = {score:,.3f}.")
        render_chart(comparison_chart(evaluation.comparison, evaluation.ranking_metric), f"{prefix}_comparison")
    if evaluation.predictions:
        render_chart(evaluation_chart(evaluation.predictions), f"{prefix}_actual_predicted")
    st.caption(f"Training end: {evaluation.training_end[:10]}. Shared holdout: {evaluation.test_start[:10]} to {evaluation.test_end[:10]}. Missing/imputed actuals are not scored; MAPE excludes zero actuals only.")


def model_training(settings, database) -> None:
    page_heading("Model Training", "Fit genuine forecasting models and evaluate them on one chronological test period.")
    dataset = require_dataset()
    if dataset is None:
        return
    if not dataset.model_ready or len(dataset.series) < 16:
        st.info("Training needs a complete modelling calendar and sufficient history. Review preprocessing or upload more observations.")
        return
    prefix = dataset.fingerprint[:12]
    selected = persistent_widget("multiselect", "Models to evaluate", ["ARIMA", "SARIMA", "Prophet"], default=["ARIMA", "SARIMA", "Prophet"], key=f"training_models_{prefix}")
    with st.form(f"training_form_{prefix}"):
        horizons, unit = horizon_options(dataset.frequency)
        test_size = int(persistent_widget("number_input", f"Chronological holdout ({unit})", min_value=2, max_value=min(366, len(dataset.series) - 14), value=min(horizons[1], len(dataset.series) - 14), key=f"test_size_{prefix}", in_form=True))
        metric = persistent_widget("selectbox", "Model selection metric", ["RMSE", "MAE", "sMAPE", "MAPE"], help="Lowest valid score wins; ties use model-name order.", key=f"selection_metric_{prefix}", in_form=True)
        configurations = []
        for name in selected:
            with st.expander(f"{name} parameters", expanded=name == "ARIMA"):
                configurations.append(model_controls(name, dataset, "training", in_form=True))
        known_regressors = persistent_widget("checkbox", "Configured regressor values are known at each forecast origin", value=False, help="Use calendar or pre-planned drivers, not subsequently realized outcomes.", key=f"known_regressors_{prefix}", in_form=True)
        backtest_folds = persistent_widget("selectbox", "Rolling-origin folds within the training partition", [0, 2, 3], help="Zero disables backtesting. The final chronological holdout remains separate.", key=f"backtest_folds_{prefix}", in_form=True)
        backtest_horizon = int(persistent_widget("number_input", "Backtesting horizon (periods)", min_value=1, max_value=90, value=min(14, test_size), key=f"backtest_horizon_{prefix}", in_form=True))
        submitted = st.form_submit_button("Train and evaluate models", type="primary", disabled=not selected, on_click=remember_controls)
    if submitted:
        try:
            with st.spinner("Fitting selected models against the shared chronological holdout…"):
                result = cached_evaluate(dataset, dataset.fingerprint, tuple(configurations), test_size, metric, backtest_folds, backtest_horizon, known_regressors)
            st.session_state["evaluation"] = result
            st.session_state["known_regressors"] = known_regressors
            for key in ["forecast", "insights", "report_package"]:
                st.session_state[key] = None
        except SalesVisionError as exc:
            st.error(str(exc))
    result = active_evaluation(dataset)
    if result is not None:
        display_evaluation(result, "training")
        st.button("Open Model Comparison", on_click=navigate, args=("Model Comparison",), type="primary")


def model_comparison(settings, database) -> None:
    page_heading("Model Comparison", "Inspect genuine errors, fitting diagnostics and explicitly metric-driven model selection.")
    dataset = require_dataset()
    if dataset is None:
        return
    result = active_evaluation(dataset)
    if result is None:
        st.info("Train models first to produce measured comparison results.")
        st.button("Open Model Training", on_click=navigate, args=("Model Training",), type="primary")
        return
    display_evaluation(result, "comparison")
    if result.predictions:
        selected = persistent_widget("selectbox", "Inspect evaluated model", list(result.predictions), key=f"inspect_model_{dataset.fingerprint[:12]}")
        st.dataframe(result.predictions[selected], use_container_width=True, hide_index=True)
        st.download_button("Download holdout predictions CSV", export_csv(result.predictions[selected]), f"{selected.lower()}_holdout.csv", "text/csv", on_click="ignore")
        with st.expander("Recorded model configuration and fitted summary"):
            st.json(result.configurations[selected].to_dict())
            st.text(result.summaries[selected])
            for note in result.warnings.get(selected, []):
                st.caption(note)
    if result.backtest is not None:
        st.subheader("Rolling-origin backtesting · training partition only")
        st.dataframe(result.backtest.comparison, use_container_width=True, hide_index=True)
        st.dataframe(result.backtest.folds, use_container_width=True, hide_index=True)
    st.button("Continue to Sales Forecasting", on_click=navigate, args=("Sales Forecasting",), type="primary", disabled=not result.predictions)


def future_inputs(dataset, config, horizon, maximum):
    regressors, holidays = None, None
    expected = future_dates(dataset.series, horizon)
    if config.regressor_columns:
        st.write("Required future regressor columns:", ", ".join(config.regressor_columns))
        uploaded = st.file_uploader("Future regressors CSV / Excel", type=["csv", "xlsx", "xls"], key="future_regressor_uploader")
        if uploaded is not None:
            loaded = read_candidate(uploaded.getvalue(), uploaded.name, "future_exog", maximum)
            detected = detect_columns(loaded.frame)
            column = persistent_widget("selectbox", "Future regressor date column", list(loaded.frame), index=list(loaded.frame).index(detected["date"]) if detected["date"] else 0, key=f"future_date_column_{loaded.sha256[:12]}")
            regressors = loaded.frame.copy()
            regressors.index = convert_dates(regressors[column]).dt.normalize()
            regressors = validate_regressors(regressors, expected, config.regressor_columns)
            st.dataframe(regressors, use_container_width=True)
        st.caption("Future values must cover every forecast date exactly, use the same period aggregation and be known before forecasting. No future covariates are fabricated.")
    if config.use_holidays:
        uploaded = st.file_uploader("Optional future holiday calendar (ds, holiday)", type=["csv", "xlsx"], key="future_holiday_uploader")
        if uploaded is not None:
            holidays = cached_load(uploaded.getvalue(), uploaded.name, max_size_mb=maximum).frame
            if not {"ds", "holiday"}.issubset(holidays):
                raise DataValidationError("The holiday calendar requires ds and holiday columns.")
            holidays = holidays[["ds", "holiday"]].copy()
            holidays["ds"] = convert_dates(holidays["ds"]).dt.normalize()
            if holidays["ds"].isna().any() or not holidays["ds"].isin(expected).all():
                raise DataValidationError("Future holiday dates must be valid and match forecast-calendar periods exactly.")
        st.caption("Historical flags do not imply future holidays. Supply known dates matching the forecasting calendar: daily dates, the selected week-ending day or monthly starts.")
    return regressors, holidays


def sales_forecasting(settings, database) -> None:
    page_heading("Sales Forecasting", "Retrain the selected model on full history and generate future sales with model uncertainty.")
    dataset = require_dataset()
    if dataset is None:
        return
    evaluation = active_evaluation(dataset)
    if evaluation is None or not evaluation.predictions:
        st.info("Evaluate at least one model on a chronological holdout before final forecasting.")
        st.button("Open Model Training", on_click=navigate, args=("Model Training",), type="primary")
        return
    st.button("Change dataset or frequency", on_click=navigate, args=("Dataset Management",))
    models = list(evaluation.predictions)
    previous = active_forecast(dataset)
    default_model = previous.model if previous is not None and previous.model in models else evaluation.best_model
    model_key = f"forecast_model_{dataset.fingerprint[:12]}"
    chosen = persistent_widget("selectbox", "Forecasting model", models, index=models.index(default_model) if default_model in models else 0, key=model_key)
    config = evaluation.configurations[chosen]
    if persistent_widget("checkbox", "Customize model configuration", key=f"customize_forecast_{dataset.fingerprint[:12]}"):
        with st.expander("Forecast configuration", expanded=True):
            config = model_controls(chosen, dataset, "forecast", config)
        if config != evaluation.configurations[chosen]:
            st.info("Evaluate the modified configuration on the same holdout before generating a final forecast.")
            if st.button("Re-evaluate modified configuration", type="primary"):
                configs = [config if name == chosen else existing for name, existing in evaluation.configurations.items()]
                with st.spinner("Evaluating the revised configuration…"):
                    st.session_state["evaluation"] = cached_evaluate(dataset, dataset.fingerprint, tuple(configs), evaluation.test_size, evaluation.ranking_metric, 0, 14, st.session_state.get("known_regressors", False))
                st.session_state["forecast"] = None
                st.session_state["insights"] = None
                st.session_state["report_package"] = None
                st.rerun()
            return
    horizons, unit = horizon_options(dataset.frequency)
    options = [*horizons, "Custom"]
    default_horizon = previous.horizon if previous is not None and previous.horizon in horizons else "Custom" if previous is not None else horizons[0]
    horizon_key = f"forecast_horizon_{dataset.fingerprint[:12]}"
    selected_horizon = persistent_widget("selectbox", "Forecast horizon", options, index=options.index(default_horizon), format_func=lambda value: f"{value} {unit}" if isinstance(value, int) else value, key=horizon_key)
    horizon = int(persistent_widget("number_input", "Custom horizon (periods)", min_value=1, max_value=366, value=previous.horizon if previous is not None else 30, key=f"custom_horizon_{dataset.fingerprint[:12]}")) if selected_horizon == "Custom" else selected_horizon
    future_regressors, future_holidays = future_inputs(dataset, config, horizon, settings.max_upload_mb)
    constraint = persistent_widget("checkbox", "Add a separate nonnegative business-planning view", value=False, help="Raw predictions remain available. Clipping does not preserve calibrated interval coverage.", key=f"nonnegative_{dataset.fingerprint[:12]}")
    save_history = persistent_widget("checkbox", "Save this execution and report snapshot to forecast history", value=True, disabled=database is None, key=f"save_history_{dataset.fingerprint[:12]}") and database is not None
    if st.button("Generate future forecast", type="primary"):
        try:
            if config.regressor_columns and future_regressors is None:
                raise DataValidationError("Upload the required future regressors before forecasting.")
            with st.spinner("Preparing the full-history model forecast…"):
                computed = cached_forecast(dataset, dataset.fingerprint, config, horizon, evaluation.metrics_for(chosen), future_regressors, future_holidays, "nonnegative" if constraint else "none")
            metadata = dict(computed.metadata)
            metadata["computation_created_at"] = computed.created_at
            run = replace(computed, run_id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), metadata=metadata)
            st.session_state["forecast"] = run
            st.session_state["insights"] = None
            st.session_state["report_package"] = None
            if save_history:
                report = build_report_context(dataset, evaluation, run, st.session_state.get("seasonal"))
                database.save_forecast(run, report)
            st.success(f"Generated {run.horizon} future {unit} using {run.model}. Full-history training observations: {run.training_observations}.")
        except SalesVisionError as exc:
            st.error(str(exc))
    run = active_forecast(dataset)
    if run is not None:
        summary = summarize_forecast(dataset.observed_series, run.predictions)
        st.caption(f"Displayed forecast: {run.model} · {run.horizon} {unit} · {summary['start'][:10]} to {summary['end'][:10]}. Generate again to apply revised inputs.")
        left, middle, right = st.columns(3)
        left.metric("Raw forecast total", compact_number(summary["forecast_total"], dataset.metadata.get("currency", "")))
        middle.metric("Mean future period sales", compact_number(summary["forecast_mean"], dataset.metadata.get("currency", "")))
        right.metric("Growth vs preceding equal horizon", f"{summary['growth_vs_preceding_horizon_pct']:+.2f}%" if summary["growth_vs_preceding_horizon_pct"] is not None else "—")
        st.caption(summary["growth_definition"])
        for warning in run.warnings:
            st.warning(warning)
        render_chart(forecast_chart(dataset.observed_series, run.predictions, run.model, run.configuration["interval_width"]), "final_forecast")
        st.dataframe(run.predictions, use_container_width=True, hide_index=True)
        st.caption(f"Model fitting time: {run.training_seconds:.3f}s. Computation created: {run.metadata.get('computation_created_at', run.created_at)}. Identical data/configuration can reuse cached computation; each button execution has its own history identifier.")
        left, right = st.columns(2)
        left.download_button("Download forecast CSV", export_csv(run.predictions), "salesvision_forecast.csv", "text/csv", on_click="ignore")
        right.download_button("Download forecast Excel", export_excel(run.predictions, "Forecast"), "salesvision_forecast.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")
        if not run.components.empty:
            with st.expander("Prophet forecast components"):
                st.dataframe(run.components, use_container_width=True, hide_index=True)
        with st.expander("Full-history model configuration and summary"):
            st.json(run.configuration)
            st.text(run.model_summary)


def gemini_insights(settings, database) -> None:
    page_heading("Gemini AI Insights", "Explain actual sales analytics and forecasts with the official Google Gemini SDK.")
    dataset = require_dataset()
    if dataset is None:
        return
    evaluation, run, seasonal = active_evaluation(dataset), active_forecast(dataset), st.session_state.get("seasonal")
    context = build_analytics_context(dataset, evaluation, run, seasonal)
    model = persistent_widget("text_input", "Gemini model", value=settings.gemini_model, key="gemini_model_override")
    st.caption("The API key is read on the server from GEMINI_API_KEY or Streamlit secrets. Requests contain aggregated statistics and forecast summaries; no customer rows or identifiers are sent.")
    if not settings.gemini_api_key:
        st.info("Gemini is not configured. Set GEMINI_API_KEY locally in .env or Streamlit secrets. Computed local explanations remain available.")
    if st.button("Check Gemini connectivity"):
        with st.spinner("Checking the configured model…"):
            with GeminiClient(settings.gemini_api_key, model) as client:
                status = client.check_connectivity()
        st.success(status.message) if status.ok else st.warning(status.message)
    topic = persistent_widget("selectbox", "Business insight topic", TOPICS, key="insight_topic")
    question = persistent_widget("text_area", "Question about current analytics", max_chars=1500, key="insight_question") if topic == "Ask a question" else ""
    if st.button("Generate business explanation", type="primary"):
        with st.spinner("Preparing an evidence-based business explanation…"):
            with GeminiClient(settings.gemini_api_key, model) as client:
                result = generate_insights(context, topic, question, client)
        st.session_state["insights"] = result
        st.session_state["report_package"] = None
    result = st.session_state.get("insights")
    if result is not None and result.context_hash == context_fingerprint(context):
        if result.status != "success":
            st.warning(result.message)
        st.caption(f"Explanation source: {result.source} · Generated (UTC): {result.generated_at}")
        st.markdown(result.text)
        if run is not None and st.button("Save current explanation with forecast history", disabled=database is None):
            database.save_forecast(run, build_report_context(dataset, evaluation, run, seasonal, result))
            st.session_state.pop(f"saved_pdf_{run.run_id}", None)
            st.success("The current report snapshot is saved with this forecast execution.")
    with st.expander("Inspect the aggregate analytics sent to Gemini"):
        st.json(context)


def reports_downloads(settings, database) -> None:
    page_heading("Reports & Downloads", "Export current historical analysis, measured model results and future forecasts.")
    dataset = require_dataset()
    if dataset is None:
        return
    evaluation, run = active_evaluation(dataset), active_forecast(dataset)
    context = build_report_context(dataset, evaluation, run, st.session_state.get("seasonal"), st.session_state.get("insights"))
    st.write("The report includes dataset provenance, quality/audit operations, observed statistics and charts, seasonal decomposition when computed, model methodology and actual holdout results, selected-model forecasts, and current business explanations.")
    if evaluation is None or run is None:
        st.info("Current exports will contain available outputs. Train models and generate a future forecast for a complete modelling report.")
    if st.button("Build report package", type="primary"):
        with st.spinner("Rendering charts, workbook and illustrated PDF…"):
            csv = export_csv(run.predictions if run else dataset.observed_series.rename_axis("date").reset_index())
            excel = generate_excel_report(context)
            pdf = generate_pdf_report(context)
        st.session_state["report_package"] = {"csv": csv, "excel": excel, "pdf": pdf, "dataset_fingerprint": dataset.fingerprint}
        st.success("CSV, Excel workbook and illustrated PDF generated from actual current outputs.")
    package = st.session_state.get("report_package")
    if package and package["dataset_fingerprint"] == dataset.fingerprint:
        columns = st.columns(3)
        columns[0].download_button("Download CSV", package["csv"], "salesvision_forecast.csv" if run else "salesvision_history.csv", "text/csv", on_click="ignore")
        columns[1].download_button("Download Excel report", package["excel"], "salesvision_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")
        columns[2].download_button("Download PDF report", package["pdf"], "salesvision_report.pdf", "application/pdf", on_click="ignore")


def forecast_history(settings, database) -> None:
    page_heading("Forecast History", "Review saved executions, reproduce report snapshots and compare overlapping forecasts.")
    if database is None:
        st.info("Forecast history is unavailable until its database path is corrected. Current analysis and downloads remain available.")
        return
    dataset = st.session_state.get("processed")
    current_only = persistent_widget("checkbox", "Only show the current dataset configuration", value=False, disabled=dataset is None, key="history_current_only")
    history = database.list_forecasts(dataset.fingerprint if dataset is not None and current_only else None)
    if history.empty:
        st.info("No forecast executions have been saved yet. Generate a forecast with history saving enabled.")
        return
    st.dataframe(history.drop(columns=["Fingerprint"]), use_container_width=True, hide_index=True)
    ids = history["Run ID"].tolist()
    labels = {row["Run ID"]: f"{row['Model']} · {row['Dataset']} · {row['Executed (UTC)'][:19]} · {row['Run ID'][:8]}" for _, row in history.iterrows()}
    selected = persistent_widget("selectbox", "View previous forecast", ids, format_func=lambda value: labels[value], key="history_selection")
    run, report = database.get_forecast(selected), database.get_report(selected)
    if run.metadata.get("kind") == "synthetic":
        st.warning("This saved execution used explicitly synthetic test data.")
    render_chart(forecast_chart(report.history, run.predictions, run.model, run.configuration["interval_width"]), "history_forecast")
    st.dataframe(run.predictions, use_container_width=True, hide_index=True)
    with st.expander("Saved metadata and evaluation metrics"):
        st.json({"execution": run.created_at, "model": run.model, "frequency": run.frequency, "horizon": run.horizon, "configuration": run.configuration, "evaluation_metrics": run.evaluation_metrics, "training_observations": run.training_observations})
    left, right = st.columns(2)
    left.download_button("Download saved forecast CSV", export_csv(run.predictions), "saved_forecast.csv", "text/csv", on_click="ignore")
    if right.button("Generate saved report PDF"):
        st.session_state[f"saved_pdf_{selected}"] = generate_pdf_report(report)
    if st.session_state.get(f"saved_pdf_{selected}"):
        st.download_button("Download saved report PDF", st.session_state[f"saved_pdf_{selected}"], "saved_salesvision_report.pdf", "application/pdf", on_click="ignore")
    comparisons = persistent_widget("multiselect", "Compare saved forecasts", ids, format_func=lambda value: labels[value], max_selections=4, key="history_comparisons")
    if len(comparisons) >= 2:
        try:
            table = database.compare_forecasts(comparisons)
            figure = px.line(table, x="date", y="prediction", color="execution", markers=True, title="Saved forecasts · common future dates")
            render_chart(figure, "history_comparison")
            st.dataframe(table, use_container_width=True, hide_index=True)
        except SalesVisionError as exc:
            st.info(str(exc))
    with st.expander("Delete selected execution"):
        confirmed = st.checkbox(f"Confirm deletion of forecast {selected[:8]} and its saved points/report", key=f"delete_confirm_{selected}")
        if st.button("Delete selected forecast", disabled=not confirmed):
            database.delete_forecast(selected, confirmed=True)
            st.session_state.pop(f"saved_pdf_{selected}", None)
            st.rerun()


def about_project(settings, database) -> None:
    page_heading("About SalesVision", "Sales Forecasting Using Python · Business Analytics · Academic minor project")
    st.markdown("### Problem and proposed system\nBusinesses need reproducible forecasts from historical sales to support planning. SalesVision combines explicit data-quality policies, exploratory and seasonal analysis, genuine ARIMA/SARIMAX/Prophet models, chronological evaluation and explainable future predictions in one Python application.")
    st.markdown("### Implemented methodology\n1. Load and inspect real CSV/Excel data.\n2. Map date/sales columns, select filters, audit cleaning and preserve unknown observations.\n3. Aggregate at the native or coarser frequency.\n4. Explore observed sales and perform seasonal decomposition when enough history exists.\n5. Compare models on one chronological holdout; optionally backtest within training history.\n6. Select by an explicitly chosen metric and retrain on full history.\n7. Forecast future sales and model intervals; use Gemini only to explain computed evidence.\n8. Export reports and store aggregate execution snapshots in SQLite.")
    st.code("CSV / Excel → Validation → Audited cleaning → Period aggregation\n             → EDA / STL → Chronological model evaluation\n             → Full-history selected model → Future forecast\n             → Plotly dashboard / PDF / Excel / SQLite history\n             → Optional Gemini explanations of aggregate analytics", language="text")
    st.markdown("### Technology stack\nPython 3.12, Pandas, NumPy, Statsmodels, official Prophet, Matplotlib, Plotly, Streamlit, SQLite, ReportLab, OpenPyXL, official google-genai SDK and Pytest.")
    st.markdown("### Dataset and interpretation\nWalmart competition files are supported through manual upload and validated ancillary joins. The included UCI Online Retail daily sample contains actual recorded GBP net revenue under CC BY 4.0. The separate synthetic weekly fixture is clearly labeled. Scores measure only the displayed holdout; intervals and business explanations are not guarantees.")
    st.markdown("### Reproducibility\nReusable services are shared with the Colab notebook and experiment scripts. Seeds, parameters, data fingerprints, actual fitting duration and preprocessing policies are recorded. No live Colab session is required to run this application.")


def render_page(page: str, settings, database) -> None:
    handlers = dict(zip(NAVIGATION, [overview, dataset_management, preprocessing, eda, seasonal_analysis, model_training, model_comparison, sales_forecasting, gemini_insights, reports_downloads, forecast_history, about_project]))
    handlers[page](settings, database)
