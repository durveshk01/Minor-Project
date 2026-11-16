"""Accessible dashboard cards, dataset labels and model-configuration controls."""

from copy import deepcopy
from html import escape

import pandas as pd
import streamlit as st

from src.data.aggregator import frequency_family
from src.features.time_features import default_seasonal_period
from src.forecasting.types import ModelConfig


def navigate(page: str) -> None:
    st.session_state["navigation"] = page


def page_heading(title: str, description: str) -> None:
    st.markdown(f'<div class="sv-page-heading"><h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def compact_number(value: float | None, currency: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = {"GBP": "£", "USD": "$", "EUR": "€", "INR": "₹"}.get(currency, "")
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.2f}M"
    if absolute >= 10_000:
        return f"{prefix}{value / 1_000:,.2f}K"
    return f"{prefix}{value:,.2f}"


def kpi_card(label: str, value: str, detail: str, category: str = "Historical") -> None:
    with st.container(border=True):
        st.caption(category.upper())
        st.metric(label, value)
        st.markdown(f'<div class="sv-card-detail">{escape(detail)}</div>', unsafe_allow_html=True)


def dataset_badge(dataset) -> None:
    kind = dataset.metadata.get("kind", "user-provided")
    if kind == "synthetic":
        st.warning("SYNTHETIC TEST DATA · Software demonstration only; these are not Walmart or empirical industry findings.")
    st.caption(f"{dataset.metadata.get('name', 'User dataset')}  |  {dataset.frequency}  |  {len(dataset.series):,} calendar periods  |  {int(dataset.observed_mask.sum()):,} observed  |  ID {dataset.fingerprint[:12]}")
    if dataset.metadata.get("aggregation") == "mean":
        st.info("The forecast target is average record sales per period. Sums of period averages are not total transaction revenue.")


def require_dataset():
    dataset = st.session_state.get("processed")
    if dataset is None:
        st.info("Load a CSV or Excel dataset, then apply preprocessing to create an analysis-ready sales series.")
        st.button("Open Dataset Management", on_click=navigate, args=("Dataset Management",), type="primary")
        return None
    dataset_badge(dataset)
    return dataset


def render_chart(figure, key: str) -> None:
    st.plotly_chart(figure, use_container_width=True, key=key, config={"displaylogo": False, "responsive": True})


def remember_controls(key: str | None = None) -> None:
    """Copy submitted widget values into non-widget state before Streamlit cleanup."""
    values = st.session_state.setdefault("control_values", {})
    for name in [key] if key is not None else list(values):
        if name in st.session_state:
            values[name] = deepcopy(st.session_state[name])


def persistent_widget(kind: str, label: str, *args, key: str, container=None, in_form: bool = False, **kwargs):
    """Restore controls without stale dynamic defaults or forbidden form callbacks."""
    values = st.session_state.setdefault("control_values", {})
    if key in values:
        remembered = deepcopy(values[key])
        if kind in {"selectbox", "radio", "multiselect"}:
            options = list(args[0] if args else kwargs["options"])
            if kind == "multiselect":
                kwargs["default"] = [value for value in remembered if value in options]
            elif remembered in options:
                kwargs["index"] = options.index(remembered)
        elif kind == "number_input":
            if remembered is not None and kwargs.get("min_value", remembered) <= remembered <= kwargs.get("max_value", remembered):
                kwargs["value"] = remembered
        else:
            kwargs["value"] = remembered
    if not in_form:
        kwargs.update({"on_change": remember_controls, "args": (key,)})
    result = getattr(container if container is not None else st, kind)(label, *args, key=key, **kwargs)
    values[key] = deepcopy(result)
    return result


def model_controls(name: str, dataset, prefix: str, base: ModelConfig | None = None, in_form: bool = False) -> ModelConfig:
    base = base or ModelConfig(name, seasonal_order=(0, 1, 1, default_seasonal_period(dataset.frequency)))
    key = f"{prefix}_{dataset.fingerprint[:12]}_{name}"
    st.markdown(f"**{name} configuration**")
    order = base.order
    seasonal = base.seasonal_order
    yearly, weekly = base.yearly_seasonality, base.weekly_seasonality
    mode, prior, use_holidays = base.seasonality_mode, base.changepoint_prior_scale, base.use_holidays
    regressors = base.regressor_columns
    maxiter = base.maxiter
    if name in {"ARIMA", "SARIMA"}:
        columns = st.columns(3)
        order = tuple(int(persistent_widget("number_input", label, container=column, min_value=0, max_value=5, value=value, key=f"{key}_{label}", in_form=in_form)) for column, label, value in zip(columns, ["p", "d", "q"], base.order))
        maxiter = int(persistent_widget("number_input", "Optimizer iterations", min_value=20, max_value=1000, value=base.maxiter, step=25, key=f"{key}_maxiter", in_form=in_form))
    if name == "SARIMA":
        columns = st.columns(4)
        seasonal = tuple(int(persistent_widget("number_input", label, container=column, min_value=2 if label == "s" else 0, max_value=366 if label == "s" else 3, value=value, key=f"{key}_{label}", in_form=in_form)) for column, label, value in zip(columns, ["P", "D", "Q", "s"], base.seasonal_order))
        st.caption(f"Seasonal terms require at least {2 * seasonal[3]} training observations. Large seasonal periods increase fitting cost.")
    if name == "Prophet":
        mapping = {"Auto (data-dependent)": None, "Off": False, "On": True}
        inverse = {value: label for label, value in mapping.items()}
        columns = st.columns(2)
        yearly = mapping[persistent_widget("selectbox", "Yearly seasonality", list(mapping), container=columns[0], index=list(mapping).index(inverse[base.yearly_seasonality]), key=f"{key}_yearly", in_form=in_form)]
        weekly = mapping[persistent_widget("selectbox", "Within-week seasonality", list(mapping), container=columns[1], index=list(mapping).index(inverse[base.weekly_seasonality]), key=f"{key}_weekly", in_form=in_form)]
        mode = persistent_widget("selectbox", "Seasonality mode", ["additive", "multiplicative"], index=["additive", "multiplicative"].index(base.seasonality_mode), key=f"{key}_mode", in_form=in_form)
        prior = float(persistent_widget("number_input", "Changepoint prior scale", min_value=0.001, max_value=1.0, value=base.changepoint_prior_scale, format="%.3f", key=f"{key}_prior", in_form=in_form))
        use_holidays = persistent_widget("checkbox", "Use supplied holiday calendar", value=base.use_holidays, disabled=dataset.holidays.empty, key=f"{key}_holidays", in_form=in_form)
        st.caption("Intraday seasonality is disabled for aggregated data. Yearly terms require two years; within-week terms require daily observations.")
    if name in {"SARIMA", "Prophet"} and not dataset.regressors.empty:
        regressors = tuple(persistent_widget("multiselect", "External regressors", list(dataset.regressors.columns), default=list(base.regressor_columns), key=f"{key}_regressors", in_form=in_form))
    interval = float(persistent_widget("selectbox", "Interval width", [0.80, 0.90, 0.95, 0.99], index=[0.80, 0.90, 0.95, 0.99].index(base.interval_width) if base.interval_width in [0.80, 0.90, 0.95, 0.99] else 2, format_func=lambda value: f"{value:.0%}", key=f"{key}_interval", in_form=in_form))
    return ModelConfig(name, order, seasonal, base.trend, interval, maxiter, yearly, weekly, False, mode, prior, use_holidays, regressors, base.seed)


def horizon_options(frequency: str) -> tuple[list[int], str]:
    family = frequency_family(frequency)
    return {"daily": ([7, 30, 90], "days"), "weekly": ([4, 8, 12, 26], "weeks"), "monthly": ([3, 6, 12], "months")}[family]
