"""Plotly dashboard charts and Matplotlib figures from supplied application data."""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf

COLORS = ["#2563EB", "#0D9488", "#8B5CF6", "#F59E0B", "#E05263"]


def style_chart(figure: go.Figure, title: str, y_title: str = "Sales") -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18}},
        template="plotly_white", colorway=COLORS,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": "#283A53"},
        margin={"l": 24, "r": 24, "t": 65, "b": 35},
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified", yaxis_title=y_title,
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E6EBF3", zerolinecolor="#CBD5E1")
    return figure


def historical_chart(series: pd.Series, title: str = "Historical sales") -> go.Figure:
    figure = go.Figure(go.Scatter(x=series.index, y=series.values, name="Observed sales", mode="lines", line={"color": COLORS[0], "width": 2}, connectgaps=False))
    return style_chart(figure, title)


def rolling_chart(series: pd.Series, window: int) -> go.Figure:
    figure = historical_chart(series, f"Sales and {window}-period trailing average")
    figure.add_trace(go.Scatter(x=series.index, y=series.rolling(window, min_periods=window).mean(), name="Trailing average", line={"color": COLORS[1], "width": 3}))
    return figure


def forecast_chart(history: pd.Series, forecast: pd.DataFrame, model: str, interval_width: float = 0.95) -> go.Figure:
    figure = historical_chart(history, f"{model} · future sales forecast")
    if {"lower", "upper"}.issubset(forecast) and forecast[["lower", "upper"]].notna().all().all():
        figure.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper"], mode="lines", line={"width": 0}, name="Upper interval", showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(37,99,235,0.14)", name=f"{interval_width:.0%} model interval"))
    figure.add_trace(go.Scatter(x=forecast["date"], y=forecast["prediction"], name="Raw model forecast", mode="lines+markers", line={"color": COLORS[1], "width": 3, "dash": "dash"}, marker={"size": 5}))
    if "business_prediction" in forecast:
        figure.add_trace(go.Scatter(x=forecast["date"], y=forecast["business_prediction"], name="Business constraint: floor at zero", line={"color": COLORS[3], "dash": "dot"}))
    return figure


def evaluation_chart(predictions: dict[str, pd.DataFrame]) -> go.Figure:
    figure = go.Figure()
    if predictions:
        actual = next(iter(predictions.values()))
        figure.add_trace(go.Scatter(x=actual["date"], y=actual["actual"], name="Actual observed sales", line={"color": "#172338", "width": 3}, connectgaps=False))
        for index, (name, table) in enumerate(predictions.items()):
            figure.add_trace(go.Scatter(x=table["date"], y=table["prediction"], name=name, line={"color": COLORS[index % len(COLORS)], "width": 2, "dash": "dash"}))
    return style_chart(figure, "Chronological holdout · actual versus predicted")


def comparison_chart(table: pd.DataFrame, metric: str = "RMSE") -> go.Figure:
    successful = table.loc[table["Status"].eq("success")].copy()
    figure = px.bar(successful, x="Model", y=metric, color="Model", color_discrete_sequence=COLORS)
    figure.update_layout(showlegend=False)
    return style_chart(figure, f"Model comparison · lower {metric} is better", metric)


def decomposition_chart(components: pd.DataFrame) -> go.Figure:
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    for row, column in enumerate(["observed", "trend", "seasonal", "residual"], 1):
        figure.add_trace(go.Scatter(x=components.index, y=components[column], name=column.title(), line={"color": COLORS[(row - 1) % len(COLORS)], "width": 2}), row=row, col=1)
    style_chart(figure, "STL seasonal decomposition")
    figure.update_layout(height=720, showlegend=False)
    return figure


def distribution_chart(series: pd.Series) -> go.Figure:
    figure = px.histogram(pd.DataFrame({"sales": series.dropna()}), x="sales", nbins=30, marginal="box", color_discrete_sequence=[COLORS[0]])
    return style_chart(figure, "Distribution of observed period sales", "Periods")


def bar_chart(frame: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    figure = px.bar(frame, x=x, y=y, color=color, barmode="group", color_discrete_sequence=COLORS)
    return style_chart(figure, title, y.replace("_", " ").title())


def statistical_figure(series: pd.Series) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    values = series.dropna()
    axes[0].hist(values, bins=min(30, max(5, len(values) // 5)), color=COLORS[0], edgecolor="white")
    axes[0].set(title="Observed-sales distribution", xlabel="Sales", ylabel="Period count")
    lags = min(30, len(values) // 2 - 1)
    if lags >= 1 and values.std() > 0 and series.notna().all():
        plot_acf(series, lags=lags, ax=axes[1], color=COLORS[1], title="Autocorrelation of processed series")
    else:
        axes[1].boxplot(values)
        axes[1].set(title="Observed-sales spread", ylabel="Sales")
    return figure


def figure_png(figure: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def report_history_figure(history: pd.Series, forecast: pd.DataFrame | None = None) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 3.4), layout="constrained")
    axis.plot(history.index, history.values, color="#172338", label="Observed history", linewidth=1.5)
    if forecast is not None and not forecast.empty:
        axis.plot(forecast["date"], forecast["prediction"], color=COLORS[0], label="Raw forecast", linewidth=2)
        if forecast[["lower", "upper"]].notna().all().all():
            axis.fill_between(forecast["date"], forecast["lower"], forecast["upper"], color=COLORS[0], alpha=0.15, label="Model interval")
    axis.set(xlabel="Date", ylabel="Sales", title="Historical sales and model forecast" if forecast is not None else "Historical sales")
    axis.grid(alpha=0.15)
    axis.legend(loc="best")
    return figure_png(figure)


def report_comparison_figure(table: pd.DataFrame) -> bytes:
    successful = table.loc[table["Status"].eq("success")]
    figure, axis = plt.subplots(figsize=(8, 3), layout="constrained")
    axis.bar(successful["Model"], successful["RMSE"], color=COLORS[:len(successful)])
    axis.set(title="Actual holdout RMSE — lower is better", ylabel="RMSE")
    return figure_png(figure)


def report_evaluation_figure(predictions: dict[str, pd.DataFrame]) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 3.4), layout="constrained")
    if predictions:
        actual = next(iter(predictions.values()))
        axis.plot(actual["date"], actual["actual"], color="#172338", label="Actual observed sales", linewidth=2)
        for index, (name, frame) in enumerate(predictions.items()):
            axis.plot(frame["date"], frame["prediction"], color=COLORS[index % len(COLORS)], label=name, linestyle="--")
    axis.set(title="Common chronological holdout: actual versus predicted", ylabel="Sales", xlabel="Date")
    axis.legend(loc="best")
    return figure_png(figure)


def report_decomposition_figure(components: pd.DataFrame) -> bytes:
    figure, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for index, column in enumerate(["observed", "trend", "seasonal", "residual"]):
        axes[index].plot(components.index, components[column], color=COLORS[index % len(COLORS)], linewidth=1)
        axes[index].set_ylabel(column.title())
    return figure_png(figure)
