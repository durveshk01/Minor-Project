"""Exercise the actual Streamlit server in Chrome, including uploads and responsive views."""

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import expect, sync_playwright
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADINGS = {
    "Overview Dashboard": "Sales intelligence, in focus",
    "Dataset Management": "Dataset Management",
    "Data Preprocessing": "Data Preprocessing",
    "Exploratory Data Analysis": "Exploratory Data Analysis",
    "Seasonal Analysis": "Seasonal Analysis",
    "Model Training": "Model Training",
    "Model Comparison": "Model Comparison",
    "Sales Forecasting": "Sales Forecasting",
    "Gemini AI Insights": "Gemini AI Insights",
    "Reports & Downloads": "Reports & Downloads",
    "Forecast History": "Forecast History",
    "About Project": "About SalesVision",
}


def ready(page) -> None:
    page.wait_for_timeout(150)
    page.wait_for_function("(() => {const app=document.querySelector('[data-testid=stApp]'); return app && app.getAttribute('data-test-script-state') !== 'running' && document.querySelectorAll('[data-testid=stSpinner]').length === 0 && document.querySelectorAll('[data-testid=stMain] [data-stale=true]').length === 0;})()", timeout=180_000)
    assert page.get_by_test_id("stException").count() == 0
    assert page.get_by_text("This operation could not be completed.", exact=False).count() == 0


def navigate(page, name: str, mobile: bool = False) -> None:
    control = page.locator('[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"]')
    if control.count() and control.first.is_visible():
        control.first.click()
    page.get_by_test_id("stSidebar").get_by_text(name, exact=True).click()
    page.locator(".sv-page-heading h1").filter(has_text=HEADINGS[name]).wait_for(timeout=180_000)
    ready(page)
    (ROOT / "outputs/qa/browser-state.html").write_text(page.content(), encoding="utf-8")
    if mobile:
        close = page.get_by_test_id("stSidebarCollapseButton")
        if close.count() and close.first.is_visible():
            close.first.click()


def select(page, label: str, value: str) -> None:
    box = page.get_by_test_id("stSelectbox").filter(has_text=label)
    box.locator('[data-baseweb="select"]').click()
    page.get_by_role("option", name=value, exact=True).click()
    ready(page)


def multiselect(page, label: str, values: list[str]) -> None:
    """Exercise the real multi-select keyboard interactions rather than injecting session state."""
    box = page.get_by_test_id("stMultiSelect").filter(has_text=label)
    for _ in range(12):
        if not box.locator('[data-baseweb="tag"]').count():
            break
        box.get_by_role("combobox").fill("")
        box.get_by_role("combobox").press("Backspace")
        box.get_by_role("combobox").press("Escape")
        ready(page)
    assert not box.locator('[data-baseweb="tag"]').count(), f"Could not clear {label}."
    for value in values:
        box.get_by_role("combobox").fill(value)
        page.get_by_role("option", name=value, exact=True).click()
        ready(page)


def download(page, label: str, target: Path) -> Path:
    ready(page)
    with page.expect_download(timeout=60_000) as received:
        page.get_by_role("button", name=label, exact=True).click()
    received.value.save_as(target)
    assert target.stat().st_size > 0
    return target


def number(page, label: str, value, scope=None) -> None:
    root = scope if scope is not None else page
    box = root.get_by_test_id("stNumberInput").filter(has=page.get_by_text(label, exact=True))
    box.get_by_role("spinbutton").fill(str(value))
    box.get_by_role("spinbutton").press("Tab")
    ready(page)


def checkbox(page, label: str, checked: bool) -> None:
    box = page.get_by_test_id("stCheckbox").filter(has_text=label)
    if box.get_by_role("checkbox").is_checked() != checked:
        box.locator('label[data-baseweb="checkbox"]').click()
        ready(page)
    assert box.get_by_role("checkbox").is_checked() == checked


def radio(page, label: str, value: str) -> None:
    page.get_by_test_id("stRadio").filter(has_text=label).get_by_text(value, exact=True).click()
    ready(page)


def text_input(page, label: str, value: str) -> None:
    box = page.get_by_test_id("stTextInput").filter(has_text=label)
    box.get_by_role("textbox").fill(value)
    box.get_by_role("textbox").press("Tab")
    ready(page)


def open_expander(page, label: str) -> None:
    box = page.get_by_test_id("stExpander").filter(has=page.get_by_text(label, exact=True))
    if box.get_attribute("open") is None:
        box.locator("summary").click()


def choices(page, label: str) -> list[str]:
    box = page.get_by_test_id("stSelectbox").filter(has_text=label)
    box.locator('[data-baseweb="select"]').click()
    values = page.get_by_role("option").all_text_contents()
    page.keyboard.press("Escape")
    return values


def inspect_charts(page, minimum: int = 1, interactive: bool = False) -> int:
    """Read rendered Plotly traces and operate its actual hover/pan/zoom controls."""
    ready(page)
    page.wait_for_function("minimum => Array.from(document.querySelectorAll('[data-testid=stPlotlyChart]')).filter(node => node.getClientRects().length && getComputedStyle(node).visibility !== 'hidden').length >= minimum", arg=minimum)
    visible = [chart for chart in page.get_by_test_id("stPlotlyChart").all() if chart.is_visible()]
    assert len(visible) >= minimum, f"Expected {minimum} visible charts; found {len(visible)}."
    for chart in visible:
        plot = chart.locator(".js-plotly-plot")
        plot.wait_for()
        page.wait_for_function("plot => plot.data && plot.data.length && plot._fullLayout && plot._fullLayout.width > 0", arg=plot.element_handle())
        state = plot.evaluate("plot => ({traces:plot.data.length,width:plot._fullLayout.width})")
        assert state["traces"] > 0 and state["width"] > 0
    if interactive:
        chart = visible[0]
        drag = chart.locator(".nsewdrag").first
        drag.scroll_into_view_if_needed()
        bounds = drag.bounding_box()
        drag.hover(position={"x": bounds["width"] * 0.5, "y": bounds["height"] * 0.5})
        chart.locator(".hoverlayer text").first.wait_for()
        for title, mode in [("Pan", "pan"), ("Zoom", "zoom")]:
            chart.locator(f'.modebar-btn[data-title="{title}"]').click()
            assert chart.locator(".js-plotly-plot").evaluate("plot => plot._fullLayout.dragmode") == mode
        chart.locator('.modebar-btn[data-title="Reset axes"]').click()
    return len(visible)


@contextmanager
def browser_runtime(current_page, qa: Path):
    """Capture real failure artifacts before Playwright tears down its transport."""
    with sync_playwright() as playwright:
        try:
            yield playwright
        except Exception:
            page = current_page()
            if page is not None:
                page.screenshot(path=str(qa / "browser-failure.png"), full_page=True)
                (qa / "browser-failure.html").write_text(page.content(), encoding="utf-8")
            raise


@contextmanager
def browser_scenario(browser, url: str, qa: Path, name: str):
    """Isolate a conditional workflow and preserve the failing page if it fails."""
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_default_timeout(30_000)
    page.goto(url, wait_until="domcontentloaded")
    page.locator(".sv-page-heading h1").wait_for()
    try:
        yield page
        assert not errors, errors
    except Exception:
        page.screenshot(path=str(qa / f"{name}-failure.png"), full_page=True)
        (qa / f"{name}-failure.html").write_text(page.content(), encoding="utf-8")
        raise
    finally:
        context.close()


def upload_history(page, path: Path) -> None:
    navigate(page, "Dataset Management")
    page.get_by_role("tab", name="Upload CSV / Excel", exact=True).click()
    page.get_by_test_id("stFileUploader").filter(has_text="Historical sales dataset").locator('input[type="file"]').set_input_files(str(path))
    page.get_by_role("button", name="Use uploaded dataset", exact=True).wait_for()
    ready(page)


def compare_history(page, identifiers: list[str]) -> None:
    multiselect(page, "Compare saved forecasts", [])
    box = page.get_by_test_id("stMultiSelect").filter(has_text="Compare saved forecasts")
    for identifier in identifiers:
        box.get_by_role("combobox").fill(identifier[:8])
        page.get_by_role("option").filter(has_text=identifier[:8]).click()
        ready(page)


def exercise_weekly_monthly(browser, url: str, qa: Path, database, passed) -> None:
    """Verify synthetic weekly joins/drivers/holidays and genuine monthly forecast controls."""
    weekly = pd.read_csv(ROOT / "data/sample/synthetic_weekly_sales.csv", parse_dates=["Date"])
    stores_path, features_path = qa / "audit-stores.csv", qa / "audit-features.csv"
    pd.DataFrame({"Store": [1, 2], "Category": ["Food", "Home"], "Region": ["North", "South"], "Size": [1000, 1500]}).to_csv(stores_path, index=False)
    features = weekly[["Store", "Date", "IsHoliday"]].drop_duplicates().copy()
    features["IsHoliday"] = features["IsHoliday"].map({True: " yes ", False: "no"})
    features["PlannedPromotion"] = 1 + 0.2 * np.sin(np.arange(len(features)) / 4)
    features.to_csv(features_path, index=False)
    with browser_scenario(browser, url, qa, "weekly-monthly") as page:
        upload_history(page, ROOT / "data/sample/synthetic_weekly_sales.csv")
        open_expander(page, "Optional Walmart ancillary files")
        for label, path in [("stores.csv", stores_path), ("features.csv", features_path)]:
            page.get_by_test_id("stFileUploader").filter(has_text=label).locator('input[type="file"]').set_input_files(str(path))
            ready(page)
        page.get_by_role("button", name="Use uploaded dataset", exact=True).click()
        page.get_by_text("Activated 1,248 uploaded records.", exact=True).wait_for()
        page.get_by_text("SYNTHETIC TEST DATA · This source is explicitly simulated", exact=False).wait_for()
        page.get_by_text("The active joined-table fingerprint is separate", exact=False).wait_for()
        passed("synthetic weekly ancillary joins preserve cardinality, holiday meaning and original provenance")
        page.get_by_role("button", name="Continue to Data Preprocessing", exact=True).click()
        ready(page)
        assert choices(page, "Forecasting frequency") == ["Weekly", "Monthly"]
        open_expander(page, "Store, department, category and region filters")
        for label, values in [("Store · Store", ["1"]), ("Department · Dept", ["1"]), ("Category · Category", ["Food"]), ("Region · Region", ["North"])]:
            multiselect(page, label, values)
        multiselect(page, "Store · Store", ["2"])
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("No records match the selected filters.", exact=True).wait_for()
        multiselect(page, "Store · Store", ["1"])
        multiselect(page, "Optional numeric regressors to aggregate", ["PlannedPromotion"])
        select(page, "Week-ending day", "W-SUN")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 208 records into 208 W-SUN periods.", exact=True).wait_for()
        shifted = pd.read_csv(download(page, "Download processed CSV", qa / "browser-weekly-sunday.csv"), parse_dates=["date"])
        assert shifted["date"].dt.dayofweek.eq(6).all()
        select(page, "Week-ending day", "W-FRI")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 208 records into 208 W-FRI periods.", exact=True).wait_for()
        processed = pd.read_csv(download(page, "Download processed CSV", qa / "browser-weekly-friday.csv"), parse_dates=["date"])
        reference = weekly.loc[weekly["Store"].eq(1) & weekly["Dept"].eq(1)]
        np.testing.assert_allclose(processed["observed_sales"], reference["Weekly_Sales"])
        assert processed["date"].dt.dayofweek.eq(4).all() and processed["is_observed"].all()
        passed("four business filters, empty-selection rejection, weekly anchors and finer-frequency prevention")
        navigate(page, "Exploratory Data Analysis")
        page.get_by_role("tab", name="Segments & holidays", exact=True).click()
        assert inspect_charts(page, minimum=5) >= 5
        for role in ["store", "department", "category", "region"]:
            page.get_by_text(f"All {role} results", exact=True).click()
        page.get_by_text("Holiday comparisons describe observed association", exact=False).wait_for()
        passed("actual store, department, category, region and holiday chart/table views")
        navigate(page, "Model Training")
        multiselect(page, "Models to evaluate", ["SARIMA", "Prophet"])
        number(page, "Chronological holdout (weeks)", 8)
        open_expander(page, "SARIMA parameters")
        sarima = page.get_by_test_id("stExpander").filter(has_text="SARIMA parameters")
        for label, value in [("p", 1), ("d", 1), ("q", 1), ("P", 0), ("D", 1), ("Q", 1), ("s", 52), ("Optimizer iterations", 150)]:
            number(page, label, value, scope=sarima)
        sarima.get_by_test_id("stMultiSelect").filter(has_text="External regressors").get_by_role("combobox").fill("PlannedPromotion")
        page.get_by_role("option", name="PlannedPromotion", exact=True).click()
        open_expander(page, "Prophet parameters")
        select(page, "Yearly seasonality", "On")
        select(page, "Within-week seasonality", "Off")
        select(page, "Seasonality mode", "multiplicative")
        number(page, "Changepoint prior scale", 0.075)
        checkbox(page, "Use supplied holiday calendar", True)
        prophet = page.get_by_test_id("stExpander").filter(has_text="Prophet parameters")
        prophet.get_by_test_id("stMultiSelect").filter(has_text="External regressors").get_by_role("combobox").fill("PlannedPromotion")
        page.get_by_role("option", name="PlannedPromotion", exact=True).click()
        select(page, "Rolling-origin folds within the training partition", "0")
        page.get_by_role("button", name="Train and evaluate models", exact=True).click()
        page.get_by_text("External regressors may be evaluated only when", exact=False).first.wait_for()
        assert page.get_by_text("Best successful model:", exact=False).count() == 0
        passed("regressor holdout values require explicit forecast-origin knowledge confirmation")
        multiselect(page, "Models to evaluate", ["ARIMA", "SARIMA", "Prophet"])
        checkbox(page, "Configured regressor values are known at each forecast origin", True)
        select(page, "Rolling-origin folds within the training partition", "2")
        number(page, "Backtesting horizon (periods)", 4)
        page.get_by_role("button", name="Train and evaluate models", exact=True).click()
        page.get_by_text("Best successful model:", exact=False).wait_for(timeout=180_000)
        ready(page)
        navigate(page, "Model Comparison")
        page.get_by_text("Rolling-origin backtesting · training partition only", exact=True).wait_for()
        for model in ["ARIMA", "SARIMA", "Prophet"]:
            select(page, "Inspect evaluated model", model)
            holdout = pd.read_csv(download(page, "Download holdout predictions CSV", qa / f"browser-weekly-{model.lower()}-holdout.csv"), parse_dates=["date"])
            assert len(holdout) == 8 and holdout["date"].dt.dayofweek.eq(4).all()
            assert np.isfinite(holdout["prediction"]).all()
        passed("all three genuine weekly parameterized holdouts and two rolling-origin folds")
        navigate(page, "Sales Forecasting")
        assert choices(page, "Forecast horizon") == ["4 weeks", "8 weeks", "12 weeks", "26 weeks", "Custom"]
        select(page, "Forecasting model", "Prophet")
        select(page, "Forecast horizon", "8 weeks")
        page.get_by_role("button", name="Generate future forecast", exact=True).click()
        page.get_by_text("Upload the required future regressors before forecasting.", exact=True).wait_for()
        expected = pd.date_range(weekly["Date"].max(), periods=9, freq="W-FRI")[1:]
        invalid_exog, valid_exog = qa / "audit-future-invalid.csv", qa / "audit-future-valid.xlsx"
        pd.DataFrame({"Date": expected[:-1], "PlannedPromotion": 1.0}).to_csv(invalid_exog, index=False)
        with pd.ExcelWriter(valid_exog) as writer:
            pd.DataFrame({"Date": expected, "PlannedPromotion": 1 + 0.2 * np.sin(np.arange(8) / 4)}).to_excel(writer, sheet_name="Future", index=False)
        exog_uploader = page.get_by_test_id("stFileUploader").filter(has_text="Future regressors CSV / Excel").locator('input[type="file"]')
        exog_uploader.set_input_files(str(invalid_exog))
        page.get_by_text("External regressors must exactly match the training or forecast dates in chronological order.", exact=True).wait_for()
        exog_uploader.set_input_files(str(valid_exog))
        ready(page)
        select(page, "Future regressor date column", "Date")
        page.get_by_role("button", name="Generate future forecast", exact=True).click()
        page.get_by_text("Generated 8 future weeks using Prophet.", exact=False).wait_for(timeout=90_000)
        page.get_by_text("No future holiday calendar was supplied;", exact=False).wait_for()
        passed("future driver uploads reject missing periods and require real aligned values")
        invalid_holiday, valid_holiday = qa / "audit-holiday-invalid.csv", qa / "audit-holiday-valid.csv"
        pd.DataFrame({"ds": [expected[0] - pd.Timedelta(days=1)], "holiday": ["observed_holiday"]}).to_csv(invalid_holiday, index=False)
        pd.DataFrame({"ds": [expected[0]], "holiday": ["observed_holiday"]}).to_csv(valid_holiday, index=False)
        holiday_uploader = page.get_by_test_id("stFileUploader").filter(has_text="Optional future holiday calendar").locator('input[type="file"]')
        holiday_uploader.set_input_files(str(invalid_holiday))
        page.get_by_text("Future holiday dates must be valid and match forecast-calendar periods exactly.", exact=True).wait_for()
        holiday_uploader.set_input_files(str(valid_holiday))
        ready(page)
        page.get_by_role("button", name="Generate future forecast", exact=True).click()
        page.get_by_text("Generated 8 future weeks using Prophet.", exact=False).wait_for(timeout=90_000)
        future = pd.read_csv(download(page, "Download forecast CSV", qa / "browser-weekly-prophet-future.csv"), parse_dates=["date"])
        assert pd.DatetimeIndex(future["date"]).equals(expected)
        assert "component_holidays" in future and future["component_holidays"].iloc[0] != 0
        assert page.get_by_text("No future holiday calendar was supplied;", exact=False).count() == 0
        open_expander(page, "Prophet forecast components")
        open_expander(page, "Full-history model configuration and summary")
        weekly_runs = database.list_forecasts()
        latest = database.get_forecast(weekly_runs.iloc[0]["Run ID"])
        assert latest.frequency == "W-FRI" and latest.training_observations == 208
        assert latest.configuration["regressor_columns"] == ["PlannedPromotion"]
        assert latest.configuration["yearly_seasonality"] and latest.configuration["use_holidays"]
        passed("known weekly future holidays affect real Prophet components with full-history drivers")
        for model in ["SARIMA", "ARIMA"]:
            select(page, "Forecasting model", model)
            page.get_by_role("button", name="Generate future forecast", exact=True).click()
            page.get_by_text(f"Generated 8 future weeks using {model}.", exact=False).wait_for(timeout=90_000)
            table = pd.read_csv(download(page, "Download forecast CSV", qa / f"browser-weekly-{model.lower()}-future.csv"), parse_dates=["date"])
            assert pd.DatetimeIndex(table["date"]).equals(expected)
            assert np.isfinite(table[["prediction", "lower", "upper"]]).all().all()
        passed("all three genuine weekly future models preserve the selected weekly anchor")
        navigate(page, "Forecast History")
        checkbox(page, "Only show the current dataset configuration", True)
        assert page.get_by_test_id("stDataFrame").count() >= 2
        checkbox(page, "Only show the current dataset configuration", False)
        different = weekly_runs.loc[weekly_runs["Frequency"].eq("D"), "Run ID"].iloc[0]
        compare_history(page, [latest.run_id, different])
        page.get_by_text("History comparison requires the same dataset, filters, preprocessing and frequency.", exact=True).wait_for()
        passed("history dataset filter and incompatible-frequency comparisons are safe")
        navigate(page, "Data Preprocessing")
        multiselect(page, "Optional numeric regressors to aggregate", [])
        select(page, "Forecasting frequency", "Monthly")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 208 records into 48 MS periods.", exact=True).wait_for()
        monthly = pd.read_csv(download(page, "Download processed CSV", qa / "browser-monthly-processed.csv"), parse_dates=["date"])
        np.testing.assert_allclose(monthly["observed_sales"], reference.set_index("Date")["Weekly_Sales"].resample("MS").sum())
        assert monthly["date"].dt.is_month_start.all()
        navigate(page, "Sales Forecasting")
        page.get_by_text("Evaluate at least one model", exact=False).wait_for()
        passed("coarser monthly aggregation preserves totals and invalidates old evaluation/forecast")
        navigate(page, "Seasonal Analysis")
        number(page, "Seasonal cycle (observations)", 12)
        page.get_by_role("button", name="Run seasonal decomposition", exact=True).click()
        page.get_by_text("Estimated seasonal strength", exact=True).wait_for()
        inspect_charts(page)
        navigate(page, "Exploratory Data Analysis")
        page.get_by_role("tab", name="Sales trends", exact=True).click()
        page.get_by_text("Daily and weekday views require daily source observations", exact=False).wait_for()
        number(page, "Trailing average window (periods)", 3)
        inspect_charts(page, minimum=3)
        passed("monthly seasonal cycle and trailing averages omit unsupported daily views")
        navigate(page, "Model Training")
        number(page, "Chronological holdout (months)", 6)
        select(page, "Model selection metric", "MAE")
        select(page, "Rolling-origin folds within the training partition", "0")
        page.get_by_role("button", name="Train and evaluate models", exact=True).click()
        page.get_by_text("Best successful model:", exact=False).wait_for(timeout=180_000)
        ready(page)
        navigate(page, "Model Comparison")
        for model in ["ARIMA", "SARIMA", "Prophet"]:
            select(page, "Inspect evaluated model", model)
            evaluated = pd.read_csv(download(page, "Download holdout predictions CSV", qa / f"browser-monthly-{model.lower()}-holdout.csv"), parse_dates=["date"])
            assert len(evaluated) == 6 and evaluated["date"].dt.is_month_start.all()
            assert np.isfinite(evaluated["prediction"]).all()
        passed("all three genuine monthly holdouts with alternative MAE ranking")
        navigate(page, "Sales Forecasting")
        checkbox(page, "Save this execution and report snapshot to forecast history", False)
        assert choices(page, "Forecast horizon") == ["3 months", "6 months", "12 months", "Custom"]
        for model, horizon in [("ARIMA", 3), ("SARIMA", 6), ("Prophet", 12)]:
            select(page, "Forecasting model", model)
            select(page, "Forecast horizon", f"{horizon} months")
            page.get_by_role("button", name="Generate future forecast", exact=True).click()
            page.get_by_text(f"Generated {horizon} future months using {model}.", exact=False).wait_for(timeout=90_000)
            table = pd.read_csv(download(page, "Download forecast CSV", qa / f"browser-monthly-{model.lower()}-future.csv"), parse_dates=["date"])
            assert len(table) == horizon and table["date"].dt.is_month_start.all()
        passed("all three real monthly future models and 3/6/12-month preset horizons")
        select(page, "Forecasting model", "ARIMA")
        checkbox(page, "Customize model configuration", True)
        for label, value in [("p", 0), ("d", 1), ("q", 0), ("Optimizer iterations", 100)]:
            number(page, label, value)
        select(page, "Interval width", "90%")
        page.get_by_text("Evaluate the modified configuration", exact=False).wait_for()
        assert page.get_by_role("button", name="Generate future forecast", exact=True).count() == 0
        page.get_by_role("button", name="Re-evaluate modified configuration", exact=True).click()
        ready(page)
        select(page, "Forecast horizon", "Custom")
        number(page, "Custom horizon (periods)", 9)
        checkbox(page, "Add a separate nonnegative business-planning view", True)
        checkbox(page, "Save this execution and report snapshot to forecast history", True)
        page.get_by_role("button", name="Generate future forecast", exact=True).click()
        page.get_by_text("Generated 9 future months using ARIMA.", exact=False).wait_for(timeout=90_000)
        custom = pd.read_csv(download(page, "Download forecast CSV", qa / "browser-monthly-custom.csv"))
        np.testing.assert_allclose(custom["prediction"], monthly["model_input"].iloc[-1])
        np.testing.assert_allclose(custom["business_prediction"], custom["prediction"].clip(lower=0))
        np.testing.assert_allclose(custom["business_lower"], custom["lower"].clip(lower=0))
        stored = database.get_forecast(database.list_forecasts().iloc[0]["Run ID"])
        assert stored.configuration["order"] == [0, 1, 0] and stored.configuration["interval_width"] == 0.90
        assert stored.horizon == 9 and stored.training_observations == 48
        navigate(page, "Overview Dashboard")
        navigate(page, "Sales Forecasting")
        assert "Custom" in page.get_by_test_id("stSelectbox").filter(has_text="Forecast horizon").inner_text()
        assert page.get_by_test_id("stNumberInput").filter(has_text="Custom horizon (periods)").get_by_role("spinbutton").input_value() == "9"
        passed("modified configurations require re-evaluation and custom horizons preserve raw/business views")


def exercise_revenue_policies(browser, url: str, qa: Path, passed) -> None:
    """Exercise derived revenue, explicit cleaning decisions, date windows and unknown periods."""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    rows = [{"Date": date.strftime("%d/%m/%Y"), "Quantity": float(5 + index % 7), "UnitPrice": 10.0 + department, "Store": store, "Dept": department, "Category": "Food" if department == 1 else "Home", "Region": "North" if store == 1 else "South", "IsHoliday": " yes " if date.day == 19 else "no", "PlannedDriver": 1 + index % 5, "Dataset_Type": "SYNTHETIC — browser revenue-policy audit"} for index, date in enumerate(dates) if date != pd.Timestamp("2023-01-12") for store in [1, 2] for department in [1, 2]]
    raw = pd.DataFrame(rows)
    chosen = raw["Store"].eq(1) & raw["Dept"].eq(1)
    raw.loc[chosen & raw["Date"].eq("14/01/2023"), "Quantity"] = np.nan
    raw.loc[chosen & raw["Date"].eq("15/01/2023"), "Quantity"] = -3
    raw.loc[chosen & raw["Date"].eq("16/01/2023"), "Quantity"] = 0
    duplicate = raw.loc[chosen & raw["Date"].eq("13/01/2023")].iloc[[0]]
    invalid = raw.iloc[[0]].copy()
    invalid["Date"] = "not-a-date"
    raw = pd.concat([raw, duplicate, invalid], ignore_index=True)
    source = qa / "audit-revenue-policies.csv"
    raw.to_csv(source, index=False)
    with browser_scenario(browser, url, qa, "revenue-policies") as page:
        upload_history(page, source)
        page.get_by_role("button", name="Use uploaded dataset", exact=True).click()
        page.get_by_text("Activated 358 uploaded records.", exact=True).wait_for()
        navigate(page, "Data Preprocessing")
        select(page, "Date column", "Date")
        radio(page, "Sales definition", "Existing sales column")
        select(page, "Sales column", "UnitPrice")
        radio(page, "Sales definition", "Quantity × UnitPrice revenue")
        select(page, "Quantity column", "Quantity")
        select(page, "Unit price column", "UnitPrice")
        text_input(page, "Date format (optional)", "%d/%m/%Y")
        checkbox(page, "Interpret ambiguous dates day-first", True)
        select(page, "Holiday flag column", "None")
        select(page, "Holiday flag column", "IsHoliday")
        open_expander(page, "Inspect invalid/missing records and negative sales")
        open_expander(page, "Store, department, category and region filters")
        for label, values in [("Store · Store", ["1"]), ("Department · Dept", ["1"]), ("Category · Category", ["Food"]), ("Region · Region", ["North"])]:
            multiselect(page, label, values)
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("records have missing/invalid dates or invalid numeric sales", exact=False).wait_for()
        select(page, "Invalid records", "Exclude and audit")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("sales inputs are missing. Choose a missing-sales policy.", exact=False).wait_for()
        select(page, "Missing sales inputs", "Retain as unknown")
        select(page, "Exact duplicate policy", "Remove exact duplicates")
        select(page, "Negative sales", "Stop and review")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Negative sales are present. Review returns/cancellations and choose a policy.", exact=True).wait_for()
        select(page, "Negative sales", "Exclude and audit")
        select(page, "Missing time periods", "Stop and review gaps")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("3 periods are absent or have incomplete sales.", exact=True).wait_for()
        passed("date/revenue mapping and invalid, missing, negative and gap policies fail safely before explicit decisions")
        select(page, "Missing time periods", "Keep unknown")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 88 records into 90 D periods.", exact=True).wait_for()
        retained = pd.read_csv(download(page, "Download processed CSV", qa / "browser-revenue-unknown.csv"), parse_dates=["date"])
        assert retained["observed_sales"].isna().sum() == 3 and retained["model_input"].isna().sum() == 3
        navigate(page, "Model Training")
        page.get_by_text("Training needs a complete modelling calendar", exact=False).wait_for()
        navigate(page, "Seasonal Analysis")
        expect(page.get_by_role("button", name="Run seasonal decomposition", exact=True)).to_be_disabled()
        passed("unknown observed targets block model fitting and seasonal decomposition until resolved")
        navigate(page, "Data Preprocessing")
        select(page, "Missing time periods", "Zero (explicit business assumption)")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 88 records into 90 D periods.", exact=True).wait_for()
        zeroed = pd.read_csv(download(page, "Download processed CSV", qa / "browser-revenue-zero.csv"))
        assert zeroed.loc[zeroed["observed_sales"].isna(), "model_input"].eq(0).all()
        select(page, "Missing time periods", "Forward-fill (past values only)")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 88 records into 90 D periods.", exact=True).wait_for()
        filled = pd.read_csv(download(page, "Download processed CSV", qa / "browser-revenue-filled.csv"))
        np.testing.assert_allclose(filled["model_input"], filled["observed_sales"].ffill())
        np.testing.assert_allclose(filled["observed_sales"], retained["observed_sales"], equal_nan=True)
        open_expander(page, "Excluded record audit")
        open_expander(page, "Flagged outliers — diagnostic only")
        passed("duplicate/exclusion audit and zero/causal-fill policies preserve genuine unknown actuals")
        checkbox(page, "Restrict historical date window", True)
        date_box = page.get_by_test_id("stDateInput").filter(has_text="Historical date window")
        inputs = date_box.locator("input")
        if inputs.count() == 1:
            inputs.fill("2023/01/08 – 2023/03/25")
            inputs.press("Enter")
        else:
            for control, value in zip(inputs.all(), ["2023/01/08", "2023/03/25"]):
                control.fill(value)
                control.press("Tab")
        ready(page)
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        ready(page)
        windowed = pd.read_csv(download(page, "Download processed CSV", qa / "browser-revenue-window.csv"), parse_dates=["date"])
        assert windowed["date"].iloc[0] == pd.Timestamp("2023-01-08") and windowed["date"].iloc[-1] == pd.Timestamp("2023-03-25")
        assert len(windowed) == 77 and windowed["observed_sales"].isna().sum() == 3
        passed("actual date-window control applies inclusive validated calendar bounds")
        checkbox(page, "Restrict historical date window", False)
        select(page, "Aggregation", "Average record sales (mean)")
        open_expander(page, "Store, department, category and region filters")
        multiselect(page, "Store · Store", [])
        multiselect(page, "Region · Region", [])
        multiselect(page, "Optional numeric regressors to aggregate", ["PlannedDriver"])
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        ready(page)
        averaged = pd.read_csv(download(page, "Download processed CSV", qa / "browser-revenue-mean.csv"), parse_dates=["date"])
        reference = raw.drop_duplicates().loc[raw.drop_duplicates()["Dept"].eq(1)].copy()
        reference["date"] = pd.to_datetime(reference["Date"], format="%d/%m/%Y", errors="coerce")
        reference["sales"] = reference["Quantity"] * reference["UnitPrice"]
        reference = reference.loc[reference["date"].notna() & ~reference["sales"].lt(0)]
        grouped = reference.set_index("date")["sales"].resample("D")
        expected = grouped.mean().mask(grouped.size() > grouped.count())
        np.testing.assert_allclose(averaged["observed_sales"], expected, equal_nan=True)
        navigate(page, "Overview Dashboard")
        page.get_by_text("Sum of Period Averages", exact=True).wait_for()
        navigate(page, "Reports & Downloads")
        page.get_by_role("button", name="Build report package", exact=True).click()
        page.get_by_text("CSV, Excel workbook and illustrated PDF generated", exact=False).wait_for(timeout=90_000)
        excel = download(page, "Download Excel report", qa / "browser-revenue-report.xlsx")
        pdf = download(page, "Download PDF report", qa / "browser-revenue-report.pdf")
        assert "not total transaction revenue" in " ".join("\n".join(part.extract_text() for part in PdfReader(pdf).pages).split())
        assert len(pd.ExcelFile(excel).sheet_names) >= 6
        passed("mean aggregation changes actual arithmetic and consistently labels revenue interpretation in reports")


def exercise_irregular_upload(browser, url: str, qa: Path, passed) -> None:
    source = qa / "audit-irregular.csv"
    pd.DataFrame({"Date": ["2023-01-10", "2023-03-15"], "Sales": [10, 20]}).to_csv(source, index=False)
    with browser_scenario(browser, url, qa, "irregular") as page:
        upload_history(page, source)
        page.get_by_role("button", name="Use uploaded dataset", exact=True).click()
        page.get_by_text("Activated 2 uploaded records.", exact=True).wait_for()
        navigate(page, "Data Preprocessing")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Select the native source frequency explicitly when it cannot be inferred.", exact=True).wait_for()
        select(page, "Native source frequency", "Monthly")
        assert choices(page, "Forecasting frequency") == ["Monthly"]
        select(page, "Missing time periods", "Forward-fill (past values only)")
        page.get_by_role("button", name="Apply preprocessing", exact=True).click()
        page.get_by_text("Processed 2 records into 3 MS periods.", exact=True).wait_for()
        output = pd.read_csv(download(page, "Download processed CSV", qa / "browser-irregular.csv"))
        np.testing.assert_allclose(output["model_input"], [10, 10, 20])
        assert output["observed_sales"].isna().sum() == 1
        passed("irregular uploads require an explicit native frequency and preserve absent months")


def main() -> None:
    from src.ai.insight_generator import TOPICS
    from src.database.db_manager import DatabaseManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", action="store_true")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "msedge"])
    parser.add_argument("--scenario", default="full", choices=["full", "revenue", "irregular"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qa/browser-results.json")
    args = parser.parse_args()
    qa = ROOT / "outputs/qa"
    screenshots = ROOT / "docs/screenshots"
    qa.mkdir(parents=True, exist_ok=True)
    screenshots.mkdir(parents=True, exist_ok=True)
    database_path = qa / f"browser-{uuid.uuid4().hex[:8]}.sqlite3"
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    environment = os.environ.copy()
    environment["SALESVISION_DATABASE_PATH"] = str(database_path)
    if not args.gemini:
        environment["GEMINI_API_KEY"] = ""
    evidence = {"executed_at": datetime.now(timezone.utc).isoformat(), "browser": args.browser, "scope": args.scenario, "database": str(database_path.relative_to(ROOT)), "gemini_requested": args.gemini, "checks": [], "screenshots": [], "console_errors": []}
    log = (qa / "streamlit-browser.log").open("w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    page = None
    started = time.perf_counter()

    def passed(name: str, **details) -> None:
        evidence["checks"].append({"name": name, "status": "passed", **details})
        print(f"PASS {name}", flush=True)

    def capture(name: str) -> None:
        ready(page)
        page.keyboard.press("Escape")
        page.mouse.move(0, 0)
        page.evaluate("document.querySelectorAll('[data-testid=stMain], [data-testid=stAppViewContainer]').forEach(node => node.scrollTo(0, 0)); window.scrollTo(0, 0)")
        page.screenshot(path=str(screenshots / name), full_page=True, animations="disabled")
        evidence["screenshots"].append(f"docs/screenshots/{name}")

    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("The Streamlit QA server exited before startup.")
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.3)
        else:
            raise TimeoutError("The Streamlit QA server did not become healthy.")
        passed("actual Streamlit server health")
        with browser_runtime(lambda: page, qa) as playwright:
            browser = playwright.chromium.launch(channel=args.browser, headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page = context.new_page()
            page.set_default_timeout(30_000)
            page.on("pageerror", lambda error: evidence["console_errors"].append(str(error)))
            page.goto(url, wait_until="domcontentloaded")
            page.locator(".sv-page-heading h1").wait_for()
            ready(page)
            page.get_by_role("button", name="Start with a dataset", exact=True).wait_for()
            capture("00-empty-overview.png")
            passed("desktop empty state")
            if args.scenario != "full":
                focused = {"revenue": exercise_revenue_policies, "irregular": exercise_irregular_upload}[args.scenario]
                focused(browser, url, qa, passed)
                evidence["status"] = "passed"
                context.close()
                browser.close()
                return
            for workspace in HEADINGS:
                navigate(page, workspace)
            passed("all twelve actual browser workspaces handle empty state")

            navigate(page, "Dataset Management")
            select(page, "Available local datasets", "UCI real retail · uci_online_retail_daily.csv")
            page.get_by_role("button", name="Load selected dataset", exact=True).click()
            page.get_by_text("Loaded 305 records from uci_online_retail_daily.csv.", exact=True).wait_for()
            capture("02-dataset-management.png")
            passed("real UCI local CSV load and source preview")

            navigate(page, "Data Preprocessing")
            select(page, "Missing time periods", "Forward-fill (past values only)")
            page.get_by_role("button", name="Apply preprocessing", exact=True).click()
            page.get_by_text("Processed 305 records into 374 D periods.", exact=True).wait_for()
            ready(page)
            processed_path = download(page, "Download processed CSV", qa / "browser-processed.csv")
            processed = pd.read_csv(processed_path)
            assert len(processed) == 374 and processed["observed_sales"].isna().sum() == 69
            assert processed["model_input"].notna().all()
            capture("03-data-preprocessing.png")
            passed("audited causal preprocessing and actual processed download", calendar_days=374, observed_days=305, unknown_days=69)

            navigate(page, "Exploratory Data Analysis")
            page.get_by_test_id("stPlotlyChart").first.wait_for()
            number(page, "Trailing average window (periods)", 14)
            inspect_charts(page, minimum=4, interactive=True)
            page.get_by_role("tab", name="Distribution & growth", exact=True).click()
            inspect_charts(page, minimum=3)
            assert page.get_by_test_id("stImage").count() > 0
            page.get_by_role("tab", name="Segments & holidays", exact=True).click()
            page.get_by_text("This source has no store, department, category, region or holiday fields.", exact=False).wait_for()
            page.get_by_role("tab", name="Sales trends", exact=True).click()
            capture("04-exploratory-analysis.png")
            passed("real EDA charts")
            passed("all EDA tabs, trailing average, distribution, growth and real chart hover/pan/zoom")

            navigate(page, "Seasonal Analysis")
            number(page, "Seasonal cycle (observations)", 14)
            page.get_by_role("button", name="Run seasonal decomposition", exact=True).click()
            page.get_by_text("Estimated seasonal strength", exact=True).wait_for()
            number(page, "Seasonal cycle (observations)", 7)
            page.get_by_role("button", name="Run seasonal decomposition", exact=True).click()
            page.get_by_text("Estimated seasonal strength", exact=True).wait_for()
            capture("05-seasonal-analysis.png")
            passed("actual STL decomposition")
            passed("seasonal-cycle control recomputes decomposition and restores the recorded seven-day method")

            navigate(page, "Model Training")
            multiselect(page, "Models to evaluate", [])
            expect(page.get_by_role("button", name="Train and evaluate models", exact=True)).to_be_disabled()
            multiselect(page, "Models to evaluate", ["ARIMA"])
            assert page.get_by_text("SARIMA parameters", exact=True).count() == 0
            assert page.get_by_text("Prophet parameters", exact=True).count() == 0
            multiselect(page, "Models to evaluate", ["ARIMA", "SARIMA", "Prophet"])
            expect(page.get_by_role("button", name="Train and evaluate models", exact=True)).to_be_enabled()
            passed("model-selection controls react to empty and restored selections")
            number(page, "Chronological holdout (days)", 31)
            number(page, "Chronological holdout (days)", 30)
            for metric in ["MAPE", "sMAPE", "RMSE"]:
                select(page, "Model selection metric", metric)
            select(page, "Rolling-origin folds within the training partition", "3")
            page.get_by_text("Prophet parameters", exact=True).click()
            select(page, "Yearly seasonality", "Off")
            select(page, "Within-week seasonality", "On")
            training_started = time.perf_counter()
            page.get_by_role("button", name="Train and evaluate models", exact=True).click()
            page.get_by_text(re.compile("Best successful model: SARIMA")).wait_for(timeout=180_000)
            ready(page)
            passed("all three genuine holdout models and rolling-origin folds", elapsed_seconds=time.perf_counter() - training_started)
            capture("06-model-training.png")

            navigate(page, "Model Comparison")
            page.get_by_text("Rolling-origin backtesting · training partition only", exact=True).wait_for()
            capture("07-model-comparison.png")
            passed("comparison, holdout charts and backtest tables")
            for model in ["ARIMA", "SARIMA", "Prophet"]:
                select(page, "Inspect evaluated model", model)
                holdout = pd.read_csv(download(page, "Download holdout predictions CSV", qa / f"browser-{model.lower()}-holdout.csv"))
                reference = pd.read_csv(ROOT / f"outputs/experiments/uci_daily/{model.lower()}_holdout.csv")
                np.testing.assert_allclose(holdout["prediction"], reference["prediction"], rtol=1e-5, atol=1e-3)
                np.testing.assert_allclose(holdout["actual"], reference["actual"], equal_nan=True)
            open_expander(page, "Recorded model configuration and fitted summary")
            passed("all model-inspection controls and actual holdout downloads match independent experiment predictions")
            navigate(page, "Model Training")
            assert "RMSE" in page.get_by_test_id("stSelectbox").filter(has_text="Model selection metric").inner_text()
            open_expander(page, "Prophet parameters")
            yearly_control = page.get_by_test_id("stSelectbox").filter(has_text="Yearly seasonality")
            assert "Off" in yearly_control.inner_text(), yearly_control.inner_text()
            assert page.get_by_test_id("stNumberInput").filter(has_text="Chronological holdout (days)").get_by_role("spinbutton").input_value() == "30"
            navigate(page, "Data Preprocessing")
            assert "Forward-fill (past values only)" in page.get_by_test_id("stSelectbox").filter(has_text="Missing time periods").inner_text()
            navigate(page, "Exploratory Data Analysis")
            assert page.get_by_test_id("stNumberInput").filter(has_text="Trailing average window (periods)").get_by_role("spinbutton").input_value() == "14"
            passed("submitted training, preprocessing and analysis controls survive actual workspace navigation")

            navigate(page, "Sales Forecasting")
            checkbox(page, "Save this execution and report snapshot to forecast history", False)
            for horizon in [7, 90, 11]:
                select(page, "Forecast horizon", f"{horizon} days" if horizon != 11 else "Custom")
                if horizon == 11:
                    number(page, "Custom horizon (periods)", 11)
                    checkbox(page, "Add a separate nonnegative business-planning view", True)
                page.get_by_role("button", name="Generate future forecast", exact=True).click()
                page.get_by_text(f"Generated {horizon} future days using SARIMA.", exact=False).wait_for(timeout=90_000)
                table = pd.read_csv(download(page, "Download forecast CSV", qa / f"browser-daily-{horizon}.csv"), parse_dates=["date"])
                assert pd.DatetimeIndex(table["date"]).equals(pd.date_range("2011-12-10", periods=horizon, freq="D"))
                if horizon == 11:
                    np.testing.assert_allclose(table["business_prediction"], table["prediction"].clip(lower=0))
            database = DatabaseManager(database_path)
            assert database.list_forecasts().empty
            checkbox(page, "Add a separate nonnegative business-planning view", False)
            checkbox(page, "Save this execution and report snapshot to forecast history", True)
            passed("daily preset/custom horizons, business-view toggle and disabled history saving work")
            select(page, "Forecast horizon", "30 days")
            for name in ["SARIMA", "ARIMA", "Prophet", "SARIMA"]:
                select(page, "Forecasting model", name)
                page.get_by_role("button", name="Generate future forecast", exact=True).click()
                page.get_by_text(re.compile(f"Generated 30 future days using {name}\\.")).wait_for(timeout=60_000)
                ready(page)
            database = DatabaseManager(database_path)
            runs = database.list_forecasts()
            assert len(runs) == 4 and set(runs["Model"]) == {"ARIMA", "SARIMA", "Prophet"}
            assert runs["Run ID"].nunique() == 4
            sarima_ids = runs.loc[runs["Model"].eq("SARIMA"), "Run ID"].tolist()
            np.testing.assert_allclose(database.get_forecast(sarima_ids[0]).predictions["prediction"], database.get_forecast(sarima_ids[1]).predictions["prediction"])
            passed("all three future models, SQLite saves and distinct cached execution IDs")
            forecast_path = download(page, "Download forecast CSV", qa / "browser-forecast.csv")
            forecast_excel = download(page, "Download forecast Excel", qa / "browser-forecast.xlsx")
            csv_forecast = pd.read_csv(forecast_path)
            excel_forecast = pd.read_excel(forecast_excel)
            assert len(csv_forecast) == 30
            np.testing.assert_allclose(csv_forecast["prediction"], excel_forecast["prediction"])
            assert csv_forecast.iloc[0]["date"].startswith("2011-12-10")
            capture("08-sales-forecasting.png")
            passed("actual future CSV and Excel download fidelity")

            navigate(page, "Gemini AI Insights")
            assert choices(page, "Business insight topic") == list(TOPICS)
            for topic in TOPICS:
                select(page, "Business insight topic", topic)
            question = "Which evidence supports the chosen model, and what remains unknown?"
            page.get_by_test_id("stTextArea").filter(has_text="Question about current analytics").get_by_role("textbox").fill(question)
            page.get_by_test_id("stTextArea").filter(has_text="Question about current analytics").get_by_role("textbox").press("Tab")
            navigate(page, "About Project")
            navigate(page, "Gemini AI Insights")
            assert "Ask a question" in page.get_by_test_id("stSelectbox").filter(has_text="Business insight topic").inner_text()
            assert page.get_by_test_id("stTextArea").get_by_role("textbox").input_value() == question
            select(page, "Business insight topic", "Executive summary")
            open_expander(page, "Inspect the aggregate analytics sent to Gemini")
            assert page.get_by_test_id("stJson").count() > 0
            passed("all insight topics, question/model controls and aggregate inspection preserve navigation state")
            model = page.get_by_test_id("stTextInput").filter(has_text="Gemini model").get_by_role("textbox").input_value()
            if args.gemini:
                text_input(page, "Gemini model", "gemini-salesvision-audit-unavailable")
            page.get_by_role("button", name="Check Gemini connectivity", exact=True).click()
            unavailable_message = "The configured Gemini model is unavailable. Select a model supported by your API account." if args.gemini else "Gemini is not configured. Set GEMINI_API_KEY in the server environment or Streamlit secrets."
            page.get_by_text(unavailable_message, exact=True).wait_for(timeout=60_000)
            page.get_by_role("button", name="Generate business explanation", exact=True).click()
            page.get_by_text("Explanation source: Local analytics", exact=False).wait_for(timeout=90_000)
            passed("unavailable or missing Gemini configuration preserves computed local explanations")
            if args.gemini:
                text_input(page, "Gemini model", model)
            if args.gemini:
                page.get_by_role("button", name="Check Gemini connectivity", exact=True).click()
                page.get_by_text("The configured Gemini model is reachable", exact=False).wait_for(timeout=60_000)
            page.get_by_role("button", name="Generate business explanation", exact=True).click()
            page.get_by_text("Explanation source:", exact=False).wait_for(timeout=90_000)
            ready(page)
            explanation = page.get_by_text("Explanation source:", exact=False).inner_text()
            evidence["gemini_ui_source"] = "Gemini" if "source: Gemini" in explanation else "Local analytics"
            if args.gemini:
                assert evidence["gemini_ui_source"] == "Gemini", "The live browser explanation used a provider fallback."
            else:
                assert evidence["gemini_ui_source"] == "Local analytics"
            page.get_by_role("button", name="Save current explanation with forecast history", exact=True).click()
            page.get_by_text("The current report snapshot is saved", exact=False).wait_for()
            capture("09-gemini-insights.png")
            passed("actual Gemini UI explanation and saved report snapshot", source=evidence["gemini_ui_source"])

            navigate(page, "Reports & Downloads")
            page.get_by_role("button", name="Build report package", exact=True).click()
            page.get_by_text("CSV, Excel workbook and illustrated PDF generated", exact=False).wait_for(timeout=90_000)
            report_csv = download(page, "Download CSV", qa / "browser-report.csv")
            report_excel = download(page, "Download Excel report", qa / "browser-report.xlsx")
            report_pdf = download(page, "Download PDF report", qa / "browser-report.pdf")
            comparison = pd.read_excel(report_excel, sheet_name="Model Comparison")
            assert set(comparison["Model"]) == {"ARIMA", "SARIMA", "Prophet"} and comparison["Status"].eq("success").all()
            reference = pd.read_csv(ROOT / "outputs/experiments/uci_daily/model_comparison.csv").set_index("Model")
            np.testing.assert_allclose(comparison.set_index("Model").loc[reference.index, ["MAE", "RMSE", "MAPE", "sMAPE"]], reference[["MAE", "RMSE", "MAPE", "sMAPE"]], rtol=1e-5, atol=1e-3)
            np.testing.assert_allclose(pd.read_csv(report_csv)["prediction"], csv_forecast["prediction"])
            pdf = PdfReader(report_pdf)
            assert len(pdf.pages) >= 5 and "SalesVision" in pdf.pages[0].extract_text()
            capture("10-reports-downloads.png")
            passed("browser report downloads, PDF parsing and matching script/notebook/UI metrics", pdf_pages=len(pdf.pages))

            navigate(page, "Forecast History")
            page.get_by_role("button", name="Generate saved report PDF", exact=True).click()
            page.get_by_role("button", name="Download saved report PDF", exact=True).wait_for(timeout=90_000)
            saved = download(page, "Download saved report PDF", qa / "browser-saved-report.pdf")
            assert len(PdfReader(saved).pages) >= 5
            saved_csv = pd.read_csv(download(page, "Download saved forecast CSV", qa / "browser-saved-forecast.csv"))
            np.testing.assert_allclose(saved_csv["prediction"], csv_forecast["prediction"])
            navigate(page, "Gemini AI Insights")
            page.get_by_role("button", name="Save current explanation with forecast history", exact=True).click()
            page.get_by_text("The current report snapshot is saved", exact=False).wait_for()
            navigate(page, "Forecast History")
            assert page.get_by_role("button", name="Download saved report PDF", exact=True).count() == 0
            page.get_by_role("button", name="Generate saved report PDF", exact=True).click()
            page.get_by_role("button", name="Download saved report PDF", exact=True).wait_for(timeout=90_000)
            refreshed_pdf = download(page, "Download saved report PDF", qa / "browser-saved-refreshed.pdf")
            assert PdfReader(refreshed_pdf).pages
            open_expander(page, "Saved metadata and evaluation metrics")
            passed("saved CSV fidelity and report re-save invalidate previously generated history PDFs")
            comparison_select = page.get_by_test_id("stMultiSelect").filter(has_text="Compare saved forecasts")
            for identifier in runs["Run ID"].iloc[:2]:
                comparison_select.get_by_role("combobox").fill(identifier[:8])
                page.get_by_role("option").filter(has_text=identifier[:8]).click()
                ready(page)
                assert identifier[:8] in comparison_select.inner_text()
            page.get_by_test_id("stPlotlyChart").nth(1).wait_for()
            capture("11-forecast-history.png")
            page.get_by_text("Delete selected execution", exact=True).click()
            assert page.get_by_role("button", name="Delete selected forecast", exact=True).is_disabled()
            confirmation = page.get_by_test_id("stCheckbox").filter(has_text="Confirm deletion of forecast")
            confirmation.locator("label").click()
            delete = page.get_by_role("button", name="Delete selected forecast", exact=True)
            expect(delete).to_be_enabled()
            delete.click()
            ready(page)
            page.wait_for_timeout(300)
            assert len(database.list_forecasts()) == 3
            passed("history view, regenerated PDF, compatible comparison and confirmed deletion")

            navigate(page, "About Project")
            capture("12-about-project.png")
            passed("About Project methodology")
            navigate(page, "Overview Dashboard")
            assert page.get_by_test_id("stMetric").count() == 8
            capture("01-overview-dashboard.png")
            passed("eight real overview KPI cards")

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(400)
            (qa / "mobile-controls.json").write_text(json.dumps(page.locator('[data-testid*="Sidebar"]').evaluate_all("nodes => nodes.map(node => ({testid:node.getAttribute('data-testid'),tag:node.tagName,aria:node.getAttribute('aria-label'),bounds:node.getBoundingClientRect().toJSON()}))"), indent=2), encoding="utf-8")
            page.screenshot(path=str(qa / "mobile-initial.png"))
            for name in HEADINGS:
                navigate(page, name, mobile=True)
                if name == "Sales Forecasting":
                    assert "30 days" in page.get_by_test_id("stSelectbox").filter(has_text="Forecast horizon").inner_text()
                    assert "Displayed forecast: SARIMA · 30 days" in page.get_by_text("Displayed forecast:", exact=False).inner_text()
                width = page.evaluate("(() => {const main=document.querySelector('[data-testid=stMain]'); return {viewport:document.documentElement.clientWidth,content:document.documentElement.scrollWidth,main_viewport:main.clientWidth,main_content:main.scrollWidth};})()")
                assert width["content"] <= width["viewport"] + 1, f"Horizontal body overflow on {name}: {width}"
                assert width["main_content"] <= width["main_viewport"] + 1, f"Horizontal main overflow on {name}: {width}"
                passed(f"mobile navigation and width: {name}", viewport=width["viewport"], content=width["content"])
                if name in {"Overview Dashboard", "Sales Forecasting", "Model Comparison"}:
                    capture({"Overview Dashboard": "13-mobile-overview.png", "Sales Forecasting": "14-mobile-forecast.png", "Model Comparison": "15-mobile-comparison.png"}[name])
            page.set_viewport_size({"width": 1440, "height": 1000})
            navigate(page, "Dataset Management")
            page.get_by_role("button", name="Load selected dataset", exact=True).click()
            page.get_by_text("Loaded 305 records from uci_online_retail_daily.csv.", exact=True).wait_for()
            navigate(page, "Overview Dashboard")
            page.get_by_role("button", name="Start with a dataset", exact=True).wait_for()
            assert page.get_by_test_id("stMetric").count() == 0
            passed("source reactivation invalidates downstream results")

            real_frame = pd.read_csv(ROOT / "data/sample/uci_online_retail_daily.csv")
            excel_upload = qa / "real_uci_upload.xlsx"
            with pd.ExcelWriter(excel_upload) as writer:
                pd.DataFrame({"Description": ["Browser worksheet-selection fixture"]}).to_excel(writer, sheet_name="Notes", index=False)
                real_frame.to_excel(writer, sheet_name="History", index=False)
            corrupted = qa / "corrupted.xlsx"
            corrupted.write_bytes(b"This is deliberately not an Excel workbook.")
            for extension, upload_path in [("CSV", ROOT / "data/sample/uci_online_retail_daily.csv"), ("Excel", excel_upload)]:
                upload_context = browser.new_context(viewport={"width": 1440, "height": 1000})
                upload_page = upload_context.new_page()
                upload_page.on("pageerror", lambda error: evidence["console_errors"].append(str(error)))
                upload_page.set_default_timeout(30_000)
                upload_page.goto(url, wait_until="domcontentloaded")
                upload_page.locator(".sv-page-heading h1").wait_for()
                navigate(upload_page, "Dataset Management")
                upload_page.get_by_role("tab", name="Upload CSV / Excel", exact=True).click()
                uploader = upload_page.get_by_test_id("stFileUploader").filter(has_text="Historical sales dataset").locator('input[type="file"]')
                if extension == "Excel":
                    uploader.set_input_files(str(corrupted))
                    upload_page.get_by_text("The uploaded Excel workbook could not be opened.", exact=True).wait_for()
                    passed("corrupt workbook is rejected gracefully")
                uploader.set_input_files(str(upload_path))
                ready(upload_page)
                if extension == "Excel":
                    select(upload_page, "Worksheet", "History")
                checkbox(upload_page, "Save the uploaded source file in data/uploads", extension == "CSV")
                upload_page.get_by_role("button", name="Use uploaded dataset", exact=True).wait_for()
                upload_page.get_by_role("button", name="Use uploaded dataset", exact=True).click()
                upload_page.get_by_text("Activated 305 uploaded records.", exact=True).wait_for()
                navigate(upload_page, "Data Preprocessing")
                text_input(upload_page, "Dataset source / description", "Actual UCI aggregate uploaded during independent browser verification")
                select(upload_page, "Currency or units", "GBP")
                select(upload_page, "Missing time periods", "Forward-fill (past values only)")
                upload_page.get_by_role("button", name="Apply preprocessing", exact=True).click()
                upload_page.get_by_text("Processed 305 records into 374 D periods.", exact=True).wait_for()
                ready(upload_page)
                passed(f"actual browser {extension} upload and processing")
                if extension == "CSV":
                    persisted = ROOT / "data/uploads" / f"{hashlib.sha256(upload_path.read_bytes()).hexdigest()[:12]}_{upload_path.name}"
                    assert persisted.read_bytes() == upload_path.read_bytes()
                upload_context.close()
            passed("worksheet selection, optional source persistence and uploaded provenance/currency controls")
            exercise_weekly_monthly(browser, url, qa, database, passed)
            exercise_revenue_policies(browser, url, qa, passed)
            exercise_irregular_upload(browser, url, qa, passed)
            evidence["model_frequency_coverage"] = {frequency: ["ARIMA", "SARIMA", "Prophet"] for frequency in ["D", "W-FRI", "MS"]}
            assert not evidence["console_errors"], evidence["console_errors"]
            passed("no browser JavaScript exceptions")
            context.close()
            browser.close()
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if page is not None:
            try:
                page.screenshot(path=str(qa / "browser-failure.png"), full_page=True)
                (qa / "browser-failure.html").write_text(page.content(), encoding="utf-8")
            except Exception:
                pass
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        evidence["duration_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps({"status": evidence["status"], "checks": len(evidence["checks"]), "screenshots": len(evidence["screenshots"]), "duration_seconds": evidence["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
