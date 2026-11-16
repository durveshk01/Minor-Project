"""Verify executed artifacts, documentation and an optionally extracted portable application."""

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ARIMA", "SARIMA", "Prophet")
METRICS = ("MAE", "RMSE", "MAPE", "sMAPE", "Bias")
API_KEY_PATTERN = re.compile(rb"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])")


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def measured_errors(actual, prediction) -> dict:
    truth = np.asarray(actual, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    require(len(truth) > 0 and truth.shape == estimate.shape, "Invalid scored-target shape.")
    require(np.isfinite(truth).all() and np.isfinite(estimate).all(), "Non-finite scored values.")
    error = estimate - truth
    nonzero = truth != 0
    symmetric_denominator = np.abs(truth) + np.abs(estimate)
    symmetric = np.zeros_like(truth)
    available = symmetric_denominator != 0
    symmetric[available] = 200 * np.abs(error[available]) / symmetric_denominator[available]
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt(np.square(error).mean()),
        "MAPE": 100 * np.abs(error[nonzero] / truth[nonzero]).mean() if nonzero.any() else None,
        "sMAPE": symmetric.mean(),
        "Bias": error.mean(),
        "Scored Observations": len(truth),
        "MAPE Observations": int(nonzero.sum()),
        "Zero Actuals": int((~nonzero).sum()),
    }


def verify_scores(row, measured: dict) -> None:
    for metric in METRICS:
        if measured[metric] is None:
            require(pd.isna(row[metric]), f"Unexpected {metric} for all-zero actuals.")
        else:
            np.testing.assert_allclose(float(row[metric]), measured[metric], rtol=1e-5, atol=1e-3)
    for count in ("Scored Observations", "MAPE Observations", "Zero Actuals"):
        require(int(row[count]) == measured[count], f"Incorrect {count}.")


def verify_prediction_table(table: pd.DataFrame, expected_dates: pd.DatetimeIndex) -> None:
    require({"date", "prediction", "lower", "upper"}.issubset(table), "Missing forecast columns.")
    require(pd.DatetimeIndex(pd.to_datetime(table["date"])).equals(expected_dates), "Forecast date mismatch.")
    values = table[["prediction", "lower", "upper"]].to_numpy(dtype=float)
    require(np.isfinite(values).all(), "Forecast values or intervals are not finite.")
    require(table["lower"].le(table["upper"]).all(), "Inverted forecast intervals.")


def verify_experiment(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "success", "Experiment did not succeed.")
    for name, item in manifest["artifacts"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Unsafe artifact path.")
        content = path.read_bytes()
        require(len(content) == item["bytes"], f"Artifact size differs: {name}")
        require(hashlib.sha256(content).hexdigest() == item["sha256"], f"Artifact checksum differs: {name}")
        require(not API_KEY_PATTERN.search(content), f"Credential-shaped value in artifact: {name}")
    source = ROOT / "data/sample/uci_online_retail_daily.csv"
    require(hashlib.sha256(source.read_bytes()).hexdigest() == manifest["data_provenance"]["dataset_file_sha256"], "Derived-source checksum mismatch.")
    calendar = pd.read_csv(directory / "processed_calendar.csv", parse_dates=["date"]).set_index("date")
    require(calendar.index.equals(pd.date_range(calendar.index[0], calendar.index[-1], freq="D")), "Incomplete daily calendar.")
    require(calendar["model_input"].notna().all(), "Unknown model inputs remain.")
    np.testing.assert_array_equal(calendar["is_observed"], calendar["observed_sales"].notna())
    recorded = pd.read_csv(source, parse_dates=["Date"]).set_index("Date")["Sales"]
    pd.testing.assert_series_equal(calendar["observed_sales"], recorded.reindex(calendar.index), check_names=False, check_freq=False)
    pd.testing.assert_series_equal(calendar["model_input"], calendar["observed_sales"].ffill(), check_names=False)
    holdout_size = manifest["holdout_size"]
    training = calendar.iloc[:-holdout_size]
    holdout = calendar.iloc[-holdout_size:]
    require(len(training) == manifest["train_size"], "Training-size mismatch.")
    require(training.index[-1] == pd.Timestamp(manifest["training_end"]), "Training end differs.")
    require(holdout.index[0] == pd.Timestamp(manifest["holdout_start"]) and holdout.index[-1] == pd.Timestamp(manifest["holdout_end"]), "Holdout boundaries differ.")
    comparison = pd.read_csv(directory / "model_comparison.csv").set_index("Model")
    require(set(comparison.index) == set(MODELS) and comparison["Status"].eq("success").all(), "Candidate fits did not all succeed.")
    folds = pd.read_csv(directory / "backtest_folds.csv", parse_dates=["Train End", "Test Start", "Test End"])
    pooled = pd.read_csv(directory / "backtest_comparison.csv").set_index("Model")
    require(folds["Status"].eq("success").all() and pooled["Status"].eq("success").all(), "Backtest failures remain.")
    require(folds["Train End"].lt(folds["Test Start"]).all() and folds["Test End"].le(training.index[-1]).all(), "Backtesting crossed an origin or final holdout boundary.")
    future_tables = {}
    for name in MODELS:
        table = pd.read_csv(directory / f"{name.lower()}_holdout.csv")
        verify_prediction_table(table, holdout.index)
        np.testing.assert_allclose(table["actual"], holdout["observed_sales"], equal_nan=True)
        np.testing.assert_array_equal(table["is_observed"], holdout["is_observed"])
        observed = table.loc[table["is_observed"]]
        verify_scores(comparison.loc[name], measured_errors(observed["actual"], observed["prediction"]))
        backtest = pd.read_csv(directory / f"{name.lower()}_backtest.csv", parse_dates=["date", "origin"])
        expected_folds = folds.loc[folds["Model"].eq(name)].set_index("Fold")
        require(set(backtest["fold"]) == set(expected_folds.index), "Backtest fold IDs differ.")
        for fold, part in backtest.groupby("fold"):
            row = expected_folds.loc[fold]
            dates = pd.date_range(row["Test Start"], row["Test End"], freq="D")
            verify_prediction_table(part, dates)
            require(part["origin"].eq(row["Train End"]).all(), "Incorrect forecast origin.")
            np.testing.assert_allclose(part["actual"], calendar.loc[dates, "observed_sales"], equal_nan=True)
            np.testing.assert_array_equal(part["is_observed"], calendar.loc[dates, "is_observed"])
            scored = part.loc[part["is_observed"]]
            verify_scores(row, measured_errors(scored["actual"], scored["prediction"]))
        scored = backtest.loc[backtest["is_observed"]]
        verify_scores(pooled.loc[name], measured_errors(scored["actual"], scored["prediction"]))
        require(int(pooled.loc[name, "Completed Folds"]) == len(expected_folds), "Completed-fold count differs.")
        run = manifest["future_forecasts"][name]
        require(run["training_observations"] == len(calendar), "Final fit did not use full history.")
        future = pd.read_csv(directory / f"{name.lower()}_future_forecast.csv")
        verify_prediction_table(future, pd.date_range(calendar.index[-1], periods=run["horizon"] + 1, freq="D")[1:])
        future_tables[name] = future
    selected = min((float(comparison.loc[name, manifest["ranking_metric"]]), name) for name in MODELS)[1]
    require(selected == manifest["best_model"], "Incorrect candidate selection.")
    baseline = pd.read_csv(directory / "baseline_comparison.csv").set_index("Baseline")
    baseline_predictions = {
        "Last-value naive": np.repeat(training["model_input"].iloc[-1], holdout_size),
        "Seven-day seasonal naive": np.resize(training["model_input"].iloc[-7:].to_numpy(), holdout_size),
    }
    observed_mask = holdout["is_observed"].to_numpy()
    for name, prediction in baseline_predictions.items():
        verify_scores(baseline.loc[name], measured_errors(holdout.loc[observed_mask, "observed_sales"], prediction[observed_mask]))
    with pd.ExcelFile(directory / "salesvision_report.xlsx") as workbook:
        workbook_comparison = pd.read_excel(workbook, sheet_name="Model Comparison").set_index("Model")
        np.testing.assert_allclose(workbook_comparison.loc[list(MODELS), list(METRICS)], comparison.loc[list(MODELS), list(METRICS)], rtol=1e-5, atol=1e-3)
        forecast_sheet = pd.read_excel(workbook, sheet_name="Forecast")
        np.testing.assert_allclose(forecast_sheet[["prediction", "lower", "upper"]], future_tables[selected][["prediction", "lower", "upper"]], rtol=1e-5, atol=1e-3)
        sheet_count = len(workbook.sheet_names)
    pdf = PdfReader(directory / "salesvision_report.pdf")
    require(len(pdf.pages) >= 5 and "SalesVision" in pdf.pages[0].extract_text(), "Illustrated report did not parse correctly.")
    statistics = json.loads((directory / "statistics.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(statistics["total"], calendar["observed_sales"].sum())
    require(statistics["observations"] == int(observed_mask.sum()) + int(training["is_observed"].sum()), "Observed statistic count differs.")
    return {
        "directory": directory.relative_to(ROOT).as_posix(),
        "status": "passed",
        "artifact_checksums": len(manifest["artifacts"]),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "calendar_periods": len(calendar),
        "observed_periods": int(calendar["is_observed"].sum()),
        "unknown_periods": int((~calendar["is_observed"]).sum()),
        "holdout_periods": holdout_size,
        "scored_holdout_targets": int(observed_mask.sum()),
        "candidate_rmse": {name: float(comparison.loc[name, "RMSE"]) for name in MODELS},
        "selected_model": selected,
        "strongest_baseline_rmse": float(baseline["RMSE"].min()),
        "selected_model_beats_strongest_baseline": bool(comparison.loc[selected, "RMSE"] < baseline["RMSE"].min()),
        "full_history_periods": len(calendar),
        "future_rows": {name: len(table) for name, table in future_tables.items()},
        "selected_raw_point_total": float(future_tables[selected]["prediction"].sum()),
        "pdf_pages": len(pdf.pages),
        "excel_sheets": sheet_count,
        "explanation_source": manifest["insights_source"],
    }


def verify_documentation(output: Path) -> dict:
    documents = [ROOT / "README.md", ROOT / "data/sample/README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    checked = 0
    for path in documents:
        text = path.read_text(encoding="utf-8")
        require(not API_KEY_PATTERN.search(text.encode()), f"Credential-shaped value in {path.name}.")
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", text):
            if urllib.parse.urlsplit(target).scheme or target.startswith("#"):
                continue
            destination = (path.parent / urllib.parse.unquote(target.split("#")[0])).resolve()
            require(destination.is_file() or destination == output.resolve(), f"Broken documentation link in {path.name}: {target}")
            checked += 1
    return {"status": "passed", "markdown_files": len(documents), "local_links": checked}


def verify_qa() -> dict:
    suites = list(ET.parse(ROOT / "outputs/qa/pytest-results.xml").getroot().iter("testsuite"))
    require(suites and all(int(suite.get("failures", 0)) + int(suite.get("errors", 0)) + int(suite.get("skipped", 0)) == 0 for suite in suites), "Pytest evidence has failures, errors or skipped checks.")
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    require(tests > 0, "No executed pytest tests were recorded.")
    browser = json.loads((ROOT / "outputs/qa/browser-results.json").read_text(encoding="utf-8"))
    require(browser["status"] == "passed" and browser.get("scope") == "full" and not browser["console_errors"] and len(browser["checks"]) >= 63 and all(check["status"] == "passed" for check in browser["checks"]), "Full browser evidence is incomplete or failed.")
    require(browser.get("model_frequency_coverage") == {frequency: list(MODELS) for frequency in ["D", "W-FRI", "MS"]}, "Browser model/frequency coverage is incomplete.")
    names = {check["name"] for check in browser["checks"]}
    require(len(names) == len(browser["checks"]), "Duplicate browser checks were recorded.")
    workspaces = ["Overview Dashboard", "Dataset Management", "Data Preprocessing", "Exploratory Data Analysis", "Seasonal Analysis", "Model Training", "Model Comparison", "Sales Forecasting", "Gemini AI Insights", "Reports & Downloads", "Forecast History", "About Project"]
    required = {
        "all twelve actual browser workspaces handle empty state",
        "all EDA tabs, trailing average, distribution, growth and real chart hover/pan/zoom",
        "model-selection controls react to empty and restored selections",
        "all model-inspection controls and actual holdout downloads match independent experiment predictions",
        "submitted training, preprocessing and analysis controls survive actual workspace navigation",
        "all three future models, SQLite saves and distinct cached execution IDs",
        "unavailable or missing Gemini configuration preserves computed local explanations",
        "saved CSV fidelity and report re-save invalidate previously generated history PDFs",
        "history view, regenerated PDF, compatible comparison and confirmed deletion",
        "worksheet selection, optional source persistence and uploaded provenance/currency controls",
        "synthetic weekly ancillary joins preserve cardinality, holiday meaning and original provenance",
        "actual store, department, category, region and holiday chart/table views",
        "all three genuine weekly parameterized holdouts and two rolling-origin folds",
        "future driver uploads reject missing periods and require real aligned values",
        "known weekly future holidays affect real Prophet components with full-history drivers",
        "all three genuine weekly future models preserve the selected weekly anchor",
        "all three genuine monthly holdouts with alternative MAE ranking",
        "all three real monthly future models and 3/6/12-month preset horizons",
        "modified configurations require re-evaluation and custom horizons preserve raw/business views",
        "actual date-window control applies inclusive validated calendar bounds",
        "mean aggregation changes actual arithmetic and consistently labels revenue interpretation in reports",
        "irregular uploads require an explicit native frequency and preserve absent months",
        "no browser JavaScript exceptions",
        *(f"mobile navigation and width: {workspace}" for workspace in workspaces),
    }
    require(required.issubset(names), "Required browser workflow checks are missing.")
    require(browser["gemini_ui_source"] == ("Gemini" if browser["gemini_requested"] else "Local analytics"), "Browser provider source does not match its requested execution mode.")
    require(len(browser["screenshots"]) == 16, "Browser screenshots are incomplete.")
    for screenshot in browser["screenshots"]:
        with Image.open(ROOT / screenshot) as image:
            image.verify()
    notebook = nbformat.read(ROOT / "notebooks/Sales_Forecasting_Experiments.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(cells and all(cell.execution_count is not None for cell in cells), "Unexecuted notebook cells remain.")
    require(not [item for cell in cells for item in cell.get("outputs", []) if item.output_type == "error"], "Notebook cell errors remain.")
    execution = json.loads((ROOT / "outputs/qa/notebook-execution.json").read_text(encoding="utf-8"))
    require(execution["errors"] == 0 and execution["executed_code_cells"] == execution["code_cells"] == len(cells), "Notebook evidence does not match its executed cells.")
    clean = json.loads((ROOT / "outputs/qa/clean-final-environment-results.json").read_text(encoding="utf-8"))
    require(clean["status"] == "passed" and all(check["status"] == "passed" for check in clean["checks"]), "Isolated-runtime verification failed.")
    require({check["name"] for check in clean["checks"]} == {"install", "dependencies", "imports", "pytest"}, "Isolated-runtime checks are incomplete.")
    require(clean["initial_packages"] and {package["name"].lower() for package in clean["initial_packages"]}.issubset({"pip", "setuptools", "wheel"}), "The isolated runtime already contained application dependencies before installation.")
    clean_suites = list(ET.parse(ROOT / "outputs/qa/clean-final-environment-tests.xml").getroot().iter("testsuite"))
    clean_counts = {attribute: sum(int(suite.get(attribute, 0)) for suite in clean_suites) for attribute in ["tests", "failures", "errors", "skipped"]}
    require(clean_counts == clean["pytest"] == {"tests": tests, "failures": 0, "errors": 0, "skipped": 0}, "Isolated-runtime test evidence differs from the current suite.")
    inventory = json.loads((ROOT / "outputs/qa/audit-inventory.json").read_text(encoding="utf-8"))
    require(inventory["status"] == "passed" and inventory["inspected_files"] == len(inventory["files"]) and not inventory["circular_imports"] and inventory["credential_scan"] == "passed", "Source inventory checks failed.")
    require(inventory["issue_counts"]["identified"] > 0 and inventory["issue_counts"]["identified"] == inventory["issue_counts"]["fixed"] and inventory["issue_counts"]["remaining_critical_high"] == 0, "The final audit has unresolved recorded issues.")
    for item in inventory["files"]:
        relative = item["path"]
        if relative.endswith((".py", ".ipynb")) or relative.startswith(("config/", ".streamlit/", "data/sample/")) or relative in {"requirements.txt", "requirements-dev.txt", "constraints.txt", "pyproject.toml", ".env.example", ".gitignore", ".gitattributes"}:
            source = (ROOT / relative).resolve()
            require(source.is_relative_to(ROOT) and source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"], f"Source inventory is stale: {relative}")
    gemini = json.loads((ROOT / "outputs/qa/gemini-audit-final.json").read_text(encoding="utf-8"))
    require(gemini["raw_records_sent"] == 0, "Live Gemini verification included raw records.")
    explanation = gemini["explanation"]
    require((explanation["source"] == "Gemini" and explanation["status"] == "success") or (explanation["source"] == "Local analytics" and explanation["status"] != "success"), "Gemini source/status evidence is inconsistent.")
    for command, label in [([sys.executable, "-m", "ruff", "check", "."], "Ruff"), ([sys.executable, "-m", "pip", "check"], "Dependencies")]:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        require(result.returncode == 0, f"{label} check failed; run its command directly for diagnostics.")
    return {
        "status": "passed", "pytest_passed": tests,
        "browser_checks": len(browser["checks"]), "browser_duration_seconds": browser["duration_seconds"],
        "browser": browser["browser"], "browser_javascript_exceptions": 0,
        "screenshots": len(browser["screenshots"]), "gemini_ui_source": browser["gemini_ui_source"],
        "notebook_executed_cells": len(cells), "notebook_errors": 0,
        "notebook_duration_seconds": execution["duration_seconds"],
        "ruff": "passed", "dependencies": "passed",
        "source_inventory": {"inspected_files": inventory["inspected_files"], "python_files": inventory["python_files"], "audit_updated_files": inventory["audit_updated_files"], "issue_counts": inventory["issue_counts"], "circular_imports": 0, "credential_scan": "passed", "current_source_checksums": "passed"},
        "isolated_runtime": {"status": clean["status"], "created_environment": clean["created_environment"], "initial_packages": clean["initial_packages"], "pytest_passed": clean_counts["tests"], "duration_seconds": clean["duration_seconds"]},
        "browser_model_frequency_coverage": browser["model_frequency_coverage"],
        "gemini_live_recheck": {"verified_at": gemini["verified_at"], "model": explanation["model"], "connectivity_status": gemini["connectivity"]["status"], "generation_status": explanation["status"], "source": explanation["source"]},
        "environment_scope": "Local Windows/Python execution and resized browser viewports; hosted Colab and physical devices were not tested.",
    }


def verify_archive(path: Path) -> dict:
    required = {"app.py", "README.md", "requirements.txt", "constraints.txt", ".env.example", ".gitignore", ".gitattributes", "docs/FINAL_PROJECT_AUDIT.md", "docs/installation.md", "docs/architecture.md", "docs/methodology.md", "docs/user_guide.md", "docs/experimental_results.md", "docs/review3_demo.md", "docs/verification_summary.json", "notebooks/Sales_Forecasting_Experiments.ipynb", "data/sample/uci_online_retail_daily.csv", "outputs/experiments/uci_daily/manifest.json"}
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "Archive CRC check failed.")
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate archive member names.")
        present = set()
        for name in names:
            item = PurePosixPath(name)
            require(not item.is_absolute() and ".." not in item.parts and "\\" not in name and item.parts[0] == "sales-forecasting-system", "Unsafe archive path.")
            relative = PurePosixPath(*item.parts[1:])
            require(not set(relative.parts) & {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "qa", "uploads", "processed"} or relative.name == ".gitkeep", f"Private/runtime archive member: {relative}")
            require(not (relative.name.startswith(".env") and relative.name != ".env.example") and relative.name != "secrets.toml", "Private configuration in archive.")
            require(relative.suffix.lower() not in {".sqlite3", ".sqlite", ".db", ".log", ".zip", ".pyc", ".pyo"} and not relative.name.endswith(("-wal", "-shm")), "Runtime file in archive.")
            require(not (relative.parts[:2] == ("data", "source") and relative.name != ".gitkeep"), "Customer-level source data in archive.")
            content = archive.read(name)
            source = ROOT.joinpath(*relative.parts)
            require(source.is_file() and content == source.read_bytes(), f"Packaged source differs: {relative}")
            require(not API_KEY_PATTERN.search(content), f"Credential-shaped value in archive member: {relative}")
            if relative.as_posix() == ".env.example":
                require(re.search(rb"(?m)^GEMINI_API_KEY=\s*$", content), "Example credential is not blank.")
            if relative.suffix == ".py":
                compile(content, str(relative), "exec")
            present.add(relative.as_posix())
        require(required.issubset(present), "Required project delivery files are missing.")
    return {"status": "passed", "archive": path.relative_to(ROOT).as_posix(), "files": len(names), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_bytes_match": True, "credential_and_runtime_exclusions": "passed"}


def smoke_archive(path: Path, channel: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="salesvision-delivery-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
        project = destination / "sales-forecasting-system"
        environment = os.environ.copy()
        environment["GEMINI_API_KEY"] = ""
        environment["SALESVISION_DATABASE_PATH"] = str(project / "outputs/smoke.sqlite3")
        with socket.socket() as available:
            available.bind(("127.0.0.1", 0))
            port = available.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with (ROOT / "outputs/qa/package-smoke.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true"], cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    require(server.poll() is None, "Extracted Streamlit app exited during startup.")
                    try:
                        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.3)
                else:
                    raise TimeoutError("Extracted Streamlit app did not become healthy.")
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(channel=channel, headless=True)
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    page.goto(url, wait_until="domcontentloaded")
                    page.get_by_role("button", name="Start with a dataset", exact=True).wait_for(timeout=60_000)
                    page.get_by_test_id("stSidebar").get_by_text("Dataset Management", exact=True).click()
                    page.get_by_role("button", name="Load selected dataset", exact=True).click()
                    page.get_by_text("Loaded 305 records from uci_online_retail_daily.csv.", exact=True).wait_for()
                    page.get_by_test_id("stSidebar").get_by_text("Data Preprocessing", exact=True).click()
                    page.get_by_test_id("stSelectbox").filter(has_text="Missing time periods").locator('[data-baseweb="select"]').click()
                    page.get_by_role("option", name="Forward-fill (past values only)", exact=True).click()
                    page.get_by_role("button", name="Apply preprocessing", exact=True).click()
                    page.get_by_text("Processed 305 records into 374 D periods.", exact=True).wait_for(timeout=60_000)
                    require(page.get_by_test_id("stException").count() == 0 and page.get_by_text("This operation could not be completed.", exact=False).count() == 0, "Extracted app raised a UI error.")
                    page.screenshot(path=str(ROOT / "outputs/qa/package-smoke.png"), full_page=True)
                    browser.close()
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=15)
    return {"status": "passed", "browser": channel, "fresh_temporary_extraction": True, "server_health": "passed", "loaded_real_records": 305, "processed_calendar_periods": 374, "gemini": "disabled", "interpreter": sys.version.split()[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="Require passing local pytest/browser/notebook evidence and check Ruff/dependencies.")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run the extracted archive in an isolated offline browser session.")
    parser.add_argument("--browser", choices=["chrome", "msedge"], default="chrome")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qa/delivery-results.json")
    args = parser.parse_args()
    require(not args.smoke or args.archive is not None, "--smoke requires --archive.")
    args.output = args.output.resolve()
    evidence = {"verified_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0]}
    try:
        directories = [ROOT / "outputs/experiments/uci_daily"]
        notebook_directory = ROOT / "outputs/experiments/uci_daily_notebook"
        if (notebook_directory / "manifest.json").is_file():
            directories.append(notebook_directory)
        evidence["experiments"] = [verify_experiment(directory) for directory in directories]
        if len(directories) == 2:
            for name in ["model_comparison.csv", "baseline_comparison.csv"]:
                current, reference = [pd.read_csv(directory / name) for directory in directories]
                np.testing.assert_allclose(current[list(METRICS)], reference[list(METRICS)], rtol=1e-5, atol=1e-3)
            for model in MODELS:
                current, reference = [pd.read_csv(directory / f"{model.lower()}_future_forecast.csv") for directory in directories]
                np.testing.assert_allclose(current[["prediction", "lower", "upper"]], reference[["prediction", "lower", "upper"]], rtol=1e-5, atol=1e-3)
            evidence["notebook_cli_reproduction"] = "passed"
        evidence["documentation"] = verify_documentation(args.output)
        if args.qa:
            evidence["qa"] = verify_qa()
        if args.archive:
            args.archive = args.archive.resolve()
            evidence["package"] = verify_archive(args.archive)
            if args.smoke:
                evidence["package_smoke"] = smoke_archive(args.archive, args.browser)
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
