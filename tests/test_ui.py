"""Streamlit navigation, integrated workflow and stale-result prevention."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data.loader import LoadedDataset
from src.ui.dashboard import NAVIGATION

ROOT = Path(__file__).resolve().parents[1]


def click(app, label):
    next(button for button in app.button if button.label == label).click().run(timeout=120)
    assert not app.exception
    assert not [error.value for error in app.error], [error.value for error in app.error]
    return app


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("SALESVISION_DATABASE_PATH", str(tmp_path / "ui.sqlite3"))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not application.exception
    return application


def test_all_navigation_pages_handle_empty_state(app):
    for name in NAVIGATION:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception, name
        assert not app.error, [error.value for error in app.error]


@pytest.mark.integration
def test_streamlit_preprocess_train_forecast_reports_history_and_ai_fallback(app, daily_frame):
    content = daily_frame.to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    app.session_state["loaded"] = LoadedDataset(daily_frame, "synthetic_ui.csv", digest, len(content))
    app.session_state["loaded_metadata"] = {"name": "Synthetic integrated UI test", "kind": "synthetic", "source_sha256": digest, "currency": "sales units"}
    app.session_state["load_operations"] = []
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    click(app, "Apply preprocessing")
    assert len(app.session_state["processed"].series) == 90
    app.sidebar.radio[0].set_value("Model Training").run()
    click(app, "Train and evaluate models")
    result = app.session_state["evaluation"]
    assert len(result.predictions) == 3, result.comparison.to_string()
    app.sidebar.radio[0].set_value("Model Comparison").run()
    assert not app.error
    app.sidebar.radio[0].set_value("Sales Forecasting").run()
    click(app, "Generate future forecast")
    run = app.session_state["forecast"]
    assert run.training_observations == 90
    assert len(run.predictions) == 7
    next(box for box in app.selectbox if box.label == "Forecast horizon").set_value(30).run()
    click(app, "Generate future forecast")
    run = app.session_state["forecast"]
    assert len(run.predictions) == 30
    for model in ["ARIMA", "Prophet", "SARIMA"]:
        next(box for box in app.selectbox if box.label == "Forecasting model").set_value(model).run()
        click(app, "Generate future forecast")
        run = app.session_state["forecast"]
        assert run.model == model and len(run.predictions) == 30
    app.sidebar.radio[0].set_value("Model Comparison").run()
    app.sidebar.radio[0].set_value("Sales Forecasting").run()
    assert next(box for box in app.selectbox if box.label == "Forecast horizon").value == 30
    assert next(box for box in app.selectbox if box.label == "Forecasting model").value == run.model
    app.sidebar.radio[0].set_value("Gemini AI Insights").run()
    next(button for button in app.button if button.label == "Generate business explanation").click().run()
    assert not app.exception and not app.error
    assert app.session_state["insights"].source == "Local analytics"
    app.sidebar.radio[0].set_value("Reports & Downloads").run()
    click(app, "Build report package")
    package = app.session_state["report_package"]
    assert package["pdf"].startswith(b"%PDF")
    assert package["excel"].startswith(b"PK")
    app.sidebar.radio[0].set_value("Forecast History").run()
    assert not app.exception and not app.error
    assert next(box for box in app.selectbox if box.label == "View previous forecast").value == run.run_id
    app.sidebar.radio[0].set_value("Dataset Management").run()
    click(app, "Load selected dataset")
    assert app.session_state["processed"] is None
    assert app.session_state["evaluation"] is None
    assert app.session_state["forecast"] is None


def test_preprocessing_policies_survive_navigation(app, daily_frame):
    content = daily_frame.to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    app.session_state["loaded"] = LoadedDataset(daily_frame, "audit.csv", digest, len(content))
    app.session_state["loaded_metadata"] = {"name": "Audit fixture", "kind": "synthetic"}
    app.session_state["load_operations"] = []
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    next(box for box in app.selectbox if box.label == "Missing time periods").set_value("Zero (explicit business assumption)").run()
    next(box for box in app.selectbox if box.label == "Exact duplicate policy").set_value("Remove exact duplicates").run()
    click(app, "Apply preprocessing")
    app.sidebar.radio[0].set_value("Overview Dashboard").run()
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    assert next(box for box in app.selectbox if box.label == "Missing time periods").value == "Zero (explicit business assumption)"
    assert next(box for box in app.selectbox if box.label == "Exact duplicate policy").value == "Remove exact duplicates"


def test_submitted_model_parameters_survive_navigation(app, daily_frame):
    content = daily_frame.to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    app.session_state["loaded"] = LoadedDataset(daily_frame, "audit.csv", digest, len(content))
    app.session_state["loaded_metadata"] = {"name": "Audit fixture", "kind": "synthetic"}
    app.session_state["load_operations"] = []
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    click(app, "Apply preprocessing")
    app.sidebar.radio[0].set_value("Model Training").run()
    next(box for box in app.multiselect if box.label == "Models to evaluate").set_value(["ARIMA"]).run()
    next(box for box in app.selectbox if box.label == "Model selection metric").set_value("MAE")
    next(box for box in app.number_input if box.label == "p").set_value(0)
    next(box for box in app.number_input if box.label == "q").set_value(0)
    next(box for box in app.number_input if box.label == "Optimizer iterations").set_value(225)
    click(app, "Train and evaluate models")
    app.sidebar.radio[0].set_value("Model Comparison").run()
    app.sidebar.radio[0].set_value("Model Training").run()
    assert next(box for box in app.multiselect if box.label == "Models to evaluate").value == ["ARIMA"]
    assert next(box for box in app.selectbox if box.label == "Model selection metric").value == "MAE"
    assert next(box for box in app.number_input if box.label == "Optimizer iterations").value == 225
    assert next(box for box in app.number_input if box.label == "p").value == 0


def test_model_selection_updates_submit_availability_and_parameter_panels(app, daily_frame):
    content = daily_frame.to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    app.session_state["loaded"] = LoadedDataset(daily_frame, "selection.csv", digest, len(content))
    app.session_state["loaded_metadata"] = {"name": "Model selection audit", "kind": "synthetic"}
    app.session_state["load_operations"] = []
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    click(app, "Apply preprocessing")
    app.sidebar.radio[0].set_value("Model Training").run()
    selection = next(box for box in app.multiselect if box.label == "Models to evaluate")
    selection.set_value([]).run()
    assert next(button for button in app.button if button.label == "Train and evaluate models").disabled
    assert not any(box.label == "p" for box in app.number_input)
    next(box for box in app.multiselect if box.label == "Models to evaluate").set_value(["Prophet"]).run()
    assert not next(button for button in app.button if button.label == "Train and evaluate models").disabled
    assert any(box.label == "Yearly seasonality" for box in app.selectbox)
    assert not any(box.label == "p" for box in app.number_input)
    click(app, "Train and evaluate models")
    assert set(app.session_state["evaluation"].predictions) == {"Prophet"}


def test_corrupt_history_does_not_block_core_pages(monkeypatch, tmp_path, daily_frame):
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"Isolated corrupt SQLite fixture")
    monkeypatch.setenv("SALESVISION_DATABASE_PATH", str(path))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert any(button.label == "Start with a dataset" for button in app.button)
    for name in NAVIGATION:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception, name
        assert not app.error, [error.value for error in app.error]
    content = daily_frame.to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    app.session_state["loaded"] = LoadedDataset(daily_frame, "degraded.csv", digest, len(content))
    app.session_state["loaded_metadata"] = {"name": "Degraded history audit", "kind": "synthetic"}
    app.session_state["load_operations"] = []
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    click(app, "Apply preprocessing")
    app.sidebar.radio[0].set_value("Model Training").run()
    next(box for box in app.multiselect if box.label == "Models to evaluate").set_value(["ARIMA"]).run()
    click(app, "Train and evaluate models")
    app.sidebar.radio[0].set_value("Sales Forecasting").run()
    assert next(box for box in app.checkbox if box.label == "Save this execution and report snapshot to forecast history").disabled
    click(app, "Generate future forecast")
    assert len(app.session_state["forecast"].predictions) == 7
    app.sidebar.radio[0].set_value("Reports & Downloads").run()
    click(app, "Build report package")
    assert app.session_state["report_package"]["pdf"].startswith(b"%PDF")
    assert path.read_bytes() == b"Isolated corrupt SQLite fixture"


def test_irregular_history_has_an_explicit_native_frequency_control(app):
    frame = pd.DataFrame({"Date": ["2023-01-10", "2023-03-15"], "Sales": [10, 20]})
    content = frame.to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    app.session_state["loaded"] = LoadedDataset(frame, "irregular.csv", digest, len(content))
    app.session_state["loaded_metadata"] = {"name": "Irregular calendar audit", "kind": "synthetic"}
    app.session_state["load_operations"] = []
    app.sidebar.radio[0].set_value("Data Preprocessing").run()
    assert any(box.label == "Native source frequency" for box in app.selectbox)
    next(box for box in app.selectbox if box.label == "Native source frequency").set_value("Monthly").run()
    frequency = next(box for box in app.selectbox if box.label == "Forecasting frequency")
    assert frequency.options == ["Monthly"]
    next(box for box in app.selectbox if box.label == "Missing time periods").set_value("Forward-fill (past values only)").run()
    click(app, "Apply preprocessing")
    dataset = app.session_state["processed"]
    assert dataset.frequency == "MS" and len(dataset.series) == 3
    assert dataset.metadata["aggregation_options"]["source_frequency"] == "MS"
