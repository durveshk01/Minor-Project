# Notebook and Google Colab

## Verified local execution

`notebooks/Sales_Forecasting_Experiments.ipynb` has 16 sections and 15 code cells. It was executed with the local project Python 3.12.10 interpreter; all code cells completed with zero errors and reproduced the CLI metrics.

Run from the project directory:

```powershell
& ".venv\Scripts\python.exe" scripts/execute_notebook.py
```

The executor validates the notebook, starts this interpreter as its kernel without changing global kernel installation, runs cells in order, saves executed outputs into the notebook, and records execution evidence in `outputs/qa/notebook-execution.json`.

The notebook writes numerical/report artifacts to `outputs/experiments/uci_daily_notebook/`, preserving the separate CLI experiment and its recorded explanation status. Re-execution creates new execution identities and can append to its local notebook history database.

## Notebook sections

1. Project introduction and objectives.
2. Environment, project ZIP upload and optional Google Drive.
3. Dataset acquisition, provenance and description.
4. Explicit validation/cleaning and calendar construction.
5. Observed exploratory analysis.
6. Robust STL seasonal analysis.
7. Prespecified chronological evaluation design.
8. Genuine ARIMA evaluation.
9. Genuine SARIMA evaluation.
10. Genuine Prophet evaluation.
11. Common model comparison, baselines and training-only backtests.
12. Full-history future forecasting.
13. Optional grounded Gemini explanation.
14. CSV, illustrated PDF, Excel and SQLite history.
15. Actual findings, reproduction check and limitations.
16. Artifact manifest, checksums and downloadable result bundle.

## Colab project-ZIP workflow

Hosted Colab execution has not been performed; the following branches are supplied for portability.

1. Build `outputs/salesvision_project.zip` locally with `python scripts/package_project.py`.
2. Upload `Sales_Forecasting_Experiments.ipynb` into Colab.
3. Select a **Python 3.12** runtime; the notebook also accepts 3.11. Local verification covered 3.12 only. Its pinned dependency guard rejects other Python versions.
4. Keep `USE_GOOGLE_DRIVE = False` in the environment cell.
5. Run that cell. If the shared project is absent, it requests exactly one project ZIP through `files.upload()`.
6. Upload `salesvision_project.zip`. The cell checks extraction paths and locates `/content/sales-forecasting-system`.
7. The cell installs the project requirements using the runtime's interpreter and sets the shared-service import path.
8. Run the remaining cells in order. On a reused runtime, restart if package installation indicates already-loaded dependencies need restarting, then rerun the environment cell.

The derived real UCI sample is already included, so no customer-level workbook is needed. The acquisition branch downloads the openly licensed source only if the derived sample is absent.

## Optional Google Drive workflow

Put the **unpacked** project under `MyDrive/sales-forecasting-system`. In the environment cell, set:

```python
USE_GOOGLE_DRIVE = True
DRIVE_PROJECT_DIRECTORY = Path('/content/drive/MyDrive/sales-forecasting-system')
```

Update the path if using a different folder. The cell mounts Drive and validates the complete project directory before importing shared services. Generated results remain under that project's `outputs/experiments/uci_daily_notebook/` path.

## Optional Gemini

The notebook defaults to `ENABLE_GEMINI = False`, producing a correctly labelled deterministic **Local analytics** explanation. No credential is required for the numerical experiment.

To request a live explanation, change that flag in section 13. In Colab, hidden `getpass` requests the key if it is not already server-configured. Locally, server settings can read the private `.env`. The default model is `gemini-3.1-flash-lite`, configurable with `GEMINI_MODEL`.

Do not type a key into a saved code cell or print it. Inspect the actual explanation source/status; a provider failure correctly uses local analytics. Successful local/browser/CLI Gemini calls do not constitute hosted-Colab verification.

## Outputs and downloads

The final cell exports all model holdout/backtest/future tables, fitted summaries, versions, provenance, seed, report snapshot, PDF/Excel reports, figures, actual findings and checksums. It verifies the manifest's artifact hashes and creates:

`outputs/experiments/uci_daily_notebook/notebook_results.zip`

In Colab, `files.download()` offers this result bundle. The generated bundle also contains the notebook's own aggregate history database. The separate **portable project ZIP** excludes runtime databases and nested ZIPs.

The Streamlit application remains independently usable with `streamlit run app.py`; the notebook is an experiment/reproducibility entry point, not a required live connection.
