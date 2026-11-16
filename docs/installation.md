# Installation and configuration

## 1. Environment

The verified environment is **Windows 11, Python 3.12.10**, with the dependency versions in `requirements.txt` and `constraints.txt`. Install a Python 3.12 interpreter and unpack the project into a writable local directory. The extracted archive contains a `sales-forecasting-system` top-level folder; use that folder as the working directory for every command below.

The project has no separate frontend build or web API process. Streamlit runs the Python application directly.

## 2. Windows PowerShell

```powershell
py -3.12 --version
py -3.12 -m venv .venv
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".venv\Scripts\python.exe" -m pip check
& ".venv\Scripts\Activate.ps1"
streamlit run app.py
```

If PowerShell does not permit the activation script, use the virtual environment interpreter directly:

```powershell
& ".venv\Scripts\python.exe" -m streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. Stop the server with `Ctrl+C`. Browser sessions hold current analysis state; saved forecast snapshots remain in SQLite after a restart.

If an existing virtual environment reports `No module named pip`, install Python's bundled installer in that environment, then rerun the installation/check commands:

```powershell
& ".venv\Scripts\python.exe" -m ensurepip
```

### Development and QA dependencies

```powershell
& ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
& ".venv\Scripts\ruff.exe" check .
& ".venv\Scripts\python.exe" -m pytest -q
```

Actual browser QA uses installed Google Chrome or Microsoft Edge through Playwright's browser channel. If neither is installed, install one before running `scripts/browser_qa.py`. An additional Playwright browser download is not needed for these channel-based commands.

## 3. macOS/Linux commands

The equivalent installation procedure is supplied below; these operating systems have not been part of the verified execution environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
streamlit run app.py
```

## 4. Optional server-side configuration

Copy `.env.example` to a private `.env` using your editor, then set values locally. The distributed example is:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
SALESVISION_DATABASE_PATH=outputs/salesvision.sqlite3
SALESVISION_MAX_UPLOAD_MB=100
```

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | Optional Gemini credential; keep it on the server |
| `GEMINI_MODEL` | Account-supported model; the default passed live generation |
| `SALESVISION_DATABASE_PATH` | Relative-to-project or absolute SQLite file path |
| `SALESVISION_MAX_UPLOAD_MB` | Integer application-side file size limit, 1–10000 MB; default 100 MB |

Alternatively, place `GEMINI_API_KEY` and `GEMINI_MODEL` in private `.streamlit/secrets.toml`. Existing environment variables take precedence; `.env` is loaded without overriding them, and Streamlit secrets provide a fallback for Gemini values.

Restart Streamlit after editing server configuration. Keep `.env` and `.streamlit/secrets.toml` out of archives and source control. Rotate credentials that have been exposed. No key is necessary for EDA, training, forecasting, reporting, history, or deterministic local explanations.

Streamlit's `server.maxUploadSize` is separately configured to 100 MB in `.streamlit/config.toml`. If increasing the allowed upload size, align that setting with `SALESVISION_MAX_UPLOAD_MB`.

Invalid configuration produces a readable application error. If the history database is corrupt or unavailable, the application preserves core analytics/forecasting/reporting and disables history controls; correct the configured path to restore persistence. The application does not delete the existing database to recover.

### Test Gemini

```powershell
& ".venv\Scripts\python.exe" scripts/verify_gemini.py --model gemini-3.1-flash-lite
```

Inspect the printed **connectivity status**, **explanation status**, and **explanation source**. A model metadata lookup alone does not establish successful generation. The verifier records sanitized results in `outputs/qa/gemini-live.json`; a local fallback is correctly labelled. The final audit reached `gemini-3.1-flash-lite`, but generation returned `service_unavailable`; earlier live generation succeeded. See [testing](testing.md) for the dated evidence.

### Verify a clean installation

Supply an unused environment directory when recording a new installation proof:

```powershell
& ".venv\Scripts\python.exe" scripts/verify_clean_environment.py --environment outputs/qa/clean-verification-new --output outputs/qa/clean-final-environment-results.json
```

The script creates a separate virtual environment, records its initial packages, installs `requirements-dev.txt`, runs `pip check`, imports native/application dependencies and executes the complete suite. A repeated command reuses its environment; choose another unused directory for a fresh-install check. The final audit's newly created environment initially contained only pip and passed all **160 tests**. Detailed installation logs remain local in `outputs/qa/`.

## 5. Data and reproducible execution

The archive includes the derived real daily UCI sample and its provenance metadata, so the demonstration can run without downloading data. To acquire the original openly licensed workbook and reproduce the derivation:

```powershell
& ".venv\Scripts\python.exe" scripts/fetch_uci_data.py
```

The original workbook is stored locally in `data/source/Online Retail.xlsx`; it contains customer identifiers. The portable archive uses the derived daily file instead. The acquisition script retains existing source/sample files; it does not overwrite them automatically.

Run the real numerical experiment and notebook:

```powershell
& ".venv\Scripts\python.exe" scripts/run_experiments.py --output outputs/experiments/uci_daily_local
& ".venv\Scripts\python.exe" scripts/execute_notebook.py
```

Use `--gemini` on the experiment command only when requesting a live provider explanation. Notebook execution is offline by default. See [Colab](colab.md) for its environment branches.

## 6. Windows native-library troubleshooting

Use the project interpreter rather than a global Anaconda or newer Python installation. In the verified machine, global Python was 3.14 and was not the compatible project runtime.

The pinned environment resolved native-wheel import failures and a Prophet/CmdStanPy compatibility issue. Important versions include NumPy `2.2.6`, SciPy `1.15.3`, Statsmodels `0.14.4`, Prophet `1.1.7`, and CmdStanPy `1.2.5`. Preserve the accompanying transitive constraints when rebuilding the environment.

If an import reports `An Application Control policy has blocked this file.`, record the failing package and rebuild using the pinned Python 3.12 requirements in an appropriate local directory. Do not substitute arbitrary dependency upgrades for the verified configuration.

Prophet/CmdStan needs a writable temporary directory. In the verified environment, the following existing directory was used successfully:

```powershell
$env:TEMP = 'C:\Users\DURVES~1\AppData\Local\Temp\opencode'
$env:TMP = $env:TEMP
```

On another computer, set these variables to an **existing, writable temporary directory on that computer** before running tests, experiments, or Streamlit. The path above is environment-specific.

## 7. Delivery archive

```powershell
& ".venv\Scripts\python.exe" scripts/package_project.py
```

Output: `outputs/salesvision_project.zip`. Extract it, create a fresh Python 3.12 virtual environment, install requirements, and launch `streamlit run app.py`. The virtual environment and runtime history are intentionally machine-local; dependencies are supplied as pinned installation requirements.
