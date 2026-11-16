# Testing and verification

## Final acceptance evidence — 2026-11-16

Verification uses pinned **Windows 11 / Python 3.12.10** dependencies. The [final audit](FINAL_PROJECT_AUDIT.md) records root causes, measured file/issue counts and limitations. Local execution records underpin the safe [verification summary](verification_summary.json).

| Check | Measured result | Evidence |
|---|---|---|
| Original baseline | 76 passed in 11.48 s | `outputs/qa/audit-baseline.xml` |
| Final automated suite | **160 passed; 0 failures/errors/skips; 19.71 s** | `outputs/qa/pytest-results.xml` |
| Fresh isolated installation | Initially only pip; pinned install, dependency/native imports passed | `outputs/qa/clean-final-environment-results.json` |
| Isolated-runtime suite | **160 passed; 0 failures/errors/skips; 21.86 s** | `outputs/qa/clean-final-environment-tests.xml` |
| Actual Chrome workflow | **63 passed; 0 JavaScript exceptions; 202.59 s** | `outputs/qa/browser-results.json` |
| Browser model coverage | D, W-FRI and MS × ARIMA/SARIMA/Prophet; genuine holdouts and futures | Browser check/matrix evidence |
| Focused revenue/report workflow | 7 passed after correcting mean-target report semantics | `outputs/qa/browser-revenue-fixed.json` |
| Screenshots / narrow layout | 16 captures; all 12 pages at 390 × 844 without body/main overflow | `docs/screenshots/` and browser evidence |
| Fresh local notebook | **15/15 code cells; 0 errors; 10.77 s** | Executed notebook and `outputs/qa/notebook-execution.json` |
| CLI / notebook / UI fidelity | Independent error/date/baseline/backtest recomputation agrees | `scripts/verify_delivery.py` |
| Reports | Both experiments: 9-page PDFs, 16-sheet workbooks | Experiment manifests and parsed exports |
| Portable delivery | 150 archive members verified; fresh extracted-app Chrome smoke passed with 305 records / 374 D periods | `outputs/qa/delivery-results.json` |
| Final live Gemini recheck | Connectivity success; generation **service_unavailable**, labelled Local analytics | `outputs/qa/gemini-audit-final.json` |

The two final suites execute **320 successful test cases across 160 unique cases**. Their fixture databases are isolated and their ordinary AI tests disable real provider requests. The original baseline grew by 84 cases covering independently identified boundaries and genuine model/UI integration.

The full browser run intentionally used **Local analytics**. Earlier live generation succeeded; later live browser/CLI/verifier requests returned provider service unavailability. The final acceptance results do not claim a fresh successful live generation. Provider errors leave numerical forecasting and exports available.

Timings are measurements of these runs on this machine, including concurrent verification load. They are not general performance guarantees. Hosted Colab, physical devices and another operating system were not executed.

## Reproduce acceptance checks

Run from the project directory. Use an existing writable temporary directory for Prophet/CmdStan as described in [installation](installation.md). Supply an unused isolated-environment path for a fresh installation proof.

```powershell
& ".venv\Scripts\python.exe" -m ruff check .
& ".venv\Scripts\python.exe" -m pip check
& ".venv\Scripts\python.exe" -m pytest -q --basetemp outputs/qa/pytest --junitxml outputs/qa/pytest-results.xml
& ".venv\Scripts\python.exe" scripts/verify_clean_environment.py --environment outputs/qa/clean-verification-new --output outputs/qa/clean-final-environment-results.json
& ".venv\Scripts\python.exe" scripts/run_experiments.py
& ".venv\Scripts\python.exe" scripts/execute_notebook.py
& ".venv\Scripts\python.exe" scripts/browser_qa.py --scenario full
& ".venv\Scripts\python.exe" scripts/verify_gemini.py --output outputs/qa/gemini-audit-final.json
& ".venv\Scripts\python.exe" scripts/audit_project.py
& ".venv\Scripts\python.exe" scripts/verify_delivery.py --qa --output docs/verification_summary.json
```

If no Gemini credential is configured, its recheck records `missing_key` and Local analytics. This is valid offline evidence. To request live browser verification, use `scripts/browser_qa.py --scenario full --gemini`; it requires actual Gemini generation and fails rather than counting a fallback as live success.

## Independent negative reproductions

| Saved run | Cases | Passed | Failed |
|---|---:|---:|---:|
| Initial defects — `audit-reproductions.xml` | 58 | 5 | 53 |
| Further discoveries — `audit-discoveries.xml` | 12 | 2 | 10 |
| Forecast/overflow boundaries — `audit-boundary-reproductions.xml` | 2 | 0 | 2 |
| Snapshot/logging boundaries — `audit-snapshot-reproductions.xml` | 7 | 1 | 6 |
| Mean PDF/Excel interpretation — `audit-mean-report-reproduction.xml` | 1 | 0 | 1 |
| **Total retained pre-fix reproductions** | **80** | **8** | **72** |

These intentionally failing pre-fix records contain 39 distinct regression functions, with parameterized cases grouped by root cause in the final audit. Final repair suites pass; these historical failures are retained as reproduction evidence. Browser training-selection and mean-report failures also have separate before/after records.

Automation-only defects were corrected separately: hidden expander/checkbox/radio targeting and waiting for Streamlit rerun/render completion before chart inspection or downloading. Assertions still check actual exported values, calendars and controls.

## Automated coverage

- CSV/Excel mapping/validation, configured limits, duplicate headers, holiday meaning and validated ancillary joins.
- Explicit cleaning, original row-reference audits, business/date filters, UTC normalization and native/coarser frequencies.
- Unknown/zero/imputed distinctions, incomplete drivers/holiday dates, overflow guards and observed summaries.
- Genuine daily/weekly/monthly model fitting, analytical ARIMA/seasonal-SARIMA expectations, finite ordered intervals and Prophet identifiability.
- Chronological holdouts, training-partition rolling origins, leakage prevention, aligned actuals, zero-safe metrics and signed bias.
- Persistent widgets/forms, immediate candidate selection, invalidation, full-history refits and modified-configuration re-evaluation.
- Formula-safe spreadsheet fields/headers, aggregation-aware PDFs/workbooks, complete summaries and precise restored snapshots.
- SQLite identity/integrity/corruption handling, graceful history unavailability, report refresh/comparison and confirmed cascading deletion.
- Gemini classified failures, structured/cited responses, non-text parts, closure, aggregate hashes and local fallback.

## Actual browser controls and conditional data

The default full run starts a real localhost Streamlit server, creates isolated browser contexts/SQLite history, drives actual controls and downloads files. Installed Chrome or Edge is used through Playwright; add `--browser msedge` for Edge.

- All twelve empty-state workspaces and all twelve narrow layouts; source reactivation invalidation.
- Real UCI CSV load and uploads, a multi-sheet Excel selection, optional source persistence, source/currency fields and corrupt workbook rejection.
- Every EDA tab, trailing windows, actual chart traces/hover/pan/zoom, STL cycles, all four business filters and segment/holiday views.
- All candidate panels, selection/holdout/metric/fold controls, model-inspection downloads and submitted/navigation state.
- Daily preset/custom horizons and history/business toggles; weekly anchor, future-driver and holiday uploads; all monthly presets and custom/re-evaluated configuration.
- Every insight-topic option, question/model state and aggregate inspection; actual fallback/report re-save/history-PDF invalidation.
- CSV/Excel/PDF matching values, history dataset filtering, compatible/incompatible comparisons and confirmed deletion.
- Derived Quantity × UnitPrice revenue, all explicit invalid/missing/negative/gap decisions, inclusive date bounds and mean arithmetic/report labels.
- Irregular uploaded history requiring explicit monthly cadence while preserving an unknown month.

Weekly store/department/category/region/driver/holiday inputs are **explicitly synthetic fixtures**. They verify functionality, not Walmart or empirical business findings. Focused subsets can be run with `--scenario revenue` or `--scenario irregular` and a distinct `--output`; only `scope=full` satisfies final delivery verification.

## Artifact, source and portable-archive verification

`scripts/audit_project.py` inventories distributable inputs, scans for credential-shaped content, compiles every Python/notebook source, parses configuration, imports project modules, checks import cycles, counts the audit ledger and summarizes retained JUnit evidence. Its timestamp-backed changed-path manifest is measured from the original baseline start.

`scripts/verify_delivery.py --qa` requires complete passing suite/browser/notebook evidence, the nine browser model/frequency combinations, final isolated-install proof, current source checksums and a closed audit ledger. It records actual Gemini availability/source instead of equating a metadata lookup with generation.

The independent experiment verifier recomputes errors from genuine observed actuals, validates masks/dates/full-history sizes/backtest boundaries, recomputes naive benchmarks, verifies **74 artifact checksums** across the two experiments and parses their reports. It compares notebook/CLI/UI measurements at `rtol=1e-5`, `atol=1e-3`.

```powershell
& ".venv\Scripts\python.exe" scripts/package_project.py
& ".venv\Scripts\python.exe" scripts/verify_delivery.py --qa --archive outputs/salesvision_project.zip --smoke
```

Archive verification checks CRCs, paths, required files, source-byte equality, blank example credentials and private/runtime/customer-source exclusions. The smoke check extracts the archive into a temporary directory, starts its app with the current interpreter, loads the packaged real sample and preprocesses **305 records into 374 daily periods**. This verifies extracted application execution on the measured machine.

The verified **150-file** archive contains the safe verification summary, final audit, screenshots, executed notebook and real CLI artifacts. Detailed QA XML/JSON, databases, logs, clean environments and downloads stay local. Final archive integrity/source/exclusion checks and fresh extracted-app Chrome smoke passed; results, byte size and SHA-256 are recorded in `outputs/qa/delivery-results.json`.
