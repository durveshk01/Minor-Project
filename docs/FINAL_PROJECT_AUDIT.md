# Final SalesVision project audit

**Date:** 2026-11-16 · **Project:** Sales Forecasting Using Python · **Platform:** Windows 11 / Python 3.12.10

## Executive result

The shared Streamlit/CLI/notebook numerical workflow passes its final local acceptance checks. Genuine ARIMA, SARIMA/SARIMAX and Prophet models were evaluated and forecasted through the actual browser on daily, anchored-weekly and monthly data. All twelve workspaces, exports, history and conditional upload workflows were exercised.

- **109 distributable files inspected**, including **65 Python files** and all **15 notebook code-cell sources**. The separately verified faculty PDF is an additional generated deliverable.
- **42 grouped issues identified and fixed**; zero remaining recorded critical/high-priority issues.
- **49 updated source/configuration/tooling/test/documentation/notebook paths**, including **7 added files** and **42 updated existing files**. Sixteen screenshots and the experiment artifacts were also regenerated.
- **160/160 final tests passed** in both the project environment and a newly created pinned environment: **320 passing acceptance executions**, zero failures/errors/skips.
- **63/63 complete actual-Chrome checks passed**, zero JavaScript exceptions; **16** screenshots; all twelve pages checked at **390 × 844**.
- Fresh local notebook: **15/15 code cells**, zero errors; independently reproduced CLI metrics.
- Refreshed experiment reports: **9-page illustrated PDFs**, **16-sheet Excel workbooks**; actual values, calendars, metrics and checksums independently verified.
- **153-file portable archive verified**, including the faculty manual/PDF/generator, CRCs, safe paths and exact current-source bytes; fresh extracted-app Chrome smoke passed with 305 real records / 374 daily periods.
- Final live Gemini lookup succeeded but generation returned **`service_unavailable`**. The complete acceptance browser run and refreshed CLI evidence use correctly labelled **Local analytics**. Earlier live success is retained as historical evidence.

The portable archive rebuild/extraction checks are recorded under **Delivery verification** below. Detailed machine-local execution records are in `outputs/qa/`; the distributable [verification summary](verification_summary.json) retains safe aggregate evidence.

## Scope and inspection method

The inventory covers the entry point, all source/configuration/tooling/tests, pinned requirements, project configuration, styles, documentation/diagrams, notebook, sample data/metadata and screenshots. Original customer-level source data and private/runtime content remain local.

`scripts/audit_project.py` records SHA-256/size/modified time for every inspected file, compiles every Python AST and notebook source, parses JSON/TOML/notebook schema, imports project modules, checks local import cycles and scans distributable content for credential-shaped values. Source inspection examined error boundaries, import/configuration assumptions, cleaning/aggregation, leakage/scoring, actual model fitting, cache/state, report serialization, SQLite transactions and official Gemini transport. No detected circular imports or credential-shaped distributable content remain.

The original baseline began at **2026-11-16T13:29:33.203198+00:00**. The updated-path manifest counts non-PNG distributable files whose modification time is at or after that baseline. Issue counts group observations by common root cause; parameterized assertions contribute multiple cases to a single issue. Findings came from saved negative regressions, actual browser failures and source/configuration inspection.

### Evidence key

- **R1:** `outputs/qa/audit-reproductions.xml` — 58 cases, 53 pre-fix failures.
- **R2:** `outputs/qa/audit-discoveries.xml` — 12 cases, 10 pre-fix failures.
- **R3:** `outputs/qa/audit-boundary-reproductions.xml` — 2 pre-fix failures.
- **R4:** `outputs/qa/audit-snapshot-reproductions.xml` — 7 cases, 6 pre-fix failures.
- **R5:** `outputs/qa/audit-mean-report-reproduction.xml` — 1 pre-fix failure.
- **B1:** `browser-training-reproduction.json` → `browser-training-fixed.json`.
- **B2:** `browser-revenue-first.json` → `browser-revenue-fixed.json`.
- **S:** final project/isolated 160-case suites and `browser-results.json` containing the complete 63-check run.
- **I:** recorded source/configuration inspection and `audit-inventory.json`; targeted guards exercised in applicable final workflows.
- **D:** `audit-delivery-reproduction.json` and final delivery-verifier execution.

QA filenames in this key are under `outputs/qa/`. Historic negative runs are intentionally retained; they do not indicate failure of the final suites.

## Root-cause issue ledger

| ID | Severity | Identified root cause | Implemented correction and verification | Status |
|---|---|---|---|---|
| SV-001 | Medium | Invalid upload-limit configuration escaped or silently changed meaning. | Integer 1–10000 validation with named `ConfigurationError`; six R1 boundaries and S. | Fixed |
| SV-002 | Medium | Malformed calendar frequencies raised untyped Pandas errors. | Validate/normalize frequency before calendar operations; R1 and S. | Fixed |
| SV-003 | Medium | Invalid or timezone-aware date windows used inconsistent coercion. | Validate two bounds and normalize through the shared UTC calendar; R1 and actual inclusive browser window. | Fixed |
| SV-004 | High | Arbitrarily named mapped targets/revenue inputs could be selected as regressors. | Forbid mapped target/date/revenue source inputs; R1 leakage regression and S. | Fixed |
| SV-005 | High | Incomplete regressor periods became partial means. | Preserve incomplete aggregated drivers as unknown; R1 and exact driver/calendar guards in S. | Fixed |
| SV-006 | High | Unknown-only monthly/yearly/weekday summaries became zero. | Observed rollups use minimum-count sums and retain unknown evidence; R1 and S. | Fixed |
| SV-007 | High | Mean-target explanations/reports treated period-average sums ambiguously as revenue totals. | Aggregation-aware local explanations, shared PDF/Excel target definition, unambiguous statistic/monthly labels; R1, R5, B2 and S. | Fixed |
| SV-008 | Medium | Model configuration shape/types were insufficiently checked before field access/fitting. | Validate actual `ModelConfig`, integer/boolean/order/seed/interval/prior/regressor fields at boundaries; R1/R2 and S. | Fixed |
| SV-009 | Medium | Prophet yearly-history checks rejected exactly two complete supported cycles. | Correct 730 D / 104 weekly / 24 monthly thresholds; R1 boundary cases and real weekly/monthly fits. | Fixed |
| SV-010 | Medium | Backtest booleans/floats and unsupported horizons reached partitioning. | Validate integer fold/horizon ranges before splitting; R1 and genuine rolling origins in S. | Fixed |
| SV-011 | Medium | An explicit zero STL cycle was silently replaced by a default. | Distinguish absent from invalid periods and validate 2–366; R1 and actual cycle changes in S. | Fixed |
| SV-012 | High | Squared large finite errors overflowed RMSE. | Stable finite-range metric calculation with analytically correct large-error regression; R1 and independent saved-error recomputation. | Fixed |
| SV-013 | Medium | Nonnumeric metric vectors escaped as conversion errors. | Typed numeric-vector validation; R1 and S. | Fixed |
| SV-014 | High | Nonfinite or negative error scores could become model winners. | Exclude invalid successful-row scores and use deterministic valid ranking; R1 and real RMSE/MAE selection. | Fixed |
| SV-015 | High | Dataframe snapshots lost tiny numeric values and nanosecond dates. | JSON-safe records preserve float/timestamp precision and unknown values; R1 and SQLite/report round trips in S. | Fixed |
| SV-016 | High | Spreadsheet headers/leading whitespace or linebreak formulas were insufficiently escaped. | Neutralize formula-shaped text in headers and cells while retaining numeric negatives; R1 and export regressions. | Fixed |
| SV-017 | High | Corrupt SQLite initialization blocked the core application. | Typed database errors and history-disabled entry-point degradation preserve database bytes and core forecasting/reporting; R1 and integrated S. | Fixed |
| SV-018 | High | A report could be saved against a different forecast execution. | Require matching fingerprint, run identity and prediction table; R1 and saved snapshot integrity in S. | Fixed |
| SV-019 | Medium | Reports omitted the actual final full-history fitted summary. | Include separately labelled final-fit diagnostics alongside holdout summaries; R1 and parsed current reports. | Fixed |
| SV-020 | Medium | Closed Gemini clients could report false success; pinned-SDK transports needed proper closure. | Closed-client status/cleared handle and official transport fallback closure; R1 and AI lifecycle tests. | Fixed |
| SV-021 | High | Holiday summaries counted incomplete/unknown dates as observed totals. | Preserve incomplete date totals and count only genuine observed dates; R2 and S. | Fixed |
| SV-022 | Medium | Timezone-aware holiday events used a different date basis. | Normalize holiday dates through UTC consistently; R2 and S. | Fixed |
| SV-023 | Medium | Holiday-enabled Prophet accepted an event table with no training-calendar match. | Validate table/event shape and require matching training periods; R2 and actual holiday-enabled weekly fitting. | Fixed |
| SV-024 | High | Leading blank CSV lines bypassed duplicate-header detection. | Inspect the first substantive header before Pandas parsing; R2 and loader regressions. | Fixed |
| SV-025 | Medium | Incompatible Walmart Store join types escaped as raw merge errors. | Typed join validation while preserving cardinality constraints; R2 and synthetic ancillary browser joins. | Fixed |
| SV-026 | Medium | Semantically equivalent holiday encodings falsely disagreed across joined files. | Normalize bool/0–1/yes–no meaning, preserve unknown flags and reject genuine disagreement; R2 and S. | Fixed |
| SV-027 | High | Irregular uploaded history had no explicit native-frequency correction control. | Native cadence selector plus native/coarser-only options; R2 and actual two-record monthly upload in S. | Fixed |
| SV-028 | Medium | Evaluation/UI holdouts exceeded the forecast service's supported limit. | Enforce integer 2–366 and sufficient remaining history in API/UI; R3 and S. | Fixed |
| SV-029 | High | Individually finite sales could overflow after aggregation. | Validate aggregate numeric range and fail safely before fitting; R3 and S. | Fixed |
| SV-030 | Medium | Restoring mixed timestamp precision failed after precise serialization. | Mixed-format strict datetime restoration; R4 and S. | Fixed |
| SV-031 | High | Valid JSON containing corrupt snapshot shapes escaped integrity checks. | Validate run/report metadata, configuration, operations, tables and calendars; R4 and S. | Fixed |
| SV-032 | Medium | Unavailable file logging silently removed server diagnostics. | Fall back to stderr `StreamHandler` with safe warning; R4 and S. | Fixed |
| SV-033 | High | Streamlit cleanup lost submitted policies/model and analysis controls on navigation. | Durable deep-copied control values, valid-value restoration and form-submit retention; R1 and actual browser navigation in S. | Fixed |
| SV-034 | Medium | Deferred model selection left training enabled when no candidate was selected. | Move candidate multiselect outside the training form; B1, seven UI regressions and S. | Fixed |
| SV-035 | High | Future holidays could fall off the selected weekly/monthly calendar and be silently ineffective. | Exact future-calendar membership checks, explicit missing-calendar warning; I, regression and real nonzero weekly Prophet holiday component in S. | Fixed |
| SV-036 | High | Joined-upload hashing replaced original provenance and lost small driver-file changes. | Retain original/ancillary byte hashes and distinct joined identity; I, tiny-driver regression and actual joins in S. | Fixed |
| SV-037 | Medium | Optional/future input loaders did not uniformly receive the configured upload limit. | Pass application maximum through ancillary/future loaders; I, loader boundaries and actual upload workflows. | Fixed |
| SV-038 | Medium | Long PDF fitted-summary lines discarded trailing evidence. | Wrap complete summary lines instead of truncating; I and parsed marker regression. | Fixed |
| SV-039 | High | Explanation re-save/deletion left stale generated history PDFs in session cache. | Invalidate per-execution PDF cache on update/delete; I and actual saved-PDF regeneration check in S. | Fixed |
| SV-040 | Medium | Project-directory write failures lacked the application configuration error contract. | Typed writable-directory error with preserved existing files and documented writable installation requirement; I. | Fixed |
| SV-041 | Low | Ruff inspected pip/vendor code inside isolated QA environments. | Exclude runtime `outputs` alongside `.venv`/notebooks; reproduced lint flood and final clean Ruff execution. | Fixed |
| SV-042 | Medium | Delivery verifier hardcoded 32 checks and lacked expanded/fresh-runtime/source evidence. | Require complete 63+ full-scope checks, nine model/frequency combinations, clean install/current hashes, closed ledger and truthful provider source; D. | Fixed |

## Measured tests and verification

| Acceptance check | Executed | Passed | Failed / errors / skipped | Measurement |
|---|---:|---:|---|---|
| Project pytest suite | 160 | 160 | 0 / 0 / 0 | 19.705 s |
| Newly installed isolated pytest suite | 160 | 160 | 0 / 0 / 0 | 21.864 s |
| Complete actual-browser checks | 63 | 63 | 0 failures; 0 JS exceptions | 202.589 s |
| Focused revenue/report browser checks | 7 | 7 | 0 failures | 20.394 s |
| Fresh notebook code cells | 15 | 15 | 0 errors | 10.771 s |
| CLI + notebook artifact hashes | 74 | 74 | 0 mismatches | 37 per experiment |
| Experiment PDFs / workbooks | 2 / 2 | 2 / 2 | 0 parse/value mismatches | 9 pages / 16 sheets each |
| Ruff / dependency consistency | 2 | 2 | 0 failures | Current project interpreter |
| Portable archive members | 153 | 153 | 0 CRC/path/source/credential failures | Current-source byte equality |
| Fresh extracted-app Chrome smoke | 1 | 1 | 0 failures | 305 real records / 374 D periods |

The initial suite had **76 cases**, all passing. Independent pre-fix records contain **80 executed cases: 8 passed, 72 failed, zero errors/skips**, spanning 39 distinct regression functions. These historic failures include parameter variants and successive discoveries. The final suite contains **84 additional cases** and passes in both environments. The audit ledger also includes source-confirmed and actual-browser/tooling defects.

First clean-install attempt encountered a pip checksum mismatch and was preserved as failed environment evidence. A normal pinned retry succeeded; a final **new** environment was then created with only **pip 25.0.1** initially installed. Its pinned install, `pip check`, native imports and full suite passed in **123.272 s** total. No pins were changed and checksum verification was not disabled.

Browser automation failures from collapsed elements or incomplete rerun rendering were investigated using actual screenshots/HTML. Stable rendering is required before chart inspection and downloading; numerical/calendar/selection assertions remain intact. Provider unavailability is recorded independently from passing offline acceptance.

## Verified browser workspaces and controls

| Workspace | Actual verified behavior |
|---|---|
| Overview Dashboard | Empty state, eight genuine KPIs, observed versus unknown data, mean-target label and source invalidation. |
| Dataset Management | Real UCI load/preview, CSV upload/persistence, multi-sheet Excel selection, corrupt workbook rejection, synthetic ancillary joins/provenance. |
| Data Preprocessing | Date/revenue mapping, format/day-first/holiday fields, four business filters, invalid/missing/duplicate/negative/gap decisions, sum/mean arithmetic, inclusive date window, native/coarser cadence and weekly anchors. |
| Exploratory Data Analysis | All tabs, native/coarser charts, trailing windows, distribution/growth/Matplotlib figure, all available segment/holiday views, actual Plotly hover/pan/zoom/reset. |
| Seasonal Analysis | Actual STL, cycle changes, monthly cycle 12, unresolved-input disabled state and descriptive interpretation. |
| Model Training | Immediate empty/restored selection, genuine three-model holdouts, parameter panels, metrics, 0/2/3 folds, known-origin driver confirmation and navigation persistence. |
| Model Comparison | Actual chronological/backtest tables/charts, all model-inspection controls, date-aligned downloads and independent score/prediction matches. |
| Sales Forecasting | All nine model/frequency combinations, preset/custom horizons, full-history snapshots, required driver/holiday uploads, raw/business outputs, history off/on and modified-configuration re-evaluation. |
| Gemini AI Insights | All topic options, question/model persistence, aggregate inspector, unavailable/missing-model fallback, current explanation saving and separate live provider recheck. |
| Reports & Downloads | Genuine CSV/Excel/PDF downloads, source/statistics/model/forecast fidelity, history-only mean-target report and complete fitted summaries. |
| Forecast History | Distinct cached execution IDs, saved CSV/PDF, explanation refresh invalidation, current-dataset filtering, compatible/incompatible comparison and confirmed deletion. |
| About Project | Implemented methodology and navigation/empty/narrow states. |

All twelve pages passed body/main width checks at **390 pixels** and empty-state navigation. Desktop tests use **1440 × 1000**. The [gallery](screenshots/README.md) contains actual captures from the final complete Local analytics run.

## Genuine forecasting and numerical findings

| Calendar | Browser history / evaluation | Actual future verification |
|---|---|---|
| Daily `D` | UCI 374 calendar / 305 observed periods; common 30-day holdout and three 14-day origins. | ARIMA/SARIMA/Prophet 30-day full fits; 7/90/custom-11 horizons and separate business/history toggles. |
| Weekly `W-FRI` | Explicitly synthetic 208-period selected store/department; eight-period holdout, two four-period origins, known promotion drivers and holiday-enabled Prophet. | All three eight-week full fits, exact anchors, real aligned Excel drivers, rejected incomplete files and nonzero genuine Prophet holiday component. |
| Monthly `MS` | Same selected synthetic weekly contributions aggregated into 48 monthly periods; six-month holdout, MAE ranking and cycle-12 STL. | Genuine 3/6/12-month model presets; modified ARIMA re-evaluation and custom nine-month random-walk analytical equality. |

Independent integration regressions also cover all three models on genuine fitted weekly series, exact random-walk ARIMA expectations and corresponding previous-year seasonal-SARIMA predictions. Missing actuals are never scored; backtests end before final holdout; final forecasts refit all history. Raw negative outputs remain available separately from any business constraint. Interval endpoints do not form a calibrated joint-total interval.

For real UCI net revenue, the refreshed CLI, notebook and browser downloads agree at `rtol=1e-5`, `atol=1e-3`:

| Candidate | MAE (GBP) | RMSE (GBP) | MAPE (%) | sMAPE (%) | Bias (GBP) |
|---|---:|---:|---:|---:|---:|
| ARIMA | 14,384.500 | 19,529.031 | 29.658 | 26.746 | −5,150.079 |
| SARIMA | 12,076.296 | **16,669.560** | 26.450 | 22.147 | 4,013.102 |
| Prophet | 12,312.633 | 17,955.768 | **21.492** | 22.147 | −7,436.187 |

Only **26 genuine targets** are scored in the 30-day holdout. **Seven-day seasonal naive RMSE is GBP 15,328.037 and beats SARIMA.** Candidate selection remains correctly SARIMA by RMSE. Thirty full-history SARIMA dates are `2011-12-10`–`2012-01-08`; raw point-total is **GBP 1,693,961.86**. These dates are future relative to the archived source, not today, and lack labelled future outcomes.

Source: Chen, D. (2015), UCI Online Retail, DOI `10.24432/C5BW33`, CC BY 4.0. The original **541,909** invoice lines retain **9,290** negative-revenue lines and **5,268** exact duplicates; the derived sample excludes customer identifiers. The [experimental results](experimental_results.md) preserve the data hashes, configuration, assumptions, fold boundaries and baseline limitation.

## Gemini, database, reports and notebook status

- **Gemini:** official `google-genai==1.26.0`; aggregate-only assembled analytics; bounded requests/retries; structured narrative/citations and actual source/status. Earlier successful `gemini-3.1-flash-lite` generation is preserved in local `gemini-3.1-flash-lite.json`. Final `gemini-audit-final.json` records successful connectivity and `service_unavailable` generation. The latest live-browser attempt also fell back and correctly failed its live-success assertion. The complete 63-check browser run deliberately used Local analytics. This external result remains visible, not counted as live success.
- **SQLite:** transactional IDs/points/snapshots, full-history metadata, compatible common-date comparison, re-save integrity, cascading deletion and corrupt-data errors pass regression/browser checks. Existing corrupt database bytes are preserved; core analytics/forecasting/reporting continue with history disabled.
- **Reports:** actual prediction/metric arrays, observed history, model configurations, holdout and full-history summaries, source/cleaning evidence and current explanations are retained. PDFs/workbooks parse and match exported evidence. Formula-shaped text is escaped; mean-target labels are explicit; stale saved PDFs are invalidated.
- **Notebook:** 15/15 cells executed in order in a fresh local Python 3.12.10 kernel, without global kernel changes, at `2026-11-16T16:26:19.119948+00:00`. CLI/notebook numerical reproduction passed; hosted Colab/Drive branches remain unexecuted.

## Updated file manifest

The timestamp-backed manifest in `outputs/qa/audit-inventory.json` records **49 non-PNG updated paths**, including Git publication configuration and faculty documentation:

```text
.gitattributes                                  added
.gitignore
app.py
config/settings.py
notebooks/Sales_Forecasting_Experiments.ipynb
pyproject.toml
scripts/audit_project.py                         added
scripts/browser_qa.py
scripts/build_faculty_guide.py                    added
scripts/package_project.py
scripts/verify_clean_environment.py              added
scripts/verify_delivery.py
src/ai/gemini_client.py
src/ai/insight_generator.py
src/data/aggregator.py
src/data/loader.py
src/data/pipeline.py
src/data/validator.py
src/database/db_manager.py
src/evaluation/backtesting.py
src/evaluation/metrics.py
src/evaluation/model_comparison.py
src/features/time_features.py
src/forecasting/common.py
src/forecasting/forecast_service.py
src/forecasting/prophet_model.py
src/reports/report_generator.py
src/ui/dashboard.py
src/ui/widgets.py
src/utils/exceptions.py
src/utils/logging_config.py
src/utils/serialization.py
tests/test_audit_regressions.py                   added
tests/test_forecasting.py
tests/test_reports.py
tests/test_ui.py
README.md
docs/FINAL_PROJECT_AUDIT.md                       added
docs/faculty_demonstration_guide.md               added
docs/architecture.md
docs/colab.md
docs/experimental_results.md
docs/installation.md
docs/methodology.md
docs/review3_demo.md
docs/screenshots/README.md
docs/testing.md
docs/user_guide.md
docs/verification_summary.json
```

The **16 regenerated PNG screenshots** and regenerated CLI/notebook outputs are additional generated artifacts. Private/runtime QA files, environments, logs, uploads and databases are excluded from the distributable source count. Source inspection and current-hash checks validate actual files rather than relying on this prose list alone.

## Delivery verification

Archive: `outputs/salesvision_project.zip`. The packaging script includes the complete source/config/tests/guides/diagrams, labelled samples, 16 screenshots, executed notebook, final audit/safe summary and genuine CLI result artifacts. It excludes `.env`, private secrets, virtual environments, caches, runtime databases/logs, uploaded sources and the customer-level workbook.

Final delivery verification **passed**: **153 archive members**, valid CRCs and safe member paths, all required delivery files present, exact byte agreement with current distributable sources, successful Python compilation, blank example credentials and private/runtime/customer-source exclusions. Git attributes preserve these bytes across clones, and generated SQLite history and database-containing notebook bundles remain local. `outputs/qa/delivery-results.json` records the final archive byte size, SHA-256 and detailed verification results.

Fresh temporary-extraction Chrome smoke **passed** using Python **3.12.10**: healthy Streamlit startup, packaged real UCI sample loading, preprocessing **305 records into 374 daily periods**, and no displayed application errors. Gemini was disabled for this isolated smoke workflow. This establishes extracted application execution on the measured machine; installation on another machine or operating system remains untested.

## Remaining external limitations

1. **Gemini provider availability:** final live generation returned service unavailability; retry with available server-side credentials/model when the provider recovers. Core features and labelled fallback pass.
2. **Walmart data unavailable:** legitimate manual acquisition/competition terms review is required at the Kaggle Walmart forecasting data page. No Walmart empirical result is claimed; weekly dimensional fixtures are explicitly synthetic. `test.csv` lacks the target for scored evaluation.
3. **Hosted Colab / physical devices / other operating systems:** provided branches and resized layouts were inspected; these external environments were not executed.
4. **Empirical scope:** one retailer, fixed settings, one final holdout, explicit causal filling and retained duplicates/returns; the seasonal-naive benchmark is stronger. Future accuracy/interval calibration and inventory units need additional outcomes/business inputs.
5. **Upload configuration:** application limit accepts 1–10000 MB; Streamlit HTTP cap defaults to 100 MB. Align both settings for a higher allowed upload size.

No remaining known critical/high-priority application defect is recorded after the implemented fixes. These external/empirical limitations remain explicit.

## Launch and repeat verification

From the extracted `sales-forecasting-system` directory with pinned Python 3.12 dependencies installed:

```powershell
streamlit run app.py
```

Direct interpreter launch:

```powershell
& ".venv\Scripts\python.exe" -m streamlit run app.py
```

The [installation guide](installation.md), [testing commands](testing.md), [user guide](user_guide.md), [notebook instructions](colab.md) and [Review 3 demonstration](review3_demo.md) contain the detailed procedures.
