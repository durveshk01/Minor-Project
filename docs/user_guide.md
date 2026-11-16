# User guide

## A reproducible first run

1. Launch `streamlit run app.py` in the project environment.
2. Open **Dataset Management** and load `UCI real retail · uci_online_retail_daily.csv`.
3. Open **Data Preprocessing**. Map `Date` and `Sales`; use daily **Total sales (sum)**, retain duplicates and negative returns, and choose **Forward-fill (past values only)** for missing periods. Apply preprocessing.
4. Inspect EDA, then run seasonal decomposition with a cycle of **7**.
5. In **Model Training**, evaluate all three models on **30** holdout days, select **RMSE**, and choose **3** rolling-origin folds with horizon **14**. Set Prophet yearly seasonality **Off**, within-week seasonality **On**; retain the other default parameters.
6. Inspect **Model Comparison**. SARIMA should have the lowest candidate RMSE with these inputs.
7. In **Sales Forecasting**, generate **30 days** using SARIMA and save history.
8. Generate a Gemini or Local analytics explanation, build reports, and inspect Forecast History.

These settings reproduce the [recorded UCI experiment](experimental_results.md). The seven-day naive benchmark, shown in the notebook/experiment evidence, outperforms the three-model winner on the final holdout.

## 1. Overview Dashboard

The empty state asks for a dataset rather than inventing KPIs. After preprocessing, eight cards show observed total, mean, maximum, minimum, endpoint growth, future raw forecast total, best evaluated model and its selection score.

Historical statistics exclude unknown periods. Negative net sales are retained according to policy. `—` means an output is not yet available or its calculation is undefined. Currency and compact `K`/`M` labels are presentation formats; downloads retain numerical values.

The **Sales Growth** card compares the first and latest observed periods. The **Forecasted Sales** card sums the displayed raw future point forecasts. Neither is a guaranteed business outcome.

## 2. Dataset Management

### Local samples

Available CSV/Excel files are discovered in `data/sample`, `data/uploads` and `data/source`. The real UCI daily sample includes companion provenance metadata. The weekly synthetic fixture is explicitly identified as test data. Click **Load selected dataset** to activate a source and invalidate earlier current analysis results.

### Uploads

Use the **Upload CSV / Excel** tab, select a `.csv`, `.xlsx` or `.xls`, and choose its worksheet when appropriate. Inspect the preview, then click **Use uploaded dataset**. File uploads are subject to the configured size limits. Invalid Excel workbooks are rejected with a readable error.

Saving uploaded source files into `data/uploads` is optional. If saved, the application sanitizes the filename and prefixes its content hash. Uploaded sources without companion metadata are labelled user-provided; supply their source and currency during preprocessing.

The source preview shows records, columns, exact duplicates, missing cells, sample rows, missingness by column, size and hash.

### Walmart inputs

Legitimately obtain the competition files from the [Kaggle data page](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data). Upload `train.csv` as history; the ancillary uploader accepts `stores.csv` and `features.csv` for checked joins on `Store` and `Store/Date` respectively.

`Weekly_Sales` is a weekly target. Preserve the native week anchor; do not choose daily analysis for weekly observations. `test.csv` has no `Weekly_Sales`, so it cannot provide labelled accuracy evaluation. Future covariates must be supplied in the exact aggregated calendar required by the selected series/model.

## 3. Data Preprocessing

Map a date column and an existing sales column, or select explicit `Quantity × UnitPrice` revenue derivation when those columns exist. Optionally map a real holiday flag and specify a date format or day-first interpretation.

Inspect quality before selecting:

- Invalid-record policy.
- Missing-sales input policy.
- Exact duplicate policy.
- Negative-sales policy.
- Native/coarser forecasting frequency and weekly week-ending day.
- Sum or mean aggregation.
- Missing-time-period policy.
- Available business filters and optional historical date window.
- Optional numeric regressors with appropriate business meaning.

**Apply preprocessing** creates the analysis dataset and an operation-by-operation audit. The preview/download distinguishes `observed_sales`, `model_input` and `is_observed`. Unknown periods remain unknown in the observed output even if explicitly filled for models. Inspect excluded rows and diagnostic-only outlier flags in their expanders.

When an irregular source's cadence cannot be inferred, select **Native source frequency** explicitly. Available target frequencies remain native or coarser. For **Average record sales (mean)**, sums of period averages are not transaction-revenue totals; overview, explanations and report exports identify this interpretation.

Selected policies, filters and date/analysis controls persist when navigating between workspaces. A source or preprocessing change invalidates affected analytical results.

**Keep unknown** is useful for inspection but prevents fitting if gaps remain. **Forward-fill** uses preceding values only. **Zero** requires a real business assumption about missing dates; absence of a record alone does not establish zero sales.

## 4. Exploratory Data Analysis

- **Sales trends**: observed native/coarser rollups, trailing average, month/year comparisons.
- **Distribution & growth**: distributions, period growth, statistical figure and weekday summaries for daily observations.
- **Segments & holidays**: actual available store/department/category/region totals and supplied holiday associations.

Daily/weekday views require daily data. Missing or boundary periods can make summaries incomplete. The UCI daily sample contains no store, department, category, region or holiday flags; appropriate transaction-level fields are required to enable those views.

## 5. Seasonal Analysis

Choose the seasonal cycle in observations and click **Run seasonal decomposition**. At least two cycles are required. Typical cycles are 7 for daily data, 52 for weekly data and 12 for monthly data.

The result shows trend, seasonal and residual components, seasonal strength and 5th–95th percentile amplitude. If model inputs were filled, the page explicitly identifies that assumption. STL describes repeating structure; it does not prove a holiday caused sales changes.

## 6. Model Training

Choose the candidates and common chronological holdout length. Configure orders, seasonal cycle, iterations, Prophet seasonalities/mode/prior, interval width and actual optional regressors/holidays. The selection score defaults to **RMSE**; MAE, sMAPE and MAPE are also supported.

For external regressors, confirm that their evaluation values were known before every origin. This is a substantive data assumption, not a way to admit realized future outcomes.

Rolling-origin folds are optional (0, 2 or 3 in the UI) and restricted to the earlier training partition. Longer cycles, larger holdouts and more folds require more history. Model-specific history/seasonality checks can reject configurations even when the general training page is available.

Click **Train and evaluate models**. The result identifies successful/failed fits and the winner under the requested metric. Fitting failures expose diagnostic messages rather than fabricated predictions.

Candidate selection updates immediately: an empty selection disables training and each selected model exposes its own parameter panel. Submitted parameter choices survive navigation. Holdouts are restricted to supported integer horizons and sufficient remaining history.

## 7. Model Comparison

Review MAE, RMSE, MAPE, sMAPE, signed bias, scored-target counts, actual fitting time and status. Inspect holdout plots and downloadable date-aligned prediction tables. Unknown actuals are shown but excluded from scoring; zeros are excluded only from MAPE.

The configuration/summary expander includes the actual fitted model diagnostics and recorded warnings/retries. Optional backtesting has separate model and fold tables.

The current selector ranks the evaluated ARIMA/SARIMA/Prophet candidates. Consult the experiment/notebook's naive benchmarks when interpreting whether model complexity improved performance.

## 8. Sales Forecasting

Select a successfully evaluated model. Standard horizons are:

| Frequency | Standard future horizons |
|---|---|
| Daily | 7, 30, 90 days |
| Weekly | 4, 8, 12, 26 weeks |
| Monthly | 3, 6, 12 months |

**Custom** accepts 1–366 periods. Configured external regressors require a CSV/Excel with every exact future date and every selected column; values must already be known. Supply an actual future `ds, holiday` calendar when modelling future holiday effects.

Future holiday dates must match the selected forecast calendar exactly: daily dates, the selected weekly week-ending dates, or month starts. A holiday-enabled model without a future event calendar explicitly warns that future holidays have not been supplied.

Optional controls enable a separate nonnegative planning view and history saving. **Generate future forecast** fits on full history or reuses the identical validated numerical computation. Each click creates a distinct execution ID/time. Changing inputs alone does not regenerate the result; read the **Displayed forecast** model/horizon/date caption. Model and horizon controls restore the last forecast after navigation.

The page shows raw total, average future period sales, conditional equal-window growth, interval chart, full prediction table, summaries/components and CSV/Excel downloads. Growth is unavailable when the preceding equal-length history is incomplete. Individual interval endpoints cannot be summed into a calibrated total interval.

If customizing the model changes its evaluated configuration, use **Re-evaluate modified configuration** before final forecasting.

## 9. Gemini AI Insights

The credential is read server-side; the page lets you configure the model name and test connectivity. Select an explanation topic or ask a question about the computed analytics. The free-text question is also sent to Gemini; keep it focused on the displayed aggregate evidence.

Click **Generate business explanation**. Check the actual **Explanation source**:

- **Gemini**: structured narrative generated by the provider from aggregate analytics.
- **Local analytics**: deterministic Python explanation, with a readable configuration/provider failure status.

The aggregate-context expander shows the evidence used. Explanations are advisory and should be reviewed against it. New analysis invalidates stale text. If a forecast exists, **Save current explanation with forecast history** refreshes that execution's aggregate report snapshot.

## 10. Reports & Downloads

Click **Build report package** for actual current results. CSV contains the selected forecast, or observed history if no forecast exists. Excel includes available provenance, quality, audit, statistics, seasonality, model comparison, holdout predictions, configuration/summary, forecast and explanation sheets. The illustrated PDF summarizes those outputs and their assumptions.

Run EDA/STL, model evaluation, final forecasting and explanation generation before building a full demonstration report. Downloads retain raw numerical predictions and use spreadsheet-safe text values. Downloading does not retrain models.

For mean-aggregated targets, PDF and Excel explain the target and label historical/monthly sums as **`sum_of_period_averages`**. Full-history fitted summaries are retained and long PDF summary lines wrap completely.

## 11. Forecast History

History persists in the configured SQLite database. You can filter to the current dataset, inspect a previous execution's points/configuration/metrics, download its forecast CSV, and generate a PDF from its saved report snapshot.

Select 2–4 saved executions to compare. They must have the same dataset/preprocessing/frequency fingerprint and overlapping future dates. The plot/table uses only their common dates. Holdout scores stored with a forecast refer to its earlier evaluation, not accuracy on future unknown dates.

Deletion requires the explicit confirmation checkbox and deletes that execution's saved points/report. Reloading source data or restarting Streamlit clears current in-memory analysis but preserves other saved executions.

Re-saving an explanation refreshes its saved report snapshot and invalidates any previously generated PDF for that execution. If the database cannot initialize, history controls are unavailable while current analytics, forecasts and reports continue.

## 12. About Project

This page summarizes the problem, implemented methodology, technology stack, supported sources, interpretation and shared notebook/script reproducibility. Use it to introduce the system during Review 3.

## Practical interpretation

- Native sales frequency determines what behavior can be observed.
- Gross sales, net revenue, average record value and item demand are distinct targets.
- Low holdout error is evidence for that window, not a promise about future sales.
- UCI's forecast dates belong to the archived study; they are not today's forecast dates.
- Revenue forecasts support scenario review, but inventory quantities require product mix, prices, demand units and lead times.

See [methodology](methodology.md) for formulas and [installation](installation.md) for environment/provider troubleshooting.
