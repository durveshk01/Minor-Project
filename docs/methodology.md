# Forecasting methodology

## Problem and objectives

Historical sales can contain gaps, returns, duplicate source lines, multiple business segments, and inconsistent date formats. A useful planning tool must make these assumptions visible, compare models fairly, and distinguish measured historical error from uncertain future outcomes.

SalesVision's objectives are to produce auditable data preparation, observed business analysis, genuine statistical/model forecasts, chronological evaluation, explainable outputs, and reproducible reports. The case study is recorded UCI daily net revenue; store-level Walmart analysis requires separately obtained competition files.

## 1. Define and validate the target

Users map a date and existing sales column, or explicitly derive revenue as `Quantity × UnitPrice`. Numeric conversion rejects non-finite values. Dates and date-window bounds are converted consistently through UTC into timezone-naive calendar dates, with optional explicit format/day-first interpretation. File hashes, source citation/license, mapping and policy choices are retained.

The real UCI derivation sums invoice-line revenue by invoice calendar date. It retains refunds/cancellations and exact source duplicates, producing **recorded net revenue**, not gross sales or item demand. Customer identifiers are excluded from the distributed daily sample.

Quality inspection measures invalid/missing dates and sales, duplicates, negative/zero values and missing cells. Rows are excluded only through selected policies, with original source-row identifiers and reasons in the rejection audit.

## 2. Cleaning, filtering and aggregation

| Issue | Implemented choices |
|---|---|
| Invalid records | Stop/review; explicitly exclude and audit |
| Missing sales | Stop/review; retain as unknown; exclude and audit |
| Exact duplicates | Retain; explicitly remove exact duplicates |
| Negative sales | Retain as net returns; stop/review; exclude and audit |
| Missing calendar periods | Keep unknown; causal forward-fill; explicitly assume zero; stop/review |
| Business scope | Store/department/category/region filters when available; optional date window |
| Target aggregation | Sum; or explicitly labelled mean record sales per period |

Filtering follows validation and selected cleaning policies in the audit. Data are stably sorted chronologically, then aggregated at the native or a coarser frequency. Weekly source observations are not expanded into fabricated daily sales. Weekly anchors are explicit (`W-MON` through `W-SUN`); monthly model dates use month starts (`MS`).

Irregular source cadence requires an explicit native-frequency decision. Mean aggregation produces average retained-record sales within each period. Sums of those period averages, including monthly comparisons, are not total transaction revenue; report exports use the label `sum_of_period_averages` and state the target definition.

A period containing unknown sales inputs remains incomplete/unknown instead of silently summing only its known contributions. A continuous calendar identifies both absent and incomplete periods. The processed output preserves:

- `observed_sales`: original complete observed period values, with unknowns retained.
- `model_input`: the explicitly processed series used for fitting.
- `is_observed`: whether an original complete target exists.

Forward-fill uses only preceding values. It is an assumption about model inputs, not an assertion that an unrecorded date had that sales value. Unknown holdout targets remain unscored regardless of the input policy. Keeping unresolved unknowns prevents model fitting until a complete modelling calendar is supplied.

IQR outliers are **diagnostic flags only**; they do not remove, replace, or clip observations. This avoids fitting a full-history outlier transformation that leaks held-out distributional information.

## 3. Observed exploratory and seasonal analysis

Historical totals, averages, extrema, distributions, growth, rollups, and available business-segment/holiday comparisons use observed data. Rollup boundaries and periods with missing observations can be incomplete. Monthly totals from weekly observations remain summaries of those recorded weekly contributions, not inferred daily transactions.

First-to-latest growth is `100 × (last - first) / abs(first)` across observed endpoints, undefined when the first observation is zero. Period-to-period growth is undefined after a zero or unknown prior period. These are descriptive calculations, not causal trend estimates.

Robust **STL** decomposes the explicitly processed series into trend, seasonal and residual components. At least two selected cycles are required. Standard diagnostic cycles are 7 daily observations, approximately 52 weekly observations, or 12 monthly observations.

Seasonal strength is based on residual and seasonal-plus-residual variance, bounded to `[0,1]`. Seasonal amplitude is the difference between the seasonal component's 95th and 5th percentiles. Filled-input STL remains descriptive and must be interpreted with the missing-period audit; neither a seasonal component nor a holiday comparison proves causality.

## 4. Prespecify candidate configurations

The case study fixes these configurations before the final holdout comparison:

| Candidate | Configuration |
|---|---|
| ARIMA | `(p,d,q)=(1,1,1)`, optimizer iterations 150 |
| SARIMA/SARIMAX | `(1,1,1)(0,1,1,7)`, optimizer iterations 150 |
| Prophet | Additive; weekly on, yearly off, intraday off; changepoint prior 0.05 |
| Shared | Seed 42; interval width 95%; no fabricated regressors/holidays |

These are reproducible example settings, not the result of a claimed exhaustive parameter search. Interactive changes are explicit. A modified final model configuration must be re-evaluated on the same holdout before forecasting.

### ARIMA and SARIMA

ARIMA combines autoregressive terms, differencing and moving-average errors. SARIMA adds seasonal terms; Statsmodels **SARIMAX** implements the seasonal model and optionally aligned external regressors.

Sales are scaled for statistical optimization by `max(training.std(ddof=0), 1.0)`, learned from the current training partition and restored to original units for prediction. SARIMAX regressor means/scales are also learned from that training partition and reused for its forecast values.

Nonconverged LBFGS fits retry the same specification with Powell. Retries and warnings are recorded; a fit that still fails convergence is reported as failed rather than assigned fabricated metrics. SARIMAX enforces stationarity/invertibility and checks minimum history for the configured seasonal cycle.

### Prophet

The official `prophet` package fits trend, changepoints and supported seasonal components. Aggregated sales cannot identify intraday seasonality, so it is disabled. Within-week seasonality requires daily observations spanning at least two weeks. The implemented yearly-history threshold is **730 daily, 104 weekly or 24 monthly observations**, with appropriate history span. Exact boundary and insufficient-history regressions verify it.

The approximately one-year UCI case study therefore explicitly disables yearly terms. Multiplicative seasonality is rejected for zero/negative sales; additive mode is appropriate for this net-revenue target. Prophet predictive intervals use 500 seeded uncertainty draws. A lock and restored NumPy RNG state keep these simulations reproducible without changing other application random state.

## 5. Chronological evaluation and leakage prevention

The final `test_size` periods are held out; every selected model gets the same earlier training calendar and predicts the same future holdout dates. No random time-series split is used.

For UCI:

```text
2010-12-01 ───────────── 2011-11-09 | 2011-11-10 ───── 2011-12-09
       344 training calendar days | 30 holdout days; 26 observed targets
```

Predictions are joined by the exact holdout calendar. Only finite original observed targets are scored. Evaluation retains all holdout prediction rows and indicates unavailable actuals, so excluded targets remain visible.

Optional rolling-origin validation uses expanding training prefixes and subsequent windows **inside the training partition**. The case study uses three nonoverlapping 14-day windows. Each fit learns its own statistical scaling; earlier-window scores do not replace the final holdout candidate selection.

External regressors require an explicit assertion that evaluation-period values were known at each origin. Mapped targets and their revenue inputs are forbidden regressors. Incomplete aggregate driver periods remain unknown. Final forecasting requires exact chronological coverage of every required date and all configured columns, without automatic exogenous filling. Historical holidays must match training-calendar periods; future holidays must match daily dates, the selected weekly anchor or month starts. Historical flags do not invent future events, and a missing future calendar is explicitly identified.

## 6. Error metrics and selection

Let `e_i = prediction_i - actual_i` for the `n` scored observed targets.

| Metric | Definition | Interpretation |
|---|---|---|
| MAE | `mean(abs(e_i))` | Typical absolute error in sales units |
| RMSE | `sqrt(mean(e_i²))` | More sensitive to large errors; sales units |
| MAPE | `100 × mean(abs(e_i)/abs(actual_i))` over nonzero actuals | Relative error; unstable near zero |
| sMAPE | `mean(200 × abs(e_i)/(abs(actual_i)+abs(prediction_i)))` | Symmetric percentage scale; both-zero pair contributes zero |
| Bias | `mean(e_i)` | Positive means average overprediction; negative underprediction |

MAPE is unavailable if every scored actual is zero. Zero targets still contribute to MAE, RMSE, sMAPE and bias. Metrics record scored observations, MAPE observations and zero-actual counts. Actual/predicted indexes must agree.

Selection uses the **lowest valid requested error** among successful candidates. Default: RMSE; supported alternatives: MAE, sMAPE and MAPE. Ties use model-name order. Failed models and invalid scores do not become winners.

The experiment also evaluates last-value naive and seven-day seasonal-naive baselines using the same train/holdout and observed target mask. These benchmarks reveal whether added model complexity helps. They are not candidate models in the current three-model dashboard selector.

## 7. Full-history forecast and uncertainty

After evaluation, the chosen model is refitted on all historical modelling periods, with the evaluated configuration. Forecast dates start strictly after the final historical period. The library accepts 1–366 periods; standard UI horizons depend on frequency.

Statsmodels returns model-based intervals; Prophet uses its predictive simulation. Intervals describe individual future periods under model assumptions, not guaranteed empirical coverage. Summing their lower/upper endpoints does not create a valid joint total-revenue interval.

Raw forecasts can be negative for a net-sales target. An optional nonnegative planning view adds separately labelled constrained columns while retaining original predictions and errors. Clipping changes interpretation and interval calibration.

Forecast growth compares the raw future point-total to the immediately preceding equal-length **fully observed** historical window. It is unavailable if that window is incomplete, too short, or sums to zero. Summed period averages are not transaction revenue when mean aggregation is selected.

## 8. Explanation and reproducibility

Gemini receives aggregate statistics, decomposition diagnostics, measured holdout errors and computed forecast summaries. It is instructed to explain this evidence and acknowledge missing information. Structured JSON, narrative bounds and available cited-section names are validated. It cannot alter the Python model output.

Classified failures return deterministic **Local analytics** with the actual failure status. The current explanation's context hash prevents stale narratives entering a later report. Requests have bounded timeouts/retries; logs expose status labels rather than raw provider credential-bearing errors.

Reproducibility records source/derived hashes, policy and parameter choices, versions, seed, split boundaries, actual model summaries/warnings, measured fitting duration and artifact checksums. Numerical comparisons use stated floating-point tolerances; timestamps, execution IDs and report bytes can change across reruns.

## Limitations and future investigation

- One archived retailer and one final holdout cannot establish general retail forecasting superiority.
- Forward-filled gaps and retained duplicates/returns are explicit target assumptions requiring sensitivity analysis.
- Current UCI history cannot support yearly Prophet seasonality; more history is needed.
- The stronger seasonal-naive baseline outperforms the selected SARIMA on this holdout.
- Future point forecasts have no labelled outcomes here, so future accuracy and interval coverage remain unmeasured.
- Revenue alone cannot determine SKU quantities, product mix, prices or lead times.
- Legitimate Walmart data, more retailers, additional temporal windows, baseline selection, interval calibration, hierarchical forecasting and known-at-origin planned covariates are useful future extensions.

## References

1. Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository. [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33), CC BY 4.0.
2. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition. <https://otexts.com/fpp3/>.
3. Statsmodels ARIMA documentation: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>.
4. Statsmodels SARIMAX documentation: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>.
5. Prophet documentation: <https://facebook.github.io/prophet/docs/quick_start.html>.
6. Statsmodels STL documentation: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html>.
7. Official Google Gen AI Python SDK: <https://googleapis.github.io/python-genai/>.
