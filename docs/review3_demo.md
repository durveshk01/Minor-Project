# Review 3 demonstration guide

## Project statement

**Title:** Sales Forecasting Using Python  
**Application:** SalesVision — Sales Intelligence & Forecasting System  
**Domain:** Business Analytics  
**Demonstrated case:** Actual UCI Online Retail daily GBP net revenue

The system integrates explicit data-quality decisions, observed EDA, robust STL, genuine ARIMA/SARIMAX/Prophet forecasts, chronological comparison, full-history retraining, optional grounded Gemini explanations, illustrated reports and persistent forecast history.

## Delivery items

- Portable `outputs/salesvision_project.zip` containing sources, pinned requirements, documentation, diagrams, screenshots, samples, executed notebook and CLI evidence.
- Streamlit entry point: `app.py`; launch command: **`streamlit run app.py`**.
- Executed notebook: `notebooks/Sales_Forecasting_Experiments.ipynb`.
- Actual results: `outputs/experiments/uci_daily/results.md`, CSVs, manifest and figures.
- Illustrated PDF and Excel report in that experiment directory.
- [Architecture/data-flow/ER diagrams](architecture.md), [methodology](methodology.md), [experimental results](experimental_results.md), [testing evidence](testing.md), and [screenshot gallery](screenshots/README.md).
- [Final project audit](FINAL_PROJECT_AUDIT.md), with root-cause fixes, source counts, 160-test results in two environments and 63 actual-browser checks.

Runtime credentials, customer-level workbook and local history are not part of the portable delivery. The derived UCI sample carries its attribution and CC BY 4.0 provenance.

## Demonstration preparation

Install the pinned Python 3.12 requirements and start Streamlit from the project directory. Verify that `uci_online_retail_daily.csv` appears in the local catalogue. Configure an available Gemini key/model server-side if needed; the entire numerical demonstration works with labelled Local analytics. The final audit's live generation recheck returned temporary provider unavailability despite successful model lookup, so prepare to demonstrate the labelled fallback.

Open the saved experimental findings and baseline table for discussing performance. Have the actual PDF/Excel and screenshot gallery available as executed evidence.

## Suggested demonstration sequence

### 1. Introduce the problem and system

Open **About Project**, then the overview's initial state. Explain the need for reproducible forecasts, visible cleaning assumptions, fair comparisons and uncertainty-aware planning. Show the architecture diagram and shared Python services.

### 2. Establish real data and provenance

In **Dataset Management**, load the UCI daily sample. Show source/citation, records, columns and hash. Explain that the original workbook has **541,909** invoice lines and the daily sample has **305** recorded dates. The target is recorded net revenue and retains returns/cancellations/duplicates.

State the actual dataset scope: one UK online retailer, not Walmart. Walmart support is implemented, but competition files were unavailable for this empirical run.

### 3. Make preprocessing assumptions visible

Select `Date`, `Sales`, daily sum, retained duplicates/negative returns, and **Forward-fill (past values only)**. Apply preprocessing and show the audit plus observed/model-input distinction.

Expected calendar: **374 days**, **305 observed**, **69 unknown**. Explain why filling model inputs does not invent genuine targets, why unknown holdout days remain unscored, and why outliers are only flagged.

### 4. Demonstrate EDA and STL

Show observed trends, distributions, monthly comparisons and weekday analysis. Run a **7-observation** robust STL cycle; seasonal strength is approximately **0.386** under the explicit input policy. Mention missing-data and descriptive/causal limits.

### 5. Fit and compare genuine models

Use all three candidates, **30-day holdout**, **RMSE**, **3** rolling-origin folds, **14-day** fold horizon. Keep ARIMA `(1,1,1)` and SARIMA `(1,1,1)(0,1,1,7)`. For Prophet, use additive mode, yearly **Off**, weekly **On**, prior `0.05` and 95% intervals.

Train, show statuses, actual holdout curves, model summaries and earlier backtest tables. Expected candidate RMSEs:

| Candidate | RMSE (GBP) |
|---|---:|
| ARIMA | 19,529.031 |
| SARIMA | **16,669.560** |
| Prophet | 17,955.768 |

All models score the same **26 observed** targets in the 30-day holdout. The selection is SARIMA by the declared criterion; Prophet has the lowest MAPE.

### 6. Discuss the baseline honestly

Open the notebook or saved `baseline_comparison.csv`: seven-day seasonal naive RMSE is **GBP 15,328.037**. It beats SARIMA on the final holdout. Explain that the application selects among its three candidates and the benchmark reveals a current modelling limitation. This is a useful empirical finding, not an improvement to hide or invent.

### 7. Generate the full-history forecast

Generate **30 days** using SARIMA, with history saving enabled. Show **374** full-history training observations, future dates **2011-12-10–2012-01-08**, raw point-total **GBP 1,693,961.86**, table and interval chart.

Navigate away/back to demonstrate control restoration and the displayed-forecast caption. Generate another model or repeat SARIMA to show distinct saved execution IDs even when numerical computation is cached.

State that dates are future relative to the archived data and have no observed outcomes for accuracy measurement. Period intervals cannot be summed into a calibrated revenue-total interval.

### 8. Explain computed evidence

Open **Gemini AI Insights**, inspect the aggregate context and generate an executive/model/forecast explanation. Show its actual **Gemini** or **Local analytics** source. Save the explanation with the existing execution. Describe the separation between Python numerical models and narrative explanation.

### 9. Export and demonstrate persistence

Build/download CSV, Excel and illustrated PDF reports. Open the workbook's Model Comparison and Forecast sheets and the PDF's figures. In **Forecast History**, inspect the saved snapshot, regenerate its PDF, compare overlapping compatible runs and demonstrate the required deletion confirmation.

### 10. Close with evidence and limitations

Return to the eight overview KPI cards. Show the notebook's **15/15** successful local code cells, artifact manifest/checksums, **160/160** tests in both project and freshly installed environments, **63** actual-browser checks and real screenshots. Daily/weekly/monthly ARIMA, SARIMA and Prophet workflows were verified; weekly dimension/driver/holiday checks use explicitly synthetic data.

Distinguish implemented software features from unverified external branches: hosted Colab and physical mobile devices were not executed; Walmart empirical analysis requires the legitimate files. Future research includes more temporal windows, baseline selection, gap sensitivity and calibrated intervals.

## Presentation outline

1. Problem statement and objectives.
2. Data source, licence and net-revenue target.
3. Architecture and shared services.
4. Audited preprocessing and unknown-target handling.
5. EDA/STL findings and assumptions.
6. Model specifications and chronological validation.
7. Actual candidate and naive-baseline results.
8. Full-history forecast, intervals and explanation source.
9. Reports, persistence, tests and browser evidence.
10. Limitations, future work and conclusion.

## Viva questions and accurate answers

| Question | Answer |
|---|---|
| Why not randomly split sales dates? | A temporal holdout reproduces forecasting from earlier to later dates and avoids mixing future data into training. |
| How are missing dates handled? | Original unknowns are preserved; an explicit policy supplies model inputs. Unknown targets never count as genuine observations in scoring. |
| Why retain negative revenue? | Returns/cancellations can produce negative net sales. Removing them changes the target and requires an explicit audited policy. |
| Is SARIMA universally best? | It has the lowest candidate RMSE on this holdout. Seasonal naive is stronger here; other criteria/windows can change the result. |
| Why disable yearly Prophet terms? | The case has approximately one year; the implementation requires two years of training history to enable yearly seasonality. |
| Does Gemini forecast sales? | Numerical predictions come from Statsmodels/Prophet. Gemini explains aggregate Python evidence and can fall back to deterministic local text. |
| Are the future values accurate? | No labelled future outcomes exist here; accuracy is measured only on historical validation/holdout targets. |
| Can this set inventory units? | Revenue alone lacks SKU mix, unit prices, demand quantities and lead-time information. |
| Are these Walmart results? | No. They are actual UCI findings; Walmart file support is implemented but the competition files were not available. |
| Was Colab tested? | The notebook ran locally with all code cells successful. Hosted upload/Drive branches are provided and remain unverified. |

## Conclusion

SalesVision demonstrates a working, reproducible end-to-end forecasting application with genuine outputs and visible assumptions. Its main Review 3 contribution is the integration of auditable preparation, temporal evaluation, full-history forecasts, grounded explanation, usable reports and saved history, supported by executed evidence and an honest baseline comparison.
