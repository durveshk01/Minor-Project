# Executed UCI Online Retail experiment

Real dataset: UCI Online Retail — recorded daily net revenue (CC BY 4.0; GBP net revenue).
History: 2010-12-01 to 2011-12-09; 305 observed dates in 374 calendar days.
69 unknown days use explicit causal forward-fill for model inputs only. Observed historical totals and scored targets retain the original unknowns.
Observed net revenue: GBP 9,747,747.93.

## Prespecified evaluation

Training: 344 calendar days, ending 2011-11-09.
Common holdout: 30 calendar days, 2011-11-10 to 2011-12-09.
No random split or holdout-directed parameter search was used. The winner is selected among ARIMA, SARIMA and Prophet by lowest valid holdout RMSE.

| Model | MAE (GBP) | RMSE (GBP) | MAPE (%) | sMAPE (%) | Bias (GBP) | Scored dates |
|---|---:|---:|---:|---:|---:|---:|
| ARIMA | 14,384.500 | 19,529.031 | 29.658 | 26.746 | -5,150.079 | 26 |
| SARIMA | 12,076.296 | 16,669.560 | 26.450 | 22.147 | 4,013.102 | 26 |
| Prophet | 12,312.633 | 17,955.768 | 21.492 | 22.147 | -7,436.187 | 26 |

Selected model: **SARIMA**, holdout RMSE **GBP 16,669.560**.
Final SARIMA is refitted on all 374 calendar days. Its 30-day forecast spans 2011-12-10 to 2012-01-08 and has a raw point-forecast total of GBP 1,693,961.86.

## Baseline benchmarks

| Baseline | MAE (GBP) | RMSE (GBP) |
|---|---:|---:|
| Last-value naive | 15,008.011 | 19,868.400 |
| Seven-day seasonal naive | 12,353.891 | 15,328.037 |

The selected model **does not beat** the stronger naive baseline on this holdout. Baselines are benchmarks, not candidates in the application's three-model selection.

## Training-partition rolling-origin results

These separate earlier windows do not replace the final holdout selection.

| Model | RMSE (GBP) | Completed folds | Status |
|---|---:|---:|---|
| ARIMA | 15,668.915 | 3 | success |
| SARIMA | 13,738.304 | 3 | success |
| Prophet | 14,887.998 | 3 | success |

## Interpretation and limits

- This archived dataset represents one UK online retailer; these are not Walmart findings.
- Missing dates are unknown; causal forward-fill is a documented modelling assumption. STL diagnostics also use that filled series and are descriptive, not causal.
- Source returns/cancellations and exact duplicates are retained, so the target is recorded net revenue.
- The final forecast is future relative to the dataset's last date, not a forecast for today's calendar.
- About one year of history does not identify yearly Prophet seasonality; it is explicitly disabled.
- MAPE can be large for small or negative net-sales days; zero targets are excluded only from MAPE.
- Parameters are fixed examples, not proof of globally optimal models; one holdout does not establish general superiority.
- Model-based period intervals are not guaranteed coverage or a joint interval for total revenue.
- Revenue predictions alone cannot determine stock quantities, prices, product mix or lead times.
- No future labelled targets exist for the final forecast; no future accuracy claim is made.

Source: https://archive.ics.uci.edu/dataset/352/online+retail

Citation: Chen, D. (2015). Online Retail. UCI Machine Learning Repository. DOI: 10.24432/C5BW33.
