# Dataset provenance

- `synthetic_weekly_sales.csv` is a deterministic testing fixture, not Walmart data.
- The included `uci_online_retail_daily.csv` is actual recorded daily net revenue from the UCI Online Retail dataset. `python scripts/fetch_uci_data.py` provides reproducible acquisition/derivation. Its companion metadata records the license, source hash and source quality counts.
- User datasets and Walmart competition files belong in `data/uploads/` or can be uploaded through Streamlit.
- Missing dates are not silently treated as zero sales. The application asks for an explicit strategy.

## Preferred Walmart data

Visit https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data, sign in, review and accept the current competition rules where permitted, then download `train.csv`, `test.csv`, `stores.csv` and `features.csv`. Extract them locally and upload the files in Dataset Management. `test.csv` has no `Weekly_Sales`; it can supply known future covariates but cannot be used as historical sales or as a scored test set.

Walmart dates represent **weekly observations**. The supported starting series is total `Weekly_Sales` across Store and Dept per Date. `stores.csv` joins on Store; `features.csv` joins on Store and Date, with cardinality validation and an audit record.

## Openly licensed real-data validation

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. License: CC BY 4.0.

The actual source has 541,909 invoice-line records for a UK-based online retailer, from December 2010 to December 2011. Revenue is `Quantity × UnitPrice` in GBP. The generated daily file sums all source lines, retains returns/cancellations and duplicates, and excludes customer identifiers. It is a secondary retail dataset, not a substitute labeled as Walmart.

The derived file contains **305 recorded dates** between `2010-12-01` and `2011-12-09`. There are **69 absent dates** in the 374-day calendar. The executed experiment explicitly forward-fills them for model inputs only; observed statistics and holdout scoring retain unknown actuals. The source includes **9,290 negative-revenue lines** and **5,268 exact source duplicates**, both retained.

Original workbook SHA-256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.

Derived CSV SHA-256: `d9b2e2d967602d1aa41fcdb48701ba5d5bc07c4bed056ce21e8f23eebaaeef5c`.

Changes from the source are calendar-date normalization, daily revenue aggregation, a recorded line-count column, and removal of customer-level fields. Preserve the Chen/UCI citation and [CC BY 4.0 attribution](https://creativecommons.org/licenses/by/4.0/) when redistributing the derived data. The original workbook is local-only in `data/source/` and is excluded from the project archive.

## Executed study and testing fixture

See [experimental results](../../docs/experimental_results.md) for the genuine temporal evaluation, baseline comparison and archived future forecast. There are no labelled outcomes for that final future horizon.

`synthetic_weekly_sales.csv` uses seed **42**, **208 Friday observations**, **2 stores** and **3 departments** (**1,248 records**). It supports deterministic software checks and is explicitly labelled synthetic in the application. Its results must not be presented as real retail or Walmart findings.
