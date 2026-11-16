"""Generate a reproducible, explicitly synthetic weekly test dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_weekly_data(periods: int = 208, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index, date in enumerate(pd.date_range("2021-01-01", periods=periods, freq="W-FRI")):
        holiday = date.month == 12 and date.day >= 17
        seasonal = 1600 * np.sin(2 * np.pi * index / 52)
        for store in (1, 2):
            for department in (1, 2, 3):
                sales = 8500 + store * 1300 + department * 650 + index * 18 + seasonal
                sales += int(holiday) * 2300 + rng.normal(0, 350)
                rows.append({
                    "Store": store, "Dept": department, "Date": date,
                    "Weekly_Sales": round(sales, 2), "IsHoliday": holiday,
                    "Dataset_Type": "SYNTHETIC — application testing only",
                })
    return pd.DataFrame(rows)


def main() -> None:
    target = ROOT / "data/sample/synthetic_weekly_sales.csv"
    if target.exists():
        print(f"Existing synthetic file preserved: {target}")
        return
    frame = generate_weekly_data()
    frame.to_csv(target, index=False, mode="x")
    metadata = {
        "name": "Synthetic weekly retail test data",
        "kind": "synthetic",
        "source_url": "Locally generated using NumPy seed 42",
        "currency": "sales units",
        "license": "Project-generated testing fixture",
        "description": "208 Fridays × 2 stores × 3 departments. Simulated trend, annual cycle, holiday uplift and Gaussian noise. NOT Walmart data and NOT empirical business findings.",
    }
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {len(frame)} explicitly synthetic transaction rows over 208 weekly dates.")


if __name__ == "__main__":
    main()
