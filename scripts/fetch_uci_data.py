"""Download the CC BY 4.0 UCI retail source and derive recorded daily revenue."""

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def main() -> None:
    source = ROOT / "data/source/Online Retail.xlsx"
    target = ROOT / "data/sample/uci_online_retail_daily.csv"
    if not source.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "SalesVision-Academic/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive_bytes = response.read(50_000_001)
        if len(archive_bytes) > 50_000_000:
            raise ValueError("The source archive exceeds the expected download size.")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(name for name in archive.namelist() if name.endswith("Online Retail.xlsx"))
            if archive.getinfo(member).file_size > 60_000_000:
                raise ValueError("The workbook is larger than expected.")
            with source.open("xb") as handle:
                handle.write(archive.read(member))
    frame = pd.read_excel(source)
    required = {"InvoiceDate", "Quantity", "UnitPrice"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected UCI schema: {list(frame.columns)}")
    dates = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    sales = pd.to_numeric(frame["Quantity"], errors="coerce") * pd.to_numeric(
        frame["UnitPrice"], errors="coerce"
    )
    if dates.isna().any() or sales.isna().any():
        raise ValueError("Unexpected invalid dates or revenue: inspect the source before deriving data.")
    transactions = pd.DataFrame({"Date": dates.dt.normalize(), "Sales": sales})
    daily = transactions.groupby("Date").agg(
        Sales=("Sales", "sum"), Transactions=("Sales", "size")
    ).reset_index()
    if not target.exists():
        daily.to_csv(target, index=False, float_format="%.4f", mode="x")
    metadata = {
        "name": "UCI Online Retail — recorded daily net revenue",
        "kind": "real",
        "source_url": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "download_url": URL,
        "citation": "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. DOI: 10.24432/C5BW33.",
        "license": "CC BY 4.0",
        "currency": "GBP",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(frame),
        "source_columns": list(frame.columns),
        "date_start": dates.min().isoformat(),
        "date_end": dates.max().isoformat(),
        "derived_rows": len(daily),
        "negative_transaction_count": int((sales < 0).sum()),
        "exact_source_duplicates": int(frame.duplicated().sum()),
        "derivation": "Sales = Quantity × UnitPrice; sum every transaction by invoice calendar date. Refunds, cancellations and exact source duplicates are retained. Customer identifiers are excluded from the derived file.",
        "missing_dates": "Dates without source transactions are absent, not automatically filled. Choose an explicit missing-period strategy in preprocessing.",
    }
    metadata_path = target.with_suffix(".metadata.json")
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(daily.head().to_string(index=False))


if __name__ == "__main__":
    main()
