"""CSV/Excel ingestion and validated Walmart ancillary-file joins."""

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.data.validator import convert_dates, convert_holiday_flags
from src.utils.exceptions import DataValidationError


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    filename: str
    sha256: str
    size_bytes: int
    sheets: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)


def load_dataset(
    source: str | Path | bytes | BinaryIO,
    filename: str | None = None,
    sheet_name: str | int = 0,
    encoding: str = "utf-8-sig",
    max_size_mb: int = 100,
) -> LoadedDataset:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataValidationError("The selected dataset file does not exist.")
        if path.stat().st_size > max_size_mb * 1024**2:
            raise DataValidationError(f"The file exceeds the {max_size_mb} MB limit.")
        filename = filename or path.name
        content = path.read_bytes()
    elif isinstance(source, bytes):
        content = source
    else:
        position = source.tell()
        source.seek(0)
        content = source.read(max_size_mb * 1024**2 + 1)
        source.seek(position)
        filename = filename or getattr(source, "name", None)
    if not filename:
        raise DataValidationError("Supply the uploaded filename including its extension.")
    if not content:
        raise DataValidationError("The uploaded file is empty.")
    if len(content) > max_size_mb * 1024**2:
        raise DataValidationError(f"The file exceeds the {max_size_mb} MB limit.")
    extension = Path(filename).suffix.lower()
    sheets: tuple[str, ...] = ()
    try:
        if extension == ".csv":
            header = next(row for row in csv.reader(io.StringIO(content.decode(encoding))) if row and not (len(row) == 1 and not row[0].strip()))
            if len(header) != len(set(header)):
                raise DataValidationError("Duplicate CSV column names must be made unique.")
            frame = pd.read_csv(io.BytesIO(content), encoding=encoding, low_memory=False)
        elif extension in {".xlsx", ".xls"}:
            with pd.ExcelFile(io.BytesIO(content)) as workbook:
                sheets = tuple(workbook.sheet_names)
                header = workbook.parse(sheet_name=sheet_name, header=None, nrows=1).iloc[0]
                present_names = header.dropna().astype(str).tolist()
                if len(present_names) != len(set(present_names)):
                    raise DataValidationError("Duplicate Excel column names must be made unique.")
                frame = workbook.parse(sheet_name=sheet_name)
        else:
            raise DataValidationError("Supported dataset formats are CSV, XLSX and XLS.")
    except DataValidationError:
        raise
    except (ValueError, UnicodeError, OSError, KeyError, StopIteration, IndexError, ImportError) as exc:
        raise DataValidationError(
            "The file could not be read. Check the format, encoding and selected worksheet."
        ) from exc
    except Exception as exc:
        raise DataValidationError("The workbook or CSV file is invalid or corrupted.") from exc
    if frame.empty or len(frame.columns) < 2:
        raise DataValidationError("A dataset needs at least one record and date/sales columns.")
    notes = []
    if not all(isinstance(column, str) for column in frame.columns):
        frame.columns = frame.columns.map(str)
        notes.append("Column labels were converted to strings for manual mapping.")
    return LoadedDataset(
        frame=frame,
        filename=Path(filename).name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        sheets=sheets,
        notes=notes,
    )


def merge_walmart_files(
    train: pd.DataFrame,
    stores: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    required = {"Store", "Dept", "Date", "Weekly_Sales"}
    if not required.issubset(train.columns):
        raise DataValidationError(
            "Walmart historical data requires Store, Dept, Date and Weekly_Sales. "
            "The competition test.csv has no sales labels."
        )
    result = train.copy()
    operations = []
    if stores is not None:
        if "Store" not in stores or stores["Store"].isna().any() or stores["Store"].duplicated().any():
            raise DataValidationError("stores.csv requires one non-null, unique row per Store.")
        unmatched = int((~result["Store"].isin(stores["Store"])).sum())
        try:
            result = result.merge(stores, on="Store", how="left", validate="many_to_one", suffixes=("", "_stores"))
        except (ValueError, TypeError) as exc:
            raise DataValidationError("Store identifiers in history and stores.csv must have compatible types.") from exc
        operations.append({"operation": "Join stores.csv on Store", "affected_rows": len(result), "detail": f"Cardinality checked; {unmatched} rows had no store metadata."})
    if features is not None:
        if not {"Store", "Date"}.issubset(features.columns):
            raise DataValidationError("features.csv requires Store and Date columns.")
        ancillary = features.copy()
        result["Date"] = convert_dates(result["Date"])
        ancillary["Date"] = convert_dates(ancillary["Date"])
        if result["Date"].isna().any() or ancillary[["Store", "Date"]].isna().any().any():
            raise DataValidationError("Invalid Walmart join dates or store identifiers must be corrected.")
        if ancillary.duplicated(["Store", "Date"]).any():
            raise DataValidationError("features.csv must contain unique Store/Date pairs.")
        try:
            result = result.merge(ancillary, on=["Store", "Date"], how="left", validate="many_to_one", suffixes=("", "_features"), indicator="_feature_join")
        except (ValueError, TypeError) as exc:
            raise DataValidationError("Store identifiers in history and features.csv must have compatible types.") from exc
        unmatched = int(result["_feature_join"].eq("left_only").sum())
        result = result.drop(columns="_feature_join")
        if {"IsHoliday", "IsHoliday_features"}.issubset(result.columns):
            history_flags = convert_holiday_flags(result["IsHoliday"])
            feature_flags = convert_holiday_flags(result["IsHoliday_features"])
            comparable = history_flags.notna() & feature_flags.notna()
            if (history_flags.loc[comparable] != feature_flags.loc[comparable]).any():
                raise DataValidationError("Holiday flags disagree between train.csv and features.csv.")
        operations.append({"operation": "Join features.csv on Store/Date", "affected_rows": len(result), "detail": f"Cardinality checked; {unmatched} rows had no feature metadata. No missing features were filled."})
    if len(result) != len(train):
        raise DataValidationError("A Walmart join changed the historical record count.")
    return result, operations
