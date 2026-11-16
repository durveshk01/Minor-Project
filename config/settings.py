"""Environment-backed configuration without exposing credentials to the UI."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = PROJECT_ROOT
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.1-flash-lite"
    database_path: Path = PROJECT_ROOT / "outputs" / "salesvision.sqlite3"
    max_upload_mb: int = 100

    def ensure_directories(self) -> None:
        try:
            for relative in (
                "data/sample", "data/source", "data/uploads", "data/processed",
                "models", "outputs", "outputs/logs", "outputs/history",
            ):
                (self.root / relative).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError("The project directory must be writable. Check application directory permissions.") from exc


def get_settings(secrets: Mapping | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    secrets = secrets or {}
    try:
        maximum = int(os.getenv("SALESVISION_MAX_UPLOAD_MB", "100"))
        if not 1 <= maximum <= 10000:
            raise ValueError("Upload limit outside supported bounds")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("SALESVISION_MAX_UPLOAD_MB must be an integer between 1 and 10000.") from exc
    configured_path = Path(os.getenv("SALESVISION_DATABASE_PATH", "outputs/salesvision.sqlite3"))
    database_path = configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or str(secrets.get("GEMINI_API_KEY", "")),
        gemini_model=os.getenv("GEMINI_MODEL") or str(secrets.get("GEMINI_MODEL", "gemini-3.1-flash-lite")),
        database_path=database_path,
        max_upload_mb=maximum,
    )
