"""Rotating local logs for core application diagnostics."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(directory: Path) -> logging.Logger:
    logger = logging.getLogger("salesvision")
    if not logger.handlers:
        file_available = True
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(directory / "salesvision.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        except OSError:
            handler = logging.StreamHandler()
            file_available = False
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not file_available:
            logger.warning("File logging is unavailable; diagnostics are being sent to server stderr.")
    return logger
