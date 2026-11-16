"""Typed, user-facing errors for recoverable application failures."""


class SalesVisionError(Exception):
    """Base application error that can be displayed safely."""


class DataValidationError(SalesVisionError):
    """Dataset or preprocessing configuration is invalid."""


class ForecastError(SalesVisionError):
    """A model could not produce a validated forecast."""


class InsufficientDataError(ForecastError):
    """More historical observations are required."""


class ReportError(SalesVisionError):
    """A report could not be generated."""


class ConfigurationError(SalesVisionError):
    """Server configuration or application directories need correction."""


class DatabaseError(SalesVisionError):
    """Forecast history is unavailable without compromising core computation."""
