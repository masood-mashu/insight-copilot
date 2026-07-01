"""DataFrame profiling utilities."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_profiler_agent(df: pd.DataFrame) -> dict:
    """Profile a pandas DataFrame and return a structured summary.

    The returned dictionary contains dataset dimensions, per-column metadata,
    a light statistical summary for numeric columns, and the number of
    duplicate rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    logger.debug("Profiling DataFrame with shape %s", df.shape)

    columns: list[dict[str, Any]] = []
    n_rows, n_cols = df.shape

    numeric_summary = df.describe(include="number") if not df.empty else pd.DataFrame()

    for column_name in df.columns:
        series = df[column_name]
        non_null_series = series.dropna()
        unique_values = []
        if not non_null_series.empty:
            unique_values = [_to_python_value(value) for value in list(pd.unique(non_null_series))[:3]]

        column_profile: dict[str, Any] = {
            "name": str(column_name),
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_pct": round((float(series.isnull().sum()) / n_rows * 100.0) if n_rows else 0.0, 2),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": unique_values,
        }

        if pd.api.types.is_numeric_dtype(series) and column_name in numeric_summary.columns:
            stats_row = numeric_summary[column_name]
            column_profile["stats"] = {
                "mean": _to_python_value(stats_row.get("mean")),
                "std": _to_python_value(stats_row.get("std")),
                "min": _to_python_value(stats_row.get("min")),
                "max": _to_python_value(stats_row.get("max")),
            }

        columns.append(column_profile)

    result = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": columns,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    logger.debug("Profile complete for %s rows and %s columns", n_rows, n_cols)
    return result


def _to_python_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value
