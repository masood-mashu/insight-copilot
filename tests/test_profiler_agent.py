"""Unit tests for agents/profiler_agent.py."""

from __future__ import annotations

import pandas as pd
import pytest

from agents.profiler_agent import run_profiler_agent


def test_profiler_basic_shape_and_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = run_profiler_agent(df)

    assert result["n_rows"] == 3
    assert result["n_cols"] == 2
    assert {col["name"] for col in result["columns"]} == {"a", "b"}
    assert result["duplicate_rows"] == 0


def test_profiler_empty_dataframe() -> None:
    df = pd.DataFrame({"a": [], "b": []})
    result = run_profiler_agent(df)

    assert result["n_rows"] == 0
    assert result["n_cols"] == 2
    for column in result["columns"]:
        assert column["null_pct"] == 0.0
        assert column["unique_count"] == 0
        assert column["sample_values"] == []


def test_profiler_all_null_column_has_no_stats() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    result = run_profiler_agent(df)

    col_b = next(col for col in result["columns"] if col["name"] == "b")
    assert col_b["null_count"] == 3
    assert col_b["null_pct"] == 100.0
    assert col_b["sample_values"] == []


def test_profiler_single_row() -> None:
    df = pd.DataFrame({"a": [42], "b": ["only"]})
    result = run_profiler_agent(df)

    assert result["n_rows"] == 1
    col_a = next(col for col in result["columns"] if col["name"] == "a")
    assert col_a["unique_count"] == 1
    assert col_a["stats"]["mean"] == 42.0


def test_profiler_detects_duplicate_rows() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = run_profiler_agent(df)

    assert result["duplicate_rows"] == 1


def test_profiler_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        run_profiler_agent({"not": "a dataframe"})  # type: ignore[arg-type]
