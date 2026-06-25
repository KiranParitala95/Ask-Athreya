"""Tests for the data-analysis tools.

A small DataFrame with known contents is loaded into the shared state, so each
tool's output can be checked against values we can compute by hand. This also
covers the boundary cases: a failing query and the row-preview truncation.
"""

import pandas as pd
import pytest

import data_state
from tools import (
    MAX_PREVIEW_ROWS,
    _format_result,
    find_data_quality_issues,
    get_data_overview,
    inspect_columns,
    run_dataframe_query,
)


@pytest.fixture
def sample_data():
    """Load a small, known dataset into the shared state for each test.

    It deliberately contains a missing value and a duplicate row so the
    null-count and quality tools have something to find.
    """
    dataframe = pd.DataFrame(
        {
            "name": ["Asha", "Ravi", "Meera", "Asha"],
            "score": [90, 70, None, 90],
            "city": ["Pune", "Delhi", "Pune", "Pune"],
        }
    )
    data_state.set_active_dataframe(dataframe)
    yield dataframe
    data_state._active_dataframe = None


def test_overview_reports_shape_and_columns(sample_data):
    result = get_data_overview.invoke({})

    assert "Rows: 4" in result
    assert "Columns: 3" in result
    assert "name" in result
    assert "score" in result


def test_inspect_columns_counts_nulls(sample_data):
    result = inspect_columns.invoke({})

    # The score column has exactly one missing value.
    assert "score: 1 nulls" in result
    assert "name: 0 nulls" in result


def test_query_computes_correct_value(sample_data):
    result = run_dataframe_query.invoke({"pandas_expression": "df['score'].max()"})

    assert "90" in result


def test_bad_query_returns_message_not_exception(sample_data):
    result = run_dataframe_query.invoke(
        {"pandas_expression": "df['missing_column'].mean()"}
    )

    # A bad expression should come back as readable text, never raise.
    assert "Query failed" in result


def test_quality_check_finds_duplicate(sample_data):
    result = find_data_quality_issues.invoke({})

    # The "Asha / 90 / Pune" row appears twice.
    assert "Duplicate rows: 1" in result


def test_format_result_truncates_long_output():
    long_series = pd.Series(range(MAX_PREVIEW_ROWS + 10))

    formatted = _format_result(long_series)

    assert "rows total" in formatted


def test_quality_check_reports_none_when_no_empty_or_constant_columns(sample_data):
    result = find_data_quality_issues.invoke({})
    assert "Fully empty columns: none" in result
