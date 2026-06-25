"""Tools the agent calls to analyse the loaded CSV.

Each tool is a small, focused function that performs one kind of lookup or
calculation and returns plain text. The language model reads each tool's
docstring to decide which one to call, so the descriptions are written to be
useful to the model, not only to a human reader.

The agent never touches the data directly; it only ever sees the strings these
functions return. All real work is ordinary pandas living behind these tools.
"""

import pandas as pd
from langchain_core.tools import tool

from data_state import get_active_dataframe

# When a query returns many rows, only the first slice is shown. This keeps tool
# output small enough for tight per-minute token budgets on hosted models.
MAX_PREVIEW_ROWS = 10


@tool
def get_data_overview() -> str:
    """Return the dataset's shape and the name and type of every column.

    Call this first when you need a high-level picture of the data before
    answering anything specific, or to confirm the exact column names to use.
    """
    dataframe = get_active_dataframe()
    row_count, column_count = dataframe.shape

    column_lines = [
        f"  - {name} ({dtype})" for name, dtype in dataframe.dtypes.items()
    ]

    return (
        f"Rows: {row_count}\n"
        f"Columns: {column_count}\n"
        "Column details:\n" + "\n".join(column_lines)
    )


@tool
def inspect_columns() -> str:
    """Return the null count for every column plus basic summary statistics.

    Use this to answer questions about missing values, averages, minimums,
    maximums, or how numeric columns are distributed.
    """
    dataframe = get_active_dataframe()

    null_counts = dataframe.isnull().sum()
    null_lines = [
        f"  - {column}: {int(null_counts[column])} nulls"
        for column in dataframe.columns
    ]

    # describe(include="all") computes top/freq/unique stats for text columns
    # too. With a column like a name that is almost entirely unique values, that
    # output balloons in size for little benefit, so only numeric columns get
    # full statistics here; text columns are covered by the null counts above.
    numeric_columns = dataframe.select_dtypes(include="number")
    if numeric_columns.empty:
        statistics = "(no numeric columns to summarise)"
    else:
        statistics = numeric_columns.describe().to_string()

    return (
        "Null counts per column:\n"
        + "\n".join(null_lines)
        + "\n\nSummary statistics:\n"
        + statistics
    )


@tool
def run_dataframe_query(pandas_expression: str) -> str:
    """Evaluate one pandas expression against the dataset and return the result.

    The DataFrame is available as `df` and pandas as `pd`. Pass a single
    expression that produces a value, for example:
      - df['salary'].mean()
      - df.groupby('department')['salary'].mean()
      - df[df['age'] > 50].shape[0]

    Provide one expression only: no assignments, imports, or multiple statements.
    Use this for any calculation the other tools do not answer directly.
    """
    dataframe = get_active_dataframe()
    result = _evaluate_pandas_expression(pandas_expression, dataframe)
    return _format_result(result)


@tool
def find_data_quality_issues() -> str:
    """Scan the dataset for common quality problems and report what it finds.

    Checks for duplicate rows, columns that are entirely empty, and columns
    where every value is identical. Use this when asked about data quality,
    duplicates, or whether the dataset looks clean.
    """
    dataframe = get_active_dataframe()

    duplicate_count = int(dataframe.duplicated().sum())

    empty_columns = [
        column for column in dataframe.columns if dataframe[column].isnull().all()
    ]
    constant_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True) == 1
    ]

    return (
        f"Duplicate rows: {duplicate_count}\n"
        f"Fully empty columns: {', '.join(empty_columns) if empty_columns else 'none'}\n"
        f"Constant (single-value) columns: {', '.join(constant_columns) if constant_columns else 'none'}"
    )


def _evaluate_pandas_expression(expression: str, dataframe: pd.DataFrame):
    """Run a pandas expression in a restricted namespace and return its value.

    Only `df` and `pd` are exposed and Python builtins are stripped out, so a
    stray expression cannot reach unrelated parts of the system. This is meant
    for local, single-user analysis, not for evaluating untrusted public input.

    A failed expression is returned as a text message rather than raised, so the
    agent can read the error and retry with a corrected expression instead of
    ending the whole session.
    """
    safe_namespace = {"df": dataframe, "pd": pd}
    try:
        return eval(expression, {"__builtins__": {}}, safe_namespace)
    except Exception as error:
        return f"Query failed: {error}. Check the column names and try again."


def _format_result(result) -> str:
    """Turn a query result into readable text for the model to summarise."""
    if isinstance(result, (pd.DataFrame, pd.Series)):
        if len(result) > MAX_PREVIEW_ROWS:
            preview = result.head(MAX_PREVIEW_ROWS).to_string()
            return (
                f"{preview}\n"
                f"... ({len(result)} rows total, showing the first {MAX_PREVIEW_ROWS})"
            )
        return result.to_string()
    return str(result)
