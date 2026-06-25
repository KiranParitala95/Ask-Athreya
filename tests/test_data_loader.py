"""Tests for the CSV and Excel loader.

These cover the happy path for both file types, plus every way loading is meant
to fail: a missing file, an unsupported extension, an empty file, and the
quote-wrapped paths that Windows "Copy as path" produces.
"""

import pandas as pd
import pytest

from data_loader import load_data_file


def test_loads_valid_csv(tmp_path):
    path = tmp_path / "good.csv"
    path.write_text("name,age\nAlice,30\nBob,25\n")

    dataframe = load_data_file(str(path))

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["name", "age"]


def test_loads_valid_excel(tmp_path):
    path = tmp_path / "good.xlsx"
    pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]}).to_excel(path, index=False)

    dataframe = load_data_file(str(path))

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["name", "age"]


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data_file(str(tmp_path / "does_not_exist.csv"))


def test_unsupported_extension_raises_value_error(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("name,age\nAlice,30\n")

    with pytest.raises(ValueError):
        load_data_file(str(path))


def test_empty_csv_raises_value_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("name,age\n")

    with pytest.raises(ValueError):
        load_data_file(str(path))


def test_empty_excel_raises_value_error(tmp_path):
    path = tmp_path / "empty.xlsx"
    pd.DataFrame({"name": [], "age": []}).to_excel(path, index=False)

    with pytest.raises(ValueError):
        load_data_file(str(path))


def test_strips_surrounding_quotes_from_csv_path(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text("name,age\nAlice,30\n")

    assert load_data_file(f'"{path}"').shape == (1, 2)


def test_strips_surrounding_quotes_from_excel_path(tmp_path):
    path = tmp_path / "quoted.xlsx"
    pd.DataFrame({"name": ["Alice"], "age": [30]}).to_excel(path, index=False)

    assert load_data_file(f'"{path}"').shape == (1, 2)
