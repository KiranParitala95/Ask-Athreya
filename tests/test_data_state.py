"""Tests for the shared active-DataFrame state.

The key behaviours are that reading before any data is set fails loudly, and
that once a DataFrame is set it is returned unchanged.
"""

import pandas as pd
import pytest

import data_state


@pytest.fixture(autouse=True)
def reset_state():
    """Clear the module-level DataFrame before and after each test.

    The active DataFrame is shared module state, so each test resets it to avoid
    leaking data between tests.
    """
    data_state._active_dataframe = None
    yield
    data_state._active_dataframe = None


def test_get_before_set_raises_runtime_error():
    with pytest.raises(RuntimeError):
        data_state.get_active_dataframe()


def test_set_then_get_returns_same_dataframe():
    dataframe = pd.DataFrame({"a": [1, 2, 3]})

    data_state.set_active_dataframe(dataframe)

    assert data_state.get_active_dataframe() is dataframe
