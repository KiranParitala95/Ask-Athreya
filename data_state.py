"""Holds the CSV that the agent is currently analysing.

A language model cannot pass a whole DataFrame around as a tool argument, so the
data it loads at startup has to live somewhere every tool can reach. Keeping that
single shared reference in one small, dedicated module is cleaner than scattering
globals across the codebase: there is exactly one place that owns the active data.
"""

import pandas as pd

# The DataFrame for the current session. None until a file is loaded.
_active_dataframe: pd.DataFrame | None = None


def set_active_dataframe(dataframe: pd.DataFrame) -> None:
    """Store the DataFrame the agent should analyse for this session."""
    global _active_dataframe
    _active_dataframe = dataframe


def get_active_dataframe() -> pd.DataFrame:
    """Return the loaded DataFrame.

    Raises:
        RuntimeError: If called before a CSV has been loaded, so the mistake
            surfaces immediately instead of as a confusing error deeper in a tool.
    """
    if _active_dataframe is None:
        raise RuntimeError(
            "No CSV has been loaded yet. Load a file before using the agent's tools."
        )
    return _active_dataframe
