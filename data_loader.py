"""Load a CSV or Excel file into a DataFrame, validating the input first.

Every check lives here so that a missing path, a wrong file type, or an empty
file fails immediately with a message that says exactly what went wrong, rather
than surfacing later as an opaque pandas error in the middle of a tool call.
"""

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_data_file(file_path: str) -> pd.DataFrame:
    """Read a CSV or Excel file from disk and return it as a DataFrame.

    Strips surrounding quotes and spaces from the path first — Windows
    "Copy as path" wraps paths in double quotes, which would otherwise
    cause a FileNotFoundError on a perfectly valid path.

    Raises:
        FileNotFoundError: If nothing exists at the given path.
        ValueError: If the file type is unsupported, or the file loads with no rows.
    """
    cleaned = file_path.strip(" \"'")
    path = Path(cleaned)

    if not path.exists():
        raise FileNotFoundError(f"No file found at: {cleaned}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported types: {supported}."
        )

    dataframe = pd.read_csv(path) if extension == ".csv" else pd.read_excel(path)

    if dataframe.empty:
        raise ValueError(f"The file '{file_path}' loaded successfully but has no rows.")

    return dataframe
