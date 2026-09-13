"""
Backend-Independent Dataset Loader for AI Analytics.

Provides a unified dataset loading utility that loads Pandas DataFrames into
any database conforming to the AnalyticsDatabase abstraction (DuckDB, PostgreSQL)
without coupling caller logic to concrete database implementations.
"""
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd

from src.analytics.database import AnalyticsDatabase


# =====================================================================
# EXCEPTIONS
# =====================================================================

class DatasetLoaderError(Exception):
    """Base exception for dataset loading errors."""
    pass


class InvalidDataFrameError(DatasetLoaderError, ValueError):
    """Raised when an invalid, corrupt, or non-DataFrame object is provided."""
    pass


# =====================================================================
# STRUCTURED LOADING METADATA
# =====================================================================

@dataclass(frozen=True)
class DatasetLoadResult:
    """
    Immutable metadata record produced upon successfully loading a dataset into
    an analytical database backend.
    """
    table_name: str
    row_count: int
    column_count: int
    columns: List[str]
    schema: Dict[str, str]


# =====================================================================
# DATASET LOADING FUNCTION
# =====================================================================

def load_dataset_into_database(
    engine: AnalyticsDatabase,
    df: pd.DataFrame,
    table_name: str = "sales",
) -> DatasetLoadResult:
    """
    Loads a Pandas DataFrame into an AnalyticsDatabase backend as a queryable
    relational table and inspects the resulting table schema.

    Parameters:
        engine: Any initialized database backend conforming to AnalyticsDatabase.
        df: The source Pandas DataFrame to load.
        table_name: Destination table identifier (default: 'sales').

    Returns:
        DatasetLoadResult: Structured metadata detailing table name, rows, columns,
                           and column data types.

    Raises:
        TypeError: If engine does not conform to AnalyticsDatabase.
        InvalidDataFrameError: If df is not a valid pandas DataFrame or has no columns.
        ValueError: If table_name is an invalid SQL identifier.
    """
    if not isinstance(engine, AnalyticsDatabase):
        raise TypeError(
            f"Expected an instance conforming to AnalyticsDatabase, but received: {type(engine).__name__}"
        )

    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError(
            f"Expected a pandas DataFrame, but received: {type(df).__name__}"
        )

    if len(df.columns) == 0:
        raise InvalidDataFrameError(
            "DataFrame contains no columns and cannot be loaded into the database."
        )

    if not table_name or not isinstance(table_name, str) or not table_name.isidentifier():
        raise InvalidDataFrameError(
            f"Invalid table name '{table_name}'. Must be a valid alphanumeric identifier."
        )

    # Work on a deep copy to ensure caller's DataFrame is strictly unmodified
    df_isolated = df.copy(deep=True)

    # Delegate loading to the database abstraction
    engine.load_dataframe(df_isolated, table_name=table_name)

    # Retrieve authoritative schema from the database
    schema = engine.get_schema(table_name=table_name)

    return DatasetLoadResult(
        table_name=table_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        schema=schema,
    )
