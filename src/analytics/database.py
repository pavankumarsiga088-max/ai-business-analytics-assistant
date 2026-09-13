"""
Database Abstraction Layer for AI Analytics Engine.

Defines a clean, unified contract (AnalyticsDatabase) for analytical database
backends (e.g. DuckDB, PostgreSQL) and a factory function (create_database_engine)
to construct the appropriate engine without binding the caller to database-specific
implementations.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set, Tuple
import pandas as pd


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class DatabaseEngineError(Exception):
    """Base exception for analytical database abstraction errors."""
    pass


class UnsupportedDatabaseError(ValueError, DatabaseEngineError):
    """Raised when an unsupported database engine type is requested."""
    pass


# =====================================================================
# ANALYTICS DATABASE ABSTRACT BASE CLASS
# =====================================================================

class AnalyticsDatabase(ABC):
    """
    Abstract Base Class defining the contract for analytical database engines.

    Every conforming engine must support:
      - Safe connection lifecycle (connect, close, is_open)
      - DataFrame loading into a queryable relational table (load_dataframe)
      - Read-only analytical SQL execution returning a pandas DataFrame (execute_query)
      - Schema inspection compatible with NL-to-SQL prompting (get_schema)
      - Safe context manager protocol (__enter__, __exit__)
    """

    @abstractmethod
    def connect(self) -> None:
        """Establishes or verifies an active database connection."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Safely closes active connections and cleans up allocated resources."""
        pass

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Returns True if the database connection is currently active and open."""
        pass

    @property
    def is_connected(self) -> bool:
        """Alias for is_open."""
        return self.is_open

    @abstractmethod
    def load_dataframe(self, df: pd.DataFrame, table_name: str = "sales") -> None:
        """
        Loads a Pandas DataFrame into the database as a queryable relational table.

        Guarantees:
          - Does NOT modify the caller's DataFrame.
          - Replaces existing table with the same name.
        """
        pass

    def register_dataframe(self, df: pd.DataFrame, table_name: str = "sales") -> None:
        """
        Parity alias for load_dataframe. Registers a Pandas DataFrame as a table.
        """
        self.load_dataframe(df, table_name=table_name)

    @abstractmethod
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a validated, read-only analytical SQL query and returns the
        result as a new Pandas DataFrame.
        """
        pass

    @abstractmethod
    def get_schema(self, table_name: str = "sales") -> Dict[str, str]:
        """
        Returns a dictionary mapping column names to data type strings for
        the specified table, compatible with NL-to-SQL schema formatting.

        Example:
            {"order_id": "VARCHAR", "revenue": "DOUBLE"}
        """
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =====================================================================
# DATABASE FACTORY FUNCTION
# =====================================================================

def create_database_engine(
    db_type: str = "duckdb",
    **kwargs: Any,
) -> AnalyticsDatabase:
    """
    Factory function to construct the appropriate analytical database engine.

    Supported database types:
      - 'duckdb': In-memory DuckDB engine for rapid zero-dependency local analytics
      - 'postgres' / 'postgresql': PostgreSQL engine for enterprise relational analytics

    Parameters:
        db_type: Target database identifier (case-insensitive).
        **kwargs: Engine-specific configuration options passed to the constructor.

    Returns:
        AnalyticsDatabase: An initialized instance conforming to AnalyticsDatabase.

    Raises:
        UnsupportedDatabaseError: If db_type is not recognized.
    """
    if not db_type or not isinstance(db_type, str):
        raise UnsupportedDatabaseError(
            f"Invalid database type '{db_type}'. Must be a non-empty string."
        )

    db_type_normalized = db_type.strip().lower()

    if db_type_normalized == "duckdb":
        # Deferred import to avoid circular dependency
        from src.analytics.duckdb_engine import DuckDBEngine
        return DuckDBEngine(**kwargs)

    elif db_type_normalized in ("postgres", "postgresql"):
        # Deferred import to avoid circular dependency
        from src.analytics.postgres_engine import PostgresEngine
        return PostgresEngine(**kwargs)

    else:
        raise UnsupportedDatabaseError(
            f"Unsupported database type: '{db_type}'. "
            f"Supported options are 'duckdb', 'postgres' (or 'postgresql')."
        )


def get_configured_database_backend() -> str:
    """
    Reads the configured analytical database backend from the DATABASE_BACKEND
    environment variable. Defaults to 'duckdb' if unset or unrecognized.
    """
    env_val = os.getenv("DATABASE_BACKEND", "duckdb").strip().lower()
    if env_val in ("postgres", "postgresql"):
        return "postgres"
    return "duckdb"


def is_backend_available(backend: str) -> Tuple[bool, str]:
    """
    Verifies if a given analytical database backend is reachable and available.

    Returns:
        Tuple[bool, str]: (True, "") if operational, or (False, reason) if unavailable.
    """
    normalized = (backend or "").strip().lower()
    if normalized == "duckdb":
        return True, ""
    elif normalized in ("postgres", "postgresql"):
        try:
            from src.analytics.postgres_engine import PostgresEngine
            engine = PostgresEngine(connect_timeout=2)
            engine.connect()
            engine.close()
            return True, ""
        except Exception as e:
            return False, str(e)
    else:
        return False, f"Unsupported database type: '{backend}'"
