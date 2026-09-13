"""
DuckDB SQL Analytics Engine.

Provides an in-memory SQL execution layer that registers Pandas DataFrames
as relational tables, safely executes read-only SQL queries, and returns
results as Pandas DataFrames without modifying the underlying data.
"""
import re
from typing import Set, Dict, Any, Optional, Union
import duckdb
import pandas as pd
from src.analytics.database import AnalyticsDatabase


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class DuckDBEngineError(Exception):
    """Base exception for all DuckDB analytics engine errors."""
    pass


class DisallowedQueryError(DuckDBEngineError):
    """Raised when a non-read-only or forbidden SQL statement is submitted."""
    pass


class QueryExecutionError(DuckDBEngineError):
    """Raised when SQL syntax parsing or execution fails in DuckDB."""
    pass


# =====================================================================
# READ-ONLY QUERY VALIDATION
# =====================================================================

# Allowed starting verbs for read-only analytical queries
ALLOWED_STARTING_VERBS: Set[str] = {
    "SELECT",
    "WITH",
    "DESCRIBE",
    "SHOW",
    "EXPLAIN",
    "SUMMARIZE",
}

# Explicitly forbidden SQL command keywords (case-insensitive)
DISALLOWED_KEYWORDS: Set[str] = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "MERGE",
    "UPSERT",
    "PRAGMA",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "IMPORT",
    "CALL",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
    "VACUUM",
    "SET",
    "RESET",
}


def _strip_comments_and_literals(sql: str) -> str:
    """
    Strips SQL comments and string literals so that keyword checks
    do not match on comments or literal text (e.g. WHERE status = 'DROP').
    """
    # 1. Remove multiline block comments: /* ... */
    no_block_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    
    # 2. Remove single-line comments: -- ...
    no_line_comments = re.sub(r"--.*$", " ", no_block_comments, flags=re.MULTILINE)
    
    # 3. Replace single-quoted string literals with empty strings
    no_single_quotes = re.sub(r"'(?:''|[^'])*'", "''", no_line_comments)
    
    # 4. Replace dollar-quoted string literals ($$...$$) if present
    no_dollar_quotes = re.sub(r"\$\$.*?\$\$", "''", no_single_quotes, flags=re.DOTALL)
    
    return no_dollar_quotes.strip()


def validate_read_only_query(sql_query: str) -> None:
    """
    Validates that a SQL query is strictly read-only:
    - Must not be empty.
    - Must not chain multiple statements via semicolons.
    - Must start with an allowed read-only verb (SELECT, WITH, etc.).
    - Must not contain any forbidden DDL / DML modifying verbs.

    Raises:
        DisallowedQueryError: If the query fails any security rule.
    """
    if not sql_query or not sql_query.strip():
        raise DisallowedQueryError("SQL query cannot be empty.")

    cleaned_sql = _strip_comments_and_literals(sql_query)
    
    if not cleaned_sql:
        raise DisallowedQueryError("SQL query contains no executable statements.")

    # Check for multiple statements separated by semicolons (e.g. "SELECT 1; DROP TABLE sales;")
    # Strip any trailing semicolon first
    statements = [s.strip() for s in cleaned_sql.split(";") if s.strip()]
    if len(statements) > 1:
        raise DisallowedQueryError(
            "Multiple SQL statements separated by semicolons are not allowed. "
            "Only single read-only queries may be executed."
        )

    statement = statements[0]
    
    # Extract the first keyword token
    first_token_match = re.match(r"^([a-zA-Z_]+)", statement)
    if not first_token_match:
        raise DisallowedQueryError(
            f"Invalid SQL syntax: Query does not begin with a valid command word: '{statement[:30]}'"
        )

    first_verb = first_token_match.group(1).upper()
    if first_verb not in ALLOWED_STARTING_VERBS:
        raise DisallowedQueryError(
            f"Disallowed statement type: '{first_verb}'. "
            f"Only read-only analytical queries ({', '.join(sorted(ALLOWED_STARTING_VERBS))}) are permitted."
        )

    # Check for forbidden mutating keywords anywhere in the sanitized statement
    for forbidden in DISALLOWED_KEYWORDS:
        pattern = rf"\b{forbidden}\b"
        if re.search(pattern, statement, re.IGNORECASE):
            raise DisallowedQueryError(
                f"Disallowed SQL keyword detected: '{forbidden}'. "
                f"Data and schema modifications (INSERT, UPDATE, DELETE, DROP, etc.) are strictly prohibited."
            )


# =====================================================================
# DUCKDB ENGINE CLASS
# =====================================================================

class DuckDBEngine(AnalyticsDatabase):
    """
    Manages an in-memory DuckDB connection for analytical querying over Pandas DataFrames.
    """

    def __init__(self, database: str = ":memory:"):
        """
        Initializes an in-memory DuckDB database connection.
        """
        self._database: str = database
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._registered_tables: Set[str] = set()
        self.connect()

    def connect(self) -> None:
        """Establishes or reopens the in-memory DuckDB connection."""
        if self._con is None:
            self._con = duckdb.connect(database=self._database)

    @property
    def is_open(self) -> bool:
        """Returns True if the connection is active and open."""
        return self._con is not None

    @property
    def is_connected(self) -> bool:
        """Alias for is_open."""
        return self.is_open

    def register_dataframe(self, df: pd.DataFrame, table_name: str = "sales") -> None:
        """
        Registers a Pandas DataFrame as a queryable table in DuckDB.
        
        Zero-copy virtual table: does NOT modify the input DataFrame.
        
        Parameters:
            df: The pandas DataFrame to register.
            table_name: The table identifier to use in SQL queries (default: 'sales').
        """
        if not self.is_open:
            raise DuckDBEngineError("DuckDB connection is closed. Please create or reopen the engine.")
            
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but received: {type(df).__name__}")
            
        if not table_name or not table_name.isidentifier():
            raise ValueError(f"Invalid table name '{table_name}'. Must be a valid alphanumeric identifier.")

        # DuckDB registers the DataFrame view without mutating the original DataFrame
        self._con.register(table_name, df)
        self._registered_tables.add(table_name)

    # Parity alias conforming to AnalyticsDatabase abstraction
    load_dataframe = register_dataframe

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Validates that the SQL query is read-only, executes it against
        registered tables, and returns the result as a new Pandas DataFrame.

        Parameters:
            sql_query: The SQL query to execute.

        Returns:
            pd.DataFrame: Query result as a new DataFrame.
            
        Raises:
            DisallowedQueryError: If the query attempts any data modification.
            QueryExecutionError: If DuckDB fails to parse or execute the query.
        """
        if not self.is_open:
            raise DuckDBEngineError("DuckDB connection is closed. Please create or reopen the engine.")

        # 1. Enforce strict read-only validation
        validate_read_only_query(sql_query)

        # 2. Execute query safely
        try:
            result_df = self._con.execute(sql_query).df()
            return result_df
        except duckdb.Error as e:
            raise QueryExecutionError(f"SQL execution error: {str(e)}") from e
        except Exception as e:
            raise QueryExecutionError(f"Unexpected query error: {str(e)}") from e

    def get_registered_tables(self) -> Set[str]:
        """Returns the set of registered table names."""
        return set(self._registered_tables)

    def get_schema(self, table_name: str = "sales") -> Dict[str, str]:
        """
        Returns a dictionary mapping column names to their DuckDB data types
        for the given registered table.
        """
        if not self.is_open:
            raise DuckDBEngineError("DuckDB connection is closed.")
            
        if table_name not in self._registered_tables:
            raise ValueError(f"Table '{table_name}' has not been registered in this DuckDB session.")

        desc_df = self._con.execute(f"DESCRIBE {table_name}").df()
        return dict(zip(desc_df["column_name"], desc_df["column_type"]))

    def close(self) -> None:
        """Safely closes the DuckDB connection."""
        if self._con is not None:
            self._con.close()
            self._con = None
            self._registered_tables.clear()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =====================================================================
# CONVENIENCE STANDALONE FUNCTIONS
# =====================================================================

def create_engine() -> DuckDBEngine:
    """Factory function to create a new DuckDBEngine instance."""
    return DuckDBEngine()


def execute_sql(df: pd.DataFrame, sql_query: str, table_name: str = "sales") -> pd.DataFrame:
    """
    Convenience function: Creates a temporary in-memory engine, registers
    the DataFrame, runs the query, closes the engine, and returns the result DataFrame.
    """
    with DuckDBEngine() as engine:
        engine.register_dataframe(df, table_name=table_name)
        return engine.execute_query(sql_query)
