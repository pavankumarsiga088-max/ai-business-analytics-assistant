"""
PostgreSQL Analytics Engine.

Provides an enterprise SQL execution layer that loads Pandas DataFrames into
a PostgreSQL database, safely executes read-only analytical SQL queries,
and returns results as Pandas DataFrames without modifying the underlying data.
"""
import os
import re
from typing import Set, Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from src.analytics.database import AnalyticsDatabase

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

try:
    import psycopg
    from psycopg import sql
    PSYCOPG_AVAILABLE = True
except ImportError:
    psycopg = None  # type: ignore
    sql = None      # type: ignore
    PSYCOPG_AVAILABLE = False


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class PostgresEngineError(Exception):
    """Base exception for all PostgreSQL analytics engine errors."""
    pass


class PostgresConnectionError(PostgresEngineError):
    """Raised when connection to PostgreSQL fails or credentials are invalid."""
    pass


class DisallowedQueryError(PostgresEngineError):
    """Raised when a non-read-only or forbidden SQL statement is submitted."""
    pass


class QueryExecutionError(PostgresEngineError):
    """Raised when SQL syntax parsing or execution fails in PostgreSQL."""
    pass


# Aliases for explicit naming flexibility
PostgresDisallowedQueryError = DisallowedQueryError
PostgresQueryExecutionError = QueryExecutionError


# =====================================================================
# READ-ONLY QUERY SECURITY VALIDATION
# =====================================================================

# Allowed starting verbs for read-only analytical queries in PostgreSQL
ALLOWED_STARTING_VERBS: Set[str] = {
    "SELECT",
    "WITH",
    "EXPLAIN",
    "SHOW",
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
    "GRANT",
    "REVOKE",
    "VACUUM",
    "COPY",
    "CALL",
    "DO",
    "EXEC",
    "EXECUTE",
    "LOCK",
    "COMMENT",
    "REINDEX",
    "CLUSTER",
    "REFRESH",
    "SECURITY",
    "SET",
    "RESET",
}


def _strip_comments_and_literals(sql_query: str) -> str:
    """
    Strips SQL comments, single-quoted string literals, and dollar-quoted
    literals ($$...$$) so that keyword checks do not match benign text
    (e.g., WHERE status = 'DROP' or inside comments).
    """
    # 1. Remove multiline block comments: /* ... */
    no_block_comments = re.sub(r"/\*.*?\*/", " ", sql_query, flags=re.DOTALL)

    # 2. Remove single-line comments: -- ...
    no_line_comments = re.sub(r"--.*$", " ", no_block_comments, flags=re.MULTILINE)

    # 3. Replace single-quoted string literals with empty strings
    no_single_quotes = re.sub(r"'(?:''|[^'])*'", "''", no_line_comments)

    # 4. Replace dollar-quoted string literals ($$...$$) if present
    no_dollar_quotes = re.sub(r"\$\$.*?\$\$", "''", no_single_quotes, flags=re.DOTALL)

    return no_dollar_quotes.strip()


def validate_read_only_query(sql_query: str) -> None:
    """
    Validates that a SQL query is strictly read-only for PostgreSQL:
    - Must not be empty.
    - Must not chain multiple statements via semicolons.
    - Must start with an allowed read-only verb (SELECT, WITH, EXPLAIN, SHOW).
    - Must not contain forbidden DDL/DML mutating verbs.

    Raises:
        DisallowedQueryError: If the query fails any security rule.
    """
    if not sql_query or not sql_query.strip():
        raise DisallowedQueryError("SQL query cannot be empty.")

    cleaned_sql = _strip_comments_and_literals(sql_query)

    if not cleaned_sql:
        raise DisallowedQueryError("SQL query contains no executable statements.")

    # Check for multiple statements separated by semicolons (e.g. "SELECT 1; DROP TABLE sales;")
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
# TYPE INFERENCE FOR PANDAS TO POSTGRESQL
# =====================================================================

def _infer_postgres_type(series: pd.Series) -> str:
    """
    Infers an appropriate PostgreSQL data type from a Pandas Series.
    Handles numeric, boolean, datetime, ISO date strings, and text.
    """
    if pd.api.types.is_integer_dtype(series.dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(series.dtype):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(series.dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "TIMESTAMP"

    # For object/string columns, inspect non-null values for clean date formats
    non_null = series.dropna()
    if not non_null.empty and series.name:
        col_name_lower = str(series.name).lower()
        if "date" in col_name_lower or "time" in col_name_lower:
            sample = non_null.head(50)
            try:
                # Check if all sample rows match ISO date YYYY-MM-DD
                parsed = pd.to_datetime(sample, format="%Y-%m-%d", errors="coerce")
                if parsed.notna().all():
                    return "DATE"
            except Exception:
                pass

    return "TEXT"


# =====================================================================
# POSTGRES ENGINE CLASS
# =====================================================================

class PostgresEngine(AnalyticsDatabase):
    """
    Manages a PostgreSQL connection for analytical querying over uploaded datasets.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[Union[int, str]] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        autocommit: bool = True,
        connect_timeout: int = 5,
        query_timeout: Optional[int] = None,
    ):
        """
        Initializes a PostgreSQL database engine.

        Parameters:
            host: PostgreSQL host (default: from POSTGRES_HOST or 'localhost').
            port: PostgreSQL port (default: from POSTGRES_PORT or 5432).
            database: Target database name (default: from POSTGRES_DB or 'postgres').
            user: PostgreSQL username (default: from POSTGRES_USER or 'postgres').
            password: Password override. If None, reads from POSTGRES_PASSWORD.
            connect_timeout: Connection timeout in seconds (default: 10).
            query_timeout: Query execution timeout in seconds (default: from QUERY_TIMEOUT_SECONDS or 30).
            autocommit: Whether to enable autocommit on the connection (default: True).
        """
        self._host = str(host or os.getenv("POSTGRES_HOST", "localhost")).strip()
        port_raw = port or os.getenv("POSTGRES_PORT", "5432")
        try:
            self._port = int(str(port_raw).strip())
        except (ValueError, TypeError):
            self._port = 5432

        self._database = str(database or os.getenv("POSTGRES_DB", "postgres")).strip()
        self._user = str(user or os.getenv("POSTGRES_USER", "postgres")).strip()

        if password is not None:
            self._password = str(password)
        else:
            self._password = os.getenv("POSTGRES_PASSWORD", "")

        self._autocommit = autocommit
        self._connect_timeout = connect_timeout

        query_timeout_raw = query_timeout if query_timeout is not None else os.getenv("QUERY_TIMEOUT_SECONDS", "30")
        try:
            self._query_timeout: int = int(str(query_timeout_raw).strip())
        except (ValueError, TypeError):
            self._query_timeout = 30

        self._con: Optional[Any] = None
        self._registered_tables: Set[str] = set()

    @property
    def query_timeout(self) -> int:
        """Query execution timeout in seconds."""
        return self._query_timeout

    @property
    def host(self) -> str:
        """PostgreSQL host."""
        return self._host

    @property
    def port(self) -> int:
        """PostgreSQL port."""
        return self._port

    @property
    def database(self) -> str:
        """Target database name."""
        return self._database

    @property
    def user(self) -> str:
        """PostgreSQL username."""
        return self._user

    @property
    def is_open(self) -> bool:
        """Returns True if the connection is active and open."""
        return self._con is not None and not self._con.closed

    @property
    def is_connected(self) -> bool:
        """Alias for is_open."""
        return self.is_open

    def _sanitize_error_message(self, msg: str) -> str:
        """Redacts password from error strings to prevent credential leaks."""
        if self._password and self._password in msg:
            return msg.replace(self._password, "******")
        return msg

    def connect(self) -> None:
        """
        Establishes a connection to PostgreSQL safely.

        Raises:
            PostgresConnectionError: If connection or authentication fails.
        """
        if not PSYCOPG_AVAILABLE:
            raise PostgresConnectionError(
                "The 'psycopg' library is not installed. Please install 'psycopg[binary]'."
            )

        if self.is_open:
            return

        try:
            self._con = psycopg.connect(
                host=self._host,
                port=self._port,
                dbname=self._database,
                user=self._user,
                password=self._password,
                connect_timeout=self._connect_timeout,
                autocommit=self._autocommit,
            )
        except Exception as e:
            sanitized = self._sanitize_error_message(str(e))
            self._con = None
            raise PostgresConnectionError(
                f"Failed to connect to PostgreSQL at {self._host}:{self._port} "
                f"(database: '{self._database}', user: '{self._user}'): {sanitized}"
            ) from None

    def _ensure_connected(self) -> None:
        """Ensures an active connection is open before query or load operations."""
        if not self.is_open:
            self.connect()

    def load_dataframe(self, df: pd.DataFrame, table_name: str = "sales") -> None:
        """
        Loads a Pandas DataFrame into a PostgreSQL table.
        The uploaded DataFrame becomes the source table for analytics.

        Parameters:
            df: The Pandas DataFrame to load.
            table_name: Destination table identifier (default: 'sales').

        Notes:
            - Does NOT modify the caller's DataFrame.
            - Replaces any existing table with the same name.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but received: {type(df).__name__}")

        if not table_name or not table_name.isidentifier():
            raise ValueError(f"Invalid table name '{table_name}'. Must be a valid alphanumeric identifier.")

        self._ensure_connected()

        # Work on an isolated copy to guarantee caller's DataFrame is unmodified
        df_copy = df.copy(deep=True)

        # Infer column types
        col_types = {col: _infer_postgres_type(df_copy[col]) for col in df_copy.columns}

        try:
            with self._con.cursor() as cur:
                # 1. Drop existing table
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name))
                )

                # 2. Create new table with inferred schema
                col_defs = [
                    sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(dtype))
                    for col, dtype in col_types.items()
                ]
                create_query = sql.SQL("CREATE TABLE {} ({});").format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(col_defs),
                )
                cur.execute(create_query)

                # 3. Batch insert records if DataFrame is not empty
                if not df_copy.empty:
                    cols_identifiers = [sql.Identifier(col) for col in df_copy.columns]
                    placeholders = [sql.SQL("%s") for _ in df_copy.columns]
                    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({});").format(
                        sql.Identifier(table_name),
                        sql.SQL(", ").join(cols_identifiers),
                        sql.SQL(", ").join(placeholders),
                    )

                    # Replace NaN / NaT / pd.NA with None for clean PostgreSQL NULLs
                    clean_records = []
                    for row in df_copy.itertuples(index=False, name=None):
                        clean_row = tuple(None if pd.isna(val) else val for val in row)
                        clean_records.append(clean_row)

                    cur.executemany(insert_query, clean_records)

            self._registered_tables.add(table_name)
        except Exception as e:
            sanitized = self._sanitize_error_message(str(e))
            raise QueryExecutionError(f"Failed to load DataFrame into table '{table_name}': {sanitized}") from None

    # Parity alias with DuckDBEngine
    register_dataframe = load_dataframe

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Validates that the SQL query is strictly read-only, executes it in
        PostgreSQL, and returns the result as a new Pandas DataFrame.

        Parameters:
            sql_query: The SQL query to execute.

        Returns:
            pd.DataFrame: Analytical query result as a new DataFrame.

        Raises:
            DisallowedQueryError: If the query contains mutating or forbidden statements.
            QueryExecutionError: If PostgreSQL syntax parsing or execution fails.
        """
        self._ensure_connected()

        # 1. Strict read-only validation
        validate_read_only_query(sql_query)

        # 2. Execute query safely
        try:
            with self._con.cursor() as cur:
                if self._query_timeout and self._query_timeout > 0:
                    cur.execute(
                        sql.SQL("SET statement_timeout = {};").format(
                            sql.Literal(self._query_timeout * 1000)
                        )
                    )
                cur.execute(sql_query)
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    return pd.DataFrame(rows, columns=columns)
                return pd.DataFrame()
        except DisallowedQueryError:
            raise
        except Exception as e:
            sanitized = self._sanitize_error_message(str(e))
            raise QueryExecutionError(f"PostgreSQL query execution error: {sanitized}") from None

    def get_registered_tables(self) -> Set[str]:
        """Returns the set of registered table names in the current session."""
        return set(self._registered_tables)

    def get_schema(self, table_name: str = "sales") -> Dict[str, str]:
        """
        Returns a dictionary mapping column names to PostgreSQL data types
        for the specified table, compatible with NL-to-SQL schema formatting.

        Example:
            {"order_id": "TEXT", "quantity": "BIGINT", "revenue": "DOUBLE PRECISION"}
        """
        self._ensure_connected()

        query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = current_schema()
        ORDER BY ordinal_position;
        """
        try:
            with self._con.cursor() as cur:
                cur.execute(query, (table_name,))
                rows = cur.fetchall()

            if not rows:
                raise ValueError(
                    f"Table '{table_name}' has not been registered or does not exist in PostgreSQL session."
                )

            return {col_name: col_type.upper() for col_name, col_type in rows}
        except ValueError:
            raise
        except Exception as e:
            sanitized = self._sanitize_error_message(str(e))
            raise QueryExecutionError(f"Failed to fetch schema for '{table_name}': {sanitized}") from None

    def close(self) -> None:
        """Safely closes the PostgreSQL connection."""
        if self._con is not None:
            try:
                if not self._con.closed:
                    self._con.close()
            except Exception:
                pass
            finally:
                self._con = None
                self._registered_tables.clear()

    def __enter__(self):
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return (
            f"<PostgresEngine host={self._host} port={self._port} "
            f"database={self._database} user={self._user} status={status}>"
        )

    def __str__(self) -> str:
        return self.__repr__()


# =====================================================================
# CONVENIENCE STANDALONE FUNCTIONS
# =====================================================================

def create_postgres_engine(**kwargs) -> PostgresEngine:
    """Factory function to create a new PostgresEngine instance."""
    return PostgresEngine(**kwargs)


def execute_postgres_sql(
    df: pd.DataFrame,
    sql_query: str,
    table_name: str = "sales",
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function: Creates a temporary engine, loads the DataFrame,
    runs the read-only query, closes the engine, and returns the result.
    """
    with PostgresEngine(**kwargs) as engine:
        engine.load_dataframe(df, table_name=table_name)
        return engine.execute_query(sql_query)
