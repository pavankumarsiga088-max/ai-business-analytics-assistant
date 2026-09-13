"""
Tests for the standalone PostgreSQL Analytics Engine (Phase 13A).

Includes:
1. Offline Unit & Security Tests (ALWAYS run unconditionally):
   - Configuration defaults and environment variable resolution
   - Credential privacy (password never exposed in repr or error messages)
   - Exception class hierarchy
   - Read-only SQL security validation (SELECT, WITH, EXPLAIN, SHOW allowed)
   - Mutating SQL rejection (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, etc.)
   - Multi-statement SQL rejection
   - Comment and string literal keyword safety (e.g. WHERE status = 'DROP')
   - Graceful offline connection error handling

2. Real PostgreSQL Integration Tests (Run when PostgreSQL is available, skipped otherwise):
   - Successful PostgreSQL connection
   - Loading sample_sales.csv into table 'sales'
   - Table row count verification == 100
   - Schema retrieval verification
   - SELECT query execution
   - Aggregation query verification (Regional revenue)
   - Caller DataFrame immutability verification
   - Invalid SQL syntax error handling
   - Connection lifecycle and cleanup
"""
import os
import pytest
import pandas as pd
from src.data_quality.loader import load_dataset
from src.analytics.postgres_engine import (
    PostgresEngine,
    PostgresEngineError,
    PostgresConnectionError,
    DisallowedQueryError,
    QueryExecutionError,
    validate_read_only_query,
    create_postgres_engine,
    execute_postgres_sql,
)


from typing import Tuple

# =====================================================================
# FIXTURES & AVAILABILITY CHECK
# =====================================================================

def check_postgres_availability() -> Tuple[bool, str]:
    """
    Attempts a brief connection to PostgreSQL using configured environment
    variables. Returns (True, "") if available, or (False, "reason") if unavailable.
    """
    engine = PostgresEngine(connect_timeout=2)
    try:
        engine.connect()
        engine.close()
        return True, ""
    except Exception as e:
        return False, str(e)


PG_AVAILABLE, PG_UNAVAILABLE_REASON = check_postgres_availability()

requires_postgres = pytest.mark.skipif(
    not PG_AVAILABLE,
    reason=f"PostgreSQL server or credentials not available ({PG_UNAVAILABLE_REASON})"
)


# =====================================================================
# OFFLINE UNIT & SECURITY TESTS (ALWAYS RUN)
# =====================================================================

class TestPostgresEngineOffline:
    """Unit tests that run offline without needing a live PostgreSQL server."""

    def test_exception_hierarchy(self):
        """Verify custom exception inheritance matches project conventions."""
        assert issubclass(PostgresConnectionError, PostgresEngineError)
        assert issubclass(DisallowedQueryError, PostgresEngineError)
        assert issubclass(QueryExecutionError, PostgresEngineError)

    def test_configuration_defaults(self, monkeypatch):
        """Verify fallback to sensible default configuration."""
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_PORT", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("POSTGRES_USER", raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

        engine = PostgresEngine()
        assert engine.host == "localhost"
        assert engine.port == 5432
        assert engine.database == "postgres"
        assert engine.user == "postgres"
        assert not engine.is_open

    def test_configuration_custom_arguments(self):
        """Verify explicit constructor arguments override defaults."""
        engine = PostgresEngine(
            host="db.example.com",
            port=5433,
            database="analytics_test",
            user="analyst",
            password="secret_password_xyz",
        )
        assert engine.host == "db.example.com"
        assert engine.port == 5433
        assert engine.database == "analytics_test"
        assert engine.user == "analyst"

    def test_password_redaction_in_repr_and_str(self):
        """Verify passwords are never leaked in repr, str, or log outputs."""
        secret = "super_secret_pg_pass_12345"
        engine = PostgresEngine(password=secret)
        repr_str = repr(engine)
        str_str = str(engine)

        assert secret not in repr_str, "Password must not appear in __repr__"
        assert secret not in str_str, "Password must not appear in __str__"
        assert "host=localhost" in repr_str

    def test_password_sanitization_in_error_messages(self):
        """Verify passwords are redacted from connection exception messages."""
        secret = "secret_db_password_never_expose"
        engine = PostgresEngine(password=secret)
        sanitized = engine._sanitize_error_message(f"Error authenticating with {secret} at host")
        assert secret not in sanitized
        assert "******" in sanitized

    def test_read_only_allowed_queries(self):
        """Verify safe analytical SQL verbs pass validation."""
        valid_queries = [
            "SELECT * FROM sales;",
            "select order_id, revenue from sales where quantity > 2",
            "WITH regional AS (SELECT region, SUM(revenue) FROM sales GROUP BY region) SELECT * FROM regional;",
            "EXPLAIN SELECT * FROM sales;",
            "SHOW search_path;",
        ]
        for query in valid_queries:
            validate_read_only_query(query)  # Should not raise

    def test_read_only_blocks_empty_queries(self):
        """Verify empty and whitespace queries are rejected."""
        with pytest.raises(DisallowedQueryError, match="cannot be empty"):
            validate_read_only_query("")
        with pytest.raises(DisallowedQueryError, match="cannot be empty"):
            validate_read_only_query("   \n\t  ")

    def test_read_only_blocks_mutating_verbs(self):
        """Verify destructive DDL and DML statements are strictly blocked."""
        disallowed = [
            ("INSERT INTO sales VALUES (1, 'Widget');", "INSERT"),
            ("UPDATE sales SET revenue = 0;", "UPDATE"),
            ("DELETE FROM sales WHERE revenue < 100;", "DELETE"),
            ("DROP TABLE sales;", "DROP"),
            ("ALTER TABLE sales ADD COLUMN status TEXT;", "ALTER"),
            ("TRUNCATE TABLE sales;", "TRUNCATE"),
            ("CREATE TABLE dummy (id INT);", "CREATE"),
            ("GRANT ALL ON sales TO public;", "GRANT"),
            ("REVOKE ALL ON sales FROM public;", "REVOKE"),
            ("VACUUM FULL sales;", "VACUUM"),
        ]
        for sql_cmd, keyword in disallowed:
            with pytest.raises(DisallowedQueryError, match=r"Disallowed SQL keyword detected|Disallowed statement type"):
                validate_read_only_query(sql_cmd)

    def test_read_only_blocks_multi_statement_injection(self):
        """Verify semicolons chaining multiple statements are rejected."""
        multi_sql = "SELECT * FROM sales; DROP TABLE sales;"
        with pytest.raises(DisallowedQueryError, match="Multiple SQL statements"):
            validate_read_only_query(multi_sql)

    def test_read_only_allows_keywords_inside_string_literals(self):
        """Verify queries with SQL keywords inside quoted strings are permitted."""
        valid_query = "SELECT * FROM sales WHERE product = 'DROP' AND category = 'CREATE';"
        validate_read_only_query(valid_query)  # Should not raise

    def test_read_only_allows_comments(self):
        """Verify queries with single-line and multi-line comments are safely sanitized."""
        query_with_comments = """
        -- Analytical report
        /* Multi-line
           comment block */
        SELECT region, SUM(revenue)
        FROM sales
        GROUP BY region;
        """
        validate_read_only_query(query_with_comments)

    def test_graceful_offline_connection_failure(self):
        """Verify connecting to an invalid port fails cleanly with PostgresConnectionError."""
        # Port 59999 is typically unused, connection should fail fast
        engine = PostgresEngine(host="127.0.0.1", port=59999, connect_timeout=1)
        with pytest.raises(PostgresConnectionError) as exc_info:
            engine.connect()
        assert "Failed to connect to PostgreSQL" in str(exc_info.value)
        assert not engine.is_open

    def test_type_validation_on_load(self):
        """Verify load_dataframe rejects non-DataFrame inputs."""
        engine = PostgresEngine()
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            engine.load_dataframe({"not": "a dataframe"})

    def test_table_name_validation_on_load(self):
        """Verify load_dataframe rejects invalid table identifiers."""
        engine = PostgresEngine()
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Invalid table name"):
            engine.load_dataframe(df, table_name="invalid table name; DROP")


# =====================================================================
# LIVE POSTGRESQL INTEGRATION TESTS (REQUIRES LIVE DB)
# =====================================================================

@requires_postgres
class TestPostgresEngineIntegration:
    """Integration tests that execute against a live PostgreSQL server."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        """Creates and cleans up a live engine for testing."""
        self.engine = PostgresEngine()
        self.engine.connect()
        yield
        self.engine.close()

    def test_live_connection_lifecycle(self):
        """Verify connection can be opened and closed properly."""
        assert self.engine.is_open
        assert self.engine.is_connected
        self.engine.close()
        assert not self.engine.is_open

    def test_load_sample_sales_dataframe(self):
        """Verify loading sample_sales.csv into 'sales' table."""
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        assert len(df) == 100, f"Expected 100 rows in sample_sales.csv, got {len(df)}"

        # Deep copy snapshot to verify immutability
        df_snapshot = df.copy(deep=True)

        self.engine.load_dataframe(df, table_name="sales")
        assert "sales" in self.engine.get_registered_tables()

        # Verify original DataFrame was not mutated
        assert df.equals(df_snapshot), "Caller DataFrame was modified by load_dataframe!"

    def test_row_count_verification(self):
        """Verify exactly 100 rows are present in the sales table."""
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        self.engine.load_dataframe(df, table_name="sales")

        result = self.engine.execute_query("SELECT COUNT(*) AS total_rows FROM sales;")
        assert isinstance(result, pd.DataFrame)
        assert "total_rows" in result.columns
        count = int(result["total_rows"].iloc[0])
        assert count == 100, f"Expected 100 rows, got {count}"

    def test_schema_retrieval(self):
        """Verify schema extraction returns column names and types compatible with NL-to-SQL."""
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        self.engine.load_dataframe(df, table_name="sales")

        schema = self.engine.get_schema("sales")
        assert isinstance(schema, dict)

        expected_cols = [
            "order_id", "order_date", "customer_name", "product",
            "category", "region", "quantity", "unit_price", "revenue", "payment_method"
        ]
        for col in expected_cols:
            assert col in schema, f"Column '{col}' missing from retrieved PostgreSQL schema"

    def test_select_limit_query(self):
        """Verify SELECT * FROM sales LIMIT 5 returns exactly 5 rows."""
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        self.engine.load_dataframe(df, table_name="sales")

        res = self.engine.execute_query("SELECT * FROM sales LIMIT 5;")
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 5
        assert len(res.columns) == 10

    def test_regional_revenue_aggregation(self):
        """
        Verify aggregation query matches DuckDB ground truth:
        East: 207141.0, South: 159251.0, North: 112362.0, Central: 82061.0, West: 80206.0
        """
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        self.engine.load_dataframe(df, table_name="sales")

        sql = """
        SELECT region, SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC;
        """
        res = self.engine.execute_query(sql)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 5
        assert list(res["region"]) == ["East", "South", "North", "Central", "West"]
        revenues = [float(r) for r in res["total_revenue"]]
        assert revenues == [207141.0, 159251.0, 112362.0, 82061.0, 80206.0]

    def test_product_revenue_aggregation(self):
        """Verify product revenue aggregation executes in descending order."""
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        self.engine.load_dataframe(df, table_name="sales")

        sql = """
        SELECT product, SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY product
        ORDER BY total_revenue DESC;
        """
        res = self.engine.execute_query(sql)
        assert isinstance(res, pd.DataFrame)
        assert len(res) > 0
        revenues = list(res["total_revenue"])
        assert revenues == sorted(revenues, reverse=True)

    def test_query_execution_error_on_invalid_sql(self):
        """Verify syntax errors in PostgreSQL raise QueryExecutionError."""
        csv_path = "sample_data/sample_sales.csv"
        df = load_dataset(csv_path)
        self.engine.load_dataframe(df, table_name="sales")

        with pytest.raises(QueryExecutionError) as exc_info:
            self.engine.execute_query("SELECT non_existent_column_xyz FROM sales;")
        assert "column" in str(exc_info.value).lower()

    def test_context_manager_usage(self):
        """Verify context manager open/close behavior."""
        with PostgresEngine() as engine:
            assert engine.is_open
        assert not engine.is_open


# =====================================================================
# STANDALONE CLI TEST RUNNER
# =====================================================================

def run_standalone():
    """Runs tests cleanly when invoked as `python test_postgres_engine.py`."""
    print("=" * 78)
    print(" POSTGRESQL ANALYTICS ENGINE TEST SUITE (PHASE 13A)")
    print("=" * 78)

    # 1. Offline Tests
    print("\n--- Running Offline Unit & Security Tests ---")
    offline = TestPostgresEngineOffline()
    offline.test_exception_hierarchy()
    print(" [PASS] Exception hierarchy verified.")
    offline.test_configuration_custom_arguments()
    print(" [PASS] Configuration resolution verified.")
    offline.test_password_redaction_in_repr_and_str()
    print(" [PASS] Password privacy in repr/str verified.")
    offline.test_password_sanitization_in_error_messages()
    print(" [PASS] Password sanitization in error strings verified.")
    offline.test_read_only_allowed_queries()
    print(" [PASS] Read-only allowed queries (SELECT, WITH, EXPLAIN, SHOW) accepted.")
    offline.test_read_only_blocks_empty_queries()
    print(" [PASS] Empty queries rejected.")
    offline.test_read_only_blocks_mutating_verbs()
    print(" [PASS] Mutating DDL/DML verbs rejected (INSERT, UPDATE, DELETE, DROP, etc.).")
    offline.test_read_only_blocks_multi_statement_injection()
    print(" [PASS] Multi-statement injection rejected.")
    offline.test_read_only_allows_keywords_inside_string_literals()
    print(" [PASS] String literals with keywords safely permitted.")
    offline.test_read_only_allows_comments()
    print(" [PASS] Comments stripped cleanly.")
    offline.test_graceful_offline_connection_failure()
    print(" [PASS] Graceful connection failure verified.")
    offline.test_type_validation_on_load()
    print(" [PASS] Type validation on load verified.")
    offline.test_table_name_validation_on_load()
    print(" [PASS] Table name validation on load verified.")

    # 2. Live Integration Tests
    print("\n--- PostgreSQL Live Integration Status ---")
    if not PG_AVAILABLE:
        print(f" [SKIP] Live PostgreSQL integration skipped: {PG_UNAVAILABLE_REASON}")
        print("  -> Tip: To enable live integration tests, set POSTGRES_PASSWORD in .env")
    else:
        print(" [INFO] PostgreSQL server is reachable. Running live integration tests...")
        integration = TestPostgresEngineIntegration()
        integration.setup_engine = lambda: None
        integration.engine = PostgresEngine()
        integration.engine.connect()
        try:
            integration.test_load_sample_sales_dataframe()
            print(" [PASS] Loaded sample_sales.csv into PostgreSQL table 'sales' (100 rows).")
            integration.test_row_count_verification()
            print(" [PASS] Verified row count == 100.")
            integration.test_schema_retrieval()
            print(" [PASS] Schema extraction verified.")
            integration.test_select_limit_query()
            print(" [PASS] SELECT * LIMIT 5 verified.")
            integration.test_regional_revenue_aggregation()
            print(" [PASS] Regional revenue aggregation verified against DuckDB ground truth.")
            integration.test_product_revenue_aggregation()
            print(" [PASS] Product revenue aggregation verified.")
            integration.test_query_execution_error_on_invalid_sql()
            print(" [PASS] Invalid SQL execution error caught cleanly.")
        finally:
            integration.engine.close()

    print("\n" + "=" * 78)
    print(" POSTGRESQL ENGINE TEST EXECUTION COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    run_standalone()
