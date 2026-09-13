"""
Tests for Analytics Database Abstraction & Factory Pattern (Phase 13B).

Validates:
1. AnalyticsDatabase abstract base class and interface definition.
2. DuckDBEngine conforms to AnalyticsDatabase (subclass and instance checks).
3. PostgresEngine conforms to AnalyticsDatabase (subclass and instance checks).
4. create_database_engine("duckdb") constructs a DuckDBEngine.
5. create_database_engine("postgres") and ("postgresql") construct a PostgresEngine.
6. Unsupported database type raises UnsupportedDatabaseError (which inherits from ValueError).
7. Common execute_query contract returns pd.DataFrame across engines.
8. Common get_schema contract returns Dict[str, str] compatible with nl_to_sql.format_schema.
9. Common load_dataframe contract loads 100 rows into table 'sales' without mutating caller df.
10. Context manager protocol works across engines.
11. Polymorphic execution test demonstrating engine-swapping with identical analytical results.
"""
import pytest
import pandas as pd
from typing import Dict, Tuple

from src.data_quality.loader import load_dataset
from src.analytics.database import (
    AnalyticsDatabase,
    create_database_engine,
    DatabaseEngineError,
    UnsupportedDatabaseError,
)
from src.analytics.duckdb_engine import DuckDBEngine
from src.analytics.postgres_engine import PostgresEngine
from src.analytics.nl_to_sql import format_schema


# =====================================================================
# POSTGRES AVAILABILITY CHECK
# =====================================================================

def check_postgres_availability() -> Tuple[bool, str]:
    """Checks if live PostgreSQL is reachable using environment variables."""
    engine = PostgresEngine(connect_timeout=2)
    try:
        engine.connect()
        engine.close()
        return True, ""
    except Exception as e:
        return False, str(e)


PG_AVAILABLE, PG_REASON = check_postgres_availability()

requires_postgres = pytest.mark.skipif(
    not PG_AVAILABLE,
    reason=f"PostgreSQL server or credentials not available ({PG_REASON})"
)


# =====================================================================
# ABSTRACTION & INTERFACE CONFORMANCE TESTS
# =====================================================================

class TestDatabaseAbstractionConformance:
    """Tests verifying the interface contract and conformance of both engines."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Verify AnalyticsDatabase cannot be instantiated directly without implementations."""
        with pytest.raises(TypeError):
            AnalyticsDatabase()  # type: ignore

    def test_incomplete_subclass_cannot_be_instantiated(self):
        """Verify subclasses missing required abstract methods cannot be instantiated."""
        class IncompleteEngine(AnalyticsDatabase):
            def connect(self) -> None: pass
            def close(self) -> None: pass

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore

    def test_duckdb_engine_conforms_to_abstraction(self):
        """Verify DuckDBEngine satisfies the AnalyticsDatabase interface."""
        assert issubclass(DuckDBEngine, AnalyticsDatabase)
        engine = DuckDBEngine()
        try:
            assert isinstance(engine, AnalyticsDatabase)
            assert hasattr(engine, "connect")
            assert hasattr(engine, "close")
            assert hasattr(engine, "is_open")
            assert hasattr(engine, "is_connected")
            assert hasattr(engine, "load_dataframe")
            assert hasattr(engine, "register_dataframe")
            assert hasattr(engine, "execute_query")
            assert hasattr(engine, "get_schema")
        finally:
            engine.close()

    def test_postgres_engine_conforms_to_abstraction(self):
        """Verify PostgresEngine satisfies the AnalyticsDatabase interface."""
        assert issubclass(PostgresEngine, AnalyticsDatabase)
        engine = PostgresEngine()
        assert isinstance(engine, AnalyticsDatabase)
        assert hasattr(engine, "connect")
        assert hasattr(engine, "close")
        assert hasattr(engine, "is_open")
        assert hasattr(engine, "is_connected")
        assert hasattr(engine, "load_dataframe")
        assert hasattr(engine, "register_dataframe")
        assert hasattr(engine, "execute_query")
        assert hasattr(engine, "get_schema")


# =====================================================================
# FACTORY FUNCTION TESTS
# =====================================================================

class TestDatabaseFactory:
    """Tests for create_database_engine factory function."""

    def test_factory_creates_duckdb(self):
        """Verify factory constructs DuckDBEngine with 'duckdb'."""
        engine = create_database_engine("duckdb")
        try:
            assert isinstance(engine, DuckDBEngine)
            assert isinstance(engine, AnalyticsDatabase)
            assert engine.is_open
        finally:
            engine.close()

    def test_factory_creates_duckdb_case_insensitive(self):
        """Verify factory handles whitespace and case insensitivity."""
        engine = create_database_engine("  DUCKDB  ")
        try:
            assert isinstance(engine, DuckDBEngine)
        finally:
            engine.close()

    def test_factory_creates_postgres(self):
        """Verify factory constructs PostgresEngine with 'postgres' and 'postgresql'."""
        engine1 = create_database_engine("postgres")
        assert isinstance(engine1, PostgresEngine)
        assert isinstance(engine1, AnalyticsDatabase)

        engine2 = create_database_engine("postgresql")
        assert isinstance(engine2, PostgresEngine)
        assert isinstance(engine2, AnalyticsDatabase)

    def test_factory_creates_postgres_case_insensitive(self):
        """Verify factory handles whitespace and case insensitivity for postgres."""
        engine = create_database_engine("  PostgreSQL  ")
        assert isinstance(engine, PostgresEngine)

    def test_factory_unsupported_database_raises_error(self):
        """Verify unsupported database identifiers raise UnsupportedDatabaseError."""
        for invalid_type in ["sqlite", "mysql", "oracle", "bigquery", "snowflake"]:
            with pytest.raises(UnsupportedDatabaseError, match="Unsupported database type"):
                create_database_engine(invalid_type)

    def test_factory_invalid_inputs_raise_error(self):
        """Verify empty string or non-string inputs raise UnsupportedDatabaseError."""
        with pytest.raises(UnsupportedDatabaseError):
            create_database_engine("")
        with pytest.raises(UnsupportedDatabaseError):
            create_database_engine("   ")
        with pytest.raises(UnsupportedDatabaseError):
            create_database_engine(None)  # type: ignore

    def test_exception_inheritance(self):
        """Verify UnsupportedDatabaseError inherits from ValueError and DatabaseEngineError."""
        assert issubclass(UnsupportedDatabaseError, ValueError)
        assert issubclass(UnsupportedDatabaseError, DatabaseEngineError)


# =====================================================================
# COMMON CONTRACT TESTS — DUCKDB
# =====================================================================

class TestDuckDBCommonContract:
    """Validates the common contract against DuckDBEngine."""

    @pytest.fixture
    def sales_df(self):
        return load_dataset("sample_data/sample_sales.csv")

    def test_duckdb_load_and_query_contract(self, sales_df):
        """Verify load_dataframe and execute_query contract with DuckDB."""
        df_copy = sales_df.copy(deep=True)

        with create_database_engine("duckdb") as db:
            assert db.is_connected
            db.load_dataframe(sales_df, table_name="sales")

            # Contract: caller DataFrame must remain unmodified
            assert sales_df.equals(df_copy)

            # Contract: execute_query returns pd.DataFrame
            count_res = db.execute_query("SELECT COUNT(*) AS total_rows FROM sales;")
            assert isinstance(count_res, pd.DataFrame)
            assert int(count_res["total_rows"].iloc[0]) == 100

            # Contract: schema returns Dict[str, str] compatible with format_schema
            schema = db.get_schema("sales")
            assert isinstance(schema, dict)
            assert "revenue" in schema and "order_id" in schema
            formatted = format_schema(schema)
            assert "Table: sales" in formatted
            assert "revenue" in formatted

        # Contract: context manager closes connection
        assert not db.is_open

    def test_duckdb_register_dataframe_parity(self, sales_df):
        """Verify register_dataframe parity alias works identically."""
        with create_database_engine("duckdb") as db:
            db.register_dataframe(sales_df, table_name="sales")
            res = db.execute_query("SELECT COUNT(*) AS total_rows FROM sales;")
            assert int(res["total_rows"].iloc[0]) == 100


# =====================================================================
# COMMON CONTRACT TESTS — POSTGRESQL (LIVE INTEGRATION)
# =====================================================================

@requires_postgres
class TestPostgresCommonContract:
    """Validates the common contract against PostgresEngine using live local DB."""

    @pytest.fixture
    def sales_df(self):
        return load_dataset("sample_data/sample_sales.csv")

    def test_postgres_load_and_query_contract(self, sales_df):
        """Verify load_dataframe and execute_query contract with PostgreSQL."""
        df_copy = sales_df.copy(deep=True)

        with create_database_engine("postgres") as db:
            assert db.is_connected
            db.load_dataframe(sales_df, table_name="sales")

            # Contract: caller DataFrame must remain unmodified
            assert sales_df.equals(df_copy)

            # Contract: execute_query returns pd.DataFrame
            count_res = db.execute_query("SELECT COUNT(*) AS total_rows FROM sales;")
            assert isinstance(count_res, pd.DataFrame)
            assert int(count_res["total_rows"].iloc[0]) == 100

            # Contract: schema returns Dict[str, str] compatible with format_schema
            schema = db.get_schema("sales")
            assert isinstance(schema, dict)
            assert "revenue" in schema and "order_id" in schema
            formatted = format_schema(schema)
            assert "Table: sales" in formatted
            assert "revenue" in formatted

        # Contract: context manager closes connection
        assert not db.is_open

    def test_postgres_register_dataframe_parity(self, sales_df):
        """Verify register_dataframe parity alias works identically on PostgreSQL."""
        with create_database_engine("postgres") as db:
            db.register_dataframe(sales_df, table_name="sales")
            res = db.execute_query("SELECT COUNT(*) AS total_rows FROM sales;")
            assert int(res["total_rows"].iloc[0]) == 100


# =====================================================================
# POLYMORPHIC ENGINE EXECUTION COMPARISON
# =====================================================================

@requires_postgres
class TestPolymorphicExecution:
    """Verifies that higher-level analytical logic produces identical results across engines."""

    def test_polymorphic_regional_analysis(self):
        """
        Executes the exact same analytical pipeline through the AnalyticsDatabase
        abstraction against both DuckDB and PostgreSQL. Verifies identical outputs.
        """
        df = load_dataset("sample_data/sample_sales.csv")

        def run_analytics_pipeline(engine: AnalyticsDatabase) -> pd.DataFrame:
            with engine as db:
                db.load_dataframe(df, table_name="sales")
                query = """
                SELECT region, SUM(revenue) AS total_revenue
                FROM sales
                GROUP BY region
                ORDER BY total_revenue DESC;
                """
                return db.execute_query(query)

        duckdb_res = run_analytics_pipeline(create_database_engine("duckdb"))
        postgres_res = run_analytics_pipeline(create_database_engine("postgres"))

        # Both must return DataFrames with 5 regions
        assert len(duckdb_res) == 5
        assert len(postgres_res) == 5

        # Region ordering must match exactly
        assert list(duckdb_res["region"]) == list(postgres_res["region"])

        # Revenue numbers must match exactly
        duck_rev = [float(x) for x in duckdb_res["total_revenue"]]
        pg_rev = [float(x) for x in postgres_res["total_revenue"]]
        assert duck_rev == pg_rev == [207141.0, 159251.0, 112362.0, 82061.0, 80206.0]


# =====================================================================
# STANDALONE CLI TEST RUNNER
# =====================================================================

def run_standalone():
    """Runs tests cleanly when invoked as `python test_database_abstraction.py`."""
    print("=" * 78)
    print(" DATABASE ABSTRACTION & FACTORY TEST SUITE (PHASE 13B)")
    print("=" * 78)

    # 1. Interface conformance
    print("\n--- Testing Interface Conformance ---")
    conformance = TestDatabaseAbstractionConformance()
    conformance.test_abstract_class_cannot_be_instantiated()
    conformance.test_incomplete_subclass_cannot_be_instantiated()
    conformance.test_duckdb_engine_conforms_to_abstraction()
    conformance.test_postgres_engine_conforms_to_abstraction()
    print(" [PASS] AnalyticsDatabase interface definition & conformance verified.")

    # 2. Factory
    print("\n--- Testing Database Factory ---")
    factory = TestDatabaseFactory()
    factory.test_factory_creates_duckdb()
    factory.test_factory_creates_duckdb_case_insensitive()
    factory.test_factory_creates_postgres()
    factory.test_factory_creates_postgres_case_insensitive()
    factory.test_factory_unsupported_database_raises_error()
    factory.test_factory_invalid_inputs_raise_error()
    factory.test_exception_inheritance()
    print(" [PASS] Factory function and error handling verified.")

    # 3. DuckDB Common Contract
    print("\n--- Testing DuckDB Common Contract ---")
    sales_df = load_dataset("sample_data/sample_sales.csv")
    duck_tests = TestDuckDBCommonContract()
    duck_tests.test_duckdb_load_and_query_contract(sales_df)
    duck_tests.test_duckdb_register_dataframe_parity(sales_df)
    print(" [PASS] DuckDB load, query, schema, and context manager contract verified.")

    # 4. PostgreSQL Common Contract
    print("\n--- Testing PostgreSQL Common Contract ---")
    if not PG_AVAILABLE:
        print(f" [SKIP] Live PostgreSQL common contract skipped: {PG_REASON}")
    else:
        pg_tests = TestPostgresCommonContract()
        pg_tests.test_postgres_load_and_query_contract(sales_df)
        pg_tests.test_postgres_register_dataframe_parity(sales_df)
        print(" [PASS] PostgreSQL load, query, schema, and context manager contract verified.")

        # 5. Polymorphic Execution
        print("\n--- Testing Polymorphic Execution (DuckDB vs PostgreSQL) ---")
        poly = TestPolymorphicExecution()
        poly.test_polymorphic_regional_analysis()
        print(" [PASS] Polymorphic pipeline execution verified: Identical results across engines.")

    print("\n" + "=" * 78)
    print(" ALL ABSTRACTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_standalone()
