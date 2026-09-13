"""
Tests for Backend-Independent Dataset Loader (Phase 13).

Validates:
1. DatasetLoadResult structured metadata.
2. Input validation (type checking on engine, DataFrame, table_name).
3. Loading clean dataset (sample_sales.csv) into DuckDB.
4. Loading clean dataset (sample_sales.csv) into PostgreSQL.
5. Immutability guarantee: caller's DataFrame is never mutated.
6. Loading dirty dataset (sample_dirty_data.xlsx) into DuckDB and PostgreSQL.
7. End-to-end schema extraction compatibility with format_schema.
"""
import pytest
import pandas as pd
from typing import Tuple

from src.data_quality.loader import load_dataset
from src.analytics.database import (
    AnalyticsDatabase,
    create_database_engine,
)
from src.analytics.postgres_engine import PostgresEngine
from src.analytics.dataset_loader import (
    load_dataset_into_database,
    DatasetLoadResult,
    DatasetLoaderError,
    InvalidDataFrameError,
)
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
# UNIT & VALIDATION TESTS
# =====================================================================

class TestDatasetLoaderValidation:
    """Tests input validation and error handling in load_dataset_into_database."""

    def test_invalid_engine_type_raises_type_error(self):
        """Verify TypeError when an invalid object is passed as engine."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError, match="AnalyticsDatabase"):
            load_dataset_into_database("not_an_engine", df)  # type: ignore

    def test_invalid_dataframe_raises_error(self):
        """Verify InvalidDataFrameError when non-DataFrame is provided."""
        engine = create_database_engine("duckdb")
        try:
            with pytest.raises(InvalidDataFrameError, match="Expected a pandas DataFrame"):
                load_dataset_into_database(engine, {"col": [1, 2]})  # type: ignore
        finally:
            engine.close()

    def test_empty_dataframe_without_columns_raises_error(self):
        """Verify DataFrame without columns is rejected."""
        engine = create_database_engine("duckdb")
        try:
            empty_df = pd.DataFrame()
            with pytest.raises(InvalidDataFrameError, match="no columns"):
                load_dataset_into_database(engine, empty_df)
        finally:
            engine.close()

    def test_invalid_table_name_raises_error(self):
        """Verify SQL injection or invalid identifiers in table_name are rejected."""
        engine = create_database_engine("duckdb")
        try:
            df = pd.DataFrame({"x": [10]})
            for bad_name in ["sales; DROP TABLE sales;", "sales table", "123sales", ""]:
                with pytest.raises(InvalidDataFrameError, match="Invalid table name"):
                    load_dataset_into_database(engine, df, table_name=bad_name)
        finally:
            engine.close()

    def test_dataset_load_result_immutability(self):
        """Verify DatasetLoadResult is a frozen dataclass."""
        res = DatasetLoadResult(
            table_name="sales",
            row_count=10,
            column_count=2,
            columns=["a", "b"],
            schema={"a": "INT", "b": "TEXT"},
        )
        assert res.table_name == "sales"
        assert res.row_count == 10
        assert res.column_count == 2
        with pytest.raises(Exception):
            res.row_count = 20  # Frozen dataclass should reject attribute mutation


# =====================================================================
# DUCKDB LOADING TESTS
# =====================================================================

class TestDuckDBDatasetLoading:
    """Tests loading datasets into DuckDB via load_dataset_into_database."""

    @pytest.fixture
    def sales_df(self):
        return load_dataset("sample_data/sample_sales.csv")

    def test_load_sample_sales_into_duckdb(self, sales_df):
        """Verify clean dataset loading into DuckDB."""
        snapshot = sales_df.copy(deep=True)

        with create_database_engine("duckdb") as engine:
            result = load_dataset_into_database(engine, sales_df, table_name="sales")

            assert isinstance(result, DatasetLoadResult)
            assert result.table_name == "sales"
            assert result.row_count == 100
            assert result.column_count == 10
            assert len(result.columns) == 10
            assert "revenue" in result.schema
            assert "order_id" in result.schema

            # Verify caller DataFrame was never mutated
            assert sales_df.equals(snapshot), "Caller DataFrame was mutated by dataset loader!"

            # Verify SQL query returns matching count
            count_df = engine.execute_query("SELECT COUNT(*) AS cnt FROM sales;")
            assert int(count_df["cnt"].iloc[0]) == 100

            # Verify format_schema works with returned schema
            formatted = format_schema(result.schema)
            assert "Table: sales" in formatted
            assert "revenue" in formatted

    def test_load_dirty_dataset_into_duckdb(self):
        """Verify dirty dataset loading into DuckDB without automatic mutation."""
        dirty_df = load_dataset("sample_data/sample_dirty_data.xlsx")
        assert len(dirty_df) == 100

        with create_database_engine("duckdb") as engine:
            result = load_dataset_into_database(engine, dirty_df, table_name="sales")
            assert result.row_count == 100
            assert result.column_count == 10
            count_df = engine.execute_query("SELECT COUNT(*) AS cnt FROM sales;")
            assert int(count_df["cnt"].iloc[0]) == 100


# =====================================================================
# POSTGRESQL LOADING TESTS (LIVE INTEGRATION)
# =====================================================================

@requires_postgres
class TestPostgresDatasetLoading:
    """Tests loading datasets into PostgreSQL via load_dataset_into_database."""

    @pytest.fixture
    def sales_df(self):
        return load_dataset("sample_data/sample_sales.csv")

    def test_load_sample_sales_into_postgres(self, sales_df):
        """Verify clean dataset loading into PostgreSQL."""
        snapshot = sales_df.copy(deep=True)

        with create_database_engine("postgres") as engine:
            result = load_dataset_into_database(engine, sales_df, table_name="sales")

            assert isinstance(result, DatasetLoadResult)
            assert result.table_name == "sales"
            assert result.row_count == 100
            assert result.column_count == 10
            assert "revenue" in result.schema
            assert "order_id" in result.schema

            # Immutability
            assert sales_df.equals(snapshot), "Caller DataFrame was mutated!"

            # SQL execution
            count_df = engine.execute_query("SELECT COUNT(*) AS cnt FROM sales;")
            assert int(count_df["cnt"].iloc[0]) == 100

            # Schema formatting
            formatted = format_schema(result.schema)
            assert "Table: sales" in formatted
            assert "revenue" in formatted

    def test_load_dirty_dataset_into_postgres(self):
        """Verify dirty dataset with mixed types loads cleanly into PostgreSQL."""
        dirty_df = load_dataset("sample_data/sample_dirty_data.xlsx")
        assert len(dirty_df) == 100
        snapshot = dirty_df.copy(deep=True)

        with create_database_engine("postgres") as engine:
            result = load_dataset_into_database(engine, dirty_df, table_name="sales")
            assert result.row_count == 100
            assert result.column_count == 10

            # Mixed unit_price column should be TEXT in PostgreSQL
            assert result.schema.get("unit_price") == "TEXT"

            # Verify count
            count_df = engine.execute_query("SELECT COUNT(*) AS cnt FROM sales;")
            assert int(count_df["cnt"].iloc[0]) == 100

            # Caller unmodified
            assert dirty_df.equals(snapshot)


# =====================================================================
# STANDALONE CLI TEST RUNNER
# =====================================================================

def run_standalone():
    """Runs tests cleanly when invoked as `python test_dataset_loader.py`."""
    print("=" * 78)
    print(" DATASET LOADER TEST SUITE (PHASE 13)")
    print("=" * 78)

    print("\n--- Testing Input Validation ---")
    val = TestDatasetLoaderValidation()
    val.test_invalid_engine_type_raises_type_error()
    val.test_invalid_dataframe_raises_error()
    val.test_empty_dataframe_without_columns_raises_error()
    val.test_invalid_table_name_raises_error()
    val.test_dataset_load_result_immutability()
    print(" [PASS] Input validation and metadata immutability verified.")

    print("\n--- Testing DuckDB Dataset Loading ---")
    sales_df = load_dataset("sample_data/sample_sales.csv")
    duck = TestDuckDBDatasetLoading()
    duck.test_load_sample_sales_into_duckdb(sales_df)
    duck.test_load_dirty_dataset_into_duckdb()
    print(" [PASS] DuckDB dataset loading (clean & dirty data) verified.")

    print("\n--- Testing PostgreSQL Dataset Loading ---")
    if not PG_AVAILABLE:
        print(f" [SKIP] Live PostgreSQL dataset loading skipped: {PG_REASON}")
    else:
        pg = TestPostgresDatasetLoading()
        pg.test_load_sample_sales_into_postgres(sales_df)
        pg.test_load_dirty_dataset_into_postgres()
        print(" [PASS] PostgreSQL dataset loading (clean & dirty data) verified.")

    print("\n" + "=" * 78)
    print(" ALL DATASET LOADER TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_standalone()
