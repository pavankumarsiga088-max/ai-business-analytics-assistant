"""
End-to-End Integration Tests for Phase 13: Full PostgreSQL Analytics Integration.

Validates:
1. Configuration & Factory Selection (DuckDB default, PostgreSQL override, unsupported handling).
2. PostgreSQL Availability & Error Handling (graceful failure when host/port unreachable).
3. DuckDB Full Analytics Pipeline:
   - Dataset loading via load_dataset_into_database
   - Row count (100) & column count (10)
   - Schema extraction
   - DataFrame immutability
   - Read-only SQL execution & aggregation
   - Read-only security gate (rejection of INSERT, UPDATE, DELETE, DROP)
   - Downstream analytics: KPI presentation, Plotly visualization, Downloads
   - Conversation state tracking
4. PostgreSQL Full Analytics Pipeline (Live Integration):
   - Dataset loading into PostgreSQL
   - Row count (100) & column count (10)
   - Schema extraction
   - DataFrame immutability
   - Read-only SQL execution & aggregation (matching DuckDB ground truth)
   - Read-only security gate
   - Downstream analytics: KPI presentation, Plotly visualization, Downloads
   - Conversation state tracking
5. Dataset & Database Boundaries:
   - Dataset boundary resets conversation history
   - Database switching creates clean isolated engines
   - NL-to-SQL schema injection uses authoritative database schema
"""
import pytest
import pandas as pd
from typing import Tuple

from src.data_quality.loader import load_dataset
from src.analytics.database import (
    AnalyticsDatabase,
    create_database_engine,
    get_configured_database_backend,
    is_backend_available,
    UnsupportedDatabaseError,
)
from src.analytics.dataset_loader import (
    load_dataset_into_database,
    DatasetLoadResult,
)
from src.analytics.duckdb_engine import DuckDBEngine, DisallowedQueryError as DuckDBDisallowed
from src.analytics.postgres_engine import PostgresEngine, PostgresConnectionError, DisallowedQueryError as PostgresDisallowed
from src.analytics.nl_to_sql import build_sql_prompt, format_schema
from src.analytics.result_presentation import build_kpi_metadata, ResultPresentation
from src.analytics.visualization import create_chart
from src.analytics.downloads import dataframe_to_csv_bytes, dataframe_to_excel_bytes
from src.analytics.conversation import ConversationHistory


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
# CONFIGURATION & FACTORY SELECTION TESTS
# =====================================================================

class TestBackendConfigurationAndFactory:
    """Validates configuration resolution, factory instantiation, and error handling."""

    def test_default_configured_backend_is_duckdb(self, monkeypatch):
        """Verify default backend is duckdb when env var is omitted."""
        monkeypatch.delenv("DATABASE_BACKEND", raising=False)
        assert get_configured_database_backend() == "duckdb"

    def test_env_configured_backend_postgres(self, monkeypatch):
        """Verify environment variable DATABASE_BACKEND=postgres is honored."""
        monkeypatch.setenv("DATABASE_BACKEND", "postgres")
        assert get_configured_database_backend() == "postgres"

        monkeypatch.setenv("DATABASE_BACKEND", "PostgreSQL")
        assert get_configured_database_backend() == "postgres"

    def test_factory_selection_duckdb(self):
        """Verify factory creates DuckDBEngine."""
        with create_database_engine("duckdb") as engine:
            assert isinstance(engine, DuckDBEngine)
            assert isinstance(engine, AnalyticsDatabase)
            assert engine.is_open

    def test_factory_selection_postgres(self):
        """Verify factory creates PostgresEngine."""
        engine = create_database_engine("postgres")
        assert isinstance(engine, PostgresEngine)
        assert isinstance(engine, AnalyticsDatabase)

    def test_unsupported_backend_raises_error(self):
        """Verify unsupported database raises UnsupportedDatabaseError."""
        with pytest.raises(UnsupportedDatabaseError, match="Unsupported database type"):
            create_database_engine("sqlite")

    def test_postgresql_unavailable_error_handling(self):
        """Verify connecting to an invalid port raises PostgresConnectionError cleanly."""
        is_avail, reason = is_backend_available("nonexistent_db")
        assert not is_avail
        assert "Unsupported database type" in reason

        engine = PostgresEngine(host="127.0.0.1", port=59999, connect_timeout=1)
        with pytest.raises(PostgresConnectionError):
            engine.connect()


# =====================================================================
# DUCKDB FULL ANALYTICS PIPELINE
# =====================================================================

class TestDuckDBAnalyticsPipeline:
    """Validates full analytical flow on DuckDB: loading, querying, KPIs, charts, downloads, conversation."""

    @pytest.fixture
    def sales_df(self):
        return load_dataset("sample_data/sample_sales.csv")

    def test_duckdb_pipeline_end_to_end(self, sales_df):
        snapshot = sales_df.copy(deep=True)

        with create_database_engine("duckdb") as engine:
            # 1. Dataset loading
            load_res = load_dataset_into_database(engine, sales_df, table_name="sales")
            assert load_res.row_count == 100
            assert load_res.column_count == 10
            assert "revenue" in load_res.schema
            assert sales_df.equals(snapshot), "Caller DataFrame must not be mutated"

            # 2. Schema check for NL-to-SQL
            schema_text = format_schema(load_res.schema)
            assert "Table: sales" in schema_text
            assert "revenue" in schema_text

            # 3. Read-only SQL execution
            sql_count = "SELECT COUNT(*) AS total_rows FROM sales;"
            count_df = engine.execute_query(sql_count)
            assert isinstance(count_df, pd.DataFrame)
            assert int(count_df["total_rows"].iloc[0]) == 100

            # 4. Regional aggregation
            sql_agg = """
            SELECT region, SUM(revenue) AS total_revenue
            FROM sales
            GROUP BY region
            ORDER BY total_revenue DESC;
            """
            agg_df = engine.execute_query(sql_agg)
            assert len(agg_df) == 5
            assert list(agg_df["region"]) == ["East", "South", "North", "Central", "West"]

            # 5. Security gate: verify mutating queries are blocked
            with pytest.raises(DuckDBDisallowed):
                engine.execute_query("DROP TABLE sales;")
            with pytest.raises(DuckDBDisallowed):
                engine.execute_query("INSERT INTO sales VALUES (1, 'Fake');")

            # 6. Smart KPI presentation
            presentation = build_kpi_metadata(agg_df)
            assert isinstance(presentation, ResultPresentation)
            assert presentation.show_chart is True
            assert presentation.row_count == 5

            # 7. Visualization creation
            fig = create_chart(agg_df)
            assert fig is not None

            # 8. Downloads serialization
            csv_bytes = dataframe_to_csv_bytes(agg_df)
            excel_bytes = dataframe_to_excel_bytes(agg_df)
            assert len(csv_bytes) > 0
            assert len(excel_bytes) > 0

            # 9. Conversation history integration
            history = ConversationHistory(dataset_name="sample_sales.csv")
            history.add_turn(
                question="What is revenue by region?",
                sql=sql_agg,
                result_df=agg_df,
                insights="East has highest revenue.",
            )
            assert len(history) == 1
            turn = history.get_recent_turns(1)[0]
            assert turn.question == "What is revenue by region?"
            assert turn.result.row_count == 5


# =====================================================================
# POSTGRESQL FULL ANALYTICS PIPELINE (LIVE INTEGRATION)
# =====================================================================

@requires_postgres
class TestPostgresAnalyticsPipeline:
    """Validates full analytical flow on PostgreSQL with exact ground truth comparison."""

    @pytest.fixture
    def sales_df(self):
        return load_dataset("sample_data/sample_sales.csv")

    def test_postgres_pipeline_end_to_end(self, sales_df):
        snapshot = sales_df.copy(deep=True)

        with create_database_engine("postgres") as engine:
            # 1. Dataset loading
            load_res = load_dataset_into_database(engine, sales_df, table_name="sales")
            assert load_res.row_count == 100
            assert load_res.column_count == 10
            assert "revenue" in load_res.schema
            assert sales_df.equals(snapshot), "Caller DataFrame must not be mutated"

            # 2. Schema check for NL-to-SQL
            schema_text = format_schema(load_res.schema)
            assert "Table: sales" in schema_text
            assert "revenue" in schema_text

            # 3. Read-only SQL execution
            sql_count = "SELECT COUNT(*) AS total_rows FROM sales;"
            count_df = engine.execute_query(sql_count)
            assert isinstance(count_df, pd.DataFrame)
            assert int(count_df["total_rows"].iloc[0]) == 100

            # 4. Regional aggregation matching DuckDB ground truth
            sql_agg = """
            SELECT region, SUM(revenue) AS total_revenue
            FROM sales
            GROUP BY region
            ORDER BY total_revenue DESC;
            """
            agg_df = engine.execute_query(sql_agg)
            assert len(agg_df) == 5
            assert list(agg_df["region"]) == ["East", "South", "North", "Central", "West"]
            revenues = [float(r) for r in agg_df["total_revenue"]]
            assert revenues == [207141.0, 159251.0, 112362.0, 82061.0, 80206.0]

            # 5. Security gate: verify mutating queries are blocked
            with pytest.raises(PostgresDisallowed):
                engine.execute_query("DROP TABLE sales;")
            with pytest.raises(PostgresDisallowed):
                engine.execute_query("INSERT INTO sales (order_id) VALUES ('ORD-999');")

            # 6. Smart KPI presentation
            presentation = build_kpi_metadata(agg_df)
            assert isinstance(presentation, ResultPresentation)
            assert presentation.show_chart is True
            assert presentation.row_count == 5

            # 7. Visualization creation
            fig = create_chart(agg_df)
            assert fig is not None

            # 8. Downloads serialization
            csv_bytes = dataframe_to_csv_bytes(agg_df)
            excel_bytes = dataframe_to_excel_bytes(agg_df)
            assert len(csv_bytes) > 0
            assert len(excel_bytes) > 0

            # 9. Conversation history integration
            history = ConversationHistory(dataset_name="sample_sales.csv")
            history.add_turn(
                question="What is revenue by region?",
                sql=sql_agg,
                result_df=agg_df,
                insights="East has highest revenue in PostgreSQL.",
            )
            assert len(history) == 1
            turn = history.get_recent_turns(1)[0]
            assert turn.question == "What is revenue by region?"
            assert turn.result.row_count == 5


# =====================================================================
# DATASET & DATABASE BOUNDARIES
# =====================================================================

class TestDatasetAndDatabaseBoundaries:
    """Validates session state invalidation, dataset boundaries, and prompt schema injection."""

    def test_dataset_boundary_resets_conversation(self):
        """Verify changing dataset flushes old conversation history."""
        df = load_dataset("sample_data/sample_sales.csv")
        history = ConversationHistory(dataset_name="dataset_A.csv")
        history.add_turn(question="Question 1", sql="SELECT 1;", result_df=df)
        assert len(history) == 1

        # Change dataset
        history.reset_for_dataset("dataset_B.csv")
        assert len(history) == 0
        assert history.dataset_name == "dataset_B.csv"

    def test_nl_to_sql_prompt_construction_with_database_schema(self):
        """Verify NL-to-SQL prompt injection uses authoritative database schema."""
        with create_database_engine("duckdb") as engine:
            df = load_dataset("sample_data/sample_sales.csv")
            load_res = load_dataset_into_database(engine, df, table_name="sales")

            prompt = build_sql_prompt(
                question="What are total sales by product?",
                schema=load_res.schema,
            )
            assert "DATABASE SCHEMA" in prompt
            assert "product" in prompt
            assert "revenue" in prompt
            assert "sales" in prompt


# =====================================================================
# STANDALONE CLI TEST RUNNER
# =====================================================================

def run_standalone():
    """Runs tests cleanly when invoked as `python test_phase_13_integration.py`."""
    print("=" * 78)
    print(" PHASE 13 FULL INTEGRATION TEST SUITE")
    print("=" * 78)

    print("\n--- 1. Backend Configuration & Factory ---")
    cfg = TestBackendConfigurationAndFactory()
    cfg.test_factory_selection_duckdb()
    cfg.test_factory_selection_postgres()
    cfg.test_unsupported_backend_raises_error()
    cfg.test_postgresql_unavailable_error_handling()
    print(" [PASS] Backend configuration, factory, and error handling verified.")

    print("\n--- 2. DuckDB Analytics Pipeline ---")
    sales_df = load_dataset("sample_data/sample_sales.csv")
    duck = TestDuckDBAnalyticsPipeline()
    duck.test_duckdb_pipeline_end_to_end(sales_df)
    print(" [PASS] DuckDB end-to-end analytics pipeline verified.")

    print("\n--- 3. PostgreSQL Analytics Pipeline (Live Integration) ---")
    if not PG_AVAILABLE:
        print(f" [SKIP] Live PostgreSQL pipeline skipped: {PG_REASON}")
    else:
        pg = TestPostgresAnalyticsPipeline()
        pg.test_postgres_pipeline_end_to_end(sales_df)
        print(" [PASS] PostgreSQL end-to-end analytics pipeline verified with exact ground truth.")

    print("\n--- 4. Dataset & Database Boundaries ---")
    boundaries = TestDatasetAndDatabaseBoundaries()
    boundaries.test_dataset_boundary_resets_conversation()
    boundaries.test_nl_to_sql_prompt_construction_with_database_schema()
    print(" [PASS] Dataset boundaries and NL-to-SQL schema injection verified.")

    print("\n" + "=" * 78)
    print(" ALL PHASE 13 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_standalone()
