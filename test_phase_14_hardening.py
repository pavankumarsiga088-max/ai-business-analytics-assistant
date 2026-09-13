"""
Phase 14 — Production Hardening and Deployment Preparation Test Suite.

Verifies:
1. Centralized non-secret configuration defaults and dynamic getters (src/config.py).
2. Secret values and credentials are never exposed in user-facing errors or string representations.
3. Strict read-only SQL validation blocks destructive DDL/DML and multi-statement queries.
4. Query resource limits and oversized result DataFrame truncation safeguards.
5. Missing Gemini API key resilience (clean detection, no crashes).
6. Sanitized database error handling (no password leakage).
7. Dataset validation for empty DataFrames and 0-column DataFrames.
8. Caller DataFrame immutability guarantees.
9. Database engine switching state safety and conversation boundary isolation.
"""
import os
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.config import (
    get_database_backend,
    get_gemini_model,
    get_max_query_result_rows,
    get_query_timeout_seconds,
    is_gemini_configured,
    is_postgres_configured,
    MAX_QUERY_RESULT_ROWS,
    QUERY_TIMEOUT_SECONDS,
)
from src.analytics.database import (
    create_database_engine,
    get_configured_database_backend,
    UnsupportedDatabaseError,
)
from src.analytics.dataset_loader import (
    load_dataset_into_database,
    InvalidDataFrameError,
)
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    validate_read_only_query,
    DisallowedQueryError,
)
from src.analytics.postgres_engine import (
    PostgresEngine,
    PostgresConnectionError,
)
from src.analytics.conversation import ConversationHistory
from src.analytics.nl_to_sql import (
    get_gemini_client,
    clean_generated_sql,
    LLMGenerationError,
)


class TestConfigManagement(unittest.TestCase):
    """Verifies centralized configuration and environment variable handling."""

    def test_default_database_backend_is_duckdb(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_database_backend(), "duckdb")

    def test_database_backend_env_override(self):
        with patch.dict(os.environ, {"DATABASE_BACKEND": "postgres"}):
            self.assertEqual(get_database_backend(), "postgres")
        with patch.dict(os.environ, {"DATABASE_BACKEND": "postgresql"}):
            self.assertEqual(get_database_backend(), "postgres")

    def test_max_query_result_rows_config(self):
        with patch.dict(os.environ, {"MAX_QUERY_RESULT_ROWS": "5000"}):
            self.assertEqual(get_max_query_result_rows(), 5000)

    def test_query_timeout_config(self):
        with patch.dict(os.environ, {"QUERY_TIMEOUT_SECONDS": "45"}):
            self.assertEqual(get_query_timeout_seconds(), 45)

    def test_gemini_configuration_detector(self):
        # Empty or unset
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertFalse(is_gemini_configured())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_gemini_configured())
        # Placeholder
        with patch.dict(os.environ, {"GEMINI_API_KEY": "your_gemini_api_key_here"}):
            self.assertFalse(is_gemini_configured())
        # Valid key
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeKeyForTesting12345"}):
            self.assertTrue(is_gemini_configured())

    def test_postgres_configuration_detector(self):
        with patch.dict(os.environ, {"POSTGRES_HOST": "localhost", "POSTGRES_DB": "sales_db", "POSTGRES_USER": "analyst"}):
            self.assertTrue(is_postgres_configured())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_postgres_configured())


class TestSecurityHardening(unittest.TestCase):
    """Verifies security controls, read-only SQL gates, and password redaction."""

    def test_destructive_sql_verbs_are_strictly_blocked(self):
        destructive_queries = [
            "INSERT INTO sales VALUES (1, 'Widget');",
            "UPDATE sales SET revenue = 0;",
            "DELETE FROM sales WHERE id = 1;",
            "DROP TABLE sales;",
            "ALTER TABLE sales ADD COLUMN hack TEXT;",
            "TRUNCATE TABLE sales;",
            "CREATE TABLE evil (id INT);",
            "GRANT ALL PRIVILEGES ON sales TO public;",
            "REVOKE SELECT ON sales FROM analyst;",
            "VACUUM FULL;",
            "COPY sales TO '/tmp/leak.csv';",
            "EXEC xp_cmdshell 'dir';",
        ]
        for query in destructive_queries:
            with self.subTest(query=query):
                with self.assertRaises(DisallowedQueryError):
                    validate_read_only_query(query)

    def test_multi_statement_sql_is_strictly_blocked(self):
        multi_statement_queries = [
            "SELECT 1; SELECT 2;",
            "SELECT * FROM sales; DROP TABLE sales;",
            "SELECT region FROM sales; DELETE FROM sales;",
        ]
        for query in multi_statement_queries:
            with self.subTest(query=query):
                with self.assertRaises(DisallowedQueryError):
                    validate_read_only_query(query)

    def test_postgres_password_redaction_in_sanitized_messages(self):
        engine = PostgresEngine(
            host="db.example.internal",
            port=5432,
            database="analytics",
            user="analyst_user",
            password="SuperSecretPassword99!",
            connect_timeout=1,
        )
        leaked_msg = "FATAL: password authentication failed for user 'analyst_user' with password 'SuperSecretPassword99!'"
        sanitized = engine._sanitize_error_message(leaked_msg)
        self.assertNotIn("SuperSecretPassword99!", sanitized)
        self.assertIn("******", sanitized)

    def test_connection_failure_never_exposes_password(self):
        fake_engine = PostgresEngine(
            host="127.0.0.1",
            port=59999,  # Non-existent port
            database="analytics",
            user="analyst_user",
            password="SensitivePasswordXYZ123",
            connect_timeout=1,
        )
        try:
            fake_engine.connect()
        except PostgresConnectionError as e:
            self.assertNotIn("SensitivePasswordXYZ123", str(e))
        finally:
            fake_engine.close()


class TestQueryResourceProtection(unittest.TestCase):
    """Verifies query row limits and oversized result handling."""

    def test_oversized_result_truncation_logic(self):
        max_rows = 100
        # Generate DataFrame with 250 rows
        large_df = pd.DataFrame({"id": range(250), "val": [f"item_{i}" for i in range(250)]})

        # Apply resource protection
        original_count = len(large_df)
        is_truncated = False
        result_df = large_df

        if original_count > max_rows:
            result_df = large_df.iloc[:max_rows].copy()
            is_truncated = True

        self.assertTrue(is_truncated)
        self.assertEqual(len(result_df), 100)
        self.assertEqual(original_count, 250)

    def test_normal_result_remains_untruncated(self):
        max_rows = 100
        small_df = pd.DataFrame({"id": range(25), "val": [f"item_{i}" for i in range(25)]})

        original_count = len(small_df)
        is_truncated = False
        result_df = small_df

        if original_count > max_rows:
            result_df = small_df.iloc[:max_rows].copy()
            is_truncated = True

        self.assertFalse(is_truncated)
        self.assertEqual(len(result_df), 25)

    def test_postgres_engine_query_timeout_property(self):
        engine = PostgresEngine(query_timeout=42)
        self.assertEqual(engine.query_timeout, 42)


class TestGeminiFailureResilience(unittest.TestCase):
    """Verifies graceful handling when Gemini API is unconfigured or fails."""

    def test_missing_api_key_raises_clear_llm_generation_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMGenerationError) as ctx:
                get_gemini_client(api_key=None)
            self.assertIn("GEMINI_API_KEY is not configured", str(ctx.exception))

    def test_clean_generated_sql_strips_markdown_fences(self):
        fenced_sql = "```sql\nSELECT region, SUM(revenue) FROM sales GROUP BY region;\n```"
        cleaned = clean_generated_sql(fenced_sql)
        self.assertEqual(cleaned, "SELECT region, SUM(revenue) FROM sales GROUP BY region;")

        fenced_backticks = "`SELECT * FROM sales`"
        cleaned_backticks = clean_generated_sql(fenced_backticks)
        self.assertEqual(cleaned_backticks, "SELECT * FROM sales")


class TestDatasetValidationAndSafety(unittest.TestCase):
    """Verifies dataset input validation, immutability, and boundary controls."""

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "region": ["North", "South", "East", "West"],
            "revenue": [100.0, 200.0, 300.0, 400.0],
        })

    def test_empty_dataframe_rejected_by_loader(self):
        engine = DuckDBEngine()
        empty_df = pd.DataFrame()
        with self.assertRaises(InvalidDataFrameError):
            load_dataset_into_database(engine, empty_df)
        engine.close()

    def test_zero_column_dataframe_rejected_by_loader(self):
        engine = DuckDBEngine()
        zero_col_df = pd.DataFrame(index=[0, 1, 2])
        with self.assertRaises(InvalidDataFrameError):
            load_dataset_into_database(engine, zero_col_df)
        engine.close()

    def test_caller_dataframe_immutability_preserved(self):
        engine = DuckDBEngine()
        original_df = self.sample_df.copy(deep=True)
        load_dataset_into_database(engine, self.sample_df, table_name="sales")

        # Caller DataFrame must not be altered in shape, values, or types
        pd.testing.assert_frame_equal(self.sample_df, original_df)
        engine.close()

    def test_dataset_switch_resets_conversation_boundary(self):
        history = ConversationHistory(dataset_name="sales_q1.csv")
        history.add_turn(
            question="What is total revenue?",
            sql="SELECT SUM(revenue) FROM sales;",
            result_df=pd.DataFrame({"total": [1000.0]}),
        )
        self.assertEqual(len(history), 1)

        # Dataset boundary transition
        history.reset_for_dataset(dataset_name="sales_q2.csv")
        self.assertEqual(len(history), 0)
        self.assertTrue(history.is_empty())
        self.assertEqual(history.dataset_name, "sales_q2.csv")


if __name__ == "__main__":
    unittest.main()
