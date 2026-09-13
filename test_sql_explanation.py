"""
Unit Tests for SQL Explanation Module (Phase 11D).

Validates:
1. Empty question handling raises EmptySQLExplanationError
2. Empty SQL handling raises EmptySQLExplanationError
3. Prompt contains current question
4. Prompt contains actual generated SQL
5. Prompt contains grounding constraints (source of truth, no hallucination)
6. Prompt does not contain full DataFrame or table dumps
7. Prompt does not contain query result rows
8. Mock LLM explanation succeeds and returns text
9. LLM failure handled cleanly as LLMExplanationGenerationError
10. Returned explanation is a string
11. SQL is never executed by sql_explanation.py
12. Explanation function does not mutate input arguments
13. Schema parameter is optional and formatted cleanly
14. Custom model handling remains compatible
15. API key parameter override is passed cleanly to SDK caller
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.analytics.sql_explanation import (
    build_sql_explanation_prompt,
    generate_sql_explanation,
    SQLExplanationError,
    EmptySQLExplanationError,
    LLMExplanationGenerationError,
)
from src.analytics.nl_to_sql import LLMGenerationError


class TestSQLExplanation(unittest.TestCase):

    def setUp(self):
        self.sample_question = "What is the total revenue by region?"
        self.sample_sql = (
            "SELECT region, SUM(revenue) AS total_revenue "
            "FROM sales GROUP BY region ORDER BY total_revenue DESC;"
        )
        self.sample_schema = {
            "sales": ["region", "revenue", "order_date", "quantity"]
        }
        self.sample_explanation = (
            "### What the query does\n"
            "Groups sales by region and calculates the total revenue for each region.\n\n"
            "### How it works\n"
            "- Groups records by the region column.\n"
            "- Sums the revenue within each group.\n"
            "- Orders the regions from highest to lowest revenue."
        )

    # -----------------------------------------------------------------
    # 1. Empty question handling
    # -----------------------------------------------------------------
    def test_1_empty_question_handling(self):
        """Empty or whitespace-only question raises EmptySQLExplanationError."""
        with self.assertRaises(EmptySQLExplanationError):
            generate_sql_explanation(
                question="",
                sql=self.sample_sql,
                llm_client=lambda p: "Explanation",
            )

        with self.assertRaises(EmptySQLExplanationError):
            generate_sql_explanation(
                question="   \n\t  ",
                sql=self.sample_sql,
                llm_client=lambda p: "Explanation",
            )

    # -----------------------------------------------------------------
    # 2. Empty SQL handling
    # -----------------------------------------------------------------
    def test_2_empty_sql_handling(self):
        """Empty or whitespace-only SQL raises EmptySQLExplanationError."""
        with self.assertRaises(EmptySQLExplanationError):
            generate_sql_explanation(
                question=self.sample_question,
                sql="",
                llm_client=lambda p: "Explanation",
            )

        with self.assertRaises(EmptySQLExplanationError):
            generate_sql_explanation(
                question=self.sample_question,
                sql="   \n  ",
                llm_client=lambda p: "Explanation",
            )

    # -----------------------------------------------------------------
    # 3. Prompt contains current question
    # -----------------------------------------------------------------
    def test_3_prompt_contains_question(self):
        """The built prompt includes the user's business question."""
        prompt = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
        )
        self.assertIn(self.sample_question, prompt)

    # -----------------------------------------------------------------
    # 4. Prompt contains actual generated SQL
    # -----------------------------------------------------------------
    def test_4_prompt_contains_sql(self):
        """The built prompt includes the exact generated SQL query to explain."""
        prompt = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
        )
        self.assertIn(self.sample_sql, prompt)

    # -----------------------------------------------------------------
    # 5. Prompt contains grounding constraints
    # -----------------------------------------------------------------
    def test_5_prompt_contains_grounding_rules(self):
        """Prompt strictly mandates SQL as source of truth and bans invented operations."""
        prompt = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
        )
        self.assertIn("SOURCE OF TRUTH", prompt)
        self.assertIn("DO NOT INVENT", prompt)
        self.assertIn("What the query does", prompt)
        self.assertIn("How it works", prompt)

    # -----------------------------------------------------------------
    # 6. Prompt does not contain full DataFrame
    # -----------------------------------------------------------------
    def test_6_prompt_does_not_contain_full_dataframe(self):
        """Prompt receives only question, SQL, and optional schema (no DataFrames)."""
        prompt = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
            schema=self.sample_schema,
        )
        self.assertNotIn("DataFrame", prompt)
        self.assertNotIn("pandas", prompt)
        self.assertNotIn("0    East", prompt)

    # -----------------------------------------------------------------
    # 7. Prompt does not contain query result rows
    # -----------------------------------------------------------------
    def test_7_prompt_does_not_contain_result_rows(self):
        """Prompt does not include execution result numbers or rows."""
        prompt = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
        )
        self.assertNotIn("207141", prompt)
        self.assertNotIn("159251", prompt)

    # -----------------------------------------------------------------
    # 8. Mock LLM explanation succeeds
    # -----------------------------------------------------------------
    def test_8_mock_llm_explanation_succeeds(self):
        """A mock LLM client successfully returns the structured explanation."""
        def mock_client(prompt: str) -> str:
            return self.sample_explanation

        result = generate_sql_explanation(
            question=self.sample_question,
            sql=self.sample_sql,
            llm_client=mock_client,
        )
        self.assertEqual(result, self.sample_explanation)
        self.assertIn("### What the query does", result)
        self.assertIn("### How it works", result)

    # -----------------------------------------------------------------
    # 9. LLM failure handled cleanly
    # -----------------------------------------------------------------
    def test_9_llm_failure_handled_cleanly(self):
        """LLM exceptions or empty responses raise LLMExplanationGenerationError."""
        def failing_client(prompt: str) -> str:
            raise RuntimeError("API connection timeout")

        with self.assertRaises(LLMExplanationGenerationError):
            generate_sql_explanation(
                question=self.sample_question,
                sql=self.sample_sql,
                llm_client=failing_client,
            )

        def empty_client(prompt: str) -> str:
            return "   "

        with self.assertRaises(LLMExplanationGenerationError):
            generate_sql_explanation(
                question=self.sample_question,
                sql=self.sample_sql,
                llm_client=empty_client,
            )

    # -----------------------------------------------------------------
    # 10. Returned explanation is text
    # -----------------------------------------------------------------
    def test_10_returned_explanation_is_text(self):
        """generate_sql_explanation always returns a string."""
        res = generate_sql_explanation(
            question=self.sample_question,
            sql=self.sample_sql,
            llm_client=lambda p: self.sample_explanation,
        )
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)

    # -----------------------------------------------------------------
    # 11. SQL is not executed by sql_explanation.py
    # -----------------------------------------------------------------
    def test_11_sql_is_never_executed(self):
        """sql_explanation.py does not connect to DuckDB or execute queries."""
        with patch("duckdb.connect") as mock_duckdb:
            res = generate_sql_explanation(
                question=self.sample_question,
                sql="DROP TABLE sales; -- adversarial attempt",
                llm_client=lambda p: "Explains query without running it",
            )
            mock_duckdb.assert_not_called()
            self.assertEqual(res, "Explains query without running it")

    # -----------------------------------------------------------------
    # 12. Explanation function does not mutate inputs
    # -----------------------------------------------------------------
    def test_12_immutability_of_inputs(self):
        """Inputs (question string, sql string, schema dict) remain unchanged."""
        orig_question = str(self.sample_question)
        orig_sql = str(self.sample_sql)
        orig_schema = dict(self.sample_schema)

        _ = generate_sql_explanation(
            question=self.sample_question,
            sql=self.sample_sql,
            schema=self.sample_schema,
            llm_client=lambda p: self.sample_explanation,
        )

        self.assertEqual(self.sample_question, orig_question)
        self.assertEqual(self.sample_sql, orig_sql)
        self.assertEqual(self.sample_schema, orig_schema)

    # -----------------------------------------------------------------
    # 13. Schema is optional
    # -----------------------------------------------------------------
    def test_13_schema_is_optional(self):
        """Functions execute successfully whether schema is provided or None."""
        prompt_with = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
            schema=self.sample_schema,
        )
        prompt_without = build_sql_explanation_prompt(
            question=self.sample_question,
            sql=self.sample_sql,
            schema=None,
        )

        self.assertIn("DATABASE SCHEMA", prompt_with)
        self.assertNotIn("DATABASE SCHEMA", prompt_without)

        res = generate_sql_explanation(
            question=self.sample_question,
            sql=self.sample_sql,
            schema=None,
            llm_client=lambda p: self.sample_explanation,
        )
        self.assertEqual(res, self.sample_explanation)

    # -----------------------------------------------------------------
    # 14. Custom model handling remains compatible
    # -----------------------------------------------------------------
    def test_14_custom_model_handling(self):
        """Custom model name parameter is accepted without error."""
        captured_model = []

        class MockModelClient:
            class models:
                @staticmethod
                def generate_content(model, contents):
                    captured_model.append(model)
                    mock_res = MagicMock()
                    mock_res.text = "Model explanation"
                    return mock_res

        res = generate_sql_explanation(
            question=self.sample_question,
            sql=self.sample_sql,
            llm_client=MockModelClient(),
            model="gemini-custom-model",
        )
        self.assertEqual(res, "Model explanation")
        self.assertEqual(captured_model, ["gemini-custom-model"])

    # -----------------------------------------------------------------
    # 15. API key handling remains compatible
    # -----------------------------------------------------------------
    def test_15_api_key_handling(self):
        """API key override parameter is passed down to _call_gemini_sdk."""
        with patch("src.analytics.sql_explanation._call_gemini_sdk") as mock_sdk:
            mock_sdk.return_value = self.sample_explanation

            res = generate_sql_explanation(
                question=self.sample_question,
                sql=self.sample_sql,
                api_key="custom_test_api_key_123",
                model="gemini-2.5-flash",
            )

            mock_sdk.assert_called_once()
            _, kwargs = mock_sdk.call_args
            self.assertEqual(kwargs.get("api_key"), "custom_test_api_key_123")
            self.assertEqual(kwargs.get("model"), "gemini-2.5-flash")
            self.assertEqual(res, self.sample_explanation)


if __name__ == "__main__":
    unittest.main()
