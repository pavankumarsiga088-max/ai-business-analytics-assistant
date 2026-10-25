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
    # 15. Deterministic execution without LLM or API call
    # -----------------------------------------------------------------
    def test_15_deterministic_execution_without_llm(self):
        """Default generate_sql_explanation produces deterministic result locally without Gemini API."""
        res = generate_sql_explanation(
            question=self.sample_question,
            sql=self.sample_sql,
            api_key="custom_test_api_key_123",
            model="gemini-2.5-flash",
        )
        self.assertIsInstance(res, str)
        self.assertIn("### What the query does", res)
        self.assertIn("### How it works", res)
        self.assertIn("total revenue", res.lower())
        self.assertIn("region", res.lower())

    # -----------------------------------------------------------------
    # 16. Aggregate query explanation
    # -----------------------------------------------------------------
    def test_16_aggregate_query_explanation(self):
        """Aggregate query explains mathematical aggregation with SUM."""
        sql = "SELECT SUM(revenue) AS total_revenue FROM sales;"
        res = generate_sql_explanation("What is total revenue?", sql)
        self.assertIn("### What the query does", res)
        self.assertIn("### How it works", res)
        self.assertIn("total revenue", res.lower())
        self.assertIn("sum", res.lower())

    # -----------------------------------------------------------------
    # 17. GROUP BY explanation
    # -----------------------------------------------------------------
    def test_17_group_by_explanation(self):
        """Query with GROUP BY explains grouping by specified column."""
        sql = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region;"
        res = generate_sql_explanation("Revenue by region?", sql)
        self.assertIn("Groups results by `region`.", res)
        self.assertIn("region", res.lower())

    # -----------------------------------------------------------------
    # 18. WHERE condition explanation
    # -----------------------------------------------------------------
    def test_18_where_condition_explanation(self):
        """Query with WHERE explains filtering condition."""
        sql = "SELECT * FROM sales WHERE region = 'East';"
        res = generate_sql_explanation("East sales?", sql)
        self.assertIn("Filters records where `region = 'East'`.", res)

    # -----------------------------------------------------------------
    # 19. ORDER BY explanation
    # -----------------------------------------------------------------
    def test_19_order_by_explanation(self):
        """Query with ORDER BY explains sorting and direction."""
        sql = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;"
        res = generate_sql_explanation("Sorted revenue by region?", sql)
        self.assertIn("Orders results by `total_revenue` in descending order", res)

    # -----------------------------------------------------------------
    # 20. LIMIT explanation
    # -----------------------------------------------------------------
    def test_20_limit_explanation(self):
        """Query with LIMIT explains row count cap."""
        sql = "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC LIMIT 5;"
        res = generate_sql_explanation("Top 5 products?", sql)
        self.assertIn("Limits the output to the top `5` records.", res)

    # -----------------------------------------------------------------
    # 21. Multiple aggregates explanation
    # -----------------------------------------------------------------
    def test_21_multiple_aggregates_explanation(self):
        """Query with multiple aggregates explains SUM, AVG, and COUNT."""
        sql = "SELECT region, SUM(revenue) AS total_rev, AVG(unit_price) AS avg_price, COUNT(*) AS total_orders FROM sales GROUP BY region;"
        res = generate_sql_explanation("Summary by region?", sql)
        self.assertIn("SUM(revenue)", res)
        self.assertIn("AVG(unit_price)", res)
        self.assertIn("COUNT(*)", res)

    # -----------------------------------------------------------------
    # 22. Scalar query explanation
    # -----------------------------------------------------------------
    def test_22_scalar_query_explanation(self):
        """Scalar aggregate query explains calculation across all records."""
        sql = "SELECT AVG(revenue) AS average_order_value FROM sales;"
        res = generate_sql_explanation("Average order value?", sql)
        self.assertIn("across all records", res.lower())
        self.assertIn("average", res.lower())

    # -----------------------------------------------------------------
    # 23. Grouped query explanation
    # -----------------------------------------------------------------
    def test_23_grouped_query_explanation(self):
        """Grouped query explains grouping and aggregation for domain entities."""
        sql = "SELECT crop, SUM(profit) AS total_profit FROM agriculture GROUP BY crop ORDER BY total_profit DESC;"
        res = generate_sql_explanation("Which crop generates the highest profit?", sql)
        self.assertIn("crop", res.lower())
        self.assertIn("profit", res.lower())
        self.assertIn("descending", res.lower())

    # -----------------------------------------------------------------
    # 24. Follow-up query explanation
    # -----------------------------------------------------------------
    def test_24_follow_up_query_explanation(self):
        """Follow-up query with IN filter explains targeted subset comparison."""
        sql = "SELECT region, SUM(revenue) AS total_revenue FROM sales WHERE region IN ('East', 'South') GROUP BY region ORDER BY total_revenue DESC;"
        res = generate_sql_explanation("Compare East and South", sql)
        self.assertIn("region IN ('East', 'South')", res)
        self.assertIn("Groups results by `region`.", res)

    # -----------------------------------------------------------------
    # 25. No database execution during explanation
    # -----------------------------------------------------------------
    def test_25_no_database_execution_in_explanation(self):
        """Explanation module never opens database connections or runs queries."""
        with patch("duckdb.connect") as mock_duck:
            res = generate_sql_explanation(
                "Adversarial question",
                "SELECT * FROM sales; DROP TABLE sales;",
            )
            mock_duck.assert_not_called()
            self.assertIn("### What the query does", res)


if __name__ == "__main__":
    unittest.main()
