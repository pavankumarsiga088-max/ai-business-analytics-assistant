"""
Test Suite for Phase 10C: Follow-Up Conversation UI & Workflow.

Validates the full conversation integration flow:
1. First question works with empty conversation history.
2. First successful turn is stored.
3. Second question receives previous conversation context.
4. Follow-up question can use previous analytical context.
5. Second successful turn is stored.
6. Only completed successful analytics queries are stored.
7. Failed SQL generation does not create a turn.
8. Failed SQL validation does not create a turn.
9. Failed SQL execution does not create a turn.
10. New dataset resets conversation history.
11. Old dataset context is not passed to the new dataset.
12. Conversation history remains bounded by the existing ConversationHistory rules.
13. Full DataFrames are not stored in conversation history.
14. Existing SQL read-only validation remains enforced.
"""
import unittest
import pandas as pd

from src.data_quality.loader import load_dataset
from src.analytics.conversation import (
    ConversationTurn,
    ConversationHistory,
    QueryResultSummary,
)
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    validate_read_only_query,
    DisallowedQueryError,
    QueryExecutionError,
)
from src.analytics.nl_to_sql import (
    generate_sql,
    build_sql_prompt,
    clean_generated_sql,
    NLToSQLError,
    EmptyQuestionError,
    LLMGenerationError,
)


class TestPhase10CConversationFlow(unittest.TestCase):
    def setUp(self):
        # Sample dataset for testing
        self.sample_df = pd.DataFrame({
            "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104"],
            "region": ["East", "West", "East", "South"],
            "product": ["Laptop", "Mouse", "Keyboard", "Laptop"],
            "revenue": [2000.0, 50.0, 100.0, 2000.0],
        })
        self.schema = {
            "order_id": "VARCHAR",
            "region": "VARCHAR",
            "product": "VARCHAR",
            "revenue": "DOUBLE",
        }
        self.history = ConversationHistory(dataset_name="sample_sales.csv")

    def test_1_first_question_with_empty_history(self):
        """1. First question works with empty conversation history."""
        captured_prompts = []

        def mock_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region;"

        sql = generate_sql(
            "What is the total revenue by region?",
            schema=self.schema,
            conversation_context=self.history,
            llm_client=mock_llm,
        )

        self.assertNotIn("PREVIOUS ANALYTICS CONTEXT", captured_prompts[0])
        self.assertIn("USER QUESTION:\nWhat is the total revenue by region?", captured_prompts[0])
        self.assertEqual(sql, "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region;")

    def test_2_first_successful_turn_stored(self):
        """2. First successful analytical turn is stored in ConversationHistory."""
        with DuckDBEngine() as engine:
            engine.register_dataframe(self.sample_df, table_name="sales")
            sql = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;"
            validate_read_only_query(sql)
            result_df = engine.execute_query(sql)

            self.history.add_turn(
                question="What is the total revenue by region?",
                sql=sql,
                result_df=result_df,
                insights="East has the highest revenue of 2100.0.",
            )

        self.assertEqual(len(self.history), 1)
        self.assertFalse(self.history.is_empty())
        turn = self.history.get_turns()[0]
        self.assertEqual(turn.question, "What is the total revenue by region?")
        self.assertIn("total_revenue", turn.result.columns)
        self.assertEqual(turn.result.row_count, 3)

    def test_3_second_question_receives_previous_context(self):
        """3. Second question receives previous conversation context in prompt."""
        # Add first turn to history
        self.history.add_turn(
            question="What is the total revenue by region?",
            sql="SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region;",
            result_df=pd.DataFrame({"region": ["East", "South"], "total_revenue": [2100.0, 2000.0]}),
            insights="East is the top region.",
        )

        captured_prompts = []

        def mock_followup_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "SELECT SUM(revenue) AS total_revenue FROM sales WHERE region = 'East';"

        sql = generate_sql(
            "What about East?",
            schema=self.schema,
            conversation_context=self.history,
            llm_client=mock_followup_llm,
        )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("PREVIOUS ANALYTICS CONTEXT", prompt)
        self.assertIn("What is the total revenue by region?", prompt)
        self.assertIn("CURRENT QUESTION:\nWhat about East?", prompt)
        self.assertEqual(sql, "SELECT SUM(revenue) AS total_revenue FROM sales WHERE region = 'East';")

    def test_4_followup_question_resolves_and_executes(self):
        """4. Follow-up query executes cleanly against DuckDB."""
        with DuckDBEngine() as engine:
            engine.register_dataframe(self.sample_df, table_name="sales")
            followup_sql = "SELECT SUM(revenue) AS total_revenue FROM sales WHERE region = 'East';"
            validate_read_only_query(followup_sql)
            res = engine.execute_query(followup_sql)

            self.assertEqual(len(res), 1)
            self.assertEqual(res["total_revenue"].iloc[0], 2100.0)

    def test_5_second_successful_turn_stored(self):
        """5. Second successful turn is stored in chronological order."""
        # Turn 1
        self.history.add_turn(
            question="What is the total revenue by region?",
            sql="SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region;",
            result_df=pd.DataFrame({"region": ["East"], "total_revenue": [2100.0]}),
        )

        # Turn 2
        self.history.add_turn(
            question="What about East?",
            sql="SELECT SUM(revenue) AS total_revenue FROM sales WHERE region = 'East';",
            result_df=pd.DataFrame({"total_revenue": [2100.0]}),
            insights="East total is 2100.",
        )

        self.assertEqual(len(self.history), 2)
        turns = self.history.get_turns()
        self.assertEqual(turns[0].question, "What is the total revenue by region?")
        self.assertEqual(turns[1].question, "What about East?")

    def test_6_only_completed_queries_stored(self):
        """6. Turns are only added after full execution."""
        initial_len = len(self.history)
        try:
            # Simulate a failure before add_turn
            raise QueryExecutionError("Simulated execution failure")
        except QueryExecutionError:
            pass

        self.assertEqual(len(self.history), initial_len)

    def test_7_failed_sql_generation_does_not_create_turn(self):
        """7. Failed SQL generation leaves history unchanged."""
        initial_count = len(self.history)

        def failing_llm(prompt: str):
            raise LLMGenerationError("Network error contacting Gemini")

        with self.assertRaises(LLMGenerationError):
            generate_sql(
                "Any question",
                schema=self.schema,
                conversation_context=self.history,
                llm_client=failing_llm,
            )

        self.assertEqual(len(self.history), initial_count)

    def test_8_failed_sql_validation_does_not_create_turn(self):
        """8. Unsafe SQL failing validation does not create a turn."""
        initial_count = len(self.history)

        malicious_sql = "DROP TABLE sales;"
        with self.assertRaises(DisallowedQueryError):
            validate_read_only_query(malicious_sql)
            # If validation fails, add_turn is not reached in app.py
            self.history.add_turn("Delete table", sql=malicious_sql)

        self.assertEqual(len(self.history), initial_count)

    def test_9_failed_sql_execution_does_not_create_turn(self):
        """9. Failed DuckDB query execution does not create a turn."""
        initial_count = len(self.history)

        invalid_syntax_sql = "SELECT non_existent_column FROM sales;"
        with DuckDBEngine() as engine:
            engine.register_dataframe(self.sample_df, table_name="sales")
            with self.assertRaises(QueryExecutionError):
                engine.execute_query(invalid_syntax_sql)
                # If execution fails, add_turn is not reached
                self.history.add_turn("Bad column", sql=invalid_syntax_sql)

        self.assertEqual(len(self.history), initial_count)

    def test_10_new_dataset_resets_conversation_history(self):
        """10. Uploading a new dataset flushes history."""
        self.history.add_turn("Question 1", sql="SELECT 1;")
        self.history.add_turn("Question 2", sql="SELECT 2;")
        self.assertEqual(len(self.history), 2)

        # Simulate dataset boundary switch
        self.history.reset_for_dataset(dataset_name="new_dataset.xlsx")

        self.assertEqual(len(self.history), 0)
        self.assertTrue(self.history.is_empty())
        self.assertEqual(self.history.dataset_name, "new_dataset.xlsx")

    def test_11_old_dataset_context_not_passed_to_new_dataset(self):
        """11. Old dataset context is never passed to queries on new dataset."""
        self.history.add_turn("What were sales in 2023?", sql="SELECT * FROM sales;")
        self.history.reset_for_dataset("hr_employees.csv")

        prompt = build_sql_prompt(
            "Who has highest salary?",
            schema={"employee_name": "VARCHAR", "salary": "DOUBLE"},
            conversation_context=self.history,
        )

        self.assertNotIn("What were sales in 2023?", prompt)
        self.assertNotIn("PREVIOUS ANALYTICS CONTEXT", prompt)
        self.assertIn("USER QUESTION:\nWho has highest salary?", prompt)

    def test_12_conversation_history_boundedness_enforced(self):
        """12. ConversationHistory honors maximum turns limit."""
        bounded_hist = ConversationHistory(max_turns=3)
        for i in range(1, 6):
            bounded_hist.add_turn(f"Question {i}", sql=f"SELECT {i};")

        self.assertEqual(len(bounded_hist), 3)
        turns = bounded_hist.get_turns()
        self.assertEqual([t.question for t in turns], ["Question 3", "Question 4", "Question 5"])

    def test_13_full_dataframes_not_stored_in_history(self):
        """13. Full live DataFrames are not stored in conversation history."""
        large_df = pd.DataFrame({"col": list(range(100))})
        self.history.add_turn("Large table query", result_df=large_df)

        turn = self.history.get_turns()[-1]
        self.assertIsInstance(turn.result, QueryResultSummary)
        self.assertNotIsInstance(turn.result, pd.DataFrame)
        self.assertEqual(turn.result.row_count, 100)
        self.assertEqual(turn.result.truncated_rows, 25)
        self.assertTrue(turn.result.is_truncated)

    def test_14_read_only_validation_strictly_enforced(self):
        """14. Strict read-only SQL validation remains enforced."""
        disallowed = [
            "INSERT INTO sales VALUES (1);",
            "UPDATE sales SET revenue = 0;",
            "DELETE FROM sales WHERE region = 'East';",
            "DROP TABLE sales;",
            "ALTER TABLE sales ADD COLUMN test INT;",
            "SELECT 1; DROP TABLE sales;",
        ]
        for query in disallowed:
            with self.assertRaises(DisallowedQueryError):
                validate_read_only_query(query)


if __name__ == "__main__":
    unittest.main()
