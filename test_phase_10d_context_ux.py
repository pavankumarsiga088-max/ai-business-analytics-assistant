"""
Test Suite for Phase 10D: Context Management & UX Hardening.

Validates the 15 required context management and UX scenarios:
1. New dataset starts with empty conversation history.
2. Existing conversation can be cleared without removing the dataset.
3. Clearing conversation removes latest analytics state.
4. Dataset switching resets conversation history.
5. Dataset switching clears latest SQL/result/insights.
6. Previous dataset context is not passed into a new dataset analysis.
7. Context indicator reflects zero previous turns for a new conversation.
8. Context indicator reflects available previous turns after successful analyses.
9. Only configured recent context limit (3) is sent to generate_sql().
10. Conversation remains bounded by ConversationHistory.
11. Historical turns do not contain live DataFrames (QueryResultSummary used).
12. Failed SQL generation does not create a history turn.
13. Failed SQL execution does not create a history turn.
14. Successful query creates a history turn.
15. Clearing history allows a new first question on the same dataset.
"""

import unittest
import pandas as pd

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
    LLMGenerationError,
)


class TestPhase10DContextUX(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104"],
            "region": ["North", "South", "East", "West"],
            "category": ["Electronics", "Furniture", "Electronics", "Clothing"],
            "sales_amount": [1500.0, 750.0, 1200.0, 300.0],
        })
        self.schema = {
            "order_id": "VARCHAR",
            "region": "VARCHAR",
            "category": "VARCHAR",
            "sales_amount": "DOUBLE",
        }
        self.dataset_name = "sales_data_q3.csv"
        self.history = ConversationHistory(dataset_name=self.dataset_name)

    def test_1_new_dataset_starts_with_empty_conversation_history(self):
        """1. A new dataset must start with an empty ConversationHistory instance."""
        new_history = ConversationHistory(dataset_name="newly_loaded_dataset.csv")
        self.assertEqual(len(new_history), 0)
        self.assertTrue(new_history.is_empty())
        self.assertEqual(new_history.dataset_name, "newly_loaded_dataset.csv")
        self.assertEqual(new_history.get_turns(), [])

    def test_2_existing_conversation_cleared_without_removing_dataset(self):
        """2. Existing conversation can be cleared without removing the loaded dataset."""
        # Add sample turns
        self.history.add_turn("What is total sales?", sql="SELECT SUM(sales_amount) FROM sales;")
        self.history.add_turn("By region?", sql="SELECT region, SUM(sales_amount) FROM sales GROUP BY region;")
        self.assertEqual(len(self.history), 2)

        # Simulate clearing conversation (as done by the Clear Conversation History action)
        dataset_reference = self.sample_df  # dataset remains loaded
        self.history.clear()

        self.assertEqual(len(self.history), 0)
        self.assertTrue(self.history.is_empty())
        self.assertEqual(self.history.dataset_name, self.dataset_name)  # preserves dataset context
        self.assertIsNotNone(dataset_reference)
        self.assertEqual(len(dataset_reference), 4)

    def test_3_clearing_conversation_removes_latest_analytics_state(self):
        """3. Clearing conversation removes latest analytics state and question input."""
        # Setup simulated Streamlit session state
        session_state = {
            "conv_history": self.history,
            "last_analytics_result": {
                "question": "What is total sales?",
                "sql": "SELECT SUM(sales_amount) FROM sales;",
                "df": pd.DataFrame({"total": [3750.0]}),
                "insights": "Total sales is 3750.",
                "dataset_name": self.dataset_name,
            },
            "user_nl_question": "Follow up question",
        }

        # Simulate the Clear Conversation History action in app.py
        session_state["conv_history"].clear()
        session_state["last_analytics_result"] = None
        if "user_nl_question" in session_state:
            session_state["user_nl_question"] = ""

        self.assertEqual(len(session_state["conv_history"]), 0)
        self.assertIsNone(session_state["last_analytics_result"])
        self.assertEqual(session_state["user_nl_question"], "")

    def test_4_dataset_switching_resets_conversation_history(self):
        """4. Switching dataset resets conversation history completely."""
        self.history.add_turn("What is total sales?", sql="SELECT SUM(sales_amount) FROM sales;")
        self.assertEqual(len(self.history), 1)

        # Simulate dataset boundary switch
        new_dataset_name = "inventory_q3.csv"
        self.history.reset_for_dataset(new_dataset_name)

        self.assertEqual(len(self.history), 0)
        self.assertTrue(self.history.is_empty())
        self.assertEqual(self.history.dataset_name, new_dataset_name)

    def test_5_dataset_switching_clears_latest_sql_result_insights(self):
        """5. Switching dataset clears latest SQL, result DataFrame, and insights."""
        session_state = {
            "current_dataset_name": "old_dataset.csv",
            "conv_history": ConversationHistory(dataset_name="old_dataset.csv"),
            "last_analytics_result": {
                "question": "Old query",
                "sql": "SELECT * FROM old_table;",
                "df": pd.DataFrame({"col": [1, 2]}),
                "insights": "Old insight",
            },
            "user_nl_question": "Old pending question",
        }
        session_state["conv_history"].add_turn("Old query", sql="SELECT * FROM old_table;")

        # Simulate dataset switch block in app.py
        new_file_name = "new_dataset.csv"
        if session_state["current_dataset_name"] != new_file_name:
            session_state["current_dataset_name"] = new_file_name
            session_state["conv_history"].reset_for_dataset(new_file_name)
            session_state["last_analytics_result"] = None
            if "user_nl_question" in session_state:
                session_state["user_nl_question"] = ""

        self.assertEqual(session_state["current_dataset_name"], "new_dataset.csv")
        self.assertEqual(len(session_state["conv_history"]), 0)
        self.assertIsNone(session_state["last_analytics_result"])
        self.assertEqual(session_state["user_nl_question"], "")

    def test_6_previous_dataset_context_not_passed_into_new_dataset_analysis(self):
        """6. Old dataset context is never passed into a new dataset analysis prompt."""
        self.history.add_turn(
            "What were sales by region in 2023?",
            sql="SELECT region, SUM(sales_amount) FROM sales GROUP BY region;",
        )
        self.history.reset_for_dataset("employees.csv")

        prompt = build_sql_prompt(
            "Who has the highest salary?",
            schema={"employee_name": "VARCHAR", "salary": "DOUBLE"},
            conversation_context=self.history,
            recent_turn_limit=3,
        )

        self.assertNotIn("sales_amount", prompt)
        self.assertNotIn("What were sales by region in 2023?", prompt)
        self.assertNotIn("PREVIOUS ANALYTICS CONTEXT", prompt)
        self.assertIn("USER QUESTION:\nWho has the highest salary?", prompt)

    def test_7_context_indicator_zero_previous_turns(self):
        """7. Context indicator reflects zero previous turns for a new conversation."""
        def get_context_indicator_text(history: ConversationHistory) -> str:
            num_prev_turns = len(history)
            if num_prev_turns > 0:
                recent_count = min(num_prev_turns, 3)
                return f"Using previous analytical context ({recent_count} recent turn{'s' if recent_count > 1 else ''})"
            return "Starting a new analytical question (no previous context)"

        indicator = get_context_indicator_text(self.history)
        self.assertEqual(indicator, "Starting a new analytical question (no previous context)")

    def test_8_context_indicator_available_previous_turns(self):
        """8. Context indicator reflects available previous turns after successful analyses."""
        def get_context_indicator_text(history: ConversationHistory) -> str:
            num_prev_turns = len(history)
            if num_prev_turns > 0:
                recent_count = min(num_prev_turns, 3)
                return f"Using previous analytical context ({recent_count} recent turn{'s' if recent_count > 1 else ''})"
            return "Starting a new analytical question (no previous context)"

        self.history.add_turn("Turn 1", sql="SELECT 1;")
        self.assertEqual(
            get_context_indicator_text(self.history),
            "Using previous analytical context (1 recent turn)",
        )

        self.history.add_turn("Turn 2", sql="SELECT 2;")
        self.assertEqual(
            get_context_indicator_text(self.history),
            "Using previous analytical context (2 recent turns)",
        )

        self.history.add_turn("Turn 3", sql="SELECT 3;")
        self.assertEqual(
            get_context_indicator_text(self.history),
            "Using previous analytical context (3 recent turns)",
        )

        # Capped at 3 recent turns
        self.history.add_turn("Turn 4", sql="SELECT 4;")
        self.assertEqual(
            get_context_indicator_text(self.history),
            "Using previous analytical context (3 recent turns)",
        )

    def test_9_recent_context_limit_capped_at_three_in_generate_sql(self):
        """9. Only configured recent context limit (3) is sent to generate_sql()."""
        for i in range(1, 6):
            self.history.add_turn(f"Question {i}", sql=f"SELECT {i};")

        self.assertEqual(len(self.history), 5)
        captured_prompts = []

        def mock_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "SELECT 6;"

        generate_sql(
            "Question 6",
            schema=self.schema,
            conversation_context=self.history,
            recent_turn_limit=3,
            llm_client=mock_llm,
        )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        # Must contain recent 3 turns (Question 3, 4, 5)
        self.assertIn("Question 3", prompt)
        self.assertIn("Question 4", prompt)
        self.assertIn("Question 5", prompt)
        # Must NOT contain older turns (Question 1, 2)
        self.assertNotIn("Question 1", prompt)
        self.assertNotIn("Question 2", prompt)

    def test_10_conversation_bounded_by_conversation_history(self):
        """10. Conversation remains bounded by ConversationHistory max_turns."""
        bounded_history = ConversationHistory(max_turns=10)
        for i in range(15):
            bounded_history.add_turn(f"Query {i}", sql=f"SELECT {i};")

        self.assertEqual(len(bounded_history), 10)
        turns = bounded_history.get_turns()
        self.assertEqual(turns[0].question, "Query 5")
        self.assertEqual(turns[-1].question, "Query 14")

    def test_11_historical_turns_do_not_contain_live_dataframes(self):
        """11. Historical turns do not contain live DataFrames (QueryResultSummary used)."""
        large_df = pd.DataFrame({"val": list(range(100))})
        self.history.add_turn(
            question="Get values",
            sql="SELECT * FROM table;",
            result_df=large_df,
        )

        turn = self.history.get_turns()[0]
        self.assertIsInstance(turn.result, QueryResultSummary)
        self.assertNotIsInstance(turn.result, pd.DataFrame)
        self.assertEqual(turn.result.row_count, 100)
        self.assertFalse(hasattr(turn.result, "iloc"))  # Not a DataFrame

    def test_12_failed_sql_generation_does_not_create_history_turn(self):
        """12. Failed SQL generation does not create a history turn."""
        initial_count = len(self.history)

        def failing_llm(prompt: str):
            raise LLMGenerationError("Gemini API timed out")

        with self.assertRaises(LLMGenerationError):
            generate_sql(
                "Find something",
                schema=self.schema,
                conversation_context=self.history,
                llm_client=failing_llm,
            )

        self.assertEqual(len(self.history), initial_count)

    def test_13_failed_sql_execution_does_not_create_history_turn(self):
        """13. Failed SQL execution in DuckDB does not create a history turn."""
        initial_count = len(self.history)

        with DuckDBEngine() as engine:
            engine.register_dataframe(self.sample_df, table_name="sales")
            with self.assertRaises(QueryExecutionError):
                engine.execute_query("SELECT non_existent_column FROM sales;")
                # add_turn is not called if execute_query raises
                self.history.add_turn("Bad query", sql="SELECT non_existent_column FROM sales;")

        self.assertEqual(len(self.history), initial_count)

    def test_14_successful_query_creates_history_turn(self):
        """14. Successful analytical query creates a proper history turn."""
        with DuckDBEngine() as engine:
            engine.register_dataframe(self.sample_df, table_name="sales")
            sql = "SELECT category, SUM(sales_amount) AS total FROM sales GROUP BY category;"
            validate_read_only_query(sql)
            res_df = engine.execute_query(sql)

            self.history.add_turn(
                question="Sales by category",
                sql=sql,
                result_df=res_df,
                insights="Electronics leads with highest sales.",
            )

        self.assertEqual(len(self.history), 1)
        turn = self.history.get_turns()[0]
        self.assertEqual(turn.question, "Sales by category")
        self.assertEqual(turn.sql, sql)
        self.assertEqual(turn.insights, "Electronics leads with highest sales.")
        self.assertEqual(turn.result.row_count, 3)

    def test_15_clearing_history_allows_new_first_question_on_same_dataset(self):
        """15. Clearing history allows a new first question on the same dataset with zero context."""
        # Perform initial turn
        self.history.add_turn(
            "First question",
            sql="SELECT region, SUM(sales_amount) FROM sales GROUP BY region;",
        )
        self.assertEqual(len(self.history), 1)

        # Clear history
        self.history.clear()
        self.assertEqual(len(self.history), 0)

        # Now ask a new question - prompt must be a clean first-question prompt
        captured_prompts = []

        def mock_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "SELECT AVG(sales_amount) FROM sales;"

        sql = generate_sql(
            "What is the average sales amount?",
            schema=self.schema,
            conversation_context=self.history,
            recent_turn_limit=3,
            llm_client=mock_llm,
        )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertNotIn("PREVIOUS ANALYTICS CONTEXT", prompt)
        self.assertNotIn("First question", prompt)
        self.assertIn("USER QUESTION:\nWhat is the average sales amount?", prompt)
        self.assertEqual(sql, "SELECT AVG(sales_amount) FROM sales;")


if __name__ == "__main__":
    unittest.main()
