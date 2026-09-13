"""
Offline Unit Tests for Phase 10A: Conversation State Foundation.

Validates:
1. Empty history.
2. Add one turn.
3. Add multiple turns.
4. Chronological ordering.
5. Retrieve recent turns.
6. Clear history.
7. Maximum turn limit.
8. Result row truncation.
9. Original row count preserved.
10. DataFrame/result immutability.
11. Stored turns remain independent.
12. Empty/None result handling.
13. No external API calls.
14. No DuckDB dependency.
15. No Streamlit dependency.
"""
import ast
import os
import unittest
import pandas as pd

from src.analytics.conversation import (
    ConversationTurn,
    ConversationHistory,
    QueryResultSummary,
)


class TestConversationState(unittest.TestCase):
    def setUp(self):
        self.history = ConversationHistory(max_turns=10, max_result_rows=25)

    def test_1_empty_history(self):
        """1. Empty history verification."""
        self.assertEqual(len(self.history), 0)
        self.assertTrue(self.history.is_empty())
        self.assertEqual(self.history.get_turns(), [])
        self.assertEqual(self.history.get_recent_turns(5), [])

    def test_2_add_one_turn(self):
        """2. Add one turn verification."""
        df = pd.DataFrame({"region": ["North"], "revenue": [1250.0]})
        turn = self.history.add_turn(
            question="What is the total revenue for North?",
            sql="SELECT region, SUM(revenue) AS revenue FROM sales WHERE region = 'North' GROUP BY region;",
            result_df=df,
            insights="North generated $1,250.00 in total revenue.",
        )

        self.assertEqual(len(self.history), 1)
        self.assertFalse(self.history.is_empty())
        self.assertEqual(turn.question, "What is the total revenue for North?")
        self.assertTrue(turn.has_sql)
        self.assertTrue(turn.has_result)
        self.assertTrue(turn.has_insights)
        self.assertEqual(turn.result.row_count, 1)
        self.assertEqual(turn.result.columns, ["region", "revenue"])
        self.assertEqual(turn.result.rows, [{"region": "North", "revenue": 1250.0}])

    def test_3_add_multiple_turns(self):
        """3. Add multiple turns."""
        self.history.add_turn("Question 1")
        self.history.add_turn("Question 2")
        self.history.add_turn("Question 3")
        self.assertEqual(len(self.history), 3)

    def test_4_chronological_ordering(self):
        """4. Chronological ordering preservation."""
        self.history.add_turn("Turn A")
        self.history.add_turn("Turn B")
        self.history.add_turn("Turn C")

        turns = self.history.get_turns()
        questions = [t.question for t in turns]
        self.assertEqual(questions, ["Turn A", "Turn B", "Turn C"])

    def test_5_retrieve_recent_turns(self):
        """5. Retrieve recent turns (limit=N, boundary cases)."""
        for i in range(1, 6):
            self.history.add_turn(f"Question {i}")

        # Limit 2 should return last 2 in chronological order
        recent_2 = self.history.get_recent_turns(limit=2)
        self.assertEqual([t.question for t in recent_2], ["Question 4", "Question 5"])

        # Limit 0 or negative should return empty list
        self.assertEqual(self.history.get_recent_turns(limit=0), [])
        self.assertEqual(self.history.get_recent_turns(limit=-1), [])

        # Limit >= total returns all turns
        recent_all = self.history.get_recent_turns(limit=10)
        self.assertEqual(len(recent_all), 5)
        self.assertEqual([t.question for t in recent_all], [f"Question {i}" for i in range(1, 6)])

        # Limit None returns all turns
        self.assertEqual(len(self.history.get_recent_turns(None)), 5)

    def test_6_clear_history(self):
        """6. Clear history operation."""
        self.history.add_turn("Turn 1")
        self.history.add_turn("Turn 2")
        self.assertEqual(len(self.history), 2)

        self.history.clear()
        self.assertEqual(len(self.history), 0)
        self.assertTrue(self.history.is_empty())
        self.assertEqual(self.history.get_turns(), [])

    def test_7_maximum_turn_limit(self):
        """7. Maximum turn limit bounded behavior."""
        bounded_history = ConversationHistory(max_turns=3)
        for i in range(1, 6):
            bounded_history.add_turn(f"Question {i}")

        self.assertEqual(len(bounded_history), 3)
        questions = [t.question for t in bounded_history.get_turns()]
        self.assertEqual(questions, ["Question 3", "Question 4", "Question 5"])

    def test_8_result_row_truncation(self):
        """8. Result row truncation when rows exceed max_result_rows."""
        large_df = pd.DataFrame({
            "id": list(range(1, 51)),
            "val": [f"val_{i}" for i in range(1, 51)],
        })
        turn = self.history.add_turn("Large result query", result_df=large_df)

        self.assertTrue(turn.result.is_truncated)
        self.assertEqual(len(turn.result.rows), 25)
        self.assertEqual(turn.result.truncated_rows, 25)

    def test_9_original_row_count_preserved(self):
        """9. Original row count preserved despite truncation."""
        large_df = pd.DataFrame({
            "id": list(range(1, 100)),
            "val": list(range(1, 100)),
        })
        turn = self.history.add_turn("100 row query", result_df=large_df)

        self.assertEqual(turn.result.row_count, 99)
        self.assertEqual(turn.result.truncated_rows, 25)
        self.assertTrue(turn.result.is_truncated)

    def test_10_dataframe_immutability(self):
        """10. Caller mutating DataFrame does not affect stored history."""
        df = pd.DataFrame({"product": ["Laptop", "Mouse"], "sales": [1200, 45]})
        turn = self.history.add_turn("Sales query", result_df=df)

        # Mutate the caller's DataFrame in place
        df.iloc[0, 0] = "MUTATED_VALUE"
        df.iloc[0, 1] = 999999
        df["injected_column"] = [1, 2]

        # Verify the stored snapshot was NOT altered
        self.assertEqual(turn.result.rows[0]["product"], "Laptop")
        self.assertEqual(turn.result.rows[0]["sales"], 1200)
        self.assertNotIn("injected_column", turn.result.columns)

    def test_11_stored_turns_remain_independent(self):
        """11. Mutating returned list from get_turns() does not mutate history."""
        self.history.add_turn("Turn 1")
        turns = self.history.get_turns()
        turns.clear()

        self.assertEqual(len(self.history), 1)
        self.assertEqual(self.history.get_turns()[0].question, "Turn 1")

    def test_12_empty_and_none_result_handling(self):
        """12. Handling of empty DataFrame and None results."""
        # Empty DataFrame
        turn_empty = self.history.add_turn("Empty result query", result_df=pd.DataFrame(columns=["a", "b"]))
        self.assertIsNotNone(turn_empty.result)
        self.assertTrue(turn_empty.result.empty)
        self.assertEqual(turn_empty.result.row_count, 0)
        self.assertEqual(turn_empty.result.columns, ["a", "b"])
        self.assertFalse(turn_empty.has_result)

        # None result
        turn_none = self.history.add_turn("None result query", result_df=None)
        self.assertIsNone(turn_none.result)
        self.assertFalse(turn_none.has_result)

    def test_13_no_external_api_calls(self):
        """13. Pure offline behavior - no network socket / API calls."""
        # Check that ConversationHistory operations require no credentials or network
        offline_hist = ConversationHistory()
        for i in range(5):
            offline_hist.add_turn(
                question=f"Question {i}",
                sql=f"SELECT {i};",
                result_df=pd.DataFrame({"x": [i]}),
                insights=f"Insight {i}",
            )
        self.assertEqual(len(offline_hist), 5)

    def test_14_no_duckdb_dependency(self):
        """14. No DuckDB imports or dependencies in src/analytics/conversation.py."""
        conv_file_path = os.path.join("src", "analytics", "conversation.py")
        with open(conv_file_path, "r", encoding="utf-8") as f:
            code = f.read()

        tree = ast.parse(code)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module)

        for mod in imported_modules:
            self.assertFalse(
                mod == "duckdb" or mod.startswith("duckdb."),
                f"Forbidden dependency on duckdb detected in conversation.py: {mod}",
            )
            self.assertFalse(
                "duckdb_engine" in mod,
                f"Forbidden internal import of duckdb_engine in conversation.py: {mod}",
            )

    def test_15_no_streamlit_dependency(self):
        """15. No Streamlit imports or dependencies in src/analytics/conversation.py."""
        conv_file_path = os.path.join("src", "analytics", "conversation.py")
        with open(conv_file_path, "r", encoding="utf-8") as f:
            code = f.read()

        tree = ast.parse(code)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module)

        for mod in imported_modules:
            self.assertFalse(
                mod == "streamlit" or mod.startswith("streamlit."),
                f"Forbidden dependency on streamlit detected in conversation.py: {mod}",
            )

    def test_16_dataset_boundary_reset(self):
        """16. Dataset boundary resetting."""
        hist = ConversationHistory(dataset_name="sales.csv")
        hist.add_turn("Turn 1")
        self.assertEqual(hist.dataset_name, "sales.csv")
        self.assertEqual(len(hist), 1)

        hist.reset_for_dataset("dirty_sales.xlsx")
        self.assertEqual(hist.dataset_name, "dirty_sales.xlsx")
        self.assertEqual(len(hist), 0)
        self.assertTrue(hist.is_empty())

    def test_17_serialization_roundtrip(self):
        """17. Serialization to/from dictionary."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        self.history.add_turn("Test roundtrip", sql="SELECT 1;", result_df=df, insights="Looks good.")
        data = self.history.to_dict()

        restored = ConversationHistory.from_dict(data)
        self.assertEqual(len(restored), 1)
        turn = restored.get_turns()[0]
        self.assertEqual(turn.question, "Test roundtrip")
        self.assertEqual(turn.sql, "SELECT 1;")
        self.assertEqual(turn.insights, "Looks good.")
        self.assertEqual(turn.result.row_count, 3)
        self.assertEqual(turn.result.rows, [{"col": 1}, {"col": 2}, {"col": 3}])

    def test_18_sql_explanation_storage_and_serialization(self):
        """18. Verify sql_explanation can be stored, checked, and serialized."""
        turn = self.history.add_turn(
            question="What is total revenue?",
            sql="SELECT SUM(revenue) FROM sales;",
            sql_explanation="Calculates total revenue across all rows.",
        )
        self.assertTrue(turn.has_sql_explanation)
        self.assertEqual(turn.sql_explanation, "Calculates total revenue across all rows.")

        d = turn.to_dict()
        self.assertEqual(d.get("sql_explanation"), "Calculates total revenue across all rows.")

        reconstructed = ConversationTurn.from_dict(d)
        self.assertTrue(reconstructed.has_sql_explanation)
        self.assertEqual(reconstructed.sql_explanation, "Calculates total revenue across all rows.")

    def test_19_existing_turns_without_explanation_remain_backward_compatible(self):
        """19. Turns created without sql_explanation default to None cleanly."""
        turn = self.history.add_turn(question="Simple question")
        self.assertIsNone(turn.sql_explanation)
        self.assertFalse(turn.has_sql_explanation)

        # Deserializing legacy dictionary without sql_explanation field
        legacy_dict = {"question": "Old turn", "sql": "SELECT 1;", "insights": "Old insight"}
        restored = ConversationTurn.from_dict(legacy_dict)
        self.assertIsNone(restored.sql_explanation)
        self.assertFalse(restored.has_sql_explanation)


if __name__ == "__main__":
    unittest.main()
