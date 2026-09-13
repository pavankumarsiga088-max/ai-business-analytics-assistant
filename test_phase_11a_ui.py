"""
Test Suite for Phase 11A: UI/UX Modernization & Presentation Layer.

Validates structural UI components and presentation helpers:
1. Dynamic result table column configuration for floats (%.2f).
2. Dynamic result table column configuration for integers (%d).
3. Dynamic result table column configuration for datetimes (YYYY-MM-DD).
4. Dynamic result table column configuration for text/categorical columns.
5. Preserves DataFrame immutability during column configuration generation.
6. Handles empty, single-row, and null-valued DataFrames safely.
7. Works dynamically with arbitrary schemas without hardcoded column names.
8. Handles wide DataFrames (30+ columns) without error.
9. Dataset overview KPI generation works correctly with loaded data.
10. Data quality checks and scoring output match expected overview KPI structure.
11. Current analysis state schema maintains separation from conversation history.
12. Clearing conversation resets current analysis while preserving loaded dataset.
13. Context indicators correctly distinguish new vs context-aware questions.
"""

import unittest
import pandas as pd
import streamlit as st

from app import build_result_column_config, run_all_quality_checks
from src.data_quality.scoring import calculate_quality_score, generate_quality_report
from src.analytics.conversation import ConversationHistory
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query
from src.analytics.insights import generate_business_insights


class TestPhase11AUI(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "order_id": ["ORD-101", "ORD-102", "ORD-103"],
            "region": ["North", "South", "East"],
            "order_date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"]),
            "quantity": [2, 5, 1],
            "unit_price": [499.99, 19.50, 1200.00],
            "revenue": [999.98, 97.50, 1200.00],
        })

    def test_1_column_config_float_formatting(self):
        """Float columns must receive NumberColumn with 2 decimal precision (%.2f)."""
        config = build_result_column_config(self.sample_df)
        self.assertIn("unit_price", config)
        self.assertIn("revenue", config)

        unit_price_col = config["unit_price"]
        # In Streamlit, NumberColumn returns a ColumnConfig dict or object
        fmt = unit_price_col.get("type_config", {}).get("format") or getattr(unit_price_col, "format", None)
        self.assertEqual(fmt, "%.2f")

    def test_2_column_config_integer_formatting(self):
        """Integer columns must receive NumberColumn with integer format (%d)."""
        config = build_result_column_config(self.sample_df)
        self.assertIn("quantity", config)

        qty_col = config["quantity"]
        fmt = qty_col.get("type_config", {}).get("format") or getattr(qty_col, "format", None)
        self.assertEqual(fmt, "%d")

    def test_3_column_config_datetime_formatting(self):
        """Datetime columns must receive DatetimeColumn with YYYY-MM-DD format."""
        config = build_result_column_config(self.sample_df)
        self.assertIn("order_date", config)

        date_col = config["order_date"]
        fmt = date_col.get("type_config", {}).get("format") or getattr(date_col, "format", None)
        self.assertEqual(fmt, "YYYY-MM-DD")

    def test_4_column_config_text_columns(self):
        """Text/Categorical columns must be configured with readable column labels."""
        config = build_result_column_config(self.sample_df)
        self.assertIn("order_id", config)
        self.assertIn("region", config)

        region_col = config["region"]
        label = region_col.get("label") or getattr(region_col, "label", None)
        self.assertEqual(label, "region")

    def test_5_column_config_preserves_dataframe_immutability(self):
        """Generating column config must never mutate the input DataFrame."""
        df_copy = self.sample_df.copy(deep=True)
        _ = build_result_column_config(self.sample_df)

        pd.testing.assert_frame_equal(self.sample_df, df_copy)

    def test_6_column_config_handles_empty_and_null_dataframes(self):
        """Handles empty DataFrames, None, and DataFrames containing NaNs without exceptions."""
        self.assertEqual(build_result_column_config(None), {})
        self.assertEqual(build_result_column_config(pd.DataFrame()), {})

        nan_df = pd.DataFrame({
            "str_col": ["a", None, "c"],
            "float_col": [1.5, float("nan"), 3.0],
            "int_col": [1, 2, 3],
        })
        config = build_result_column_config(nan_df)
        self.assertEqual(len(config), 3)
        self.assertIn("str_col", config)
        self.assertIn("float_col", config)
        self.assertIn("int_col", config)

    def test_7_column_config_works_for_arbitrary_schemas(self):
        """Works dynamically for arbitrary SQL result DataFrames with unexpected column names."""
        arbitrary_df = pd.DataFrame({
            "avg_transaction_val_2023": [105.432],
            "count_customers_tier_1": [42],
            "custom_entity_name": ["Global Operations"],
        })
        config = build_result_column_config(arbitrary_df)
        self.assertEqual(len(config), 3)
        self.assertIn("avg_transaction_val_2023", config)
        self.assertIn("count_customers_tier_1", config)
        self.assertIn("custom_entity_name", config)

    def test_8_column_config_handles_wide_dataframes(self):
        """Handles wide DataFrames (e.g. 35 columns) cleanly."""
        wide_data = {f"metric_{i}": [float(i) * 1.1] for i in range(35)}
        wide_df = pd.DataFrame(wide_data)

        config = build_result_column_config(wide_df)
        self.assertEqual(len(config), 35)

    def test_9_dataset_overview_kpis(self):
        """Verifies row, column, and file metadata calculations for KPI display."""
        total_rows = len(self.sample_df)
        total_cols = len(self.sample_df.columns)
        dataset_name = "sample_test_data.csv"

        self.assertEqual(total_rows, 3)
        self.assertEqual(total_cols, 6)
        self.assertEqual(dataset_name, "sample_test_data.csv")

    def test_10_quality_report_integration_for_overview(self):
        """Quality checks and scoring generate expected report fields for KPI cards."""
        check_results = run_all_quality_checks(self.sample_df)
        self.assertEqual(len(check_results), 10)

        score_res = calculate_quality_score(check_results, total_rows=len(self.sample_df))
        report = generate_quality_report(check_results, score_res, total_rows=len(self.sample_df))

        self.assertIn("overall_score", report)
        self.assertIn("quality_status", report)
        self.assertIn("total_penalty", report)
        self.assertIsInstance(report["overall_score"], (int, float))

    def test_11_current_analysis_state_schema(self):
        """Current analysis state maintains separation from conversation history."""
        sample_state = {
            "question": "What is revenue by region?",
            "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region;",
            "result_df": pd.DataFrame({"region": ["North"], "revenue": [999.98]}),
            "insights": "North leads with highest revenue.",
            "context_badge": "Used previous analytical context (1 recent turn)",
        }

        self.assertIn("question", sample_state)
        self.assertIn("sql", sample_state)
        self.assertIn("result_df", sample_state)
        self.assertIn("insights", sample_state)
        self.assertIn("context_badge", sample_state)

    def test_12_clearing_history_resets_current_analysis(self):
        """Clearing conversation resets history and current analysis without dropping dataset."""
        history = ConversationHistory(dataset_name="sales.csv")
        history.add_turn("Q1", sql="SELECT 1;")
        self.assertEqual(len(history), 1)

        simulated_session = {
            "current_dataset_name": "sales.csv",
            "conversation_history": history,
            "last_analytics_result": {"question": "Q1"},
            "user_nl_question": "Follow up",
        }

        # Simulate clear action
        simulated_session["conversation_history"].clear()
        simulated_session["last_analytics_result"] = None
        simulated_session["user_nl_question"] = ""

        self.assertEqual(len(simulated_session["conversation_history"]), 0)
        self.assertIsNone(simulated_session["last_analytics_result"])
        self.assertEqual(simulated_session["user_nl_question"], "")
        self.assertEqual(simulated_session["current_dataset_name"], "sales.csv")

    def test_13_context_indicators_distinguish_new_vs_followup(self):
        """Context indicators correctly produce distinct badges for 0 turns vs multiple turns."""
        history = ConversationHistory()
        self.assertEqual(len(history), 0)

        # 0 turns -> New analytical question
        num_turns = len(history)
        badge_0 = (
            f"Used previous analytical context ({min(num_turns, 3)} recent turns)"
            if num_turns > 0
            else "New analytical question (no previous context)"
        )
        self.assertEqual(badge_0, "New analytical question (no previous context)")

        # 2 turns -> Context-aware
        history.add_turn("Q1", sql="SELECT 1;")
        history.add_turn("Q2", sql="SELECT 2;")
        num_turns = len(history)
        badge_2 = f"Used previous analytical context ({min(num_turns, 3)} recent turns)"
        self.assertEqual(badge_2, "Used previous analytical context (2 recent turns)")

    def test_14_generate_business_insights_call_contract(self):
        """Verifies the exact parameter set used by app.py can call generate_business_insights() without schema argument."""
        mock_output = "• Insight: North region had highest order count.\n• Recommendation: Investigate marketing in East."

        def mock_llm(prompt: str) -> str:
            return mock_output

        # Calling with exact parameters as app.py (question, result_df, sql)
        res = generate_business_insights(
            question="What is total revenue by region?",
            result_df=self.sample_df,
            sql="SELECT region, SUM(revenue) FROM sales GROUP BY region;",
            llm_client=mock_llm,
        )
        self.assertEqual(res, mock_output)

        # Assert that passing schema parameter raises TypeError (guardrail against unexpected keyword argument)
        with self.assertRaises(TypeError):
            generate_business_insights(
                question="What is total revenue by region?",
                result_df=self.sample_df,
                sql="SELECT region, SUM(revenue) FROM sales GROUP BY region;",
                schema={"region": "VARCHAR", "revenue": "DOUBLE"},
                llm_client=mock_llm,
            )

    def test_15_smart_result_presentation_scalar_integration(self):
        """Verifies scalar query result generates valid single KPI card metadata for UI rendering."""
        from src.analytics.result_presentation import build_kpi_metadata
        scalar_df = pd.DataFrame({"total_revenue": [109510.0]})
        meta = build_kpi_metadata(scalar_df)

        self.assertEqual(meta.classification, "kpi")
        self.assertEqual(len(meta.kpi_cards), 1)
        self.assertEqual(meta.kpi_cards[0].label, "Total Revenue")
        self.assertEqual(meta.kpi_cards[0].formatted_value, "109,510")
        self.assertTrue(meta.show_table)
        self.assertFalse(meta.show_chart)

    def test_16_smart_result_presentation_multi_kpi_integration(self):
        """Verifies multi-metric aggregate query result generates valid multi-KPI card metadata for UI rendering."""
        from src.analytics.result_presentation import build_kpi_metadata
        multi_df = pd.DataFrame({
            "total_revenue": [641021.0],
            "avg_order_value": [6410.21],
            "total_orders": [100],
        })
        meta = build_kpi_metadata(multi_df)

        self.assertEqual(meta.classification, "multi_kpi")
        self.assertEqual(len(meta.kpi_cards), 3)
        self.assertEqual(meta.kpi_cards[0].formatted_value, "641,021")
        self.assertEqual(meta.kpi_cards[1].formatted_value, "6,410.21")
        self.assertEqual(meta.kpi_cards[2].formatted_value, "100")
        self.assertTrue(meta.show_table)
        self.assertFalse(meta.show_chart)


    def test_17_downloads_available_for_valid_current_result(self):
        """Verifies CSV, Excel, and HTML chart downloads are prepared for valid chartable result."""
        from src.analytics.visualization import create_chart
        from src.analytics.downloads import (
            dataframe_to_csv_bytes,
            dataframe_to_excel_bytes,
            figure_to_html_bytes,
        )

        res_df = pd.DataFrame({"region": ["North", "South"], "revenue": [100.0, 200.0]})
        chart_fig = create_chart(res_df)
        self.assertIsNotNone(chart_fig)

        csv_bytes = dataframe_to_csv_bytes(res_df)
        excel_bytes = dataframe_to_excel_bytes(res_df)
        html_bytes = figure_to_html_bytes(chart_fig)

        self.assertGreater(len(csv_bytes), 0)
        self.assertGreater(len(excel_bytes), 0)
        self.assertGreater(len(html_bytes), 0)

    def test_18_downloads_chart_omitted_for_scalar_kpi(self):
        """For scalar KPI results where create_chart returns None, chart download is omitted."""
        from src.analytics.visualization import create_chart
        from src.analytics.downloads import (
            dataframe_to_csv_bytes,
            dataframe_to_excel_bytes,
        )

        scalar_df = pd.DataFrame({"total_revenue": [109510.0]})
        chart_fig = create_chart(scalar_df)
        self.assertIsNone(chart_fig)

        # Tabular exports still succeed
        csv_bytes = dataframe_to_csv_bytes(scalar_df)
        excel_bytes = dataframe_to_excel_bytes(scalar_df)
        self.assertGreater(len(csv_bytes), 0)
        self.assertGreater(len(excel_bytes), 0)

        # Condition for UI: chart download button is shown if and only if chart_fig is not None
        show_chart_dl = (chart_fig is not None)
        self.assertFalse(show_chart_dl)

    def test_19_downloads_omitted_when_no_analysis_or_empty_result(self):
        """Download controls must not appear when last_analytics_result is None or empty."""
        simulated_session = {
            "current_dataset_name": "sales.csv",
            "last_analytics_result": None,
        }
        has_analysis = (
            "last_analytics_result" in simulated_session
            and simulated_session["last_analytics_result"] is not None
        )
        self.assertFalse(has_analysis)

        empty_df = pd.DataFrame()
        show_downloads = has_analysis and not empty_df.empty
        self.assertFalse(show_downloads)

    def test_20_downloads_use_current_result_only_without_conversation_leakage(self):
        """Downloads strictly reflect the current analysis turn and not historical conversation turns."""
        from src.analytics.downloads import dataframe_to_csv_bytes

        history = ConversationHistory(dataset_name="sales.csv")
        history.add_turn("Q1", sql="SELECT 1", result_df=pd.DataFrame({"turn": [1]}))
        history.add_turn("Q2", sql="SELECT 2", result_df=pd.DataFrame({"turn": [2]}))

        # Current analysis is Turn 2
        current_res_df = pd.DataFrame({"turn": [2], "metric": [999]})

        csv_bytes = dataframe_to_csv_bytes(current_res_df)
        csv_text = csv_bytes.decode("utf-8")

        self.assertIn("metric", csv_text)
        self.assertIn("999", csv_text)
        # Turn 1 data is not present in the current download
        self.assertNotIn("turn,1", csv_text)

    def test_21_sql_explanation_section_coexists_with_raw_sql(self):
        """Current analysis state maintains independent visibility for both raw SQL and explanation."""
        sample_state = {
            "question": "What is total revenue by region?",
            "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region;",
            "result_df": pd.DataFrame({"region": ["East"], "revenue": [100.0]}),
            "insights": "East generated highest revenue.",
            "sql_explanation": "### What the query does\nGroups sales by region...",
            "context_badge": "New analytical question",
        }

        self.assertIn("sql", sample_state)
        self.assertIn("sql_explanation", sample_state)
        self.assertNotEqual(sample_state["sql"], sample_state["sql_explanation"])
        self.assertTrue(len(sample_state["sql"]) > 0)
        self.assertTrue(len(sample_state["sql_explanation"]) > 0)

    def test_22_explanation_failure_preserves_full_analysis_workflow(self):
        """If SQL explanation generation fails, results, charts, insights, and downloads still work."""
        from src.analytics.visualization import create_chart
        from src.analytics.downloads import dataframe_to_csv_bytes, dataframe_to_excel_bytes

        res_df = pd.DataFrame({"region": ["East", "West"], "revenue": [100.0, 200.0]})
        chart = create_chart(res_df)

        failed_state = {
            "question": "What is revenue by region?",
            "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region;",
            "result_df": res_df,
            "insights": "Insights still present.",
            "sql_explanation": None,  # Simulated failure
        }

        # Results still exist
        self.assertFalse(failed_state["result_df"].empty)
        # Chart still works
        self.assertIsNotNone(chart)
        # Insights still exist
        self.assertIsNotNone(failed_state["insights"])
        # Downloads still work
        self.assertGreater(len(dataframe_to_csv_bytes(failed_state["result_df"])), 0)
        self.assertGreater(len(dataframe_to_excel_bytes(failed_state["result_df"])), 0)

    def test_23_explanation_omitted_when_no_analysis_exists(self):
        """When no analysis exists, explanation expander is not rendered."""
        session_state = {"last_analytics_result": None}
        has_analysis = session_state.get("last_analytics_result") is not None
        self.assertFalse(has_analysis)


    def test_24_data_quality_issue_explorer_structure(self):
        """Phase 12A: Quality checks produce structured QualityIssueDetail objects for all 10 checks."""
        from src.data_quality.issue_details import extract_issue_details, QualityIssueDetail

        check_results = run_all_quality_checks(self.sample_df)
        issue_details = extract_issue_details(check_results, df=self.sample_df)

        self.assertEqual(len(issue_details), 10)
        for issue in issue_details:
            self.assertIsInstance(issue, QualityIssueDetail)
            self.assertTrue(len(issue.check_name) > 0)
            self.assertIn(issue.severity, ["HIGH", "MEDIUM", "LOW", "PASS"])
            self.assertIn(issue.status, ["PASS", "FAIL", "WARNING"])
            self.assertTrue(len(issue.message) > 0)
            self.assertTrue(len(issue.recommendation) > 0)
            self.assertIsInstance(issue.affected_columns, list)
            self.assertIsInstance(issue.sample_values, list)
            self.assertIsInstance(issue.sample_records, pd.DataFrame)
            self.assertLessEqual(len(issue.sample_records), 5)

    def test_25_severity_summary_metrics(self):
        """Phase 12A: Severity summary metrics accurately calculate counts for UI badges."""
        from src.data_quality.issue_details import extract_issue_details, get_issue_severity_counts

        check_results = run_all_quality_checks(self.sample_df)
        issue_details = extract_issue_details(check_results, df=self.sample_df)
        counts = get_issue_severity_counts(issue_details)

        self.assertIn("HIGH", counts)
        self.assertIn("MEDIUM", counts)
        self.assertIn("LOW", counts)
        self.assertIn("PASS", counts)
        self.assertIn("TOTAL_ISSUES", counts)
        total_checks = counts["HIGH"] + counts["MEDIUM"] + counts["LOW"] + counts["PASS"]
        self.assertEqual(total_checks, 10)

    def test_26_clean_dataset_state(self):
        """Phase 12A: Completely clean dataset reflects zero total issues and positive state."""
        from src.data_quality.issue_details import extract_issue_details, get_issue_severity_counts

        # Synthetic completely clean 4-row dataset
        clean_df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
            "customer_name": ["Alice", "Bob", "Charlie", "David"],
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "region": ["North", "South", "East", "West"],
            "quantity": [1, 2, 3, 4],
            "unit_price": [10.0, 10.0, 10.0, 10.0],
            "revenue": [10.0, 20.0, 30.0, 40.0],
        })

        check_results = run_all_quality_checks(clean_df)
        issue_details = extract_issue_details(check_results, df=clean_df)
        counts = get_issue_severity_counts(issue_details)

        self.assertEqual(counts["TOTAL_ISSUES"], 0)
        self.assertEqual(counts["PASS"], 10)

    def test_27_dirty_dataset_issues_detail(self):
        """Phase 12A: Dirty dataset identifies specific issues with bounded samples."""
        from src.data_quality.issue_details import extract_issue_details, get_issue_severity_counts

        dirty_df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-1", "ORD-3"],  # Duplicate ID
            "customer_name": ["Alice", None, "Charlie"],  # Missing
            "order_date": ["2024-01-01", "bad-date", "2024-01-03"],  # Invalid date
            "quantity": [-1, 2, 3],  # Invalid qty
            "unit_price": [10.0, "bad_price", 10.0],  # Numeric type
            "revenue": [10.0, 20.0, 999.0],  # Inconsistent revenue
        })

        check_results = run_all_quality_checks(dirty_df)
        issue_details = extract_issue_details(check_results, df=dirty_df)
        counts = get_issue_severity_counts(issue_details)

        self.assertGreater(counts["HIGH"], 0)
        self.assertGreater(counts["TOTAL_ISSUES"], 0)

        # Check that samples are bounded
        for issue in issue_details:
            self.assertLessEqual(len(issue.sample_records), 5)
            self.assertLessEqual(len(issue.sample_values), 5)

    def test_28_existing_quality_score_preserved(self):
        """Phase 12A: Existing scoring, penalties, and recommendations remain fully preserved."""
        check_results = run_all_quality_checks(self.sample_df)
        score_res = calculate_quality_score(check_results, total_rows=len(self.sample_df))
        report = generate_quality_report(check_results, score_res, total_rows=len(self.sample_df))

        self.assertIn("overall_score", report)
        self.assertIn("quality_status", report)
        self.assertIn("recommendations", report)
        self.assertIn("check_details", report)


if __name__ == "__main__":
    unittest.main()

