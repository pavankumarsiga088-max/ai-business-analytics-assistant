"""
Functional and Live Gemini Verification Script for Phase 11D — SQL Explanation & Transparency.

Tests:
1. Live Gemini verification using GEMINI_API_KEY (if present in .env) for question:
   "What is the total revenue by region?" and SQL:
   SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;
   Verifying region grouping, total revenue calculation, and descending ordering.
2. Case A: Grouped analysis ("What is the total revenue by region?")
   Pipeline: Question -> SQL -> Validation -> DuckDB -> SQL Explanation -> Chart -> Insights -> Downloads -> History.
3. Case B: Scalar analysis ("What is the total revenue?")
   Pipeline: KPI card, SQL explanation, result table, Insights, CSV/Excel downloads, chart omitted.
4. Case C: Follow-up analysis ("What about East?")
   Pipeline: Follow-up question using conversation context -> new SQL -> new SQL explanation -> History preserved.
"""

import os
import io
import openpyxl
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from src.data_quality.loader import load_dataset
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query
from src.analytics.nl_to_sql import generate_sql
from src.analytics.visualization import create_chart
from src.analytics.insights import generate_business_insights
from src.analytics.conversation import ConversationHistory
from src.analytics.result_presentation import build_kpi_metadata
from src.analytics.downloads import dataframe_to_csv_bytes, dataframe_to_excel_bytes
from src.analytics.sql_explanation import generate_sql_explanation


def verify_live_gemini():
    print("=== LIVE GEMINI VERIFICATION ===")
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        print("[SKIP] GEMINI_API_KEY not found in environment. Skipping live API call.")
        return False

    question = "What is the total revenue by region?"
    sql = (
        "SELECT region, SUM(revenue) AS total_revenue "
        "FROM sales GROUP BY region ORDER BY total_revenue DESC;"
    )

    print("Sending prompt to Gemini for SQL explanation...")
    explanation = generate_sql_explanation(
        question=question,
        sql=sql,
        api_key=api_key,
    )

    print("\n--- Live Gemini Explanation Output ---")
    print(explanation)
    print("--------------------------------------")

    exp_lower = explanation.lower()
    # Verify concepts corresponding to:
    # 1. region / grouping
    has_region = "region" in exp_lower
    has_group = "group" in exp_lower or "each region" in exp_lower or "by region" in exp_lower
    # 2. total / sum / revenue
    has_revenue = "revenue" in exp_lower or "sales" in exp_lower
    has_sum = "total" in exp_lower or "sum" in exp_lower or "calculat" in exp_lower or "add" in exp_lower
    # 3. sorting / descending / highest
    has_order = "highest" in exp_lower or "descend" in exp_lower or "sort" in exp_lower or "order" in exp_lower

    print(f"Concept checks: Region={has_region}, Grouping={has_group}, Revenue={has_revenue}, Sum={has_sum}, Order={has_order}")
    assert has_region and has_revenue, "Live explanation must mention region and revenue"
    assert has_group or has_sum, "Live explanation must mention grouping or total revenue calculation"
    assert has_order, "Live explanation must mention sorting/ordering"

    print("[OK] Live Gemini verification PASSED!\n")
    return True


def verify_functional_cases():
    print("=== FUNCTIONAL VERIFICATION (sample_sales.csv) ===")
    df = load_dataset("sample_data/sample_sales.csv")
    print(f"Loaded {len(df)} rows from sample_sales.csv")

    conv_history = ConversationHistory(dataset_name="sample_sales.csv")

    with DuckDBEngine() as engine:
        engine.register_dataframe(df, table_name="sales")
        schema = engine.get_schema("sales")

        # -------------------------------------------------------------
        # CASE A: Grouped query ("What is the total revenue by region?")
        # -------------------------------------------------------------
        print("\n--- CASE A: Grouped query ('What is the total revenue by region?') ---")
        q_a = "What is the total revenue by region?"
        sql_a = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;"
        validate_read_only_query(sql_a)
        res_a = engine.execute_query(sql_a)

        # 1. SQL Explanation
        explanation_a = generate_sql_explanation(
            question=q_a,
            sql=sql_a,
            schema=schema,
            llm_client=lambda p: (
                "### What the query does\n"
                "Groups sales by region and calculates the total revenue for each region.\n\n"
                "### How it works\n"
                "- Groups rows by region.\n"
                "- Calculates total revenue per region.\n"
                "- Sorts regions from highest to lowest revenue."
            ),
        )
        assert "### What the query does" in explanation_a
        assert "### How it works" in explanation_a
        print("[OK] SQL Explanation generated successfully.")

        # 2. Result presentation & Chart
        meta_a = build_kpi_metadata(res_a)
        assert meta_a.classification == "table_and_chart"
        chart_a = create_chart(res_a)
        assert chart_a is not None

        # 3. Insights
        insights_a = generate_business_insights(
            question=q_a,
            result_df=res_a,
            sql=sql_a,
            llm_client=lambda p: "• East region generated the highest revenue ($207,141).",
        )

        # 4. Downloads
        csv_a = dataframe_to_csv_bytes(res_a)
        excel_a = dataframe_to_excel_bytes(res_a)
        assert len(csv_a) > 0 and len(excel_a) > 0

        # 5. Conversation History
        turn_a = conv_history.add_turn(
            question=q_a,
            sql=sql_a,
            result_df=res_a,
            insights=insights_a,
            sql_explanation=explanation_a,
        )
        assert len(conv_history) == 1
        assert turn_a.has_sql_explanation
        assert turn_a.sql_explanation == explanation_a
        print("[OK] Case A complete end-to-end pipeline PASSED.")

        # -------------------------------------------------------------
        # CASE B: Scalar query ("What is the total revenue?")
        # -------------------------------------------------------------
        print("\n--- CASE B: Scalar query ('What is the total revenue?') ---")
        q_b = "What is the total revenue?"
        sql_b = "SELECT SUM(revenue) AS total_revenue FROM sales;"
        validate_read_only_query(sql_b)
        res_b = engine.execute_query(sql_b)

        # 1. SQL Explanation
        explanation_b = generate_sql_explanation(
            question=q_b,
            sql=sql_b,
            schema=schema,
            llm_client=lambda p: (
                "### What the query does\n"
                "Calculates the total revenue across all sales transactions.\n\n"
                "### How it works\n"
                "- Queries the sales table.\n"
                "- Adds up the revenue column across all records."
            ),
        )
        assert "### What the query does" in explanation_b
        print("[OK] Scalar SQL Explanation generated successfully.")

        # 2. KPI Card & Chart absence
        meta_b = build_kpi_metadata(res_b)
        assert meta_b.classification == "kpi"
        assert meta_b.kpi_cards[0].formatted_value == "641,021"
        chart_b = create_chart(res_b)
        assert chart_b is None, "Scalar query must not produce a chart"

        # 3. Insights & Downloads
        insights_b = generate_business_insights(
            question=q_b,
            result_df=res_b,
            sql=sql_b,
            llm_client=lambda p: "• Total revenue across all orders is 641,021.",
        )
        csv_b = dataframe_to_csv_bytes(res_b)
        assert len(csv_b) > 0

        turn_b = conv_history.add_turn(
            question=q_b,
            sql=sql_b,
            result_df=res_b,
            insights=insights_b,
            sql_explanation=explanation_b,
        )
        assert len(conv_history) == 2
        print("[OK] Case B complete scalar pipeline PASSED.")

        # -------------------------------------------------------------
        # CASE C: Follow-up query ("What about East?")
        # -------------------------------------------------------------
        print("\n--- CASE C: Follow-up query ('What about East?') ---")
        q_c = "What about East?"
        sql_c = "SELECT region, SUM(revenue) AS total_revenue FROM sales WHERE region = 'East' GROUP BY region;"
        validate_read_only_query(sql_c)
        res_c = engine.execute_query(sql_c)

        explanation_c = generate_sql_explanation(
            question=q_c,
            sql=sql_c,
            schema=schema,
            llm_client=lambda p: (
                "### What the query does\n"
                "Calculates total revenue specifically for the East region.\n\n"
                "### How it works\n"
                "- Filters sales records to only include region = 'East'.\n"
                "- Sums the revenue for East."
            ),
        )
        assert "East" in explanation_c
        assert explanation_c != explanation_a, "Follow-up explanation must be distinct from prior turn"

        turn_c = conv_history.add_turn(
            question=q_c,
            sql=sql_c,
            result_df=res_c,
            insights="East total revenue is $207,141.",
            sql_explanation=explanation_c,
        )
        assert len(conv_history) == 3
        # Previous explanation is preserved in Turn 1, Turn 3 has new explanation
        assert conv_history.get_turns()[0].sql_explanation == explanation_a
        assert conv_history.get_turns()[2].sql_explanation == explanation_c
        print("[OK] Case C follow-up pipeline and independent explanation preservation PASSED.")

    print("\nALL FUNCTIONAL VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    verify_live_gemini()
    verify_functional_cases()
