"""
Offline functional verification script for Phase 11B.
Tests:
Case A: "What is the total revenue by region?" (Grouped result -> table_and_chart, Plotly chart, insights, history)
Case B: "What is the total revenue in January 2024?" (Scalar aggregate -> kpi, ~109,510, no chart warning, insights)
Case C: "What are the total revenue and average order value?" (Multi-metric aggregate -> multi_kpi, multiple cards, table, insights)
"""
import pandas as pd
from src.data_quality.loader import load_dataset
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query
from src.analytics.visualization import create_chart
from src.analytics.insights import generate_business_insights
from src.analytics.conversation import ConversationHistory
from src.analytics.result_presentation import (
    build_kpi_metadata,
    classify_result_shape,
    identify_kpi_columns,
)

def run_verification():
    print("=== Loading sample_sales.csv ===")
    df = load_dataset("sample_data/sample_sales.csv")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    conv_history = ConversationHistory(dataset_name="sample_sales.csv")

    with DuckDBEngine() as engine:
        engine.register_dataframe(df, table_name="sales")

        # -------------------------------------------------------------
        # Case A: "What is the total revenue by region?"
        # -------------------------------------------------------------
        print("\n--- CASE A: Total revenue by region ---")
        sql_a = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;"
        validate_read_only_query(sql_a)
        res_a = engine.execute_query(sql_a)
        print("Result A:\n", res_a)

        meta_a = build_kpi_metadata(res_a)
        print(f"Classification A: {meta_a.classification}")
        assert meta_a.classification == "table_and_chart", f"Expected 'table_and_chart', got {meta_a.classification}"
        assert meta_a.show_chart is True
        assert meta_a.show_table is True
        assert len(meta_a.kpi_cards) == 0

        chart_a = create_chart(res_a)
        assert chart_a is not None, "Expected Plotly chart figure for grouped result"
        print("Plotly chart generated successfully.")

        mock_insights_a = "• East generated the highest revenue ($207,141).\n• South followed with $159,251."
        insights_a = generate_business_insights(
            question="What is the total revenue by region?",
            result_df=res_a,
            sql=sql_a,
            llm_client=lambda prompt: mock_insights_a,
        )
        assert insights_a == mock_insights_a
        print("AI Insights generated successfully.")

        turn_a = conv_history.add_turn(
            question="What is the total revenue by region?",
            sql=sql_a,
            result_df=res_a,
            insights=insights_a,
        )
        assert len(conv_history) == 1
        print("Conversation turn recorded. Case A PASSED.")

        # -------------------------------------------------------------
        # Case B: "What is the total revenue in January 2024?"
        # -------------------------------------------------------------
        print("\n--- CASE B: Total revenue in January 2024 ---")
        sql_b = "SELECT SUM(revenue) AS total_revenue FROM sales WHERE order_date LIKE '2024-01%';"
        validate_read_only_query(sql_b)
        res_b = engine.execute_query(sql_b)
        print("Result B:\n", res_b)

        meta_b = build_kpi_metadata(res_b)
        print(f"Classification B: {meta_b.classification}")
        assert meta_b.classification == "kpi", f"Expected 'kpi', got {meta_b.classification}"
        assert meta_b.show_chart is False
        assert meta_b.show_table is True
        assert len(meta_b.kpi_cards) == 1

        card_b = meta_b.kpi_cards[0]
        print(f"KPI Card: Label='{card_b.label}', Value='{card_b.formatted_value}'")
        assert card_b.label == "Total Revenue"
        assert card_b.formatted_value == "109,510", f"Expected '109,510', got {card_b.formatted_value}"

        chart_b = create_chart(res_b)
        assert chart_b is None, "Expected NO chart for scalar result"
        print("Chart is None (no confusing chart warning displayed).")

        mock_insights_b = "• January 2024 generated total revenue of 109,510 across all transactions."
        insights_b = generate_business_insights(
            question="What is the total revenue in January 2024?",
            result_df=res_b,
            sql=sql_b,
            llm_client=lambda prompt: mock_insights_b,
        )
        assert insights_b == mock_insights_b
        print("AI Insights generated successfully.")

        turn_b = conv_history.add_turn(
            question="What is the total revenue in January 2024?",
            sql=sql_b,
            result_df=res_b,
            insights=insights_b,
        )
        assert len(conv_history) == 2
        print("Conversation turn recorded. Case B PASSED.")

        # -------------------------------------------------------------
        # Case C: "What are the total revenue and average order value?"
        # -------------------------------------------------------------
        print("\n--- CASE C: Total revenue and average order value ---")
        sql_c = "SELECT SUM(revenue) AS total_revenue, AVG(revenue) AS avg_order_value, COUNT(*) AS total_orders FROM sales;"
        validate_read_only_query(sql_c)
        res_c = engine.execute_query(sql_c)
        print("Result C:\n", res_c)

        meta_c = build_kpi_metadata(res_c)
        print(f"Classification C: {meta_c.classification}")
        assert meta_c.classification == "multi_kpi", f"Expected 'multi_kpi', got {meta_c.classification}"
        assert meta_c.show_chart is False
        assert meta_c.show_table is True
        assert len(meta_c.kpi_cards) == 3

        for card in meta_c.kpi_cards:
            print(f"  KPI Card: Label='{card.label}', Value='{card.formatted_value}'")

        assert meta_c.kpi_cards[0].label == "Total Revenue"
        assert meta_c.kpi_cards[0].formatted_value == "641,021"
        assert meta_c.kpi_cards[1].label == "Avg Order Value"
        assert meta_c.kpi_cards[1].formatted_value == "6,410.21"
        assert meta_c.kpi_cards[2].label == "Total Orders"
        assert meta_c.kpi_cards[2].formatted_value == "100"

        mock_insights_c = "• Total business revenue stands at 641,021 across 100 orders, averaging 6,410.21 per order."
        insights_c = generate_business_insights(
            question="What are the total revenue and average order value?",
            result_df=res_c,
            sql=sql_c,
            llm_client=lambda prompt: mock_insights_c,
        )
        assert insights_c == mock_insights_c
        print("AI Insights generated successfully.")

        turn_c = conv_history.add_turn(
            question="What are the total revenue and average order value?",
            sql=sql_c,
            result_df=res_c,
            insights=insights_c,
        )
        assert len(conv_history) == 3
        print("Conversation turn recorded. Case C PASSED.")

    print("\nALL FUNCTIONAL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
