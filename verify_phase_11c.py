"""
Functional verification script for Phase 11C — Downloadable Analysis Results.

Verifies end-to-end against sample_sales.csv:
Case A: Grouped query ("What is the total revenue by region?")
  - Table
  - Plotly chart
  - CSV download bytes
  - Excel download bytes (re-readable with openpyxl)
  - Standalone Plotly HTML download bytes
  - AI Business Insights
  - Conversation turn recorded

Case B: Scalar query ("What is the total revenue?")
  - Scalar KPI presentation (109,510 / 641,021)
  - Table
  - CSV download bytes
  - Excel download bytes
  - NO chart download (figure is None)
  - AI Business Insights
  - Conversation turn recorded
"""

import io
import openpyxl
import pandas as pd
from src.data_quality.loader import load_dataset
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query
from src.analytics.visualization import create_chart
from src.analytics.insights import generate_business_insights
from src.analytics.conversation import ConversationHistory
from src.analytics.result_presentation import build_kpi_metadata
from src.analytics.downloads import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    figure_to_html_bytes,
    sanitize_filename,
    build_download_filename,
)


def run_verification():
    print("=== Loading sample_sales.csv ===")
    df = load_dataset("sample_data/sample_sales.csv")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    conv_history = ConversationHistory(dataset_name="sample_sales.csv")

    with DuckDBEngine() as engine:
        engine.register_dataframe(df, table_name="sales")

        # -------------------------------------------------------------
        # CASE A: Grouped query ("What is the total revenue by region?")
        # -------------------------------------------------------------
        print("\n--- CASE A: Total revenue by region (Grouped / Chartable) ---")
        sql_a = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;"
        validate_read_only_query(sql_a)
        res_a = engine.execute_query(sql_a)
        print("Result A:\n", res_a)

        # 1. Result presentation & visualization
        meta_a = build_kpi_metadata(res_a)
        assert meta_a.classification == "table_and_chart", f"Expected 'table_and_chart', got {meta_a.classification}"
        chart_a = create_chart(res_a)
        assert chart_a is not None, "Expected Plotly figure for grouped query"

        # 2. CSV Export
        csv_bytes_a = dataframe_to_csv_bytes(res_a)
        assert len(csv_bytes_a) > 0, "CSV bytes must not be empty"
        csv_text_a = csv_bytes_a.decode("utf-8")
        assert "region,total_revenue" in csv_text_a
        assert "East,207141.0" in csv_text_a
        print("[OK] CSV export valid (contains header and rows, excludes index).")

        # 3. Excel Export
        excel_bytes_a = dataframe_to_excel_bytes(res_a)
        assert len(excel_bytes_a) > 0, "Excel bytes must not be empty"
        wb_a = openpyxl.load_workbook(io.BytesIO(excel_bytes_a))
        sheet_a = wb_a.active
        headers_a = [cell.value for cell in sheet_a[1]]
        assert headers_a == ["region", "total_revenue"], f"Unexpected headers: {headers_a}"
        assert sheet_a.max_row == 6, f"Expected 6 rows (1 header + 5 data), got {sheet_a.max_row}"
        print("[OK] Excel export valid (readable by openpyxl with correct schema and rows).")

        # 4. Standalone Plotly HTML Export
        html_bytes_a = figure_to_html_bytes(chart_a)
        assert len(html_bytes_a) > 0, "HTML bytes must not be empty"
        html_text_a = html_bytes_a.decode("utf-8")
        assert "<html" in html_text_a.lower()
        assert "plotly" in html_text_a.lower()
        assert "East" in html_text_a
        print("[OK] Plotly HTML export valid (standalone HTML document).")

        # 5. Filename sanitization
        prefix = sanitize_filename("sample_sales.csv")
        csv_fn = build_download_filename("analysis_result", "csv", prefix=prefix)
        excel_fn = build_download_filename("analysis_result", "xlsx", prefix=prefix)
        html_fn = build_download_filename("chart", "html", prefix=prefix)
        assert csv_fn == "sample_sales_analysis_result.csv"
        assert excel_fn == "sample_sales_analysis_result.xlsx"
        assert html_fn == "sample_sales_chart.html"
        print(f"[OK] Dynamic filenames: {csv_fn}, {excel_fn}, {html_fn}")

        # 6. AI Insights & History
        insights_a = generate_business_insights(
            question="What is the total revenue by region?",
            result_df=res_a,
            sql=sql_a,
            llm_client=lambda prompt: "• East region leads with highest revenue.",
        )
        assert insights_a == "• East region leads with highest revenue."

        turn_a = conv_history.add_turn(
            question="What is the total revenue by region?",
            sql=sql_a,
            result_df=res_a,
            insights=insights_a,
        )
        assert len(conv_history) == 1
        print("[OK] AI Insights and conversation memory preserved. Case A PASSED.")

        # -------------------------------------------------------------
        # CASE B: Scalar query ("What is the total revenue?")
        # -------------------------------------------------------------
        print("\n--- CASE B: Total revenue (Scalar / KPI) ---")
        sql_b = "SELECT SUM(revenue) AS total_revenue FROM sales;"
        validate_read_only_query(sql_b)
        res_b = engine.execute_query(sql_b)
        print("Result B:\n", res_b)

        # 1. KPI presentation & chart absence
        meta_b = build_kpi_metadata(res_b)
        assert meta_b.classification == "kpi", f"Expected 'kpi', got {meta_b.classification}"
        assert len(meta_b.kpi_cards) == 1
        assert meta_b.kpi_cards[0].label == "Total Revenue"
        assert meta_b.kpi_cards[0].formatted_value == "641,021"
        print(f"[OK] KPI Card rendered: {meta_b.kpi_cards[0].label} = {meta_b.kpi_cards[0].formatted_value}")

        chart_b = create_chart(res_b)
        assert chart_b is None, "Expected NO chart figure for scalar query"
        print("[OK] Plotly chart is None (chart download button omitted cleanly).")

        # 2. CSV Export
        csv_bytes_b = dataframe_to_csv_bytes(res_b)
        assert len(csv_bytes_b) > 0
        csv_text_b = csv_bytes_b.decode("utf-8")
        assert "total_revenue" in csv_text_b
        assert "641021.0" in csv_text_b
        print("[OK] CSV export valid for scalar result.")

        # 3. Excel Export
        excel_bytes_b = dataframe_to_excel_bytes(res_b)
        assert len(excel_bytes_b) > 0
        wb_b = openpyxl.load_workbook(io.BytesIO(excel_bytes_b))
        sheet_b = wb_b.active
        assert [cell.value for cell in sheet_b[1]] == ["total_revenue"]
        assert sheet_b.max_row == 2  # 1 header + 1 data row
        print("[OK] Excel export valid for scalar result.")

        # 4. Confirm chart export condition
        show_chart_dl = (chart_b is not None)
        assert show_chart_dl is False, "Chart download MUST not be offered when figure is None"
        print("[OK] Chart download button condition verified (hidden for scalar KPI).")

        # 5. AI Insights & History
        insights_b = generate_business_insights(
            question="What is the total revenue?",
            result_df=res_b,
            sql=sql_b,
            llm_client=lambda prompt: "• Total revenue across all transactions is 641,021.",
        )
        assert insights_b == "• Total revenue across all transactions is 641,021."

        turn_b = conv_history.add_turn(
            question="What is the total revenue?",
            sql=sql_b,
            result_df=res_b,
            insights=insights_b,
        )
        assert len(conv_history) == 2
        print("[OK] AI Insights and conversation memory preserved. Case B PASSED.")

    print("\nALL PHASE 11C FUNCTIONAL VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    run_verification()
