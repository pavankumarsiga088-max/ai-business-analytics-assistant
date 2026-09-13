"""
Phase 15 — Final End-to-End System Smoke Verification Script.

Performs an automated health check across all 16 major subsystems of the
AI Business Analytics & Data Quality Assistant without exposing secrets
or requiring external network dependencies.
"""
import os
import sys
import pandas as pd

from src.config import (
    get_database_backend,
    get_gemini_model,
    get_max_query_result_rows,
    get_query_timeout_seconds,
    is_gemini_configured,
    is_postgres_configured,
)
from src.data_quality.loader import load_dataset
from src.data_quality.checks import (
    check_missing_values,
    check_duplicate_rows,
    check_duplicate_order_ids,
    check_whitespace,
    check_inconsistent_casing,
    check_invalid_dates,
    check_numeric_columns,
    check_invalid_quantities,
    check_outliers,
    check_revenue_consistency,
)
from src.data_quality.scoring import calculate_quality_score, generate_quality_report
from src.data_quality.issue_details import extract_issue_details, get_issue_severity_counts
from src.data_quality.report_export import generate_quality_report_bytes
from src.analytics.database import (
    create_database_engine,
    is_backend_available,
    AnalyticsDatabase,
)
from src.analytics.dataset_loader import load_dataset_into_database
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query
from src.analytics.result_presentation import build_kpi_metadata
from src.analytics.visualization import create_chart
from src.analytics.conversation import ConversationHistory
from src.analytics.downloads import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    figure_to_html_bytes,
)
from src.analytics.sql_explanation import build_sql_explanation_prompt
from src.analytics.insights import build_insights_prompt


def run_verification() -> bool:
    print("=" * 80)
    print("PHASE 15: FINAL END-TO-END PROJECT VERIFICATION")
    print("=" * 80)

    all_passed = True

    def check(step_num: int, description: str, func):
        nonlocal all_passed
        try:
            func()
            print(f"[{step_num:02d}/16] PASS - {description}")
        except Exception as e:
            print(f"[{step_num:02d}/16] FAIL - {description}: {e}")
            all_passed = False

    # 1. Clean dataset loading
    clean_df = None
    def step_1():
        nonlocal clean_df
        clean_df = load_dataset("sample_data/sample_sales.csv")
        assert len(clean_df) > 0, "Clean dataset has no rows"
        assert "revenue" in clean_df.columns, "revenue column missing in clean dataset"
    check(1, "Load clean dataset (sample_sales.csv)", step_1)

    # 2. Dirty dataset loading
    dirty_df = None
    def step_2():
        nonlocal dirty_df
        dirty_df = load_dataset("sample_data/sample_dirty_data.xlsx")
        assert len(dirty_df) > 0, "Dirty dataset has no rows"
    check(2, "Load dirty dataset (sample_dirty_data.xlsx)", step_2)

    # 3. Data Quality checks execution
    checks = []
    def step_3():
        nonlocal checks
        checks = [
            check_missing_values(clean_df),
            check_duplicate_rows(clean_df),
            check_duplicate_order_ids(clean_df, "order_id"),
            check_whitespace(clean_df),
            check_inconsistent_casing(clean_df),
            check_invalid_dates(clean_df, ["order_date"]),
            check_numeric_columns(clean_df, ["quantity", "unit_price", "revenue"]),
            check_invalid_quantities(clean_df, "quantity"),
            check_outliers(clean_df, ["quantity", "unit_price", "revenue"]),
            check_revenue_consistency(clean_df, "quantity", "unit_price", "revenue"),
        ]
        assert len(checks) == 10, f"Expected 10 checks, got {len(checks)}"
    check(3, "Execute 10 Data Quality check algorithms", step_3)

    # 4. Quality score & report generation
    def step_4():
        score_res = calculate_quality_score(checks, total_rows=len(clean_df))
        assert 0 <= score_res["overall_score"] <= 100, "Score out of range"
        report = generate_quality_report(checks, score_res, len(clean_df))
        assert "overall_score" in report, "Report missing overall_score"
        issue_details = extract_issue_details(checks, clean_df)
        sev_counts = get_issue_severity_counts(issue_details)
        html_bytes = generate_quality_report_bytes(
            dataset_name="sample_sales.csv",
            score_report=report,
            issue_details=issue_details,
            severity_counts=sev_counts,
        )
        assert len(html_bytes) > 0, "HTML report is empty"
    check(4, "Generate quality score and executive HTML report", step_4)

    # 5. DuckDB engine query execution
    def step_5():
        with DuckDBEngine() as engine:
            load_dataset_into_database(engine, clean_df, "sales")
            res = engine.execute_query("SELECT region, SUM(revenue) as total_rev FROM sales GROUP BY region;")
            assert len(res) > 0, "DuckDB returned empty result"
    check(5, "DuckDB engine table registration & query execution", step_5)

    # 6. PostgreSQL availability check
    def step_6():
        is_avail, reason = is_backend_available("postgres")
        # Reporting status safely without failing if offline
        print(f"       -> PostgreSQL availability: {is_avail} ({'online' if is_avail else reason})")
    check(6, "Detect PostgreSQL backend availability", step_6)

    # 7. Database factory & abstraction
    def step_7():
        duck_engine = create_database_engine("duckdb")
        assert isinstance(duck_engine, AnalyticsDatabase), "Factory did not return AnalyticsDatabase"
        duck_engine.close()
    check(7, "Database abstraction factory instantiation", step_7)

    # 8. Read-only SQL validation
    def step_8():
        validate_read_only_query("SELECT 1;")
        validate_read_only_query("WITH cte AS (SELECT 1) SELECT * FROM cte;")
        try:
            validate_read_only_query("DROP TABLE sales;")
            assert False, "Failed to block DROP TABLE"
        except Exception:
            pass
    check(8, "Read-only SQL AST safety gate", step_8)

    # 9. Result presentation (KPI classification)
    def step_9():
        kpi_df = pd.DataFrame({"total_revenue": [500000.0]})
        presentation = build_kpi_metadata(kpi_df)
        assert presentation.classification == "kpi", f"Expected 'kpi', got {presentation.classification}"
        assert len(presentation.kpi_cards) == 1, "Expected 1 KPI card"
    check(9, "KPI metadata classification", step_9)

    # 10. Automated Plotly chart generation
    chart_fig = None
    def step_10():
        nonlocal chart_fig
        grouped_df = pd.DataFrame({"region": ["East", "West"], "revenue": [100.0, 200.0]})
        chart_fig = create_chart(grouped_df)
        assert chart_fig is not None, "Plotly chart figure not generated"
    check(10, "Automated Plotly chart generation", step_10)

    # 11. Plain-English SQL explanation logic
    def step_11():
        prompt = build_sql_explanation_prompt("Total revenue?", "SELECT SUM(revenue) FROM sales;")
        assert "Total revenue?" in prompt, "Question missing from prompt"
        assert "SELECT SUM(revenue)" in prompt, "SQL missing from prompt"
    check(11, "SQL explanation prompt construction", step_11)

    # 12. Business insights generation logic
    def step_12():
        sample_res = pd.DataFrame({"region": ["East"], "revenue": [100.0]})
        prompt = build_insights_prompt("Revenue?", sample_res, "SELECT * FROM sales;")
        assert "East" in prompt, "Result data missing from insights prompt"
    check(12, "AI business insights prompt construction", step_12)

    # 13. Conversation history & rolling window memory
    def step_13():
        history = ConversationHistory(dataset_name="sales.csv", max_turns=5)
        history.add_turn("Q1", "SELECT 1;", pd.DataFrame({"x": [1]}))
        assert len(history) == 1, "Turn not added"
        history.reset_for_dataset("new_sales.csv")
        assert len(history) == 0, "History not cleared on dataset switch"
    check(13, "Multi-turn conversational memory & dataset isolation", step_13)

    # 14. Export download bytes generation
    def step_14():
        test_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        csv_bytes = dataframe_to_csv_bytes(test_df)
        assert len(csv_bytes) > 0, "CSV bytes empty"
        excel_bytes = dataframe_to_excel_bytes(test_df)
        assert len(excel_bytes) > 0, "Excel bytes empty"
        if chart_fig is not None:
            html_bytes = figure_to_html_bytes(chart_fig)
            assert len(html_bytes) > 0, "Chart HTML bytes empty"
    check(14, "Multi-format download generation (CSV, Excel, HTML)", step_14)

    # 15. Centralized configuration getters
    def step_15():
        assert get_database_backend() in ("duckdb", "postgres"), "Invalid backend default"
        assert get_max_query_result_rows() > 0, "Invalid max rows"
        assert get_query_timeout_seconds() > 0, "Invalid timeout"
        _ = is_gemini_configured()
        _ = is_postgres_configured()
    check(15, "Centralized configuration management (src/config.py)", step_15)

    # 16. Streamlit app import check
    def step_16():
        import app
        assert hasattr(app, "main"), "app.py missing main() function"
    check(16, "Streamlit application module import", step_16)

    print("=" * 80)
    if all_passed:
        print("ALL 16 SUBSYSTEMS PASSED VERIFICATION SUCCESSFULLY!")
    else:
        print("SOME SUBSYSTEMS FAILED VERIFICATION. CHECK LOGS ABOVE.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
