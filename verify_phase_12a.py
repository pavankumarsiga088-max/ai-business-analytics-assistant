"""
Verification Script for Phase 12A — Data Quality Issue Explorer.

Tests:
1. Clean Dataset (sample_data/sample_sales.csv)
2. Dirty Dataset (sample_data/sample_dirty_data.xlsx)
3. Full AI Analytics Pipeline Regression ("What is the total revenue by region?")
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 stdout on Windows to prevent cp1252 character encode crashes
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

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
from src.data_quality.issue_details import (
    extract_issue_details,
    get_issue_severity_counts,
    QualityIssueDetail,
)
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query
from src.analytics.nl_to_sql import generate_sql
from src.analytics.sql_explanation import generate_sql_explanation
from src.analytics.visualization import create_chart
from src.analytics.insights import generate_business_insights
from src.analytics.downloads import dataframe_to_csv_bytes, dataframe_to_excel_bytes, figure_to_html_bytes
from src.analytics.conversation import ConversationHistory


def run_checks(df: pd.DataFrame) -> list:
    return [
        check_missing_values(df),
        check_duplicate_rows(df),
        check_duplicate_order_ids(df, id_column="order_id"),
        check_whitespace(df),
        check_inconsistent_casing(df),
        check_invalid_dates(df, date_columns=["order_date"]),
        check_numeric_columns(df, numeric_columns=["quantity", "unit_price", "revenue"]),
        check_invalid_quantities(df, quantity_column="quantity"),
        check_outliers(df, numeric_columns=["quantity", "unit_price", "revenue"]),
        check_revenue_consistency(df, quantity_column="quantity", unit_price_column="unit_price", revenue_column="revenue"),
    ]


def verify_clean_dataset():
    print("\n=== 1. VERIFYING CLEAN DATASET (sample_sales.csv) ===")
    csv_path = "sample_data/sample_sales.csv"
    if not os.path.exists(csv_path):
        csv_path = "sample_sales.csv"
    
    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    checks = run_checks(df)
    score_res = calculate_quality_score(checks, total_rows=len(df))
    print(f"Quality Score: {score_res['overall_score']} / 100.0, Status: {score_res['status']}")

    issue_details = extract_issue_details(checks, df=df)
    sev_counts = get_issue_severity_counts(issue_details)
    print(f"Severity Counts: {sev_counts}")

    assert len(issue_details) == 10, "Should have 10 issue details."
    assert sev_counts["HIGH"] == 0, "Clean dataset should have 0 HIGH severity issues."
    assert sev_counts["MEDIUM"] == 0, "Clean dataset should have 0 MEDIUM severity issues."

    # Verify each issue detail attributes
    for iss in issue_details:
        assert isinstance(iss, QualityIssueDetail)
        assert len(iss.sample_records) <= 5, "Sample records must be <= 5 rows."
        assert len(iss.sample_values) <= 5, "Sample values must be <= 5."

    print("[OK] Clean dataset verification PASSED.")


def verify_dirty_dataset():
    print("\n=== 2. VERIFYING DIRTY DATASET (sample_dirty_data.xlsx) ===")
    xlsx_path = "sample_data/sample_dirty_data.xlsx"
    if not os.path.exists(xlsx_path):
        xlsx_path = "sample_dirty_data.xlsx"
    
    df = load_dataset(xlsx_path)
    df_copy = df.copy(deep=True)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    checks = run_checks(df)
    score_res = calculate_quality_score(checks, total_rows=len(df))
    print(f"Quality Score: {score_res['overall_score']} / 100.0, Status: {score_res['status']}")

    issue_details = extract_issue_details(checks, df=df)
    sev_counts = get_issue_severity_counts(issue_details)
    print(f"Severity Counts: {sev_counts}")

    assert sev_counts["HIGH"] > 0, "Dirty dataset must detect HIGH severity issues."
    assert sev_counts["MEDIUM"] > 0, "Dirty dataset must detect MEDIUM severity issues."
    assert sev_counts["LOW"] > 0, "Dirty dataset must detect LOW severity issues."
    assert sev_counts["TOTAL_ISSUES"] > 0, "Total issues must be > 0."

    # Check specific issue details
    for iss in issue_details:
        if iss.has_issues:
            print(f"  [{iss.severity}] {iss.check_name}: {iss.issue_count} issues | Cols: {iss.affected_columns} | Rows: {iss.affected_row_count}")
            assert iss.message, "Explanation message must be present."
            assert iss.recommendation, "Recommendation must be present."
            assert len(iss.sample_records) <= 5, "Sample records must be <= 5."
            assert len(iss.sample_values) <= 5, "Sample values must be <= 5."

    # Verify input DataFrame immutability
    pd.testing.assert_frame_equal(df, df_copy)
    print("[OK] Input DataFrame was not mutated.")
    print("[OK] Dirty dataset verification PASSED.")


def verify_ai_analytics_regression():
    print("\n=== 3. VERIFYING AI ANALYTICS REGRESSION ===")
    csv_path = "sample_data/sample_sales.csv"
    if not os.path.exists(csv_path):
        csv_path = "sample_sales.csv"
    
    df = load_dataset(csv_path)
    engine = DuckDBEngine()
    engine.register_dataframe(df, "sales")

    question = "What is the total revenue by region?"
    print(f"Question: {question}")

    # Generate SQL
    schema = engine.get_schema("sales")
    sql = generate_sql(question, schema=schema, model="gemini-3.5-flash-lite")
    print(f"Generated SQL: {sql.strip()}")
    assert "sales" in sql.lower(), "Generated SQL must reference 'sales'."

    # Validate read-only
    validate_read_only_query(sql)
    print("[OK] SQL validation passed.")

    # Execute
    res_df = engine.execute_query(sql)
    print(f"Query Result: {len(res_df)} rows, columns: {list(res_df.columns)}")
    assert not res_df.empty, "Query result should not be empty."

    # SQL Explanation
    explanation = generate_sql_explanation(question=question, sql=sql, schema=schema, model="gemini-3.5-flash-lite")
    print(f"Explanation Preview: {explanation[:120]}...")
    assert "### What the query does" in explanation, "Must include ### What the query does"

    # Visualization
    chart = create_chart(res_df)
    print(f"Chart generated: {type(chart).__name__}")
    assert chart is not None, "Chart must be generated."

    # Insights
    insights = generate_business_insights(question, res_df, sql=sql, model="gemini-3.5-flash-lite")
    print(f"Insights Preview: {insights[:120]}...")
    assert len(insights) > 0, "Insights must be generated."

    # Downloads
    csv_b = dataframe_to_csv_bytes(res_df)
    excel_b = dataframe_to_excel_bytes(res_df)
    html_b = figure_to_html_bytes(chart)
    print(f"Download Bytes: CSV={len(csv_b)}, Excel={len(excel_b)}, HTML={len(html_b)}")
    assert len(csv_b) > 0 and len(excel_b) > 0 and len(html_b) > 0

    # Conversation history
    history = ConversationHistory(dataset_name="sample_sales.csv")
    history.add_turn(
        question=question,
        sql=sql,
        result_df=res_df,
        insights=insights,
        sql_explanation=explanation,
    )
    assert len(history) == 1, "Turn must be saved in history."
    assert history.get_turns()[0].has_sql_explanation, "Turn must have sql_explanation."

    print("[OK] AI Analytics regression verification PASSED.")


def main():
    print("=" * 60)
    print("PHASE 12A END-TO-END VERIFICATION")
    print("=" * 60)
    verify_clean_dataset()
    verify_dirty_dataset()
    verify_ai_analytics_regression()
    print("\n" + "=" * 60)
    print("ALL PHASE 12A CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
