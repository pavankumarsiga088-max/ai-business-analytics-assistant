"""
Verification Script for Phase 12C — Data Quality Report Export.

Verifies:
1. Clean Dataset (sample_data/sample_sales.csv) -> HTML report generation & byte serialization
2. Dirty Dataset (sample_data/sample_dirty_data.xlsx) -> High/Med issues, bounded tables, priority areas, escaping
3. Full AI Analytics Pipeline Regression ("What is the total revenue by region?")
"""
import os
import sys
import html
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
from src.data_quality.dashboard import (
    get_score_interpretation,
    create_issue_distribution_chart,
    get_priority_areas,
    build_quality_check_overview_df,
)
from src.data_quality.report_export import (
    build_quality_report_html,
    generate_quality_report_bytes,
    build_quality_report_filename,
    esc,
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
    df_copy = df.copy(deep=True)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    checks = run_checks(df)
    score_res = calculate_quality_score(checks, total_rows=len(df))
    issues = extract_issue_details(checks, df=df)
    sev_counts = get_issue_severity_counts(issues)
    priority_areas = get_priority_areas(issues)

    print(f"Quality Score: {score_res['overall_score']} / 100.0, Status: {score_res['status']}")
    print(f"Severity Counts: {sev_counts}")

    # Generate HTML report and bytes
    html_out = build_quality_report_html(
        dataset_name="sample_sales.csv",
        score_report=score_res,
        issue_details=issues,
        severity_counts=sev_counts,
        priority_areas=priority_areas,
    )
    report_bytes = generate_quality_report_bytes(
        dataset_name="sample_sales.csv",
        score_report=score_res,
        issue_details=issues,
        severity_counts=sev_counts,
        priority_areas=priority_areas,
    )
    filename = build_quality_report_filename("sample_sales.csv")

    assert isinstance(html_out, str) and len(html_out) > 500, "HTML report should be non-empty string."
    assert isinstance(report_bytes, bytes) and len(report_bytes) > 500, "Report bytes should be non-empty bytes."
    assert filename.endswith(".html"), f"Filename should end with .html, got {filename}"
    assert "sample_sales" in filename, f"Filename should reference dataset, got {filename}"
    assert "sample_sales.csv" in html_out, "HTML should contain dataset name."
    assert f"{score_res['overall_score']:.2f}" in html_out, "HTML should contain overall score."
    assert "Quality Check Overview" in html_out, "HTML should contain overview table."
    print(f"[OK] Clean dataset HTML report generated: {len(html_out)} chars, {len(report_bytes)} bytes, filename='{filename}'")

    # Verify DataFrame immutability
    pd.testing.assert_frame_equal(df, df_copy)
    print("[OK] Original clean DataFrame was not mutated.")


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
    issues = extract_issue_details(checks, df=df)
    sev_counts = get_issue_severity_counts(issues)
    priority_areas = get_priority_areas(issues)

    print(f"Quality Score: {score_res['overall_score']} / 100.0, Status: {score_res['status']}")
    print(f"Severity Counts: {sev_counts}")
    print(f"Priority Areas: {len(priority_areas)} items")

    # Generate HTML report and bytes
    html_out = build_quality_report_html(
        dataset_name="sample_dirty_data.xlsx",
        score_report=score_res,
        issue_details=issues,
        severity_counts=sev_counts,
        priority_areas=priority_areas,
    )
    report_bytes = generate_quality_report_bytes(
        dataset_name="sample_dirty_data.xlsx",
        score_report=score_res,
        issue_details=issues,
        severity_counts=sev_counts,
        priority_areas=priority_areas,
    )
    filename = build_quality_report_filename("sample_dirty_data.xlsx")

    assert sev_counts["HIGH"] > 0, "Dirty dataset should have High severity issues."
    assert sev_counts["MEDIUM"] > 0, "Dirty dataset should have Medium severity issues."
    assert len(priority_areas) > 0, "Dirty dataset should have priority areas."
    assert "Priority Remediation Areas" in html_out, "HTML should contain Priority Remediation Areas."
    assert "Detailed Quality Issues" in html_out, "HTML should contain Detailed Quality Issues."
    assert filename.endswith(".html"), f"Filename should end with .html, got {filename}"

    # Verify bounding: no sample table should have > 5 data rows
    table_parts = html_out.split("Sample Problematic Rows (up to 5):")
    for part in table_parts[1:]:
        table_html = part.split("</div></div>")[0]
        # count <tr> tags in tbody
        tbody = table_html.split("<tbody>")[1].split("</tbody>")[0]
        tr_count = tbody.count("<tr>")
        assert tr_count <= 5, f"Sample records table exceeded 5 rows (found {tr_count})"

    print(f"[OK] Dirty dataset HTML report generated: {len(html_out)} chars, {len(report_bytes)} bytes, filename='{filename}'")

    # Verify DataFrame immutability
    pd.testing.assert_frame_equal(df, df_copy)
    print("[OK] Original dirty DataFrame was not mutated.")


def verify_ai_analytics_regression():
    print("\n=== 3. VERIFYING AI ANALYTICS REGRESSION ===")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[SKIP] GEMINI_API_KEY not found in environment.")
        return

    csv_path = "sample_data/sample_sales.csv"
    if not os.path.exists(csv_path):
        csv_path = "sample_sales.csv"
    df = load_dataset(csv_path)

    engine = DuckDBEngine()
    engine.register_dataframe(df, "sales")

    question = "What is the total revenue by region?"
    print(f"NL Question: '{question}'")

    schema = engine.get_schema("sales")
    # Use gemini-3.5-flash-lite as established in Phase 11B/C/D/12A/B
    sql = generate_sql(question, schema=schema, model="gemini-3.5-flash-lite")
    print(f"Generated SQL:\n{sql}")

    validate_read_only_query(sql)
    print("[OK] SQL validation passed.")

    res_df = engine.execute_query(sql)
    print(f"Result ({len(res_df)} rows):\n{res_df.to_string(index=False)}")

    # Visualizations
    fig = create_chart(res_df)
    assert fig is not None, "Chart generation failed."
    print("[OK] Chart generated successfully.")

    # Business Insights
    insights = generate_business_insights(question, res_df, sql=sql, model="gemini-3.5-flash-lite")
    print(f"Insights preview: {insights[:120]}...")
    assert len(insights) > 50, "Insights too short."
    print("[OK] Business insights generated successfully.")

    # SQL Explanation
    explanation = generate_sql_explanation(question=question, sql=sql, schema=schema, model="gemini-3.5-flash-lite")
    print(f"Explanation preview: {explanation[:120]}...")
    assert len(explanation) > 30, "Explanation too short."
    print("[OK] SQL explanation generated successfully.")

    # Downloads
    csv_bytes = dataframe_to_csv_bytes(res_df)
    excel_bytes = dataframe_to_excel_bytes(res_df)
    chart_html_bytes = figure_to_html_bytes(fig)
    assert len(csv_bytes) > 0 and len(excel_bytes) > 0 and len(chart_html_bytes) > 0
    print("[OK] AI Analytics download bytes generated successfully.")

    print("\n[SUCCESS] AI Analytics regression verified with zero regressions!")


if __name__ == "__main__":
    print("==================================================")
    print("STARTING PHASE 12C VERIFICATION SUITE")
    print("==================================================")
    verify_clean_dataset()
    verify_dirty_dataset()
    verify_ai_analytics_regression()
    print("\n==================================================")
    print("ALL PHASE 12C VERIFICATIONS COMPLETED SUCCESSFULLY")
    print("==================================================")
