"""
Offline Unit Tests for Data Quality Report Export (Phase 12C).

Covers all 21 required conditions:
1. test_report_html_generation_returns_non_empty_string
2. test_report_contains_dataset_name
3. test_report_contains_quality_score_and_status
4. test_report_contains_severity_breakdown
5. test_report_contains_check_overview_table
6. test_report_contains_priority_areas
7. test_report_contains_detailed_issues
8. test_report_bounds_sample_records_to_max_5
9. test_report_bounds_example_values_to_max_5
10. test_report_escapes_html_in_column_names
11. test_report_escapes_html_in_sample_values
12. test_report_escapes_html_in_dataset_name
13. test_report_handles_special_characters_cleanly
14. test_generate_report_bytes_returns_utf8_bytes
15. test_safe_filename_generation_removes_path_traversal
16. test_safe_filename_generation_handles_special_chars
17. test_report_on_clean_dataset
18. test_report_on_dirty_dataset
19. test_report_on_empty_dataset
20. test_report_with_unusual_column_types
21. test_original_dataframe_is_not_mutated
"""
import unittest
import pandas as pd

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
from src.data_quality.scoring import calculate_quality_score
from src.data_quality.issue_details import (
    extract_issue_details,
    get_issue_severity_counts,
    QualityIssueDetail,
)
from src.data_quality.dashboard import get_priority_areas
from src.data_quality.report_export import (
    build_quality_report_html,
    generate_quality_report_bytes,
    build_quality_report_filename,
    esc,
    _render_sample_records_table,
)


class TestPhase12CReportExport(unittest.TestCase):

    def setUp(self):
        # 1. Clean dataset
        self.clean_df = pd.DataFrame({
            "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104"],
            "customer_name": ["Alice", "Bob", "Charlie", "David"],
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "region": ["North", "South", "East", "West"],
            "quantity": [1, 2, 3, 4],
            "unit_price": [50.0, 50.0, 50.0, 50.0],
            "revenue": [50.0, 100.0, 150.0, 200.0],
        })

        # 2. Dirty dataset
        self.dirty_df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-1", "ORD-3", "ORD-4", "ORD-5", "ORD-6", "ORD-7"],
            "customer_name": ["Alice", None, "Charlie", "  David ", "Eve", "Frank", "Grace"],
            "order_date": ["2024-01-01", "bad-date-1", "bad-date-2", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"],
            "region": ["North", "north", "East", "West", "North", "South", "East"],
            "quantity": [-5, 0, 2, 4, 10, 15, 20],
            "unit_price": [10.0, "text_price", 20.0, 25.0, 30.0, 35.0, 40.0],
            "revenue": [50.0, 0.0, 999.0, 100.0, 300.0, 525.0, 800.0],
        })

    def test_1_report_html_generation_returns_non_empty_string(self):
        """1. HTML report generator returns a non-empty string with standard HTML document structure."""
        checks = [check_missing_values(self.clean_df)]
        score_res = calculate_quality_score(checks, total_rows=len(self.clean_df))
        issues = extract_issue_details(checks, df=self.clean_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="sales.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIsInstance(html_out, str)
        self.assertGreater(len(html_out), 100)
        self.assertTrue(html_out.strip().startswith("<!DOCTYPE html>"))
        self.assertTrue(html_out.strip().endswith("</html>"))

    def test_2_report_contains_dataset_name(self):
        """2. The exported HTML report includes the specified dataset name."""
        checks = [check_missing_values(self.clean_df)]
        score_res = calculate_quality_score(checks, total_rows=len(self.clean_df))
        issues = extract_issue_details(checks, df=self.clean_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="my_custom_dataset.xlsx",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIn("my_custom_dataset.xlsx", html_out)

    def test_3_report_contains_quality_score_and_status(self):
        """3. The report prominently contains the overall quality score and status label."""
        checks = [check_missing_values(self.clean_df)]
        score_res = calculate_quality_score(checks, total_rows=len(self.clean_df))
        issues = extract_issue_details(checks, df=self.clean_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="sales.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIn("100.00", html_out)
        self.assertIn("out of 100", html_out)
        self.assertIn("Excellent", html_out)

    def test_4_report_contains_severity_breakdown(self):
        """4. Severity breakdown counts for High, Medium, Low, and Passed are present."""
        checks = [
            check_missing_values(self.dirty_df),
            check_duplicate_order_ids(self.dirty_df, id_column="order_id"),
            check_duplicate_rows(self.dirty_df),
        ]
        score_res = calculate_quality_score(checks, total_rows=len(self.dirty_df))
        issues = extract_issue_details(checks, df=self.dirty_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="dirty_sales.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIn("Severity Summary", html_out)
        self.assertIn("High Severity", html_out)
        self.assertIn("Medium Severity", html_out)
        self.assertIn("Low Severity", html_out)
        self.assertIn("Passed Checks", html_out)

    def test_5_report_contains_check_overview_table(self):
        """5. Quality Check Overview table contains check names, severity, status, and columns."""
        checks = [
            check_missing_values(self.clean_df),
            check_duplicate_rows(self.clean_df),
        ]
        score_res = calculate_quality_score(checks, total_rows=len(self.clean_df))
        issues = extract_issue_details(checks, df=self.clean_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="clean.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIn("Quality Check Overview", html_out)
        self.assertIn("Missing Values", html_out)
        self.assertIn("Duplicate Rows", html_out)

    def test_6_report_contains_priority_areas(self):
        """6. Priority areas for remediation are rendered when quality issues exist."""
        checks = [
            check_missing_values(self.dirty_df),
            check_duplicate_order_ids(self.dirty_df, id_column="order_id"),
        ]
        score_res = calculate_quality_score(checks, total_rows=len(self.dirty_df))
        issues = extract_issue_details(checks, df=self.dirty_df)
        counts = get_issue_severity_counts(issues)
        priority_areas = get_priority_areas(issues)

        html_out = build_quality_report_html(
            dataset_name="dirty.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
            priority_areas=priority_areas,
        )

        self.assertIn("Priority Remediation Areas", html_out)
        self.assertIn("Missing Values", html_out)

    def test_7_report_contains_detailed_issues(self):
        """7. Detailed quality issue cards render descriptions, suggestions, and affected columns."""
        checks = [check_missing_values(self.dirty_df)]
        score_res = calculate_quality_score(checks, total_rows=len(self.dirty_df))
        issues = extract_issue_details(checks, df=self.dirty_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="dirty.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIn("Detailed Quality Issues", html_out)
        self.assertIn("Recommended action", html_out)
        self.assertIn("Affected columns", html_out)

    def test_8_report_bounds_sample_records_to_max_5(self):
        """8. An issue with > 5 sample records renders at most 5 rows in the HTML sample table."""
        large_sample_df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(15)],
            "val": [f"Val_{i}" for i in range(15)],
        })

        table_html = _render_sample_records_table(large_sample_df, max_rows=5)
        # Verify 5 tbody rows (6 <tr> tags total including 1 in <thead>)
        row_count = table_html.count("<tr>") - 1
        self.assertEqual(row_count, 5)
        self.assertIn("ID_4", table_html)
        self.assertNotIn("ID_5", table_html)

    def test_9_report_bounds_example_values_to_max_5(self):
        """9. An issue with > 5 example values only displays at most 5 values plus a truncation badge."""
        detail = QualityIssueDetail(
            check_name="Many Examples Check",
            catalog_key="test_catalog_key",
            status="FAIL",
            severity="MEDIUM",
            category="WARNING",
            message="Many examples message",
            issue_count=10,
            affected_columns=["test_col"],
            affected_row_count=10,
            affected_row_indices=list(range(10)),
            sample_values=[f"Ex_{i}" for i in range(10)],
            sample_records=pd.DataFrame(),
            recommendation="Review values",
        )

        score_res = {"overall_score": 85.0, "status": "Good"}
        counts = {"HIGH": 0, "MEDIUM": 1, "LOW": 0, "PASS": 0, "TOTAL_ISSUES": 1}

        html_out = build_quality_report_html(
            dataset_name="examples.csv",
            score_report=score_res,
            issue_details=[detail],
            severity_counts=counts,
        )

        self.assertIn("Ex_0", html_out)
        self.assertIn("Ex_4", html_out)
        self.assertNotIn("Ex_5", html_out)
        self.assertIn("+5 more", html_out)

    def test_10_report_escapes_html_in_column_names(self):
        """10. Malicious or HTML-like column names are safely escaped against XSS."""
        malicious_col = "<script>alert('xss')</script>"
        detail = QualityIssueDetail(
            check_name="XSS Col Check",
            catalog_key="test_catalog_key",
            status="FAIL",
            severity="HIGH",
            category="ERROR",
            message="XSS test detail",
            issue_count=1,
            affected_columns=[malicious_col],
            affected_row_count=1,
            affected_row_indices=[0],
            sample_values=["safe_val"],
            sample_records=pd.DataFrame(),
            recommendation="sanitize inputs",
        )

        score_res = {"overall_score": 80.0, "status": "Good"}
        counts = {"HIGH": 1, "MEDIUM": 0, "LOW": 0, "PASS": 0, "TOTAL_ISSUES": 1}

        html_out = build_quality_report_html(
            dataset_name="xss_test.csv",
            score_report=score_res,
            issue_details=[detail],
            severity_counts=counts,
        )

        self.assertNotIn("<script>alert('xss')</script>", html_out)
        self.assertIn(esc(malicious_col), html_out)

    def test_11_report_escapes_html_in_sample_values(self):
        """11. Sample values containing HTML or script tags are strictly escaped."""
        xss_val = "<img src=x onerror=alert(1)>"
        df_xss = pd.DataFrame({"col1": [xss_val]})

        detail = QualityIssueDetail(
            check_name="XSS Val Check",
            catalog_key="test_catalog_key",
            status="FAIL",
            severity="HIGH",
            category="ERROR",
            message="XSS val",
            issue_count=1,
            affected_columns=["col1"],
            affected_row_count=1,
            affected_row_indices=[0],
            sample_values=[xss_val],
            sample_records=df_xss,
            recommendation="clean val",
        )

        score_res = {"overall_score": 75.0, "status": "Needs Review"}
        counts = {"HIGH": 1, "MEDIUM": 0, "LOW": 0, "PASS": 0, "TOTAL_ISSUES": 1}

        html_out = build_quality_report_html(
            dataset_name="xss_val.csv",
            score_report=score_res,
            issue_details=[detail],
            severity_counts=counts,
        )

        self.assertNotIn("<img src=x onerror=alert(1)>", html_out)
        self.assertIn(esc(xss_val), html_out)

    def test_12_report_escapes_html_in_dataset_name(self):
        """12. Dataset name containing HTML characters is properly escaped."""
        malicious_name = '"><b onmouseover=alert(1)>sales</b>.csv'
        score_res = {"overall_score": 100.0, "status": "Excellent"}
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "PASS": 1, "TOTAL_ISSUES": 0}

        html_out = build_quality_report_html(
            dataset_name=malicious_name,
            score_report=score_res,
            issue_details=[],
            severity_counts=counts,
        )

        self.assertNotIn('"><b onmouseover=alert(1)>sales</b>.csv', html_out)
        self.assertIn(esc(malicious_name), html_out)

    def test_13_report_handles_special_characters_cleanly(self):
        """13. Unicode, currency symbols, and accents render cleanly."""
        special_text = "Vente € 1,200.50 · 日本 · España — café & résumé"
        detail = QualityIssueDetail(
            check_name="Unicode Check",
            catalog_key="test_catalog_key",
            status="WARNING",
            severity="LOW",
            category="WARNING",
            message=special_text,
            issue_count=1,
            affected_columns=["notes"],
            affected_row_count=1,
            affected_row_indices=[0],
            sample_values=[special_text],
            sample_records=pd.DataFrame({"notes": [special_text]}),
            recommendation=special_text,
        )

        score_res = {"overall_score": 95.0, "status": "Excellent"}
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 1, "PASS": 0, "TOTAL_ISSUES": 1}

        html_out = build_quality_report_html(
            dataset_name="intl_dataset.csv",
            score_report=score_res,
            issue_details=[detail],
            severity_counts=counts,
        )

        self.assertIn("Vente € 1,200.50", html_out)
        self.assertIn("España — café &amp; résumé", html_out)

    def test_14_generate_report_bytes_returns_utf8_bytes(self):
        """14. generate_quality_report_bytes returns bytes decodable as UTF-8 matching build_quality_report_html."""
        score_res = {"overall_score": 100.0, "status": "Excellent"}
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "PASS": 1, "TOTAL_ISSUES": 0}

        report_bytes = generate_quality_report_bytes(
            dataset_name="bytes_test.csv",
            score_report=score_res,
            issue_details=[],
            severity_counts=counts,
            timestamp="2026-09-12 12:00:00 UTC",
        )

        self.assertIsInstance(report_bytes, bytes)
        decoded = report_bytes.decode("utf-8")
        self.assertTrue(decoded.startswith("<!DOCTYPE html>"))
        self.assertIn("bytes_test.csv", decoded)

    def test_15_safe_filename_generation_removes_path_traversal(self):
        """15. Filename sanitization strips path traversal attempts (e.g. ../../)."""
        filename = build_quality_report_filename("../../etc/passwd.csv")
        self.assertNotIn("..", filename)
        self.assertNotIn("/", filename)
        self.assertNotIn("\\", filename)
        self.assertTrue(filename.endswith(".html"))
        self.assertIn("passwd", filename)

    def test_16_safe_filename_generation_handles_special_chars(self):
        """16. Filename generator handles spaces, symbols, and punctuation cleanly."""
        filename = build_quality_report_filename("My Sales Report @ 2024! (v2.0).xlsx")
        self.assertTrue(filename.endswith(".html"))
        self.assertNotIn(" ", filename)
        self.assertNotIn("@", filename)
        self.assertNotIn("!", filename)
        self.assertNotIn("(", filename)
        self.assertNotIn(")", filename)
        self.assertIn("My_Sales_Report", filename)

    def test_17_report_on_clean_dataset(self):
        """17. Report generation on clean dataset yields 100/100 score, passed banner, and no errors."""
        all_checks = [
            check_missing_values(self.clean_df),
            check_duplicate_rows(self.clean_df),
            check_duplicate_order_ids(self.clean_df, id_column="order_id"),
            check_whitespace(self.clean_df),
            check_inconsistent_casing(self.clean_df),
            check_invalid_dates(self.clean_df, date_columns=["order_date"]),
            check_numeric_columns(self.clean_df, numeric_columns=["quantity", "unit_price", "revenue"]),
            check_invalid_quantities(self.clean_df, quantity_column="quantity"),
            check_outliers(self.clean_df, numeric_columns=["revenue"]),
            check_revenue_consistency(self.clean_df),
        ]
        score_res = calculate_quality_score(all_checks, total_rows=len(self.clean_df))
        issues = extract_issue_details(all_checks, df=self.clean_df)
        counts = get_issue_severity_counts(issues)
        priority_areas = get_priority_areas(issues)

        html_out = build_quality_report_html(
            dataset_name="clean_dataset.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
            priority_areas=priority_areas,
        )

        self.assertIn("100.00", html_out)
        self.assertIn("out of 100", html_out)
        self.assertIn("No quality issues detected", html_out)
        self.assertEqual(counts["TOTAL_ISSUES"], 0)

    def test_18_report_on_dirty_dataset(self):
        """18. Report on dirty dataset renders issues across severities and priority areas."""
        all_checks = [
            check_missing_values(self.dirty_df),
            check_duplicate_rows(self.dirty_df),
            check_duplicate_order_ids(self.dirty_df, id_column="order_id"),
            check_whitespace(self.dirty_df),
            check_inconsistent_casing(self.dirty_df),
            check_invalid_dates(self.dirty_df, date_columns=["order_date"]),
            check_numeric_columns(self.dirty_df, numeric_columns=["quantity", "unit_price", "revenue"]),
            check_invalid_quantities(self.dirty_df, quantity_column="quantity"),
            check_outliers(self.dirty_df, numeric_columns=["revenue"]),
            check_revenue_consistency(self.dirty_df),
        ]
        score_res = calculate_quality_score(all_checks, total_rows=len(self.dirty_df))
        issues = extract_issue_details(all_checks, df=self.dirty_df)
        counts = get_issue_severity_counts(issues)
        priority_areas = get_priority_areas(issues)

        html_out = build_quality_report_html(
            dataset_name="dirty_dataset.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
            priority_areas=priority_areas,
        )

        self.assertGreater(counts["TOTAL_ISSUES"], 0)
        self.assertIn("Priority Remediation Areas", html_out)
        self.assertIn("Detailed Quality Issues", html_out)

    def test_19_report_on_empty_dataset(self):
        """19. Report on empty dataset (0 rows) does not crash and displays graceful state."""
        empty_df = pd.DataFrame(columns=["col_a", "col_b"])
        checks = [check_missing_values(empty_df)]
        score_res = calculate_quality_score(checks, total_rows=0)
        issues = extract_issue_details(checks, df=empty_df)
        counts = get_issue_severity_counts(issues)

        html_out = build_quality_report_html(
            dataset_name="empty.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
        )

        self.assertIsInstance(html_out, str)
        self.assertIn("empty.csv", html_out)

    def test_20_report_with_unusual_column_types(self):
        """20. Dataset with datetimes, booleans, and nulls formats cleanly in sample tables."""
        unusual_df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "flag": [True, False],
            "complex_val": [{"key": "val"}, None],
        })
        table_html = _render_sample_records_table(unusual_df, max_rows=5)

        self.assertIn("timestamp", table_html)
        self.assertIn("flag", table_html)
        self.assertIn("True", table_html)
        self.assertIn("<td></td>", table_html)

    def test_21_original_dataframe_is_not_mutated(self):
        """21. Generating the report does not alter or mutate the original input DataFrame."""
        orig_copy = self.dirty_df.copy(deep=True)

        checks = [
            check_missing_values(self.dirty_df),
            check_duplicate_rows(self.dirty_df),
            check_duplicate_order_ids(self.dirty_df, id_column="order_id"),
            check_whitespace(self.dirty_df),
        ]
        score_res = calculate_quality_score(checks, total_rows=len(self.dirty_df))
        issues = extract_issue_details(checks, df=self.dirty_df)
        counts = get_issue_severity_counts(issues)
        priority_areas = get_priority_areas(issues)

        _ = build_quality_report_html(
            dataset_name="mutate_test.csv",
            score_report=score_res,
            issue_details=issues,
            severity_counts=counts,
            priority_areas=priority_areas,
        )

        pd.testing.assert_frame_equal(self.dirty_df, orig_copy)


if __name__ == "__main__":
    unittest.main()
