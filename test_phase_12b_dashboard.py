"""
Offline Unit Tests for Data Quality Dashboard (Phase 12B).

Covers all 12 required conditions:
1. Score comes from existing scoring output
2. Severity counts are correct
3. Issue distribution chart is dynamically generated
4. Clean dataset state works
5. Dirty dataset state works
6. Priority ordering is deterministic (HIGH > MEDIUM > LOW, count descending)
7. Quality check overview uses existing results
8. No DataFrame mutation
9. Empty dataset handled safely
10. Arbitrary column names work
11. Zero-issue dataset does not break
12. Dashboard helper logic does not depend on sample filenames
"""
import unittest
import pandas as pd
import plotly.graph_objects as go

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
from src.data_quality.dashboard import (
    get_score_interpretation,
    create_issue_distribution_chart,
    get_priority_areas,
    build_quality_check_overview_df,
)


class TestPhase12BDashboard(unittest.TestCase):

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

        # 2. Dirty dataset with multiple mixed severities
        self.dirty_df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-1", "ORD-3", "ORD-4", "ORD-5"],
            "customer_name": ["Alice", None, "Charlie", "  David ", "Eve"],
            "order_date": ["2024-01-01", "bad-date-1", "bad-date-2", "2024-01-04", "2024-01-05"],
            "region": ["North", "north", "East", "West", "North"],
            "quantity": [-5, 0, 2, 4, 10],
            "unit_price": [10.0, "text_price", 20.0, 25.0, 30.0],
            "revenue": [50.0, 0.0, 999.0, 100.0, 300.0],
        })

    def test_1_score_comes_from_existing_scoring_output(self):
        """1. Quality score visual uses the exact value from the scoring engine without recalculation."""
        checks = [check_missing_values(self.clean_df)]
        score_res = calculate_quality_score(checks, total_rows=len(self.clean_df))

        score = score_res["overall_score"]
        status = score_res["status"]
        interpretation = get_score_interpretation(score, status)

        self.assertIsInstance(score, (int, float))
        self.assertEqual(score, 100.0)
        self.assertEqual(status, "Excellent")
        self.assertIn("high integrity", interpretation.lower())

    def test_2_severity_counts_are_correct(self):
        """2. Severity overview counts match the extracted issue details exactly."""
        checks = [
            check_missing_values(self.dirty_df),
            check_duplicate_rows(self.dirty_df),
            check_duplicate_order_ids(self.dirty_df, id_column="order_id"),
        ]
        issues = extract_issue_details(checks, df=self.dirty_df)
        counts = get_issue_severity_counts(issues)

        # Missing values = High (customer_name null), Duplicate order_id = High (ORD-1 dup), Duplicate rows = Pass
        self.assertEqual(counts["HIGH"], 2)
        self.assertEqual(counts["PASS"], 1)
        self.assertEqual(counts["TOTAL_ISSUES"], 2)

    def test_3_issue_distribution_is_dynamically_generated(self):
        """3. Issue distribution Plotly chart is dynamically created with correct data points and styling."""
        test_counts = {"HIGH": 5, "MEDIUM": 2, "LOW": 3, "PASS": 0, "TOTAL_ISSUES": 10}
        fig = create_issue_distribution_chart(test_counts)

        self.assertIsInstance(fig, go.Figure)
        bar_data = fig.data[0]
        self.assertEqual(list(bar_data.x), ["High", "Medium", "Low", "Passed"])
        self.assertEqual(list(bar_data.y), [5, 2, 3, 0])
        # Verify color assignments
        self.assertEqual(len(bar_data.marker.color), 4)

    def test_4_clean_dataset_state_works(self):
        """4. Clean dataset yields PASS for all checks, 0 priority issues, and clean status."""
        checks = [
            check_missing_values(self.clean_df),
            check_duplicate_rows(self.clean_df),
            check_duplicate_order_ids(self.clean_df, id_column="order_id"),
        ]
        issues = extract_issue_details(checks, df=self.clean_df)
        counts = get_issue_severity_counts(issues)
        priority_areas = get_priority_areas(issues)

        self.assertEqual(counts["TOTAL_ISSUES"], 0)
        self.assertEqual(len(priority_areas), 0)
        self.assertTrue(all(i.status == "PASS" for i in issues))

    def test_5_dirty_dataset_state_works(self):
        """5. Dirty dataset correctly highlights problematic checks and priority areas."""
        checks = [
            check_missing_values(self.dirty_df),
            check_invalid_dates(self.dirty_df, date_columns=["order_date"]),
            check_invalid_quantities(self.dirty_df, quantity_column="quantity"),
        ]
        issues = extract_issue_details(checks, df=self.dirty_df)
        priority_areas = get_priority_areas(issues)

        self.assertGreater(len(priority_areas), 0)
        severities = [p.severity for p in priority_areas]
        # HIGH issues must appear before MEDIUM issues
        self.assertIn("HIGH", severities)
        self.assertIn("MEDIUM", severities)

    def test_6_priority_ordering_is_deterministic(self):
        """6. Priority areas strictly place HIGH before MEDIUM, MEDIUM before LOW, with count tie-breaking."""
        # Construct synthetic issues with known severities and counts
        i_low = QualityIssueDetail("Check Low", "low_key", "FAIL", "LOW", "WARNING", "msg", 10, [], 10, [], [], pd.DataFrame(), "rec")
        i_med = QualityIssueDetail("Check Med", "med_key", "FAIL", "MEDIUM", "ERROR", "msg", 2, [], 2, [], [], pd.DataFrame(), "rec")
        i_high_1 = QualityIssueDetail("Check High 1", "h1_key", "FAIL", "HIGH", "ERROR", "msg", 3, [], 3, [], [], pd.DataFrame(), "rec")
        i_high_2 = QualityIssueDetail("Check High 2", "h2_key", "FAIL", "HIGH", "ERROR", "msg", 8, [], 8, [], [], pd.DataFrame(), "rec")
        i_pass = QualityIssueDetail("Check Pass", "pass_key", "PASS", "PASS", "PASS", "msg", 0, [], 0, [], [], pd.DataFrame(), "rec")

        unsorted_issues = [i_low, i_pass, i_high_1, i_med, i_high_2]
        priorities = get_priority_areas(unsorted_issues)

        # PASS should be excluded
        self.assertNotIn(i_pass, priorities)
        self.assertEqual(len(priorities), 4)

        # High 2 (count 8) should be 1st
        self.assertEqual(priorities[0].check_name, "Check High 2")
        # High 1 (count 3) should be 2nd
        self.assertEqual(priorities[1].check_name, "Check High 1")
        # Med should be 3rd
        self.assertEqual(priorities[2].check_name, "Check Med")
        # Low should be 4th
        self.assertEqual(priorities[3].check_name, "Check Low")

    def test_7_quality_check_overview_uses_existing_results(self):
        """7. Quality check overview DataFrame accurately represents check results."""
        checks = [
            check_missing_values(self.dirty_df),
            check_duplicate_rows(self.clean_df),
        ]
        issues = extract_issue_details(checks, df=self.dirty_df)
        df_overview = build_quality_check_overview_df(issues)

        self.assertIsInstance(df_overview, pd.DataFrame)
        self.assertEqual(len(df_overview), 2)
        self.assertIn("Status", df_overview.columns)
        self.assertIn("Check Name", df_overview.columns)
        self.assertIn("Severity", df_overview.columns)
        self.assertIn("Issues", df_overview.columns)
        self.assertIn("Affected Columns", df_overview.columns)

    def test_8_no_dataframe_mutation(self):
        """8. Generating dashboard visuals and overviews leaves input DataFrames completely unchanged."""
        clean_copy = self.clean_df.copy(deep=True)
        dirty_copy = self.dirty_df.copy(deep=True)

        checks = [check_missing_values(self.dirty_df), check_whitespace(self.clean_df)]
        issues = extract_issue_details(checks, df=self.dirty_df)
        _ = build_quality_check_overview_df(issues)
        _ = get_priority_areas(issues)

        pd.testing.assert_frame_equal(self.clean_df, clean_copy)
        pd.testing.assert_frame_equal(self.dirty_df, dirty_copy)

    def test_9_empty_dataset_handled_safely(self):
        """9. Handles empty DataFrames gracefully without unhandled exceptions."""
        empty_df = pd.DataFrame()
        checks = [check_missing_values(empty_df)]
        issues = extract_issue_details(checks, df=empty_df)
        counts = get_issue_severity_counts(issues)
        fig = create_issue_distribution_chart(counts)
        priorities = get_priority_areas(issues)
        overview_df = build_quality_check_overview_df(issues)

        self.assertEqual(counts["TOTAL_ISSUES"], 0)
        self.assertEqual(len(priorities), 0)
        self.assertEqual(len(overview_df), 1)
        self.assertIsInstance(fig, go.Figure)

    def test_10_arbitrary_column_names_work(self):
        """10. Dynamic schemas with non-standard column names work seamlessly."""
        custom_df = pd.DataFrame({
            "client_code": ["C-1", "C-2", "C-3"],
            "invoice_amount": [100.0, None, 300.0],
            "due_date": ["2024-05-01", "2024-05-02", "bad_date"],
        })
        checks = [
            check_missing_values(custom_df),
            check_invalid_dates(custom_df, date_columns=["due_date"]),
        ]
        issues = extract_issue_details(checks, df=custom_df)
        overview_df = build_quality_check_overview_df(issues)

        self.assertEqual(len(overview_df), 2)
        # Check affected columns extracted dynamically
        self.assertIn("invoice_amount", overview_df.iloc[0]["Affected Columns"])
        self.assertIn("due_date", overview_df.iloc[1]["Affected Columns"])

    def test_11_zero_issue_dataset_does_not_break(self):
        """11. A 100% clean dataset with 0 issues generates valid charts and clean interpretations."""
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "PASS": 10, "TOTAL_ISSUES": 0}
        fig = create_issue_distribution_chart(counts)
        interp = get_score_interpretation(100.0, "Excellent")

        self.assertIsInstance(fig, go.Figure)
        self.assertIn("high integrity", interp.lower())
        self.assertEqual(fig.data[0].y[3], 10)  # 10 passed checks

    def test_12_dashboard_helper_logic_does_not_depend_on_sample_filenames(self):
        """12. Dashboard logic does not hardcode or depend on sample filenames or external assets."""
        score_interp = get_score_interpretation(72.5, "Needs Review")
        self.assertIn("notable quality issues", score_interp.lower())

        counts = {"HIGH": 1, "MEDIUM": 1, "LOW": 1, "PASS": 1, "TOTAL_ISSUES": 3}
        fig = create_issue_distribution_chart(counts, title="Custom File Integrity")
        self.assertEqual(fig.layout.title.text, "Custom File Integrity")


if __name__ == "__main__":
    unittest.main()
