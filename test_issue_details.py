"""
Offline Unit Tests for Data Quality Issue Explorer (Phase 12A).

Verifies that QualityIssueDetail objects accurately translate check results
into structured, bounded, non-destructive, and actionable insights.
"""
import unittest
import pandas as pd
import numpy as np

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
from src.data_quality.scoring import CHECK_CATALOG
from src.data_quality.issue_details import (
    QualityIssueDetail,
    build_issue_detail,
    extract_issue_details,
    get_issue_severity_counts,
)


class TestDataQualityIssueDetails(unittest.TestCase):

    def setUp(self):
        # Clean sample dataframe
        self.clean_df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
            "customer_name": ["Alice", "Bob", "Charlie", "David"],
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "region": ["North", "South", "East", "West"],
            "quantity": [10, 20, 30, 40],
            "unit_price": [100.0, 100.0, 100.0, 100.0],
            "revenue": [1000.0, 2000.0, 3000.0, 4000.0],
        })

        # Synthetic dirty dataframe with controlled anomalies
        self.dirty_df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-2", "ORD-2", "ORD-4", "ORD-5", "ORD-6", "ORD-7", "ORD-8"],
            "customer_name": ["Alice", None, "Bob", "Charlie", "David", " Eve ", "Frank", "Grace"],
            "order_date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"],
            "region": ["North", "north", "South", "East", None, "West", "North", "South"],
            "quantity": [10, 5, -2, 0, 12, 15, 18, 20],
            "unit_price": [100.0, "INR 500", 50.0, 75.0, 100.0, 100.0, 100.0, 100.0],
            "revenue": [1000.0, 2500.0, 99999.0, 0.0, 1200.0, 1500.0, 1800.0, 2000.0],
        })

    def test_1_clean_dataset_produces_pass_details(self):
        """1. Clean dataset produces PASS status and severity for all checks without anomalies."""
        check_res = check_missing_values(self.clean_df)
        detail = build_issue_detail(check_res, df=self.clean_df)

        self.assertEqual(detail.status, "PASS")
        self.assertEqual(detail.severity, "PASS")
        self.assertEqual(detail.issue_count, 0)
        self.assertEqual(detail.affected_columns, [])
        self.assertEqual(detail.affected_row_count, 0)
        self.assertEqual(detail.sample_values, [])
        self.assertTrue(detail.sample_records.empty)
        self.assertFalse(detail.has_issues)

    def test_2_missing_values_produce_correct_issue_count(self):
        """2. Missing values check identifies exact issue count."""
        check_res = check_missing_values(self.dirty_df)
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertTrue(detail.has_issues)
        # customer_name has 1 missing, region has 1 missing -> total 2
        self.assertEqual(detail.issue_count, 2)
        self.assertEqual(detail.severity, "HIGH")

    def test_3_missing_values_identify_affected_columns(self):
        """3. Missing values check accurately identifies which columns have nulls."""
        check_res = check_missing_values(self.dirty_df)
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertIn("customer_name", detail.affected_columns)
        self.assertIn("region", detail.affected_columns)
        self.assertNotIn("quantity", detail.affected_columns)

    def test_4_duplicate_ids_produce_correct_details(self):
        """4. Duplicate primary key check identifies the duplicate ID and affected column."""
        check_res = check_duplicate_order_ids(self.dirty_df, id_column="order_id")
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertTrue(detail.has_issues)
        self.assertEqual(detail.severity, "HIGH")
        self.assertIn("order_id", detail.affected_columns)
        self.assertIn("ORD-2", detail.sample_values)
        self.assertGreater(detail.affected_row_count, 0)

    def test_5_invalid_dates_identify_date_columns(self):
        """5. Invalid dates check flags the date column and bad values."""
        check_res = check_invalid_dates(self.dirty_df, date_columns=["order_date"])
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertTrue(detail.has_issues)
        self.assertEqual(detail.severity, "HIGH")
        self.assertIn("order_date", detail.affected_columns)
        self.assertTrue(any("not-a-date" in str(v) for v in detail.sample_values))

    def test_6_numeric_type_problems_identify_numeric_columns(self):
        """6. Numeric type problems flags strings in numeric columns."""
        check_res = check_numeric_columns(self.dirty_df, numeric_columns=["unit_price"])
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertTrue(detail.has_issues)
        self.assertEqual(detail.severity, "HIGH")
        self.assertIn("unit_price", detail.affected_columns)
        self.assertTrue(any("INR 500" in str(v) for v in detail.sample_values))

    def test_7_invalid_quantities_identify_quantity_column(self):
        """7. Invalid quantities flags negative and zero quantities in quantity column."""
        check_res = check_invalid_quantities(self.dirty_df, quantity_column="quantity")
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertTrue(detail.has_issues)
        self.assertEqual(detail.severity, "MEDIUM")
        self.assertIn("quantity", detail.affected_columns)
        self.assertEqual(detail.issue_count, 2)  # -2 and 0
        self.assertTrue(any("negative: -2" in str(v) for v in detail.sample_values))

    def test_8_revenue_consistency_identifies_relevant_columns(self):
        """8. Revenue consistency identifies quantity, unit_price, revenue discrepancy."""
        check_res = check_revenue_consistency(
            self.dirty_df,
            quantity_column="quantity",
            unit_price_column="unit_price",
            revenue_column="revenue",
        )
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertTrue(detail.has_issues)
        self.assertEqual(detail.severity, "HIGH")
        for col in ["quantity", "unit_price", "revenue"]:
            self.assertIn(col, detail.affected_columns)
        self.assertGreater(len(detail.sample_values), 0)

    def test_9_outlier_details_are_low_severity(self):
        """9. Outlier detection is categorized as LOW severity warning."""
        # Create dataset with an outlier
        df_outliers = pd.DataFrame({
            "quantity": [10, 11, 12, 10, 11, 12, 10, 11, 12, 1000]
        })
        check_res = check_outliers(df_outliers, numeric_columns=["quantity"])
        detail = build_issue_detail(check_res, df=df_outliers)

        self.assertTrue(detail.has_issues)
        self.assertEqual(detail.severity, "LOW")
        self.assertEqual(detail.category, "WARNING")
        self.assertIn("quantity", detail.affected_columns)

    def test_10_issue_samples_are_limited(self):
        """10. Sample values and sample records are strictly bounded to at most 5 items."""
        large_dirty = pd.DataFrame({
            "order_date": [f"invalid_date_{i}" for i in range(20)]
        })
        check_res = check_invalid_dates(large_dirty, date_columns=["order_date"])
        detail = build_issue_detail(check_res, df=large_dirty, sample_limit=5)

        self.assertLessEqual(len(detail.sample_values), 5)
        self.assertLessEqual(len(detail.sample_records), 5)

    def test_11_issue_detail_objects_do_not_contain_full_dataframe(self):
        """11. QualityIssueDetail does not store the complete input DataFrame."""
        large_df = pd.DataFrame({
            "order_id": [f"ORD-{i}" for i in range(100)],
            "val": [None] * 100
        })
        check_res = check_missing_values(large_df)
        detail = build_issue_detail(check_res, df=large_df)

        # sample_records should only have 5 rows, not all 100
        self.assertEqual(len(detail.sample_records), 5)
        self.assertNotEqual(len(detail.sample_records), len(large_df))

    def test_12_input_dataframe_is_not_mutated(self):
        """12. Building issue details leaves input DataFrame completely untouched."""
        df_copy = self.dirty_df.copy(deep=True)
        check_res = check_missing_values(self.dirty_df)
        _ = build_issue_detail(check_res, df=self.dirty_df)

        pd.testing.assert_frame_equal(self.dirty_df, df_copy)

    def test_13_empty_dataset_handled_safely(self):
        """13. Handles empty DataFrames without exceptions or crashes."""
        empty_df = pd.DataFrame()
        check_res = check_missing_values(empty_df)
        detail = build_issue_detail(check_res, df=empty_df)

        self.assertEqual(detail.status, "PASS")
        self.assertEqual(detail.severity, "PASS")
        self.assertEqual(detail.issue_count, 0)
        self.assertTrue(detail.sample_records.empty)

    def test_14_dynamic_column_names_work_with_non_sample_datasets(self):
        """14. Works on arbitrary non-e-commerce schema column names."""
        custom_df = pd.DataFrame({
            "patient_uuid": ["P-1", "P-1"],  # Duplicate ID
            "blood_pressure_sys": [120, None],  # Missing
            "visit_timestamp": ["2023-05-01", "not_a_time"],  # Invalid date
        })
        res_dup = check_duplicate_order_ids(custom_df, id_column="patient_uuid")
        detail_dup = build_issue_detail(res_dup, df=custom_df)
        self.assertIn("patient_uuid", detail_dup.affected_columns)

        res_miss = check_missing_values(custom_df)
        detail_miss = build_issue_detail(res_miss, df=custom_df)
        self.assertIn("blood_pressure_sys", detail_miss.affected_columns)

        res_date = check_invalid_dates(custom_df, date_columns=["visit_timestamp"])
        detail_date = build_issue_detail(res_date, df=custom_df)
        self.assertIn("visit_timestamp", detail_date.affected_columns)

    def test_15_recommendations_are_present(self):
        """15. Every issue detail provides a non-empty, actionable recommendation."""
        check_res = check_missing_values(self.dirty_df)
        detail = build_issue_detail(check_res, df=self.dirty_df)

        self.assertIsInstance(detail.recommendation, str)
        self.assertGreater(len(detail.recommendation.strip()), 10)

    def test_16_explanations_are_present(self):
        """16. Every issue detail provides a clear business explanation message."""
        check_res = check_whitespace(pd.DataFrame({"cat": [" A ", "B"]}))
        detail = build_issue_detail(check_res)

        self.assertIsInstance(detail.message, str)
        self.assertIn("whitespace", detail.message.lower())

    def test_17_severity_mapping_is_consistent_with_scoring(self):
        """17. Severity mapping aligns directly with CHECK_CATALOG severity levels."""
        # High severity checks
        for key in ["missing_values", "duplicate_order_ids", "invalid_dates", "numeric_columns", "revenue_consistency"]:
            catalog_entry = CHECK_CATALOG[key]
            self.assertEqual(catalog_entry["severity"], "HIGH")

        # Medium severity checks
        for key in ["duplicate_rows", "invalid_quantities"]:
            catalog_entry = CHECK_CATALOG[key]
            self.assertEqual(catalog_entry["severity"], "MEDIUM")

        # Low severity checks
        for key in ["whitespace", "inconsistent_casing", "outliers"]:
            catalog_entry = CHECK_CATALOG[key]
            self.assertEqual(catalog_entry["severity"], "LOW")

    def test_18_extract_issue_details_and_severity_counts(self):
        """18. Helper functions extract_issue_details and get_issue_severity_counts aggregate correctly."""
        checks = [
            check_missing_values(self.clean_df),
            check_duplicate_rows(self.clean_df),
        ]
        details = extract_issue_details(checks, df=self.clean_df)
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0].severity, "PASS")
        self.assertEqual(details[1].severity, "PASS")

        counts = get_issue_severity_counts(details)
        self.assertEqual(counts["PASS"], 2)
        self.assertEqual(counts["HIGH"], 0)
        self.assertEqual(counts["TOTAL_ISSUES"], 0)


if __name__ == "__main__":
    unittest.main()
