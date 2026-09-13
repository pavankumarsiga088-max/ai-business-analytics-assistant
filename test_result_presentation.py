"""
Unit Tests for Result Presentation Module (Phase 11B).

Validates:
1. Empty DataFrame -> "empty"
2. 1x1 numeric scalar -> "kpi"
3. 1x1 text scalar -> non-KPI ("table")
4. 1x3 numeric result -> "multi_kpi"
5. Grouped categorical + numeric result -> "table_and_chart"
6. Table-only result -> "table"
7. Identifier columns are not incorrectly treated as KPIs
8. KPI labels convert correctly (snake_case to Title Case)
9. Integer formatting (commas, large values)
10. Float formatting (two decimal places, fractional values)
11. Negative values formatting
12. Null and NaN values formatting
13. Original DataFrame is not mutated
14. Multiple KPI limit behaves correctly (capping displayed cards)
"""

import unittest
import numpy as np
import pandas as pd

from src.analytics.result_presentation import (
    classify_result_shape,
    identify_kpi_columns,
    format_kpi_label,
    format_kpi_value,
    build_kpi_metadata,
    is_identifier_or_temporal_column,
    is_kpi_candidate_column,
    KPICard,
    ResultPresentation,
)


class TestResultPresentation(unittest.TestCase):

    # -----------------------------------------------------------------
    # 1. Empty DataFrame
    # -----------------------------------------------------------------
    def test_1_empty_dataframe(self):
        """Empty DataFrame, None, or 0 rows must classify as 'empty'."""
        self.assertEqual(classify_result_shape(None), "empty")
        self.assertEqual(classify_result_shape(pd.DataFrame()), "empty")
        self.assertEqual(classify_result_shape(pd.DataFrame(columns=["total_revenue"])), "empty")

        meta = build_kpi_metadata(pd.DataFrame())
        self.assertEqual(meta.classification, "empty")
        self.assertEqual(len(meta.kpi_cards), 0)
        self.assertFalse(meta.show_chart)
        self.assertFalse(meta.show_table)

    # -----------------------------------------------------------------
    # 2. 1x1 numeric scalar -> "kpi"
    # -----------------------------------------------------------------
    def test_2_single_numeric_scalar(self):
        """1 row x 1 numeric column must classify as 'kpi'."""
        df = pd.DataFrame({"total_revenue": [109510.0]})
        self.assertEqual(classify_result_shape(df), "kpi")

        meta = build_kpi_metadata(df)
        self.assertEqual(meta.classification, "kpi")
        self.assertEqual(len(meta.kpi_cards), 1)
        self.assertEqual(meta.kpi_cards[0].column, "total_revenue")
        self.assertEqual(meta.kpi_cards[0].label, "Total Revenue")
        self.assertEqual(meta.kpi_cards[0].formatted_value, "109,510")
        self.assertFalse(meta.show_chart)
        self.assertTrue(meta.show_table)

    # -----------------------------------------------------------------
    # 3. 1x1 text scalar -> appropriate non-KPI classification
    # -----------------------------------------------------------------
    def test_3_single_text_scalar(self):
        """1 row x 1 text column must classify as 'table' (non-KPI)."""
        df = pd.DataFrame({"status": ["Completed"]})
        self.assertEqual(classify_result_shape(df), "table")

        meta = build_kpi_metadata(df)
        self.assertEqual(meta.classification, "table")
        self.assertEqual(len(meta.kpi_cards), 0)
        self.assertFalse(meta.show_chart)
        self.assertTrue(meta.show_table)

    # -----------------------------------------------------------------
    # 4. 1x3 numeric result -> "multi_kpi"
    # -----------------------------------------------------------------
    def test_4_multi_kpi_result(self):
        """1 row x multiple numeric columns must classify as 'multi_kpi'."""
        df = pd.DataFrame({
            "total_revenue": [641021.0],
            "avg_order_value": [6410.21],
            "total_orders": [100],
        })
        self.assertEqual(classify_result_shape(df), "multi_kpi")

        meta = build_kpi_metadata(df)
        self.assertEqual(meta.classification, "multi_kpi")
        self.assertEqual(len(meta.kpi_cards), 3)
        self.assertEqual(meta.kpi_cards[0].label, "Total Revenue")
        self.assertEqual(meta.kpi_cards[0].formatted_value, "641,021")
        self.assertEqual(meta.kpi_cards[1].label, "Avg Order Value")
        self.assertEqual(meta.kpi_cards[1].formatted_value, "6,410.21")
        self.assertEqual(meta.kpi_cards[2].label, "Total Orders")
        self.assertEqual(meta.kpi_cards[2].formatted_value, "100")
        self.assertFalse(meta.show_chart)
        self.assertTrue(meta.show_table)

    # -----------------------------------------------------------------
    # 5. Grouped categorical + numeric result -> "table_and_chart"
    # -----------------------------------------------------------------
    def test_5_grouped_categorical_numeric_result(self):
        """Grouped multi-row results with categorical and numeric columns classify as 'table_and_chart'."""
        df = pd.DataFrame({
            "region": ["East", "South", "North", "West"],
            "total_revenue": [207141.0, 159251.0, 148520.0, 126109.0],
        })
        self.assertEqual(classify_result_shape(df), "table_and_chart")

        meta = build_kpi_metadata(df)
        self.assertEqual(meta.classification, "table_and_chart")
        self.assertEqual(len(meta.kpi_cards), 0)
        self.assertTrue(meta.show_chart)
        self.assertTrue(meta.show_table)

    # -----------------------------------------------------------------
    # 6. Table-only result -> "table"
    # -----------------------------------------------------------------
    def test_6_table_only_result(self):
        """Multi-row results not suitable for automated charts classify as 'table'."""
        df = pd.DataFrame({
            "order_id": [101, 102, 103],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "product": ["Widget", "Gadget", "Doodad"],
            "status": ["Shipped", "Pending", "Delivered"],
        })
        self.assertEqual(classify_result_shape(df), "table")

        meta = build_kpi_metadata(df)
        self.assertEqual(meta.classification, "table")
        self.assertEqual(len(meta.kpi_cards), 0)
        self.assertFalse(meta.show_chart)
        self.assertTrue(meta.show_table)

    # -----------------------------------------------------------------
    # 7. Identifier columns are not incorrectly treated as KPIs
    # -----------------------------------------------------------------
    def test_7_identifier_columns_excluded_from_kpis(self):
        """Identifiers (order_id, customer_id, zip_code, year) must not be treated as KPI metrics."""
        id_cols = [
            "order_id", "customer_id", "product_id", "store_id", "user_id",
            "zip_code", "postal_code", "pincode", "year", "fiscal_year",
            "country_code", "account_num",
        ]
        for col in id_cols:
            self.assertTrue(
                is_identifier_or_temporal_column(col),
                f"Expected {col} to be identified as an identifier/temporal column",
            )
            s = pd.Series([2024, 2025])
            self.assertFalse(
                is_kpi_candidate_column(s, col),
                f"Expected {col} to NOT be a KPI candidate",
            )

        # 1-row table of only an identifier should NOT classify as kpi
        df_id = pd.DataFrame({"order_id": [101]})
        self.assertEqual(classify_result_shape(df_id), "table")

    # -----------------------------------------------------------------
    # 8. KPI labels convert correctly
    # -----------------------------------------------------------------
    def test_8_kpi_label_formatting(self):
        """snake_case and function alias strings convert to Title Case labels."""
        self.assertEqual(format_kpi_label("total_revenue"), "Total Revenue")
        self.assertEqual(format_kpi_label("avg_order_value"), "Avg Order Value")
        self.assertEqual(format_kpi_label("total_orders"), "Total Orders")
        self.assertEqual(format_kpi_label("sum(revenue)"), "Sum Revenue")
        self.assertEqual(format_kpi_label("AVG(unit_price)"), "Avg Unit Price")
        self.assertEqual(format_kpi_label("quarterly_sales_kpi"), "Quarterly Sales KPI")
        self.assertEqual(format_kpi_label(""), "")

    # -----------------------------------------------------------------
    # 9. Integer formatting
    # -----------------------------------------------------------------
    def test_9_integer_formatting(self):
        """Integers are formatted with thousands separators without decimal points."""
        self.assertEqual(format_kpi_value(109510), "109,510")
        self.assertEqual(format_kpi_value(100), "100")
        self.assertEqual(format_kpi_value(0), "0")
        self.assertEqual(format_kpi_value(1000000), "1,000,000")
        self.assertEqual(format_kpi_value(1000000000), "1,000,000,000")
        # Floating point exact integers format cleanly
        self.assertEqual(format_kpi_value(109510.0), "109,510")
        self.assertEqual(format_kpi_value(50.0), "50")

    # -----------------------------------------------------------------
    # 10. Float formatting
    # -----------------------------------------------------------------
    def test_10_float_formatting(self):
        """Floating point numbers format with two decimal places and commas."""
        self.assertEqual(format_kpi_value(6410.21), "6,410.21")
        self.assertEqual(format_kpi_value(0.2345), "0.23")
        self.assertEqual(format_kpi_value(1234567.891), "1,234,567.89")
        self.assertEqual(format_kpi_value(19.5), "19.50")
        # Percentage column formatting if col_name specifies pct
        self.assertEqual(format_kpi_value(0.2345, col_name="profit_pct"), "23.45%")
        self.assertEqual(format_kpi_value(15.5, col_name="margin_percent"), "15.50%")

    # -----------------------------------------------------------------
    # 11. Negative values
    # -----------------------------------------------------------------
    def test_11_negative_values(self):
        """Negative numbers preserve minus sign and format appropriately."""
        self.assertEqual(format_kpi_value(-109510), "-109,510")
        self.assertEqual(format_kpi_value(-6410.21), "-6,410.21")
        self.assertEqual(format_kpi_value(-0.5), "-0.50")
        self.assertEqual(format_kpi_value(-100.0), "-100")

    # -----------------------------------------------------------------
    # 12. Null values
    # -----------------------------------------------------------------
    def test_12_null_values(self):
        """Null, None, and NaN format safely as 'N/A'."""
        self.assertEqual(format_kpi_value(None), "N/A")
        self.assertEqual(format_kpi_value(float("nan")), "N/A")
        self.assertEqual(format_kpi_value(np.nan), "N/A")

    # -----------------------------------------------------------------
    # 13. Original DataFrame is not mutated
    # -----------------------------------------------------------------
    def test_13_dataframe_immutability(self):
        """Result presentation functions never mutate the input DataFrame."""
        df = pd.DataFrame({
            "region": ["East", "South"],
            "revenue": [207141.0, 159251.0],
            "orders": [50, 40],
        })
        df_copy = df.copy(deep=True)

        _ = classify_result_shape(df)
        _ = identify_kpi_columns(df)
        _ = build_kpi_metadata(df)

        pd.testing.assert_frame_equal(df, df_copy)

    # -----------------------------------------------------------------
    # 14. Multiple KPI limit behaves correctly
    # -----------------------------------------------------------------
    def test_14_multiple_kpi_limit(self):
        """When multiple KPIs exceed max_kpis (e.g. 4), cards are capped while total_kpis tracks all."""
        df = pd.DataFrame({
            f"metric_{i}": [float(i * 100)] for i in range(1, 7)
        })
        self.assertEqual(len(df.columns), 6)

        meta = build_kpi_metadata(df, max_kpis=4)
        self.assertEqual(meta.classification, "multi_kpi")
        self.assertEqual(meta.total_kpis, 6)
        self.assertEqual(meta.displayed_kpis, 4)
        self.assertEqual(len(meta.kpi_cards), 4)
        self.assertEqual(meta.kpi_cards[0].label, "Metric 1")
        self.assertEqual(meta.kpi_cards[3].label, "Metric 4")

        # Serializability check
        as_dict = meta.to_dict()
        self.assertEqual(as_dict["classification"], "multi_kpi")
        self.assertEqual(len(as_dict["kpi_cards"]), 4)


if __name__ == "__main__":
    unittest.main()
