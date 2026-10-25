"""
Unit Tests for Dataset-Aware Question Suggestion Engine (src/analytics/question_suggestions.py).

Validates:
1. agriculture-like schema gets agriculture-relevant suggestions
2. revenue + region gets regional revenue suggestion
3. revenue + product gets product/revenue suggestion
4. profit + crop gets crop/profit suggestion
5. water + irrigation gets water/irrigation suggestion
6. risk + crop gets risk suggestion
7. date + revenue gets time-based suggestion
8. unknown schema gets fallback suggestions
9. suggestions are deterministic
10. no DataFrame mutation
"""

import unittest
import pandas as pd

from src.analytics.question_suggestions import (
    generate_suggested_questions,
    extract_column_names,
)


class TestQuestionSuggestions(unittest.TestCase):

    def setUp(self):
        # Agriculture-like dataset schema
        self.agri_columns = [
            "Crop",
            "State",
            "Season",
            "Irrigation_Method",
            "Water_Used",
            "Production",
            "Yield",
            "Disease_Risk",
            "Pest_Risk",
            "Cost",
            "Revenue",
            "Profit",
        ]
        self.agri_df = pd.DataFrame(columns=self.agri_columns)

    # 1. Agriculture-like schema gets agriculture-relevant suggestions
    def test_1_agriculture_schema_suggestions(self):
        """Agriculture dataset schema yields agriculture-relevant domain questions."""
        suggestions = generate_suggested_questions(self.agri_df)
        self.assertTrue(4 <= len(suggestions) <= 6)
        text = " ".join(suggestions).lower()
        self.assertIn("revenue", text)
        self.assertIn("crop", text)
        self.assertIn("profit", text)
        self.assertIn("state", text)
        self.assertIn("water", text)
        self.assertIn("season", text)

    # 2. Revenue + region gets regional revenue suggestion
    def test_2_revenue_and_region_suggestion(self):
        """Schema containing revenue and region suggests regional revenue query."""
        schema = ["region", "revenue"]
        suggestions = generate_suggested_questions(schema)
        text = " ".join(suggestions).lower()
        self.assertTrue(
            "region" in text and "revenue" in text or "which region sells the most" in text
        )

    # 3. Revenue + product gets product/revenue suggestion
    def test_3_revenue_and_product_suggestion(self):
        """Schema containing product and revenue suggests product/revenue queries."""
        schema = ["product", "revenue"]
        suggestions = generate_suggested_questions(schema)
        text = " ".join(suggestions).lower()
        self.assertTrue("product" in text)
        self.assertTrue("revenue" in text or "sales" in text)

    # 4. Profit + crop gets crop/profit suggestion
    def test_4_profit_and_crop_suggestion(self):
        """Schema containing crop and profit suggests crop profitability query."""
        schema = ["crop", "profit"]
        suggestions = generate_suggested_questions(schema)
        text = " ".join(suggestions).lower()
        self.assertIn("crop", text)
        self.assertIn("profit", text)

    # 5. Water + irrigation gets water/irrigation suggestion
    def test_5_water_and_irrigation_suggestion(self):
        """Schema containing water and irrigation suggests water efficiency query."""
        schema = ["water_used", "irrigation_method"]
        suggestions = generate_suggested_questions(schema)
        text = " ".join(suggestions).lower()
        self.assertIn("irrigation", text)
        self.assertIn("water", text)

    # 6. Risk + crop gets risk suggestion
    def test_6_risk_and_crop_suggestion(self):
        """Schema containing risk and crop suggests agricultural risk query."""
        schema = ["crop", "disease_risk", "pest_risk"]
        suggestions = generate_suggested_questions(schema)
        text = " ".join(suggestions).lower()
        self.assertIn("crop", text)
        self.assertIn("risk", text)

    # 7. Date + revenue gets time-based suggestion
    def test_7_date_and_revenue_suggestion(self):
        """Schema containing date and revenue suggests time-based trend query."""
        schema = ["order_date", "revenue"]
        suggestions = generate_suggested_questions(schema)
        text = " ".join(suggestions).lower()
        self.assertTrue("trend" in text or "monthly" in text or "time" in text)

    # 8. Unknown schema gets fallback suggestions
    def test_8_unknown_schema_fallback(self):
        """Unrecognized schema returns sensible generic fallback questions without error."""
        schema = ["foo_col", "bar_metric_99", "custom_hash"]
        suggestions = generate_suggested_questions(schema)
        self.assertTrue(4 <= len(suggestions) <= 6)
        # Should gracefully return valid non-empty questions
        for q in suggestions:
            self.assertIsInstance(q, str)
            self.assertTrue(len(q) > 0)

    # 9. Suggestions are deterministic
    def test_9_suggestions_are_deterministic(self):
        """Consecutive calls with the same schema produce identical questions in the same order."""
        run_1 = generate_suggested_questions(self.agri_columns)
        run_2 = generate_suggested_questions(self.agri_columns)
        run_3 = generate_suggested_questions(self.agri_columns)
        self.assertEqual(run_1, run_2)
        self.assertEqual(run_2, run_3)

    # 10. No DataFrame mutation
    def test_10_no_dataframe_mutation(self):
        """Input DataFrame and its metadata are completely unmodified by suggestion generation."""
        df = pd.DataFrame({
            "Crop": ["Wheat", "Rice"],
            "Revenue": [1000.0, 2000.0],
            "Profit": [300.0, 600.0],
        })
        original_shape = df.shape
        original_columns = list(df.columns)
        original_data = df.to_dict(orient="records")

        _ = generate_suggested_questions(df)

        self.assertEqual(df.shape, original_shape)
        self.assertEqual(list(df.columns), original_columns)
        self.assertEqual(df.to_dict(orient="records"), original_data)


if __name__ == "__main__":
    unittest.main()
