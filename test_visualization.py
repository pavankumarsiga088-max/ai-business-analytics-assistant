"""
Test Suite for Automatic Plotly Visualization Engine (Phase 9B).

Validates all 8 core visualization rules and constraints:
1. categorical + numeric -> bar chart
2. date + numeric -> line chart
3. multiple categories/numeric columns -> grouped bar chart
4. numeric-only result -> no chart (None)
5. empty DataFrame -> no chart (None)
6. one-column DataFrame -> no chart (None)
7. invalid/unsupported data -> no chart (None)
8. original DataFrame is not mutated
"""
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.analytics.visualization import (
    create_chart,
    generate_chart_title,
    is_date_column,
    is_numeric_measure_column,
    is_categorical_dimension_column,
)


def run_all_visualization_tests():
    print("=" * 78)
    print(" AUTOMATIC PLOTLY VISUALIZATION (PHASE 9B) TEST SUITE")
    print("=" * 78)
    print("Running all offline visualization tests...\n")

    # -----------------------------------------------------------------
    # TEST 1: Categorical + Numeric -> Bar Chart
    # -----------------------------------------------------------------
    print("[TEST 1] Verifying categorical + numeric generates a bar chart...")
    bar_df = pd.DataFrame({
        "region": ["East", "South", "North", "Central", "West"],
        "total_revenue": [207141.0, 159251.0, 112362.0, 82061.0, 80206.0],
    })
    bar_fig = create_chart(bar_df)
    assert bar_fig is not None, "Expected a Plotly Figure for categorical + numeric data"
    assert isinstance(bar_fig, go.Figure), "Result must be a plotly.graph_objects.Figure"
    # Check trace type
    trace_types = [t.type for t in bar_fig.data]
    assert "bar" in trace_types, f"Expected 'bar' trace in chart, got: {trace_types}"
    # Verify title generated
    assert "Revenue" in bar_fig.layout.title.text and "Region" in bar_fig.layout.title.text
    print(f"  -> SUCCESS: Bar chart created with title: '{bar_fig.layout.title.text}'\n")

    # -----------------------------------------------------------------
    # TEST 2: Date + Numeric -> Line Chart
    # -----------------------------------------------------------------
    print("[TEST 2] Verifying date + numeric generates a line chart...")
    line_df = pd.DataFrame({
        "order_date": ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"],
        "weekly_sales": [12500.0, 18900.0, 15400.0, 22100.0],
    })
    line_fig = create_chart(line_df)
    assert line_fig is not None, "Expected a Plotly Figure for date + numeric data"
    assert isinstance(line_fig, go.Figure), "Result must be a plotly.graph_objects.Figure"
    trace_types = [t.type for t in line_fig.data]
    assert "scatter" in trace_types, f"Expected 'scatter' (line) trace in chart, got: {trace_types}"
    assert "Over Time" in line_fig.layout.title.text or "Order Date" in line_fig.layout.title.text
    print(f"  -> SUCCESS: Line chart created with title: '{line_fig.layout.title.text}'\n")

    # -----------------------------------------------------------------
    # TEST 3: Categorical + Multiple Numeric -> Grouped Bar Chart
    # -----------------------------------------------------------------
    print("[TEST 3] Verifying categorical + multiple numeric generates a grouped bar chart...")
    multi_df = pd.DataFrame({
        "category": ["Electronics", "Furniture", "Home Decor"],
        "total_revenue": [150000.0, 120000.0, 65000.0],
        "total_profit": [45000.0, 36000.0, 19500.0],
    })
    multi_fig = create_chart(multi_df)
    assert multi_fig is not None, "Expected a Plotly Figure for categorical + multiple numeric data"
    assert isinstance(multi_fig, go.Figure)
    trace_types = [t.type for t in multi_fig.data]
    assert all(t == "bar" for t in trace_types), f"All traces should be 'bar', got: {trace_types}"
    assert len(multi_fig.data) == 2, f"Expected 2 grouped series, got {len(multi_fig.data)}"
    assert multi_fig.layout.barmode == "group", "Expected layout barmode='group'"
    print(f"  -> SUCCESS: Grouped bar chart created with title: '{multi_fig.layout.title.text}'\n")

    # -----------------------------------------------------------------
    # TEST 4: Numeric-only result -> No Chart (None)
    # -----------------------------------------------------------------
    print("[TEST 4] Verifying numeric-only DataFrame returns None...")
    numeric_only_df = pd.DataFrame({
        "min_price": [10.0, 20.0, 30.0],
        "max_price": [100.0, 200.0, 300.0],
    })
    num_fig = create_chart(numeric_only_df)
    assert num_fig is None, "Numeric-only DataFrame without dimensions should return None"
    print("  -> SUCCESS: Numeric-only data correctly returned None (no meaningless chart).\n")

    # -----------------------------------------------------------------
    # TEST 5: Empty DataFrame -> No Chart (None)
    # -----------------------------------------------------------------
    print("[TEST 5] Verifying empty DataFrame returns None...")
    empty_df = pd.DataFrame()
    assert create_chart(empty_df) is None, "Empty DataFrame should return None"
    empty_with_cols = pd.DataFrame(columns=["region", "revenue"])
    assert create_chart(empty_with_cols) is None, "Empty DataFrame with columns should return None"
    assert create_chart(None) is None, "None input should return None"
    print("  -> SUCCESS: Empty DataFrames and None correctly return None.\n")

    # -----------------------------------------------------------------
    # TEST 6: One-column DataFrame -> No Chart (None)
    # -----------------------------------------------------------------
    print("[TEST 6] Verifying single-column DataFrame (e.g. COUNT(*)) returns None...")
    single_col_df = pd.DataFrame({"total_orders": [100]})
    assert create_chart(single_col_df) is None, "Single-cell / single-column DataFrame should return None"
    single_cat_col = pd.DataFrame({"region": ["East", "West", "North"]})
    assert create_chart(single_cat_col) is None, "Single categorical column DataFrame should return None"
    print("  -> SUCCESS: Single-column DataFrames correctly return None.\n")

    # -----------------------------------------------------------------
    # TEST 7: Identifier columns (order_id, etc.) not treated as numeric measures
    # -----------------------------------------------------------------
    print("[TEST 7] Verifying that identifier columns are not plotted as numeric measures...")
    id_df = pd.DataFrame({
        "order_id": ["ORD-1001", "ORD-1002", "ORD-1003"],
        "customer_id": [501, 502, 503],
    })
    assert create_chart(id_df) is None, "ID-only DataFrame should return None"
    print("  -> SUCCESS: Identifier columns are not treated as quantitative measures.\n")

    # -----------------------------------------------------------------
    # TEST 8: Immutability - Original DataFrame is not mutated
    # -----------------------------------------------------------------
    print("[TEST 8] Verifying that input DataFrame is completely unmodified...")
    test_dates = ["2024-03-01", "2024-01-01", "2024-02-01"]  # Unsorted dates
    df_immutable = pd.DataFrame({
        "order_date": test_dates,
        "revenue": [300.0, 100.0, 200.0],
    })
    df_snapshot = df_immutable.copy(deep=True)
    _ = create_chart(df_immutable)
    assert df_immutable.equals(df_snapshot), "Original DataFrame was mutated during chart creation!"
    # Ensure original order_date was not sorted in place
    assert list(df_immutable["order_date"]) == test_dates, "Original row ordering was mutated!"
    print("  -> SUCCESS: Input DataFrame remained completely unmodified.\n")

    print("=" * 78)
    print(" ALL 8 VISUALIZATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_visualization_tests()
