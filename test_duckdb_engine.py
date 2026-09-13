"""
Test script for the DuckDB SQL Analytics Engine.

Validates all 12 test requirements:
1. Load sample_sales.csv using existing loader.
2. Register DataFrame as a DuckDB table named 'sales'.
3. Run SELECT * FROM sales LIMIT 5 (verify exactly 5 rows).
4. Run SELECT COUNT(*) AS total_orders FROM sales (verify count equals dataset length).
5. Run product revenue aggregation (verify columns, success, descending order).
6. Run region revenue aggregation (verify successful execution).
7. Verify INSERT query is rejected.
8. Verify UPDATE query is rejected.
9. Verify DELETE query is rejected.
10. Verify DROP query is rejected.
11. Verify invalid SQL produces a clear error rather than crashing.
12. Verify original DataFrame is completely unmodified.
"""
import pandas as pd
from src.data_quality.loader import load_dataset
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    DisallowedQueryError,
    QueryExecutionError,
)


def run_all_tests():
    print("=" * 78)
    print(" DUCKDB SQL ANALYTICS ENGINE TEST SUITE")
    print("=" * 78)
    print("Executing all 12 tests against 'sample_data/sample_sales.csv'...\n")

    # -----------------------------------------------------------------
    # TEST 1: Load sample_sales.csv using existing loader
    # -----------------------------------------------------------------
    print("[TEST 1] Loading sample_sales.csv using existing loader...")
    csv_path = "sample_data/sample_sales.csv"
    df = load_dataset(csv_path)
    
    assert isinstance(df, pd.DataFrame), "Loaded dataset must be a pandas DataFrame"
    assert len(df) == 100, f"Expected 100 rows in sample_sales.csv, got {len(df)}"
    assert len(df.columns) == 10, f"Expected 10 columns, got {len(df.columns)}"
    
    # Keep an exact deep copy to test immutability in TEST 12
    df_snapshot = df.copy(deep=True)
    print(f"  -> SUCCESS: Loaded {len(df)} rows, {len(df.columns)} columns.\n")

    # -----------------------------------------------------------------
    # TEST 2: Register DataFrame as DuckDB table named 'sales'
    # -----------------------------------------------------------------
    print("[TEST 2] Registering DataFrame as DuckDB table 'sales'...")
    engine = DuckDBEngine()
    engine.register_dataframe(df, table_name="sales")
    
    assert "sales" in engine.get_registered_tables(), "Table 'sales' must be in registered tables"
    schema = engine.get_schema("sales")
    assert "revenue" in schema and "quantity" in schema and "product" in schema
    print(f"  -> SUCCESS: Table 'sales' registered. Detected schema columns: {list(schema.keys())}\n")

    # -----------------------------------------------------------------
    # TEST 3: SELECT * FROM sales LIMIT 5 (exactly 5 rows)
    # -----------------------------------------------------------------
    print("[TEST 3] Running: SELECT * FROM sales LIMIT 5;")
    res_limit5 = engine.execute_query("SELECT * FROM sales LIMIT 5;")
    
    assert isinstance(res_limit5, pd.DataFrame), "Result must be a pandas DataFrame"
    assert len(res_limit5) == 5, f"Expected exactly 5 rows, got {len(res_limit5)}"
    assert len(res_limit5.columns) == len(df.columns), "Columns count must match original DataFrame"
    print("  -> SUCCESS: Exactly 5 rows returned.")
    print(res_limit5[["order_id", "product", "quantity", "unit_price", "revenue"]].to_string(index=False))
    print("\n")

    # -----------------------------------------------------------------
    # TEST 4: SELECT COUNT(*) AS total_orders FROM sales
    # -----------------------------------------------------------------
    print("[TEST 4] Running: SELECT COUNT(*) AS total_orders FROM sales;")
    res_count = engine.execute_query("SELECT COUNT(*) AS total_orders FROM sales;")
    
    assert "total_orders" in res_count.columns, "'total_orders' column missing in result"
    total_orders = int(res_count["total_orders"].iloc[0])
    assert total_orders == len(df), f"Expected total_orders == {len(df)}, got {total_orders}"
    print(f"  -> SUCCESS: Count returned {total_orders}, exactly matching dataset length.\n")

    # -----------------------------------------------------------------
    # TEST 5: Product revenue aggregation query
    # -----------------------------------------------------------------
    print("[TEST 5] Running Product Revenue Aggregation:")
    sql_product = """
    SELECT product,
           SUM(revenue) AS total_revenue
    FROM sales
    GROUP BY product
    ORDER BY total_revenue DESC;
    """
    res_product = engine.execute_query(sql_product)
    
    assert isinstance(res_product, pd.DataFrame), "Result must be a pandas DataFrame"
    assert "product" in res_product.columns, "'product' column missing in result"
    assert "total_revenue" in res_product.columns, "'total_revenue' column missing in result"
    assert len(res_product) > 0, "Result must not be empty"
    
    # Verify ordered descending by total_revenue
    revenues = res_product["total_revenue"].tolist()
    assert revenues == sorted(revenues, reverse=True), "Results are not ordered descending by total_revenue"
    print("  -> SUCCESS: Product aggregation executed and verified descending.")
    print(res_product.to_string(index=False))
    print("\n")

    # -----------------------------------------------------------------
    # TEST 6: Region revenue aggregation query
    # -----------------------------------------------------------------
    print("[TEST 6] Running Regional Revenue Aggregation:")
    sql_region = """
    SELECT region,
           SUM(revenue) AS total_revenue
    FROM sales
    GROUP BY region
    ORDER BY total_revenue DESC;
    """
    res_region = engine.execute_query(sql_region)
    
    assert isinstance(res_region, pd.DataFrame), "Result must be a pandas DataFrame"
    assert "region" in res_region.columns, "'region' column missing in result"
    assert "total_revenue" in res_region.columns, "'total_revenue' column missing in result"
    assert len(res_region) > 0, "Result must not be empty"
    print("  -> SUCCESS: Regional aggregation executed successfully.")
    print(res_region.to_string(index=False))
    print("\n")

    # -----------------------------------------------------------------
    # TEST 7: Verify that an INSERT query is rejected
    # -----------------------------------------------------------------
    print("[TEST 7] Verifying that an INSERT query is rejected...")
    insert_sql = "INSERT INTO sales (order_id, product) VALUES ('ORD-9999', 'Fake Product');"
    try:
        engine.execute_query(insert_sql)
        raise AssertionError("SECURITY FAILURE: INSERT query was not rejected!")
    except DisallowedQueryError as e:
        print(f"  -> SUCCESS: INSERT properly blocked with error: {e}")
    print("\n")

    # -----------------------------------------------------------------
    # TEST 8: Verify that an UPDATE query is rejected
    # -----------------------------------------------------------------
    print("[TEST 8] Verifying that an UPDATE query is rejected...")
    update_sql = "UPDATE sales SET revenue = 0 WHERE quantity = 1;"
    try:
        engine.execute_query(update_sql)
        raise AssertionError("SECURITY FAILURE: UPDATE query was not rejected!")
    except DisallowedQueryError as e:
        print(f"  -> SUCCESS: UPDATE properly blocked with error: {e}")
    print("\n")

    # -----------------------------------------------------------------
    # TEST 9: Verify that a DELETE query is rejected
    # -----------------------------------------------------------------
    print("[TEST 9] Verifying that a DELETE query is rejected...")
    delete_sql = "DELETE FROM sales WHERE revenue < 1000;"
    try:
        engine.execute_query(delete_sql)
        raise AssertionError("SECURITY FAILURE: DELETE query was not rejected!")
    except DisallowedQueryError as e:
        print(f"  -> SUCCESS: DELETE properly blocked with error: {e}")
    print("\n")

    # -----------------------------------------------------------------
    # TEST 10: Verify that a DROP query is rejected
    # -----------------------------------------------------------------
    print("[TEST 10] Verifying that a DROP query is rejected...")
    drop_sql = "DROP TABLE sales;"
    try:
        engine.execute_query(drop_sql)
        raise AssertionError("SECURITY FAILURE: DROP query was not rejected!")
    except DisallowedQueryError as e:
        print(f"  -> SUCCESS: DROP properly blocked with error: {e}")
        
    # Also verify that multi-statement SQL injection (SELECT ... ; DROP ...) is rejected
    injection_sql = "SELECT * FROM sales; DROP TABLE sales;"
    try:
        engine.execute_query(injection_sql)
        raise AssertionError("SECURITY FAILURE: Multi-statement SQL injection was not rejected!")
    except DisallowedQueryError as e:
        print(f"  -> SUCCESS: Multi-statement injection properly blocked with error: {e}")
    print("\n")

    # -----------------------------------------------------------------
    # TEST 11: Verify that invalid SQL produces a clear error
    # -----------------------------------------------------------------
    print("[TEST 11] Verifying that invalid SQL produces a clear error rather than crashing...")
    invalid_sql = "SELECT non_existent_column FORM sales;"  # typo in FORM and missing column
    try:
        engine.execute_query(invalid_sql)
        raise AssertionError("FAILURE: Invalid SQL did not raise an exception!")
    except QueryExecutionError as e:
        print(f"  -> SUCCESS: Invalid SQL caught cleanly as QueryExecutionError:\n     {e}")
    except DisallowedQueryError as e:
        # Also acceptable if caught by syntax validator
        print(f"  -> SUCCESS: Invalid SQL caught as DisallowedQueryError:\n     {e}")
    print("\n")

    # -----------------------------------------------------------------
    # TEST 12: Verify original Pandas DataFrame has not been modified
    # -----------------------------------------------------------------
    print("[TEST 12] Verifying that the original Pandas DataFrame is completely unmodified...")
    assert df.shape == df_snapshot.shape, f"DataFrame shape changed from {df_snapshot.shape} to {df.shape}"
    assert list(df.columns) == list(df_snapshot.columns), "DataFrame columns were modified"
    assert df.equals(df_snapshot), "DataFrame contents differ from original snapshot!"
    print("  -> SUCCESS: df.equals(df_snapshot) is True. The original DataFrame was not mutated in any way.")
    print("\n")

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    engine.close()
    assert not engine.is_open, "Engine must be closed"
    print("Engine closed safely.")

    print("=" * 78)
    print(" ALL 12 TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_tests()
