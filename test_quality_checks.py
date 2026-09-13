"""
Test script to run all 10 Data Quality checks against both
clean and dirty sample datasets.
"""
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


def run_all_checks(name: str, file_path: str):
    print("=" * 75)
    print(f"RUNNING ALL 10 QUALITY CHECKS ON: {name}")
    print(f"Source file: {file_path}")
    print("=" * 75)
    
    # 1. Load the dataset using our loader
    df = load_dataset(file_path)
    print(f"Loaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    
    # 2. Execute all 10 checks
    results = [
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
    
    # 3. Print formatted report for each check
    for idx, r in enumerate(results, 1):
        status_symbol = "[FAIL]" if r["has_issues"] else "[PASS]"
        print(f"{status_symbol} Check {idx}: {r['check_name']}")
        
        if r["has_issues"]:
            for k, v in r.items():
                if k not in ("check_name", "has_issues"):
                    print(f"     * {k}: {v}")
        else:
            print("     * No issues detected.")
        print("-" * 75)
    print("\n")


def main():
    # 1. Test Clean Dataset
    run_all_checks("Clean E-Commerce Dataset (CSV)", "sample_data/sample_sales.csv")
    
    # 2. Test Dirty Dataset
    run_all_checks("Dirty Dataset with Intentional Anomalies (Excel)", "sample_data/sample_dirty_data.xlsx")


if __name__ == "__main__":
    main()
