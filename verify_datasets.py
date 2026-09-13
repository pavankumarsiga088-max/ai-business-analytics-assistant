"""
Verification script for Phase 2 datasets.
Loads sample_sales.csv and sample_dirty_data.xlsx using Pandas and inspects basic properties.
"""
import pandas as pd

def inspect_dataset(name: str, file_path: str, is_excel: bool = False):
    print("=" * 60)
    print(f"INSPECTING: {name} ({file_path})")
    print("=" * 60)

    # 1. Load data
    if is_excel:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # 2. Shape
    num_rows, num_cols = df.shape
    print(f"Rows: {num_rows}")
    print(f"Columns: {num_cols}")
    print("-" * 60)

    # 3. Columns & Data types
    print("Column Names & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<16} : {dtype}")
    print("-" * 60)

    # 4. Missing values
    missing_series = df.isna().sum()
    total_missing = missing_series.sum()
    print(f"Total Missing Values: {total_missing}")
    if total_missing > 0:
        print("Missing Values Per Column:")
        for col, count in missing_series.items():
            if count > 0:
                print(f"  - {col:<16} : {count} missing")
    else:
        print("  None detected.")
    print("-" * 60)

    # 5. Duplicate rows
    num_duplicates = df.duplicated().sum()
    print(f"Duplicate Rows (all columns matching): {num_duplicates}")
    order_id_duplicates = df["order_id"].duplicated().sum()
    print(f"Duplicate 'order_id' entries: {order_id_duplicates}")
    print("-" * 60)

    # 6. First 10 rows preview
    print("First 10 Rows Preview:")
    print(df.head(10).to_string())
    print("\n")

def main():
    inspect_dataset("Clean Dataset (CSV)", "sample_data/sample_sales.csv", is_excel=False)
    inspect_dataset("Dirty Dataset (Excel)", "sample_data/sample_dirty_data.xlsx", is_excel=True)

if __name__ == "__main__":
    main()
