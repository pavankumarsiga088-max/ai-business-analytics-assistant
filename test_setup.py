"""
Test script to verify Phase 1 environment setup.
"""
import sys
import pandas as pd
import duckdb
import plotly

def main():
    print("=" * 50)
    print("Environment Verification Successful!")
    print("=" * 50)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Pandas version : {pd.__version__}")
    print(f"DuckDB version : {duckdb.__version__}")
    print(f"Plotly version : {plotly.__version__}")
    print("All core dependencies loaded cleanly.")

if __name__ == "__main__":
    main()
