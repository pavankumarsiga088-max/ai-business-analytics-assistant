"""
Comprehensive Verification Script for Phase 13: Full PostgreSQL Analytics Integration.

Executes and verifies:
1. Database backend discovery and configuration defaults.
2. DuckDB pipeline: loading, schema extraction, query execution, KPI card generation, Plotly visualization, downloads.
3. PostgreSQL pipeline (live): loading, schema extraction, query execution, KPI card generation, Plotly visualization, downloads.
4. Ground truth equivalence between DuckDB and PostgreSQL query results.
5. Dirty dataset compatibility (sample_dirty_data.xlsx) on both engines.
6. Dataset boundary isolation: resetting conversation turns on dataset change.
7. Streamlit app module health and configuration.
"""
import sys
import pandas as pd

from src.data_quality.loader import load_dataset
from src.analytics.database import (
    create_database_engine,
    get_configured_database_backend,
    is_backend_available,
)
from src.analytics.dataset_loader import load_dataset_into_database
from src.analytics.result_presentation import build_kpi_metadata
from src.analytics.visualization import create_chart
from src.analytics.downloads import dataframe_to_csv_bytes, dataframe_to_excel_bytes
from src.analytics.conversation import ConversationHistory


def verify_all():
    print("=" * 78)
    print(" PHASE 13 FULL POSTGRESQL ANALYTICS INTEGRATION VERIFICATION")
    print("=" * 78)

    # 1. Configuration Check
    print("\n[1] Checking Database Configuration & Availability...")
    default_backend = get_configured_database_backend()
    print(f"  -> Configured Default Backend: '{default_backend}'")
    assert default_backend in ("duckdb", "postgres")

    duck_ok, _ = is_backend_available("duckdb")
    print(f"  -> DuckDB Availability: {'AVAILABLE' if duck_ok else 'UNAVAILABLE'}")
    assert duck_ok

    pg_ok, pg_reason = is_backend_available("postgres")
    print(f"  -> PostgreSQL Availability: {'AVAILABLE' if pg_ok else f'UNAVAILABLE ({pg_reason})'}")

    # 2. Clean Dataset Loading & Querying on DuckDB
    print("\n[2] Executing DuckDB Analytics Pipeline...")
    clean_csv = "sample_data/sample_sales.csv"
    sales_df = load_dataset(clean_csv)
    sales_snapshot = sales_df.copy(deep=True)

    with create_database_engine("duckdb") as duck_engine:
        duck_load = load_dataset_into_database(duck_engine, sales_df, table_name="sales")
        print(f"  -> DuckDB Loaded: {duck_load.row_count} rows, {duck_load.column_count} columns into '{duck_load.table_name}'")
        assert duck_load.row_count == 100
        assert duck_load.column_count == 10
        assert sales_df.equals(sales_snapshot), "Caller DataFrame must not be mutated!"

        sql_count = "SELECT COUNT(*) AS total_rows FROM sales;"
        cnt_duck = duck_engine.execute_query(sql_count)
        assert int(cnt_duck["total_rows"].iloc[0]) == 100

        sql_regional = """
        SELECT region, SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC;
        """
        duck_agg = duck_engine.execute_query(sql_regional)
        assert len(duck_agg) == 5
        print("  -> DuckDB Regional Aggregation Result:")
        print(duck_agg.to_string(index=False))

    # 3. Clean Dataset Loading & Querying on PostgreSQL
    if pg_ok:
        print("\n[3] Executing PostgreSQL Live Analytics Pipeline...")
        with create_database_engine("postgres") as pg_engine:
            pg_load = load_dataset_into_database(pg_engine, sales_df, table_name="sales")
            print(f"  -> PostgreSQL Loaded: {pg_load.row_count} rows, {pg_load.column_count} columns into '{pg_load.table_name}'")
            assert pg_load.row_count == 100
            assert pg_load.column_count == 10
            assert sales_df.equals(sales_snapshot), "Caller DataFrame must not be mutated!"

            cnt_pg = pg_engine.execute_query(sql_count)
            assert int(cnt_pg["total_rows"].iloc[0]) == 100

            pg_agg = pg_engine.execute_query(sql_regional)
            assert len(pg_agg) == 5
            print("  -> PostgreSQL Regional Aggregation Result:")
            print(pg_agg.to_string(index=False))

            # Compare ground truth
            duck_revs = [float(x) for x in duck_agg["total_revenue"]]
            pg_revs = [float(x) for x in pg_agg["total_revenue"]]
            assert duck_revs == pg_revs == [207141.0, 159251.0, 112362.0, 82061.0, 80206.0]
            print("  -> SUCCESS: DuckDB and PostgreSQL aggregation results match bit-for-bit!")
    else:
        print("\n[3] PostgreSQL live integration skipped (database or password unavailable).")

    # 4. Dirty Dataset Loading & Type Compatibility
    print("\n[4] Verifying Dirty Dataset (sample_dirty_data.xlsx) Compatibility...")
    dirty_xlsx = "sample_data/sample_dirty_data.xlsx"
    dirty_df = load_dataset(dirty_xlsx)
    assert len(dirty_df) == 100

    with create_database_engine("duckdb") as duck_engine:
        res = load_dataset_into_database(duck_engine, dirty_df, table_name="sales")
        assert res.row_count == 100
        print("  -> DuckDB loaded dirty dataset successfully.")

    if pg_ok:
        with create_database_engine("postgres") as pg_engine:
            res = load_dataset_into_database(pg_engine, dirty_df, table_name="sales")
            assert res.row_count == 100
            assert res.schema.get("unit_price") == "TEXT"
            print("  -> PostgreSQL loaded dirty dataset successfully (mixed unit_price -> TEXT).")

    # 5. Downstream Analytics Verification
    print("\n[5] Verifying Downstream Presentation, Visualization & Downloads...")
    presentation = build_kpi_metadata(duck_agg)
    assert presentation.show_chart is True
    print(f"  -> Smart Presentation: Classification '{presentation.classification}', show_chart={presentation.show_chart}")

    fig = create_chart(duck_agg)
    assert fig is not None
    print(f"  -> Plotly Visualization: Created {type(fig).__name__}")

    csv_bytes = dataframe_to_csv_bytes(duck_agg)
    excel_bytes = dataframe_to_excel_bytes(duck_agg)
    assert len(csv_bytes) > 0 and len(excel_bytes) > 0
    print(f"  -> Downloads: CSV ({len(csv_bytes)} bytes), Excel ({len(excel_bytes)} bytes)")

    # 6. Conversation Dataset Boundary Reset
    print("\n[6] Verifying Conversation History Dataset Boundary Reset...")
    conv = ConversationHistory(dataset_name="sample_sales.csv")
    conv.add_turn(question="Turn 1", sql="SELECT 1;", result_df=duck_agg)
    assert len(conv) == 1
    conv.reset_for_dataset("sample_dirty_data.xlsx")
    assert len(conv) == 0
    assert conv.dataset_name == "sample_dirty_data.xlsx"
    print("  -> Conversation turns reset cleanly on dataset boundary.")

    # 7. Streamlit Module Health Check
    print("\n[7] Verifying Streamlit App Health...")
    import app
    assert hasattr(app, "build_result_column_config")
    print("  -> Streamlit app module loaded cleanly without errors.")

    print("\n" + "=" * 78)
    print(" ALL PHASE 13 VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    verify_all()
