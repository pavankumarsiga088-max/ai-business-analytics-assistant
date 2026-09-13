"""
Integration test for Real Google Gemini API Connection.

Verifies live connectivity to Google Gemini API via official google-genai SDK.

SECURITY RULES:
- NEVER prints, displays, or exposes the GEMINI_API_KEY.
- Sends only safe analytical questions.
- Does not modify any underlying datasets.
"""
import os
import sys

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from src.analytics.nl_to_sql import (
    generate_sql,
    get_gemini_client,
    clean_generated_sql,
    LLMGenerationError,
)
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    validate_read_only_query,
)
from src.data_quality.loader import load_dataset


def run_gemini_integration_test():
    print("=" * 78)
    print(" REAL GEMINI API INTEGRATION TEST")
    print("=" * 78)

    # 1. Verify GEMINI_API_KEY presence safely (NEVER print the value)
    api_key_configured = bool(os.getenv("GEMINI_API_KEY"))
    if not api_key_configured:
        print("[SKIP] GEMINI_API_KEY is not configured in .env or environment.")
        print("Please configure GEMINI_API_KEY in your root .env file to run live tests.")
        print("=" * 78)
        return

    print("[STEP 1] Confirmed GEMINI_API_KEY is configured in the environment.")
    print("         (Key value hidden for security)")

    # 2. Verify client initialization
    print("\n[STEP 2] Initializing official Google GenAI client...")
    try:
        client = get_gemini_client()
        print("  -> SUCCESS: Client initialized cleanly using official google-genai SDK.")
    except Exception as e:
        print(f"  -> ERROR: Client initialization failed: {e}")
        raise

    # 3. Send a simple, safe business question
    question = "What are the total sales revenue by region?"
    print(f"\n[STEP 3] Sending business question to Gemini:")
    print(f"         Question: '{question}'")

    try:
        generated_sql = generate_sql(question=question)
    except LLMGenerationError as e:
        print(f"  -> ERROR: Gemini API generation failed: {e}")
        raise

    # 4. Verify non-empty, clean response
    print("\n[STEP 4] Verifying generated SQL response:")
    assert isinstance(generated_sql, str), "Expected string output from generate_sql"
    assert len(generated_sql.strip()) > 0, "Generated SQL must not be empty"
    assert not generated_sql.startswith("```"), "Generated SQL must not contain opening markdown fence"
    assert not generated_sql.endswith("```"), "Generated SQL must not contain closing markdown fence"

    print("  -> Generated SQL:")
    for line in generated_sql.splitlines():
        print(f"     {line}")

    # 5. Verify read-only SQL validation gate
    print("\n[STEP 5] Passing generated SQL through DuckDB read-only validation gate...")
    validate_read_only_query(generated_sql)
    print("  -> SUCCESS: SQL passed strict read-only validation (SELECT/WITH only, no mutations).")

    # 6. Execute against sample dataset in DuckDB
    print("\n[STEP 6] Executing generated query against DuckDB sample dataset...")
    df = load_dataset("sample_data/sample_sales.csv")
    with DuckDBEngine() as engine:
        engine.register_dataframe(df, table_name="sales")
        result_df = engine.execute_query(generated_sql)
        print(f"  -> SUCCESS: Query executed! Result shape: {result_df.shape[0]} rows, {result_df.shape[1]} columns.")
        print("\n  Sample Result Preview:")
        print(result_df.to_string(index=False))

    print("\n" + "=" * 78)
    print(" REAL GEMINI API INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_gemini_integration_test()
