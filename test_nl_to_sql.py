"""
Test suite for Natural Language to SQL Generation (Phase 7).

Verifies all 7 core requirements:
1. TEST 1: Empty or whitespace-only question is rejected.
2. TEST 2: Mocked LLM response containing valid SQL is returned correctly.
3. TEST 3: The generated SQL does not contain markdown code fences.
4. TEST 4: Mocked LLM/API failure is converted into a clear application-level error (LLMGenerationError).
5. TEST 5: Mocked SQL generation for "Which product generated the highest revenue?" contains
           the expected analytical concepts: sales, product, revenue, SUM, GROUP BY, ORDER BY, LIMIT.
6. TEST 6: Verify that nl_to_sql.py itself does NOT execute SQL (no DB connections or execution).
7. TEST 7: Verify that a generated INSERT/UPDATE/DELETE/DROP statement is NOT executed
           and is rejected by the existing DuckDBEngine read-only validation layer.
"""
import sys
import re
import pandas as pd

from src.data_quality.loader import load_dataset
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    DisallowedQueryError,
)
from src.analytics.nl_to_sql import (
    generate_sql,
    build_sql_prompt,
    format_schema,
    clean_generated_sql,
    NLToSQLError,
    EmptyQuestionError,
    LLMGenerationError,
    DEFAULT_SALES_SCHEMA,
)


def run_all_tests():
    print("=" * 78)
    print(" NATURAL LANGUAGE TO SQL (PHASE 7) TEST SUITE")
    print("=" * 78)
    print("Executing all 7 security and functionality tests...\n")

    # -----------------------------------------------------------------
    # TEST 1: Empty question is rejected
    # -----------------------------------------------------------------
    print("[TEST 1] Verifying that empty or whitespace-only questions are rejected...")
    empty_cases = ["", "   ", "\n\t  ", None]
    for case in empty_cases:
        try:
            generate_sql(case, llm_client=lambda p: "SELECT 1;")
            raise AssertionError(f"Expected EmptyQuestionError for input: {repr(case)}")
        except EmptyQuestionError as e:
            # Expected behavior
            pass
        except Exception as e:
            raise AssertionError(f"Expected EmptyQuestionError but got {type(e).__name__}: {e}")
    print("  -> SUCCESS: Empty, whitespace, and None questions rejected with EmptyQuestionError.\n")

    # -----------------------------------------------------------------
    # TEST 2: Mocked LLM response containing valid SQL is returned correctly
    # -----------------------------------------------------------------
    print("[TEST 2] Verifying that a mocked LLM response is returned correctly...")
    mock_sql = "SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product;"
    mock_client = lambda prompt: mock_sql

    result_sql = generate_sql(
        question="Show total revenue by product",
        llm_client=mock_client,
    )
    assert result_sql == mock_sql, f"Expected '{mock_sql}', got '{result_sql}'"
    print(f"  -> SUCCESS: Mocked response returned correctly: {result_sql}\n")

    # -----------------------------------------------------------------
    # TEST 3: Generated SQL does not contain markdown code fences
    # -----------------------------------------------------------------
    print("[TEST 3] Verifying that markdown code fences are stripped from LLM output...")
    fenced_responses = [
        # Standard ```sql fence
        "```sql\nSELECT * FROM sales LIMIT 5;\n```",
        # Generic ``` fence without language specifier
        "```\nSELECT region, COUNT(*) FROM sales GROUP BY region;\n```",
        # Fence with trailing / leading spaces
        "   ```sql\nSELECT order_id, revenue FROM sales;\n```   ",
        # Single backtick wrapped
        "`SELECT category FROM sales LIMIT 10;`",
    ]

    for fenced in fenced_responses:
        client = lambda p, r=fenced: r
        cleaned = generate_sql(
            question="Any question",
            llm_client=client,
        )
        assert not cleaned.startswith("```"), f"Cleaned SQL starts with fence: {cleaned}"
        assert not cleaned.endswith("```"), f"Cleaned SQL ends with fence: {cleaned}"
        assert not cleaned.startswith("`"), f"Cleaned SQL starts with backtick: {cleaned}"
        assert not cleaned.endswith("`"), f"Cleaned SQL ends with backtick: {cleaned}"
        assert "SELECT" in cleaned, f"Expected SQL query content preserved: {cleaned}"

    # Specifically test the clean_generated_sql function directly as well
    test_raw = "```sql\nSELECT 1;\n```"
    assert clean_generated_sql(test_raw) == "SELECT 1;"
    print("  -> SUCCESS: All markdown code fences and backticks cleanly removed.\n")

    # -----------------------------------------------------------------
    # TEST 4: Mocked LLM/API failure is converted into application error
    # -----------------------------------------------------------------
    print("[TEST 4] Verifying that LLM/API failure is converted into LLMGenerationError...")
    def failing_client(prompt):
        raise ConnectionError("Simulated network timeout connecting to provider API.")

    try:
        generate_sql(
            question="What is the top selling category?",
            llm_client=failing_client,
        )
        raise AssertionError("Expected LLMGenerationError was not raised on client failure!")
    except LLMGenerationError as e:
        assert "Simulated network timeout" in str(e)
        print(f"  -> SUCCESS: Exception cleanly caught and converted: {type(e).__name__}: {e}\n")

    # Also test with missing API key when no client is passed
    try:
        generate_sql(
            question="What is the top selling category?",
            llm_client=None,
            api_key="",  # empty key override
        )
        raise AssertionError("Expected LLMGenerationError when API key is missing or empty")
    except LLMGenerationError as e:
        assert "GEMINI_API_KEY is not configured" in str(e)
        print(f"  -> SUCCESS: Missing API key cleanly caught as LLMGenerationError: {e}\n")

    # -----------------------------------------------------------------
    # TEST 5: Mocked SQL for 'Which product generated the highest revenue?'
    # -----------------------------------------------------------------
    print("[TEST 5] Verifying analytical concepts for 'Which product generated the highest revenue?'...")
    mock_highest_rev_sql = """
    SELECT product,
           SUM(revenue) AS total_revenue
    FROM sales
    GROUP BY product
    ORDER BY total_revenue DESC
    LIMIT 1;
    """

    sql_out = generate_sql(
        question="Which product generated the highest revenue?",
        llm_client=lambda p: mock_highest_rev_sql,
    )

    expected_concepts = ["sales", "product", "revenue", "SUM", "GROUP BY", "ORDER BY", "LIMIT"]
    sql_upper = sql_out.upper()
    for concept in expected_concepts:
        assert concept.upper() in sql_upper, f"Missing analytical concept '{concept}' in generated SQL:\n{sql_out}"

    # Also verify that the prompt sent to the LLM contains full schema details
    captured_prompt = []
    def recording_client(prompt):
        captured_prompt.append(prompt)
        return mock_highest_rev_sql

    generate_sql(
        question="Which product generated the highest revenue?",
        llm_client=recording_client,
    )
    assert len(captured_prompt) == 1
    prompt_text = captured_prompt[0]
    for col in ["order_id", "product", "category", "region", "quantity", "unit_price", "revenue"]:
        assert col in prompt_text, f"Schema column '{col}' missing from generated prompt!"
    assert "sales" in prompt_text, "Table name 'sales' missing from prompt!"

    # Execute this SQL in DuckDBEngine against the real sample dataset to confirm full analytical correctness
    df = load_dataset("sample_data/sample_sales.csv")
    engine = DuckDBEngine()
    engine.register_dataframe(df, table_name="sales")
    res_df = engine.execute_query(sql_out)
    engine.close()

    assert len(res_df) == 1, f"Expected exactly 1 row, got {len(res_df)}"
    assert "product" in res_df.columns and "total_revenue" in res_df.columns
    top_product = res_df.iloc[0]["product"]
    top_revenue = res_df.iloc[0]["total_revenue"]
    assert top_product == "Ergonomic Office Chair"
    assert top_revenue == 153000.0
    print(f"  -> SUCCESS: SQL generated and executed. Top product: {top_product} (Revenue: {top_revenue})\n")

    # -----------------------------------------------------------------
    # TEST 6: Verify nl_to_sql.py itself does NOT execute SQL
    # -----------------------------------------------------------------
    print("[TEST 6] Verifying that nl_to_sql.py itself does NOT execute SQL...")
    import src.analytics.nl_to_sql as nl_module

    # 1. Inspect imported modules in nl_to_sql
    module_src_file = nl_module.__file__
    with open(module_src_file, "r", encoding="utf-8") as f:
        src_content = f.read()

    # nl_to_sql must NOT import duckdb or call database execute methods
    assert "import duckdb" not in src_content, "nl_to_sql.py must NOT import duckdb!"
    assert "from duckdb" not in src_content, "nl_to_sql.py must NOT import from duckdb!"
    assert ".execute(" not in src_content, "nl_to_sql.py must NOT call .execute() on any database connection!"
    assert "DuckDBEngine" not in src_content, "nl_to_sql.py must NOT instantiate or call DuckDBEngine directly!"

    # 2. Verify generate_sql returns purely a string and never a DataFrame or cursor
    gen_result = generate_sql(
        question="How many orders were placed?",
        llm_client=lambda p: "SELECT COUNT(*) FROM sales;",
    )
    assert isinstance(gen_result, str), f"Expected str return type, got {type(gen_result).__name__}"
    assert not isinstance(gen_result, pd.DataFrame), "generate_sql must not return a DataFrame!"
    print("  -> SUCCESS: nl_to_sql.py contains no database connections or execution logic.\n")

    # -----------------------------------------------------------------
    # TEST 7: Verify generated INSERT/UPDATE/DELETE/DROP statements are rejected
    # -----------------------------------------------------------------
    print("[TEST 7] Verifying that mutating statements from LLM are blocked by DuckDBEngine...")
    mutating_payloads = [
        "DROP TABLE sales;",
        "DELETE FROM sales WHERE revenue < 5000;",
        "UPDATE sales SET revenue = 999999 WHERE product = 'Mechanical Keyboard';",
        "INSERT INTO sales (order_id, product) VALUES ('ORD-TEST', 'Hacked');",
        "TRUNCATE TABLE sales;",
        "ALTER TABLE sales DROP COLUMN revenue;",
    ]

    engine = DuckDBEngine()
    engine.register_dataframe(df, table_name="sales")

    for harmful_sql in mutating_payloads:
        # Step A: generate_sql only outputs the string (it does NOT execute it)
        generated_harmful = generate_sql(
            question="Dangerous command",
            llm_client=lambda p, s=harmful_sql: s,
        )
        assert generated_harmful == harmful_sql

        # Step B: Passing it to DuckDBEngine must be blocked by read-only security validation
        try:
            engine.execute_query(generated_harmful)
            raise AssertionError(f"SECURITY FAILURE: Mutating query was not blocked: {generated_harmful}")
        except DisallowedQueryError as e:
            # Expected behavior: strict read-only validation caught the mutation
            pass

    engine.close()
    print("  -> SUCCESS: All generated DDL/DML mutating statements were blocked by read-only security layer.\n")

    print("=" * 78)
    print(" ALL 7 NL-TO-SQL TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_tests()
