"""
Unit Tests for Phase 9C: AI Business Insights Engine (Offline).

Verifies prompt construction, data trust guardrails, anti-hallucination rules,
DataFrame immutability, error wrapping, truncation, and architectural isolation.
All tests are 100% offline and use mock LLM clients (zero Gemini API calls).
Uses standard Python assert/exceptions without requiring external test runners.
"""
import sys
import os
import inspect
import pandas as pd

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.analytics.insights import (
    generate_business_insights,
    build_insights_prompt,
    format_dataframe_for_prompt,
    InsightsError,
    EmptyResultError,
    LLMInsightGenerationError,
)
import src.analytics.insights as insights_module


# =====================================================================
# FIXTURES & MOCKS
# =====================================================================

def sample_sales_df() -> pd.DataFrame:
    """Returns a realistic sample aggregated DataFrame."""
    return pd.DataFrame({
        "region": ["East", "South", "North", "Central", "West"],
        "total_revenue": [207141.0, 159251.0, 112362.0, 82061.0, 80206.0],
    })


class MockLLMClient:
    """Mock client that records the prompt and returns a predetermined response."""
    def __init__(self, response_text: str = "- East had the highest revenue.\n- West had the lowest revenue."):
        self.response_text = response_text
        self.last_prompt = None

    def __call__(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response_text


class MockFailingClient:
    """Mock client that simulates an API / network error."""
    def __call__(self, prompt: str) -> str:
        raise RuntimeError("Network timeout or upstream LLM service error")


# =====================================================================
# TEST CASES
# =====================================================================

def test_1_normal_result_dataframe_generates_insights():
    """1. Normal result DataFrame generates insights via mock LLM client."""
    df = sample_sales_df()
    mock = MockLLMClient("- East leads sales with $207k.\n- Revenue is concentrated in top regions.")
    
    insights = generate_business_insights(
        question="What is total revenue by region?",
        result_df=df,
        sql="SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC;",
        llm_client=mock,
    )
    
    assert isinstance(insights, str), "Insights must be returned as a string"
    assert "- East leads sales with $207k." in insights
    assert mock.last_prompt is not None, "Mock client must receive the generated prompt"


def test_2_prompt_contains_original_question():
    """2. Prompt contains the original business question verbatim."""
    df = sample_sales_df()
    question = "Which region generated the highest revenue in Q3?"
    
    prompt = build_insights_prompt(question=question, result_df=df)
    assert question in prompt, f"Prompt must contain original question: {question}"


def test_3_prompt_contains_actual_result_columns():
    """3. Prompt contains all actual result columns."""
    df = sample_sales_df()
    prompt = build_insights_prompt(question="Regional breakdown", result_df=df)
    
    assert "region" in prompt, "Prompt must contain column name 'region'"
    assert "total_revenue" in prompt, "Prompt must contain column name 'total_revenue'"


def test_4_prompt_contains_actual_result_values():
    """4. Prompt contains actual result values."""
    df = sample_sales_df()
    prompt = build_insights_prompt(question="Regional breakdown", result_df=df)
    
    assert "East" in prompt, "Prompt must contain data value 'East'"
    assert "207141" in prompt, "Prompt must contain data value '207141'"
    assert "West" in prompt, "Prompt must contain data value 'West'"
    assert "80206" in prompt, "Prompt must contain data value '80206'"


def test_5_prompt_instructs_gemini_not_to_invent_facts():
    """5. Prompt contains explicit instructions not to invent causes or facts."""
    df = sample_sales_df()
    prompt = build_insights_prompt(question="Regional revenue", result_df=df)
    
    # Must instruct against inventing causality and customer/market facts
    assert "DO NOT INVENT CAUSALITY" in prompt
    assert "Do not invent causes" in prompt
    assert "OBSERVATION" in prompt
    assert "CAUSATION" in prompt


def test_6_prompt_instructs_gemini_not_to_recalculate_results():
    """6. Prompt instructs Gemini not to recalculate or replace numerical results."""
    df = sample_sales_df()
    prompt = build_insights_prompt(question="Regional revenue", result_df=df)
    
    assert "already been calculated by SQL/DuckDB" in prompt
    assert "Do not invent, recalculate, or replace" in prompt


def test_7_empty_dataframe_handled_safely():
    """7. Empty DataFrame raises EmptyResultError safely without misleading insights."""
    empty_df = pd.DataFrame()
    mock = MockLLMClient("Some insight")
    
    caught_empty = False
    try:
        generate_business_insights(
            question="What is total revenue?",
            result_df=empty_df,
            llm_client=mock,
        )
    except EmptyResultError:
        caught_empty = True
    assert caught_empty, "Expected EmptyResultError for empty DataFrame"

    # Also test None DataFrame
    caught_none = False
    try:
        generate_business_insights(
            question="What is total revenue?",
            result_df=None,
            llm_client=mock,
        )
    except EmptyResultError:
        caught_none = True
    assert caught_none, "Expected EmptyResultError for None DataFrame"


def test_8_gemini_failure_wrapped_into_clear_exception():
    """8. LLM client/API failure is wrapped into LLMInsightGenerationError."""
    df = sample_sales_df()
    failing_mock = MockFailingClient()
    
    caught_error = False
    try:
        generate_business_insights(
            question="What is total revenue by region?",
            result_df=df,
            llm_client=failing_mock,
        )
    except LLMInsightGenerationError as e:
        caught_error = True
        assert "execution failed" in str(e).lower() or "network" in str(e).lower() or "error" in str(e).lower()
    assert caught_error, "Expected LLMInsightGenerationError on client failure"


def test_9_large_result_limited_before_sending_to_gemini():
    """9. Large result DataFrame is truncated in the prompt to prevent payload overflow."""
    large_df = pd.DataFrame({
        "item_id": [f"ITEM_{i:04d}" for i in range(100)],
        "sales_count": [i * 10 for i in range(100)],
    })
    
    prompt = build_insights_prompt(
        question="Top items by sales",
        result_df=large_df,
        max_rows=15,
    )
    
    # First rows must be present
    assert "ITEM_0000" in prompt
    assert "ITEM_0014" in prompt
    # Truncation notice must be present
    assert "truncated" in prompt.lower()
    # Row beyond max_rows must NOT be in prompt
    assert "ITEM_0099" not in prompt


def test_10_original_dataframe_is_not_mutated():
    """10. Original DataFrame is not mutated in any way during insight generation."""
    df = sample_sales_df()
    original_copy = df.copy(deep=True)
    mock = MockLLMClient("- Key insight")
    
    generate_business_insights(
        question="What is regional revenue?",
        result_df=df,
        sql="SELECT region, total_revenue FROM sales",
        llm_client=mock,
    )
    
    # Strict immutability check
    assert original_copy.equals(df), "Result DataFrame was mutated!"
    assert list(original_copy.columns) == list(df.columns)
    assert len(original_copy) == len(df)


def test_11_sql_is_optional_context():
    """11. SQL query is optional context; included when provided, omitted cleanly when None."""
    df = sample_sales_df()
    
    # Provided SQL
    prompt_with_sql = build_insights_prompt(
        question="Regional revenue",
        result_df=df,
        sql="SELECT region, SUM(revenue) FROM sales GROUP BY region;",
    )
    assert "SELECT region, SUM(revenue) FROM sales GROUP BY region;" in prompt_with_sql
    assert "Executed SQL Query:" in prompt_with_sql
    
    # Omitted SQL
    prompt_without_sql = build_insights_prompt(
        question="Regional revenue",
        result_df=df,
        sql=None,
    )
    assert "Executed SQL Query" not in prompt_without_sql
    assert "SELECT" not in prompt_without_sql


def test_12_no_duckdb_import_or_connection_in_insights():
    """12. Strictly verify insights.py contains zero duckdb imports or connections."""
    source = inspect.getsource(insights_module)
    assert "import duckdb" not in source, "insights.py must not import duckdb"
    assert "from duckdb" not in source, "insights.py must not import from duckdb"
    assert "DuckDBEngine" not in source, "insights.py must not reference DuckDBEngine"


def test_13_no_streamlit_import_in_insights():
    """13. Strictly verify insights.py contains zero streamlit imports or UI code."""
    source = inspect.getsource(insights_module)
    assert "import streamlit" not in source, "insights.py must not import streamlit"
    assert "from streamlit" not in source, "insights.py must not import from streamlit"
    assert "st." not in source, "insights.py must not use streamlit commands"


def test_14_empty_question_raises_error():
    """Empty or whitespace-only question raises InsightsError."""
    df = sample_sales_df()
    mock = MockLLMClient()
    
    caught = False
    try:
        generate_business_insights(question="   ", result_df=df, llm_client=mock)
    except InsightsError:
        caught = True
    assert caught, "Expected InsightsError for whitespace question"


def test_15_empty_llm_response_raises_error():
    """Empty response from LLM client raises LLMInsightGenerationError."""
    df = sample_sales_df()
    mock = MockLLMClient("")
    
    caught = False
    try:
        generate_business_insights(question="Top regions?", result_df=df, llm_client=mock)
    except LLMInsightGenerationError:
        caught = True
    assert caught, "Expected LLMInsightGenerationError for empty LLM response"


def test_16_handles_nan_and_null_safely():
    """Missing or NaN values in result DataFrame are formatted cleanly as NULL."""
    df_with_nan = pd.DataFrame({
        "category": ["Electronics", "Clothing", None],
        "profit": [5000.0, None, 1200.0],
    })
    
    formatted = format_dataframe_for_prompt(df_with_nan)
    assert "NULL" in formatted
    assert "Electronics" in formatted


# =====================================================================
# RUNNER
# =====================================================================

def run_all_insights_tests():
    print("=" * 78)
    print(" AI BUSINESS INSIGHTS ENGINE (PHASE 9C) OFFLINE TEST SUITE")
    print("=" * 78)
    tests = [
        test_1_normal_result_dataframe_generates_insights,
        test_2_prompt_contains_original_question,
        test_3_prompt_contains_actual_result_columns,
        test_4_prompt_contains_actual_result_values,
        test_5_prompt_instructs_gemini_not_to_invent_facts,
        test_6_prompt_instructs_gemini_not_to_recalculate_results,
        test_7_empty_dataframe_handled_safely,
        test_8_gemini_failure_wrapped_into_clear_exception,
        test_9_large_result_limited_before_sending_to_gemini,
        test_10_original_dataframe_is_not_mutated,
        test_11_sql_is_optional_context,
        test_12_no_duckdb_import_or_connection_in_insights,
        test_13_no_streamlit_import_in_insights,
        test_14_empty_question_raises_error,
        test_15_empty_llm_response_raises_error,
        test_16_handles_nan_and_null_safely,
    ]

    passed = 0
    failed = 0
    for idx, t in enumerate(tests, 1):
        print(f"[{idx:02d}/{len(tests):02d}] {t.__name__}...", end=" ")
        try:
            t()
            print("PASS")
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print("\n" + "-" * 78)
    print(f"Summary: {passed} passed, {failed} failed out of {len(tests)} tests.")
    print("-" * 78)
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL Phase 9C offline tests passed successfully!\n")


if __name__ == "__main__":
    run_all_insights_tests()
