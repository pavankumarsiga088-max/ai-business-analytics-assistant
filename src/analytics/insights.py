"""
AI Business Insights Module.

Generates concise, evidence-based business insights from pre-computed SQL query results.

DESIGN & ARCHITECTURAL PRINCIPLES:
- Gemini is strictly an INTERPRETER, not a calculator.
- SQL/DuckDB performs 100% of mathematical aggregations and data retrieval.
- Gemini is explicitly instructed that results are pre-calculated and must not be recalculated.
- Gemini must never invent causality, market conditions, or customer behavior without evidence.
- This module NEVER executes SQL, NEVER connects to DuckDB, and NEVER imports Streamlit.
- The input DataFrame is treated as strictly read-only and immutable.
"""
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.analytics.nl_to_sql import _call_gemini_sdk, LLMGenerationError


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class InsightsError(Exception):
    """Base exception for all business insight generation errors."""
    pass


class EmptyResultError(InsightsError):
    """Raised when the query result DataFrame is empty or contains no data."""
    pass


class LLMInsightGenerationError(InsightsError):
    """Raised when the LLM provider fails, times out, or returns an invalid response."""
    pass


# =====================================================================
# DATA SERIALIZATION & PROMPT CONSTRUCTION
# =====================================================================

def format_dataframe_for_prompt(df: pd.DataFrame, max_rows: int = 25) -> str:
    """
    Serializes a DataFrame into a compact, human-readable tabular representation
    suitable for LLM prompts.

    Guarantees:
    - Never mutates the input DataFrame.
    - Limits prompt payload size by capping rows at max_rows.
    - Formats missing values cleanly as NULL.
    """
    if df is None or df.empty:
        return "(empty result set)"

    truncated = len(df) > max_rows
    view = df.iloc[:max_rows]

    columns = [str(col) for col in view.columns]
    lines: List[str] = [
        f"Columns ({len(columns)}): {', '.join(columns)}",
        "Rows:",
    ]

    for _, row in view.iterrows():
        row_vals = [str(val) if pd.notna(val) else "NULL" for val in row]
        lines.append(" | ".join(row_vals))

    if truncated:
        lines.append(f"... (truncated: showing first {max_rows} of {len(df)} total rows)")

    return "\n".join(lines)


def build_insights_prompt(
    question: str,
    result_df: pd.DataFrame,
    sql: Optional[str] = None,
    max_rows: int = 25,
) -> str:
    """
    Builds a structured prompt instructing Gemini to interpret the pre-computed SQL results.

    Enforces critical data trust and anti-hallucination guardrails:
    1. Pre-calculated data must not be recalculated or altered.
    2. Observations must be strictly supported by the supplied data.
    3. Causation must not be invented without direct empirical evidence.
    4. Observations must be distinguished from business recommendations.
    """
    cleaned_question = question.strip() if question else ""
    data_str = format_dataframe_for_prompt(result_df, max_rows=max_rows)

    sql_block = ""
    if sql and sql.strip():
        sql_block = f"\nExecuted SQL Query:\n```sql\n{sql.strip()}\n```\n"

    prompt = f"""You are an expert Data Analyst interpreting pre-computed query results for business stakeholders.

Original Business Question:
{cleaned_question}
{sql_block}
Query Result Data:
{data_str}

CRITICAL RULES & DATA TRUST (MANDATORY):
1. PRE-CALCULATED RESULTS: The supplied result data has already been calculated by SQL/DuckDB. Do not invent, recalculate, or replace the numerical results.
2. EVIDENCE-BASED ONLY: Ground every statement strictly in the provided data. Only make statements supported by the supplied data.
3. DO NOT INVENT CAUSALITY:
   - Clearly distinguish OBSERVATION (e.g., "Region A generated the highest revenue") from unsupported CAUSATION (e.g., "because Region A has higher customer loyalty").
   - Do not invent causes, customer behavior, market conditions, economic factors, or business facts not present in the data.
   - Do not claim causation without direct evidence.
4. RECOMMENDATIONS: Clearly distinguish factual observations from business recommendations or next steps to investigate.
5. SUFFICIENCY: If the result is small, simple, or contains insufficient information for complex patterns, state the key facts plainly rather than inventing speculative insights.

OUTPUT FORMAT:
- Return approximately 3 to 5 concise, actionable bullet points in Markdown format.
- Return ONLY the business insights. Do NOT return SQL, Python code, JSON, or chart configurations.
"""
    return prompt.strip()


# =====================================================================
# CORE BUSINESS INSIGHTS GENERATION FUNCTION
# =====================================================================

def generate_business_insights(
    question: str,
    result_df: pd.DataFrame,
    sql: Optional[str] = None,
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    max_rows: int = 25,
) -> str:
    """
    Generates concise, evidence-based business insights from pre-computed SQL results.

    Parameters:
        question: The user's original business question.
        result_df: The actual Pandas DataFrame produced by DuckDB execution.
        sql: Optional executed SQL query string for additional context.
        llm_client: Optional custom LLM client or callable (for mocking/dependency injection).
        api_key: Optional Gemini API key override. If None, resolves from GEMINI_API_KEY.
        model: Gemini model identifier (default: "gemini-2.5-flash").
        max_rows: Maximum number of rows to include in the LLM prompt.

    Returns:
        str: Concise business insights formatted as Markdown bullet points.

    Raises:
        EmptyResultError: If result_df is None or empty.
        InsightsError: If the question is empty or invalid.
        LLMInsightGenerationError: If LLM invocation or parsing fails.
    """
    # 1. Validate result DataFrame
    if result_df is None or not isinstance(result_df, pd.DataFrame) or result_df.empty:
        raise EmptyResultError("Cannot generate insights for an empty or missing result DataFrame.")

    # 2. Validate question
    if question is None or not isinstance(question, str) or not question.strip():
        raise InsightsError("Question cannot be empty or contain only whitespace.")

    # 3. Build structured prompt (guarantees immutability of result_df)
    prompt = build_insights_prompt(
        question=question,
        result_df=result_df,
        sql=sql,
        max_rows=max_rows,
    )

    # 4. Invoke LLM (mock client or official Gemini SDK)
    raw_response: str = ""

    if llm_client is not None:
        try:
            if callable(llm_client):
                raw_response = llm_client(prompt)
            elif hasattr(llm_client, "generate_insights"):
                raw_response = llm_client.generate_insights(prompt)
            elif hasattr(llm_client, "models") and hasattr(llm_client.models, "generate_content"):
                res = llm_client.models.generate_content(model=model, contents=prompt)
                raw_response = getattr(res, "text", str(res))
            elif hasattr(llm_client, "generate_content"):
                res = llm_client.generate_content(prompt)
                raw_response = getattr(res, "text", str(res))
            elif hasattr(llm_client, "generate"):
                raw_response = llm_client.generate(prompt)
            elif hasattr(llm_client, "chat"):
                raw_response = llm_client.chat(prompt)
            else:
                raise LLMInsightGenerationError(
                    f"Provided llm_client '{type(llm_client).__name__}' does not provide a callable "
                    f"interface or supported methods (generate_insights, generate_content, generate, chat)."
                )
        except Exception as e:
            if isinstance(e, InsightsError):
                raise
            raise LLMInsightGenerationError(f"LLM client execution failed: {str(e)}") from e
    else:
        try:
            raw_response = _call_gemini_sdk(
                prompt=prompt,
                api_key=api_key,
                model=model,
            )
        except LLMGenerationError as e:
            raise LLMInsightGenerationError(f"Gemini insight generation failed: {str(e)}") from e
        except Exception as e:
            raise LLMInsightGenerationError(f"Unexpected error during insight generation: {str(e)}") from e

    if not raw_response or not raw_response.strip():
        raise LLMInsightGenerationError("Gemini returned an empty insight response.")

    return raw_response.strip()
