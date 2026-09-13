"""
SQL Explanation & Transparency Module (Phase 11D).

Translates generated, validated DuckDB SQL queries into clear, concise,
and business-friendly explanations for non-technical stakeholders.

DESIGN & ARCHITECTURAL PRINCIPLES:
- Zero SQL execution: This module NEVER connects to DuckDB and NEVER executes SQL.
- Grounded strictly in the generated SQL: The SQL query is the sole source of truth.
- Zero hallucination of operations: Never invents filters, joins, metrics, or causality.
- Compact prompt: Receives only the business question, generated SQL, and optional schema.
  Never receives raw DataFrames, full result rows, or conversation history.
- Pure-Python module: Contains zero Streamlit UI code.
"""

from typing import Any, Dict, List, Optional, Union
from src.analytics.nl_to_sql import format_schema, _call_gemini_sdk, LLMGenerationError


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class SQLExplanationError(Exception):
    """Base exception for all SQL explanation generation errors."""
    pass


class EmptySQLExplanationError(SQLExplanationError):
    """Raised when the question or SQL query is empty or contains only whitespace."""
    pass


class LLMExplanationGenerationError(SQLExplanationError):
    """Raised when the LLM provider fails, times out, or returns an invalid response."""
    pass


# =====================================================================
# PROMPT CONSTRUCTION
# =====================================================================

def build_sql_explanation_prompt(
    question: str,
    sql: str,
    schema: Optional[Union[Dict[str, Any], str]] = None,
) -> str:
    """
    Constructs a concise, strictly grounded prompt instructing Gemini to explain
    the provided SQL query in plain business terms.

    Enforces critical transparency and anti-hallucination guardrails:
    1. The provided SQL query is the absolute source of truth.
    2. Never invent filters, columns, joins, metrics, or causality not present in the SQL.
    3. Translate technical clauses (WHERE, GROUP BY, aggregations, ORDER BY, LIMIT)
       into plain business language without unnecessary database jargon.
    4. Do not output raw SQL or code blocks.
    """
    cleaned_question = question.strip() if question else ""
    cleaned_sql = sql.strip() if sql else ""

    schema_block = ""
    if schema:
        try:
            formatted = format_schema(schema)
            schema_block = f"\nDATABASE SCHEMA (Authoritative reference):\n{formatted}\n"
        except Exception:
            schema_block = ""

    prompt = f"""You are an expert Data Analytics Translator helping non-technical business stakeholders understand how an AI system answered their question using SQL.

Original Business Question:
"{cleaned_question}"
{schema_block}
Generated SQL Query to Explain:
```sql
{cleaned_sql}
```

CRITICAL GROUNDING RULES (MANDATORY):
1. THE SQL IS THE SOURCE OF TRUTH: Explain ONLY what this specific SQL query actually does.
2. DO NOT INVENT OPERATIONS:
   - Do NOT invent or assume filters, conditions, or WHERE clauses not in the query.
   - Do NOT invent joins, tables, or columns not present in the SQL.
   - Do NOT invent business conclusions, recommendations, or causality.
   - Do NOT calculate or claim additional metrics.
3. BUSINESS-FRIENDLY TRANSLATION:
   - If the query has WHERE: explain the exact filtering condition in plain English.
   - If the query has GROUP BY: explain that the data is grouped into categories/regions/segments.
   - If the query has SUM, AVG, COUNT, MIN, MAX: explain the specific mathematical aggregation (e.g. "calculates total revenue", "averages order values", "counts total orders").
   - If the query has ORDER BY: explain how the rows are sorted (e.g. "from highest to lowest revenue").
   - If the query has LIMIT: explain that only the top/limited number of results are shown.
   - If the query has DISTINCT: explain that duplicate values are removed.
   - If the query has CASE: explain the conditional logic applied.
   - Avoid database jargon (prefer "groups sales by region" over "performs a GROUP BY clause").
4. OUTPUT FORMAT:
   - Return structured Markdown with exactly two sections:
     ### What the query does
     <1 to 2 clear sentences summarizing the overall business action>

     ### How it works
     - <Bullet point explaining step 1 (filtering/table used)>
     - <Bullet point explaining step 2 (grouping/aggregations)>
     - <Bullet point explaining step 3 (sorting/limiting if present)>
5. STRICT CONSTRAINTS:
   - Do NOT output SQL code blocks (do NOT use ```sql).
   - Do NOT generate a replacement or alternative SQL query.
   - Output ONLY the Markdown explanation.
"""
    return prompt.strip()


# =====================================================================
# CORE SQL EXPLANATION FUNCTION
# =====================================================================

def generate_sql_explanation(
    question: str,
    sql: str,
    schema: Optional[Union[Dict[str, Any], str]] = None,
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Generates a concise, business-friendly explanation of a validated SQL query.

    Parameters:
        question: The user's original business question.
        sql: The validated, read-only SQL query to explain.
        schema: Optional database schema mapping for reference.
        llm_client: Optional custom LLM client or callable (for mocking/dependency injection).
        api_key: Optional Gemini API key override. If None, resolves from GEMINI_API_KEY.
        model: Gemini model identifier (default: "gemini-2.5-flash").

    Returns:
        str: Business-friendly explanation formatted as Markdown.

    Raises:
        EmptySQLExplanationError: If question or sql is empty or whitespace.
        LLMExplanationGenerationError: If LLM invocation or parsing fails.
    """
    # 1. Validate question
    if question is None or not isinstance(question, str) or not question.strip():
        raise EmptySQLExplanationError("Question cannot be empty or contain only whitespace.")

    # 2. Validate SQL
    if sql is None or not isinstance(sql, str) or not sql.strip():
        raise EmptySQLExplanationError("SQL query cannot be empty or contain only whitespace.")

    # 3. Build structured prompt
    prompt = build_sql_explanation_prompt(
        question=question,
        sql=sql,
        schema=schema,
    )

    # 4. Invoke LLM (mock client or official Gemini SDK)
    raw_response: str = ""

    if llm_client is not None:
        try:
            if callable(llm_client):
                raw_response = llm_client(prompt)
            elif hasattr(llm_client, "generate_explanation"):
                raw_response = llm_client.generate_explanation(prompt)
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
                raise LLMExplanationGenerationError(
                    f"Provided llm_client '{type(llm_client).__name__}' does not provide a callable "
                    f"interface or supported methods (generate_explanation, generate_content, generate, chat)."
                )
        except Exception as e:
            if isinstance(e, SQLExplanationError):
                raise
            raise LLMExplanationGenerationError(f"LLM client execution failed: {str(e)}") from e
    else:
        try:
            raw_response = _call_gemini_sdk(
                prompt=prompt,
                api_key=api_key,
                model=model,
            )
        except LLMGenerationError as e:
            raise LLMExplanationGenerationError(f"Gemini SQL explanation failed: {str(e)}") from e
        except Exception as e:
            raise LLMExplanationGenerationError(f"Unexpected error during SQL explanation generation: {str(e)}") from e

    if not raw_response or not raw_response.strip():
        raise LLMExplanationGenerationError("Gemini returned an empty SQL explanation response.")

    return raw_response.strip()
