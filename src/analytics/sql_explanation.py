"""
Deterministic SQL Explanation & Transparency Module.

Translates generated, validated SQL queries into clear, concise,
and business-friendly explanations for non-technical stakeholders.

DESIGN & ARCHITECTURAL PRINCIPLES:
- 100% Local & Deterministic: Generates human-readable explanations directly
  from the actual SQL query without making any Gemini API calls.
- Zero SQL execution: This module NEVER connects to DuckDB or PostgreSQL and NEVER executes SQL.
- Grounded strictly in the generated SQL: The SQL query is the sole source of truth.
- Zero hallucination of operations: Never invents filters, joins, metrics, or causality.
- Compact & Fast: Generates explanations in sub-millisecond local execution time.
- Pure-Python module: Contains zero Streamlit UI code.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from src.analytics.nl_to_sql import format_schema, LLMGenerationError


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
# DETERMINISTIC SQL EXPLANATION ENGINE
# =====================================================================

def _strip_sql_comments(sql: str) -> str:
    """Removes single-line and multi-line SQL comments."""
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--.*$", " ", no_block, flags=re.MULTILINE)
    return " ".join(no_line.split())


def _extract_clause(sql: str, start_kw: str, stop_kws: List[str]) -> Optional[str]:
    """
    Extracts the text of a SQL clause starting with start_kw up to any stop_kws.
    Case-insensitive search.
    """
    if stop_kws:
        pattern = rf"\b{start_kw}\b\s+(.*?)(?=\b(?:{'|'.join(stop_kws)})\b|;|$)"
    else:
        pattern = rf"\b{start_kw}\b\s+(.*?)(?=;|$)"
    match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_aggregates(select_clause: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Extracts aggregate functions (func, column_expr, alias) from a SELECT clause.
    Supports SUM, AVG, COUNT, MIN, MAX.
    """
    agg_pattern = re.compile(
        r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*(DISTINCT\s+[^)]+|\*|[^)]+)\s*\)(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?",
        re.IGNORECASE,
    )
    results = []
    for match in agg_pattern.finditer(select_clause):
        func = match.group(1).upper()
        expr = match.group(2).strip()
        alias = match.group(3) if match.group(3) else None
        results.append((func, expr, alias))
    return results


def _extract_selected_dimensions(select_clause: str) -> List[str]:
    """Extracts non-aggregate column names selected in the query."""
    # Split select clause by top-level commas (outside parentheses)
    items = []
    current: List[str] = []
    paren_depth = 0
    for char in select_clause:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "," and paren_depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())

    dimensions = []
    agg_names = {"SUM", "AVG", "COUNT", "MIN", "MAX"}
    for item in items:
        # Check if item contains an aggregate function
        has_agg = any(re.search(rf"\b{a}\b", item, re.IGNORECASE) for a in agg_names)
        if not has_agg and item and item != "*":
            # Extract column identifier before optional alias
            m = re.match(r"^([a-zA-Z0-9_\.]+)(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?", item, re.IGNORECASE)
            if m:
                col_name = m.group(1)
                # Strip table prefix if present (e.g. sales.region -> region)
                if "." in col_name:
                    col_name = col_name.split(".")[-1]
                dimensions.append(col_name)
    return dimensions


def explain_sql_deterministically(sql: str, question: Optional[str] = None) -> str:
    """
    Analyzes the structure and clauses of a validated SQL query to produce a
    grounded, deterministic explanation formatted as Markdown.

    Zero LLM calls, zero database connections.
    """
    clean_sql = _strip_sql_comments(sql).strip()
    if clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1].strip()

    # Identify clauses
    stop_clauses = ["WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]

    select_clause = _extract_clause(clean_sql, "SELECT", ["FROM"]) or ""
    from_clause = _extract_clause(clean_sql, "FROM", stop_clauses) or ""
    where_clause = _extract_clause(clean_sql, "WHERE", ["GROUP BY", "HAVING", "ORDER BY", "LIMIT"])
    group_by_clause = _extract_clause(clean_sql, "GROUP BY", ["HAVING", "ORDER BY", "LIMIT"])
    order_by_clause = _extract_clause(clean_sql, "ORDER BY", ["LIMIT"])
    limit_clause = _extract_clause(clean_sql, "LIMIT", [])

    # Identify Joins
    joins = []
    join_matches = re.finditer(r"\b(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|JOIN)\s+([a-zA-Z0-9_]+)(?:\s+ON\s+([^,]+?))?(?=\b(?:LEFT|RIGHT|INNER|JOIN|WHERE|GROUP|HAVING|ORDER|LIMIT|$))", from_clause, re.IGNORECASE)
    for jm in join_matches:
        j_type = jm.group(1).upper()
        j_table = jm.group(2)
        j_on = jm.group(3).strip() if jm.group(3) else None
        joins.append((j_type, j_table, j_on))

    # Base table name (first token of from_clause)
    table_name = "the dataset"
    if from_clause:
        first_token = from_clause.split()[0]
        if first_token.isidentifier():
            table_name = first_token

    # Analyze Aggregates & Dimensions
    aggregates = _extract_aggregates(select_clause)
    dimensions = _extract_selected_dimensions(select_clause)

    # -------------------------------------------------------------
    # 1. Build "How it works" bullets
    # -------------------------------------------------------------
    bullets: List[str] = []

    # Table & Joins
    if table_name != "the dataset":
        bullets.append(f"Retrieves records from the `{table_name}` table.")
    else:
        bullets.append("Retrieves records from the active dataset.")

    for j_type, j_table, j_on in joins:
        if j_on:
            bullets.append(f"Joins `{j_table}` on `{j_on}`.")
        else:
            bullets.append(f"Joins `{j_table}`.")

    # WHERE filter
    if where_clause:
        bullets.append(f"Filters records where `{where_clause}`.")

    # Aggregations
    for func, expr, alias in aggregates:
        func_upper = func.upper()
        clean_expr = expr.replace("DISTINCT ", "distinct ").strip()
        alias_str = f" as `{alias}`" if alias else ""
        if func_upper == "SUM":
            bullets.append(f"Calculates total {clean_expr} using `SUM({expr})`{alias_str}.")
        elif func_upper == "AVG":
            bullets.append(f"Calculates average {clean_expr} using `AVG({expr})`{alias_str}.")
        elif func_upper == "COUNT":
            if clean_expr == "*":
                bullets.append(f"Counts total matching records using `COUNT(*)`{alias_str}.")
            else:
                bullets.append(f"Counts occurrences of {clean_expr} using `COUNT({expr})`{alias_str}.")
        elif func_upper == "MIN":
            bullets.append(f"Finds the minimum {clean_expr} using `MIN({expr})`{alias_str}.")
        elif func_upper == "MAX":
            bullets.append(f"Finds the maximum {clean_expr} using `MAX({expr})`{alias_str}.")

    # GROUP BY
    if group_by_clause:
        bullets.append(f"Groups results by `{group_by_clause}`.")

    # ORDER BY
    if order_by_clause:
        is_desc = "DESC" in order_by_clause.upper()
        col_sort = re.sub(r"\b(ASC|DESC)\b", "", order_by_clause, flags=re.IGNORECASE).strip()
        direction = "descending order (highest first)" if is_desc else "ascending order (lowest first)"
        bullets.append(f"Orders results by `{col_sort}` in {direction}.")

    # LIMIT
    if limit_clause:
        bullets.append(f"Limits the output to the top `{limit_clause}` records.")

    # -------------------------------------------------------------
    # 2. Build "What the query does" summary sentence
    # -------------------------------------------------------------
    summary_parts: List[str] = []

    # Filter prefix
    if where_clause:
        summary_parts.append(f"Filters records matching `{where_clause}` and")

    # Aggregate description
    agg_summaries: List[str] = []
    for func, expr, alias in aggregates:
        clean_expr = expr.replace("DISTINCT ", "").strip()
        f_up = func.upper()
        if f_up == "SUM":
            agg_summaries.append(f"the total {clean_expr}")
        elif f_up == "AVG":
            agg_summaries.append(f"the average {clean_expr}")
        elif f_up == "COUNT":
            agg_summaries.append(f"total count of {clean_expr}")
        elif f_up == "MIN":
            agg_summaries.append(f"the minimum {clean_expr}")
        elif f_up == "MAX":
            agg_summaries.append(f"the maximum {clean_expr}")

    agg_text = ", ".join(agg_summaries) if agg_summaries else "selected metrics"

    if group_by_clause:
        dim_text = group_by_clause
        if agg_summaries:
            summary_parts.append(f"calculates {agg_text} grouped by {dim_text}")
        else:
            summary_parts.append(f"groups records by {dim_text}")
    elif agg_summaries:
        summary_parts.append(f"calculates {agg_text} across all records")
    else:
        if dimensions:
            summary_parts.append(f"retrieves {', '.join(dimensions)}")
        else:
            summary_parts.append("retrieves data")

    # Order/Limit suffix
    if order_by_clause:
        is_desc = "DESC" in order_by_clause.upper()
        sort_col = re.sub(r"\b(ASC|DESC)\b", "", order_by_clause, flags=re.IGNORECASE).strip()
        order_desc = f"ordered by {sort_col} descending" if is_desc else f"ordered by {sort_col} ascending"
        summary_parts.append(f", {order_desc}")

    if limit_clause:
        summary_parts.append(f" (limited to top {limit_clause} results)")

    raw_summary = " ".join(summary_parts).strip()
    raw_summary = re.sub(r"\s+,", ",", raw_summary)
    raw_summary = re.sub(r"\s+", " ", raw_summary)
    # Clean up leading lowercase or awkward joining
    if raw_summary.startswith("and "):
        raw_summary = raw_summary[4:]
    if raw_summary:
        full_summary = raw_summary[0].upper() + raw_summary[1:] + "."
    else:
        full_summary = "Executes an analytical query against the dataset."

    # Format output as standard Markdown
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    return f"### What the query does\n{full_summary}\n\n### How it works\n{bullet_text}"


# =====================================================================
# PROMPT CONSTRUCTION (BACKWARD COMPATIBILITY & VERIFICATION)
# =====================================================================

def build_sql_explanation_prompt(
    question: str,
    sql: str,
    schema: Optional[Union[Dict[str, Any], str]] = None,
) -> str:
    """
    Constructs a concise, strictly grounded prompt instructing an LLM to explain
    the provided SQL query in plain business terms.

    Preserved for backward compatibility with verification scripts and tests.
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
   - If the query has SUM, AVG, COUNT, MIN, MAX: explain the specific mathematical aggregation.
   - If the query has ORDER BY: explain how the rows are sorted.
   - If the query has LIMIT: explain that only the top/limited number of results are shown.
   - Avoid database jargon.
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
# CORE SQL EXPLANATION ENTRYPOINT
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

    By default, generates deterministic human-readable explanations locally
    without consuming Gemini API quota. If a custom llm_client is explicitly
    provided (e.g. for dependency injection / mock testing), invokes the client.

    Parameters:
        question: The user's original business question.
        sql: The validated, read-only SQL query to explain.
        schema: Optional database schema mapping for reference.
        llm_client: Optional custom LLM client or callable (for mock testing).
        api_key: Optional Gemini API key override (ignored in deterministic mode).
        model: Model identifier (ignored in deterministic mode).

    Returns:
        str: Business-friendly explanation formatted as Markdown.

    Raises:
        EmptySQLExplanationError: If question or sql is empty or whitespace.
        LLMExplanationGenerationError: If a custom llm_client fails.
    """
    # 1. Validate question
    if question is None or not isinstance(question, str) or not question.strip():
        raise EmptySQLExplanationError("Question cannot be empty or contain only whitespace.")

    # 2. Validate SQL
    if sql is None or not isinstance(sql, str) or not sql.strip():
        raise EmptySQLExplanationError("SQL query cannot be empty or contain only whitespace.")

    # 3. Custom llm_client provided (for mock testing / dependency injection)
    if llm_client is not None:
        prompt = build_sql_explanation_prompt(
            question=question,
            sql=sql,
            schema=schema,
        )
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
                    f"Provided llm_client '{type(llm_client).__name__}' does not provide a supported interface."
                )
        except Exception as e:
            if isinstance(e, SQLExplanationError):
                raise
            raise LLMExplanationGenerationError(f"LLM client execution failed: {str(e)}") from e

        if not raw_response or not raw_response.strip():
            raise LLMExplanationGenerationError("LLM client returned an empty explanation.")
        return raw_response.strip()

    # 4. Deterministic local explanation (no LLM, no API call, no DB connection)
    return explain_sql_deterministically(sql=sql, question=question)
