"""
Dataset-Aware Question Suggestion Engine.

Analyzes dataset column names and data types to deterministically generate
domain-specific, natural-language analytical questions without calling an LLM.

DESIGN & ARCHITECTURAL PRINCIPLES:
- 100% Deterministic: Zero randomness, zero LLM / external API calls.
- Read-Only & Pure: Never mutates the input dataset or schema.
- Schema Agnostic: Works with pandas DataFrames, schema dictionaries, or column lists.
- Domain Adaptive: Identifies semantic concepts (agriculture, e-commerce, finance,
  operations) from column naming patterns and generates tailored questions.
- Graceful Fallback: Returns sensible generic analytical questions if no recognizable
  domain columns are detected.
"""

import re
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd


# =====================================================================
# SEMANTIC KEYWORD DEFINITIONS
# =====================================================================

REVENUE_PATTERNS: Set[str] = {
    "revenue", "sales", "sales_amount", "total_sales", "income", "turnover"
}

PROFIT_PATTERNS: Set[str] = {
    "profit", "net_profit", "profit_amount", "margin", "net_income"
}

CROP_PATTERNS: Set[str] = {
    "crop", "crops", "variety", "crop_name", "crop_type"
}

PRODUCT_PATTERNS: Set[str] = {
    "product", "product_name", "item", "item_name", "sku", "category"
}

LOCATION_PATTERNS: Set[str] = {
    "state", "region", "district", "city", "location", "territory", "zone", "country"
}

SEASON_PATTERNS: Set[str] = {
    "season", "month", "year", "date", "order_date", "quarter"
}

PRODUCTION_PATTERNS: Set[str] = {
    "production", "quantity", "units", "output", "tonnes", "tons", "volume"
}

YIELD_PATTERNS: Set[str] = {
    "yield", "yield_t_per_ha", "productivity", "efficiency"
}

WATER_PATTERNS: Set[str] = {
    "water", "water_used", "water_usage", "water_consumption"
}

IRRIGATION_PATTERNS: Set[str] = {
    "irrigation", "irrigation_method", "irrigation_type"
}

RISK_PATTERNS: Set[str] = {
    "disease", "pest", "disease_risk", "pest_risk", "risk", "hazard", "threat"
}

COST_PATTERNS: Set[str] = {
    "cost", "expense", "expenses", "production_cost", "unit_cost"
}

CUSTOMER_PATTERNS: Set[str] = {
    "customer", "customer_name", "customer_id", "client", "client_name"
}

ORDER_PATTERNS: Set[str] = {
    "order", "order_id", "orders", "transaction_id"
}


# =====================================================================
# COLUMN EXTRACTION & NORMALIZATION
# =====================================================================

def _normalize_column_name(col: str) -> str:
    """Normalizes column name to lowercase alphanumeric with single underscores."""
    col_str = str(col).strip().lower()
    # Strip non-alphanumeric except underscore
    col_clean = re.sub(r"[^\w\s]", "", col_str)
    # Convert spaces/hyphens to underscore
    col_clean = re.sub(r"[\s\-]+", "_", col_clean)
    return col_clean.strip("_")


def extract_column_names(
    schema: Optional[Union[Dict[str, Any], List[str], pd.DataFrame]]
) -> List[str]:
    """
    Extracts a list of column names safely from DataFrame, dict, or list.
    Guarantees zero mutation of input objects.
    """
    if schema is None:
        return []

    if isinstance(schema, pd.DataFrame):
        return [str(c) for c in schema.columns]

    if isinstance(schema, (list, tuple, set)):
        return [str(c) for c in schema]

    if isinstance(schema, dict):
        has_table_level = any(isinstance(v, (list, tuple, dict)) for v in schema.values())
        cols: List[str] = []
        if has_table_level:
            for v in schema.values():
                if isinstance(v, dict):
                    cols.extend(str(k) for k in v.keys())
                elif isinstance(v, (list, tuple)):
                    cols.extend(str(k) for k in v)
                elif isinstance(v, str):
                    cols.append(v)
            return cols
        else:
            return [str(k) for k in schema.keys()]

    return []


# =====================================================================
# CORE QUESTION SUGGESTION ENGINE
# =====================================================================

def generate_suggested_questions(
    schema: Optional[Union[Dict[str, Any], List[str], pd.DataFrame]] = None,
    max_questions: int = 6,
) -> List[str]:
    """
    Deterministically generates 4 to 6 natural-language analytical questions
    tailored to the schema and detected business concepts.

    Parameters:
        schema: A pandas DataFrame, dictionary schema, list of column names, or None.
        max_questions: Maximum number of suggested questions to return (default: 6).

    Returns:
        List[str]: List of 4-6 human-readable natural language questions.
    """
    raw_columns = extract_column_names(schema)
    if not raw_columns:
        return _get_fallback_questions(max_questions)

    normalized_cols = {_normalize_column_name(c): c for c in raw_columns}
    norm_set = set(normalized_cols.keys())

    # Detect presence of semantic concepts
    def find_matching_col(patterns: Set[str]) -> Optional[str]:
        for p in patterns:
            if p in norm_set:
                return normalized_cols[p]
        for col_norm in norm_set:
            for p in patterns:
                if p in col_norm or col_norm in p:
                    return normalized_cols[col_norm]
        return None

    def find_all_matching_cols(patterns: Set[str]) -> List[str]:
        matched = []
        for col_norm, original in normalized_cols.items():
            for p in patterns:
                if p == col_norm or p in col_norm:
                    matched.append(original)
                    break
        return matched

    revenue_col = find_matching_col(REVENUE_PATTERNS)
    profit_col = find_matching_col(PROFIT_PATTERNS)
    crop_col = find_matching_col(CROP_PATTERNS)
    product_col = find_matching_col(PRODUCT_PATTERNS)
    location_col = find_matching_col(LOCATION_PATTERNS)
    season_col = find_matching_col(SEASON_PATTERNS)
    production_col = find_matching_col(PRODUCTION_PATTERNS)
    yield_col = find_matching_col(YIELD_PATTERNS)
    water_col = find_matching_col(WATER_PATTERNS)
    irrigation_col = find_matching_col(IRRIGATION_PATTERNS)
    risk_cols = find_all_matching_cols(RISK_PATTERNS)
    cost_col = find_matching_col(COST_PATTERNS)
    customer_col = find_matching_col(CUSTOMER_PATTERNS)
    order_col = find_matching_col(ORDER_PATTERNS)

    questions: List[str] = []

    # 1. Overall Total Metric
    if revenue_col:
        questions.append("What is the total revenue?")
    elif profit_col:
        questions.append("What is the total profit?")
    elif production_col:
        questions.append(f"What is the total {production_col.lower()}?")

    # 2. Entity / Crop / Product Profitability or Revenue
    if crop_col and profit_col:
        questions.append("Which crop generates the highest profit?")
    elif crop_col and revenue_col:
        questions.append("Which crop generates the highest revenue?")
    elif product_col and profit_col:
        questions.append("Which product generates the highest profit?")
    elif product_col and revenue_col:
        # Check if sales dataset style or general
        if crop_col is None and location_col and location_col.lower() == "region":
            questions.append("Top 5 products")
        else:
            questions.append("Which product has the highest sales?")

    # 3. Location / Regional Analysis
    if location_col and revenue_col:
        loc_lower = location_col.lower()
        if "state" in loc_lower:
            questions.append("Which state has the highest revenue?")
        elif "region" in loc_lower:
            questions.append("Which region sells the most?")
        else:
            questions.append(f"Which {loc_lower} has the highest revenue?")
    elif location_col and profit_col:
        questions.append(f"Which {location_col.lower()} generates the highest profit?")

    # 4. Risk Analysis
    if risk_cols and crop_col:
        # Check if both disease and pest are present
        has_disease = any("disease" in c.lower() for c in risk_cols)
        has_pest = any("pest" in c.lower() for c in risk_cols)
        if has_disease and has_pest:
            questions.append("Which crop has the highest disease and pest risk?")
        elif has_disease:
            questions.append("Which crop has the highest disease risk?")
        elif has_pest:
            questions.append("Which crop has the highest pest risk?")
        else:
            questions.append("Which crop has the highest risk?")
    elif risk_cols and product_col:
        questions.append("Which product has the highest risk score?")

    # 5. Resource / Water / Efficiency / Customer / Order
    if irrigation_col and water_col:
        questions.append("Which irrigation method uses the least water?")
    elif water_col and production_col:
        questions.append("What is the water usage per production unit?")
    elif water_col and crop_col:
        questions.append("Which crop uses the most water?")
    elif customer_col and revenue_col:
        questions.append("Which customer generated the most revenue?")
    elif order_col and revenue_col:
        questions.append("Average order value")
    elif cost_col and profit_col:
        questions.append("What is the overall profit margin?")

    # 6. Temporal / Seasonal Analysis
    if season_col:
        s_lower = season_col.lower()
        if "season" in s_lower:
            if profit_col:
                questions.append("Which season is the most profitable?")
            elif revenue_col:
                questions.append("Which season generated the highest revenue?")
            else:
                questions.append("Which season has the highest production?")
        elif "date" in s_lower or "month" in s_lower or "year" in s_lower:
            if revenue_col:
                questions.append("What is the monthly revenue trend?")
            else:
                questions.append("What is the trend over time?")

    # 7. Additional Domain / Efficiency Questions if room remains
    if len(questions) < max_questions:
        if yield_col and crop_col and "Which crop has the highest yield?" not in questions:
            questions.append("Which crop has the highest yield?")
        elif cost_col and crop_col:
            questions.append("Which crop has the highest production cost?")
        elif order_col:
            questions.append("How many total orders were placed?")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped_questions: List[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            deduped_questions.append(q)

    # 8. Graceful Fallbacks if fewer than 4 questions generated
    if len(deduped_questions) < 4:
        fallbacks = _get_fallback_questions(max_questions)
        for fb in fallbacks:
            if fb not in seen:
                seen.add(fb)
                deduped_questions.append(fb)
            if len(deduped_questions) >= 4:
                break

    return deduped_questions[:max_questions]


def _get_fallback_questions(max_questions: int = 6) -> List[str]:
    """Default fallback questions for unrecognized or empty schemas."""
    defaults = [
        "What is the total revenue?",
        "Which region sells the most?",
        "Top 5 products",
        "Average order value",
        "How many total orders were placed?",
        "What is the overall summary of this dataset?",
    ]
    return defaults[:max_questions]
