"""
Result Presentation Module (Phase 11B).

Analyzes the shape and data types of query result DataFrames to determine
the optimal presentation strategy:
- "kpi": 1 row x 1 numeric metric -> prominent KPI card
- "multi_kpi": 1 row x multiple numeric metrics -> multiple KPI cards
- "table_and_chart": Grouped/chartable multi-row results -> Plotly chart + table
- "table": Non-chartable or record detail results -> tabular display only
- "empty": Zero-row or null result set

DESIGN & ARCHITECTURAL PRINCIPLES:
- Zero Streamlit UI code in this backend module (returns structured metadata).
- Input DataFrames are strictly read-only and immutable.
- Dynamic detection of numeric measures vs. identifiers (order_id, zip_code, year, etc.).
- Deterministic, locale-safe numeric formatting with zero currency assumptions.
- Coexists alongside existing visualization engine (create_chart) without replacing it.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.analytics.visualization import create_chart, is_date_column


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass(frozen=True)
class KPICard:
    """
    Represents a single KPI card's presentation metadata.

    Attributes:
        column: Original column name in the result DataFrame.
        label: Human-readable Title Case display label.
        raw_value: Unmodified raw value from the DataFrame.
        formatted_value: Formatted, human-readable display string.
    """
    column: str
    label: str
    raw_value: Any
    formatted_value: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "label": self.label,
            "raw_value": self.raw_value,
            "formatted_value": self.formatted_value,
        }


@dataclass(frozen=True)
class ResultPresentation:
    """
    Structured metadata describing how a query result should be presented.

    Attributes:
        classification: One of "kpi", "multi_kpi", "table_and_chart", "table", "empty".
        kpi_cards: List of KPICard objects (up to max_kpis).
        total_kpis: Total number of KPI candidate columns detected.
        displayed_kpis: Number of KPI cards displayed.
        show_chart: True if classification is "table_and_chart".
        show_table: True if the result is non-empty.
        row_count: Total rows in the result DataFrame.
        column_count: Total columns in the result DataFrame.
    """
    classification: str
    kpi_cards: List[KPICard]
    total_kpis: int
    displayed_kpis: int
    show_chart: bool
    show_table: bool
    row_count: int
    column_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "kpi_cards": [c.to_dict() for c in self.kpi_cards],
            "total_kpis": self.total_kpis,
            "displayed_kpis": self.displayed_kpis,
            "show_chart": self.show_chart,
            "show_table": self.show_table,
            "row_count": self.row_count,
            "column_count": self.column_count,
        }


# =====================================================================
# IDENTIFIER & KPI DETECTION
# =====================================================================

def is_identifier_or_temporal_column(col_name: str) -> bool:
    """
    Determines if a column name represents an identifier, code, key,
    or temporal unit (such as a calendar year) rather than a quantitative measure.
    """
    col_lower = col_name.lower().strip()

    # Standard identifier suffixes
    id_suffixes = [
        "_id", " id", "_key", "_code", "_zip", "_postal",
        "_num", "_no", "_number", "id", "key", "code",
    ]
    if any(col_lower.endswith(s) or col_lower == s for s in id_suffixes):
        return True

    # Common postal / zip codes
    if col_lower in ("zip", "zip_code", "zipcode", "postal_code", "postalcode", "pin_code", "pincode"):
        return True

    # Calendar & temporal partition names (e.g. year, month, quarter)
    temporal_names = [
        "year", "quarter", "month", "day", "week",
        "fiscal_year", "calendar_year", "yr",
    ]
    if col_lower in temporal_names:
        return True
    if any(col_lower.endswith(f"_{t}") for t in temporal_names):
        return True

    # Geographic coordinates and contact numbers
    geo_phone = ["latitude", "longitude", "lat", "lon", "phone", "phone_number", "mobile"]
    if col_lower in geo_phone:
        return True

    return False


def is_kpi_candidate_column(series: pd.Series, col_name: str) -> bool:
    """
    Determines if a Series is a genuine quantitative KPI measure candidate.

    Criteria:
    - Must be a numeric dtype (int, float).
    - Must not be a boolean (pandas bools are technically numeric subtypes).
    - Must not be an identifier, code, or postal column.
    - Must not be a temporal calendar column (year, date, etc.).
    """
    if series is None:
        return False

    if not pd.api.types.is_numeric_dtype(series):
        return False

    if pd.api.types.is_bool_dtype(series):
        return False

    if is_identifier_or_temporal_column(col_name):
        return False

    if is_date_column(series, col_name):
        return False

    return True


def identify_kpi_columns(df: Optional[pd.DataFrame]) -> List[str]:
    """
    Scans a DataFrame and dynamically identifies all columns suitable as KPI measures.
    Never mutates the input DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    kpi_cols: List[str] = []
    for col in df.columns:
        if is_kpi_candidate_column(df[col], str(col)):
            kpi_cols.append(str(col))

    return kpi_cols


# =====================================================================
# FORMATTING HELPERS
# =====================================================================

def format_kpi_label(col_name: str) -> str:
    """
    Converts snake_case or raw SQL alias strings into clean, human-readable Title Case labels.

    Examples:
        total_revenue -> "Total Revenue"
        avg_order_value -> "Avg Order Value"
        total_orders -> "Total Orders"
        sum(revenue) -> "Sum Revenue"
    """
    if not col_name:
        return ""

    # Replace separators and SQL function parentheses with spaces
    cleaned = re.sub(r"[_\(\)\.\-]+", " ", str(col_name)).strip()
    words = cleaned.split()

    formatted_words: List[str] = []
    for w in words:
        w_lower = w.lower()
        if w_lower in ("id", "kpi", "sql", "ai"):
            formatted_words.append(w.upper())
        else:
            formatted_words.append(w.capitalize())

    return " ".join(formatted_words)


def format_kpi_value(val: Any, col_name: str = "") -> str:
    """
    Deterministically formats a numeric KPI value into a clean, human-readable string.

    Rules:
    - Null / NaN / None -> "N/A"
    - Integers (e.g. 109510) -> "109,510"
    - Floats (e.g. 6410.21) -> "6,410.21"
    - Small fractional floats (e.g. 0.2345) -> "0.23"
    - Negative values (e.g. -109510) -> "-109,510"
    - Large values (e.g. 1000000000) -> "1,000,000,000"
    - Percentages (if column name indicates pct/percent) -> formatted with '%'
    - Zero currency assumptions.
    """
    if val is None or pd.isna(val):
        return "N/A"

    try:
        float_val = float(val)
    except (ValueError, TypeError):
        return str(val)

    # Check for percentage column indication
    col_lower = col_name.lower().strip() if col_name else ""
    is_pct = any(
        col_lower.endswith(s) or col_lower == s
        for s in ("_pct", "_percent", "_percentage", "pct", "percent", "percentage")
    )

    if is_pct:
        if abs(float_val) <= 1.0 and float_val != 0.0:
            return f"{float_val * 100:,.2f}%"
        return f"{float_val:,.2f}%"

    # Format integer values (native int or exact integer floats)
    if isinstance(val, (int, np.integer)) or float_val.is_integer():
        return f"{int(float_val):,}"

    # Format standard floating point values
    return f"{float_val:,.2f}"


# =====================================================================
# RESULT CLASSIFICATION & METADATA BUILDER
# =====================================================================

def classify_result_shape(df: Optional[pd.DataFrame]) -> str:
    """
    Classifies the structural shape and presentation intent of a query result DataFrame.

    Categories:
    - "empty": DataFrame is None, empty, or has 0 rows.
    - "kpi": Exactly 1 row and 1 numeric KPI column.
    - "multi_kpi": Exactly 1 row and multiple numeric KPI columns (all columns are KPI candidates).
    - "table_and_chart": Result is suitable for an automatic Plotly visualization.
    - "table": Result is tabular only (not empty and not suitable for KPI or chart).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) == 0:
        return "empty"

    kpi_cols = identify_kpi_columns(df)

    if len(df) == 1:
        if len(df.columns) == 1 and len(kpi_cols) == 1:
            return "kpi"
        if len(df.columns) > 1 and len(kpi_cols) == len(df.columns):
            return "multi_kpi"

    # Check if visualization engine can create a chart for this DataFrame
    chart = create_chart(df)
    if chart is not None:
        return "table_and_chart"

    return "table"


def build_kpi_metadata(
    df: Optional[pd.DataFrame],
    max_kpis: int = 4,
) -> ResultPresentation:
    """
    Analyzes a query result DataFrame and generates structured presentation metadata.
    Guarantees strict DataFrame immutability.

    Parameters:
        df: The query result DataFrame to inspect.
        max_kpis: Maximum number of KPI cards to display for multi-KPI results (default: 4).

    Returns:
        ResultPresentation with classification, cards, and presentation flags.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) == 0:
        return ResultPresentation(
            classification="empty",
            kpi_cards=[],
            total_kpis=0,
            displayed_kpis=0,
            show_chart=False,
            show_table=False,
            row_count=0,
            column_count=0,
        )

    classification = classify_result_shape(df)
    kpi_cols = identify_kpi_columns(df)
    row_count = len(df)
    col_count = len(df.columns)

    cards: List[KPICard] = []
    if classification in ("kpi", "multi_kpi"):
        first_row = df.iloc[0]
        # Cap displayed cards at max_kpis
        displayed_columns = kpi_cols[:max_kpis]
        for col in displayed_columns:
            raw_val = first_row[col]
            cards.append(
                KPICard(
                    column=col,
                    label=format_kpi_label(col),
                    raw_value=raw_val,
                    formatted_value=format_kpi_value(raw_val, col_name=col),
                )
            )

    return ResultPresentation(
        classification=classification,
        kpi_cards=cards,
        total_kpis=len(kpi_cols),
        displayed_kpis=len(cards),
        show_chart=(classification == "table_and_chart"),
        show_table=True,
        row_count=row_count,
        column_count=col_count,
    )
