"""
Automatic Data Visualization Engine for Analytics Query Results.

Inspects query result DataFrames and selects appropriate Plotly visualizations
based on column data types, cardinalities, and structural rules.

RULES:
- Rule A/C: 1 categorical + 1 numeric -> Bar chart.
- Rule B: 1 date/datetime + 1 numeric -> Line chart (chronologically sorted).
- Rule D: 1 categorical + multiple numeric -> Grouped bar chart.
- Rule E: Single column, numeric-only, all-categorical, or ambiguous -> None.
- Immutability: Never mutates the input DataFrame.
- Zero external LLM calls for chart generation.
"""
import re
from typing import Optional, List, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _clean_label(col_name: str) -> str:
    """Converts a snake_case column name into a clean, human-readable Title Case label."""
    words = col_name.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words)


def generate_chart_title(x_col: str, y_col: str, chart_type: str = "bar") -> str:
    """
    Generates a clear, professional chart title from column names and chart type.
    """
    x_label = _clean_label(x_col)
    y_label = _clean_label(y_col)

    if chart_type == "line":
        return f"{y_label} Over Time ({x_label})"
    elif chart_type == "grouped_bar":
        return f"{y_label} by {x_label}"
    else:
        return f"{y_label} by {x_label}"


def is_date_column(series: pd.Series, col_name: str = "") -> bool:
    """
    Detects whether a Series represents date or timestamp information.
    """
    # 1. Already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # 2. String/object columns with temporal names
    col_lower = col_name.lower()
    date_keywords = ["date", "time", "day", "month", "year", "quarter", "week"]
    has_date_name = any(kw in col_lower for kw in date_keywords)

    if series.dtype == "object" or isinstance(series.dtype, pd.StringDtype):
        non_nulls = series.dropna()
        if len(non_nulls) == 0:
            return False

        if has_date_name:
            # Verify if samples can be parsed as dates
            sample = non_nulls.head(20)
            try:
                converted = pd.to_datetime(sample, errors="coerce")
                if converted.notna().mean() >= 0.7:
                    return True
            except Exception:
                pass

        # Check standard date formats like YYYY-MM-DD
        sample_val = str(non_nulls.iloc[0]).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", sample_val) or re.match(r"^\d{2}/\d{2}/\d{4}", sample_val):
            try:
                converted = pd.to_datetime(non_nulls.head(20), errors="coerce")
                if converted.notna().mean() >= 0.8:
                    return True
            except Exception:
                pass

    return False


def is_identifier_column(col_name: str) -> bool:
    """
    Identifies if a column name indicates an identifier rather than a quantitative measure.
    """
    col_lower = col_name.lower().strip()
    id_suffixes = ["_id", " id", "_key", "_code", "id"]
    return any(col_lower.endswith(s) or col_lower == s for s in id_suffixes)


def is_numeric_measure_column(series: pd.Series, col_name: str) -> bool:
    """
    Determines if a Series is a genuine quantitative measure (not an ID or categorical int).
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False

    # Boolean is technically a subtype of numeric in pandas, exclude it
    if pd.api.types.is_bool_dtype(series):
        return False

    # Filter out columns that are clearly identifiers (e.g. order_id, customer_id)
    if is_identifier_column(col_name):
        return False

    return True


def is_categorical_dimension_column(series: pd.Series, col_name: str) -> bool:
    """
    Determines if a Series is a suitable categorical dimension for grouping/axes.
    """
    # Exclude date columns
    if is_date_column(series, col_name):
        return False

    # Text / Category / Boolean types
    if (
        series.dtype == "object"
        or isinstance(series.dtype, (pd.CategoricalDtype, pd.StringDtype))
        or pd.api.types.is_bool_dtype(series)
    ):
        return True

    # Numeric columns that act as codes/identifiers can sometimes be categories
    if pd.api.types.is_integer_dtype(series) and is_identifier_column(col_name):
        return True

    return False


def create_chart(df: Optional[pd.DataFrame], custom_title: Optional[str] = None) -> Optional[go.Figure]:
    """
    Analyzes a query result DataFrame and generates an appropriate Plotly Figure.

    Returns:
        go.Figure if an appropriate visualization can be determined.
        None if the structure is not suitable, single-cell, or ambiguous.
    """
    # Guard 1: Null or empty DataFrame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    # Guard 2: Single-column DataFrame (e.g. SELECT COUNT(*) FROM sales)
    if len(df.columns) < 2:
        return None

    # Work on a shallow copy to strictly prevent input mutation
    df_plot = df.copy()

    # Classify columns
    date_cols: List[str] = []
    cat_cols: List[str] = []
    num_cols: List[str] = []

    for col in df_plot.columns:
        series = df_plot[col]
        if is_date_column(series, col):
            date_cols.append(col)
        elif is_numeric_measure_column(series, col):
            num_cols.append(col)
        elif is_categorical_dimension_column(series, col):
            cat_cols.append(col)
        else:
            # Fallback check
            if pd.api.types.is_numeric_dtype(series):
                num_cols.append(col)
            else:
                cat_cols.append(col)

    # -----------------------------------------------------------------
    # CASE B: Time Series Line Chart (1 date column + 1 numeric column)
    # -----------------------------------------------------------------
    if len(date_cols) == 1 and len(num_cols) == 1 and len(cat_cols) == 0:
        d_col = date_cols[0]
        n_col = num_cols[0]

        # Convert date column safely and sort chronologically
        try:
            df_plot[d_col] = pd.to_datetime(df_plot[d_col])
            df_plot = df_plot.sort_values(by=d_col)
        except Exception:
            pass

        title = custom_title or generate_chart_title(d_col, n_col, chart_type="line")
        fig = px.line(
            df_plot,
            x=d_col,
            y=n_col,
            title=title,
            markers=True,
            labels={d_col: _clean_label(d_col), n_col: _clean_label(n_col)},
        )
        fig.update_layout(
            title_font_size=16,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # -----------------------------------------------------------------
    # CASE A & C: Standard Bar Chart (1 categorical + 1 numeric column)
    # -----------------------------------------------------------------
    if len(cat_cols) == 1 and len(num_cols) == 1 and len(date_cols) == 0:
        c_col = cat_cols[0]
        n_col = num_cols[0]

        title = custom_title or generate_chart_title(c_col, n_col, chart_type="bar")
        fig = px.bar(
            df_plot,
            x=c_col,
            y=n_col,
            title=title,
            labels={c_col: _clean_label(c_col), n_col: _clean_label(n_col)},
            color=n_col,
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            title_font_size=16,
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_tickangle=-30 if len(df_plot) > 5 else 0,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # -----------------------------------------------------------------
    # CASE D: Grouped Bar Chart (1 categorical + 2 to 4 numeric metrics)
    # -----------------------------------------------------------------
    if len(cat_cols) == 1 and 2 <= len(num_cols) <= 4 and len(date_cols) == 0:
        c_col = cat_cols[0]
        measures = num_cols

        title = custom_title or f"Metrics by {_clean_label(c_col)}"
        fig = px.bar(
            df_plot,
            x=c_col,
            y=measures,
            barmode="group",
            title=title,
            labels={c_col: _clean_label(c_col), "value": "Value", "variable": "Metric"},
        )
        fig.update_layout(
            title_font_size=16,
            margin=dict(l=40, r=40, t=50, b=40),
            legend_title_text="Metric",
            xaxis_tickangle=-30 if len(df_plot) > 5 else 0,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # -----------------------------------------------------------------
    # CASE D2: Grouped Bar Chart (2 categorical columns + 1 numeric metric)
    # e.g. region + category -> revenue
    # -----------------------------------------------------------------
    if len(cat_cols) == 2 and len(num_cols) == 1 and len(date_cols) == 0:
        primary_cat = cat_cols[0]
        group_cat = cat_cols[1]
        n_col = num_cols[0]

        # Only group if secondary category cardinality is manageable (<= 8)
        if df_plot[group_cat].nunique() <= 8:
            title = custom_title or f"{_clean_label(n_col)} by {_clean_label(primary_cat)} and {_clean_label(group_cat)}"
            fig = px.bar(
                df_plot,
                x=primary_cat,
                y=n_col,
                color=group_cat,
                barmode="group",
                title=title,
                labels={
                    primary_cat: _clean_label(primary_cat),
                    n_col: _clean_label(n_col),
                    group_cat: _clean_label(group_cat),
                },
            )
            fig.update_layout(
                title_font_size=16,
                margin=dict(l=40, r=40, t=50, b=40),
                legend_title_text=_clean_label(group_cat),
                xaxis_tickangle=-30 if len(df_plot) > 5 else 0,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            return fig

    # -----------------------------------------------------------------
    # CASE E: Numeric-only, All-categorical, or Ambiguous -> No chart
    # -----------------------------------------------------------------
    return None
