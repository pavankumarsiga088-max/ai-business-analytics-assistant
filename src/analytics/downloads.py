"""
Download Utilities Module (Phase 11C).

Provides pure-Python, in-memory serialization utilities for exporting analytical
query results and Plotly visualizations:
- CSV export: UTF-8 encoded bytes, header preserved, index excluded.
- Excel export: openpyxl in-memory workbook bytes via BytesIO, zero disk footprint.
- Plotly HTML export: Standalone, browser-ready HTML bytes with CDN-linked Plotly.js.
- Filename sanitization: Removes path separators and illegal filesystem characters.

DESIGN & ARCHITECTURAL PRINCIPLES:
- Zero Streamlit dependencies in this backend module.
- Input DataFrames and Plotly figures are strictly read-only and immutable.
- In-memory buffers only: zero temporary files left on disk.
- Robust error handling for empty sets, None, and Unicode strings.
"""

import io
import re
from typing import Optional
import pandas as pd
import plotly.graph_objects as go


def sanitize_filename(name: str, default: str = "analytics_result") -> str:
    """
    Sanitizes a candidate filename by removing filesystem-illegal characters,
    path traversal sequences, and whitespace.

    Forbidden characters stripped/replaced:
    - Path separators: / and \\
    - OS reserved symbols: : * ? " < > |
    - Control characters and excessive whitespace

    Parameters:
        name: The candidate raw string (e.g. dataset name or question fragment).
        default: Fallback filename if sanitized result is empty.

    Returns:
        Clean, safe filename string without extension.
    """
    if not name or not isinstance(name, str):
        return default

    # Strip existing extensions if present (e.g. 'sales.csv' -> 'sales')
    cleaned = re.sub(r"\.[a-zA-Z0-9]+$", "", name.strip())

    # Replace forbidden path/filesystem characters with underscores
    cleaned = re.sub(r'[\/\\:\*\?"<>\|\s]+', "_", cleaned)

    # Restrict to safe alphanumeric, underscore, hyphen
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", cleaned)

    # Collapse consecutive underscores and strip edge delimiters
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")

    if not cleaned:
        return default

    # Limit filename length to 64 characters for safety across all OS limits
    return cleaned[:64]


def build_download_filename(
    base_name: str,
    extension: str,
    prefix: Optional[str] = None,
) -> str:
    """
    Builds a sanitized, predictable download filename with the requested extension.

    Examples:
        build_download_filename("result", "csv") -> "result.csv"
        build_download_filename("analysis", "xlsx", prefix="sales") -> "sales_analysis.xlsx"
    """
    clean_base = sanitize_filename(base_name, default="analytics_result")
    clean_ext = extension.lstrip(".").lower()

    if prefix and prefix.strip():
        clean_prefix = sanitize_filename(prefix.strip(), default="")
        if clean_prefix:
            return f"{clean_prefix}_{clean_base}.{clean_ext}"

    return f"{clean_base}.{clean_ext}"


def dataframe_to_csv_bytes(df: Optional[pd.DataFrame]) -> bytes:
    """
    Serializes a DataFrame into UTF-8 encoded CSV bytes.

    Guarantees:
    - Never mutates the input DataFrame.
    - Excludes pandas row index (index=False).
    - Preserves exact column names and row values.
    - Handles Unicode/special characters cleanly.

    Returns:
        bytes: UTF-8 encoded CSV file bytes.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return b""

    if df.empty:
        if len(df.columns) > 0:
            # Empty DataFrame with declared columns produces header row
            return df.to_csv(index=False).encode("utf-8")
        return b""

    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(
    df: Optional[pd.DataFrame],
    sheet_name: str = "Query Results",
) -> bytes:
    """
    Serializes a DataFrame into Microsoft Excel (.xlsx) file bytes using openpyxl.

    Guarantees:
    - Pure in-memory generation via io.BytesIO (no disk I/O).
    - Never mutates the input DataFrame.
    - Excludes pandas row index (index=False).
    - Preserves exact column names and row values.
    - Excel sheet name is safely clamped to 31 characters without illegal symbols.

    Returns:
        bytes: Valid .xlsx binary workbook bytes.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return b""

    # Sanitize sheet name (Excel limits sheet names to 31 chars and bans : \ / ? * [ ])
    safe_sheet = re.sub(r"[:\\/\?\*\[\]]", "_", sheet_name.strip())[:31] or "Query Results"

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe_sheet)

    return buffer.getvalue()


def figure_to_html_bytes(fig: Optional[go.Figure]) -> bytes:
    """
    Exports a Plotly Figure to standalone, browser-ready HTML bytes.

    Guarantees:
    - Never mutates the input Plotly Figure.
    - Standalone HTML document ready to open directly in any web browser.
    - Uses CDN-linked Plotly JavaScript to keep exported file lightweight (~5-10 KB).

    Returns:
        bytes: UTF-8 encoded HTML document bytes.
    """
    if fig is None or not isinstance(fig, go.Figure):
        return b""

    html_str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
    )
    return html_str.encode("utf-8")
