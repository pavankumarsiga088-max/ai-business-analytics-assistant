"""
Data Quality Issue Details Module.

Transforms raw check results into structured, actionable QualityIssueDetail objects
for exploration and analysis without mutating the underlying dataset, calling LLMs,
or embedding Streamlit UI code.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd

from src.data_quality.scoring import (
    CHECK_CATALOG,
    identify_check_key,
    extract_issue_count,
)


# =====================================================================
# USER-FRIENDLY EXPLANATIONS & RECOMMENDATIONS
# =====================================================================

ISSUE_EXPLANATIONS: Dict[str, str] = {
    "missing_values": (
        "Some records contain empty or unpopulated values in one or more columns, "
        "which may cause errors or skewed results in downstream aggregations."
    ),
    "duplicate_rows": (
        "Identical records appear multiple times across all columns, "
        "which can lead to artificial double-counting in business reports."
    ),
    "duplicate_order_ids": (
        "Identifier values appear multiple times in the primary key column, "
        "which violates entity uniqueness and may distort transaction tracking."
    ),
    "whitespace": (
        "Leading or trailing whitespace exists in text fields, "
        "which can cause filter mismatches and join failures in SQL queries."
    ),
    "inconsistent_casing": (
        "Categorical values use inconsistent capitalization (e.g. 'North' vs 'north'), "
        "causing fragmented grouping in summary analyses."
    ),
    "invalid_dates": (
        "Some values in the date column cannot be interpreted as valid calendar dates, "
        "preventing accurate time-series trend analysis."
    ),
    "numeric_columns": (
        "Values are stored as text or strings in columns expected to be numeric, "
        "preventing mathematical aggregations and calculations."
    ),
    "invalid_quantities": (
        "Zero or negative quantities detected. These may represent returns, refunds, "
        "or data-entry anomalies that require business verification."
    ),
    "outliers": (
        "Some numeric values are unusually far from the typical range (IQR method). "
        "These may be legitimate high-value transactions and should be reviewed rather than deleted."
    ),
    "revenue_consistency": (
        "Calculated revenue does not match quantity multiplied by unit_price for certain records, "
        "indicating potential billing discrepancies or discount omissions."
    ),
}

DEFAULT_PASS_EXPLANATION = "No data quality issues detected for this check."


# =====================================================================
# DATA QUALITY ISSUE DETAIL DATACLASS
# =====================================================================

@dataclass
class QualityIssueDetail:
    """
    Structured representation of an individual data quality check result.

    Attributes:
        check_name: Human-readable check title.
        catalog_key: Normalized identifier matching CHECK_CATALOG.
        status: Outcome status ('PASS', 'FAIL', 'WARNING').
        severity: Authoritative severity ('HIGH', 'MEDIUM', 'LOW', 'PASS').
        category: Check category ('ERROR', 'WARNING', 'PASS').
        message: Plain-English explanation of what this issue means.
        issue_count: Number of flagged items or anomalies.
        affected_columns: List of column names impacted by this check.
        affected_row_count: Number of dataset rows affected.
        affected_row_indices: Row index positions of affected records.
        sample_values: Small list of sample problematic values (capped at 5).
        sample_records: Small DataFrame of sample problematic rows (capped at 5 rows).
        recommendation: Practical, non-destructive guidance for data analysts.
    """
    check_name: str
    catalog_key: str
    status: str
    severity: str
    category: str
    message: str
    issue_count: int
    affected_columns: List[str]
    affected_row_count: int
    affected_row_indices: List[int]
    sample_values: List[str]
    sample_records: pd.DataFrame
    recommendation: str

    @property
    def has_issues(self) -> bool:
        return self.status != "PASS" and self.issue_count > 0


# =====================================================================
# TRANSFORMATION LOGIC
# =====================================================================

def _extract_affected_columns(
    catalog_key: str,
    check_result: Dict[str, Any],
    df: Optional[pd.DataFrame] = None
) -> List[str]:
    """Dynamically extracts affected column names for arbitrary datasets."""
    if catalog_key == "missing_values":
        cols = list(check_result.get("columns_with_missing", {}).keys())
        return cols

    elif catalog_key == "duplicate_rows":
        if df is not None and not df.empty:
            return list(df.columns)
        return []

    elif catalog_key == "duplicate_order_ids":
        col = check_result.get("column_checked")
        return [col] if col else []

    elif catalog_key == "whitespace":
        aff = check_result.get("columns_affected", {})
        if isinstance(aff, dict):
            return list(aff.keys())
        elif isinstance(aff, list):
            return list(aff)
        return []

    elif catalog_key == "inconsistent_casing":
        aff = check_result.get("columns_affected", [])
        if isinstance(aff, list) and aff:
            return list(aff)
        conf = check_result.get("casing_conflicts", {})
        return list(conf.keys()) if isinstance(conf, dict) else []

    elif catalog_key in ("invalid_dates", "numeric_columns", "outliers"):
        details = check_result.get("details", {})
        if isinstance(details, dict) and details:
            return list(details.keys())
        cols_checked = check_result.get("columns_checked", [])
        return [c for c in cols_checked if c]

    elif catalog_key == "invalid_quantities":
        col = check_result.get("column_checked")
        return [col] if col else []

    elif catalog_key == "revenue_consistency":
        cols = []
        if "inconsistent_rows" in check_result and check_result["inconsistent_rows"]:
            first_row = check_result["inconsistent_rows"][0]
            candidate_keys = ["quantity", "unit_price", "revenue"]
            cols = [k for k in candidate_keys if k in first_row or (df is not None and k in df.columns)]
        if not cols and df is not None:
            cols = [c for c in ["quantity", "unit_price", "revenue"] if c in df.columns]
        if not cols:
            cols = ["quantity", "unit_price", "revenue"]
        return cols

    return []


def _extract_affected_indices(
    catalog_key: str,
    check_result: Dict[str, Any],
    affected_cols: List[str],
    df: Optional[pd.DataFrame] = None
) -> List[int]:
    """Extracts affected row indices for detailed inspection without modifying df."""
    if not check_result.get("has_issues", False):
        return []

    if catalog_key == "duplicate_rows":
        return list(check_result.get("duplicate_row_indices", []))

    elif catalog_key == "duplicate_order_ids":
        repeated_ids = check_result.get("repeated_ids", [])
        col = check_result.get("column_checked", "order_id")
        if df is not None and col in df.columns and repeated_ids:
            return df[df[col].isin(repeated_ids)].index.tolist()
        return []

    elif catalog_key == "missing_values":
        if df is not None and affected_cols:
            valid_cols = [c for c in affected_cols if c in df.columns]
            if valid_cols:
                return df[df[valid_cols].isna().any(axis=1)].index.tolist()
        return []

    elif catalog_key == "whitespace":
        if df is not None and affected_cols:
            idx_set = set()
            for c in affected_cols:
                if c in df.columns:
                    s = df[c].dropna().astype(str)
                    mask = s != s.str.strip()
                    idx_set.update(s[mask].index.tolist())
            return sorted(list(idx_set))
        return []

    elif catalog_key == "inconsistent_casing":
        conf = check_result.get("casing_conflicts", {})
        if df is not None and conf:
            idx_set = set()
            for col, variations_map in conf.items():
                if col in df.columns:
                    all_variants = set()
                    for var_list in variations_map.values():
                        all_variants.update(var_list)
                    s = df[col].dropna().astype(str).str.strip()
                    mask = s.isin(all_variants)
                    idx_set.update(s[mask].index.tolist())
            return sorted(list(idx_set))
        return []

    elif catalog_key in ("invalid_dates", "numeric_columns", "outliers"):
        details = check_result.get("details", {})
        idx_set = set()
        for col_data in details.values():
            if isinstance(col_data, dict):
                idx_set.update(col_data.get("affected_row_indices", []))
        return sorted(list(idx_set))

    elif catalog_key == "invalid_quantities":
        neg = check_result.get("negative_row_indices", [])
        zero = check_result.get("zero_row_indices", [])
        return sorted(list(set(neg + zero)))

    elif catalog_key == "revenue_consistency":
        inconsistent = check_result.get("inconsistent_rows", [])
        return sorted(list(set(r["row_index"] for r in inconsistent if isinstance(r, dict) and "row_index" in r)))

    return []


def _extract_sample_values(
    catalog_key: str,
    check_result: Dict[str, Any],
    limit: int = 5
) -> List[str]:
    """Extracts up to `limit` readable examples of problematic values."""
    if not check_result.get("has_issues", False):
        return []

    samples: List[str] = []

    if catalog_key == "missing_values":
        cols = check_result.get("columns_with_missing", {})
        for col, count in cols.items():
            samples.append(f"{col}: NULL ({count} missing)")
            if len(samples) >= limit:
                break
        return samples[:limit]

    elif catalog_key == "duplicate_rows":
        return ["Exact duplicate record across all columns"]

    elif catalog_key == "duplicate_order_ids":
        repeated = check_result.get("repeated_ids", [])
        return [str(item) for item in repeated[:limit]]

    elif catalog_key == "whitespace":
        offending = check_result.get("sample_offending_values", {})
        for col, vals in offending.items():
            for v in vals:
                samples.append(f"{col}: '{v}'")
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break
        return samples[:limit]

    elif catalog_key == "inconsistent_casing":
        conf = check_result.get("casing_conflicts", {})
        for col, groups in conf.items():
            for lowered_key, variants in groups.items():
                samples.append(f"{col}: {' vs '.join(repr(v) for v in variants)}")
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break
        return samples[:limit]

    elif catalog_key == "invalid_dates":
        details = check_result.get("details", {})
        for col, d in details.items():
            for v in d.get("sample_invalid_values", []):
                samples.append(f"{col}: '{v}'")
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break
        return samples[:limit]

    elif catalog_key == "numeric_columns":
        details = check_result.get("details", {})
        for col, d in details.items():
            for v in d.get("sample_problematic_values", []):
                samples.append(f"{col}: '{v}'")
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break
        return samples[:limit]

    elif catalog_key == "invalid_quantities":
        for v in check_result.get("negative_sample_values", []):
            samples.append(f"negative: {v}")
            if len(samples) >= limit:
                break
        for v in check_result.get("zero_sample_values", []):
            if len(samples) >= limit:
                break
            samples.append(f"zero: {v}")
        return samples[:limit]

    elif catalog_key == "outliers":
        details = check_result.get("details", {})
        for col, d in details.items():
            for v in d.get("sample_outlier_values", []):
                samples.append(f"{col}: {v}")
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break
        return samples[:limit]

    elif catalog_key == "revenue_consistency":
        for r in check_result.get("inconsistent_rows", []):
            if isinstance(r, dict):
                row_idx = r.get("row_index", "?")
                q = r.get("quantity", 0)
                p = r.get("unit_price", 0)
                act = r.get("actual_revenue", 0)
                exp = r.get("expected_revenue", 0)
                samples.append(f"Row {row_idx}: {q} × {p} = {exp} (recorded as {act})")
                if len(samples) >= limit:
                    break
        return samples[:limit]

    return samples[:limit]


# =====================================================================
# PUBLIC API FUNCTIONS
# =====================================================================

def build_issue_detail(
    check_result: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    sample_limit: int = 5,
) -> QualityIssueDetail:
    """
    Transforms a single quality check dictionary into a QualityIssueDetail object.

    Parameters:
        check_result: Raw result dictionary from one of the 10 checks.
        df: Optional original DataFrame (never mutated; used only to slice sample rows).
        sample_limit: Maximum number of sample rows to include (default 5).

    Returns:
        QualityIssueDetail dataclass instance.
    """
    raw_name = check_result.get("check_name", "Unknown Check")
    catalog_key = identify_check_key(raw_name)
    catalog_entry = CHECK_CATALOG.get(catalog_key, {})

    has_issues = bool(check_result.get("has_issues", False))
    issue_count = extract_issue_count(check_result, catalog_entry) if has_issues else 0

    if not has_issues or issue_count == 0:
        status = "PASS"
        severity = "PASS"
        category = "PASS"
        message = DEFAULT_PASS_EXPLANATION
        recommendation = "No action required. This check passed completely."
        affected_cols: List[str] = []
        affected_indices: List[int] = []
        sample_vals: List[str] = []
        sample_df = pd.DataFrame()
        row_count = 0
    else:
        severity = catalog_entry.get("severity", "MEDIUM")
        category = catalog_entry.get("category", "ERROR" if severity != "LOW" else "WARNING")
        status = "WARNING" if category == "WARNING" else "FAIL"
        message = ISSUE_EXPLANATIONS.get(
            catalog_key,
            "Anomalies were detected during this data quality inspection."
        )
        recommendation = catalog_entry.get("recommendation", "Review flagged records before analysis.")

        affected_cols = _extract_affected_columns(catalog_key, check_result, df)
        affected_indices = _extract_affected_indices(catalog_key, check_result, affected_cols, df)
        sample_vals = _extract_sample_values(catalog_key, check_result, limit=sample_limit)

        row_count = len(affected_indices) if affected_indices else issue_count

        # Bound sample rows strictly to at most `sample_limit` rows
        if df is not None and not df.empty and affected_indices:
            valid_indices = [idx for idx in affected_indices if idx in df.index]
            if valid_indices:
                sample_df = df.loc[valid_indices[:sample_limit]].copy()
            else:
                sample_df = pd.DataFrame()
        else:
            sample_df = pd.DataFrame()

    return QualityIssueDetail(
        check_name=catalog_entry.get("title", raw_name),
        catalog_key=catalog_key,
        status=status,
        severity=severity,
        category=category,
        message=message,
        issue_count=issue_count,
        affected_columns=affected_cols,
        affected_row_count=row_count,
        affected_row_indices=affected_indices,
        sample_values=sample_vals,
        sample_records=sample_df,
        recommendation=recommendation,
    )


def extract_issue_details(
    check_results: List[Dict[str, Any]],
    df: Optional[pd.DataFrame] = None,
    sample_limit: int = 5,
) -> List[QualityIssueDetail]:
    """
    Transforms a list of check result dictionaries into a list of QualityIssueDetail objects.

    Parameters:
        check_results: List of dictionaries from run_all_quality_checks.
        df: Optional DataFrame for extracting small row samples (never mutated).
        sample_limit: Maximum sample records per issue (default 5).

    Returns:
        List of QualityIssueDetail instances in original check order.
    """
    return [
        build_issue_detail(res, df=df, sample_limit=sample_limit)
        for res in check_results
    ]


def get_issue_severity_counts(issue_details: List[QualityIssueDetail]) -> Dict[str, int]:
    """
    Aggregates issue counts across severity levels for compact UI summaries.

    Returns:
        Dict with keys 'HIGH', 'MEDIUM', 'LOW', 'PASS', and 'TOTAL_ISSUES'.
    """
    counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "PASS": 0,
        "TOTAL_ISSUES": 0,
    }

    for issue in issue_details:
        if issue.status == "PASS" or issue.severity == "PASS":
            counts["PASS"] += 1
        else:
            sev = issue.severity.upper()
            if sev in counts:
                counts[sev] += 1
            counts["TOTAL_ISSUES"] += 1

    return counts
