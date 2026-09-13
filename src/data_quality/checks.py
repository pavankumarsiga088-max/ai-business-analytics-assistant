"""
Data Quality Checks Module.

Provides pure, independent functions to inspect a Pandas DataFrame for data issues:
Phase 1 Checks:
1. Missing values
2. Exact duplicate rows
3. Duplicate primary key / order_id values
4. Extra leading/trailing whitespace
5. Inconsistent capitalization in text columns

Phase 2 Checks:
6. Invalid date formats and unparseable dates
7. Numeric type problems (numbers stored as text or strings)
8. Invalid quantity values (negative or zero)
9. Outlier detection using IQR (Interquartile Range)
10. Revenue business rule consistency (quantity * unit_price == revenue)

IMPORTANT: None of these functions modify the input DataFrame.
"""
from typing import Dict, Any, List, Optional
import pandas as pd


# =====================================================================
# CHECK 1: Missing Values
# =====================================================================
def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects missing (null / NaN / None) values in each column of the DataFrame.
    """
    total_rows = len(df)
    missing_per_col = df.isna().sum()
    
    columns_with_missing = {
        col: int(count)
        for col, count in missing_per_col.items()
        if count > 0
    }
    
    percentages = {
        col: round((count / total_rows) * 100, 2)
        for col, count in columns_with_missing.items()
    }
    
    total_missing = int(missing_per_col.sum())
    
    return {
        "check_name": "Missing Values",
        "has_issues": total_missing > 0,
        "total_missing_values": total_missing,
        "columns_with_missing": columns_with_missing,
        "missing_percentages": percentages,
        "total_rows_inspected": total_rows
    }


# =====================================================================
# CHECK 2: Exact Duplicate Rows
# =====================================================================
def check_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects exact duplicate rows where every single column value is identical.
    """
    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    duplicate_indices: List[int] = df[duplicate_mask].index.tolist()
    
    return {
        "check_name": "Exact Duplicate Rows",
        "has_issues": duplicate_count > 0,
        "duplicate_row_count": duplicate_count,
        "duplicate_row_indices": duplicate_indices,
        "total_rows_inspected": len(df)
    }


# =====================================================================
# CHECK 3: Duplicate Primary Key (e.g. order_id)
# =====================================================================
def check_duplicate_order_ids(df: pd.DataFrame, id_column: str = "order_id") -> Dict[str, Any]:
    """
    Detects duplicate identifier values in a designated primary key column.
    """
    if id_column not in df.columns:
        return {
            "check_name": f"Duplicate '{id_column}' Check",
            "has_issues": False,
            "error": f"Column '{id_column}' not found in dataset.",
            "duplicate_id_count": 0,
            "repeated_ids": []
        }
    
    id_dup_mask = df.duplicated(subset=[id_column], keep=False)
    duplicate_ids = df.loc[id_dup_mask, id_column].unique().tolist()
    extra_duplicates_count = int(df.duplicated(subset=[id_column]).sum())
    
    return {
        "check_name": f"Duplicate '{id_column}' Check",
        "has_issues": extra_duplicates_count > 0,
        "column_checked": id_column,
        "duplicate_id_count": extra_duplicates_count,
        "repeated_ids": duplicate_ids,
        "total_rows_inspected": len(df)
    }


# =====================================================================
# CHECK 4: Whitespace Detection
# =====================================================================
def check_whitespace(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects strings with leading or trailing whitespace in object/string columns.
    """
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    columns_with_whitespace = {}
    sample_values = {}
    total_whitespace_issues = 0
    
    for col in text_columns:
        series_clean = df[col].dropna().astype(str)
        has_extra_spaces = series_clean != series_clean.str.strip()
        count = int(has_extra_spaces.sum())
        
        if count > 0:
            total_whitespace_issues += count
            columns_with_whitespace[col] = count
            offending_samples = series_clean[has_extra_spaces].head(3).tolist()
            sample_values[col] = offending_samples
            
    return {
        "check_name": "Whitespace Detection",
        "has_issues": total_whitespace_issues > 0,
        "total_whitespace_issues": total_whitespace_issues,
        "columns_affected": columns_with_whitespace,
        "sample_offending_values": sample_values
    }


# =====================================================================
# CHECK 5: Inconsistent Capitalization
# =====================================================================
def check_inconsistent_casing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects inconsistent capitalization within text columns.
    """
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    casing_issues_by_column = {}
    total_inconsistent_groups = 0
    
    for col in text_columns:
        raw_unique = df[col].dropna().astype(str).str.strip().unique()
        
        lower_map: Dict[str, List[str]] = {}
        for val in raw_unique:
            if not val:
                continue
            lowered = val.lower()
            lower_map.setdefault(lowered, []).append(val)
            
        conflicts = {
            lowered: variations
            for lowered, variations in lower_map.items()
            if len(variations) > 1
        }
        
        if conflicts:
            total_inconsistent_groups += len(conflicts)
            casing_issues_by_column[col] = conflicts
            
    return {
        "check_name": "Inconsistent Capitalization",
        "has_issues": total_inconsistent_groups > 0,
        "total_inconsistent_groups": total_inconsistent_groups,
        "columns_affected": list(casing_issues_by_column.keys()),
        "casing_conflicts": casing_issues_by_column
    }


# =====================================================================
# CHECK 6: Invalid Dates
# =====================================================================
def check_invalid_dates(df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Identifies date columns and detects values that cannot be parsed as valid calendar dates.
    Uses pd.to_datetime with errors='coerce' on a temporary series.
    Does NOT modify the original DataFrame.
    """
    # If date columns not explicitly provided, detect columns with 'date' or 'time' in their name
    if date_columns is None:
        date_columns = [
            col for col in df.columns
            if "date" in col.lower() or "time" in col.lower()
        ]
        
    date_issues = {}
    total_invalid_dates = 0
    
    for col in date_columns:
        if col not in df.columns:
            continue
            
        # Inspect only non-null values
        non_null_series = df[col].dropna().astype(str)
        if non_null_series.empty:
            continue
            
        # Attempt date parsing on a temporary series with errors='coerce'
        # Any string that fails date parsing becomes NaT (Not a Time)
        parsed_dates = pd.to_datetime(non_null_series, errors="coerce", format="mixed")
        
        # An invalid date is a non-null string that failed parsing (resulted in NaT)
        invalid_mask = parsed_dates.isna()
        invalid_count = int(invalid_mask.sum())
        
        if invalid_count > 0:
            total_invalid_dates += invalid_count
            affected_indices = non_null_series[invalid_mask].index.tolist()
            sample_values = non_null_series[invalid_mask].head(5).tolist()
            
            date_issues[col] = {
                "invalid_count": invalid_count,
                "affected_row_indices": affected_indices,
                "sample_invalid_values": sample_values
            }
            
    return {
        "check_name": "Invalid Dates",
        "has_issues": total_invalid_dates > 0,
        "columns_checked": date_columns,
        "total_invalid_dates": total_invalid_dates,
        "details": date_issues
    }


# =====================================================================
# CHECK 7: Numeric Type Problems
# =====================================================================
def check_numeric_columns(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detects values stored as text or strings in columns expected to be numeric.
    Flags entries like '4999' (string) or 'INR 8500' (text with currency).
    Does NOT modify the original DataFrame.
    """
    if numeric_columns is None:
        # Default expected business numeric columns
        default_candidates = ["quantity", "unit_price", "revenue", "price", "amount", "cost"]
        numeric_columns = [col for col in default_candidates if col in df.columns]
        
    problematic_details = {}
    total_problematic_values = 0
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        non_null_series = df[col].dropna()
        if non_null_series.empty:
            continue
            
        # Condition 1: Check if any value is stored as a Python string type
        is_string_type = non_null_series.apply(lambda v: isinstance(v, str))
        
        # Condition 2: Check if value cannot be converted to numeric at all
        coerced_numeric = pd.to_numeric(non_null_series, errors="coerce")
        is_unparseable = coerced_numeric.isna()
        
        # A cell has a type problem if it is a string or unparseable
        type_problem_mask = is_string_type | is_unparseable
        problem_count = int(type_problem_mask.sum())
        
        if problem_count > 0:
            total_problematic_values += problem_count
            affected_indices = non_null_series[type_problem_mask].index.tolist()
            sample_values = non_null_series[type_problem_mask].head(5).tolist()
            
            problematic_details[col] = {
                "problematic_count": problem_count,
                "affected_row_indices": affected_indices,
                "sample_problematic_values": sample_values
            }
            
    return {
        "check_name": "Numeric Type Problems",
        "has_issues": total_problematic_values > 0,
        "columns_checked": numeric_columns,
        "total_problematic_values": total_problematic_values,
        "details": problematic_details
    }


# =====================================================================
# CHECK 8: Invalid Quantity Values (Negative or Zero)
# =====================================================================
def check_invalid_quantities(df: pd.DataFrame, quantity_column: str = "quantity") -> Dict[str, Any]:
    """
    Detects negative and zero quantities in a quantity column.
    Does NOT modify the original DataFrame.
    """
    if quantity_column not in df.columns:
        return {
            "check_name": "Invalid Quantities",
            "has_issues": False,
            "error": f"Column '{quantity_column}' not found in dataset.",
            "negative_count": 0,
            "zero_count": 0
        }
        
    # Coerce to numeric for safe mathematical evaluation
    numeric_qty = pd.to_numeric(df[quantity_column], errors="coerce")
    
    neg_mask = numeric_qty < 0
    zero_mask = numeric_qty == 0
    
    neg_count = int(neg_mask.sum())
    zero_count = int(zero_mask.sum())
    
    neg_indices = df[neg_mask].index.tolist()
    zero_indices = df[zero_mask].index.tolist()
    
    neg_samples = df.loc[neg_mask, quantity_column].tolist()
    zero_samples = df.loc[zero_mask, quantity_column].tolist()
    
    total_issues = neg_count + zero_count
    
    return {
        "check_name": "Invalid Quantities",
        "has_issues": total_issues > 0,
        "column_checked": quantity_column,
        "negative_quantity_count": neg_count,
        "negative_row_indices": neg_indices,
        "negative_sample_values": neg_samples,
        "zero_quantity_count": zero_count,
        "zero_row_indices": zero_indices,
        "zero_sample_values": zero_samples,
        "business_note": (
            "Negative quantities may represent refunds or cancellations in some accounting systems, "
            "but are flagged as warnings for retail sales analysis."
        )
    }


# =====================================================================
# CHECK 9: Outlier Detection (IQR Method)
# =====================================================================
def check_outliers(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detects statistical outliers using the Interquartile Range (IQR) method:
      IQR = Q3 - Q1
      Lower Bound = Q1 - 1.5 * IQR
      Upper Bound = Q3 + 1.5 * IQR
    Values outside [Lower Bound, Upper Bound] are flagged as suspicious outliers.
    Does NOT modify the original DataFrame.
    """
    if numeric_columns is None:
        default_candidates = ["quantity", "unit_price", "revenue"]
        numeric_columns = [col for col in default_candidates if col in df.columns]
        
    outlier_details = {}
    total_outliers = 0
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        # Coerce to numeric and drop NaN values for the quantile calculation
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
            
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        
        lower_bound = round(q1 - (1.5 * iqr), 2)
        upper_bound = round(q3 + (1.5 * iqr), 2)
        
        # Outliers fall strictly below lower bound or strictly above upper bound
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        
        if outlier_count > 0:
            total_outliers += outlier_count
            affected_indices = series[outlier_mask].index.tolist()
            sample_values = series[outlier_mask].head(5).tolist()
            
            outlier_details[col] = {
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "affected_row_indices": affected_indices,
                "sample_outlier_values": sample_values
            }
            
    return {
        "check_name": "Outlier Detection (IQR)",
        "has_issues": total_outliers > 0,
        "columns_checked": numeric_columns,
        "total_outliers_detected": total_outliers,
        "details": outlier_details,
        "method": "IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)",
        "interpretation": "Outliers are statistically unusual data points that require business review, not necessarily errors."
    }


# =====================================================================
# CHECK 10: Revenue Business Rule Consistency
# =====================================================================
def check_revenue_consistency(
    df: pd.DataFrame,
    quantity_column: str = "quantity",
    unit_price_column: str = "unit_price",
    revenue_column: str = "revenue",
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validates the business rule: expected_revenue = quantity * unit_price.
    Compares expected_revenue with actual revenue for valid numeric rows.
    Flags rows where abs(expected_revenue - actual_revenue) > tolerance.
    Does NOT modify the original DataFrame.
    """
    required_cols = [quantity_column, unit_price_column, revenue_column]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        return {
            "check_name": "Revenue Consistency Check",
            "has_issues": False,
            "error": f"Required columns missing: {missing_cols}",
            "inconsistent_row_count": 0,
            "inconsistent_rows": []
        }
        
    # Coerce columns safely to numeric for calculation
    q = pd.to_numeric(df[quantity_column], errors="coerce")
    p = pd.to_numeric(df[unit_price_column], errors="coerce")
    r = pd.to_numeric(df[revenue_column], errors="coerce")
    
    # We can only test rows where all three columns are valid, non-null numbers
    valid_mask = q.notna() & p.notna() & r.notna()
    
    inconsistent_records = []
    
    if valid_mask.sum() > 0:
        expected_revenue = q[valid_mask] * p[valid_mask]
        diff = (r[valid_mask] - expected_revenue).abs()
        
        # Rows where difference exceeds the allowed numerical tolerance
        mismatch_mask = diff > tolerance
        mismatch_indices = q[valid_mask][mismatch_mask].index.tolist()
        
        for idx in mismatch_indices:
            inconsistent_records.append({
                "row_index": idx,
                "quantity": float(q[idx]),
                "unit_price": float(p[idx]),
                "actual_revenue": float(r[idx]),
                "expected_revenue": round(float(expected_revenue[idx]), 2),
                "difference": round(float(r[idx] - expected_revenue[idx]), 2)
            })
            
    return {
        "check_name": "Revenue Business Rule Consistency",
        "has_issues": len(inconsistent_records) > 0,
        "inconsistent_row_count": len(inconsistent_records),
        "inconsistent_rows": inconsistent_records,
        "tolerance": tolerance,
        "formula": f"{revenue_column} == {quantity_column} * {unit_price_column}"
    }
