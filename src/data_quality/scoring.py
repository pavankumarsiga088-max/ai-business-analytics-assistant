"""
Data Quality Scoring Engine.

Calculates an overall Data Quality Score (0 to 100) from check results,
applies dataset-size normalized penalties with severity weights,
categorizes quality status, and generates rule-based recommendations.
"""
from typing import Dict, Any, List, Optional, Tuple


# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================

# Severity weights used to multiply the normalized issue rate (issue_count / total_rows)
DEFAULT_SEVERITY_WEIGHTS = {
    "HIGH": 50.0,
    "MEDIUM": 25.0,
    "LOW": 10.0,
}

# Quality status thresholds (lower_bound_inclusive, status_label)
DEFAULT_STATUS_THRESHOLDS: List[Tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Needs Review"),
    (0.0, "Poor"),
]

# Check catalog configuration: maps check identity to severity, classification,
# issue count extractor, and rule-based recommendation.
CHECK_CATALOG: Dict[str, Dict[str, Any]] = {
    "missing_values": {
        "title": "Missing Values",
        "severity": "HIGH",
        "category": "ERROR",
        "count_keys": ["total_missing_values"],
        "recommendation": "Review columns with missing values before analysis.",
    },
    "duplicate_rows": {
        "title": "Exact Duplicate Rows",
        "severity": "MEDIUM",
        "category": "ERROR",
        "count_keys": ["duplicate_row_count"],
        "recommendation": "Remove or consolidate exact duplicate rows to avoid double-counting.",
    },
    "duplicate_order_ids": {
        "title": "Duplicate Order IDs",
        "severity": "HIGH",
        "category": "ERROR",
        "count_keys": ["duplicate_id_count"],
        "recommendation": "Investigate duplicate order IDs because they may inflate sales metrics.",
    },
    "whitespace": {
        "title": "Whitespace Detection",
        "severity": "LOW",
        "category": "WARNING",
        "count_keys": ["total_whitespace_issues"],
        "recommendation": "Trim extra leading and trailing whitespace from text fields to avoid join and filtering errors.",
    },
    "inconsistent_casing": {
        "title": "Inconsistent Capitalization",
        "severity": "LOW",
        "category": "WARNING",
        "count_keys": ["total_inconsistent_groups"],
        "recommendation": "Standardize text capitalization to avoid fragmented categorical aggregations.",
    },
    "invalid_dates": {
        "title": "Invalid Dates",
        "severity": "HIGH",
        "category": "ERROR",
        "count_keys": ["total_invalid_dates"],
        "recommendation": "Review invalid dates before performing time-series analysis.",
    },
    "numeric_columns": {
        "title": "Numeric Type Problems",
        "severity": "HIGH",
        "category": "ERROR",
        "count_keys": ["total_problematic_values"],
        "recommendation": "Convert or investigate non-numeric values before numerical calculations.",
    },
    "invalid_quantities": {
        "title": "Invalid Quantities",
        "severity": "MEDIUM",
        "category": "WARNING",
        "count_keys": ["negative_quantity_count", "zero_quantity_count"],
        "recommendation": "Verify negative or zero quantities to ensure return or refund transactions are handled correctly.",
    },
    "outliers": {
        "title": "Outlier Detection (IQR)",
        "severity": "LOW",
        "category": "WARNING",
        "count_keys": ["total_outliers_detected"],
        "recommendation": "Review statistical outliers to determine whether they are genuine business events or data-entry errors.",
    },
    "revenue_consistency": {
        "title": "Revenue Business Rule Consistency",
        "severity": "HIGH",
        "category": "ERROR",
        "count_keys": ["inconsistent_row_count"],
        "recommendation": "Investigate rows where revenue does not match quantity * unit_price.",
    },
}


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def identify_check_key(check_name: str) -> str:
    """
    Matches the `check_name` returned by any check function to our internal catalog key.
    """
    normalized = check_name.lower().replace("_", " ")
    
    if "missing" in normalized:
        return "missing_values"
    elif "exact duplicate" in normalized:
        return "duplicate_rows"
    elif "duplicate" in normalized and ("id" in normalized or "order" in normalized):
        return "duplicate_order_ids"
    elif "whitespace" in normalized:
        return "whitespace"
    elif "casing" in normalized or "capitalization" in normalized:
        return "inconsistent_casing"
    elif "date" in normalized:
        return "invalid_dates"
    elif "numeric" in normalized or "type" in normalized:
        return "numeric_columns"
    elif "quantit" in normalized:
        return "invalid_quantities"
    elif "outlier" in normalized:
        return "outliers"
    elif "revenue" in normalized:
        return "revenue_consistency"
    
    return normalized


def extract_issue_count(check_result: Dict[str, Any], catalog_entry: Optional[Dict[str, Any]] = None) -> int:
    """
    Extracts the integer count of issues found in a check result dictionary.
    """
    if not check_result.get("has_issues", False):
        return 0

    if catalog_entry and "count_keys" in catalog_entry:
        count = 0
        for key in catalog_entry["count_keys"]:
            count += int(check_result.get(key, 0))
        if count > 0:
            return count

    # Fallback to common key names
    for key in [
        "issue_count",
        "total_issues",
        "total_missing_values",
        "duplicate_row_count",
        "duplicate_id_count",
        "total_whitespace_issues",
        "total_inconsistent_groups",
        "total_invalid_dates",
        "total_problematic_values",
        "total_outliers_detected",
        "inconsistent_row_count",
    ]:
        if key in check_result:
            return int(check_result[key])

    return 1 if check_result.get("has_issues", False) else 0


def detect_total_rows(check_results: List[Dict[str, Any]], fallback: int = 100) -> int:
    """
    Attempts to detect the total number of rows in the dataset from check results.
    """
    for r in check_results:
        if "total_rows_inspected" in r and r["total_rows_inspected"]:
            return int(r["total_rows_inspected"])
    return fallback


def get_quality_status(
    score: float,
    thresholds: Optional[List[Tuple[float, str]]] = None
) -> str:
    """
    Maps a numeric quality score (0-100) to a qualitative status label.

    Default thresholds:
      90.0 - 100.0 -> Excellent
      75.0 - 89.9  -> Good
      60.0 - 74.9  -> Needs Review
      Below 60.0   -> Poor
    """
    if thresholds is None:
        thresholds = DEFAULT_STATUS_THRESHOLDS

    # Sort thresholds descending by minimum score
    sorted_thresholds = sorted(thresholds, key=lambda t: t[0], reverse=True)
    for min_score, label in sorted_thresholds:
        if score >= min_score:
            return label
    return "Poor"


# =====================================================================
# CORE SCORING LOGIC
# =====================================================================

def calculate_quality_score(
    check_results: List[Dict[str, Any]],
    total_rows: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Calculates an overall Data Quality Score (0 to 100) using a transparent,
    penalty-based formula normalized by dataset size.

    Parameters:
      check_results: List of result dictionaries from the 10 quality checks.
      total_rows: Total number of rows in the dataset. If None, auto-detected.
      weights: Optional custom severity weights dict (e.g. {"HIGH": 40.0, "MEDIUM": 20.0, "LOW": 10.0}).

    Returns:
      Dict containing:
        - overall_score: float (0.0 to 100.0)
        - status: str (e.g., 'Excellent', 'Good', 'Needs Review', 'Poor')
        - total_rows: int
        - total_penalty: float
        - check_details: List of dictionaries detailing each check's penalty
    """
    if weights is None:
        weights = DEFAULT_SEVERITY_WEIGHTS

    # Determine total rows for normalization
    n_rows = total_rows if (total_rows is not None and total_rows > 0) else detect_total_rows(check_results)
    n_rows = max(1, n_rows)  # Prevent division by zero

    check_details = []
    total_penalty = 0.0

    for result in check_results:
        raw_name = result.get("check_name", "Unknown Check")
        catalog_key = identify_check_key(raw_name)
        catalog_entry = CHECK_CATALOG.get(catalog_key, {})

        severity = catalog_entry.get("severity", "MEDIUM")
        category = catalog_entry.get("category", "WARNING" if severity == "LOW" else "ERROR")
        has_issues = bool(result.get("has_issues", False))
        issue_count = extract_issue_count(result, catalog_entry) if has_issues else 0

        # Step 1: Normalize issue count against dataset size
        # Issue rate is bounded between 0.0 and 1.0 (100% of rows)
        issue_rate = min(1.0, issue_count / n_rows) if has_issues else 0.0

        # Step 2: Apply severity weight multiplier
        severity_weight = float(weights.get(severity, 20.0))
        penalty = round(issue_rate * severity_weight, 2) if has_issues else 0.0

        # Determine check outcome status
        if not has_issues:
            check_status = "PASS"
        elif category == "WARNING":
            check_status = "WARNING"
        else:
            check_status = "FAIL"

        check_details.append({
            "check_name": raw_name,
            "catalog_key": catalog_key,
            "has_issues": has_issues,
            "status": check_status,
            "severity": severity,
            "category": category,
            "issue_count": issue_count,
            "issue_rate": round(issue_rate, 4),
            "severity_weight": severity_weight,
            "penalty": round(penalty, 2),
            "recommendation": catalog_entry.get("recommendation", "Review flagged records."),
        })

    # Total penalty equals the exact sum of individual check penalties
    total_penalty = round(sum(c["penalty"] for c in check_details), 2)

    # Step 3: Compute final score (clamped between 0.0 and 100.0)
    raw_score = 100.0 - total_penalty
    overall_score = round(max(0.0, min(100.0, raw_score)), 2)
    status = get_quality_status(overall_score)

    return {
        "overall_score": overall_score,
        "status": status,
        "total_rows": n_rows,
        "total_penalty": total_penalty,
        "check_details": check_details,
    }


# =====================================================================
# REPORT GENERATION
# =====================================================================

def generate_quality_report(
    check_results: List[Dict[str, Any]],
    score_result: Optional[Dict[str, Any]] = None,
    total_rows: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[List[Tuple[float, str]]] = None,
) -> Dict[str, Any]:
    """
    Compiles a comprehensive Data Quality Report containing scores, passed/failed checks,
    warnings, issue counts, severities, and actionable recommendations.

    Parameters:
      check_results: Results from the 10 data quality checks.
      score_result: Optional pre-calculated result from calculate_quality_score.
      total_rows: Dataset size (optional, auto-detected if None).
      weights: Optional custom severity weights.
      thresholds: Optional custom status thresholds.

    Returns:
      Structured report dictionary.
    """
    if score_result is None:
        score_result = calculate_quality_score(check_results, total_rows=total_rows, weights=weights)

    overall_score = score_result["overall_score"]
    status = get_quality_status(overall_score, thresholds=thresholds)

    check_details = score_result["check_details"]
    total_checks = len(check_details)

    passed_checks: List[str] = []
    failed_checks: List[str] = []
    warnings: List[str] = []
    issue_counts: Dict[str, int] = {}
    severity_breakdown: Dict[str, str] = {}
    recommendations: List[Dict[str, str]] = []

    for item in check_details:
        name = item["check_name"]
        issue_counts[name] = item["issue_count"]
        severity_breakdown[name] = item["severity"]

        if not item["has_issues"]:
            passed_checks.append(name)
        else:
            if item["status"] == "WARNING":
                warnings.append(name)
            else:
                failed_checks.append(name)

            recommendations.append({
                "check": name,
                "severity": item["severity"],
                "category": item["category"],
                "issue_count": item["issue_count"],
                "recommendation": item["recommendation"],
            })

    return {
        "overall_score": overall_score,
        "quality_status": status,
        "total_rows": score_result["total_rows"],
        "total_penalty": score_result["total_penalty"],
        "total_checks": total_checks,
        "passed_count": len(passed_checks),
        "failed_count": len(failed_checks),
        "warning_count": len(warnings),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "issue_counts": issue_counts,
        "severity": severity_breakdown,
        "recommendations": recommendations,
        "check_details": check_details,
    }


# =====================================================================
# FORMATTED OUTPUT
# =====================================================================

def format_report_text(report: Dict[str, Any], title: str = "DATA QUALITY REPORT") -> str:
    """
    Formats the quality report dictionary into a clean, human-readable terminal display.
    """
    lines = []
    bar = "=" * 94
    sub_bar = "-" * 94

    lines.append(bar)
    lines.append(f" {title.upper()}")
    lines.append(bar)
    lines.append(f" Overall Quality Score : {report['overall_score']} / 100.0")
    lines.append(f" Quality Status        : {report['quality_status']}")
    lines.append(f" Dataset Size          : {report['total_rows']} rows inspected")
    lines.append(f" Total Deductions      : -{report['total_penalty']:.2f} points")
    lines.append(
        f" Checks Summary        : {report['total_checks']} Total | "
        f"{report['passed_count']} Passed | {report['failed_count']} Failed | {report['warning_count']} Warnings"
    )
    lines.append(sub_bar)

    lines.append(" CHECK BREAKDOWN (Individual Penalty = Issue Rate * Severity Weight):")
    lines.append(
        f"  {'Check Name':<35} {'Severity':<8} {'Weight':<6} {'Issues':<6} {'Issue Rate':<15} {'Penalty':<11} {'Status'}"
    )
    lines.append(
        f"  {'-'*35} {'-'*8} {'-'*6} {'-'*6} {'-'*15} {'-'*11} {'-'*9}"
    )

    for c in report["check_details"]:
        status_tag = f"[{c['status']}]"
        rate_str = f"{c['issue_rate']:.4f} ({c['issue_rate'] * 100:.1f}%)"
        penalty_str = f"-{c['penalty']:.2f} pts" if c["penalty"] > 0 else " 0.00 pts"
        lines.append(
            f"  {c['check_name']:<35} {c['severity']:<8} {c['severity_weight']:<6.1f} {c['issue_count']:<6} {rate_str:<15} {penalty_str:<11} {status_tag}"
        )

    lines.append(sub_bar)
    lines.append(" MATHEMATICAL VERIFICATION:")
    lines.append(f"  * Total Penalty = Sum of Individual Penalties : {report['total_penalty']:.2f} pts")
    lines.append(f"  * Score Formula = 100.0 - {report['total_penalty']:.2f} = {report['overall_score']:.2f} (Clamped [0.0, 100.0])")
    lines.append(f"  * Final Quality Score                         : {report['overall_score']} / 100.0")
    lines.append(f"  * Resulting Status                            : {report['quality_status']}")
    lines.append(sub_bar)

    if report["passed_checks"]:
        lines.append(f" PASSED CHECKS ({len(report['passed_checks'])}):")
        for p in report["passed_checks"]:
            lines.append(f"  [PASS] {p}")
        lines.append(sub_bar)

    if report["failed_checks"]:
        lines.append(f" FAILED CHECKS (ERRORS - {len(report['failed_checks'])}):")
        for f in report["failed_checks"]:
            lines.append(f"  [FAIL] {f} ({report['issue_counts'][f]} issues)")
        lines.append(sub_bar)

    if report["warnings"]:
        lines.append(f" WARNING CHECKS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            lines.append(f"  [WARN] {w} ({report['issue_counts'][w]} flagged items)")
        lines.append(sub_bar)

    if report["recommendations"]:
        lines.append(" ACTIONABLE RECOMMENDATIONS:")
        for idx, rec in enumerate(report["recommendations"], 1):
            tag = f"[{rec['category']}]"
            lines.append(
                f"  {idx}. {tag} {rec['check']} ({rec['severity']} Severity, {rec['issue_count']} affected):"
            )
            lines.append(f"     -> {rec['recommendation']}")
    else:
        lines.append(" ACTIONABLE RECOMMENDATIONS:")
        lines.append("  All quality checks passed. No corrective actions required.")

    lines.append(bar)
    return "\n".join(lines)
