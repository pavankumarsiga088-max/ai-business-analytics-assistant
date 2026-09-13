"""
Data Quality module: Handles file loading and quality validation checks.
"""
from src.data_quality.loader import load_csv, load_excel, load_dataset
from src.data_quality.checks import (
    check_missing_values,
    check_duplicate_rows,
    check_duplicate_order_ids,
    check_whitespace,
    check_inconsistent_casing,
    check_invalid_dates,
    check_numeric_columns,
    check_invalid_quantities,
    check_outliers,
    check_revenue_consistency,
)

from src.data_quality.scoring import (
    calculate_quality_score,
    generate_quality_report,
    format_report_text,
    get_quality_status,
)
from src.data_quality.issue_details import (
    QualityIssueDetail,
    build_issue_detail,
    extract_issue_details,
    get_issue_severity_counts,
)
from src.data_quality.dashboard import (
    get_score_interpretation,
    create_issue_distribution_chart,
    get_priority_areas,
    build_quality_check_overview_df,
)
from src.data_quality.report_export import (
    build_quality_report_html,
    generate_quality_report_bytes,
    build_quality_report_filename,
)

__all__ = [
    "load_csv",
    "load_excel",
    "load_dataset",
    "check_missing_values",
    "check_duplicate_rows",
    "check_duplicate_order_ids",
    "check_whitespace",
    "check_inconsistent_casing",
    "check_invalid_dates",
    "check_numeric_columns",
    "check_invalid_quantities",
    "check_outliers",
    "check_revenue_consistency",
    "calculate_quality_score",
    "generate_quality_report",
    "format_report_text",
    "get_quality_status",
    "QualityIssueDetail",
    "build_issue_detail",
    "extract_issue_details",
    "get_issue_severity_counts",
    "get_score_interpretation",
    "create_issue_distribution_chart",
    "get_priority_areas",
    "build_quality_check_overview_df",
    "build_quality_report_html",
    "generate_quality_report_bytes",
    "build_quality_report_filename",
]

