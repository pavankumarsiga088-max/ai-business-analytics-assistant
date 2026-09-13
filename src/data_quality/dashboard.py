"""
Data Quality Dashboard Presentation Module (Phase 12B).

Provides pure, deterministic presentation helpers for the Data Quality Dashboard:
- Visual score interpretation
- Severity issue distribution charts (Plotly)
- Deterministic priority remediation ordering
- Formatted quality check overview tables

Zero AI dependencies, zero database connections, and zero DataFrame mutations.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go

from src.data_quality.issue_details import QualityIssueDetail


# =====================================================================
# SCORE INTERPRETATION
# =====================================================================

def get_score_interpretation(score: float, status: Optional[str] = None) -> str:
    """
    Provides a concise, professional assessment of dataset health
    based on the overall Data Quality Score (0 to 100).
    """
    if score >= 90.0:
        return "Dataset demonstrates high integrity with minimal to no quality anomalies."
    elif score >= 75.0:
        return "Dataset is generally healthy but contains some issues that should be reviewed."
    elif score >= 60.0:
        return "Dataset contains notable quality issues that may distort downstream analysis."
    else:
        return "Dataset has critical quality flaws requiring remediation prior to business analysis."


# =====================================================================
# ISSUE DISTRIBUTION VISUALIZATION
# =====================================================================

SEVERITY_COLOR_PALETTE = {
    "High": "#EF4444",      # Vibrant Red
    "Medium": "#F59E0B",    # Warm Amber
    "Low": "#EAB308",       # Warning Yellow
    "Passed": "#10B981",    # Emerald Green
}

def create_issue_distribution_chart(
    severity_counts: Dict[str, int],
    title: str = "Checks by Severity Level"
) -> go.Figure:
    """
    Builds a clean, compact Plotly bar chart displaying the distribution
    of quality checks across severity levels (High, Medium, Low, Passed).

    Parameters:
        severity_counts: Dict containing counts for 'HIGH', 'MEDIUM', 'LOW', and 'PASS'.
        title: Chart title string.

    Returns:
        plotly.graph_objects.Figure
    """
    categories = ["High", "Medium", "Low", "Passed"]
    counts = [
        int(severity_counts.get("HIGH", 0)),
        int(severity_counts.get("MEDIUM", 0)),
        int(severity_counts.get("LOW", 0)),
        int(severity_counts.get("PASS", 0)),
    ]
    colors = [
        SEVERITY_COLOR_PALETTE["High"],
        SEVERITY_COLOR_PALETTE["Medium"],
        SEVERITY_COLOR_PALETTE["Low"],
        SEVERITY_COLOR_PALETTE["Passed"],
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=counts,
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(0,0,0,0.15)", width=1),
                ),
                text=counts,
                textposition="auto",
                textfont=dict(size=13, weight="bold"),
                hovertemplate="<b>%{x} Severity</b><br>Checks: %{y}<extra></extra>",
            )
        ]
    )

    max_count = max(counts) if counts else 10
    y_upper = max(5, int(max_count * 1.25) + 1)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14),
            x=0.02,
            y=0.95,
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            tickfont=dict(size=12, weight="bold"),
        ),
        yaxis=dict(
            title="Number of Checks",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            range=[0, y_upper],
            dtick=1 if y_upper <= 10 else None,
        ),
        margin=dict(l=35, r=20, t=35, b=25),
        height=240,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =====================================================================
# PRIORITY / ATTENTION AREAS
# =====================================================================

def get_priority_areas(
    issue_details: List[QualityIssueDetail]
) -> List[QualityIssueDetail]:
    """
    Extracts and ranks identified data quality issues in strict order of priority:
    1. HIGH severity issues appear before MEDIUM severity issues.
    2. MEDIUM severity issues appear before LOW severity issues.
    3. Within the same severity level, issues are sorted by issue_count descending.
    4. Alphabetical check name is used as a deterministic tie-breaker.
    5. Checks with status 'PASS' or 0 issues are excluded.

    Parameters:
        issue_details: List of QualityIssueDetail objects.

    Returns:
        Sorted list of QualityIssueDetail objects requiring attention.
    """
    severity_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    problematic_issues = [iss for iss in issue_details if iss.has_issues]

    return sorted(
        problematic_issues,
        key=lambda x: (
            severity_order.get(x.severity.upper(), 99),
            -x.issue_count,
            x.check_name,
        ),
    )


# =====================================================================
# QUALITY CHECK OVERVIEW TABLE
# =====================================================================

def build_quality_check_overview_df(
    issue_details: List[QualityIssueDetail]
) -> pd.DataFrame:
    """
    Constructs a clean summary DataFrame representing all 10 quality checks
    for the dashboard overview table.

    Columns:
        - Status: Formatted tag (e.g. '✓ PASS', '✗ FAIL', '⚠ REVIEW')
        - Check Name: Title of the check
        - Severity: Authoritative severity level
        - Issues: Number of flagged items
        - Affected Columns: Comma-separated list of affected columns or 'None'
    """
    rows = []
    for issue in issue_details:
        if issue.status == "PASS":
            status_tag = "✓ PASS"
        elif issue.severity == "LOW":
            status_tag = "⚠ REVIEW"
        else:
            status_tag = "✗ FAIL"

        cols_str = ", ".join(issue.affected_columns) if issue.affected_columns else "None"

        rows.append({
            "Status": status_tag,
            "Check Name": issue.check_name,
            "Severity": issue.severity,
            "Issues": issue.issue_count,
            "Affected Columns": cols_str,
        })

    return pd.DataFrame(rows)
