"""
Data Quality Report Export Module (Phase 12C).

Generates standalone, browser-viewable HTML reports of the current dataset's
data quality assessment.

Key Principles:
- Pure Python and in-memory export (zero disk footprint, zero temporary files).
- Strict HTML escaping on all dynamic data to prevent XSS / markup corruption.
- Bounded sample sizes (max 5 example values, max 5 sample rows per issue).
- Self-contained styling: no external CSS, JavaScript, or CDN dependencies.
- Zero AI dependencies, zero DuckDB connections, zero SQL execution.
- Input DataFrames are strictly read-only and never mutated.
"""
from datetime import datetime, timezone
import html
import re
from typing import Dict, Any, List, Optional
import pandas as pd

from src.data_quality.issue_details import QualityIssueDetail
from src.data_quality.dashboard import (
    get_score_interpretation,
    get_priority_areas,
    build_quality_check_overview_df,
)


# =====================================================================
# HELPER UTILITIES & FILENAME SANITIZATION
# =====================================================================

def esc(val: Any) -> str:
    """Safely escapes arbitrary text for secure HTML insertion."""
    if val is None:
        return ""
    return html.escape(str(val))


def build_quality_report_filename(
    dataset_name: str,
    prefix: str = "data_quality_report"
) -> str:
    """
    Constructs a safe, sanitized filename for downloading the HTML quality report.
    Strips path traversal patterns and filesystem-illegal characters.
    """
    if not dataset_name or not isinstance(dataset_name, str):
        clean_base = "dataset"
    else:
        # Strip existing file extensions
        clean_base = re.sub(r"\.[a-zA-Z0-9]+$", "", dataset_name.strip())
        # Replace illegal filesystem and path characters with underscores
        clean_base = re.sub(r'[\/\\:\*\?"<>\|\s]+', "_", clean_base)
        # Keep only alphanumeric, hyphen, underscore
        clean_base = re.sub(r"[^a-zA-Z0-9_\-]", "_", clean_base)
        clean_base = re.sub(r"_+", "_", clean_base).strip("_-")
        if not clean_base:
            clean_base = "dataset"

    return f"{prefix}_{clean_base[:48]}.html"


# =====================================================================
# EMBEDDED STYLING
# =====================================================================

REPORT_CSS = """
:root {
    --primary: #1E3A8A;
    --primary-light: #EFF6FF;
    --text-main: #1F2937;
    --text-muted: #4B5563;
    --border-color: #E5E7EB;
    --bg-card: #FFFFFF;
    --bg-page: #F9FAFB;
    --high: #DC2626;
    --high-bg: #FEF2F2;
    --high-border: #FCA5A5;
    --med: #D97706;
    --med-bg: #FFFBEB;
    --med-border: #FCD34D;
    --low: #CA8A04;
    --low-bg: #FEFCE8;
    --low-border: #FDE047;
    --pass: #16A34A;
    --pass-bg: #F0FDF4;
    --pass-border: #86EFAC;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-main);
    background-color: var(--bg-page);
    line-height: 1.5;
    padding: 32px 16px;
}

.container {
    max-width: 960px;
    margin: 0 auto;
    background: var(--bg-card);
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border: 1px solid var(--border-color);
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    color: #FFFFFF;
    padding: 36px 32px;
}

.header h1 {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.header p {
    font-size: 14px;
    opacity: 0.9;
}

.metadata-bar {
    background: #F3F4F6;
    border-bottom: 1px solid var(--border-color);
    padding: 12px 32px;
    font-size: 13px;
    color: var(--text-muted);
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    justify-content: space-between;
}

.content {
    padding: 32px;
}

.section {
    margin-bottom: 36px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--primary);
    border-bottom: 2px solid var(--primary-light);
    padding-bottom: 8px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Score Visual Card */
.score-card {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 24px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 24px;
    align-items: center;
    margin-bottom: 24px;
}

.score-box {
    text-align: center;
    border-right: 2px solid #E2E8F0;
    padding-right: 24px;
}

.score-number {
    font-size: 42px;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
}

.score-max {
    font-size: 14px;
    color: var(--text-muted);
    font-weight: 500;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 700;
    margin-top: 8px;
    text-transform: uppercase;
}

.status-excellent { background: var(--pass-bg); color: var(--pass); border: 1px solid var(--pass-border); }
.status-good { background: var(--med-bg); color: var(--med); border: 1px solid var(--med-border); }
.status-review { background: var(--low-bg); color: var(--low); border: 1px solid var(--low-border); }
.status-poor { background: var(--high-bg); color: var(--high); border: 1px solid var(--high-border); }

.score-details h4 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 6px;
}

.score-details p {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 12px;
}

/* Progress bar */
.progress-container {
    width: 100%;
    height: 10px;
    background: #E2E8F0;
    border-radius: 9999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%);
    border-radius: 9999px;
}

/* Severity Grid */
.severity-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.severity-card {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid var(--border-color);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.severity-card.high { border-top: 4px solid var(--high); }
.severity-card.medium { border-top: 4px solid var(--med); }
.severity-card.low { border-top: 4px solid var(--low); }
.severity-card.passed { border-top: 4px solid var(--pass); }

.severity-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
    color: var(--text-muted);
}

.severity-count {
    font-size: 28px;
    font-weight: 800;
}

.severity-card.high .severity-count { color: var(--high); }
.severity-card.medium .severity-count { color: var(--med); }
.severity-card.low .severity-count { color: var(--low); }
.severity-card.passed .severity-count { color: var(--pass); }

/* Tables */
.table-container {
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 16px;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: left;
}

th {
    background: #F9FAFB;
    color: var(--text-main);
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    white-space: nowrap;
}

td {
    padding: 10px 14px;
    border-bottom: 1px solid #F3F4F6;
    color: var(--text-muted);
}

tr:last-child td {
    border-bottom: none;
}

tr:nth-child(even) {
    background: #FAFAFA;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-high { background: var(--high-bg); color: var(--high); border: 1px solid var(--high-border); }
.badge-medium { background: var(--med-bg); color: var(--med); border: 1px solid var(--med-border); }
.badge-low { background: var(--low-bg); color: var(--low); border: 1px solid var(--low-border); }
.badge-pass { background: var(--pass-bg); color: var(--pass); border: 1px solid var(--pass-border); }

/* Priority Areas */
.priority-list {
    list-style: none;
}

.priority-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-bottom: 10px;
}

.priority-rank {
    background: var(--primary);
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 700;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.priority-content h5 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 2px;
}

.priority-content p {
    font-size: 13px;
    color: var(--text-muted);
}

/* Detailed Issue Cards */
.issue-card {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 18px;
    background: #FFFFFF;
}

.issue-card.high { border-left: 5px solid var(--high); }
.issue-card.medium { border-left: 5px solid var(--med); }
.issue-card.low { border-left: 5px solid var(--low); }
.issue-card.pass { border-left: 5px solid var(--pass); }

.issue-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.issue-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-main);
}

.issue-meta {
    font-size: 13px;
    color: var(--text-muted);
    margin-bottom: 10px;
}

.issue-meta strong {
    color: var(--text-main);
}

.issue-desc {
    font-size: 13px;
    color: var(--text-muted);
    margin-bottom: 12px;
}

.issue-samples {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    margin-bottom: 12px;
}

.sample-tag {
    display: inline-block;
    background: #E2E8F0;
    color: #1E293B;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    margin: 2px 4px 2px 0;
}

.recommendation-box {
    background: #EFF6FF;
    border-left: 3px solid #3B82F6;
    padding: 8px 12px;
    font-size: 13px;
    color: #1E40AF;
    border-radius: 0 6px 6px 0;
}

/* Clean Alert Banner */
.clean-alert {
    background: var(--pass-bg);
    border: 1px solid var(--pass-border);
    color: #14532D;
    padding: 16px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 20px;
}

.footer {
    background: #F9FAFB;
    border-top: 1px solid var(--border-color);
    padding: 20px 32px;
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
}

@media print {
    body { padding: 0; background: #FFFFFF; }
    .container { box-shadow: none; border: none; max-width: 100%; }
    .header { padding: 20px; }
    .content { padding: 20px; }
}
"""


# =====================================================================
# TABLE BUILDER FOR SAMPLE RECORDS
# =====================================================================

def _render_sample_records_table(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Renders a clean, bounded, HTML-escaped table of sample problematic rows.
    Caps rows at max_rows (default 5).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return ""

    bounded_df = df.head(max_rows)
    headers = "".join(f"<th>{esc(col)}</th>" for col in bounded_df.columns)

    rows_html = []
    for _, row in bounded_df.iterrows():
        cells = "".join(f"<td>{esc(row[col])}</td>" for col in bounded_df.columns)
        rows_html.append(f"<tr>{cells}</tr>")

    return (
        f'<div style="margin-top: 8px; margin-bottom: 12px;">'
        f'<span style="font-size: 12px; font-weight: 600; color: #4B5563;">Sample Problematic Rows (up to 5):</span>'
        f'<div class="table-container" style="margin-top: 4px;">'
        f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows_html)}</tbody></table>'
        f'</div></div>'
    )


# =====================================================================
# HTML REPORT GENERATOR
# =====================================================================

def build_quality_report_html(
    dataset_name: str,
    score_report: Dict[str, Any],
    issue_details: List[QualityIssueDetail],
    severity_counts: Dict[str, int],
    priority_areas: Optional[List[QualityIssueDetail]] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Compiles an executive-grade, standalone HTML document representing the
    current dataset's data quality assessment.

    Parameters:
        dataset_name: Name of the uploaded dataset file.
        score_report: Dictionary from calculate_quality_score or generate_quality_report.
        issue_details: List of QualityIssueDetail instances from extract_issue_details.
        severity_counts: Dict with counts for HIGH, MEDIUM, LOW, and PASS.
        priority_areas: Optional ordered priority issues list (auto-computed if None).
        timestamp: Optional formatted timestamp string (defaults to current UTC time).

    Returns:
        str: Full standalone HTML document string.
    """
    safe_dataset_name = dataset_name if dataset_name else "Unknown Dataset"
    report_time = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    score = float(score_report.get("overall_score", 100.0))
    status = str(score_report.get("status") or score_report.get("quality_status", "Excellent"))
    score_interp = get_score_interpretation(score, status)
    total_rows = int(score_report.get("total_rows", 0))
    total_penalty = float(score_report.get("total_penalty", 0.0))

    if priority_areas is None:
        priority_areas = get_priority_areas(issue_details)

    # Status badge CSS class
    status_lower = status.lower()
    if "excellent" in status_lower:
        status_css = "status-excellent"
    elif "good" in status_lower:
        status_css = "status-good"
    elif "review" in status_lower:
        status_css = "status-review"
    else:
        status_css = "status-poor"

    # Progress bar width (0 to 100%)
    progress_width = max(0.0, min(100.0, score))

    # Build 10-check overview table rows
    overview_df = build_quality_check_overview_df(issue_details)
    overview_rows_html = []
    for _, row in overview_df.iterrows():
        sev = str(row["Severity"]).upper()
        badge_class = f"badge-{sev.lower()}"
        overview_rows_html.append(
            f"<tr>"
            f"<td><strong>{esc(row['Status'])}</strong></td>"
            f"<td>{esc(row['Check Name'])}</td>"
            f"<td><span class=\"badge {badge_class}\">{esc(sev)}</span></td>"
            f"<td>{esc(row['Issues'])}</td>"
            f"<td>{esc(row['Affected Columns'])}</td>"
            f"</tr>"
        )
    overview_table_html = "".join(overview_rows_html)

    # Build Priority Areas HTML
    if not priority_areas:
        priority_html = (
            '<div class="clean-alert">'
            '<strong>✓ No Priority Actions Required:</strong> '
            'All 10 quality checks passed with zero anomalies. The dataset is ready for analysis.'
            '</div>'
        )
    else:
        priority_items = []
        for rank, p in enumerate(priority_areas, 1):
            sev_class = f"badge-{p.severity.lower()}"
            cols_text = f" ({', '.join(p.affected_columns)})" if p.affected_columns else ""
            priority_items.append(
                f'<li class="priority-item">'
                f'<div class="priority-rank">{rank}</div>'
                f'<div class="priority-content">'
                f'<h5>{esc(p.check_name)} <span class="badge {sev_class}">{esc(p.severity)}</span> '
                f'<span style="font-size: 12px; color: #6B7280;">({p.issue_count} issue{"s" if p.issue_count != 1 else ""}{esc(cols_text)})</span></h5>'
                f'<p>{esc(p.recommendation)}</p>'
                f'</div>'
                f'</li>'
            )
        priority_html = f'<ul class="priority-list">{"".join(priority_items)}</ul>'

    # Build Detailed Issue Cards
    issue_cards = []
    for issue in issue_details:
        sev_lower = issue.severity.lower()
        badge_class = f"badge-{sev_lower}"

        cols_str = ", ".join(issue.affected_columns) if issue.affected_columns else "None"

        # Bounded sample values (max 5)
        samples_html = ""
        if issue.status != "PASS" and issue.sample_values:
            bounded_samples = issue.sample_values[:5]
            sample_tags = "".join(f'<span class="sample-tag">{esc(val)}</span>' for val in bounded_samples)
            if len(issue.sample_values) > 5:
                sample_tags += f'<span class="sample-tag" style="background:#E5E7EB; color:#4B5563;">+{len(issue.sample_values) - 5} more</span>'
            samples_html = (
                f'<div class="issue-samples">'
                f'<strong style="color: #374151;">Example problematic values (up to 5):</strong><br>'
                f'{sample_tags}'
                f'</div>'
            )

        # Bounded sample records table (max 5 rows)
        records_table_html = ""
        if issue.status != "PASS" and issue.sample_records is not None and isinstance(issue.sample_records, pd.DataFrame) and not issue.sample_records.empty:
            records_table_html = _render_sample_records_table(issue.sample_records, max_rows=5)

        rows_info = f" · Affected rows: <strong>{issue.affected_row_count}</strong>" if issue.status != "PASS" else ""

        issue_cards.append(
            f'<div class="issue-card {sev_lower}">'
            f'<div class="issue-header">'
            f'<div class="issue-title">{esc(issue.check_name)}</div>'
            f'<span class="badge {badge_class}">{esc(issue.severity)}</span>'
            f'</div>'
            f'<div class="issue-meta">'
            f'Status: <strong>{esc(issue.status)}</strong> · '
            f'Issues: <strong>{issue.issue_count}</strong> · '
            f'Affected columns: <strong>{esc(cols_str)}</strong>'
            f'{rows_info}'
            f'</div>'
            f'<div class="issue-desc"><strong>What this means:</strong> {esc(issue.message)}</div>'
            f'{samples_html}'
            f'{records_table_html}'
            f'<div class="recommendation-box"><strong>Recommended action:</strong> {esc(issue.recommendation)}</div>'
            f'</div>'
        )
    detailed_issues_html = "".join(issue_cards)

    # Clean dataset banner if 0 issues
    clean_banner = ""
    if severity_counts.get("TOTAL_ISSUES", 0) == 0:
        clean_banner = (
            '<div class="clean-alert">'
            '<strong>✓ Excellent Data Quality:</strong> '
            'No quality issues detected across any of the 10 checks.'
            '</div>'
        )

    # Assembly
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Assessment — {esc(safe_dataset_name)}</title>
    <style>
{REPORT_CSS}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1>Data Quality Assessment Report</h1>
            <p>Comprehensive automated health check, integrity scoring, and remediation plan</p>
        </header>

        <!-- Metadata Bar -->
        <div class="metadata-bar">
            <span><strong>Dataset:</strong> {esc(safe_dataset_name)}</span>
            <span><strong>Rows Inspected:</strong> {total_rows:,}</span>
            <span><strong>Generated:</strong> {esc(report_time)}</span>
        </div>

        <main class="content">
            {clean_banner}

            <!-- 1. Score Visual -->
            <section class="section">
                <h3 class="section-title">Data Quality Score</h3>
                <div class="score-card">
                    <div class="score-box">
                        <div class="score-number">{score:.2f}</div>
                        <div class="score-max">out of 100.0</div>
                        <div class="status-badge {status_css}">{esc(status)}</div>
                    </div>
                    <div class="score-details">
                        <h4>Health Assessment</h4>
                        <p>{esc(score_interp)}</p>
                        <div class="progress-container">
                            <div class="progress-fill" style="width: {progress_width:.1f}%;"></div>
                        </div>
                        <div style="font-size: 12px; color: #6B7280; margin-top: 8px;">
                            Total deductions: <strong>-{total_penalty:.2f} points</strong> across 10 evaluation checks
                        </div>
                    </div>
                </div>
            </section>

            <!-- 2. Severity Overview -->
            <section class="section">
                <h3 class="section-title">Severity Summary</h3>
                <div class="severity-grid">
                    <div class="severity-card high">
                        <div class="severity-label">High Severity</div>
                        <div class="severity-count">{severity_counts.get("HIGH", 0)}</div>
                    </div>
                    <div class="severity-card medium">
                        <div class="severity-label">Medium Severity</div>
                        <div class="severity-count">{severity_counts.get("MEDIUM", 0)}</div>
                    </div>
                    <div class="severity-card low">
                        <div class="severity-label">Low Severity</div>
                        <div class="severity-count">{severity_counts.get("LOW", 0)}</div>
                    </div>
                    <div class="severity-card passed">
                        <div class="severity-label">Passed Checks</div>
                        <div class="severity-count">{severity_counts.get("PASS", 0)}</div>
                    </div>
                </div>
            </section>

            <!-- 3. Priority Remediation Areas -->
            <section class="section">
                <h3 class="section-title">Priority Remediation Areas</h3>
                {priority_html}
            </section>

            <!-- 4. Quality Check Overview Table -->
            <section class="section">
                <h3 class="section-title">Quality Check Overview</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Status</th>
                                <th>Check Name</th>
                                <th>Severity</th>
                                <th>Issues</th>
                                <th>Affected Columns</th>
                            </tr>
                        </thead>
                        <tbody>
                            {overview_table_html}
                        </tbody>
                    </table>
                </div>
            </section>

            <!-- 5. Detailed Quality Issues -->
            <section class="section">
                <h3 class="section-title">Detailed Quality Issues</h3>
                {detailed_issues_html}
            </section>
        </main>

        <!-- Footer -->
        <footer class="footer">
            Generated by AI Business Analytics & Data Quality Assistant · Fully Deterministic Quality Engine
        </footer>
    </div>
</body>
</html>
"""


def generate_quality_report_bytes(
    dataset_name: str,
    score_report: Dict[str, Any],
    issue_details: List[QualityIssueDetail],
    severity_counts: Dict[str, int],
    priority_areas: Optional[List[QualityIssueDetail]] = None,
    timestamp: Optional[str] = None,
) -> bytes:
    """
    Generates UTF-8 encoded bytes of the standalone HTML report
    ready for Streamlit download buttons and file export.
    """
    html_content = build_quality_report_html(
        dataset_name=dataset_name,
        score_report=score_report,
        issue_details=issue_details,
        severity_counts=severity_counts,
        priority_areas=priority_areas,
        timestamp=timestamp,
    )
    return html_content.encode("utf-8")
