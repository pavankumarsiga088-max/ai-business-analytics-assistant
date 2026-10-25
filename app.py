"""
Streamlit Data Quality & AI Business Analytics Dashboard.

Phase 11A: Professional UI/UX Modernization for Data Analyst Portfolio.
Connects the data quality backend (loader, checks, scoring) and analytical
backend (DuckDB engine, NL-to-SQL, Plotly visualization, AI business insights,
and conversational memory) into a clean, modern, and professional analytics application.
"""
import os
import re
import pandas as pd
import streamlit as st

from src.data_quality.loader import load_dataset
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
)
from src.data_quality.issue_details import (
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
    generate_quality_report_bytes,
    build_quality_report_filename,
)
from src.analytics.database import (
    AnalyticsDatabase,
    create_database_engine,
    get_configured_database_backend,
    is_backend_available,
    UnsupportedDatabaseError,
)
from src.analytics.dataset_loader import (
    load_dataset_into_database,
    DatasetLoadResult,
)
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    validate_read_only_query,
    DisallowedQueryError,
    QueryExecutionError,
)
from src.analytics.postgres_engine import (
    PostgresEngine,
    PostgresConnectionError,
)
from src.analytics.nl_to_sql import (
    generate_sql,
    EmptyQuestionError,
    LLMGenerationError,
    LLMQuotaExceededError,
    LLMAuthenticationError,
    LLMServiceUnavailableError,
    QuestionInterpretationError,
)
from src.analytics.visualization import create_chart
from src.analytics.insights import (
    generate_business_insights,
    InsightsError,
    EmptyResultError,
    LLMInsightGenerationError,
)
from src.analytics.conversation import ConversationHistory
from src.analytics.result_presentation import build_kpi_metadata
from src.analytics.downloads import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    figure_to_html_bytes,
    sanitize_filename,
    build_download_filename,
)
from src.analytics.sql_explanation import generate_sql_explanation
from src.analytics.question_suggestions import generate_suggested_questions
from src.config import (
    get_database_backend,
    get_gemini_model,
    get_max_query_result_rows,
    get_query_timeout_seconds,
    is_gemini_configured,
    is_postgres_configured,
    MAX_QUERY_RESULT_ROWS,
)


def build_result_column_config(df: pd.DataFrame) -> dict:
    """
    Dynamically generates a Streamlit column configuration dictionary for arbitrary
    query result DataFrames to maximize readability without mutating the DataFrame.

    - Formats floating-point measures to 2 decimal places (%.2f).
    - Formats integer quantities/counts cleanly (%d).
    - Formats datetime columns as YYYY-MM-DD.
    - Preserves column headers without excessive compression.
    """
    if df is None or df.empty:
        return {}

    config = {}
    for col in df.columns:
        col_str = str(col)
        dtype = df[col].dtype

        if pd.api.types.is_float_dtype(dtype):
            config[col] = st.column_config.NumberColumn(
                label=col_str,
                format="%.2f",
                help=f"Numeric measure ({col_str})",
            )
        elif pd.api.types.is_integer_dtype(dtype):
            config[col] = st.column_config.NumberColumn(
                label=col_str,
                format="%d",
                help=f"Integer value ({col_str})",
            )
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            config[col] = st.column_config.DatetimeColumn(
                label=col_str,
                format="YYYY-MM-DD",
                help=f"Date/Time ({col_str})",
            )
        else:
            config[col] = st.column_config.TextColumn(
                label=col_str,
                help=f"Text/Categorical ({col_str})",
            )
    return config


def run_all_quality_checks(df: pd.DataFrame) -> list:
    """
    Executes all 10 independent quality checks on the provided DataFrame.
    Does not modify or clean the input DataFrame.
    """
    return [
        check_missing_values(df),
        check_duplicate_rows(df),
        check_duplicate_order_ids(df, id_column="order_id"),
        check_whitespace(df),
        check_inconsistent_casing(df),
        check_invalid_dates(df, date_columns=["order_date"]),
        check_numeric_columns(df, numeric_columns=["quantity", "unit_price", "revenue"]),
        check_invalid_quantities(df, quantity_column="quantity"),
        check_outliers(df, numeric_columns=["quantity", "unit_price", "revenue"]),
        check_revenue_consistency(
            df,
            quantity_column="quantity",
            unit_price_column="unit_price",
            revenue_column="revenue",
        ),
    ]


def main():
    # =========================================================================
    # 1. APPLICATION HEADER
    # =========================================================================
    st.set_page_config(
        page_title="AI Business Analytics Assistant",
        page_icon="📊",
        layout="wide",
    )

    st.title("AI Business Analytics Assistant")
    st.markdown(
        "**Ask business questions, explore your data, and generate actionable insights using natural language.**"
    )
    st.caption(
        "Automated Data Quality Profiling & Dual-Engine AI Analytics (DuckDB / PostgreSQL) | 100% Safe Read-Only Execution"
    )

    # =========================================================================
    # 2. DATASET INGESTION & OVERVIEW
    # =========================================================================
    st.header("Dataset Overview")

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload your e-commerce, sales, or business transaction dataset.",
        )

    sample_choice = None
    with col_sample:
        st.markdown("**Or test with sample data:**")
        sample_options = ["None", "Clean Dataset (sample_sales.csv)", "Dirty Dataset (sample_dirty_data.xlsx)"]
        selected_sample = st.selectbox("Quick Sample Loader", sample_options)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if selected_sample == "Clean Dataset (sample_sales.csv)":
            candidate = os.path.join(app_dir, "sample_data", "sample_sales.csv")
            sample_choice = candidate if os.path.exists(candidate) else "sample_data/sample_sales.csv"
        elif selected_sample == "Dirty Dataset (sample_dirty_data.xlsx)":
            candidate = os.path.join(app_dir, "sample_data", "sample_dirty_data.xlsx")
            sample_choice = candidate if os.path.exists(candidate) else "sample_data/sample_dirty_data.xlsx"

    # Determine file source
    df = None
    dataset_name = ""

    if uploaded_file is not None:
        try:
            dataset_name = uploaded_file.name
            df = load_dataset(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return
    elif sample_choice and os.path.exists(sample_choice):
        try:
            dataset_name = os.path.basename(sample_choice)
            df = load_dataset(sample_choice)
        except Exception as e:
            st.error(f"Error loading sample file: {e}")
            return

    # Dataset boundary enforcement: Reset cached analytics result and conversation history
    # whenever a new dataset is selected or uploaded
    if "current_dataset_name" not in st.session_state or st.session_state["current_dataset_name"] != dataset_name:
        st.session_state["current_dataset_name"] = dataset_name
        st.session_state.pop("last_analytics_result", None)
        st.session_state["user_nl_question"] = ""
        if "conversation_history" in st.session_state:
            st.session_state["conversation_history"].reset_for_dataset(dataset_name=dataset_name)
        else:
            st.session_state["conversation_history"] = ConversationHistory(dataset_name=dataset_name)

    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = ConversationHistory(dataset_name=dataset_name)

    if df is None:
        st.session_state.pop("last_analytics_result", None)
        st.markdown("---")
        st.subheader("👋 Welcome to the AI Business Analytics & Data Quality Assistant")
        st.markdown(
            """
            This platform bridges data quality engineering and conversational business intelligence:
            
            **How to get started in 4 simple steps:**
            1. **Select or Upload a Dataset**: Choose **Clean Dataset** or **Dirty Dataset** from the Quick Sample Loader above (or upload your own `.csv` / `.xlsx`).
            2. **Inspect Data Quality**: Review the automated 10-point quality scorecard, overall letter grade, and interactive Issue Explorer.
            3. **Select Analytical Engine**: Choose **DuckDB (In-Memory)** for instant local analysis or **PostgreSQL** for enterprise relational execution.
            4. **Ask Natural Language Questions**: Type business questions or click suggested queries to generate validated SQL, automated charts, and AI business takeaways.
            """
        )
        st.info("💡 **Quick Demo**: Select **'Clean Dataset (sample_sales.csv)'** in the dropdown above to test all features instantly.")
        return

    if df.empty or len(df.columns) == 0:
        st.session_state.pop("last_analytics_result", None)
        st.error("The selected dataset contains no rows or no columns. Please upload a valid dataset with at least one column and row.")
        return

    # Compute Data Quality report once for overview metrics and detailed checks
    with st.spinner("Analyzing dataset quality..."):
        check_results = run_all_quality_checks(df)
        score_result = calculate_quality_score(check_results, total_rows=len(df))
        report = generate_quality_report(
            check_results=check_results,
            score_result=score_result,
            total_rows=len(df),
        )

    # KPI Summary Cards (Requirement 2)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    display_name = dataset_name if len(dataset_name) <= 24 else dataset_name[:21] + "..."
    kpi_col1.metric("📁 Dataset Name", display_name, help=dataset_name)
    kpi_col2.metric("📊 Total Rows", f"{len(df):,}")
    kpi_col3.metric("📐 Total Columns", f"{len(df.columns):,}")
    kpi_col4.metric(
        "🛡️ Data Quality Score",
        f"{report['overall_score']:.1f} / 100",
        delta=report["quality_status"],
        delta_color="normal" if report["overall_score"] >= 80 else "inverse",
    )

    with st.expander("Preview Dataset (First 10 Rows)", expanded=False):
        st.dataframe(
            df.head(10),
            column_config=build_result_column_config(df.head(10)),
            width='stretch',
            hide_index=True,
        )

    # =========================================================================
    # 3. DATA QUALITY DASHBOARD (Phase 12B)
    # =========================================================================
    st.divider()
    st.header("Data Quality Dashboard")
    st.caption("Comprehensive data health diagnostics, severity distribution, and prioritized remediation areas.")

    # Extract issue details & severity counts
    issue_details = extract_issue_details(check_results, df=df)
    sev_counts = get_issue_severity_counts(issue_details)
    score = report["overall_score"]
    status = report["quality_status"]
    score_interp = get_score_interpretation(score, status)

    # 1. Prominent Data Quality Score Visual
    score_card_col1, score_card_col2 = st.columns([1, 2])
    with score_card_col1:
        st.metric(
            "Overall Quality Score",
            f"{score:.2f} / 100.0",
            delta=f"{status} Health",
            delta_color="normal" if score >= 75 else "inverse",
        )
        st.progress(min(1.0, max(0.0, score / 100.0)))
    with score_card_col2:
        st.markdown(f"**Health Assessment:** `{status.upper()}`")
        st.markdown(f"*{score_interp}*")
        sub_c1, sub_c2 = st.columns(2)
        sub_c1.caption(f"📊 **Dataset Size:** {report['total_rows']:,} rows inspected")
        sub_c2.caption(f"📉 **Total Deductions:** -{report['total_penalty']:.2f} pts")

    # 2. Severity Overview
    st.subheader("Severity Overview")
    sev_col1, sev_col2, sev_col3, sev_col4 = st.columns(4)
    sev_col1.metric("🔴 High Severity", sev_counts.get("HIGH", 0))
    sev_col2.metric("🟠 Medium Severity", sev_counts.get("MEDIUM", 0))
    sev_col3.metric("🟡 Low Severity", sev_counts.get("LOW", 0))
    sev_col4.metric("🟢 Passed Checks", sev_counts.get("PASS", 0))

    # Clean dataset state distinction
    if sev_counts.get("TOTAL_ISSUES", 0) == 0:
        st.success("✓ **Excellent Data Quality**: No significant data quality issues were detected across any of the 10 checks.")
    elif sev_counts.get("HIGH", 0) == 0 and sev_counts.get("MEDIUM", 0) == 0:
        st.info("ℹ️ **Dataset is Generally Healthy**: All critical high and medium severity checks passed. Minor observations noted below.")

    # 3. Issue Distribution & Priority Areas
    col_dist, col_prio = st.columns([1, 1])

    with col_dist:
        st.markdown("### Issue Distribution")
        fig_dist = create_issue_distribution_chart(sev_counts)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    with col_prio:
        st.markdown("### Priority Areas")
        priority_areas = get_priority_areas(issue_details)
        if not priority_areas:
            st.success("✓ **No Priority Actions Required**\n\nAll 10 checks passed with zero quality issues detected.")
        else:
            st.caption("Ordered by urgency: High > Medium > Low, prioritized by issue volume.")
            for idx, prio in enumerate(priority_areas, 1):
                badge = "🔴" if prio.severity == "HIGH" else ("🟠" if prio.severity == "MEDIUM" else "🟡")
                col_tag = f" ({', '.join(prio.affected_columns)})" if prio.affected_columns else ""
                st.markdown(
                    f"**{idx}. {badge} {prio.check_name} — {prio.severity}**  \n"
                    f"`{prio.issue_count}` issue{'s' if prio.issue_count != 1 else ''}{col_tag} · *{prio.recommendation}*"
                )

    # 4. Quality Check Overview Table
    st.markdown("### Quality Check Overview")
    overview_df = build_quality_check_overview_df(issue_details)
    st.dataframe(
        overview_df,
        width='stretch',
        hide_index=True,
    )

    # 5. Detailed Issue Explorer (Phase 12A Preserved)
    st.markdown("### 🔍 Data Quality Issue Explorer")
    st.caption("Investigate specific columns, inspect affected records, and review sample problematic values.")
    filter_choice = st.radio(
        "Filter Issues by Severity:",
        ["All", "High", "Medium", "Low", "Passed"],
        horizontal=True,
        key="dq_issue_filter",
    )

    if filter_choice == "High":
        displayed_issues = [iss for iss in issue_details if iss.severity == "HIGH"]
    elif filter_choice == "Medium":
        displayed_issues = [iss for iss in issue_details if iss.severity == "MEDIUM"]
    elif filter_choice == "Low":
        displayed_issues = [iss for iss in issue_details if iss.severity == "LOW"]
    elif filter_choice == "Passed":
        displayed_issues = [iss for iss in issue_details if iss.severity == "PASS"]
    else:
        displayed_issues = issue_details

    severity_icons = {
        "HIGH": "🔴",
        "MEDIUM": "🟠",
        "LOW": "🟡",
        "PASS": "🟢",
    }

    if not displayed_issues:
        st.caption(f"No checks matching filter '{filter_choice}'.")
    else:
        for issue in displayed_issues:
            icon = severity_icons.get(issue.severity, "⚪")
            if issue.status == "PASS":
                header_title = f"{icon} {issue.check_name} — Passed"
            else:
                cols_text = f" ({', '.join(issue.affected_columns)})" if issue.affected_columns else ""
                header_title = f"{icon} {issue.check_name} — {issue.issue_count} issue{'s' if issue.issue_count != 1 else ''}{cols_text}"

            with st.expander(header_title, expanded=(issue.severity == "HIGH")):
                st.markdown(f"**What this means:**\n\n{issue.message}")

                if issue.affected_columns:
                    st.markdown(f"**Affected columns:** `{', '.join(issue.affected_columns)}`")
                else:
                    st.markdown("**Affected columns:** *None*")

                if issue.status != "PASS":
                    st.markdown(f"**Affected rows:** `{issue.affected_row_count}`")

                    if issue.sample_values:
                        formatted_samples = ", ".join(f"`{v}`" for v in issue.sample_values)
                        st.markdown(f"**Example values:** {formatted_samples}")

                    if not issue.sample_records.empty:
                        st.markdown("**Sample affected records (up to 5 rows):**")
                        st.dataframe(
                            issue.sample_records,
                            column_config=build_result_column_config(issue.sample_records),
                            width='stretch',
                            hide_index=True,
                        )

                st.markdown(f"**Recommended action:**\n\n{issue.recommendation}")

    with st.expander("View 10 Quality Checks Overview & Penalty Table", expanded=False):
        st.subheader("Summary of 10 Quality Checks")
        summary_rows = []
        for c in report["check_details"]:
            penalty_display = f"-{c['penalty']:.2f} pts" if c["penalty"] > 0 else "0.00 pts"
            summary_rows.append({
                "Check Name": c["check_name"],
                "Status": f"[{c['status']}]",
                "Issue Count": c["issue_count"],
                "Severity": c["severity"],
                "Penalty": penalty_display,
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, width='stretch', hide_index=True)

        st.subheader("Actionable Recommendations")
        if report["recommendations"]:
            for idx, rec in enumerate(report["recommendations"], 1):
                category = rec["category"]
                check_name = rec["check"]
                severity = rec["severity"]
                count = rec["issue_count"]
                msg = rec["recommendation"]
                card_text = f"**{idx}. [{category}] {check_name}** ({severity} Severity, {count} issues)\n\n-> {msg}"
                if category == "ERROR":
                    st.error(card_text)
                else:
                    st.warning(card_text)
        else:
            st.success("All quality checks passed! No corrective actions required for this dataset.")

    # 6. Data Quality Report Export (Phase 12C)
    st.markdown("### 📥 Export Data Quality Report")
    st.caption("Download a standalone, executive-ready HTML data quality report containing the score, severity breakdown, 10-check overview, priority actions, and issue details.")

    dq_report_bytes = generate_quality_report_bytes(
        dataset_name=dataset_name,
        score_report=report,
        issue_details=issue_details,
        severity_counts=sev_counts,
        priority_areas=priority_areas,
    )
    dq_report_filename = build_quality_report_filename(dataset_name)

    st.download_button(
        label="📥 Download Data Quality Report (HTML)",
        data=dq_report_bytes,
        file_name=dq_report_filename,
        mime="text/html",
        key="btn_download_dq_report",
        use_container_width=False,
    )

    # =========================================================================
    # 4. AI ANALYTICS WORKSPACE (Requirements 3, 4, 5, 6)
    # =========================================================================
    st.divider()
    st.header("AI Analytics")

    # Analytical Database Selection (Phase 13)
    configured_default = get_configured_database_backend()
    if "selected_database_backend" not in st.session_state:
        st.session_state["selected_database_backend"] = "PostgreSQL" if configured_default == "postgres" else "DuckDB"

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader("Ask a Business Question")
        st.caption("Ask questions about your uploaded dataset in plain English.")
    with header_col2:
        selected_db_display = st.selectbox(
            "Analytical Database",
            options=["DuckDB", "PostgreSQL"],
            index=0 if st.session_state["selected_database_backend"] == "DuckDB" else 1,
            key="db_backend_selectbox",
            help="Select the analytical SQL execution engine (DuckDB in-memory or PostgreSQL relational).",
        )

    # Invalidate cached analytics result when switching databases (Requirement 8)
    if selected_db_display != st.session_state["selected_database_backend"]:
        st.session_state["selected_database_backend"] = selected_db_display
        st.session_state.pop("last_analytics_result", None)

    backend_key = "postgres" if selected_db_display == "PostgreSQL" else "duckdb"

    # Verify availability if PostgreSQL is selected
    if backend_key == "postgres":
        is_avail, avail_reason = is_backend_available("postgres")
        if not is_avail:
            safe_reason = re.sub(r"password=([^\s]+)", "password=***", avail_reason, flags=re.IGNORECASE)
            warn_col, switch_col = st.columns([3, 1])
            with warn_col:
                st.warning(
                    f"⚠️ **PostgreSQL is currently unreachable or unconfigured:** {safe_reason}. "
                    "Queries will automatically run on DuckDB (In-Memory). "
                    "You can configure PostgreSQL connection parameters in Streamlit Secrets or switch back to DuckDB."
                )
            with switch_col:
                if st.button("🔄 Switch to DuckDB", key="btn_switch_to_duckdb", use_container_width=True):
                    st.session_state["selected_database_backend"] = "DuckDB"
                    st.session_state["db_backend_selectbox"] = "DuckDB"
                    st.session_state.pop("last_analytics_result", None)
                    st.rerun()
            backend_key = "duckdb"
            backend_display_name = "DuckDB (Fallback)"
        else:
            pg_host = os.getenv("POSTGRES_HOST", "localhost")
            pg_port = os.getenv("POSTGRES_PORT", "5432")
            backend_display_name = f"PostgreSQL ({pg_host}:{pg_port})"
    else:
        backend_display_name = "DuckDB (In-Memory)"

    conv_history = st.session_state["conversation_history"]
    num_turns = len(conv_history)

    # Gemini configuration advisory
    if not is_gemini_configured():
        st.info(
            "ℹ️ **Gemini API is not configured.** Natural language query generation, SQL explanation, and AI business insights are disabled. "
            "Add `GEMINI_API_KEY` to your Streamlit secrets or `.env` file to enable AI analytics. (Data Quality features remain fully functional)."
        )

    # Conversation and Engine status summary
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.caption(f"📁 **Active Dataset:** `{dataset_name}`")
    with stat_col2:
        st.caption(f"⚡ **Engine:** `{backend_display_name}`")
    with stat_col3:
        if num_turns == 0:
            st.caption("💬 **Conversation Status:** *New (0 questions recorded)*")
        else:
            st.caption(f"💬 **Conversation Status:** `{num_turns}` question{'s' if num_turns != 1 else ''} recorded")

    # Suggested Questions (Dataset-Aware, Deterministic)
    st.markdown("**Suggested Questions:**")
    suggested_questions = generate_suggested_questions(schema=df)
    if len(suggested_questions) <= 4:
        example_cols = st.columns(len(suggested_questions))
        for i, ex_q in enumerate(suggested_questions):
            if example_cols[i].button(ex_q, key=f"ex_btn_{i}", use_container_width=True):
                st.session_state["user_nl_question"] = ex_q
                st.rerun()
    else:
        row1_cols = st.columns(3)
        for i, ex_q in enumerate(suggested_questions[:3]):
            if row1_cols[i].button(ex_q, key=f"ex_btn_{i}", use_container_width=True):
                st.session_state["user_nl_question"] = ex_q
                st.rerun()
        row2_cols = st.columns(len(suggested_questions) - 3)
        for j, ex_q in enumerate(suggested_questions[3:]):
            idx = 3 + j
            if row2_cols[j].button(ex_q, key=f"ex_btn_{idx}", use_container_width=True):
                st.session_state["user_nl_question"] = ex_q
                st.rerun()

    # Follow-up Suggestions if conversation turns exist
    if not conv_history.is_empty():
        st.markdown("**Follow-up Suggestions:**")
        follow_up_examples = [
            "What about East?",
            "Compare East and South",
            "Show top 3 instead",
            "Break down by product",
        ]
        fu_cols = st.columns(len(follow_up_examples))
        for i, fu_q in enumerate(follow_up_examples):
            if fu_cols[i].button(fu_q, key=f"fu_btn_{i}", use_container_width=True):
                st.session_state["user_nl_question"] = fu_q
                st.rerun()

    if "user_nl_question" not in st.session_state:
        st.session_state["user_nl_question"] = ""

    user_question = st.text_area(
        "Ask a business question about your data:",
        key="user_nl_question",
        placeholder="e.g. What is the total revenue by region? (Or follow up with: What about East?)",
        help=f"Type a natural-language analytical question. Gemini will generate a read-only SQL query executed by {backend_display_name}.",
    )

    analyze_clicked = st.button("🚀 Analyze Question", type="primary", use_container_width=True)

    # =========================================================================
    # 5. EXECUTION & CURRENT ANALYSIS (Requirements 7, 8, 9, 10, 11)
    # =========================================================================
    if analyze_clicked:
        if not is_gemini_configured():
            st.error(
                "Cannot analyze question: Gemini API key is not configured. "
                "Please set GEMINI_API_KEY in Streamlit Secrets or your .env file to enable natural language query generation."
            )
        else:
            cleaned_question = user_question.strip() if user_question else ""
            if not cleaned_question:
                st.warning("Please enter a business question to analyze.")
            else:
                if len(conv_history) > 0:
                    used_turns = min(len(conv_history), 3)
                    context_badge = f"Used previous analytical context ({used_turns} recent turn{'s' if used_turns != 1 else ''})"
                    st.info(f"Using previous analytical context ({used_turns} recent turn{'s' if used_turns != 1 else ''})")
                else:
                    context_badge = "New analytical question (no previous context)"
                    st.info("Starting a new analytical question (no previous context)")

                try:
                    with create_database_engine(backend_key) as engine:
                        load_result = load_dataset_into_database(engine, df, table_name="sales")
                        schema = load_result.schema

                        with st.spinner("Generating SQL query with Gemini..."):
                            generated_sql = generate_sql(
                                cleaned_question,
                                schema=schema,
                                conversation_context=conv_history,
                                recent_turn_limit=3,
                            )

                        validate_read_only_query(generated_sql)

                        with st.spinner(f"Executing query in {backend_display_name}..."):
                            result_df = engine.execute_query(generated_sql)

                        # Query resource protection: enforce max query result rows limit
                        max_allowed_rows = get_max_query_result_rows()
                        original_row_count = len(result_df)
                        is_truncated = False
                        if original_row_count > max_allowed_rows:
                            result_df = result_df.iloc[:max_allowed_rows].copy()
                            is_truncated = True

                        # Generate business-friendly SQL explanation (Phase 11D)
                        sql_explanation_text = None
                        try:
                            with st.spinner("Explaining query logic..."):
                                sql_explanation_text = generate_sql_explanation(
                                    question=cleaned_question,
                                    sql=generated_sql,
                                    schema=schema,
                                )
                        except Exception:
                            sql_explanation_text = None

                        # Store insights
                        insights_text = None
                        if not result_df.empty:
                            try:
                                with st.spinner("Generating AI business insights..."):
                                    insights_text = generate_business_insights(
                                        question=cleaned_question,
                                        result_df=result_df,
                                        sql=generated_sql,
                                    )
                            except EmptyResultError:
                                insights_text = None
                            except Exception:
                                insights_text = None

                        # Append completed turn to conversation history
                        conv_history.add_turn(
                            question=cleaned_question,
                            sql=generated_sql,
                            result_df=result_df,
                            insights=insights_text,
                            sql_explanation=sql_explanation_text,
                        )

                        # Cache latest analytical result
                        st.session_state["last_analytics_result"] = {
                            "question": cleaned_question,
                            "sql": generated_sql,
                            "result_df": result_df,
                            "insights": insights_text,
                            "sql_explanation": sql_explanation_text,
                            "context_badge": context_badge,
                            "backend": backend_display_name,
                            "is_truncated": is_truncated,
                            "original_row_count": original_row_count,
                        }

                except EmptyQuestionError:
                    st.warning("Please enter a business question to analyze.")
                except LLMQuotaExceededError:
                    st.error("AI service quota reached. Please try again later.")
                except LLMAuthenticationError:
                    st.error("AI service authentication failed. Please check the Gemini API configuration.")
                except LLMServiceUnavailableError:
                    st.error("Gemini is temporarily busy. Please try again in a few moments.")
                except QuestionInterpretationError:
                    st.error("I couldn't understand that business question. Try asking about revenue, sales, products, regions, orders, or dates.")
                except LLMGenerationError as e:
                    err_lower = str(e).lower()
                    if "429" in err_lower or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
                        st.error("AI service quota reached. Please try again later.")
                    elif "api key" in err_lower or "api_key" in err_lower or "auth" in err_lower or "401" in err_lower or "403" in err_lower or "not configured" in err_lower:
                        st.error("AI service authentication failed. Please check the Gemini API configuration.")
                    elif "unavailable" in err_lower or "timeout" in err_lower or "timed out" in err_lower or "503" in err_lower or "502" in err_lower or "504" in err_lower or "connection" in err_lower or "network" in err_lower:
                        st.error("Gemini is temporarily busy. Please try again in a few moments.")
                    else:
                        st.error("I couldn't understand that business question. Try asking about revenue, sales, products, regions, orders, or dates.")
                except DisallowedQueryError:
                    st.error("Security Alert: The generated query was blocked because only safe, read-only analytical queries are permitted.")
                except QueryExecutionError:
                    st.error(f"The generated query could not be executed against the current dataset ({backend_display_name}). Please try a simpler analytical question.")
                except PostgresConnectionError:
                    st.error("Unable to connect to PostgreSQL. Please check that PostgreSQL is running and configured correctly in your Streamlit secrets or .env file, or switch to DuckDB.")
                except Exception:
                    st.error("An unexpected error occurred while processing your request. Please check your question or try again.")

    # Render Current Analysis (from fresh execution or session state)
    if "last_analytics_result" in st.session_state and st.session_state["last_analytics_result"] is not None:
        last = st.session_state["last_analytics_result"]
        res_df = last["result_df"]

        st.divider()
        st.header("Current Analysis")

        # Truncation warning if row limit was exceeded
        if last.get("is_truncated"):
            st.warning(
                f"⚠️ Result contains {last['original_row_count']:,} rows. "
                f"Truncated to first {len(res_df):,} rows for memory protection and display performance."
            )


        # Analyzed Question Callout (Requirement 7)
        badge_str = f" • *{last['context_badge']}*" if last.get("context_badge") else ""
        engine_str = f" • **Engine:** `{last['backend']}`" if last.get("backend") else ""
        st.info(f"**Analyzed Question:** {last['question']}{badge_str}{engine_str}")

        # SQL Display inside expandable section (Requirement 8)
        sql_title = f"View Generated SQL ({last.get('backend', 'SQL')})"
        with st.expander(sql_title, expanded=False):
            st.code(last["sql"], language="sql")

        # Business-friendly SQL Explanation (Phase 11D)
        with st.expander("🔍 How the AI Answered This", expanded=True):
            if last.get("sql_explanation"):
                st.markdown(last["sql_explanation"])
            else:
                st.info("SQL explanation could not be generated, but the analysis result is still available.")

        # Query Results & Smart Presentation (Phase 11B)
        if res_df.empty:
            st.subheader("Query Results")
            st.info("Query executed successfully, but returned 0 rows.")
        else:
            presentation = build_kpi_metadata(res_df)

            # Prominent KPI Cards for Scalar and Multi-KPI results
            if presentation.classification in ("kpi", "multi_kpi"):
                if presentation.classification == "kpi":
                    st.subheader("Key Metric")
                    card = presentation.kpi_cards[0]
                    st.metric(label=card.label, value=card.formatted_value)
                else:
                    st.subheader("Key Performance Indicators")
                    kpi_cols = st.columns(len(presentation.kpi_cards))
                    for idx, card in enumerate(presentation.kpi_cards):
                        kpi_cols[idx].metric(label=card.label, value=card.formatted_value)

            # Query Results Table (Preserves full DataFrame and Phase 11A formatting)
            st.subheader("Query Results")
            st.caption(
                f"Showing {len(res_df):,} row{'s' if len(res_df) != 1 else ''} × {len(res_df.columns)} column{'s' if len(res_df.columns) != 1 else ''}"
            )
            st.dataframe(
                res_df,
                column_config=build_result_column_config(res_df),
                width='stretch',
                hide_index=True,
            )

            # Data Visualization (Plotly chart for chartable/grouped results)
            chart_fig = create_chart(res_df)
            if chart_fig is not None:
                st.subheader("Data Visualization")
                st.plotly_chart(chart_fig, width='stretch')

            # AI Business Insights (Requirement 11)
            st.subheader("AI Business Insights")
            if last.get("insights"):
                st.markdown(last["insights"])
            else:
                st.warning("Business insights could not be generated, but the query result is still available.")

            # Download Analysis (Phase 11C)
            st.subheader("Download Analysis")
            safe_ds_prefix = sanitize_filename(dataset_name, default="")
            csv_filename = build_download_filename("analysis_result", "csv", prefix=safe_ds_prefix)
            excel_filename = build_download_filename("analysis_result", "xlsx", prefix=safe_ds_prefix)
            chart_filename = build_download_filename("chart", "html", prefix=safe_ds_prefix)

            has_chart = chart_fig is not None
            dl_col_count = 3 if has_chart else 2
            dl_cols = st.columns(dl_col_count)

            # CSV Download
            try:
                csv_data = dataframe_to_csv_bytes(res_df)
                dl_cols[0].download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_csv",
                )
            except Exception:
                dl_cols[0].warning("CSV export could not be generated.")

            # Excel Download
            try:
                excel_data = dataframe_to_excel_bytes(res_df)
                dl_cols[1].download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="btn_download_excel",
                )
            except Exception:
                dl_cols[1].warning("Excel export could not be generated.")

            # Chart HTML Download (only when Plotly figure exists)
            if has_chart:
                try:
                    html_data = figure_to_html_bytes(chart_fig)
                    dl_cols[2].download_button(
                        label="📈 Download Chart",
                        data=html_data,
                        file_name=chart_filename,
                        mime="text/html",
                        use_container_width=True,
                        key="btn_download_chart",
                    )
                except Exception:
                    dl_cols[2].warning("Chart HTML export could not be generated.")

    elif df is not None:
        # Empty State (Requirement 13)
        st.divider()
        st.info("💡 **Ask your first business question** above or click one of the suggested questions to get started!")

    # =========================================================================
    # 6. CONVERSATION HISTORY (Requirement 12)
    # =========================================================================
    if not conv_history.is_empty():
        st.divider()
        st.header("Conversation History")
        st.caption("Chronological record of analytical questions and findings from this session.")

        turns = conv_history.get_turns()
        total_turns = len(turns)
        header_label = f"Saved Queries ({total_turns} question{'s' if total_turns != 1 else ''})"
        if total_turns > 3:
            header_label += " — latest 3 turns used for AI context"

        with st.expander(header_label, expanded=False):
            for idx, turn in enumerate(turns, 1):
                st.markdown(f"**Question {idx}:** {turn.question}")
                if turn.sql:
                    with st.expander(f"View Generated SQL (Question {idx})", expanded=False):
                        st.code(turn.sql, language="sql")
                if getattr(turn, "sql_explanation", None):
                    with st.expander(f"How the AI Answered (Question {idx})", expanded=False):
                        st.markdown(turn.sql_explanation)
                if turn.insights:
                    st.caption(f"💡 {turn.insights}")
                if idx < total_turns:
                    st.divider()

            st.write("")
            if st.button("🗑️ Clear Conversation History", key="btn_clear_history"):
                conv_history.clear()
                st.session_state.pop("last_analytics_result", None)
                st.session_state["user_nl_question"] = ""
                st.success("Conversation history and latest analysis cleared.")
                st.rerun()


if __name__ == "__main__":
    main()
