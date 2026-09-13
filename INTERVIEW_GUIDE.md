# AI Business Analytics & Data Quality Assistant — Interview Preparation Guide

This document provides concise, technically rigorous answers for job interviews, technical screens, and portfolio demonstrations for Data Analyst, Analytics Engineer, and BI Developer roles.

---

## 1. Fast Project Explanations

### The 60-Second Elevator Pitch
> *"I built the AI Business Analytics & Data Quality Assistant to solve a classic operational bottleneck in business intelligence: analysts spend up to 80% of their time verifying data cleanliness before answering stakeholders' questions. This application combines automated data quality profiling with conversational AI analytics. 
> When a user uploads a business dataset, the system runs 10 deterministic data quality checks and calculates an executive quality score (0–100). Stakeholders can then ask natural language questions—like 'What is total revenue by region?' or 'What about East?'. The platform translates questions into strictly validated, read-only analytical SQL, displays KPI cards and Plotly charts, breaks down the query logic in plain English, and produces AI business takeaways. To ensure flexibility, queries execute across interchangeable database backends—DuckDB for instant in-memory analysis and PostgreSQL for enterprise relational processing."*

### The 2-Minute Technical Architecture Deep Dive
> *"Under the hood, the platform adheres to three key engineering principles: Separation of Concerns, Defense-in-Depth Security, and the Dependency Inversion Principle.
> First, on the Data Quality side, 10 modular, rule-based algorithms inspect missing values, duplicates, casing anomalies, date formats, numeric anomalies, and cross-column business rules (such as revenue consistency). Rather than guessing, the scoring engine calculates weighted deduction penalties based on error rates and returns an executive score and downloadable HTML report.
> Second, on the Analytics side, the frontend never accesses database drivers directly. An `AnalyticsDatabase` abstract base class and factory pattern unify connection lifecycle, schema discovery, and query execution across DuckDB and PostgreSQL. Datasets are loaded as deep copies to guarantee caller immutability.
> Third, for Generative AI, we use Google Gemini 2.5 Flash via the official `google-genai` SDK. Crucially, the LLM has zero database execution authority. It generates a SQL string from schema metadata and a rolling 3-turn conversational context. The generated SQL passes through an output cleaner and a deterministic AST safety validator that strictly permits read-only verbs (SELECT, WITH, EXPLAIN) while blocking multi-statement injections and destructive operations (DROP, DELETE, UPDATE, ALTER, etc.).
> Finally, the result DataFrame is inspected for KPI classification, rendered into interactive Plotly visualizations, interpreted for business takeaways, and cached with strict dataset and database boundaries to prevent state leakage."*

---

## 2. Core Technical Questions & Answers

### A. Project Overview
**Q: What is this project?**
**A:** It is an interactive business analytics and data quality auditing application built with Streamlit, DuckDB, PostgreSQL, and Google Gemini 2.5 Flash. It allows non-technical stakeholders to upload tabular datasets, identify dirty-data issues, ask questions in plain English, and receive verified SQL queries, KPI summaries, interactive charts, and business takeaways.

### B. Why did you build it?
**A:** Self-service BI tools either lack automated data quality guardrails (leading stakeholders to make decisions on corrupt data) or require users to know SQL. I built this to bridge the gap: verifying data health *before* analysis, and democratizing data exploration through natural language without compromising database security.

### C. What business problem does it solve?
**A:** It eliminates the manual cycle of dataset validation and repetitive ad-hoc querying. Business teams can audit transaction records, catch data anomalies (like duplicate orders, negative quantities, or revenue discrepancies), and answer operational questions in seconds rather than submitting backlog tickets to the data team.

### D. Why use SQL instead of having the LLM answer directly?
**A:** LLMs are linguistic models, not mathematical calculators. If an LLM attempts to aggregate 50,000 transaction rows directly in its context window, it will hallucinate figures, saturate token limits, and introduce non-deterministic errors. By using SQL, 100% of mathematical computation is offloaded to a proven relational engine. The LLM acts purely as a translator (NL to SQL) and qualitative interpreter.

### E. Why DuckDB?
**A:** DuckDB is an embedded, columnar analytical database (the "SQLite for analytics"). It executes vectorized analytical queries directly on in-memory Pandas DataFrames with zero external server dependencies, sub-millisecond query latency, and zero infrastructure cost. It provides an ideal out-of-the-box local experience.

### F. Why PostgreSQL?
**A:** PostgreSQL represents enterprise, production-grade relational storage. Integrating PostgreSQL demonstrates client-server database connectivity via `psycopg` (v3), connection parameter management, session timeouts, and the ability to route analytics workloads to an external relational database.

### G. Why create a database abstraction?
**A:** Applying the **Dependency Inversion Principle (DIP)** decouples the Streamlit frontend, prompt generation, visualization, and insights modules from concrete database drivers. The analytics layer depends solely on the `AnalyticsDatabase` interface (`load_dataframe`, `execute_query`, `get_schema`, `connect`, `close`). This allows adding new engines (e.g. Snowflake, BigQuery) without changing a single line of business logic.

### H. Why Gemini (Gemini 2.5 Flash)?
**A:** Gemini 2.5 Flash provides exceptional reasoning latency, high SQL syntax accuracy, a large context window, and native support via the modern official `google-genai` SDK. It generates concise analytical queries and plain-English explanations cost-effectively.

### I. How does Natural Language to SQL work?
**A:** 
1. The active engine introspects the table schema (`get_schema()`), mapping column names to data types.
2. If previous conversational turns exist, the most recent 3 turns are formatted as untrusted context.
3. A structured prompt instructs Gemini to output raw SQL adhering to the schema, with strict constraints prohibiting DDL/DML.
4. The raw response is stripped of Markdown code fences and backticks via `clean_generated_sql()`.

### J. How do you prevent dangerous AI-generated SQL?
**A:** Through a deterministic AST-based security gate: `validate_read_only_query()`.
- Strips comments and string literals so keywords cannot be disguised.
- Blocks multiple statements separated by semicolons (preventing stacked SQL injection).
- Requires the query to begin with an approved read-only verb (`SELECT`, `WITH`, `EXPLAIN`).
- Scans for disallowed mutating keywords anywhere in the statement: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `GRANT`, `REVOKE`, `VACUUM`, `COPY`, `EXEC`, `PRAGMA`.
If a disallowed keyword is detected, execution is immediately aborted and a security alert is shown.

### K. Why not let the LLM execute SQL directly?
**A:** Principle of Least Privilege. Giving an LLM direct execution access opens the system to prompt injection attacks, accidental table destruction, and data exfiltration. The application enforces a strict unidirectional pipeline: LLM generates text $\rightarrow$ Application cleans text $\rightarrow$ Security gate validates $\rightarrow$ Application executes query.

### L. How does Data Quality scoring work?
**A:** The engine starts with a perfect score of 100 points. Points are deducted based on severity and issue prevalence:
- **Completeness (Missing Values)**: Weighted deduction based on missing cell ratio.
- **Uniqueness (Duplicates / PK)**: High-penalty deductions for duplicate primary keys or repeated transactions.
- **Validity (Invalid Dates, Non-numeric, Quantities $\le 0$)**: Proportional deductions per invalid record.
- **Consistency (Revenue check, Casing, Whitespace)**: Deductions for business rule violations ($Revenue \ne Qty \times Price$).
Scores map to standard grades: A (90–100), B (80–89), C (70–79), D (60–69), and F (<60).

### M. What checks are performed?
**A:** Exactly 10 deterministic checks:
1. Missing values per column
2. Duplicate rows across all columns
3. Duplicate primary key (`order_id`)
4. Leading/trailing whitespace
5. Inconsistent casing in categorical columns
6. Invalid or unparseable date formats
7. Non-numeric values in numeric fields
8. Invalid (negative or zero) quantities
9. Statistical outliers using the Interquartile Range (IQR) method
10. Cross-column revenue consistency ($Revenue = Quantity \times UnitPrice$)

### N. Why are outliers classified as warnings rather than automatic errors?
**A:** In e-commerce and retail, high transaction values or bulk orders are often legitimate business events (e.g. B2B purchases or holiday spikes) rather than data entry corruptions. Labeling them as **Warnings** alerts the analyst to inspect potential data skew without artificially penalizing the dataset's quality score as heavily as broken primary keys or missing values.

### O. How do follow-up questions work?
**A:** Follow-up questions (e.g. *"What about East?"* or *"Break that down by product"*) are resolved by injecting the recent conversational history (up to 3 recent turns) into the prompt. The model sees the previous question, previous SQL query, and previous result summary, allowing it to correctly infer that "East" is a filter on the `region` column from the previous regional revenue query.

### P. How does conversation memory work?
**A:** `ConversationHistory` maintains a rolling window of recent turns (configurable, default 10). To prevent context window explosion, query results stored in memory are truncated to the first 25 rows (`QueryResultSummary`), while the actual live `DataFrame` displayed to the user retains 100% of rows.

### Q. How do you prevent cross-dataset context leakage?
**A:** `ConversationHistory.reset_for_dataset(dataset_name)` binds conversational state to the active dataset name. When a new file is uploaded or selected, the session state callback flushes all conversational history and clears cached query results (`last_analytics_result`).

### R. How do KPI cards get selected?
**A:** In `src/analytics/result_presentation.py`, `build_kpi_metadata()` inspects the shape and data types of the query result:
- **1 row $\times$ 1 numeric column**: Classified as `kpi` (single prominent metric card).
- **1 row $\times$ 2–4 numeric columns**: Classified as `multi_kpi` (side-by-side metric cards).
- **Multiple rows**: Classified as `table` or `chartable`.

### S. How are charts selected?
**A:** `create_chart()` in `src/analytics/visualization.py` inspects the structure:
- 1 categorical/date column + 1 numeric measure $\rightarrow$ Horizontal or vertical Bar chart.
- Date/datetime column + numeric measure $\rightarrow$ Line trend chart.
- Scalar (1 row $\times$ 1 column) $\rightarrow$ No chart (metric card is preferred).
All charts use a cohesive dark navy/slate theme matching the Streamlit palette.

### T. How do you handle Gemini API failure?
**A:** Graceful degradation. If `GEMINI_API_KEY` is missing or the service is unreachable:
1. The Data Quality dashboard, issue explorer, and report downloads remain 100% functional.
2. In the AI Analytics tab, an informational advisory clearly explains that AI features require an API key.
3. If an API call times out or fails mid-analysis, user-friendly error banners explain the issue without crashing the application or showing Python tracebacks.

### U. How are PostgreSQL credentials protected?
**A:** 
1. Credentials are never hardcoded; they are retrieved from environment variables (`.env`).
2. `.env` and `.streamlit/secrets.toml` are strictly ignored in `.gitignore`.
3. `PostgresEngine` implements `_sanitize_error_message()` to replace passwords with `******` in any error or exception message.
4. Passwords are never stored in session state or displayed in Streamlit widgets.

### V. What happens when PostgreSQL is unavailable?
**A:** When PostgreSQL is selected, `is_backend_available("postgres")` verifies socket connectivity. If unreachable, the UI displays an informational warning (`⚠️ PostgreSQL is currently unavailable...`) and automatically routes the query to DuckDB as a fallback.

### W. How do you protect against large queries or runaway results?
**A:** 
- **Row limit safeguard**: `MAX_QUERY_RESULT_ROWS` (default: 10,000). If a query returns 50,000 rows, the result DataFrame is safely truncated to 10,000 for display and memory protection, and an explicit warning banner notifies the user.
- **Query execution timeout**: `QUERY_TIMEOUT_SECONDS` (default: 30s). `PostgresEngine` sets session `statement_timeout = 30000;` to kill queries exceeding 30 seconds.

### X. What would you improve next?
**A:**
1. **Semantic Layer Integration**: Incorporate a formal metrics repository (like dbt semantic layer or Cube) so metric definitions (e.g. "active customer", "churn") are standardized outside LLM prompts.
2. **Scheduled Quality Audits**: Run Data Quality checks on automated batch schedules with Slack/email alerting for pipeline regressions.
3. **DuckDB Persistent Storage**: Support querying MotherDuck or remote Parquet/Iceberg lakehouses on Cloud Storage.
4. **Vector Search Schema RAG**: For databases with hundreds of tables, use embedding-based retrieval to select relevant tables dynamically rather than injecting the full schema into the prompt.
