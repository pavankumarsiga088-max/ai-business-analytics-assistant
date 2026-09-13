# AI Business Analytics & Data Quality Assistant

An interactive, portfolio-grade business intelligence application that enables users to upload tabular business datasets, automatically audit data-quality flaws, ask natural language questions, generate validated read-only SQL, visualize results with interactive charts, and receive AI-powered business takeaways across interchangeable database backends.

---

## 1. Overview
The **AI Business Analytics & Data Quality Assistant** is an end-to-end data analytics platform built with **Streamlit**, **DuckDB**, **PostgreSQL**, and **Google Gemini 2.5 Flash**. Designed to address real-world business intelligence challenges, it bridges data engineering (automated data profiling and quality scoring) with self-service exploratory analysis (natural language to SQL translation, automated charting, and contextual follow-ups).

---

## 2. Business Problem
In modern data teams, business analysts and data scientists spend up to **80% of their working hours** manually discovering, cleaning, and verifying raw transaction data before conducting analysis. When stakeholders attempt self-service reporting without automated data health checks, they frequently make strategic decisions on corrupted data (e.g. duplicate transactions, null revenues, broken date formats, or currency anomalies). Furthermore, non-technical stakeholders often depend heavily on data engineers to write repetitive SQL queries for straightforward questions.

---

## 3. Solution
This platform provides a unified, two-pillar solution:
1. **Automated Quality Audit First**: Evaluates incoming datasets against 10 deterministic data quality checks, computes a transparent 0–100 quality score, and generates executive HTML audit reports before querying begins.
2. **Safe, Natural Language Analytics Second**: Enables users to ask plain-English questions (*"What is total revenue by region?"*, *"What about East?"*). The system translates them into strictly validated read-only SQL, executes against DuckDB or PostgreSQL, renders KPI cards and Plotly charts, and explains the query logic in plain English.

---

## 4. Key Features
- **10 Deterministic Data Quality Checks**: Missing values, duplicate rows, duplicate order IDs, whitespace anomalies, inconsistent casing, invalid date formats, non-numeric field values, negative/zero quantities, IQR-based statistical outliers, and cross-column revenue consistency.
- **Weighted Quality Scoring (0–100)**: Grade classifications (A: Excellent to F: Critical) based on error rates and business impact.
- **Interactive Issue Explorer**: Drill-down inspection filtered by severity (Critical, Warning, Info) with sample records and remediation advice.
- **Standalone HTML Report Export**: One-click downloadable executive data quality reports with responsive CSS styling.
- **Dual-Engine Relational Analytics**: Polymorphic execution across in-memory **DuckDB** (for instant zero-dependency local speed) and **PostgreSQL** (for enterprise relational storage).
- **Gemini 2.5 Flash NL-to-SQL**: Generates accurate analytical SQL grounded strictly in introspected database schemas.
- **Defense-in-Depth AST Validator**: Strictly restricts execution to read-only queries (`SELECT`, `WITH`, `EXPLAIN`) and blocks all mutating DDL/DML verbs and stacked injections.
- **Transparent Analytics Presentation**: Auto-classifies queries into single/multi-metric KPI cards, interactive Plotly visualizations, and tabular views.
- **Plain-English Query Explanation**: Translates SQL joins, groupings, and filters into accessible business summaries.
- **Multi-Turn Conversational Memory**: Resolves pronouns and follow-up queries using rolling context with strict dataset isolation.
- **Multi-Format Downloads**: One-click exports for CSV, Excel (.xlsx), and interactive Plotly HTML figures.

---

## 5. Architecture
```
                                  ┌────────────────────────┐
                                  │   Streamlit Frontend   │
                                  │        (app.py)        │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
        ┌───────────────────────────┐                   ┌───────────────────────────┐
        │    Data Quality Engine    │                   │   Unified Dataset Loader  │
        │  (src/data_quality/...)   │                   │ (src/analytics/dataset...)│
        │ - 10 Deterministic Checks │                   └─────────────┬─────────────┘
        │ - Weighted Quality Score  │                                 │
        │ - Issue Explorer          │                                 ▼
        │ - HTML Report Generator   │                   ┌───────────────────────────┐
        └───────────────────────────┘                   │    AnalyticsDatabase      │
                                                        │       Abstraction         │
                                                        └──────┬─────────────┬──────┘
                                                               │             │
                                            ┌──────────────────┴──┐       ┌──┴──────────────────┐
                                            ▼                     ▼       ▼                     ▼
                                 ┌────────────────────┐                 ┌────────────────────┐
                                 │    DuckDBEngine    │                 │   PostgresEngine   │
                                 │ (In-Memory Virtual)│                 │ (Enterprise RDBMS) │
                                 └─────────┬──────────┘                 └─────────┬──────────┘
                                           └──────────────────┬───────────────────┘
                                                              │ Standardized pd.DataFrame
                                                              ▼
                                                ┌───────────────────────────┐
                                                │   Gemini NL-to-SQL Engine │
                                                │ - Context-Aware Prompt    │
                                                │ - Read-Only AST Validator │
                                                └─────────────┬─────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ Analytics Presentation    │
                                                │ - Smart KPI Cards         │
                                                │ - Plotly Visualizations   │
                                                │ - Plain-English SQL Logic │
                                                │ - AI Business Insights    │
                                                │ - Multi-Turn History      │
                                                │ - Data & Chart Downloads  │
                                                └───────────────────────────┘
```

---

## 6. Tech Stack
| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit 1.40+ | Interactive web interface, session state, metric cards |
| **In-Memory SQL** | DuckDB 1.0+ | Fast, columnar execution over in-memory DataFrames |
| **Relational SQL** | PostgreSQL 18.6+ | Relational enterprise database via `psycopg` (v3) |
| **Generative AI** | Google GenAI SDK (`google-genai`) | Gemini 2.5 Flash for NL-to-SQL, insights, and explanations |
| **Data Processing** | Pandas, OpenPyXL | Ingestion, data profiling, type mapping, Excel export |
| **Visualizations** | Plotly Express | Dynamic bar, line, and grouped visualizations |
| **Configuration** | Python-Dotenv | Environment-based non-secret configuration |
| **Testing** | Pytest | 289 automated unit, integration, and security tests |

---

## 7. Data Quality Workflow
1. **Ingestion**: Supports CSV and Excel (`.xlsx`, `.xls`) files without in-place mutation.
2. **Profiling**: Executes 10 deterministic check algorithms across all columns.
3. **Scoring**: Calculates total deductions from a base of 100 points, classifying the score into an academic grade (A through F).
4. **Issue Exploration**: Displays issues grouped by severity (Critical, Warning, Info) with affected row counts and actionable remediation tips.
5. **Report Export**: Compiles an executive HTML quality report complete with embedded CSS and metric tables.

---

## 8. AI Analytics Workflow
1. **Schema Introspection**: The active database engine extracts column names and data types.
2. **Prompt Assembly**: The schema, rolling conversation history (last 3 turns), and user question are composed into an untrusted prompt.
3. **LLM Generation**: Gemini 2.5 Flash generates a raw SQL query.
4. **Cleaning & AST Gate**: Markdown code fences are stripped, and `validate_read_only_query()` blocks multi-statements and DDL/DML verbs.
5. **Execution**: The query runs against DuckDB or PostgreSQL, returning a standardized Pandas DataFrame capped at `MAX_QUERY_RESULT_ROWS = 10000`.
6. **Presentation**: Generates KPI cards, Plotly charts, plain-English SQL breakdowns, and AI business takeaways.

---

## 9. Database Architecture
The platform enforces the **Dependency Inversion Principle (DIP)**:
- **`AnalyticsDatabase` Interface**: Standardizes `connect()`, `close()`, `load_dataframe()`, `execute_query()`, and `get_schema()`.
- **`DuckDBEngine`**: High-performance embedded columnar database registering virtual views directly from DataFrames.
- **`PostgresEngine`**: Connects via `psycopg`, maps Pandas types to native PostgreSQL types, enforces session `statement_timeout = 30000;`, and sanitizes connection logs.
- **Engine Switching**: Switching engines clears cached query results while preserving conversational turns.

---

## 10. Security Model & Defense-in-Depth
The application enforces strict isolation between generative AI and database execution:
- **Zero Direct Execution Authority**: The LLM generates query strings only; it never has direct database access.
- **Deterministic AST Validator**: Queries must start with `SELECT`, `WITH`, or `EXPLAIN`. Semicolon-separated statement chaining is blocked.
- **Destructive Verbs Blocked**: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `GRANT`, `REVOKE`, `VACUUM`, `COPY`, `EXEC`, `PRAGMA` are unconditionally rejected.
- **Credential Masking**: Passwords and secrets are never printed in error logs, UI alerts, or serialized state.
- **Caller Immutability**: Uploaded DataFrames are deep-copied before relational registration; source data is never modified in-place.

---

## 11. Example Questions
Demonstrate the application using the pre-loaded clean dataset (`sample_sales.csv`):
- *"What is total revenue by region?"*
- *"Which product generated the highest revenue?"*
- *"What are the top 5 products by quantity sold?"*
- *"What is the average order value across categories?"*
- **Follow-up 1**: *"What about the East region?"*
- **Follow-up 2**: *"Break that down by product."*

---

## 12. Demo & Screenshot Guide
When preparing portfolio demonstrations or capturing documentation screenshots, capture these 7 key views:
1. **Data Quality Dashboard**: Executive quality score card (Score 98/100, Grade A) with radar/distribution chart.
2. **Dirty Dataset Issue Explorer**: Flaws identified on `sample_dirty_data.xlsx` (missing customer names, negative quantities, revenue inconsistencies).
3. **AI Analytics Query & Result**: Natural language question input with analyzed question badge and execution engine indicator.
4. **Interactive Plotly Visualization**: Dynamic bar chart showing revenue by region with dark-navy styling.
5. **Transparent SQL Explanation**: Expanded `"🔍 How the AI Answered This"` showing plain-English query breakdown.
6. **AI Business Insights**: Grounded, bulleted takeaways derived from pre-calculated metrics.
7. **PostgreSQL Database Selector**: Dropdown showing live engine metadata and seamless backend switching.

---

## 13. Local Setup Guide
```powershell
# 1. Clone repository and navigate to directory
cd DA_Assistant

# 2. Create and activate a Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 14. Environment Variables Reference
Copy `.env.example` to create your local `.env`:
```powershell
cp .env.example .env
```
Configure your settings:
```ini
# Gemini API Key (Required for AI features)
GEMINI_API_KEY=AIzaSyYourActualKeyHere
GEMINI_MODEL=gemini-2.5-flash

# Database Backend (duckdb or postgres)
DATABASE_BACKEND=duckdb

# PostgreSQL Configuration (Optional)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_CONNECT_TIMEOUT=10

# Resource Limits
MAX_QUERY_RESULT_ROWS=10000
QUERY_TIMEOUT_SECONDS=30
```

---

## 15. Running the App
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 16. Running Tests
The automated test suite includes **289 tests** covering data quality, database abstraction, AST security, conversation state, downloads, and production hardening.
```powershell
# Run full regression suite
pytest -q

# Run Phase 14 hardening suite
pytest test_phase_14_hardening.py -v

# Run final end-to-end verification script
python verify_final_project.py
```

---

## 17. Project Structure
```
DA_Assistant/
├── app.py                      # Streamlit interactive application
├── requirements.txt            # Production dependencies (including pytest)
├── .env.example                # Sample configuration template
├── .gitignore                  # Git ignore rules for secrets & caches
├── README.md                   # Comprehensive project documentation
├── INTERVIEW_GUIDE.md          # 24 technical interview Q&As & elevator pitches
├── RELEASE_CHECKLIST.md        # Pre-release verification checklist
├── walkthrough.md              # Historical phase-by-phase implementation log
│
├── .streamlit/
│   └── config.toml             # Streamlit server & UI theme tokens
│
├── src/
│   ├── config.py               # Centralized non-secret configuration
│   ├── analytics/
│   │   ├── database.py         # AnalyticsDatabase abstraction & factory
│   │   ├── dataset_loader.py   # Universal dataset loader & validator
│   │   ├── duckdb_engine.py    # DuckDB in-memory analytical engine
│   │   ├── postgres_engine.py  # PostgreSQL relational engine
│   │   ├── nl_to_sql.py        # Gemini NL-to-SQL translation & cleaning
│   │   ├── result_presentation.py # KPI classification & formatting
│   │   ├── visualization.py    # Automated Plotly chart generator
│   │   ├── insights.py         # Grounded AI business insights
│   │   ├── sql_explanation.py  # Plain-English SQL transparency module
│   │   ├── conversation.py     # Multi-turn conversational history
│   │   └── downloads.py        # CSV, Excel, and chart export utilities
│   │
│   └── data_quality/
│       ├── loader.py           # CSV and Excel dataset loading
│       ├── checks.py           # 10 deterministic quality check algorithms
│       ├── scoring.py          # Quality score calculation & grading
│       ├── issue_details.py    # Issue extraction & severity categorization
│       ├── dashboard.py        # Quality charts & metric overview
│       └── report_export.py    # Executive HTML report generator
│
├── sample_data/
│   ├── sample_sales.csv        # Clean e-commerce reference dataset
│   └── sample_dirty_data.xlsx  # Dirty dataset with intentional quality flaws
│
└── tests/                      # Automated test suite (289 tests)
```

---

## 18. Limitations
- **Single-Tenant Application**: Designed for single-analyst exploratory sessions; does not include user authentication or multi-tenant database isolation.
- **External Database Setup**: PostgreSQL execution requires a reachable PostgreSQL instance; if absent, queries automatically fall back to DuckDB.
- **Single-Table Focus**: Analytical queries currently target the primary uploaded table (`sales`); multi-table joins require joining within the dataset prior to upload.

---

## 19. Future Improvements
- **Semantic Layer Integration**: Incorporate a formal metrics layer (like Cube or dbt) for standardized KPI definitions.
- **Scheduled Quality Pipelines**: Run automated data quality audits on recurring batch schedules with alerting webhooks.
- **Cloud Lakehouse Federation**: Support querying remote Parquet/Iceberg tables directly on cloud object storage (S3/GCS) via DuckDB or BigQuery.

---

## 20. Interview Talking Points
- **Automated Data Health**: *"I engineered the application to address data cleanliness before analysis, computing an academic-grade quality score across 10 deterministic checks."*
- **Defense-in-Depth Security**: *"The LLM has zero database execution privileges. All generated SQL must pass an AST parser that strictly permits read-only verbs and blocks destructive DDL/DML mutations."*
- **Polymorphic Database Architecture**: *"By adhering to the Dependency Inversion Principle, the analytics layer operates interchangeably across in-memory DuckDB and relational PostgreSQL without changing any business logic."*
- **Grounded AI Insights**: *"Gemini acts purely as an analytical interpreter, not a calculator; 100% of mathematical computation is executed by the relational database."*
