"""
Analytics module: DuckDB SQL engine and analytical utilities.
"""
from src.analytics.database import (
    AnalyticsDatabase,
    create_database_engine,
    get_configured_database_backend,
    is_backend_available,
    DatabaseEngineError,
    UnsupportedDatabaseError,
)
from src.analytics.dataset_loader import (
    load_dataset_into_database,
    DatasetLoadResult,
    DatasetLoaderError,
    InvalidDataFrameError,
)
from src.analytics.duckdb_engine import (
    DuckDBEngine,
    DuckDBEngineError,
    DisallowedQueryError,
    QueryExecutionError,
    validate_read_only_query,
    create_engine,
    execute_sql,
)
from src.analytics.postgres_engine import (
    PostgresEngine,
    PostgresEngineError,
    PostgresConnectionError,
    PostgresDisallowedQueryError,
    PostgresQueryExecutionError,
    create_postgres_engine,
    execute_postgres_sql,
)
from src.analytics.nl_to_sql import (
    generate_sql,
    build_sql_prompt,
    format_schema,
    format_conversation_context,
    clean_generated_sql,
    NLToSQLError,
    EmptyQuestionError,
    LLMGenerationError,
    DEFAULT_SALES_SCHEMA,
)
from src.analytics.visualization import (
    create_chart,
    generate_chart_title,
    is_date_column,
    is_numeric_measure_column,
    is_categorical_dimension_column,
)
from src.analytics.insights import (
    generate_business_insights,
    build_insights_prompt,
    format_dataframe_for_prompt,
    InsightsError,
    EmptyResultError,
    LLMInsightGenerationError,
)
from src.analytics.conversation import (
    ConversationTurn,
    ConversationHistory,
    QueryResultSummary,
)
from src.analytics.result_presentation import (
    KPICard,
    ResultPresentation,
    classify_result_shape,
    identify_kpi_columns,
    format_kpi_label,
    format_kpi_value,
    build_kpi_metadata,
)
from src.analytics.downloads import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    figure_to_html_bytes,
    sanitize_filename,
    build_download_filename,
)
from src.analytics.sql_explanation import (
    generate_sql_explanation,
    build_sql_explanation_prompt,
    SQLExplanationError,
    EmptySQLExplanationError,
    LLMExplanationGenerationError,
)

__all__ = [
    "AnalyticsDatabase",
    "create_database_engine",
    "get_configured_database_backend",
    "is_backend_available",
    "DatabaseEngineError",
    "UnsupportedDatabaseError",
    "load_dataset_into_database",
    "DatasetLoadResult",
    "DatasetLoaderError",
    "InvalidDataFrameError",
    "DuckDBEngine",
    "DuckDBEngineError",
    "DisallowedQueryError",
    "QueryExecutionError",
    "validate_read_only_query",
    "create_engine",
    "execute_sql",
    "PostgresEngine",
    "PostgresEngineError",
    "PostgresConnectionError",
    "PostgresDisallowedQueryError",
    "PostgresQueryExecutionError",
    "create_postgres_engine",
    "execute_postgres_sql",
    "generate_sql",
    "build_sql_prompt",
    "format_schema",
    "format_conversation_context",
    "clean_generated_sql",
    "NLToSQLError",
    "EmptyQuestionError",
    "LLMGenerationError",
    "DEFAULT_SALES_SCHEMA",
    "create_chart",
    "generate_chart_title",
    "is_date_column",
    "is_numeric_measure_column",
    "is_categorical_dimension_column",
    "generate_business_insights",
    "build_insights_prompt",
    "format_dataframe_for_prompt",
    "InsightsError",
    "EmptyResultError",
    "LLMInsightGenerationError",
    "ConversationTurn",
    "ConversationHistory",
    "QueryResultSummary",
    "KPICard",
    "ResultPresentation",
    "classify_result_shape",
    "identify_kpi_columns",
    "format_kpi_label",
    "format_kpi_value",
    "build_kpi_metadata",
    "dataframe_to_csv_bytes",
    "dataframe_to_excel_bytes",
    "figure_to_html_bytes",
    "sanitize_filename",
    "build_download_filename",
    "generate_sql_explanation",
    "build_sql_explanation_prompt",
    "SQLExplanationError",
    "EmptySQLExplanationError",
    "LLMExplanationGenerationError",
]




