"""
Application Configuration Module.

Centralizes non-secret environment and application settings with sensible defaults.
Secrets (API keys, database passwords) are strictly retrieved via os.getenv()
and are never hardcoded or stored in static configuration objects.
"""
import os
from typing import Optional

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass


# =====================================================================
# DEFAULT CONFIGURATION VALUES
# =====================================================================

DEFAULT_DATABASE_BACKEND: str = os.getenv("DATABASE_BACKEND", "duckdb").strip().lower()
DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

# Query Resource Limits & Protection
MAX_QUERY_RESULT_ROWS: int = int(os.getenv("MAX_QUERY_RESULT_ROWS", "10000"))
MAX_DISPLAYED_ROWS: int = int(os.getenv("MAX_DISPLAYED_ROWS", "1000"))
QUERY_TIMEOUT_SECONDS: int = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))

# Conversational Analytics Limits
CONVERSATION_MAX_TURNS: int = int(os.getenv("CONVERSATION_MAX_TURNS", "10"))
CONVERSATION_RECENT_TURN_LIMIT: int = int(os.getenv("CONVERSATION_RECENT_TURN_LIMIT", "3"))
INSIGHTS_MAX_ROWS: int = int(os.getenv("INSIGHTS_MAX_ROWS", "25"))

# PostgreSQL Non-Secret Connection Defaults
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost").strip()
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "postgres").strip()
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres").strip()
POSTGRES_CONNECT_TIMEOUT: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))


# =====================================================================
# DYNAMIC GETTERS (Allowing runtime environment overrides in tests)
# =====================================================================

def get_database_backend() -> str:
    """Returns the configured database backend ('duckdb' or 'postgres')."""
    backend = os.getenv("DATABASE_BACKEND", DEFAULT_DATABASE_BACKEND).strip().lower()
    if backend in ("postgres", "postgresql"):
        return "postgres"
    return "duckdb"


def get_gemini_model() -> str:
    """Returns the configured Gemini model identifier."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()


def get_max_query_result_rows() -> int:
    """Returns the configured maximum query result rows limit."""
    try:
        return int(os.getenv("MAX_QUERY_RESULT_ROWS", str(MAX_QUERY_RESULT_ROWS)))
    except (ValueError, TypeError):
        return 10000


def get_query_timeout_seconds() -> int:
    """Returns the configured query timeout in seconds."""
    try:
        return int(os.getenv("QUERY_TIMEOUT_SECONDS", str(QUERY_TIMEOUT_SECONDS)))
    except (ValueError, TypeError):
        return 30


def is_gemini_configured() -> bool:
    """
    Returns True if GEMINI_API_KEY is present, non-empty, and not a placeholder.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        return False
    # Check for placeholder strings
    if "your_gemini_api_key" in key.lower():
        return False
    return True


def is_postgres_configured() -> bool:
    """
    Returns True if PostgreSQL host and credentials appear configured in environment.
    """
    host = os.getenv("POSTGRES_HOST")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    return bool(host and db and user)
