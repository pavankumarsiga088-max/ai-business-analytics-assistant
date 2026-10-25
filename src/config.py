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
# SECURE CONFIGURATION & SECRETS RESOLUTION
# =====================================================================

def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieves a configuration setting or secret from environment variables or st.secrets.

    Resolution order:
      1. os.environ (if present, non-empty, and not a known placeholder string)
      2. streamlit.secrets (if Streamlit is running and st.secrets contains the key)
      3. default fallback value

    Gracefully handles environments where Streamlit is not running, secrets.toml does not exist,
    or st.secrets raises an exception. Never prints or exposes secret values.
    """
    # 1. Check OS environment variable
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        val_clean = env_val.strip()
        val_lower = val_clean.lower()
        # Avoid treating placeholder values as configured secrets
        if not ("your_gemini_api_key" in val_lower or "your_postgres_password" in val_lower):
            return val_clean

    # 2. Check Streamlit secrets (Streamlit Community Cloud or .streamlit/secrets.toml)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Check exact key first
            if key in st.secrets:
                val = st.secrets[key]
                if val is not None and str(val).strip():
                    return str(val).strip()
            # Also check case-insensitive match (e.g. gemini_api_key)
            key_lower = key.lower()
            if key_lower in st.secrets:
                val = st.secrets[key_lower]
                if val is not None and str(val).strip():
                    return str(val).strip()
    except Exception:
        # Gracefully handle StreamlitSecretNotFoundError, FileNotFoundError, etc.
        pass

    return default


def get_gemini_api_key() -> Optional[str]:
    """
    Safely retrieves the Gemini API key from environment variables or st.secrets.
    Returns None if unset, empty, or a placeholder. Never logs or prints the key.
    """
    key = get_secret_or_env("GEMINI_API_KEY")
    if not key or not key.strip() or "your_gemini_api_key" in key.lower():
        return None
    return key.strip()


# =====================================================================
# DYNAMIC GETTERS (Allowing runtime environment / secrets overrides)
# =====================================================================

def get_database_backend() -> str:
    """Returns the configured database backend ('duckdb' or 'postgres')."""
    backend = (get_secret_or_env("DATABASE_BACKEND", DEFAULT_DATABASE_BACKEND) or "duckdb").strip().lower()
    if backend in ("postgres", "postgresql"):
        return "postgres"
    return "duckdb"


def get_gemini_model() -> str:
    """Returns the configured Gemini model identifier."""
    model = get_secret_or_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return (model or DEFAULT_GEMINI_MODEL).strip()


def get_max_query_result_rows() -> int:
    """Returns the configured maximum query result rows limit."""
    try:
        val = get_secret_or_env("MAX_QUERY_RESULT_ROWS", str(MAX_QUERY_RESULT_ROWS))
        return int(val) if val is not None else 10000
    except (ValueError, TypeError):
        return 10000


def get_query_timeout_seconds() -> int:
    """Returns the configured query timeout in seconds."""
    try:
        val = get_secret_or_env("QUERY_TIMEOUT_SECONDS", str(QUERY_TIMEOUT_SECONDS))
        return int(val) if val is not None else 30
    except (ValueError, TypeError):
        return 30


def is_gemini_configured() -> bool:
    """
    Returns True if GEMINI_API_KEY is present, non-empty, and not a placeholder.
    Works seamlessly with both local .env files and Streamlit Community Cloud st.secrets.
    """
    return get_gemini_api_key() is not None


def is_postgres_configured() -> bool:
    """
    Returns True if PostgreSQL host and credentials appear configured in environment or secrets.
    """
    host = get_secret_or_env("POSTGRES_HOST")
    db = get_secret_or_env("POSTGRES_DB")
    user = get_secret_or_env("POSTGRES_USER")
    return bool(host and db and user)
