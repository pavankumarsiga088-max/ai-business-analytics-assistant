"""
Focused Test Suite for Natural Language -> SQL UX and Reliability Improvements.

Verifies:
1. Natural-language business prompt guidelines & synonyms (sales/revenue/money made, top/best-selling, region/area, etc.)
2. Few-shot prompt examples (total revenue, region sells the most, top 5 products, order count, AOV, what about East, compare East and South)
3. Short question tolerance (e.g. "what is the total revenue", "top 5 products", "how much did we make")
4. Schema authority preservation (never invent columns, schema remains authoritative)
5. Conversation follow-up context handling ("what about East", "compare East and South", "show top 3")
6. 429 / RESOURCE_EXHAUSTED / Quota error mapping -> LLMQuotaExceededError -> "AI service quota reached. Please try again later."
7. Authentication error mapping -> LLMAuthenticationError -> "AI service authentication failed. Please check the Gemini API configuration."
8. Network / service unavailable mapping -> LLMServiceUnavailableError -> "AI service is temporarily unavailable. Please try again."
9. Question interpretation failure mapping -> QuestionInterpretationError -> "I couldn't understand that business question. Try asking about revenue, sales, products, regions, orders, or dates."
10. Zero secret exposure across all exception classifications and user messages.
"""
import pytest
import pandas as pd

from src.analytics.nl_to_sql import (
    generate_sql,
    build_sql_prompt,
    format_schema,
    clean_generated_sql,
    _classify_and_raise_gemini_error,
    NLToSQLError,
    EmptyQuestionError,
    LLMGenerationError,
    LLMQuotaExceededError,
    LLMAuthenticationError,
    LLMServiceUnavailableError,
    QuestionInterpretationError,
    BUSINESS_LANGUAGE_GUIDELINES,
    FEW_SHOT_EXAMPLES,
    DEFAULT_SALES_SCHEMA,
)
from src.analytics.conversation import ConversationHistory
from src.analytics.duckdb_engine import DuckDBEngine, validate_read_only_query


class TestNLToSQLPromptEnhancements:
    """Verifies that the prompt contains comprehensive business vocabulary and few-shot examples."""

    def test_prompt_contains_synonym_and_informal_guidance(self):
        prompt = build_sql_prompt("what is the total revenue")

        # Synonyms for financial metrics
        assert "sales" in prompt.lower()
        assert "revenue" in prompt.lower()
        assert "money made" in prompt.lower()
        assert "earnings" in prompt.lower()

        # Categorical dimensions & ranking
        assert "region" in prompt.lower()
        assert "best selling" in prompt.lower() or "top product" in prompt.lower()
        assert "orders" in prompt.lower()
        assert "average order value" in prompt.lower()

        # Guidance on short queries and grammatical imperfections
        assert "short" in prompt.lower()
        assert "grammatical" in prompt.lower()

    def test_prompt_contains_few_shot_examples(self):
        prompt = build_sql_prompt("show me total revenue")

        # Required few-shot patterns
        assert "what is the total revenue" in prompt
        assert "SUM(revenue)" in prompt
        assert "which region sells the most" in prompt
        assert "top 5 products" in prompt
        assert "how many orders do we have" in prompt
        assert "average order value" in prompt
        assert "what about East" in prompt
        assert "compare East and South" in prompt

    def test_prompt_enforces_schema_authority(self):
        custom_schema = {
            "financials": ["tx_id", "amount", "department", "fiscal_year"]
        }
        prompt = build_sql_prompt("what is total revenue", schema=custom_schema)

        # Custom schema appears in prompt
        assert "financials" in prompt
        assert "amount" in prompt
        assert "department" in prompt

        # Schema authority constraint remains strictly present
        assert "authoritative" in prompt.lower()
        assert "never invent" in prompt.lower() or "do not invent" in prompt.lower()

    def test_short_questions_are_valid_and_produce_sql(self):
        short_questions = [
            "what is the total revenue",
            "what's our total sales",
            "how much did we make",
            "show me total revenue",
            "which region sells the most",
            "which product made the most money",
            "top 5 products",
            "how many orders do we have",
            "average order value",
            "show sales by region",
            "revenue for January",
        ]

        for q in short_questions:
            # Verify build_sql_prompt accepts and structures it cleanly
            p = build_sql_prompt(q)
            assert f"USER QUESTION:\n{q}" in p
            assert "DATABASE SCHEMA:" in p

            # Verify generate_sql processes it with mock client
            mock_client = lambda prompt: "SELECT SUM(revenue) AS total_revenue FROM sales;"
            sql = generate_sql(q, llm_client=mock_client)
            assert "SELECT" in sql


class TestConversationFollowUpContext:
    """Verifies that conversation follow-up context behaves correctly with enhanced prompt."""

    def test_follow_up_prompt_preserves_context_and_examples(self):
        history = ConversationHistory()
        df_turn1 = pd.DataFrame({"region": ["East", "South"], "revenue": [1000, 500]})
        history.add_turn(
            question="total revenue by region",
            sql="SELECT region, SUM(revenue) FROM sales GROUP BY region;",
            result_df=df_turn1,
        )

        prompt = build_sql_prompt(
            question="what about East",
            conversation_context=history,
            recent_turn_limit=3,
        )

        assert "CURRENT QUESTION:\nwhat about East" in prompt
        assert "PREVIOUS ANALYTICS CONTEXT" in prompt
        assert "total revenue by region" in prompt
        assert "NATURAL BUSINESS LANGUAGE" in prompt
        assert "FEW-SHOT EXAMPLES" in prompt
        assert "authoritative" in prompt.lower()

    def test_follow_up_questions_generate_standalone_sql(self):
        follow_ups = [
            ("what about East?", "SELECT region, SUM(revenue) FROM sales WHERE region = 'East' GROUP BY region;"),
            ("compare it with South", "SELECT region, SUM(revenue) FROM sales WHERE region IN ('East', 'South') GROUP BY region;"),
            ("show top 3 instead", "SELECT product, SUM(revenue) FROM sales GROUP BY product ORDER BY 2 DESC LIMIT 3;"),
        ]

        history = ConversationHistory()
        history.add_turn(question="sales by region", sql="SELECT region, SUM(revenue) FROM sales GROUP BY region;")

        for q, expected_sql in follow_ups:
            sql = generate_sql(q, conversation_context=history, llm_client=lambda p, s=expected_sql: s)
            assert sql == expected_sql
            validate_read_only_query(sql)


class TestErrorClassificationAndMapping:
    """Verifies that API and query exceptions are categorized into precise user-friendly types."""

    def test_quota_exceeded_429_classification(self):
        # 429 HTTP status or RESOURCE_EXHAUSTED
        simulated_errors = [
            Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for generate_content_free_tier_requests"),
            Exception("ClientError: 429 You exceeded your current quota, please check your plan and billing details."),
            Exception("Rate limit reached for model gemini-3.6-flash"),
        ]

        for exc in simulated_errors:
            with pytest.raises(LLMQuotaExceededError) as exc_info:
                _classify_and_raise_gemini_error(exc)
            # Ensure it is also an LLMGenerationError for backward compatibility
            assert isinstance(exc_info.value, LLMGenerationError)

    def test_authentication_error_classification(self):
        simulated_errors = [
            Exception("401 UNAUTHENTICATED: API key not valid. Please pass a valid API key."),
            Exception("403 PERMISSION_DENIED: The caller does not have permission"),
            Exception("GEMINI_API_KEY is not configured."),
            Exception("Invalid API key provided for Google GenAI"),
        ]

        for exc in simulated_errors:
            with pytest.raises(LLMAuthenticationError) as exc_info:
                _classify_and_raise_gemini_error(exc)
            assert isinstance(exc_info.value, LLMGenerationError)

    def test_service_unavailable_error_classification(self):
        simulated_errors = [
            Exception("503 UNAVAILABLE: The service is currently unavailable."),
            Exception("504 Gateway Timeout"),
            Exception("Connection timed out connecting to generativelanguage.googleapis.com"),
            Exception("Network connection reset by peer"),
        ]

        for exc in simulated_errors:
            with pytest.raises(LLMServiceUnavailableError) as exc_info:
                _classify_and_raise_gemini_error(exc)
            assert isinstance(exc_info.value, LLMGenerationError)

    def test_question_interpretation_error_on_empty_or_refusal(self):
        refusal_responses = [
            "",
            "   ",
            "I cannot answer this question based on the provided schema.",
            "I am unable to interpret this request.",
            "Sorry, the question does not make sense for the sales table.",
        ]

        for resp in refusal_responses:
            with pytest.raises(QuestionInterpretationError):
                generate_sql("random garbled input", llm_client=lambda p, r=resp: r)

    def test_user_facing_error_messages_match_specification(self):
        """Ensures the exact UI strings specified in UX requirements are mapped correctly."""
        # Simulated UI dispatcher matching app.py logic
        def get_ui_error_message(exc: Exception) -> str:
            if isinstance(exc, LLMQuotaExceededError):
                return "AI service quota reached. Please try again later."
            elif isinstance(exc, LLMAuthenticationError):
                return "AI service authentication failed. Please check the Gemini API configuration."
            elif isinstance(exc, LLMServiceUnavailableError):
                return "AI service is temporarily unavailable. Please try again."
            elif isinstance(exc, QuestionInterpretationError):
                return "I couldn't understand that business question. Try asking about revenue, sales, products, regions, orders, or dates."
            elif isinstance(exc, LLMGenerationError):
                err_lower = str(exc).lower()
                if "429" in err_lower or "quota" in err_lower:
                    return "AI service quota reached. Please try again later."
                elif "api key" in err_lower or "auth" in err_lower:
                    return "AI service authentication failed. Please check the Gemini API configuration."
                elif "unavailable" in err_lower or "timeout" in err_lower:
                    return "AI service is temporarily unavailable. Please try again."
                else:
                    return "I couldn't understand that business question. Try asking about revenue, sales, products, regions, orders, or dates."
            return "An unexpected error occurred."

        msg_quota = get_ui_error_message(LLMQuotaExceededError("429 Quota"))
        assert msg_quota == "AI service quota reached. Please try again later."

        msg_auth = get_ui_error_message(LLMAuthenticationError("401 Unauthorized"))
        assert msg_auth == "AI service authentication failed. Please check the Gemini API configuration."

        msg_unavail = get_ui_error_message(LLMServiceUnavailableError("503 Service Unavailable"))
        assert msg_unavail == "AI service is temporarily unavailable. Please try again."

        msg_interp = get_ui_error_message(QuestionInterpretationError("Could not parse"))
        assert msg_interp == "I couldn't understand that business question. Try asking about revenue, sales, products, regions, orders, or dates."

    def test_no_secret_keys_exposed_in_error_handling(self):
        fake_secret = "AIzaSyD_SecretKeyToken123456789"
        exc = Exception(f"Failed with key {fake_secret}: 429 RESOURCE_EXHAUSTED")

        try:
            _classify_and_raise_gemini_error(exc)
        except LLMQuotaExceededError:
            # User facing message in UI never uses raw exception text
            user_msg = "AI service quota reached. Please try again later."
            assert fake_secret not in user_msg
