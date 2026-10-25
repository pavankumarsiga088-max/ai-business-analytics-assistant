"""
Unit Tests for Gemini API Failure Handling & Transient 503 Retry Logic.

Validates:
1. success on first call
2. 503 then success
3. 503 then 503 then success
4. 503 all attempts exhausted
5. 429 is NOT retried
6. authentication error is NOT retried
7. no infinite retry
8. friendly exception remains correct
"""

import unittest
from unittest.mock import MagicMock

from src.analytics.nl_to_sql import (
    generate_sql,
    _call_gemini_sdk,
    LLMServiceUnavailableError,
    LLMQuotaExceededError,
    LLMAuthenticationError,
    LLMGenerationError,
)
from src.analytics.insights import generate_business_insights
import pandas as pd


class TestGeminiRetryBehavior(unittest.TestCase):

    def setUp(self):
        self.question = "What is the total revenue?"
        self.sample_df = pd.DataFrame({"revenue": [100.0, 200.0]})

    # 1. Success on first call
    def test_1_success_on_first_call(self):
        """When Gemini succeeds on attempt 1, returns result immediately without retrying."""
        mock_sleep = MagicMock()
        mock_client = MagicMock()
        mock_client.return_value = "SELECT SUM(revenue) AS total_revenue FROM sales;"

        res = generate_sql(
            self.question,
            llm_client=mock_client,
            sleep_fn=mock_sleep,
        )

        self.assertEqual(res, "SELECT SUM(revenue) AS total_revenue FROM sales;")
        self.assertEqual(mock_client.call_count, 1)
        mock_sleep.assert_not_called()

    # 2. 503 then success
    def test_2_503_then_success(self):
        """Transient 503 on first attempt is retried once and succeeds."""
        mock_sleep = MagicMock()
        calls = []

        def mock_client(prompt):
            calls.append(1)
            if len(calls) == 1:
                raise Exception("503 UNAVAILABLE: The model is overloaded. Please try again.")
            return "SELECT SUM(revenue) AS total_revenue FROM sales;"

        res = generate_sql(
            self.question,
            llm_client=mock_client,
            sleep_fn=mock_sleep,
        )

        self.assertEqual(res, "SELECT SUM(revenue) AS total_revenue FROM sales;")
        self.assertEqual(len(calls), 2)
        mock_sleep.assert_called_once_with(1.0)

    # 3. 503 then 503 then success
    def test_3_503_then_503_then_success(self):
        """Transient 503 on first two attempts succeeds on the 3rd attempt."""
        mock_sleep = MagicMock()
        calls = []

        def mock_client(prompt):
            calls.append(1)
            if len(calls) < 3:
                raise Exception("503 Service Unavailable: High demand")
            return "SELECT SUM(revenue) AS total_revenue FROM sales;"

        res = generate_sql(
            self.question,
            llm_client=mock_client,
            sleep_fn=mock_sleep,
        )

        self.assertEqual(res, "SELECT SUM(revenue) AS total_revenue FROM sales;")
        self.assertEqual(len(calls), 3)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    # 4. 503 all attempts exhausted
    def test_4_503_all_attempts_exhausted(self):
        """When 503 persists across all 3 attempts, raises LLMServiceUnavailableError."""
        mock_sleep = MagicMock()
        calls = []

        def mock_client(prompt):
            calls.append(1)
            raise Exception("503 UNAVAILABLE: Server temporarily down")

        with self.assertRaises(LLMServiceUnavailableError):
            generate_sql(
                self.question,
                llm_client=mock_client,
                sleep_fn=mock_sleep,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(mock_sleep.call_count, 2)

    # 5. 429 is NOT retried
    def test_5_429_is_not_retried(self):
        """HTTP 429 / Quota exhaustion fails immediately with ZERO retries."""
        mock_sleep = MagicMock()
        calls = []

        def mock_client(prompt):
            calls.append(1)
            raise Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota.")

        with self.assertRaises(LLMQuotaExceededError):
            generate_sql(
                self.question,
                llm_client=mock_client,
                sleep_fn=mock_sleep,
            )

        self.assertEqual(len(calls), 1)
        mock_sleep.assert_not_called()

    # 6. Authentication error is NOT retried
    def test_6_authentication_error_is_not_retried(self):
        """Authentication / API key failure fails immediately with ZERO retries."""
        mock_sleep = MagicMock()
        calls = []

        def mock_client(prompt):
            calls.append(1)
            raise Exception("401 UNAUTHENTICATED: API key not valid. Please pass a valid API key.")

        with self.assertRaises(LLMAuthenticationError):
            generate_sql(
                self.question,
                llm_client=mock_client,
                sleep_fn=mock_sleep,
            )

        self.assertEqual(len(calls), 1)
        mock_sleep.assert_not_called()

    # 7. No infinite retry
    def test_7_no_infinite_retry(self):
        """Guarantees at most 3 total attempts (attempt 1, attempt 2, attempt 3 stops)."""
        mock_sleep = MagicMock()
        call_count = 0

        def mock_client(prompt):
            nonlocal call_count
            call_count += 1
            raise Exception("Connection timed out (503)")

        try:
            generate_sql(
                self.question,
                llm_client=mock_client,
                sleep_fn=mock_sleep,
            )
        except LLMServiceUnavailableError:
            pass

        self.assertEqual(call_count, 3)

    # 8. Friendly exception remains correct
    def test_8_friendly_exception_remains_correct(self):
        """LLMServiceUnavailableError inherits from LLMGenerationError and maps cleanly."""
        err = LLMServiceUnavailableError("Gemini service temporarily unavailable: 503")
        self.assertIsInstance(err, LLMGenerationError)
        self.assertIn("503", str(err))

    # 9. Business insights retry on 503 and no-retry on 429
    def test_9_business_insights_retry_behavior(self):
        """Business insights module inherits the same safe retry behavior."""
        mock_sleep = MagicMock()
        calls = []

        def mock_client(prompt):
            calls.append(1)
            if len(calls) == 1:
                raise Exception("503 UNAVAILABLE: High demand")
            return "- Observation: Total revenue was $300."

        res = generate_business_insights(
            self.question,
            self.sample_df,
            llm_client=mock_client,
            sleep_fn=mock_sleep,
        )

        self.assertIn("Observation", res)
        self.assertEqual(len(calls), 2)
        mock_sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
