import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.rag import workflow
from langchain_core.messages import AIMessage
from app.rag.openrouter_errors import (
    classify_openrouter_failure,
    http_retry_after,
    invoke_with_transient_retries,
)


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeOpenRouterError(Exception):
    def __init__(self, status_code, body, headers=None):
        super().__init__("sensitive prompt and authorization must never be logged")
        self.status_code = status_code
        self.response = FakeResponse(status_code, body, headers)


class FakeRunnable:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class OpenRouterErrorTests(unittest.TestCase):
    def test_greeting_routes_direct_but_answer_is_generated_by_configured_llm(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, _prompt):
                self.calls += 1
                return AIMessage(content="Model-generated greeting.")

        fake_llm = FakeLLM()
        with patch.object(workflow, "llm", return_value=fake_llm):
            result = workflow.ask("Hello!", graph=workflow.build_graph())
        self.assertEqual(result["source_used"], "direct")
        self.assertEqual(result["answer"], "Model-generated greeting.")
        self.assertEqual(fake_llm.calls, 1)

    @staticmethod
    def _limit(message, *, metadata=None, retry_after="17"):
        return FakeOpenRouterError(
            429,
            {"error": {"message": message, "metadata": metadata or {}}},
            {"retry-after": retry_after},
        )

    def test_account_free_model_limit_is_classified_without_raw_text(self):
        error = self._limit("Rate limit exceeded: free-models-per-day")
        failure = classify_openrouter_failure(error)
        self.assertEqual(failure.category, "openrouter_account_or_model_limit")
        self.assertEqual(failure.retry_after, "17")

    def test_provider_limit_is_classified_from_allowlisted_metadata(self):
        error = self._limit(
            "Provider returned error",
            metadata={"provider_name": "Safe Provider 1", "raw": "do not log this"},
        )
        failure = classify_openrouter_failure(error)
        self.assertEqual(failure.category, "upstream_provider_rate_limit")
        self.assertEqual(failure.provider, "SafeProvider1")

    def test_account_limit_is_not_retried_even_with_retry_after(self):
        error = self._limit("Daily quota exceeded")
        runnable = FakeRunnable([error])
        with (
            patch("app.rag.openrouter_errors.sleep") as sleep,
            self.assertLogs("app.rag.openrouter_errors", level="WARNING") as captured,
        ):
            with self.assertRaises(FakeOpenRouterError):
                invoke_with_transient_retries(runnable, "private prompt")
        self.assertEqual(runnable.calls, 1)
        sleep.assert_not_called()
        self.assertIn("openrouter_account_or_model_limit", "\n".join(captured.output))
        self.assertNotIn("sensitive prompt", "\n".join(captured.output))

    def test_provider_429_respects_retry_after_and_retries_once(self):
        error = self._limit(
            "Provider returned error", metadata={"provider_name": "Provider A"}, retry_after="2"
        )
        runnable = FakeRunnable([error, "ok"])
        with patch("app.rag.openrouter_errors.sleep") as sleep:
            self.assertEqual(invoke_with_transient_retries(runnable, "private prompt"), "ok")
        self.assertEqual(runnable.calls, 2)
        sleep.assert_called_once_with(2.0)

    def test_retry_after_above_bound_is_not_retried(self):
        error = FakeOpenRouterError(503, {}, {"Retry-After": "60"})
        runnable = FakeRunnable([error])
        with patch("app.rag.openrouter_errors.sleep") as sleep:
            with self.assertRaises(FakeOpenRouterError):
                invoke_with_transient_retries(runnable, "private prompt")
        self.assertEqual(runnable.calls, 1)
        sleep.assert_not_called()

    def test_transient_server_error_uses_bounded_backoff_once(self):
        error = FakeOpenRouterError(503, {})
        runnable = FakeRunnable([error, "ok"])
        with patch("app.rag.openrouter_errors.sleep") as sleep:
            self.assertEqual(invoke_with_transient_retries(runnable, "private prompt"), "ok")
        self.assertEqual(runnable.calls, 2)
        sleep.assert_called_once_with(0.5)

    def test_client_preserves_configured_model_and_disables_sdk_retries(self):
        captured = {}

        def make_client(**kwargs):
            captured.update(kwargs)
            return object()

        configured = SimpleNamespace(
            openrouter_api_key="unit-test-key",
            openrouter_model="configured-model",
            openrouter_base_url="https://openrouter.example/v1",
        )
        with (
            patch.object(workflow, "_llm", None),
            patch.object(workflow, "settings", configured),
            patch("app.rag.workflow.ChatOpenAI", side_effect=make_client),
        ):
            workflow.llm()
        self.assertEqual(captured["model"], "configured-model")
        self.assertEqual(captured["base_url"], "https://openrouter.example/v1")
        self.assertEqual(captured["max_retries"], 0)

    def test_http_retry_after_rejects_untrusted_header_text(self):
        error = FakeOpenRouterError(429, {}, {"Retry-After": "secret-token"})
        self.assertIsNone(http_retry_after(error))


if __name__ == "__main__":
    unittest.main()
