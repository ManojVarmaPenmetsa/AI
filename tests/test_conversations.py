from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.core.config import get_settings
from app import main as app_main
from app.rag import workflow


class FakeRouter:
    def invoke(self, _prompt):
        return SimpleNamespace(route="direct")


class FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def with_structured_output(self, *_args, **_kwargs):
        return FakeRouter()

    def invoke(self, prompt):
        text = "\n".join(str(getattr(message, "content", message)) for message in prompt) if isinstance(prompt, list) else str(prompt)
        self.prompts.append(text)
        if "Rewrite the latest message" in text:
            return AIMessage(content="resolved standalone follow-up")
        return AIMessage(content="A stable mocked answer")


class FakeServiceError(Exception):
    def __init__(self, status_code, body, headers=None):
        super().__init__("api-key=must-not-be-exposed prompt=private")
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers or {},
            json=lambda: body,
        )


class ConversationPersistenceTests(unittest.TestCase):
    def test_chat_translates_rate_limit_to_safe_retryable_http_error(self):
        cases = (
            (
                {"error": {"message": "free-models-per-day quota exceeded"}},
                "openrouter_account_or_model_limit",
                "OpenRouter account or model",
            ),
            (
                {"error": {"message": "Provider returned error", "metadata": {"provider_name": "Provider A"}}},
                "upstream_provider_rate_limit",
                "AI provider",
            ),
        )
        for body, _category, expected_message in cases:
            with self.subTest(expected_message=expected_message), tempfile.TemporaryDirectory() as temp_dir:
                settings = get_settings().model_copy(update={
                    "conversation_db_path": str(Path(temp_dir) / "conversations.db"),
                    "audit_db_path": str(Path(temp_dir) / "audit.db"),
                })
                failure = FakeServiceError(429, body, {"Retry-After": "12"})
                with (
                    patch.object(app_main, "get_settings", return_value=settings),
                    patch.object(app_main, "init_db"),
                    patch("app.api.routes.ask", side_effect=failure),
                    patch("app.api.routes.write_audit"),
                ):
                    with TestClient(app_main.create_app()) as client:
                        response = client.post("/api/chat", json={"question": "Test question"})
                self.assertEqual(response.status_code, 429)
                self.assertIn(expected_message, response.json()["detail"])
                self.assertEqual(response.headers["retry-after"], "12")
                self.assertNotIn("api-key", response.text)
                self.assertNotIn("private", response.text)

    def test_chat_maps_upstream_server_error_to_service_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = get_settings().model_copy(update={
                "conversation_db_path": str(Path(temp_dir) / "conversations.db"),
                "audit_db_path": str(Path(temp_dir) / "audit.db"),
            })
            failure = FakeServiceError(502, {"error": {"message": "sensitive provider detail"}})
            with (
                patch.object(app_main, "get_settings", return_value=settings),
                patch.object(app_main, "init_db"),
                patch("app.api.routes.ask", side_effect=failure),
                patch("app.api.routes.write_audit"),
            ):
                with TestClient(app_main.create_app()) as client:
                    response = client.post("/api/chat", json={"question": "Test question"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("temporarily unavailable", response.json()["detail"])
            self.assertNotIn("sensitive provider detail", response.text)

    def test_conversation_resume_context_and_owner_isolation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "conversations.db")
            settings = get_settings().model_copy(
                update={
                    "conversation_db_path": database_path,
                    "audit_db_path": str(Path(temp_dir) / "audit.db"),
                }
            )
            fake_llm = FakeLLM()

            with (
                patch.object(app_main, "get_settings", return_value=settings),
                patch.object(app_main, "init_db"),
                patch.object(workflow, "llm", return_value=fake_llm),
                patch("app.api.routes.write_audit"),
            ):
                first_app = app_main.create_app()
                with TestClient(first_app) as client:
                    created = client.post("/api/conversations")
                    self.assertEqual(created.status_code, 201)
                    first_id = created.json()["conversation_id"]

                    response = client.post(
                        "/api/chat",
                        json={"question": "How do I reset my password?", "conversation_id": first_id},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["conversation_id"], first_id)

                    follow_up = client.post(
                        "/api/chat",
                        json={"question": "What if that doesn't work?", "conversation_id": first_id},
                    )
                    self.assertEqual(follow_up.status_code, 200, follow_up.text)
                    self.assertEqual(
                        follow_up.json()["rewritten_query"],
                        "resolved standalone follow-up",
                    )
                    self.assertIn("User: How do I reset my password?", fake_llm.prompts[-2])
                    self.assertIn("Assistant: A stable mocked answer", fake_llm.prompts[-2])

                    history = client.get(f"/api/conversations/{first_id}")
                    self.assertEqual(history.status_code, 200)
                    self.assertEqual(
                        [message["role"] for message in history.json()["messages"]],
                        ["user", "assistant", "user", "assistant"],
                    )
                    self.assertEqual(
                        history.json()["messages"][2]["content"],
                        "What if that doesn't work?",
                    )
                    owner_cookie = client.cookies.get("conversation_owner")

                    second_id = client.post("/api/conversations").json()["conversation_id"]
                    self.assertNotEqual(first_id, second_id)

                with TestClient(app_main.create_app()) as other_client:
                    other_client.cookies.set("conversation_owner", "different-owner")
                    denied = other_client.get(f"/api/conversations/{first_id}")
                    self.assertEqual(denied.status_code, 404)
                    other_client.cookies.set("conversation_owner", owner_cookie)
                    resumed = other_client.get(f"/api/conversations/{first_id}")
                    self.assertEqual(resumed.status_code, 200, resumed.text)
                    self.assertEqual(len(resumed.json()["messages"]), 4)


if __name__ == "__main__":
    unittest.main()
