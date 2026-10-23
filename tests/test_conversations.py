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
        self.prompts.append(prompt)
        if "Rewrite the latest user message" in prompt:
            return AIMessage(content="resolved standalone follow-up")
        return AIMessage(content="A stable mocked answer")


class ConversationPersistenceTests(unittest.TestCase):
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
