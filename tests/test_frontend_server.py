"""Tests for frontend/server.py — FastAPI endpoints with TestClient.

Covers:
- /api/health endpoint
- /api/chat endpoint (slash commands, normal chat, error handling)
- /api/chat/stream SSE endpoint (data: format, session confirm, done event)
- /api/state endpoint
- /api/session/{id} DELETE endpoint
- Session management (create / reuse / delete)
- No API key / agent error → correct HTTP error

Requires: fastapi, httpx (TestClient), pydantic — all are runtime deps.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock agents.harness_core before importing server
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _mock_harness_core():
    """Patch agents.harness_core in sys.modules before frontend.server imports."""
    # Build a fake agents.harness_core module
    fake_core = types.ModuleType("agents.harness_core")
    fake_core.agent_loop = MagicMock(
        side_effect=lambda messages, **kw: _fake_agent_loop(messages, **kw)
    )
    fake_core.WORKDIR = Path("/tmp/test_workspace")
    fake_core.MODEL = "claude-3-5-sonnet-20241022"
    fake_core.TASK_MGR = MagicMock()
    fake_core.TASK_MGR.list_all.return_value = "No tasks."
    fake_core.BG = MagicMock()
    fake_core.BG.check.return_value = "No background tasks."
    fake_core.TEAM = MagicMock()
    fake_core.TEAM.list_all.return_value = "No teammates."
    fake_core.BUS = MagicMock()
    fake_core.BUS.read_inbox.return_value = []
    fake_core.TODO = MagicMock()
    fake_core.TODO.render.return_value = ""

    prev_core = sys.modules.get("agents.harness_core")
    sys.modules["agents.harness_core"] = fake_core

    # Ensure agents package exists as a namespace
    if "agents" not in sys.modules:
        fake_agents = types.ModuleType("agents")
        sys.modules["agents"] = fake_agents

    yield fake_core

    # Cleanup
    if prev_core is not None:
        sys.modules["agents.harness_core"] = prev_core
    elif "agents.harness_core" in sys.modules:
        del sys.modules["agents.harness_core"]


def _fake_agent_loop(messages, **kwargs):
    """Simulates agent_loop: appends an assistant message to messages list."""
    event_callback = kwargs.get("event_callback")
    if event_callback:
        event_callback({"type": "tool_call", "name": "bash", "input": {"command": "ls"}})
        event_callback({"type": "tool_result", "name": "bash", "output": "file1.txt"})
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from agent!"}],
    })


def _failing_agent_loop(messages, **kwargs):
    """Simulates agent_loop failure."""
    raise RuntimeError("API key missing or invalid")


# ---------------------------------------------------------------------------
# Import server AFTER mocks are set up
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app(_mock_harness_core):
    """Import and return the FastAPI app (after mocks are in place)."""
    # Remove any cached import
    for mod in list(sys.modules.keys()):
        if mod.startswith("frontend"):
            del sys.modules[mod]

    from frontend.server import app
    return app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


# ===================================================================
# /api/health
# ===================================================================


class TestHealth:
    def test_health_status_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "workspace" in data


# ===================================================================
# /api/state
# ===================================================================


class TestState:
    def test_state_endpoint(self, client):
        resp = client.get("/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "teammates" in data
        assert "bg_tasks" in data
        assert "inbox" in data
        assert "todo" in data


# ===================================================================
# /api/chat — normal and slash commands
# ===================================================================


class TestChat:
    def test_new_session_created(self, client):
        resp = client.post("/api/chat", json={"message": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["reply"] == "Hello from agent!"

    def test_existing_session_reused(self, client):
        # First call creates session
        resp1 = client.post("/api/chat", json={"message": "hi", "session_id": "sess1"})
        assert resp1.json()["session_id"] == "sess1"
        # Second call reuses it
        resp2 = client.post("/api/chat", json={"message": "again", "session_id": "sess1"})
        assert resp2.json()["session_id"] == "sess1"

    def test_slash_tasks(self, client, _mock_harness_core):
        resp = client.post("/api/chat", json={"message": "/tasks"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "No tasks."
        assert data["tool_calls"] == []

    def test_slash_team(self, client, _mock_harness_core):
        resp = client.post("/api/chat", json={"message": "/team"})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "No teammates."

    def test_slash_inbox_empty(self, client, _mock_harness_core):
        resp = client.post("/api/chat", json={"message": "/inbox"})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "Inbox is empty."

    def test_agent_error_returns_500(self, client, _mock_harness_core):
        from frontend import server
        original_loop = server.agent_loop
        server.agent_loop = _failing_agent_loop
        try:
            resp = client.post("/api/chat", json={"message": "fail please"})
            assert resp.status_code == 500
            assert "API key missing" in resp.json()["detail"]
        finally:
            server.agent_loop = original_loop


# ===================================================================
# /api/session/{id} DELETE
# ===================================================================


class TestDeleteSession:
    def test_delete_existing(self, client):
        # Create a session
        client.post("/api/chat", json={"message": "hi", "session_id": "del1"})
        # Delete it
        resp = client.delete("/api/session/del1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/session/does_not_exist")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


# ===================================================================
# /api/chat/stream — SSE
# ===================================================================


class TestChatStreamSSE:
    def test_sse_format(self, client, _mock_harness_core):
        """SSE stream must emit valid 'event: X\ndata: {...}\n\n' format."""
        import asyncio
        from httpx import Timeout

        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "hello", "session_id": "sse1"},
            timeout=10.0,
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            full_text = ""
            for chunk in resp.iter_text():
                full_text += chunk

        # Must start with session event
        assert "event: session" in full_text
        assert '"session_id"' in full_text
        assert "sse1" in full_text

        # Must contain done event
        assert "event: done" in full_text

    def test_sse_data_lines_are_valid_json(self, client, _mock_harness_core):
        """Each 'data:' line must contain valid JSON."""
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "test json data"},
            timeout=10.0,
        ) as resp:
            full_text = ""
            for chunk in resp.iter_text():
                full_text += chunk

        # Parse SSE events
        data_lines = [
            line[5:].strip()  # remove "data: "
            for line in full_text.split("\n")
            if line.startswith("data: ")
        ]
        assert len(data_lines) > 0

        for data_line in data_lines:
            parsed = json.loads(data_line)  # Should not raise
            assert isinstance(parsed, dict)

    def test_sse_slash_command(self, client, _mock_harness_core):
        """Slash commands in SSE mode should emit session → text → done."""
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "/tasks", "session_id": "sse_cmd"},
            timeout=10.0,
        ) as resp:
            full_text = ""
            for chunk in resp.iter_text():
                full_text += chunk

        assert "event: session" in full_text
        assert "event: text" in full_text
        assert "No tasks." in full_text
        assert "event: done" in full_text

    def test_sse_error_event_on_failure(self, client, _mock_harness_core):
        """Agent failure should emit an error event."""
        from frontend import server
        original_loop = server.agent_loop
        server.agent_loop = _failing_agent_loop
        try:
            with client.stream(
                "POST",
                "/api/chat/stream",
                json={"message": "fail", "session_id": "sse_err"},
                timeout=10.0,
            ) as resp:
                full_text = ""
                for chunk in resp.iter_text():
                    full_text += chunk

            assert "event: error" in full_text
            assert "event: session" in full_text
        finally:
            server.agent_loop = original_loop

    def test_sse_contains_tool_events(self, client, _mock_harness_core):
        """Agent emits tool_call and tool_result events via callback."""
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "run tools please", "session_id": "sse_tools"},
            timeout=10.0,
        ) as resp:
            full_text = ""
            for chunk in resp.iter_text():
                full_text += chunk

        # The fake agent_loop emits tool_call and tool_result events
        assert "tool_call" in full_text
        assert "tool_result" in full_text

    def test_sse_session_id_confirmed_first(self, client, _mock_harness_core):
        """First SSE event must be session confirmation."""
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "hi", "session_id": "first_event"},
            timeout=10.0,
        ) as resp:
            full_text = ""
            for chunk in resp.iter_text():
                full_text += chunk

        # First event line should be 'event: session'
        event_lines = [l for l in full_text.split("\n") if l.startswith("event: ")]
        assert len(event_lines) > 0
        assert event_lines[0] == "event: session"
