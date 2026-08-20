"""
server.py — FastAPI server that wraps the harness_core agent loop.

Start with:   uv run uvicorn frontend.server:app --reload --port 8765
Or:           python -m uvicorn frontend.server:app --reload --port 8765
"""

from __future__ import annotations

import asyncio
import io
import json
import queue
import re
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure the repo root is on sys.path so "agents" imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Import agent_loop (module-level globals initialise on first import)
# ---------------------------------------------------------------------------
from agents.harness_core import (  # noqa: E402
    agent_loop,
    WORKDIR,
    MODEL,
    TASK_MGR,
    BG,
    TEAM,
    BUS,
    TODO,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Agent Chat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store  (session_id → list of Anthropic message dicts)
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict[str, Any]]] = {}

# Phase 3D-1: per-session model selection (alias for the next agent run).
from agents.session import SessionState, handle_model_command  # noqa: E402
from agents.providers.model_spec import UnknownModelError  # noqa: E402

session_states: dict[str, SessionState] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # None → new session


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict[str, Any]]  # each: {tool, input, output}


class StateResponse(BaseModel):
    tasks: str
    teammates: str
    bg_tasks: str
    inbox: list[dict[str, Any]]
    todo: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from an Anthropic response.content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _parse_tool_calls(captured_output: str) -> list[dict[str, Any]]:
    """
    Parse the captured stdout from agent_loop into structured tool calls.

    agent_loop prints:
        > tool_name:
        <first 200 chars of output>
    """
    calls: list[dict[str, Any]] = []
    lines = captured_output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^>\s+(\S+):", line)
        if m:
            tool_name = m.group(1)
            # Collect output lines until next "> tool:" or end
            output_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if re.match(r"^>\s+\S+:", next_line):
                    break
                if next_line:
                    output_lines.append(next_line)
                i += 1
            calls.append({
                "tool": tool_name,
                "output": "\n".join(output_lines)[:500],
            })
        else:
            i += 1
    return calls


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "workspace": str(WORKDIR),
    }


@app.get("/api/state", response_model=StateResponse)
async def get_state():
    """Return current agent state: tasks, teammates, background tasks, inbox, todos."""
    return StateResponse(
        tasks=TASK_MGR.list_all(),
        teammates=TEAM.list_all(),
        bg_tasks=BG.check(),
        inbox=BUS.read_inbox("lead"),
        todo=TODO.render(),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # --- obtain or create session ---
    sid = req.session_id or str(uuid.uuid4())[:8]
    if sid not in sessions:
        sessions[sid] = []
    if sid not in session_states:
        session_states[sid] = SessionState(session_id=sid)

    messages = sessions[sid]
    session = session_states[sid]

    # --- handle slash commands ---
    raw = req.message.strip()
    if raw == "/tasks":
        return ChatResponse(
            session_id=sid,
            reply=TASK_MGR.list_all(),
            tool_calls=[],
        )
    if raw == "/team":
        return ChatResponse(
            session_id=sid,
            reply=TEAM.list_all(),
            tool_calls=[],
        )
    if raw == "/inbox":
        inbox = BUS.read_inbox("lead")
        return ChatResponse(
            session_id=sid,
            reply=json.dumps(inbox, indent=2) if inbox else "Inbox is empty.",
            tool_calls=[],
        )
    if raw.startswith("/model"):
        try:
            reply = handle_model_command(raw, session)
        except UnknownModelError as exc:
            reply = f"Unknown model: {exc.args[0] if exc.args else ''}"
        return ChatResponse(session_id=sid, reply=reply, tool_calls=[])

    # --- append user message ---
    messages.append({"role": "user", "content": req.message})

    # --- capture stdout from agent_loop (it prints tool results) ---
    captured = io.StringIO()

    try:
        with redirect_stdout(captured):
            agent_loop(messages, session=session)
    except Exception as exc:
        if messages and messages[-1].get("role") == "user":
            messages.pop()
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    # --- extract final reply from the last assistant message ---
    reply = "(no response)"
    if messages:
        last = messages[-1]
        if last.get("role") == "assistant":
            reply = _extract_text_from_content(last.get("content", ""))

    # --- parse tool calls with their outputs ---
    tool_calls = _parse_tool_calls(captured.getvalue())

    return ChatResponse(session_id=sid, reply=reply, tool_calls=tool_calls)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# SSE streaming chat endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream agent events via Server-Sent Events.

    Event types:
      - session:    {session_id}              — confirm session id
      - tool_call:  {name, input}             — agent is calling a tool
      - tool_result:{name, output}            — tool finished
      - text:       {text}                    — model text output block
      - status:     {message}                 — internal status (compact etc.)
      - error:      {message}                 — something went wrong
      - done:       {}                        — stream finished
    """
    sid = req.session_id or str(uuid.uuid4())[:8]
    if sid not in sessions:
        sessions[sid] = []
    if sid not in session_states:
        session_states[sid] = SessionState(session_id=sid)
    messages = sessions[sid]
    session = session_states[sid]

    # Handle slash commands synchronously
    raw = req.message.strip()
    if raw in ("/tasks", "/team", "/inbox", "/model") or raw.startswith("/model"):
        async def cmd_stream():
            if raw == "/tasks":
                reply = TASK_MGR.list_all()
            elif raw == "/team":
                reply = TEAM.list_all()
            elif raw.startswith("/model"):
                try:
                    reply = handle_model_command(raw, session)
                except UnknownModelError as exc:
                    reply = f"Unknown model: {exc.args[0] if exc.args else ''}"
            else:
                inbox_data = BUS.read_inbox("lead")
                reply = json.dumps(inbox_data, indent=2) if inbox_data else "Inbox is empty."
            yield f"event: session\ndata: {json.dumps({'session_id': sid})}\n\n"
            yield f"event: text\ndata: {json.dumps({'text': reply})}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(cmd_stream(), media_type="text/event-stream")

    messages.append({"role": "user", "content": req.message})
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            def cb(event):
                q.put_nowait(event)
            agent_loop(messages, event_callback=cb, session=session)
            q.put_nowait({"type": "done"})
        except Exception as exc:
            # Remove the user message on failure
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            q.put_nowait({"type": "error", "message": str(exc)})

    async def event_stream():
        loop = asyncio.get_running_loop()
        # Confirm session id first
        yield f"event: session\ndata: {json.dumps({'session_id': sid})}\n\n"
        while True:
            try:
                event = await loop.run_in_executor(None, q.get, True, 300)
            except Exception:
                yield f"event: error\ndata: {json.dumps({'message': 'timeout'})}\n\n"
                break
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Serve static frontend files
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
