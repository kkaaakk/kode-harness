"""
subagent.py - Lightweight subagent spawning for isolated exploration or work.

Supports two agent types:
- ``Explore``: read-only (bash + read_file)
- ``general-purpose``: read/write (+ write_file + edit_file)

Depends on: config (client, MODEL), base_tools
"""

from agents.config import client, MODEL
from agents.base_tools import run_bash, run_read, run_write, run_edit
from agents.providers import AnthropicAdapter, ModelRequest, StopReason

# Phase 3A-1: the ONLY model-call path for the subagent loop. Resolves
# ``client`` lazily from this module's globals so tests that swap
# run_subagent.__globals__["client"] keep working unchanged. D0/D1
# runtime boundary untouched: own 30-round loop, own hardcoded tool set,
# max_tokens=8000, does NOT call agent_loop.
_SUBAGENT_ADAPTER = AnthropicAdapter(client_provider=lambda: client)


def run_subagent(prompt: str, agent_type: str = "Explore") -> str:
    sub_tools = [
        {
            "name": "bash",
            "description": "Run command.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": "Read file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ]
    if agent_type != "Explore":
        sub_tools += [
            {
                "name": "write_file",
                "description": "Write file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": "Edit file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        ]
    sub_handlers = {
        "bash": lambda **kw: run_bash(kw["command"]),
        "read_file": lambda **kw: run_read(kw["path"]),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }
    sub_msgs = [{"role": "user", "content": prompt}]
    resp = None
    for _ in range(30):
        resp = _SUBAGENT_ADAPTER.complete(ModelRequest(
            model=MODEL,
            messages=sub_msgs,
            tools=sub_tools,
            max_tokens=8000,
        ))
        sub_msgs.append({"role": "assistant", "content": resp.raw_response.content})
        if resp.stop_reason != StopReason.TOOL_CALL:
            break
        results = []
        for tc in resp.tool_calls:
            h = sub_handlers.get(tc.name, lambda **kw: "Unknown tool")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": str(h(**tc.arguments))[:50000],
                }
            )
        sub_msgs.append({"role": "user", "content": results})
    if resp:
        return resp.text or "(no summary)"
    return "(subagent failed)"
