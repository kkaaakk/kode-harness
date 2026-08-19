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

# Phase 3A-1: legacy default adapter used when no ModelRuntimeContext is
# passed (keeps direct/standalone callers and fake-client tests working).
# Phase 3C-3B: when the Parent passes ``model_runtime``, the subagent
# creates its OWN adapter via ctx.create_adapter() - the model selection
# is inherited, the adapter instance is NOT shared.
_SUBAGENT_ADAPTER = AnthropicAdapter(client_provider=lambda: client)


def run_subagent(prompt: str, agent_type: str = "Explore",
                 model_runtime=None) -> str:
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
    # Phase 3C-3B: inherit Parent model SELECTION, own adapter instance.
    adapter = (
        model_runtime.create_adapter()
        if model_runtime is not None
        else _SUBAGENT_ADAPTER
    )
    for _ in range(30):
        resp = adapter.complete(ModelRequest(
            model=MODEL if model_runtime is None else model_runtime.model_id,
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
