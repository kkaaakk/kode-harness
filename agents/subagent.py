"""
subagent.py - Lightweight subagent spawning for isolated exploration or work.

Supports two agent types:
- ``Explore``: read-only (bash + read_file)
- ``general-purpose``: read/write (+ write_file + edit_file)

Depends on: config (client, MODEL), base_tools
"""

from agents.config import client, MODEL
from agents.base_tools import run_bash, run_read, run_write, run_edit


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
        resp = client.messages.create(
            model=MODEL, messages=sub_msgs, tools=sub_tools, max_tokens=8000
        )
        sub_msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                h = sub_handlers.get(b.name, lambda **kw: "Unknown tool")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": str(h(**b.input))[:50000],
                    }
                )
        sub_msgs.append({"role": "user", "content": results})
    if resp:
        return (
            "".join(b.text for b in resp.content if hasattr(b, "text"))
            or "(no summary)"
        )
    return "(subagent failed)"
