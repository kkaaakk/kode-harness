"""message_codec.py - Canonical conversation message conversion.

Phase 3B bridge between the Harness's legacy wire-shaped message history
and the canonical dataclasses in types.py.

The Harness Agent Loop still produces/consumes the legacy shape (D-1):

    {"role": "user", "content": "string"}
    {"role": "user", "content": [{"type": "text", ...},
                                 {"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
    {"role": "assistant", "content": "string"}
    {"role": "assistant", "content": [{"type": "text", ...},
                                      {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]}

This module normalizes BOTH that legacy shape AND the canonical
dataclasses (UserMessage / AssistantMessage / ToolResultMessage) into a
single canonical list. Each adapter then renders the canonical list into
its own wire format.

Constraints (Phase 3B):
  - NEVER rewrite session / compression / trace message structures.
  - Provider-specific conversion (canonical -> wire) lives ONLY inside
    the adapters.
  - Unknown message roles / shapes are passed through unchanged
    (fail-open, never dropped silently - see ``pass_through``).
"""

from __future__ import annotations

import json
from typing import Any

from agents.providers.types import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def to_canonical(messages: list[Any]) -> list[Any]:
    """Convert a mixed legacy/canonical message list to canonical form.

    Returns a NEW list. Input items that are already canonical are kept
    as-is; legacy dicts are converted; anything unrecognized is passed
    through unchanged (so provider-agnostic content like the REPL's raw
    blocks still flows).
    """
    out: list[Any] = []
    for msg in messages:
        if isinstance(msg, (UserMessage, AssistantMessage, ToolResultMessage)):
            out.append(msg)
            continue
        if isinstance(msg, dict):
            converted = _legacy_dict_to_canonical(msg)
            if isinstance(converted, list):
                out.extend(converted)
            else:
                out.append(converted)
            continue
        out.append(msg)
    return out


def _legacy_dict_to_canonical(msg: dict) -> Any | list[Any]:
    role = msg.get("role")
    content = msg.get("content")

    if role == "user":
        if isinstance(content, str):
            return UserMessage(content=content)
        if isinstance(content, list):
            # Split a multi-block user message into a UserMessage for any
            # text blocks and a ToolResultMessage per tool_result block.
            text_parts: list[str] = []
            results: list[ToolResultMessage] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(str(block["text"]))
                elif block.get("type") == "tool_result":
                    results.append(
                        ToolResultMessage(
                            tool_call_id=str(block.get("tool_use_id", "")),
                            content=str(block.get("content", "")),
                            is_error=_looks_like_error(str(block.get("content", ""))),
                        )
                    )
            converted: list[Any] = []
            if text_parts:
                converted.append(UserMessage(content="\n".join(text_parts)))
            converted.extend(results)
            return converted
        # Unknown content shape: pass through.
        return msg

    if role == "assistant":
        if isinstance(content, str):
            return AssistantMessage(text=content)
        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    text_parts.append(str(block["text"]))
                elif btype == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=str(block.get("id", "")),
                            name=str(block.get("name", "")),
                            arguments=dict(block.get("input", {}) or {}),
                        )
                    )
            return AssistantMessage(text="\n".join(text_parts), tool_calls=tool_calls)
        return msg

    # Unknown role: pass through.
    return msg


def _looks_like_error(content: str) -> bool:
    lowered = content.strip().lower()
    return lowered.startswith("error") or "error:" in lowered[:40]


def render_openai(messages: list[Any]) -> list[dict]:
    """Render a canonical message list into OpenAI Chat Completions wire
    format. Accepts canonical dataclasses AND legacy dicts (converted
    on the fly). Tool results become ``{"role": "tool", ...}`` messages.

    IMPORTANT: this renders ONLY the canonical-to-wire direction. It is
    deliberately NOT symmetric with to_canonical (no wire->canonical
    needed in 3B).
    """
    wire: list[dict] = []
    for msg in to_canonical(messages):
        if isinstance(msg, UserMessage):
            wire.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.text}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            wire.append(entry)
        elif isinstance(msg, ToolResultMessage):
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
        else:
            # Pass-through of unrecognized items (legacy dicts of unknown
            # shape, REPL raw blocks, etc.).
            wire.append(msg)
    return wire
