"""
compression.py - Context compression utilities.

Provides token estimation, micro-compaction (clearing old tool results),
and full auto-compaction via LLM summarisation.

Depends on: config (TRANSCRIPT_DIR, client, MODEL)
"""

import json
import time

from agents.config import TRANSCRIPT_DIR, client, MODEL
from agents.providers import AnthropicAdapter, ModelRequest

# Phase 3A-1: the ONLY model-call path for auto_compact. Resolves
# ``client`` lazily from this module's globals so module-global client
# injection (tests swapping compression.__globals__["client"]) keeps
# working unchanged.
_ANTHROPIC_ADAPTER = AnthropicAdapter(client_provider=lambda: client)


def estimate_tokens(messages: list) -> int:
    return len(json.dumps(messages, default=str)) // 4


def microcompact(messages: list):
    """Clear old tool_result content, keeping only the 3 most recent."""
    indices = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    indices.append(part)
    if len(indices) <= 3:
        return
    for part in indices[:-3]:
        if isinstance(part.get("content"), str) and len(part["content"]) > 100:
            part["content"] = "[cleared]"


def auto_compact(messages: list) -> list:
    """Persist a transcript and replace *messages* with an LLM summary."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    conv_text = json.dumps(messages, default=str)[-80000:]
    response = _ANTHROPIC_ADAPTER.complete(ModelRequest(
        model=MODEL,
        messages=[{"role": "user", "content": f"Summarize for continuity:\n{conv_text}"}],
        max_tokens=2000,
    ))
    summary = response.text
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {path}]\n{summary}"},
    ]
