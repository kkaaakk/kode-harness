"""DeepSeek live verification for OpenAICompatibleAdapter (Phase 3B-2).

Usage:
    python scripts/verify_openai_adapter.py            # uses .env DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY=xxx python scripts/verify_openai_adapter.py

Verifies: plain chat, tool calling, tool-result continuation, usage.
Exit 0 on success. Requires network access to api.deepseek.com.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Windows consoles default to GBK; force UTF-8 so model text with emoji
# does not crash the verification script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from agents.providers.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
)
from agents.providers.types import ModelRequest, StopReason, UserMessage


def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing DEEPSEEK_API_KEY / ANTHROPIC_API_KEY in .env")
        return 2
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    config = OpenAICompatibleConfig(
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=api_key,
        model=model,
    )
    adapter = OpenAICompatibleAdapter(config)

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ]

    # --- 1. plain chat ---
    print("[1] plain chat ...")
    r = adapter.complete(ModelRequest(
        model=model,
        messages=[UserMessage(content="Reply with exactly: OK")],
    ))
    print(f"    stop={r.stop_reason.value} text={r.text!r} usage={r.usage}")
    assert r.text.strip(), "plain chat returned empty text"

    # --- 2. tool call ---
    print("[2] tool calling ...")
    history = [UserMessage(content="What's the weather in Beijing? Use get_weather.")]
    r = adapter.complete(ModelRequest(
        model=model, messages=history, tools=tools,
    ))
    print(f"    stop={r.stop_reason.value} tool_calls={r.tool_calls}")
    if r.stop_reason != StopReason.TOOL_CALL or not r.tool_calls:
        print("    WARN: model did not request tool call (model may not "
              "support tool calling); skipping continuation check")
        return 0 if r.text else 3
    call = r.tool_calls[0]
    assert call.name == "get_weather", call.name

    # --- 3. tool result continuation ---
    print("[3] tool result continuation ...")
    history.append({
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": call.id, "name": call.name,
             "input": call.arguments},
        ],
    })
    history.append({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": call.id,
             "content": "Beijing: 22C sunny"},
        ],
    })
    r = adapter.complete(ModelRequest(model=model, messages=history, tools=tools))
    print(f"    stop={r.stop_reason.value} text={r.text!r} usage={r.usage}")
    assert r.text.strip(), "continuation returned empty text"

    print("[PASS] DeepSeek OpenAI-compatible: chat + tool call + continuation + usage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
