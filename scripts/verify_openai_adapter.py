"""Live verification for OpenAICompatibleAdapter (Phase 3B-2 / 3B-3).

Verifies the SAME OpenAICompatibleAdapter against any OpenAI-compatible
endpoint: plain chat, tool calling, tool-result continuation, usage.

Usage:
    # DeepSeek (Phase 3B-2)
    python scripts/verify_openai_adapter.py --provider deepseek

    # OpenRouter (Phase 3B-3) - set OPENROUTER_API_KEY in .env first.
    # Use a NON-OpenAI model routed through OpenRouter (e.g. Qwen) to
    # prove adapter genericity, not just OpenRouter's OpenAI path.
    python scripts/verify_openai_adapter.py --provider openrouter \
        --model qwen/qwen-2.5-72b-instruct

    # Any other OpenAI-compatible endpoint
    python scripts/verify_openai_adapter.py \
        --base-url https://myhost/v1 --api-key-env MY_KEY --model my-model

Exit codes: 0 = pass, 2 = missing config, 3 = protocol/assertion failure.

Known provider presets (env vars):
  deepseek : DEEPSEEK_API_KEY (fallback ANTHROPIC_API_KEY),
             base https://api.deepseek.com, model deepseek-chat
  openrouter: OPENROUTER_API_KEY,
             base https://openrouter.ai/api/v1, model qwen/qwen-2.5-72b-instruct
             (override with --model)
"""

import argparse
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
    OpenAICompatibleHTTPError,
)
from agents.providers.types import ModelRequest, StopReason, UserMessage

PRESETS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen-2.5-72b-instruct",
    },
}


def _resolve_config(args) -> OpenAICompatibleConfig:
    preset = PRESETS.get(args.provider)
    base_url = args.base_url or (preset["base_url"] if preset else None)
    api_key_env = args.api_key_env or (preset["api_key_env"] if preset else "OPENAI_API_KEY")
    model = args.model or (preset["model"] if preset else "deepseek-chat")
    if not base_url:
        print("Missing base_url (use --provider or --base-url)")
        raise SystemExit(2)
    api_key = os.getenv(api_key_env) or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"Missing {api_key_env} (or ANTHROPIC_API_KEY) in .env")
        raise SystemExit(2)
    return OpenAICompatibleConfig(base_url=base_url, api_key=api_key, model=model)


def _print_result(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PRESETS),
                        help="Known endpoint preset (deepseek|openrouter)")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", help="Env var holding the API key")
    parser.add_argument("--model", help="Model id to verify")
    args = parser.parse_args()

    config = _resolve_config(args)
    print(f"Verifying OpenAICompatibleAdapter against {config.chat_url} "
          f"(model={config.model})")
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

    all_ok = True

    # --- 1. plain chat ---
    print("[1] plain chat ...")
    try:
        r = adapter.complete(ModelRequest(
            model=config.model,
            messages=[UserMessage(content="Reply with exactly: OK")],
        ))
        detail = f"stop={r.stop_reason.value} text={r.text!r} usage={r.usage}"
        ok = bool(r.text.strip()) and r.stop_reason == StopReason.END
        all_ok &= _print_result("text + END + usage", ok, detail)
    except Exception as exc:  # noqa: BLE001 - verification script
        all_ok &= _print_result("text + END + usage", False, repr(exc))

    # --- 2. tool call ---
    print("[2] tool calling ...")
    history = [UserMessage(content="What's the weather in Beijing? Use get_weather.")]
    try:
        r = adapter.complete(ModelRequest(
            model=config.model, messages=history, tools=tools,
        ))
        print(f"    stop={r.stop_reason.value} tool_calls={r.tool_calls}")
        if r.stop_reason != StopReason.TOOL_CALL or not r.tool_calls:
            all_ok &= _print_result(
                "tool call", False,
                "model did not request tool call (model may lack tool "
                "calling support or the provider restricts it)",
            )
            return 3 if all_ok else 3
        call = r.tool_calls[0]
        ok = call.name == "get_weather" and isinstance(call.arguments, dict)
        all_ok &= _print_result(
            f"tool call {call.name} {call.arguments}", ok,
            f"id={call.id}",
        )
    except Exception as exc:  # noqa: BLE001
        all_ok &= _print_result("tool call", False, repr(exc))
        return 3

    # --- 3. tool result continuation (full two-round wire) ---
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
    try:
        r = adapter.complete(ModelRequest(
            model=config.model, messages=history, tools=tools,
        ))
        detail = f"stop={r.stop_reason.value} text={r.text!r} usage={r.usage}"
        ok = bool(r.text.strip())
        all_ok &= _print_result("final text after tool round", ok, detail)
    except Exception as exc:  # noqa: BLE001
        all_ok &= _print_result("final text after tool round", False, repr(exc))

    print()
    if all_ok:
        print(f"[PASS] OpenAICompatibleAdapter against {config.chat_url} "
              f"(model={config.model}): chat + tool call + continuation + usage")
        return 0
    print(f"[FAIL] OpenAICompatibleAdapter against {config.chat_url}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
