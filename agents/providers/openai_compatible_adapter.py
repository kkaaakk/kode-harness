"""openai_compatible_adapter.py - OpenAI Chat Completions compatible adapter.

The ONLY place in production code that knows OpenAI-compatible wire
dialects:
  - function.parameters (tool schema)
  - tool_calls[].function.arguments (JSON string)
  - finish_reason
  - prompt_tokens / completion_tokens / prompt_tokens_details.cached_tokens

DeepSeek / OpenRouter / any local OpenAI-compatible server are just
different ``OpenAICompatibleConfig`` values - one adapter, many
providers. Errors (HTTP, JSON, SDK) pass through unwrapped (D-3).
"""

from __future__ import annotations

import json
import types
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from agents.providers.message_codec import render_openai
from agents.providers.types import (
    ModelRequest,
    ModelResponse,
    StopReason,
    TokenUsage,
    ToolCall,
)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def _to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    """Canonical tool schema -> OpenAI function schema.

    Canonical (Anthropic-shaped) tools look like:
        {"name": ..., "description": ..., "input_schema": {...}}
    OpenAI wants:
        {"type": "function", "function": {"name": ..., "description": ...,
                                           "parameters": {...}}}

    NEVER mutates the input tools. Returns None when tools is None/empty.
    """
    if not tools:
        return None
    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
        )
    return converted


_FINISH_REASON_MAP = {
    "stop": StopReason.END,
    "tool_calls": StopReason.TOOL_CALL,
    "length": StopReason.LENGTH,
    "content_filter": StopReason.CONTENT_FILTER,
    "stop_sequence": StopReason.STOP_SEQUENCE,
}


class _HttpClient:
    """Minimal stdlib HTTP client for OpenAI-compatible endpoints.

    ``complete(payload)`` POSTs to {base_url}/chat/completions and
    returns the parsed JSON body. Non-2xx statuses raise
    ``OpenAICompatibleHTTPError`` (unwrapped upstream)."""

    def __init__(self, config: OpenAICompatibleConfig, timeout: float = 120.0):
        self._config = config
        self._timeout = timeout

    def complete(self, payload: dict) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._config.chat_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise OpenAICompatibleHTTPError(
                f"HTTP {exc.code} from {self._config.chat_url}: {detail}"
            ) from exc
        return json.loads(body)


class OpenAICompatibleHTTPError(Exception):
    """Raised for non-2xx responses from an OpenAI-compatible endpoint."""


class OpenAICompatibleAdapter:
    provider = "openai-compatible"

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        client_provider: Callable[[], Any] | None = None,
    ):
        """``client_provider``: zero-arg callable returning an object with
        ``complete(payload: dict) -> dict``. Defaults to a stdlib HTTP
        client bound to *config*. Tests inject a fake client_provider so
        no network is touched."""
        self._config = config
        if client_provider is None:
            client_provider = lambda: _HttpClient(config)  # noqa: E731
        self._client_provider = client_provider

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": render_openai(request.messages),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.system is not None:
            # Some compatible servers accept top-level "system"; the
            # safest is a system role message. Keep BOTH when possible.
            payload["messages"].insert(
                0, {"role": "system", "content": request.system}
            )
        openai_tools = _to_openai_tools(request.tools)
        if openai_tools is not None:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"
        # 3A-style bridge: extra request kwargs from extension patches.
        payload.update(request.metadata)

        raw = self._client_provider().complete(payload)
        return self._to_model_response(raw, model=request.model)

    def _to_model_response(self, raw: dict, *, model: str | None) -> ModelResponse:
        choices = raw.get("choices") or []
        if not choices:
            return ModelResponse(
                stop_reason=StopReason.UNKNOWN,
                usage=_parse_usage(raw.get("usage")),
                model=model,
                provider=self.provider,
                raw_response=raw,
                provider_metadata={"raw_finish_reason": None},
            )
        choice = choices[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        stop = _FINISH_REASON_MAP.get(finish, StopReason.UNKNOWN)

        text = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            # fail-fast on malformed arguments (never silently {}).
            arguments = json.loads(fn.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"OpenAI-compatible tool arguments must be a JSON object, "
                    f"got: {type(arguments).__name__} for tool {fn.get('name')!r}"
                )
            tool_calls.append(
                ToolCall(
                    id=str(raw_call.get("id", "")),
                    name=str(fn.get("name", "")),
                    arguments=arguments,
                )
            )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=_parse_usage(raw.get("usage")),
            model=model,
            provider=self.provider,
            raw_response=_wire_response(raw, text, tool_calls),
            provider_metadata={
                "raw_finish_reason": finish,
                "raw_message": message,
            },
        )


class _Block:
    """Attribute-style content block (matches the Anthropic SDK's block
    surface that the Agent Loop reads for history writeback)."""

    __slots__ = ("type", "text", "name", "id", "input")

    def __init__(self, *, type_, text=None, name=None, id_=None, input_=None):
        self.type = type_
        self.text = text
        self.name = name
        self.id = id_
        self.input = input_


def _wire_response(raw: dict, text: str, tool_calls: list[ToolCall]):
    """Wrap the raw OpenAI response dict with a ``content`` list of
    Anthropic-style blocks, so the Agent Loop's verbatim history writeback
    (``response.raw_response.content``) works uniformly across adapters.

    The wrapped object retains the original dict fields (access via
    attribute on the original keys) and exposes ``.content``.
    """
    blocks = []
    if text:
        blocks.append(_Block(type_="text", text=text))
    for tc in tool_calls:
        blocks.append(_Block(type_="tool_use", name=tc.name,
                             id_=tc.id, input_=tc.arguments))
    namespace = types.SimpleNamespace(content=blocks)
    namespace.__dict__.update(raw)
    return namespace


def _parse_usage(raw_usage: dict | None) -> TokenUsage | None:
    """OpenAI-compatible usage -> TokenUsage.

    prompt_tokens -> input_tokens
    completion_tokens -> output_tokens
    prompt_tokens_details.cached_tokens -> cache_read_tokens (optional)

    Provider-specific extras stay in the raw response (provider_metadata),
    NOT in TokenUsage. usage=None (or non-dict) -> None."""
    if not raw_usage or not isinstance(raw_usage, dict):
        return None
    details = raw_usage.get("prompt_tokens_details") or {}
    return TokenUsage(
        input_tokens=int(raw_usage.get("prompt_tokens") or 0),
        output_tokens=int(raw_usage.get("completion_tokens") or 0),
        cache_read_tokens=int(
            details.get("cached_tokens") or 0
        ) if isinstance(details, dict) else 0,
        cache_write_tokens=0,
    )
