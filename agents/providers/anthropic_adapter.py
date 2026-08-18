"""anthropic_adapter.py - Anthropic SDK adapter.

The ONLY place in production code allowed to call
``client.messages.create()`` (3A-1 hard acceptance). Translates between
the unified types (types.py) and the Anthropic SDK.

Mapping notes:
  - stop_reason: end_turn -> END, tool_use -> TOOL_CALL,
    max_tokens -> LENGTH, stop_sequence -> STOP_SEQUENCE, else UNKNOWN.
    The raw value is preserved in provider_metadata["raw_stop_reason"].
  - text blocks: concatenated in order into ModelResponse.text
    (multi-text-block responses emit a single aggregated text).
  - tool_use blocks -> ToolCall(id=block.id, name=block.name,
    arguments=dict(block.input)).
  - usage: input/output tokens mapped; cache_read_tokens uses
    ``cache_read_input_tokens``, cache_write_tokens uses
    ``cache_creation_input_tokens`` if present else 0. If the response
    has NO usage object at all, ModelResponse.usage is None.
  - unknown content block types are preserved in
    provider_metadata["unknown_blocks"] (never dropped).
  - Exceptions from the SDK pass through unchanged (D-3).

The adapter resolves the client lazily via ``client_provider`` at each
call (call sites pass ``lambda: client``), so the module-global client
stays THE injection point: tests that patch ``module.client`` /
``run_subagent.__globals__["client"]`` keep working unchanged.
"""

from __future__ import annotations

from typing import Any

from agents.providers.types import (
    ModelRequest,
    ModelResponse,
    StopReason,
    TokenUsage,
    ToolCall,
)

_STOP_REASON_MAP = {
    "end_turn": StopReason.END,
    "tool_use": StopReason.TOOL_CALL,
    "max_tokens": StopReason.LENGTH,
    "stop_sequence": StopReason.STOP_SEQUENCE,
}


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, client: Any = None, *, client_provider: Any = None):
        """Bind to a client object OR a client provider callable.

        ``client``: fixed reference (e.g. the config singleton).
        ``client_provider``: zero-arg callable returning the current
        client; called on EVERY complete(). This lets call-site modules
        resolve ``client`` lazily from their module globals, so tests
        that swap ``module.client`` (or ``run_subagent.__globals__``)
        keep working unchanged — the same injection point as pre-3A.
        """
        if client is not None and client_provider is not None:
            raise ValueError("pass either client or client_provider, not both")
        self._client = client
        self._client_provider = client_provider

    def _resolve_client(self) -> Any:
        if self._client_provider is not None:
            return self._client_provider()
        return self._client

    def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict[str, Any] = {}
        for key in (
            "model",
            "system",
            "messages",
            "tools",
            "max_tokens",
            "temperature",
        ):
            value = getattr(request, key)
            if value is not None:
                kwargs[key] = value
        # 3A bridge: extension patches may add request kwargs with no
        # unified field yet; forward them verbatim.
        kwargs.update(request.metadata)
        raw = self._resolve_client().messages.create(**kwargs)
        return self._to_model_response(raw, model=request.model)

    def _to_model_response(self, raw: Any, *, model: str | None) -> ModelResponse:
        raw_stop = getattr(raw, "stop_reason", None)
        stop = _STOP_REASON_MAP.get(raw_stop, StopReason.UNKNOWN)

        usage: TokenUsage | None = None
        raw_usage = getattr(raw, "usage", None)
        if raw_usage is not None:
            usage = TokenUsage(
                input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
                output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(
                    raw_usage, "cache_read_input_tokens", 0
                ) or 0,
                cache_write_tokens=getattr(
                    raw_usage, "cache_creation_input_tokens", 0
                ) or 0,
            )

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        unknown_blocks: list[Any] = []
        for block in getattr(raw, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    text_chunks.append(text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", None) or {}),
                    )
                )
            else:
                unknown_blocks.append(block)

        return ModelResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=usage,
            model=model,
            provider=self.provider,
            raw_response=raw,
            provider_metadata={
                "raw_stop_reason": raw_stop,
                "unknown_blocks": unknown_blocks,
            },
        )
