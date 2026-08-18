"""types.py - Provider-neutral model types.

The unified vocabulary between Agent Loop and any ProviderAdapter.
Phase 3A deliberately keeps the *request* wire format as-is (D-1 in
docs/phase3a0-provider-contract-audit.md): ``ModelRequest.messages`` is
still the existing Harness canonical/Anthropic-compatible shape. What
3A-1 unifies is the *response*: ``ModelResponse`` hides Anthropic
content-block/stop_reason/usage dialects behind one shape, with
``raw_response`` retained for verbatim history writeback (D-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    """Unified stop reasons. Adapters map provider strings onto these;
    Agent Loop must never see raw provider strings."""

    END = "end"
    TOOL_CALL = "tool_call"
    LENGTH = "length"
    STOP_SEQUENCE = "stop_sequence"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting. Missing provider fields default to 0 (never
    None) so TokenBudget stays simple. If the provider reported NO usage
    object at all, ``ModelResponse.usage`` is None instead."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ModelRequest:
    """A model call. ``messages``/``system``/``tools`` keep the existing
    Harness wire shape (D-1); ``metadata`` carries extra request kwargs
    from extension patches that have no unified field yet (3A bridge)."""

    model: str
    messages: list[Any]
    tools: list[dict] | None = None
    system: Any | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Unified model reply. ``text`` and ``tool_calls`` may BOTH be
    present (one assistant message can carry text + tool calls).
    ``raw_response`` retains the provider object for verbatim history
    writeback; ``provider_metadata`` holds un-unifiable fields."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.UNKNOWN
    usage: TokenUsage | None = None
    model: str | None = None
    provider: str = "unknown"
    raw_response: Any = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
