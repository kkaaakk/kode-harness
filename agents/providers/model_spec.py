"""model_spec.py - Phase 3C-0 domain model: ModelSpec / ModelCapabilities /
ModelRegistry.

Three concepts are kept SEPARATE (this is the core 3C modeling decision):

    Model    -> which model am I using
    Provider -> which service/endpoint serves it
    Adapter  -> which wire protocol that endpoint speaks

Examples:
    Qwen via OpenRouter        Model: qwen/qwen-2.5-72b-instruct
                               Provider: openrouter
                               Adapter (later): openai-compatible

    DeepSeek                   Model: deepseek-chat
                               Provider: deepseek
                               Adapter (later): openai-compatible

Therefore ``provider`` is NEVER collapsed into "openai-compatible": two
different providers may resolve to the SAME adapter type later, but they
are distinct provider identities.

Constraints for 3C-0:
  - ModelSpec holds NO endpoint configuration (no base_url / api_key /
    api_key_env / client / adapter instance). Endpoint binding belongs to
    ProviderRouter (3C-1).
  - ModelCapabilities describe a MODEL, not an Adapter. They are
    DESCRIPTIVE only - no runtime enforcement here (no tool pruning).
  - ModelRegistry: duplicate alias fails fast, unknown alias fails fast
    (no fallback default), registration order stable, never mutates a
    registered ModelSpec, never reads env vars, never creates clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    """Capability facts about a concrete model (NOT about its provider
    or adapter). All default to False; fill only what is known."""

    tool_calling: bool = False
    parallel_tool_calls: bool = False
    vision: bool = False
    reasoning: bool = False
    prompt_cache: bool = False
    structured_output: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """A stable, Harness-facing model identity.

    ``alias``: stable name used by users/Harness (e.g. "deepseek").
    ``model_id``: the exact string sent to the provider (e.g.
    "deepseek-chat" or "qwen/qwen-2.5-72b-instruct"). They are decoupled:
    if a provider renames a model_id, only this spec changes, not every
    reference to the alias.

    No endpoint config here - provider routing + binding come in 3C-1.
    """

    alias: str
    provider: str
    model_id: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


class UnknownModelError(KeyError):
    """Raised when an alias is not registered. NEVER fall back silently."""


class DuplicateModelError(ValueError):
    """Raised when registering an alias that already exists."""


class ModelRegistry:
    """Ordered registry of ModelSpecs keyed by alias.

    Thread-safe. Registration order is preserved for ``list()``.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.alias in self._specs:
            raise DuplicateModelError(
                f"Model alias {spec.alias!r} already registered "
                f"(existing provider={self._specs[spec.alias].provider!r})"
            )
        # Store a frozen spec; callers can keep their own reference but
        # cannot mutate it, and we never expose internal state that could
        # be edited (all entries are frozen dataclasses).
        self._specs[spec.alias] = spec

    def get(self, alias: str) -> ModelSpec:
        if alias not in self._specs:
            raise UnknownModelError(f"Unknown model alias: {alias!r}")
        return self._specs[alias]

    def list(self) -> list[ModelSpec]:
        """Registration order is stable and authoritative for /model list."""
        return [self._specs[k] for k in self._specs]

    def contains(self, alias: str) -> bool:
        return alias in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, alias: object) -> bool:
        return isinstance(alias, str) and alias in self._specs


# ---------------------------------------------------------------------------
# Fixed reference samples (3C-0). Used by tests and as the canonical
# starting registry for the Harness. Endpoint binding is NOT here.
# ---------------------------------------------------------------------------

def default_model_registry() -> ModelRegistry:
    """The canonical starting registry for the Harness.

    claude            : provider=anthropic,        adapter later = anthropic
    deepseek          : provider=deepseek,         adapter later = openai-compatible
    qwen-openrouter   : provider=openrouter,       adapter later = openai-compatible
    """
    registry = ModelRegistry()
    registry.register(ModelSpec(
        alias="claude",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        capabilities=ModelCapabilities(
            tool_calling=True,
            parallel_tool_calls=True,
            vision=True,
            reasoning=False,
            prompt_cache=True,
            structured_output=False,
        ),
    ))
    registry.register(ModelSpec(
        alias="deepseek",
        provider="deepseek",
        model_id="deepseek-chat",
        capabilities=ModelCapabilities(
            tool_calling=True,
            parallel_tool_calls=True,
            vision=False,
            reasoning=False,
            prompt_cache=True,
            structured_output=False,
        ),
    ))
    registry.register(ModelSpec(
        alias="qwen-openrouter",
        provider="openrouter",
        model_id="qwen/qwen-2.5-72b-instruct",
        capabilities=ModelCapabilities(
            tool_calling=True,
            parallel_tool_calls=False,
            vision=True,
            reasoning=False,
            prompt_cache=False,
            structured_output=False,
        ),
    ))
    return registry
