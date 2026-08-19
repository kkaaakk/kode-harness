"""provider_router.py - Phase 3C-1 ProviderBinding + ProviderRouter.

Builds the Provider -> Binding -> Adapter resolution chain. This is a
PURE lookup layer: it does NOT touch the Agent Loop (3C-2), does NOT
enforce capabilities (3C-2/3C-3), and holds NO secrets.

Three-layer responsibility split:
    ProviderRouter   -> table lookup only
    ProviderBinding  -> knows how to obtain an Adapter for a provider
    Adapter          -> knows how to speak the wire protocol

Key modeling invariants (from 3C-0):
    - ``provider`` is the SERVICE identity (deepseek / openrouter /
      anthropic), NEVER the protocol ("openai-compatible").
    - A ProviderBinding is CONFIGURATION, holds no secrets: it stores
      api_key_env (a NAME), not the key value. Credentials are resolved
      lazily only when create_adapter() runs.
    - A ProviderBinding does NOT hold model_id: the model's source of
      truth stays in ModelSpec.model_id, passed into create_adapter().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from agents.providers.anthropic_adapter import AnthropicAdapter
from agents.providers.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
)


@dataclass(frozen=True)
class ProviderBinding:
    """Immutable provider binding: how to obtain an Adapter.

    ``adapter_type``: a stable protocol identifier, e.g. "anthropic" or
    "openai-compatible" (used for tests/validation; the real construction
    happens via ``adapter_factory``).

    ``adapter_factory``: ``callable(model_id: str) -> Adapter``. It is
    responsible for resolving credentials from ``api_key_env`` LAZILY at
    creation time, so:
      - importing/building the router never requires a key,
      - /model list never touches credentials,
      - tests can build the whole registry with no key,
      - secrets have a short lifetime.
    """

    provider: str
    adapter_type: str
    base_url: str | None = None
    api_key_env: str | None = None
    adapter_factory: Callable[[str], Any] | None = None


class UnknownProviderError(KeyError):
    """Raised when a provider is not bound. NEVER fall back silently."""


class DuplicateProviderError(ValueError):
    """Raised when binding a provider that already exists."""


class MissingCredentialError(RuntimeError):
    """Raised when create_adapter() needs a key that is not set.

    The message mentions only the env var NAME - never any secret value.
    """


class ProviderRouter:
    """Ordered registry of ProviderBindings keyed by provider name."""

    def __init__(self) -> None:
        self._bindings: dict[str, ProviderBinding] = {}

    def register(self, binding: ProviderBinding) -> None:
        if binding.provider in self._bindings:
            raise DuplicateProviderError(
                f"Provider {binding.provider!r} already bound "
                f"(adapter_type={self._bindings[binding.provider].adapter_type!r})"
            )
        self._bindings[binding.provider] = binding

    def get(self, provider: str) -> ProviderBinding:
        if provider not in self._bindings:
            raise UnknownProviderError(f"Unknown provider: {provider!r}")
        return self._bindings[provider]

    def contains(self, provider: str) -> bool:
        return provider in self._bindings

    def list(self) -> list[ProviderBinding]:
        """Registration order is stable (authoritative for /model list)."""
        return [self._bindings[k] for k in self._bindings]

    def __len__(self) -> int:
        return len(self._bindings)

    def __contains__(self, provider: object) -> bool:
        return isinstance(provider, str) and provider in self._bindings


# ---------------------------------------------------------------------------
# Adapter factories (lazy credential resolution)
# ---------------------------------------------------------------------------

def _resolve_env(name: str | None, provider: str) -> str:
    if not name:
        raise MissingCredentialError(
            f"Provider {provider!r} has no api_key_env configured"
        )
    value = os.environ.get(name)
    if not value:
        raise MissingCredentialError(
            f"Provider {provider!r} requires env var {name!r} to be set "
            f"(refusing to build adapter without credential)"
        )
    return value


def make_openai_compatible_factory(
    base_url: str,
    api_key_env: str,
) -> Callable[[str], OpenAICompatibleAdapter]:
    """Factory for an OpenAICompatibleAdapter bound to one endpoint.

    Credential is resolved from ``api_key_env`` ONLY when the factory is
    called (create_adapter time), never at router build time.
    """

    def factory(model_id: str) -> OpenAICompatibleAdapter:
        api_key = _resolve_env(api_key_env, base_url)
        config = OpenAICompatibleConfig(
            base_url=base_url,
            api_key=api_key,
            model=model_id,
        )
        return OpenAICompatibleAdapter(config)

    return factory


def _make_anthropic_adapter(client_provider, model_id: str) -> AnthropicAdapter:
    """Anthropic adapter bound to the given client_provider (resolved
    lazily per call). Tests inject fakes via client_provider."""
    return AnthropicAdapter(client_provider=client_provider)


# ---------------------------------------------------------------------------
# Default router (3C-1 canonical starting point)
# ---------------------------------------------------------------------------

def default_provider_router(
    anthropic_client_provider: Callable[[], Any] | None = None,
) -> ProviderRouter:
    """The canonical starting router:

        anthropic  -> AnthropicAdapter
        deepseek   -> OpenAICompatibleAdapter (api.deepseek.com)
        openrouter -> OpenAICompatibleAdapter (openrouter.ai/api/v1)

    ``model_id`` is NOT stored here - it comes from ModelSpec at call
    time via create_adapter(model_id). Credentials resolved lazily.

    ``anthropic_client_provider``: optional zero-arg callable returning
    the Anthropic client to use. If None, the anthropic binding resolves
    the client from ``agents.config`` lazily (the historic behavior).
    Harness callers should pass their own module-level client so fake-
    client tests keep working unchanged.
    """
    if anthropic_client_provider is None:
        # Historic default: resolve from agents.config at adapter-creation
        # time. NOTE: only safe when agents.config is importable under the
        # current anthropic module - harness callers should inject their
        # own client_provider to stay testable.
        def _lazy_config_client() -> Any:
            from agents.config import client  # noqa: PLC0415
            return client

        anthropic_client_provider = _lazy_config_client

    router = ProviderRouter()
    router.register(ProviderBinding(
        provider="anthropic",
        adapter_type="anthropic",
        adapter_factory=lambda model_id: _make_anthropic_adapter(
            anthropic_client_provider, model_id
        ),
    ))
    router.register(ProviderBinding(
        provider="deepseek",
        adapter_type="openai-compatible",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        adapter_factory=make_openai_compatible_factory(
            "https://api.deepseek.com", "DEEPSEEK_API_KEY"
        ),
    ))
    router.register(ProviderBinding(
        provider="openrouter",
        adapter_type="openai-compatible",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        adapter_factory=make_openai_compatible_factory(
            "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
        ),
    ))
    return router
