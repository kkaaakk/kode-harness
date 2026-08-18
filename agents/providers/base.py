"""base.py - ProviderAdapter protocol.

Phase 3A keeps the interface SYNCHRONOUS (the Anthropic SDK used here is
sync; we do not convert the whole Harness to async in this phase).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.providers.types import ModelRequest, ModelResponse


@runtime_checkable
class ProviderAdapter(Protocol):
    """Anything that turns a ModelRequest into a ModelResponse.

    Contract:
      - Errors pass through UNWRAPPED (D-3): whatever the provider SDK
        raises (APIConnectionError, timeout, ...) propagates unchanged.
      - The response's ``raw_response`` retains the provider-native
        object so callers can write wire-format history verbatim.
    """

    provider: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...
