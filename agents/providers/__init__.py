"""providers - Provider abstraction layer.

Phase 3A: unified types + AnthropicAdapter only. OpenAI-compatible
adapter, router, model registry and /model switching come in 3B/3C/3D.
"""

from agents.providers.anthropic_adapter import AnthropicAdapter
from agents.providers.base import ProviderAdapter
from agents.providers.types import (
    ModelRequest,
    ModelResponse,
    StopReason,
    TokenUsage,
    ToolCall,
)

__all__ = [
    "AnthropicAdapter",
    "ProviderAdapter",
    "ModelRequest",
    "ModelResponse",
    "StopReason",
    "TokenUsage",
    "ToolCall",
]
