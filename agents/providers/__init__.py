"""providers - Provider abstraction layer.

Phase 3A: unified types + AnthropicAdapter only. Phase 3B adds the
OpenAI-compatible adapter + canonical message conversion. Router, model
registry and /model switching come in 3C/3D.
"""

from agents.providers.anthropic_adapter import AnthropicAdapter
from agents.providers.base import ProviderAdapter
from agents.providers.message_codec import render_openai, to_canonical
from agents.providers.model_spec import (
    DuplicateModelError,
    ModelCapabilities,
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
    default_model_registry,
)
from agents.providers.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleHTTPError,
)
from agents.providers.provider_router import (
    DuplicateProviderError,
    MissingCredentialError,
    ProviderBinding,
    ProviderRouter,
    UnknownProviderError,
    default_provider_router,
    make_openai_compatible_factory,
)
from agents.providers.types import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

__all__ = [
    "AnthropicAdapter",
    "ProviderAdapter",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleConfig",
    "OpenAICompatibleHTTPError",
    "ModelCapabilities",
    "ModelRegistry",
    "ModelSpec",
    "DuplicateModelError",
    "UnknownModelError",
    "default_model_registry",
    "ProviderBinding",
    "ProviderRouter",
    "DuplicateProviderError",
    "UnknownProviderError",
    "MissingCredentialError",
    "default_provider_router",
    "make_openai_compatible_factory",
    "ModelRequest",
    "ModelResponse",
    "StopReason",
    "TokenUsage",
    "ToolCall",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "to_canonical",
    "render_openai",
]
