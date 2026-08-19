"""model_runtime.py - Phase 3C-3A ModelRuntimeContext.

An immutable, per-run snapshot of the model selection:

    ModelRuntimeContext
        ├── model_spec        (ModelSpec: alias/provider/model_id/caps)
        └── provider_binding  (ProviderBinding: how to build an Adapter)

Inherited by Main Agent / Subagent / Team member / Compression /
TokenBudget as the SHARED model SELECTION — but each runtime creates its
OWN Adapter instance via ``create_adapter()``. We never share client /
adapter objects across runtimes (Team member runs on its own thread;
adapters may later hold pools/retry/streaming state).

Invariants:
  - frozen; no API key; no Adapter instance; no mutable current model
  - ModelSpec.model_id is the single source of truth for the model id
  - create_adapter() returns a FRESH adapter per call
  - resolution fails fast (UnknownModelError / UnknownProviderError /
    MissingCredentialError) exactly like 3C-2, before any model request
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.providers.model_spec import (
    ModelRegistry,
    ModelSpec,
    default_model_registry,
)
from agents.providers.provider_router import (
    ProviderBinding,
    ProviderRouter,
    default_provider_router,
)


@dataclass(frozen=True)
class ModelRuntimeContext:
    """Model selection snapshot for one agent run.

    Passed down to child runtimes (subagent/team/compression/token-budget)
    so they inherit the model SELECTION while building their own adapters.
    """

    model_spec: ModelSpec
    provider_binding: ProviderBinding

    @property
    def model_id(self) -> str:
        """THE single source of truth for the model id."""
        return self.model_spec.model_id

    def create_adapter(self) -> Any:
        """Create a FRESH adapter for this context's provider+model.

        Every call returns a new instance; adapters are never shared
        between runtimes or reused across runs."""
        if self.provider_binding.adapter_factory is None:
            raise RuntimeError(
                f"Provider {self.provider_binding.provider!r} has no "
                f"adapter_factory configured"
            )
        return self.provider_binding.adapter_factory(self.model_spec.model_id)


def resolve_model_runtime(
    model_alias: str,
    model_registry: ModelRegistry | None = None,
    provider_router: ProviderRouter | None = None,
) -> ModelRuntimeContext:
    """Resolve an alias into a ModelRuntimeContext.

    Chain: alias -> ModelRegistry.get -> ModelSpec
                 -> ProviderRouter.get(provider) -> ProviderBinding
                 -> ModelRuntimeContext

    Fail-fast, no fallback, no model request:
      - unknown alias      -> UnknownModelError
      - unknown provider   -> UnknownProviderError
      - missing credential -> (at create_adapter() time) MissingCredentialError
    """
    registry = model_registry or default_model_registry()
    router = provider_router or default_provider_router()
    spec = registry.get(model_alias)
    binding = router.get(spec.provider)
    return ModelRuntimeContext(model_spec=spec, provider_binding=binding)
