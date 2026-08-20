"""session.py - Phase 3D-0 Session model selection contract.

SessionState holds ONLY the model *alias* for the NEXT agent run. It
never stores a ModelSpec / ProviderBinding / Adapter - those are
re-resolved into a fresh immutable ModelRuntimeContext at each agent_loop
startup.

The critical 3D separation:

    SessionState.model_alias    -> "which model for the NEXT run"
    ModelRuntimeContext         -> "the frozen model of the CURRENT run"

These are NEVER the same mutable state. Changing session.model_alias
while an agent is running does NOT affect the running ModelRuntimeContext;
only the next agent_loop() picks it up.

Selection precedence (highest first):
    1. explicit agent_loop(model_alias=...)
    2. SessionState.model_alias
    3. DEFAULT_MODEL_ALIAS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.providers.model_spec import ModelRegistry, UnknownModelError

# The system default model alias (Session selection's fallback entry
# point). Kept distinct from legacy config.MODEL (the actual Anthropic
# model id used by the historic default path).
DEFAULT_MODEL_ALIAS = "claude"


@dataclass
class SessionState:
    """Mutable per-session selection state. Only the alias is stored."""

    session_id: str | None = None
    model_alias: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def get_session_model_alias(session: SessionState | None) -> str | None:
    """Return the session's selected model alias (None = use default)."""
    if session is None:
        return None
    return session.model_alias


def set_session_model_alias(
    session: SessionState,
    alias: str,
    registry: ModelRegistry | None = None,
) -> None:
    """Select *alias* for the next agent run of *session*.

    Validates against the ModelRegistry BEFORE mutating the session:
    an unknown alias raises UnknownModelError and leaves the session's
    previous selection unchanged (no state corruption).

    This only SELECTS a model - it does NOT read API keys or build any
    adapter (credential validation happens at create_adapter() time in
    the next run).
    """
    from agents.providers.model_spec import default_model_registry

    reg = registry or default_model_registry()
    # Fail fast if unknown - do not touch session state.
    reg.get(alias)  # raises UnknownModelError
    session.model_alias = alias


def resolve_session_model_alias(
    session: SessionState | None,
    explicit_alias: str | None,
) -> str:
    """Resolve the effective model alias for an agent_loop startup.

    Precedence: explicit parameter > session selection > default.
    """
    if explicit_alias is not None:
        return explicit_alias
    if session is not None and session.model_alias is not None:
        return session.model_alias
    return DEFAULT_MODEL_ALIAS
