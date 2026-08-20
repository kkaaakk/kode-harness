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

from agents.providers.model_spec import ModelRegistry, ModelSpec, UnknownModelError

# The system default model alias (Session selection's fallback entry
# point). Kept distinct from legacy config.MODEL (the actual Anthropic
# model id used by the historic default path).
DEFAULT_MODEL_ALIAS = "claude"


@dataclass(frozen=True)
class ModelChangedPayload:
    """Phase 3D-2 after-change notification payload.

    Describes the change to the model selection for the SESSION's NEXT
    agent run. It NEVER describes a change to a running
    ModelRuntimeContext (the current run is untouched).

    Two layers are kept distinct:
      selection       = the raw session.model_alias (may be None)
      effective_alias = the alias actually used next run (None resolves
                        to DEFAULT_MODEL_ALIAS)

    Carries ModelSpec so extensions do not re-resolve the registry
    (snapshot/consistency). Deliberately does NOT carry ProviderBinding /
    Adapter / API key / ModelRuntimeContext.
    """

    session_id: str | None = None
    old_selection: str | None = None
    new_selection: str | None = None
    old_effective_alias: str | None = None
    new_effective_alias: str | None = None
    old_model_spec: ModelSpec | None = None
    new_model_spec: ModelSpec | None = None
    reason: str = "user_command"


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


def describe_model_alias(
    session: SessionState | None,
    registry: ModelRegistry | None = None,
) -> str:
    """Human-readable description of the current effective model.

    Distinguishes the raw session selection (None vs an alias) from the
    effective alias actually used by the next run, so a default-model
    session reads as 'claude / source=default', not 'None'."""
    from agents.providers.model_spec import default_model_registry

    reg = registry or default_model_registry()
    session_alias = get_session_model_alias(session)
    effective = resolve_session_model_alias(session, None)
    spec = reg.get(effective)
    source = "session" if session_alias is not None else "default"
    return (
        f"Current model: {effective}\n"
        f"Provider: {spec.provider}\n"
        f"Model ID: {spec.model_id}\n"
        f"Source: {source}"
    )


def list_model_aliases(
    session: SessionState | None,
    registry: ModelRegistry | None = None,
) -> str:
    """List all registered models, marking the current effective alias.

    The list is ALWAYS derived from ModelRegistry.list() (registration
    order), never hardcoded."""
    from agents.providers.model_spec import default_model_registry

    reg = registry or default_model_registry()
    current = resolve_session_model_alias(session, None)
    lines = []
    for spec in reg.list():
        marker = "*" if spec.alias == current else " "
        lines.append(f"{marker} {spec.alias}")
        lines.append(f"    provider: {spec.provider}")
        lines.append(f"    model: {spec.model_id}")
    return "\n".join(lines)


def handle_model_command(
    command: str,
    session: SessionState,
    registry: ModelRegistry | None = None,
    extensions=None,
) -> str:
    """Handle a ``/model ...`` command (Phase 3D-1 + 3D-2).

    Supported forms:
        /model             -> current (same as /model current)
        /model current     -> current effective model
        /model list        -> ModelRegistry.list() with current marker
        /model <alias>     -> validate alias, set session.model_alias,
                              "applies to the next agent run"

    Boundaries:
      - reads ModelRegistry, writes SessionState only
      - does NOT read API keys / build adapters (selection != credential)
      - does NOT touch the running ModelRuntimeContext
      - after a successful set with an EFFECTIVE model change, fires
        MODEL_CHANGED (notification only) via ``extensions`` (an
        ExtensionRegistry-like object with emit()); a handler failure or
        block does NOT roll back the already-committed session selection
      - unknown alias -> UnknownModelError, session value unchanged
      - /model current / list never fire the event
    """
    from agents.providers.model_spec import default_model_registry
    from agents.types.events import Event

    reg = registry or default_model_registry()
    parts = command.strip().split()
    # /model -> current
    if len(parts) == 1:
        return describe_model_alias(session, reg)
    sub = parts[1]
    if sub == "current":
        return describe_model_alias(session, reg)
    if sub == "list":
        return list_model_aliases(session, reg)
    # /model <alias>: compute old state, commit, then notify.
    old_selection = get_session_model_alias(session)
    old_effective = resolve_session_model_alias(session, None)
    old_spec = reg.get(old_effective)
    set_session_model_alias(session, sub, reg)  # raises UnknownModelError
    new_selection = get_session_model_alias(session)
    new_effective = resolve_session_model_alias(session, None)
    new_spec = reg.get(new_effective)

    # Notification ONLY when the effective model actually changes.
    if old_effective != new_effective and extensions is not None:
        extensions.emit(Event.MODEL_CHANGED, {
            "event": Event.MODEL_CHANGED,
            "payload": ModelChangedPayload(
                session_id=session.session_id,
                old_selection=old_selection,
                new_selection=new_selection,
                old_effective_alias=old_effective,
                new_effective_alias=new_effective,
                old_model_spec=old_spec,
                new_model_spec=new_spec,
                reason="user_command",
            ),
        })

    return (
        f"Model selected: {sub}\n"
        f"Provider: {new_spec.provider}\n"
        f"Model ID: {new_spec.model_id}\n"
        f"Applies to the next agent run."
    )
