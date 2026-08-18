"""
events.py - Extension system event contracts.

Defines the 8 hook points for stage 1, their input/output contracts,
priority conventions, and the HookResult structure.

Design rules (stage 1):
- Kernel safety policy stays in Kernel; Extensions can only ADD restrictions.
- block=True stops subsequent non-kernel handlers.
- Extensions are fail-open by default; safety-class extensions are fail-closed.
- Every handler has a timeout (default 10s).
- Sync and async handlers are both supported.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Awaitable, Union


# ---------------------------------------------------------------------------
# Hook points (stage 1: 8 nodes)
# ---------------------------------------------------------------------------

class HookPoint(str):
    """String enum for hook points. Using str subclass for easy comparison."""


class Event(HookPoint):
    # Agent lifecycle
    BEFORE_AGENT_START = "before_agent_start"
    AGENT_END = "agent_end"
    # Per-turn lifecycle
    TURN_END = "turn_end"
    # Model call
    BEFORE_MODEL_REQUEST = "before_model_request"
    AFTER_MODEL_RESPONSE = "after_model_response"
    # Tool call
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_RESULT = "after_tool_result"
    # Compaction
    BEFORE_COMPACTION = "before_compaction"


# Ordered list for documentation / iteration
ALL_HOOKS = [
    Event.BEFORE_AGENT_START,
    Event.BEFORE_MODEL_REQUEST,
    Event.AFTER_MODEL_RESPONSE,
    Event.BEFORE_TOOL_CALL,
    Event.AFTER_TOOL_RESULT,
    Event.TURN_END,
    Event.AGENT_END,
    Event.BEFORE_COMPACTION,
]


# ---------------------------------------------------------------------------
# Priority conventions
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Priority conventions. Higher number = runs earlier.

    IMPORTANT: Real Kernel safety checks (dangerous-command blocking, path
    validation) live OUTSIDE the ExtensionRegistry, in base_tools.run_bash
    and harness_core. They are NOT registered as extensions and cannot be
    removed by clear() / unregister().

    The KERNEL_AUDIT priority below is for OBSERVER/AUDIT handlers inside
    the registry (e.g. tracing, logging) that should run first and cannot
    be bypassed by ordinary extensions. It is NOT where security policy
    is enforced — security is enforced in the Kernel layer before/after
    emit() is called.
    """
    OBSERVER = 0          # logging, tracing, metrics (read-only)
    NORMAL = 100          # ordinary extensions
    PROJECT = 500         # project-level extensions
    KERNEL_AUDIT = 1000   # kernel-level audit/observer (highest in registry)


# ---------------------------------------------------------------------------
# Hook result - structured return, NOT direct context mutation
# ---------------------------------------------------------------------------

@dataclass
class HookResult:
    """Structured result returned by hook handlers.

    Extensions should NOT mutate the context dict directly. Instead they
    return a HookResult describing how the loop should adjust.

    Fields:
        block: If True, stop processing further handlers with lower priority
               for this event. The triggering action (tool call / model call)
               is aborted. NOTE: this only affects Extension handlers; real
               Kernel safety checks run OUTSIDE the registry and are never
               bypassed by an extension block.
        reason: Human-readable reason for block or modification.
        context_patch: Dict merged into the event context AFTER this handler.
                       Higher-priority handlers set fields first; lower-priority
                       handlers CANNOT override fields already set by a higher
                       priority handler (first-wins by priority).
        tool_args_patch: (BEFORE_TOOL_CALL only) Dict merged into tool args.
                         Same first-wins-by-priority rule applies.
        tool_result_patch: (AFTER_TOOL_RESULT only) Replacement ToolResult-like
                           dict {"content": ..., "is_error": ...}. Higher
                           priority wins; lower priority cannot override.
        model_request_patch: (BEFORE_MODEL_REQUEST only) Dict merged into the
                             kwargs passed to provider.messages.create.
                             First-wins by priority.
        model_response_patch: (AFTER_MODEL_RESPONSE only) Replacement response
                              object (rarely used; prefer context_patch).
        skip_action: If True, skip the actual action (tool execution / model
                     call) but continue subsequent handlers. Used by read-only
                     or plan-mode extensions.
        metadata: Free-form per-extension metadata (e.g. trace span info).
    """
    block: bool = False
    reason: str | None = None
    context_patch: dict | None = None
    tool_args_patch: dict | None = None
    tool_result_patch: dict | None = None
    model_request_patch: dict | None = None
    model_response_patch: Any | None = None
    skip_action: bool = False
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Event context contracts
# ---------------------------------------------------------------------------

@dataclass
class EventContext:
    """Mutable per-event context passed through handlers.

    Each handler receives `context: dict` (this dataclass is the canonical
    shape; in practice we pass a plain dict for forward-compat). Handlers
    may read freely but should return a HookResult rather than mutating.

    Different events populate different fields. Optional fields are None
    when not applicable to the current event.
    """
    event: str
    # Common
    trace_id: str | None = None
    turn_index: int | None = None
    # Agent lifecycle
    messages: list | None = None           # BEFORE_AGENT_START / AGENT_END
    # Model call
    model: str | None = None               # BEFORE_MODEL_REQUEST
    system_prompt: str | None = None       # BEFORE_MODEL_REQUEST
    tools: list | None = None              # BEFORE_MODEL_REQUEST
    request_kwargs: dict | None = None     # BEFORE_MODEL_REQUEST
    response: Any | None = None            # AFTER_MODEL_RESPONSE
    # Tool call
    tool_name: str | None = None           # BEFORE_TOOL_CALL / AFTER_TOOL_RESULT
    tool_use_id: str | None = None         # BEFORE_TOOL_CALL / AFTER_TOOL_RESULT
    tool_args: dict | None = None          # BEFORE_TOOL_CALL
    tool_result: Any | None = None         # AFTER_TOOL_RESULT
    tool_error: str | None = None          # AFTER_TOOL_RESULT
    actor: str | None = None               # BEFORE_TOOL_CALL (who invoked)
    # Compaction
    pre_compact_messages: list | None = None  # BEFORE_COMPACTION
    # Extension-settable flags
    skip_action: bool = False


# ---------------------------------------------------------------------------
# Handler signature
# ---------------------------------------------------------------------------

# A handler is either sync (returns HookResult | None) or async (returns
# Awaitable[HookResult | None]). Returning None == no-op HookResult.
SyncHandler = Callable[[dict], Union[HookResult, None]]
AsyncHandler = Callable[[dict], Awaitable[Union[HookResult, None]]]
Handler = Union[SyncHandler, AsyncHandler]


# ---------------------------------------------------------------------------
# Handler timeout error
# ---------------------------------------------------------------------------

class HandlerTimeoutError(Exception):
    """Raised when a handler exceeds its timeout."""

    def __init__(self, extension_id: str, event: str, timeout: float):
        self.extension_id = extension_id
        self.event = event
        self.timeout = timeout
        super().__init__(
            f"Handler '{extension_id}' on '{event}' exceeded {timeout}s timeout"
        )


class HandlerError(Exception):
    """Wraps any exception raised by a handler, with extension_id for logging."""

    def __init__(self, extension_id: str, event: str, cause: Exception):
        self.extension_id = extension_id
        self.event = event
        self.cause = cause
        super().__init__(
            f"Handler '{extension_id}' on '{event}' raised {type(cause).__name__}: {cause}"
        )


# ---------------------------------------------------------------------------
# Fail policy
# ---------------------------------------------------------------------------

class FailPolicy(str):
    NORMAL = "fail_open"       # skip handler, log, continue
    SAFETY = "fail_closed"     # abort action, log, raise to caller


__all__ = [
    "HookPoint",
    "Event",
    "ALL_HOOKS",
    "Priority",
    "HookResult",
    "EventContext",
    "Handler",
    "SyncHandler",
    "AsyncHandler",
    "HandlerTimeoutError",
    "HandlerError",
    "FailPolicy",
]
