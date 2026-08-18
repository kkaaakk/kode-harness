"""
extension_system.py - Extension registry and event dispatch.

Stage 1 design:
- Function-style registration: registry.on(event, fn, priority=N)
- Optional class encapsulation via registry.install(obj) where obj has
  .register(registry) method. NO base class, NO inheritance tree.
- Stable ordering: sort by (priority DESC, registration_order ASC).
- block=True stops subsequent non-kernel handlers (priority < KERNEL_SAFETY).
  Kernel safety handlers (priority >= KERNEL_SAFETY) ALWAYS run, even if a
  lower-priority handler blocked. This enforces "extensions can only ADD
  restrictions, never bypass kernel safety".
- fail-open for normal handlers (log + skip); fail-closed for safety handlers.
- Handler timeout enforced via thread (sync) or asyncio.wait_for (async).
- Sync and async handlers both supported; dispatch is sync-first, async
  handlers are run via a fresh event loop with timeout.

This module has NO dependency on harness_core, so it can be unit-tested
in isolation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agents.types.events import (
    ALL_HOOKS,
    Event,
    FailPolicy,
    Handler,
    HandlerError,
    HandlerTimeoutError,
    HookResult,
    Priority,
)


logger = logging.getLogger("agents.extension_system")


# ---------------------------------------------------------------------------
# Registered handler entry
# ---------------------------------------------------------------------------

@dataclass
class _HandlerEntry:
    extension_id: str
    event: str
    handler: Handler
    priority: int
    fail_policy: str              # FailPolicy.NORMAL or FailPolicy.SAFETY
    timeout: float                # seconds
    registration_order: int       # tie-breaker for stable sort


@dataclass
class DispatchOutcome:
    """Result of dispatching one event to all matching handlers."""
    event: str
    results: list[HookResult] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
    skip_action: bool = False
    errors: list[HandlerError] = field(default_factory=list)
    timeouts: list[HandlerTimeoutError] = field(default_factory=list)
    # Patches accumulated from results
    context_patch: dict = field(default_factory=dict)
    tool_args_patch: dict = field(default_factory=dict)
    tool_result_patch: dict | None = None
    model_request_patch: dict = field(default_factory=dict)
    model_response_patch: Any | None = None


# ---------------------------------------------------------------------------
# ExtensionRegistry
# ---------------------------------------------------------------------------

class ExtensionRegistry:
    """Registry + dispatcher for extension hooks.

    Thread-safety: registration uses a lock; dispatch is lock-free once
    a snapshot of handlers is taken.
    """

    def __init__(self):
        # event -> list of entries (kept sorted on insertion)
        self._handlers: dict[str, list[_HandlerEntry]] = {
            hook: [] for hook in ALL_HOOKS
        }
        self._lock = threading.RLock()
        self._registration_counter = 0
        # Per-event logger
        self._log = logger

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(
        self,
        event: str,
        handler: Handler,
        *,
        priority: int = Priority.NORMAL,
        extension_id: str | None = None,
        fail_policy: str = FailPolicy.NORMAL,
        timeout: float = 10.0,
    ) -> str:
        """Register a handler for an event.

        Args:
            event: One of Event.* constants.
            handler: Sync or async callable taking a context dict.
            priority: Higher runs earlier. See Priority.
            extension_id: Stable identifier for logging. Auto-generated if None.
            fail_policy: fail_open (default) or fail_closed. Safety handlers
                         should use fail_closed.
            timeout: Max seconds the handler may run.

        Returns:
            The extension_id (useful for unregister).
        """
        if event not in self._handlers:
            raise ValueError(f"Unknown event: {event}. Known: {ALL_HOOKS}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        ext_id = extension_id or f"ext_{uuid.uuid4().hex[:8]}"
        entry = _HandlerEntry(
            extension_id=ext_id,
            event=event,
            handler=handler,
            priority=int(priority),
            fail_policy=fail_policy,
            timeout=float(timeout),
            registration_order=self._next_registration_order(),
        )
        with self._lock:
            # Insert maintaining sort: priority DESC, registration ASC
            bucket = self._handlers[event]
            idx = 0
            for i, existing in enumerate(bucket):
                if (entry.priority, -entry.registration_order) > (
                    existing.priority,
                    -existing.registration_order,
                ):
                    idx = i
                    break
                idx = i + 1
            bucket.insert(idx, entry)
        return ext_id

    def install(self, obj: Any) -> None:
        """Install an extension object that exposes .register(registry).

        No base class required. The object's register() method calls
        registry.on(...) for whatever events it cares about.
        """
        if not hasattr(obj, "register") or not callable(obj.register):
            raise TypeError(
                f"Extension object must have a callable .register(registry) method. "
                f"Got {type(obj).__name__}"
            )
        obj.register(self)

    def unregister(self, extension_id: str) -> int:
        """Remove all handlers with the given extension_id. Returns count removed."""
        removed = 0
        with self._lock:
            for event_key, bucket in self._handlers.items():
                kept = []
                for entry in bucket:
                    if entry.extension_id == extension_id:
                        removed += 1
                    else:
                        kept.append(entry)
                self._handlers[event_key] = kept
        return removed

    def clear(self) -> None:
        """Remove all handlers (for tests)."""
        with self._lock:
            for key in self._handlers:
                self._handlers[key] = []

    def handlers_for(self, event: str) -> list[_HandlerEntry]:
        """Snapshot of handlers for an event (for inspection / tests)."""
        with self._lock:
            return list(self._handlers.get(event, []))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def emit(self, event: str, context: dict) -> DispatchOutcome:
        """Dispatch an event to all registered handlers.

        Semantics:
        - Handlers run in priority order (desc), then registration order (asc).
        - A handler returning HookResult(block=True) sets outcome.blocked=True
          and stops further handlers with LOWER priority. Handlers with
          priority >= KERNEL_AUDIT still run (they are audit/observer handlers
          inside the registry; real Kernel safety checks run OUTSIDE the
          registry and are never affected by clear()/unregister()).
        - A handler returning HookResult(skip_action=True) sets
          outcome.skip_action; subsequent handlers still run.
        - Handler exceptions: fail_open -> log + continue; fail_closed ->
          record + abort the whole dispatch (return early).
        - Handler timeouts: treated like exceptions under their fail_policy.
        - Patch merging: FIRST-WINS BY PRIORITY. Higher-priority handlers run
          first and set fields; lower-priority handlers CANNOT override fields
          already set by a higher-priority handler. This ensures security/
          project-level extensions have final authority over ordinary ones.

        The caller (harness_core) is responsible for applying patches from
        the outcome to the actual agent loop state, and for running any
        real Kernel safety checks OUTSIDE this registry.
        """
        outcome = DispatchOutcome(event=event)
        entries = self.handlers_for(event)
        if not entries:
            return outcome

        for entry in entries:
            # If blocked by a higher-priority handler, skip remaining
            # lower-priority handlers. KERNEL_AUDIT handlers (priority >=
            # KERNEL_AUDIT) ALWAYS run so they can audit even after a block.
            if outcome.blocked and entry.priority < Priority.KERNEL_AUDIT:
                continue

            try:
                result = self._invoke_handler(entry, context)
            except HandlerTimeoutError as te:
                outcome.timeouts.append(te)
                if entry.fail_policy == FailPolicy.SAFETY:
                    self._log_safety_failure(entry, te)
                    outcome.blocked = True
                    outcome.block_reason = f"safety handler timeout: {te}"
                    return outcome
                self._log_normal_failure(entry, te)
                continue
            except HandlerError as he:
                outcome.errors.append(he)
                if entry.fail_policy == FailPolicy.SAFETY:
                    self._log_safety_failure(entry, he)
                    outcome.blocked = True
                    outcome.block_reason = f"safety handler error: {he}"
                    return outcome
                self._log_normal_failure(entry, he)
                continue

            if result is None:
                continue

            outcome.results.append(result)

            # Merge patches: FIRST-WINS BY PRIORITY.
            # Higher-priority handlers run first and lock fields; lower-priority
            # handlers cannot override fields already set.
            if result.context_patch:
                for k, v in result.context_patch.items():
                    if k not in outcome.context_patch:
                        outcome.context_patch[k] = v
            if result.tool_args_patch:
                for k, v in result.tool_args_patch.items():
                    if k not in outcome.tool_args_patch:
                        outcome.tool_args_patch[k] = v
            if result.tool_result_patch is not None and outcome.tool_result_patch is None:
                outcome.tool_result_patch = result.tool_result_patch
            if result.model_request_patch:
                for k, v in result.model_request_patch.items():
                    if k not in outcome.model_request_patch:
                        outcome.model_request_patch[k] = v
            if result.model_response_patch is not None and outcome.model_response_patch is None:
                outcome.model_response_patch = result.model_response_patch
            if result.skip_action:
                outcome.skip_action = True

            # Block: stop lower-priority handlers
            if result.block:
                outcome.blocked = True
                if result.reason:
                    outcome.block_reason = result.reason

        return outcome

    # ------------------------------------------------------------------
    # Handler invocation (sync + async + timeout)
    # ------------------------------------------------------------------

    def _invoke_handler(
        self, entry: _HandlerEntry, context: dict
    ) -> HookResult | None:
        handler = entry.handler
        if inspect.iscoroutinefunction(handler):
            return self._invoke_async(entry, context)
        return self._invoke_sync(entry, context)

    def _invoke_sync(
        self, entry: _HandlerEntry, context: dict
    ) -> HookResult | None:
        """Run a sync handler with a wait timeout via a worker thread.

        LIMITATION (important): Python has NO reliable way to interrupt a
        sync function from another thread. On timeout we abandon the worker
        thread (daemon=True) and raise HandlerTimeoutError, but the handler
        may CONTINUE running in the background and produce side-effects
        after the timeout. Its return value is discarded.

        Implications for extension authors:
        - Sync handlers MUST be fast and non-blocking (e.g. dict lookups,
          logging, simple validation). Do NOT do network I/O, file reads
          on large files, or long computations in sync handlers.
        - If a handler may block or produce delayed side-effects, use an
          async handler instead — asyncio.wait_for() can cancel the task.
        - If true isolation from untrusted/misbehaving handlers is needed,
          the handler must run in a subprocess, not a thread. This is out
          of scope for stage 1.

        The log message on timeout says "wait timed out, background
        execution may still continue", NOT "handler terminated".
        """
        result_box: dict[str, Any] = {}

        def worker():
            try:
                result_box["value"] = entry.handler(context)
            except Exception as e:  # noqa: BLE001
                result_box["error"] = e

        t = threading.Thread(
            target=worker,
            name=f"ext-{entry.extension_id}",
            daemon=True,
        )
        t.start()
        t.join(timeout=entry.timeout)

        if t.is_alive():
            # Worker still running past timeout. Abandon it.
            raise HandlerTimeoutError(
                extension_id=entry.extension_id,
                event=entry.event,
                timeout=entry.timeout,
            )

        if "error" in result_box:
            raise HandlerError(
                extension_id=entry.extension_id,
                event=entry.event,
                cause=result_box["error"],
            )
        return result_box.get("value")

    def _invoke_async(
        self, entry: _HandlerEntry, context: dict
    ) -> HookResult | None:
        """Run an async handler in a fresh event loop with timeout."""
        async def runner():
            return await asyncio.wait_for(
                entry.handler(context), timeout=entry.timeout
            )

        try:
            return asyncio.run(runner())
        except asyncio.TimeoutError:
            raise HandlerTimeoutError(
                extension_id=entry.extension_id,
                event=entry.event,
                timeout=entry.timeout,
            )
        except Exception as e:  # noqa: BLE001
            raise HandlerError(
                extension_id=entry.extension_id,
                event=entry.event,
                cause=e,
            ) from e

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_registration_order(self) -> int:
        with self._lock:
            n = self._registration_counter
            self._registration_counter += 1
            return n

    def _log_normal_failure(self, entry: _HandlerEntry, err: Exception) -> None:
        self._log.warning(
            "extension %s on %s failed (fail_open, skipped): %s",
            entry.extension_id, entry.event, err,
        )

    def _log_safety_failure(self, entry: _HandlerEntry, err: Exception) -> None:
        self._log.error(
            "SAFETY extension %s on %s failed (fail_closed, aborted): %s",
            entry.extension_id, entry.event, err,
        )


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

# A single default registry instance. harness_core will use this; tests can
# construct their own ExtensionRegistry() for isolation.
default_registry = ExtensionRegistry()


__all__ = [
    "ExtensionRegistry",
    "DispatchOutcome",
    "default_registry",
]
