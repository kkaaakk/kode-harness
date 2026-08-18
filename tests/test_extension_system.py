"""
test_extension_system.py - Stage 1 tests for the extension system.

Covers:
- 8 hook points: input/output contracts
- priority + stable ordering (priority DESC, registration ASC)
- block stops non-kernel handlers; kernel safety handlers still run
- fail-open for normal handlers; fail-closed for safety handlers
- handler timeout (sync + async)
- Extension ID and error logging
- sync + async handler handling
- registry install() with class encapsulation
- empty registry is a no-op (backward compat with pre-stage-1 harness)
"""

from __future__ import annotations

import asyncio
import time
import unittest

from agents.extension_system import ExtensionRegistry, DispatchOutcome
from agents.types.events import (
    ALL_HOOKS,
    Event,
    FailPolicy,
    HandlerError,
    HandlerTimeoutError,
    HookResult,
    Priority,
)


# ---------------------------------------------------------------------------
# Empty registry is a no-op (backward compat)
# ---------------------------------------------------------------------------

class EmptyRegistryNoOpTests(unittest.TestCase):
    """Default (empty) registry must not alter agent behavior."""

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_all_8_hooks_emit_returns_empty_outcome(self):
        for event in ALL_HOOKS:
            outcome = self.reg.emit(event, {"event": event})
            self.assertIsInstance(outcome, DispatchOutcome)
            self.assertEqual(outcome.event, event)
            self.assertFalse(outcome.blocked)
            self.assertFalse(outcome.skip_action)
            self.assertEqual(outcome.results, [])
            self.assertEqual(outcome.context_patch, {})
            self.assertEqual(outcome.tool_args_patch, {})
            self.assertIsNone(outcome.tool_result_patch)
            self.assertEqual(outcome.model_request_patch, {})
            self.assertIsNone(outcome.model_response_patch)
            self.assertEqual(outcome.errors, [])
            self.assertEqual(outcome.timeouts, [])

    def test_unknown_event_rejected_at_registration(self):
        with self.assertRaises(ValueError):
            self.reg.on("not_a_real_event", lambda ctx: None)


# ---------------------------------------------------------------------------
# Priority and stable ordering
# ---------------------------------------------------------------------------

class PriorityOrderingTests(unittest.TestCase):
    """Higher priority runs first; same priority preserves registration order."""

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_priority_desc_runs_higher_first(self):
        calls = []
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: calls.append("normal"),
                    priority=Priority.NORMAL, extension_id="normal")
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: calls.append("audit"),
                    priority=Priority.KERNEL_AUDIT, extension_id="audit")
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: calls.append("observer"),
                    priority=Priority.OBSERVER, extension_id="observer")

        self.reg.emit(Event.BEFORE_TOOL_CALL, {"event": Event.BEFORE_TOOL_CALL})
        self.assertEqual(calls, ["audit", "normal", "observer"])

    def test_same_priority_preserves_registration_order(self):
        calls = []
        self.reg.on(Event.TURN_END, lambda ctx: calls.append("first"),
                    priority=Priority.NORMAL, extension_id="first")
        self.reg.on(Event.TURN_END, lambda ctx: calls.append("second"),
                    priority=Priority.NORMAL, extension_id="second")
        self.reg.on(Event.TURN_END, lambda ctx: calls.append("third"),
                    priority=Priority.NORMAL, extension_id="third")

        self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(calls, ["first", "second", "third"])

    def test_handlers_for_returns_sorted_snapshot(self):
        self.reg.on(Event.AGENT_END, lambda ctx: None,
                    priority=Priority.NORMAL, extension_id="a")
        self.reg.on(Event.AGENT_END, lambda ctx: None,
                    priority=Priority.KERNEL_AUDIT, extension_id="b")
        entries = self.reg.handlers_for(Event.AGENT_END)
        self.assertEqual(len(entries), 2)
        # KERNEL_AUDIT (1000) first, NORMAL (100) second
        self.assertEqual(entries[0].extension_id, "b")
        self.assertEqual(entries[1].extension_id, "a")


# ---------------------------------------------------------------------------
# Block semantics
# ---------------------------------------------------------------------------

class BlockSemanticsTests(unittest.TestCase):
    """block=True stops lower-priority handlers; KERNEL_AUDIT handlers still run.

    NOTE: Real Kernel safety checks (dangerous-command blocking) live OUTSIDE
    the ExtensionRegistry, in base_tools.run_bash. They are NOT registered as
    extensions and cannot be removed by clear()/unregister(). The tests below
    verify the Registry's INTERNAL priority semantics, not the production
    security boundary.
    """

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_block_stops_subsequent_lower_priority_handlers(self):
        calls = []
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: calls.append("first"),
                    priority=Priority.NORMAL, extension_id="first")
        self.reg.on(Event.BEFORE_TOOL_CALL,
                    lambda ctx: HookResult(block=True, reason="vetoed"),
                    priority=Priority.NORMAL, extension_id="blocker")
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: calls.append("third"),
                    priority=Priority.NORMAL, extension_id="third")

        outcome = self.reg.emit(Event.BEFORE_TOOL_CALL,
                                {"event": Event.BEFORE_TOOL_CALL})
        self.assertEqual(calls, ["first"])
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.block_reason, "vetoed")

    def test_block_does_not_stop_kernel_audit_handlers(self):
        """KERNEL_AUDIT handlers (priority >= KERNEL_AUDIT) ALWAYS run, even
        after a lower-priority handler blocked. This is for audit/observer
        handlers INSIDE the registry. Real Kernel safety checks run OUTSIDE
        the registry and are never affected by clear()/unregister()."""
        calls = []
        self.reg.on(Event.BEFORE_TOOL_CALL,
                    lambda ctx: HookResult(block=True, reason="extension veto"),
                    priority=Priority.NORMAL, extension_id="blocker")
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: calls.append("audit"),
                    priority=Priority.KERNEL_AUDIT, extension_id="audit")

        outcome = self.reg.emit(Event.BEFORE_TOOL_CALL,
                                {"event": Event.BEFORE_TOOL_CALL})
        self.assertEqual(calls, ["audit"])
        self.assertTrue(outcome.blocked)

    def test_block_reason_carried_through(self):
        self.reg.on(Event.BEFORE_MODEL_REQUEST,
                    lambda ctx: HookResult(block=True, reason="rate limited"),
                    priority=Priority.NORMAL, extension_id="rl")
        outcome = self.reg.emit(Event.BEFORE_MODEL_REQUEST,
                                {"event": Event.BEFORE_MODEL_REQUEST})
        self.assertEqual(outcome.block_reason, "rate limited")

    def test_clear_does_not_affect_production_safety(self):
        """Regression: clear() only removes Registry-internal handlers.
        Production safety checks (in base_tools.run_bash) are NOT in the
        registry, so clear() cannot disable them. This test just verifies
        clear() empties the registry; the actual safety check is tested
        in test_security.py."""
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: None,
                    priority=Priority.KERNEL_AUDIT, extension_id="audit")
        self.assertEqual(len(self.reg.handlers_for(Event.BEFORE_TOOL_CALL)), 1)
        self.reg.clear()
        self.assertEqual(len(self.reg.handlers_for(Event.BEFORE_TOOL_CALL)), 0)
        # Production safety (base_tools.run_bash _is_dangerous) is unaffected
        # because it lives in a different module entirely.


# ---------------------------------------------------------------------------
# Patches accumulate
# ---------------------------------------------------------------------------

class PatchAccumulationTests(unittest.TestCase):
    """Patches merge with FIRST-WINS BY PRIORITY semantics.

    Higher-priority handlers run first and set fields; lower-priority handlers
    CANNOT override fields already set by a higher-priority handler. Different
    keys from different handlers accumulate.
    """

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_context_patch_higher_priority_wins_same_key(self):
        # Both set the same key "k". Higher priority (PROJECT) runs first and
        # locks the value; lower priority (NORMAL) cannot override.
        self.reg.on(Event.TURN_END,
                    lambda ctx: HookResult(context_patch={"k": "project"}),
                    priority=Priority.PROJECT, extension_id="proj")
        self.reg.on(Event.TURN_END,
                    lambda ctx: HookResult(context_patch={"k": "normal"}),
                    priority=Priority.NORMAL, extension_id="norm")
        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(outcome.context_patch, {"k": "project"})

    def test_context_patch_different_keys_accumulate(self):
        self.reg.on(Event.TURN_END,
                    lambda ctx: HookResult(context_patch={"a": 1}),
                    priority=Priority.PROJECT, extension_id="proj")
        self.reg.on(Event.TURN_END,
                    lambda ctx: HookResult(context_patch={"b": 2}),
                    priority=Priority.NORMAL, extension_id="norm")
        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(outcome.context_patch, {"a": 1, "b": 2})

    def test_tool_args_patch_first_wins_by_priority(self):
        # Higher priority sets "path" first; lower priority cannot override.
        self.reg.on(Event.BEFORE_TOOL_CALL,
                    lambda ctx: HookResult(tool_args_patch={"path": "secure", "limit": 100}),
                    priority=Priority.PROJECT, extension_id="proj")
        self.reg.on(Event.BEFORE_TOOL_CALL,
                    lambda ctx: HookResult(tool_args_patch={"path": "evil", "extra": True}),
                    priority=Priority.NORMAL, extension_id="norm")
        outcome = self.reg.emit(Event.BEFORE_TOOL_CALL,
                                {"event": Event.BEFORE_TOOL_CALL})
        # "path" from PROJECT wins; "limit" and "extra" accumulate.
        self.assertEqual(outcome.tool_args_patch,
                         {"path": "secure", "limit": 100, "extra": True})

    def test_model_request_patch_first_wins_by_priority(self):
        self.reg.on(Event.BEFORE_MODEL_REQUEST,
                    lambda ctx: HookResult(model_request_patch={"max_tokens": 1000, "temperature": 0.1}),
                    priority=Priority.PROJECT, extension_id="proj")
        self.reg.on(Event.BEFORE_MODEL_REQUEST,
                    lambda ctx: HookResult(model_request_patch={"max_tokens": 9999, "top_p": 0.9}),
                    priority=Priority.NORMAL, extension_id="norm")
        outcome = self.reg.emit(Event.BEFORE_MODEL_REQUEST,
                                {"event": Event.BEFORE_MODEL_REQUEST})
        # max_tokens from PROJECT wins; temperature + top_p accumulate.
        self.assertEqual(outcome.model_request_patch,
                         {"max_tokens": 1000, "temperature": 0.1, "top_p": 0.9})

    def test_tool_result_patch_higher_priority_wins(self):
        self.reg.on(Event.AFTER_TOOL_RESULT,
                    lambda ctx: HookResult(tool_result_patch={"content": "project"}),
                    priority=Priority.PROJECT, extension_id="proj")
        self.reg.on(Event.AFTER_TOOL_RESULT,
                    lambda ctx: HookResult(tool_result_patch={"content": "normal"}),
                    priority=Priority.NORMAL, extension_id="norm")
        outcome = self.reg.emit(Event.AFTER_TOOL_RESULT,
                                {"event": Event.AFTER_TOOL_RESULT})
        self.assertEqual(outcome.tool_result_patch, {"content": "project"})

    def test_model_response_patch_higher_priority_wins(self):
        self.reg.on(Event.AFTER_MODEL_RESPONSE,
                    lambda ctx: HookResult(model_response_patch="project_resp"),
                    priority=Priority.PROJECT, extension_id="proj")
        self.reg.on(Event.AFTER_MODEL_RESPONSE,
                    lambda ctx: HookResult(model_response_patch="normal_resp"),
                    priority=Priority.NORMAL, extension_id="norm")
        outcome = self.reg.emit(Event.AFTER_MODEL_RESPONSE,
                                {"event": Event.AFTER_MODEL_RESPONSE})
        self.assertEqual(outcome.model_response_patch, "project_resp")

    def test_skip_action_sets_flag(self):
        self.reg.on(Event.BEFORE_TOOL_CALL,
                    lambda ctx: HookResult(skip_action=True),
                    priority=Priority.NORMAL, extension_id="a")
        outcome = self.reg.emit(Event.BEFORE_TOOL_CALL,
                                {"event": Event.BEFORE_TOOL_CALL})
        self.assertTrue(outcome.skip_action)


# ---------------------------------------------------------------------------
# Fail policies
# ---------------------------------------------------------------------------

class FailPolicyTests(unittest.TestCase):
    """Normal handlers: fail-open (skip + log). Safety handlers: fail-closed (abort)."""

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_normal_handler_exception_fail_open(self):
        def boom(ctx):
            raise RuntimeError("boom")
        self.reg.on(Event.TURN_END, boom,
                    priority=Priority.NORMAL,
                    fail_policy=FailPolicy.NORMAL,
                    extension_id="boom_normal")

        # Should NOT raise; should record error and continue.
        self.reg.on(Event.TURN_END, lambda ctx: HookResult(context_patch={"after": True}),
                    priority=Priority.NORMAL, extension_id="after")

        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(len(outcome.errors), 1)
        self.assertEqual(outcome.errors[0].extension_id, "boom_normal")
        self.assertEqual(outcome.errors[0].event, Event.TURN_END)
        self.assertIsInstance(outcome.errors[0].cause, RuntimeError)
        # Subsequent handler still ran
        self.assertEqual(outcome.context_patch, {"after": True})

    def test_safety_handler_exception_fail_closed(self):
        def boom(ctx):
            raise RuntimeError("safety boom")
        self.reg.on(Event.BEFORE_TOOL_CALL, boom,
                    priority=Priority.KERNEL_AUDIT,
                    fail_policy=FailPolicy.SAFETY,
                    extension_id="boom_safety")
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: HookResult(context_patch={"should_not_run": True}),
                    priority=Priority.NORMAL, extension_id="after")

        outcome = self.reg.emit(Event.BEFORE_TOOL_CALL,
                                {"event": Event.BEFORE_TOOL_CALL})
        self.assertEqual(len(outcome.errors), 1)
        self.assertTrue(outcome.blocked)
        self.assertIn("safety handler", outcome.block_reason or "")
        # Subsequent non-kernel handler did NOT run
        self.assertEqual(outcome.context_patch, {})


# ---------------------------------------------------------------------------
# Handler timeout (sync + async)
# ---------------------------------------------------------------------------

class TimeoutTests(unittest.TestCase):
    """Handlers must respect their timeout."""

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_sync_handler_timeout_recorded(self):
        def slow(ctx):
            time.sleep(5)
            return HookResult()
        self.reg.on(Event.TURN_END, slow,
                    priority=Priority.NORMAL,
                    timeout=0.5,
                    extension_id="slow_sync")

        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(len(outcome.timeouts), 1)
        self.assertEqual(outcome.timeouts[0].extension_id, "slow_sync")
        self.assertEqual(outcome.timeouts[0].event, Event.TURN_END)
        self.assertAlmostEqual(outcome.timeouts[0].timeout, 0.5)

    def test_sync_handler_timeout_fail_open_normal(self):
        def slow(ctx):
            time.sleep(5)
        self.reg.on(Event.TURN_END, slow,
                    priority=Priority.NORMAL,
                    fail_policy=FailPolicy.NORMAL,
                    timeout=0.3,
                    extension_id="slow")
        self.reg.on(Event.TURN_END, lambda ctx: HookResult(context_patch={"after": True}),
                    priority=Priority.NORMAL, extension_id="after")

        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(len(outcome.timeouts), 1)
        self.assertEqual(outcome.context_patch, {"after": True})

    def test_sync_handler_timeout_fail_closed_safety(self):
        def slow(ctx):
            time.sleep(5)
        self.reg.on(Event.BEFORE_TOOL_CALL, slow,
                    priority=Priority.KERNEL_AUDIT,
                    fail_policy=FailPolicy.SAFETY,
                    timeout=0.3,
                    extension_id="slow_safety")

        outcome = self.reg.emit(Event.BEFORE_TOOL_CALL,
                                {"event": Event.BEFORE_TOOL_CALL})
        self.assertEqual(len(outcome.timeouts), 1)
        self.assertTrue(outcome.blocked)
        self.assertIn("safety handler timeout", outcome.block_reason or "")

    def test_async_handler_timeout(self):
        async def slow(ctx):
            await asyncio.sleep(5)
            return HookResult()
        self.reg.on(Event.TURN_END, slow,
                    priority=Priority.NORMAL,
                    timeout=0.3,
                    extension_id="slow_async")

        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(len(outcome.timeouts), 1)
        self.assertEqual(outcome.timeouts[0].extension_id, "slow_async")

    def test_async_handler_returns_result(self):
        async def good(ctx):
            return HookResult(context_patch={"async_ran": True})
        self.reg.on(Event.TURN_END, good,
                    priority=Priority.NORMAL, extension_id="good_async")

        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(outcome.context_patch, {"async_ran": True})


# ---------------------------------------------------------------------------
# install() class encapsulation
# ---------------------------------------------------------------------------

class InstallClassTests(unittest.TestCase):
    """registry.install(obj) calls obj.register(registry). No base class needed."""

    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_install_calls_register(self):
        calls = []

        class MyExt:
            def register(self, registry):
                registry.on(Event.TURN_END,
                            lambda ctx: calls.append("ran"),
                            priority=Priority.NORMAL,
                            extension_id="my_ext")

        self.reg.install(MyExt())
        self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(calls, ["ran"])

    def test_install_rejects_object_without_register(self):
        with self.assertRaises(TypeError):
            self.reg.install(object())


# ---------------------------------------------------------------------------
# unregister / clear
# ---------------------------------------------------------------------------

class UnregisterTests(unittest.TestCase):
    def setUp(self):
        self.reg = ExtensionRegistry()

    def test_unregister_removes_all_handlers_for_id(self):
        self.reg.on(Event.TURN_END, lambda ctx: None,
                    extension_id="ext_a")
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: None,
                    extension_id="ext_a")
        self.reg.on(Event.TURN_END, lambda ctx: None,
                    extension_id="ext_b")

        removed = self.reg.unregister("ext_a")
        self.assertEqual(removed, 2)
        self.assertEqual(len(self.reg.handlers_for(Event.TURN_END)), 1)
        self.assertEqual(len(self.reg.handlers_for(Event.BEFORE_TOOL_CALL)), 0)

    def test_clear_removes_all(self):
        self.reg.on(Event.TURN_END, lambda ctx: None)
        self.reg.on(Event.BEFORE_TOOL_CALL, lambda ctx: None)
        self.reg.clear()
        for event in ALL_HOOKS:
            self.assertEqual(len(self.reg.handlers_for(event)), 0)


# ---------------------------------------------------------------------------
# Extension ID auto-generation
# ---------------------------------------------------------------------------

class ExtensionIdTests(unittest.TestCase):
    def test_auto_generated_id_when_none(self):
        reg = ExtensionRegistry()
        ext_id = reg.on(Event.TURN_END, lambda ctx: None)
        self.assertTrue(ext_id.startswith("ext_"))
        entries = reg.handlers_for(Event.TURN_END)
        self.assertEqual(entries[0].extension_id, ext_id)

    def test_explicit_id_preserved(self):
        reg = ExtensionRegistry()
        ext_id = reg.on(Event.TURN_END, lambda ctx: None,
                        extension_id="my_explicit_id")
        self.assertEqual(ext_id, "my_explicit_id")


# ---------------------------------------------------------------------------
# Hook input contract (events receive the context dict)
# ---------------------------------------------------------------------------

class HookInputContractTests(unittest.TestCase):
    """Each hook receives a context dict with event-specific fields."""

    def setUp(self):
        self.reg = ExtensionRegistry()
        self.received = []

    def test_handler_receives_context_dict(self):
        self.reg.on(Event.BEFORE_TOOL_CALL,
                    lambda ctx: self.received.append(ctx),
                    extension_id="recorder")
        ctx = {
            "event": Event.BEFORE_TOOL_CALL,
            "tool_name": "bash",
            "tool_use_id": "t1",
            "tool_args": {"command": "ls"},
            "actor": "lead",
        }
        self.reg.emit(Event.BEFORE_TOOL_CALL, ctx)
        self.assertEqual(self.received[0], ctx)

    def test_handler_returning_none_is_noop(self):
        self.reg.on(Event.TURN_END, lambda ctx: None, extension_id="noop")
        outcome = self.reg.emit(Event.TURN_END, {"event": Event.TURN_END})
        self.assertEqual(outcome.results, [])
        self.assertFalse(outcome.blocked)


if __name__ == "__main__":
    unittest.main()
