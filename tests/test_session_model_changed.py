"""test_session_model_changed.py - Phase 3D-2 MODEL_CHANGED event tests.

Locks the after-change notification semantics:

  - MODEL_CHANGED fires when the effective model for the NEXT run truly
    changes (old_effective_alias != new_effective_alias)
  - /model deepseek from None(default claude):
      old_selection=None, new_selection='deepseek',
      old_effective_alias='claude', new_effective_alias='deepseek'
  - explicit claude from default claude -> NO event (effective unchanged)
  - deepseek -> deepseek -> NO event
  - unknown alias -> NO event, session unchanged
  - session is committed BEFORE the event fires (handler sees new alias)
  - notification only: handler failure/block does NOT roll back session
  - event does NOT build adapters / read credentials / touch current run
  - session A/B events isolated (no global state)
  - /model current / list do NOT fire the event
  - payload carries ModelSpec (no binding/adapter/key/ModelRuntimeContext)
"""

import os
import unittest

from agents.session import (
    SessionState,
    handle_model_command,
)
from agents.providers.model_spec import (
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
)
from agents.types.events import Event


class FakeExtensions:
    """Records MODEL_CHANGED emissions; simulates handler outcomes."""

    def __init__(self, handler=None):
        self.events = []
        self.handler = handler  # called with (event_name, context)

    def emit(self, event_name, context):
        self.events.append((event_name, context))
        if self.handler is not None:
            self.handler(event_name, context)
        from agents.extension_system import DispatchOutcome
        return DispatchOutcome(event=event_name)


def make_registry():
    reg = ModelRegistry()
    reg.register(ModelSpec(
        alias="claude", provider="anthropic", model_id="claude-x"))
    reg.register(ModelSpec(
        alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
    reg.register(ModelSpec(
        alias="qwen-openrouter", provider="openrouter",
        model_id="qwen/qwen-2.5-72b-instruct"))
    return reg


class ModelChangedFiringTests(unittest.TestCase):

    def setUp(self):
        self.reg = make_registry()

    def test_default_to_deepseek_fires_once(self):
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        self.assertEqual(len(exts.events), 1)
        name, ctx = exts.events[0]
        self.assertEqual(name, Event.MODEL_CHANGED)
        payload = ctx["payload"]
        self.assertIsNone(payload.old_selection)
        self.assertEqual(payload.new_selection, "deepseek")
        self.assertEqual(payload.old_effective_alias, "claude")
        self.assertEqual(payload.new_effective_alias, "deepseek")

    def test_payload_model_specs_correct(self):
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        payload = exts.events[0][1]["payload"]
        self.assertEqual(payload.old_model_spec.alias, "claude")
        self.assertEqual(payload.old_model_spec.provider, "anthropic")
        self.assertEqual(payload.new_model_spec.alias, "deepseek")
        self.assertEqual(payload.new_model_spec.provider, "deepseek")
        self.assertEqual(payload.new_model_spec.model_id, "deepseek-chat")

    def test_payload_reason_and_session_id(self):
        s = SessionState(session_id="sess-1")
        exts = FakeExtensions()
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        payload = exts.events[0][1]["payload"]
        self.assertEqual(payload.reason, "user_command")
        self.assertEqual(payload.session_id, "sess-1")

    def test_explicit_claude_no_event(self):
        # default claude -> explicit claude: effective unchanged, no event.
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model claude", s, self.reg, extensions=exts)
        self.assertEqual(s.model_alias, "claude")
        self.assertEqual(exts.events, [])

    def test_same_model_no_event(self):
        s = SessionState()
        s.model_alias = "deepseek"
        exts = FakeExtensions()
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        self.assertEqual(exts.events, [])

    def test_unknown_alias_no_event_and_unchanged(self):
        s = SessionState()
        s.model_alias = "deepseek"
        exts = FakeExtensions()
        with self.assertRaises(UnknownModelError):
            handle_model_command("/model nope", s, self.reg, extensions=exts)
        self.assertEqual(s.model_alias, "deepseek")
        self.assertEqual(exts.events, [])

    def test_current_and_list_no_event(self):
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model", s, self.reg, extensions=exts)
        handle_model_command("/model current", s, self.reg, extensions=exts)
        handle_model_command("/model list", s, self.reg, extensions=exts)
        self.assertEqual(exts.events, [])

    def test_no_extensions_no_crash(self):
        s = SessionState()
        out = handle_model_command("/model deepseek", s, self.reg)
        self.assertEqual(s.model_alias, "deepseek")
        self.assertIn("Model selected: deepseek", out)


class ModelChangedOrderingTests(unittest.TestCase):
    """Session commits BEFORE the event fires; handler sees new state."""

    def setUp(self):
        self.reg = make_registry()

    def test_handler_sees_committed_session(self):
        s = SessionState()
        seen = {}
        exts = FakeExtensions(handler=lambda name, ctx: seen.update(
            session_alias=ctx["payload"].session_id or "",
            new_selection=ctx["payload"].new_selection,
        ))
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        # Session was already committed when handler ran.
        self.assertEqual(s.model_alias, "deepseek")
        self.assertEqual(seen["new_selection"], "deepseek")

    def test_handler_failure_does_not_rollback_session(self):
        s = SessionState()

        def boom(name, ctx):
            raise RuntimeError("handler exploded")

        exts = FakeExtensions(handler=boom)
        # The command itself raises the handler exception (fail-open
        # behavior depends on registry); the KEY contract is that the
        # session stays committed regardless.
        try:
            handle_model_command("/model deepseek", s, self.reg,
                                 extensions=exts)
        except RuntimeError:
            pass
        self.assertEqual(s.model_alias, "deepseek")

    def test_event_does_not_build_adapter(self):
        # No ProviderBinding / adapter construction involved: the handler
        # only needs a ModelRegistry. If /model tried to build an adapter
        # it would need a binding - we pass only a registry.
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model qwen-openrouter", s, self.reg,
                             extensions=exts)
        self.assertEqual(s.model_alias, "qwen-openrouter")

    def test_event_does_not_read_credential(self):
        saved = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            s = SessionState()
            exts = FakeExtensions()
            handle_model_command("/model qwen-openrouter", s, self.reg,
                                 extensions=exts)
            self.assertEqual(s.model_alias, "qwen-openrouter")
        finally:
            if saved is not None:
                os.environ["OPENROUTER_API_KEY"] = saved


class SessionIsolationTests(unittest.TestCase):

    def setUp(self):
        self.reg = make_registry()

    def test_session_a_b_events_isolated(self):
        a = SessionState(session_id="A")
        b = SessionState(session_id="B")
        exts_a = FakeExtensions()
        exts_b = FakeExtensions()
        handle_model_command("/model deepseek", a, self.reg, extensions=exts_a)
        handle_model_command("/model qwen-openrouter", b, self.reg,
                             extensions=exts_b)
        # A's event is about A -> deepseek.
        pa = exts_a.events[0][1]["payload"]
        self.assertEqual(pa.session_id, "A")
        self.assertEqual(pa.new_effective_alias, "deepseek")
        # B's event is about B -> qwen.
        pb = exts_b.events[0][1]["payload"]
        self.assertEqual(pb.session_id, "B")
        self.assertEqual(pb.new_effective_alias, "qwen-openrouter")
        # No global CURRENT_MODEL coupling: a and b selections differ.
        self.assertEqual(a.model_alias, "deepseek")
        self.assertEqual(b.model_alias, "qwen-openrouter")


class PayloadShapeTests(unittest.TestCase):
    """Payload carries ModelSpec; no binding/adapter/key/context."""

    def setUp(self):
        self.reg = make_registry()

    def test_payload_has_no_adapter_or_binding(self):
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        payload = exts.events[0][1]["payload"]
        self.assertNotIn("adapter", payload.__dict__)
        self.assertNotIn("binding", payload.__dict__)
        self.assertNotIn("provider_binding", payload.__dict__)
        self.assertNotIn("api_key", payload.__dict__)
        self.assertNotIn("model_runtime", payload.__dict__)

    def test_payload_fields_exact(self):
        s = SessionState()
        exts = FakeExtensions()
        handle_model_command("/model deepseek", s, self.reg, extensions=exts)
        payload = exts.events[0][1]["payload"]
        self.assertEqual(
            set(payload.__dict__),
            {
                "session_id", "old_selection", "new_selection",
                "old_effective_alias", "new_effective_alias",
                "old_model_spec", "new_model_spec", "reason",
            },
        )


class ModelChangedEventContractTests(unittest.TestCase):
    """MODEL_CHANGED is in the Event enum / ALL_HOOKS."""

    def test_event_registered(self):
        self.assertTrue(hasattr(Event, "MODEL_CHANGED"))
        self.assertIn(Event.MODEL_CHANGED, Event.__dict__.values())
        from agents.types.events import ALL_HOOKS
        self.assertIn(Event.MODEL_CHANGED, ALL_HOOKS)

    def test_event_name(self):
        self.assertEqual(Event.MODEL_CHANGED, "model_changed")


if __name__ == "__main__":
    unittest.main()
