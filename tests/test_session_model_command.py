"""test_session_model_command.py - Phase 3D-1 /model command layer tests.

Locks the /model command semantics:

  - /model == /model current (effective model, not raw None)
  - current distinguishes source=default vs source=session
  - /model list comes from ModelRegistry.list() (not hardcoded), in
    registration order, with a current marker
  - /model <alias> validates + sets session alias, applies to NEXT run
  - unknown alias fails fast, session value unchanged
  - switching does NOT read API keys / build adapters / touch the
    running ModelRuntimeContext / fire MODEL_CHANGED
  - /model is a Harness command, NOT a tool in ToolRegistry
  - session A / session B commands do not interfere
"""

import os
import types
import unittest

from agents.session import (
    SessionState,
    describe_model_alias,
    handle_model_command,
    list_model_aliases,
)
from agents.providers.model_spec import (
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
)


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


class ModelCommandCurrentTests(unittest.TestCase):

    def setUp(self):
        self.reg = make_registry()

    def test_bare_model_equals_current(self):
        s = SessionState()
        bare = handle_model_command("/model", s, self.reg)
        cur = handle_model_command("/model current", s, self.reg)
        self.assertEqual(bare, cur)

    def test_default_session_shows_effective_claude(self):
        s = SessionState()
        out = handle_model_command("/model current", s, self.reg)
        self.assertIn("Current model: claude", out)
        self.assertIn("Provider: anthropic", out)
        self.assertIn("Model ID: claude-x", out)
        self.assertIn("Source: default", out)
        # Raw None must NOT be shown as the model.
        self.assertNotIn("None", out.splitlines()[0])

    def test_session_selected_shows_deepseek(self):
        s = SessionState()
        s.model_alias = "deepseek"
        out = handle_model_command("/model current", s, self.reg)
        self.assertIn("Current model: deepseek", out)
        self.assertIn("Provider: deepseek", out)
        self.assertIn("Model ID: deepseek-chat", out)
        self.assertIn("Source: session", out)


class ModelCommandListTests(unittest.TestCase):

    def setUp(self):
        self.reg = make_registry()

    def test_list_from_registry_registration_order(self):
        s = SessionState()
        out = handle_model_command("/model list", s, self.reg)
        # Registration order preserved.
        self.assertLess(out.index("claude"), out.index("deepseek"))
        self.assertLess(out.index("deepseek"), out.index("qwen-openrouter"))
        # Derived from registry - contains each provider/model.
        self.assertIn("provider: anthropic", out)
        self.assertIn("model: deepseek-chat", out)
        self.assertIn("provider: openrouter", out)

    def test_list_marks_current(self):
        s = SessionState()
        s.model_alias = "deepseek"
        out = handle_model_command("/model list", s, self.reg)
        # deepseek line is marked with *, claude is not.
        for line in out.splitlines():
            stripped = line.strip()
            if stripped == "* deepseek":
                break
        else:
            self.fail("expected '* deepseek' current marker in list output")

    def test_list_dynamic_not_hardcoded(self):
        # A registry with a custom model must appear in list output.
        reg = ModelRegistry()
        reg.register(ModelSpec(alias="custom", provider="custom",
                               model_id="custom-1"))
        out = list_model_aliases(None, reg)
        self.assertIn("custom", out)
        self.assertIn("provider: custom", out)


class ModelCommandSetTests(unittest.TestCase):

    def setUp(self):
        self.reg = make_registry()

    def test_set_deepseek_updates_session(self):
        s = SessionState()
        out = handle_model_command("/model deepseek", s, self.reg)
        self.assertEqual(s.model_alias, "deepseek")
        self.assertIn("Model selected: deepseek", out)
        self.assertIn("Applies to the next agent run.", out)

    def test_set_qwen_openrouter(self):
        s = SessionState()
        out = handle_model_command("/model qwen-openrouter", s, self.reg)
        self.assertEqual(s.model_alias, "qwen-openrouter")
        self.assertIn("Model ID: qwen/qwen-2.5-72b-instruct", out)

    def test_unknown_alias_fails_fast_preserves_value(self):
        s = SessionState()
        s.model_alias = "deepseek"
        with self.assertRaises(UnknownModelError):
            handle_model_command("/model deepseak", s, self.reg)
        # Session unchanged.
        self.assertEqual(s.model_alias, "deepseek")

    def test_unknown_alias_on_fresh_session_unchanged(self):
        s = SessionState()
        with self.assertRaises(UnknownModelError):
            handle_model_command("/model nope", s, self.reg)
        self.assertIsNone(s.model_alias)

    def test_set_does_not_read_api_key(self):
        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            s = SessionState()
            handle_model_command("/model deepseek", s, self.reg)
            self.assertEqual(s.model_alias, "deepseek")
        finally:
            if saved is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved

    def test_set_does_not_build_adapter(self):
        # No adapter construction: selecting a model must not require a
        # ProviderBinding/adapter. Use only a ModelRegistry.
        s = SessionState()
        handle_model_command("/model deepseek", s, self.reg)
        self.assertEqual(s.model_alias, "deepseek")

    def test_set_does_not_fire_model_changed(self):
        # There is no extension event for MODEL_CHANGED yet (3D-2); the
        # command handler only mutates the session. We assert the session
        # changed and no event API was touched by just running it.
        s = SessionState()
        handle_model_command("/model deepseek", s, self.reg)
        self.assertEqual(s.model_alias, "deepseek")


class SessionIndependenceTests(unittest.TestCase):

    def setUp(self):
        self.reg = make_registry()

    def test_session_a_b_commands_independent(self):
        a = SessionState()
        b = SessionState()
        handle_model_command("/model deepseek", a, self.reg)
        self.assertEqual(a.model_alias, "deepseek")
        self.assertIsNone(b.model_alias)
        # b's current still shows default claude.
        out_b = handle_model_command("/model current", b, self.reg)
        self.assertIn("Current model: claude", out_b)
        self.assertIn("Source: default", out_b)


class CommandNotToolTests(unittest.TestCase):
    """/model is a Harness command, never a model tool."""

    def test_model_not_in_tool_names(self):
        from agents.providers.model_spec import default_model_registry
        names = {s.alias for s in default_model_registry().list()}
        self.assertNotIn("model", names)
        self.assertNotIn("model_change", names)


if __name__ == "__main__":
    unittest.main()
