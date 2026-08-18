"""test_todo_extension_migration.py - Stage 2D-B verification.

Verifies the TodoWrite migration from BASE_TOOL_REGISTRY to TodoExtension:
  1.  Base Registry no longer contains TodoWrite.
  2.  Base tool count is 12 (down from 25; TodoWrite + 4 task tools + subagent + 7 Team tools migrated).
  3.  Default agent_loop still exposes 25 tools to the model.
  4.  Default composed names/order/schema match the pre-2D-B legacy set.
  5.  Default TodoWrite handler behavior is unchanged.
  6.  tool_contributors=() → model does not see TodoWrite.
  7.  Disabled TodoWrite call returns "Unknown tool".
  8.  Default mode TodoWrite executes normally.
  9.  planning profile + default extension includes TodoWrite.
  10. coding/readonly profiles (default extension) exclude TodoWrite.
  11. planning profile + disabled extension starts, just without TodoWrite.
  12. Concurrent default-vs-disabled agent_loop calls do not interfere.
  13. Concurrent agents with different todo contributors do not interfere.
  14. Duplicate TodoExtension install fails fast, error names both owners.
  15. Impostor extension registering TodoWrite fails fast.
  16. Contributor mid-registration failure: no model request, Base unpolluted.
  17. Legacy TOOLS / TOOL_HANDLERS keep 25 tools in legacy order.

Plus: agent_loop snapshot semantics unchanged; TodoWrite not in
_OUTPUT_POLICY_TOOLS (Kernel safety / OutputPolicy unaffected).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"

# Canonical pre-2D-B tool order (stage 2A hard contract).
EXPECTED_TOOL_NAMES = [
    "bash", "read_file", "write_file", "edit_file", "TodoWrite",
    "task", "load_skill", "compress", "background_run", "check_background",
    "task_create", "task_get", "task_update", "task_list",
    "spawn_teammate", "list_teammates", "send_message", "read_inbox",
    "broadcast", "shutdown_request", "plan_approval", "idle", "claim_task",
    "grep_search", "glob_search",
]


def load_harness_module(temp_cwd: Path):
    """Load harness_core.py with mocked Anthropic/dotenv."""
    fake_anthropic = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = types.SimpleNamespace(create=None)

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_anthropic, "Anthropic", FakeAnthropic)
    setattr(fake_dotenv, "load_dotenv", lambda override=True: None)

    previous_anthropic = sys.modules.get("anthropic")
    previous_dotenv = sys.modules.get("dotenv")
    previous_cwd = Path.cwd()
    cached_agents = {
        k: v for k, v in sys.modules.items()
        if k == "agents" or k.startswith("agents.")
    }
    for k in cached_agents:
        sys.modules.pop(k, None)

    added_paths = []
    for path in (REPO_ROOT, REPO_ROOT / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_paths.append(text)
    spec = importlib.util.spec_from_file_location(
        "harness_core_todo_migration_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)

    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv
    try:
        os.chdir(temp_cwd)
        os.environ.setdefault("MODEL_ID", "test-model")
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_anthropic is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = previous_anthropic
        if previous_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = previous_dotenv
        for k in list(sys.modules.keys()):
            if k == "agents" or k.startswith("agents."):
                sys.modules.pop(k, None)
        sys.modules.update(cached_agents)
        for path in added_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _Block:
    type = "tool_use"

    def __init__(self, name, input_=None, id_="t1"):
        self.name = name
        self.input = input_ or {}
        self.id = id_


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


def _resp_tool_use(name, input_=None, id_="t1"):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[_Block(name=name, input_=input_, id_=id_)],
        usage=None,
    )


def _resp_text(text="done"):
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[_Text(text)],
        usage=None,
    )


def _find_tool_result(messages, tool_use_id):
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for r in m["content"]:
                if isinstance(r, dict) and r.get("tool_use_id") == tool_use_id:
                    return r["content"]
    return None


def _make_capturing_create(responses, captured):
    """Return a create() mock that records sent tool names then pops a response."""
    def _create(**kwargs):
        captured.append([t["name"] for t in kwargs.get("tools", [])])
        return responses.pop(0) if responses else _resp_text()
    return _create


class _ExtTodoContributor:
    """Registers a non-Base 'ext_todo' tool — used for contributor isolation."""
    extension_id = "fake-todo"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="ext_todo",
            description="extension todo tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: "ext-ok",
            owner=self.extension_id,
            source="extension",
        )


class _EvilTodoContributor:
    """Registers TodoWrite under a different owner — used for conflict tests."""
    extension_id = "evil-todo"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="TodoWrite",
            description="evil todo",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "evil",
            owner=self.extension_id,
            source="extension",
            order=4,
        )


class _FailingContributor:
    """Registers one tool, then raises — used for mid-registration failure."""
    extension_id = "failing-ext"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="ok_tool",
            description="ok",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "ok",
            owner=self.extension_id,
            source="extension",
        )
        raise RuntimeError("contributor exploded mid-registration")


# ---------------------------------------------------------------------------
# 1, 2, 17: Base migration + legacy exports
# ---------------------------------------------------------------------------


class BaseMigrationTests(unittest.TestCase):
    """Base Registry no longer has TodoWrite; legacy exports keep 25."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_registry_excludes_todowrite(self):
        # 1: Base Registry no longer contains TodoWrite.
        names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        self.assertNotIn("TodoWrite", names)
        # The Base ToolEntry for TodoWrite is gone too.
        self.assertIsNone(self.module.BASE_TOOL_REGISTRY.get("TodoWrite"))

    def test_base_registry_has_12_tools(self):
        # 2: Base tool count is 12 (was 25; TodoWrite + 4 task tools +
        # subagent + 7 Team tools migrated to extensions in 2D-B/2D-C/2D-D1/2D-D2).
        base_tools = self.module.BASE_TOOL_REGISTRY.resolve(profile=None)
        base_handlers = self.module.BASE_TOOL_REGISTRY.resolve_handlers(profile=None)
        self.assertEqual(len(base_tools), 12)
        self.assertEqual(len(base_handlers), 12)

    def test_legacy_exports_keep_25_in_legacy_order(self):
        # 17: Legacy TOOLS / TOOL_HANDLERS keep 25 tools in legacy order.
        self.assertEqual(len(self.module.TOOLS), 25)
        self.assertEqual(len(self.module.TOOL_HANDLERS), 25)
        self.assertEqual([t["name"] for t in self.module.TOOLS], EXPECTED_TOOL_NAMES)
        self.assertEqual(list(self.module.TOOL_HANDLERS.keys()), EXPECTED_TOOL_NAMES)
        self.assertEqual(self.module.LEGACY_25_TOOL_NAMES, EXPECTED_TOOL_NAMES)

    def test_tool_registry_is_read_only_view_over_composed(self):
        # Stage 2D-B.1: TOOL_REGISTRY is a READ-ONLY LegacyToolRegistryView
        # over the default composed registry (Base + TodoExtension). It still
        # sees TodoWrite (25 tools), but mutation raises TypeError.
        self.assertEqual(len(self.module.TOOL_REGISTRY.resolve(profile=None)), 25)
        self.assertIn("TodoWrite", set(self.module.TOOL_REGISTRY.all_names()))
        # Read-only: register/clear raise TypeError.
        with self.assertRaises(TypeError):
            self.module.TOOL_REGISTRY.register(
                "x", "x", {"type": "object"}, lambda **kw: None
            )
        with self.assertRaises(TypeError):
            self.module.TOOL_REGISTRY.clear()
        # Iteration / indexing work (the stage 2A debt is fixed).
        self.assertIn("TodoWrite", set(self.module.TOOL_REGISTRY))
        self.assertEqual(
            self.module.TOOL_REGISTRY["TodoWrite"].owner, "todo-extension"
        )

    def test_todowrite_not_in_output_policy_tools(self):
        # Kernel safety / OutputPolicy unaffected: TodoWrite output is never
        # artifacted (only read_file/bash/grep_search go through the policy).
        self.assertNotIn("TodoWrite", self.module._OUTPUT_POLICY_TOOLS)


# ---------------------------------------------------------------------------
# 3, 4, 5: default composition — count, order, schema, handler
# ---------------------------------------------------------------------------


class DefaultCompositionTests(unittest.TestCase):
    """Default agent_loop exposes 25 tools; order/schema/handler unchanged."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_agent_loop_exposes_25_tools_in_legacy_order(self):
        # 3 & 4: default agent_loop sends exactly 25 tools in legacy order.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(len(captured[0]), 25)
        self.assertEqual(captured[0], EXPECTED_TOOL_NAMES)

    def test_default_composed_schema_for_todowrite_unchanged(self):
        # 4: TodoWrite schema in the composed set equals TODO_WRITE_SCHEMA.
        tools_by_name = {t["name"]: t for t in self.module.TOOLS}
        self.assertEqual(
            tools_by_name["TodoWrite"]["input_schema"],
            self.module.TODO_WRITE_SCHEMA,
        )
        self.assertEqual(
            tools_by_name["TodoWrite"]["input_schema"]["required"], ["items"]
        )

    def test_default_composed_schemas_all_well_formed(self):
        # 4: every composed tool exposes name/description/input_schema.
        for tool in self.module.TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)
            self.assertIsInstance(tool["input_schema"], dict)

    def test_default_todowrite_handler_behavior_unchanged(self):
        # 5: default TodoWrite handler updates the module-level TODO store
        # and returns its render — same behavior as pre-2D-B.
        handler = self.module.TOOL_HANDLERS["TodoWrite"]
        self.module.TODO.items = []
        result = handler(items=[
            {"content": "task A", "status": "in_progress", "activeForm": "doing A"},
            {"content": "task B", "status": "pending", "activeForm": "will do B"},
        ])
        self.assertIn("task A", result)
        self.assertIn("[>]", result)  # in_progress marker
        self.assertEqual(len(self.module.TODO.items), 2)
        self.assertEqual(self.module.TODO.items[0]["content"], "task A")

    def test_default_todowrite_handler_is_same_callable_as_extension(self):
        # 5: the handler exposed via TOOL_HANDLERS is the one TodoExtension
        # registered (callable behavior identical, not a stale Base copy).
        # Rebuild a fresh composed overlay and compare behavior.
        fresh = self.module.build_default_tool_registry()
        fresh_handler = fresh.get_handler("TodoWrite")
        self.assertIsNotNone(fresh_handler)
        self.module.TODO.items = []
        out = fresh_handler(items=[
            {"content": "x", "status": "completed", "activeForm": "done x"},
        ])
        self.assertIn("x", out)
        self.assertIn("[x]", out)  # completed marker


# ---------------------------------------------------------------------------
# 6, 7, 11: disable semantics — tool_contributors=()
# ---------------------------------------------------------------------------


class DisableSemanticsTests(unittest.TestCase):
    """tool_contributors=() disables TodoWrite; call returns Unknown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_empty_disables_todowrite(self):
        # 6: tool_contributors=() → model does not see TodoWrite (12 tools:
        # Base only — TodoWrite, 4 task tools, subagent, and 7 Team tools are extensions).
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_contributors=()
        )
        self.assertNotIn("TodoWrite", captured[0])
        self.assertEqual(len(captured[0]), 12)

    def test_disabled_todowrite_call_returns_unknown(self):
        # 7: with extensions disabled, a forged TodoWrite call returns
        # "Unknown tool" (not "unavailable").
        responses = [
            _resp_tool_use(name="TodoWrite", input_={"items": []}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "x"}]
        self.module.agent_loop(messages, tool_contributors=())
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("Unknown tool", result)
        self.assertNotIn("unavailable", result.lower())

    def test_planning_profile_with_disabled_extensions_starts_without_todo(self):
        # 11: planning profile + disabled extensions runs fine, just no TodoWrite.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_profile="planning",
            tool_contributors=(),
        )
        # planning whitelists TodoWrite, but the extension is disabled → absent.
        self.assertNotIn("TodoWrite", captured[0])
        # Other planning tools still present.
        self.assertIn("read_file", captured[0])
        self.assertIn("grep_search", captured[0])


# ---------------------------------------------------------------------------
# 8: default execution
# ---------------------------------------------------------------------------


class DefaultExecutionTests(unittest.TestCase):
    """Default mode TodoWrite is executable end-to-end."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_todowrite_executes_in_loop(self):
        # 8: default agent_loop can call TodoWrite and it runs.
        self.module.TODO.items = []
        responses = [
            _resp_tool_use(
                name="TodoWrite",
                input_={"items": [
                    {"content": "ship it", "status": "in_progress",
                     "activeForm": "shipping"},
                ]},
                id_="t1",
            ),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "update todo"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("ship it", result)
        self.assertIn("[>]", result)
        self.assertEqual(len(self.module.TODO.items), 1)


# ---------------------------------------------------------------------------
# 9, 10: profile interaction with default extension
# ---------------------------------------------------------------------------


class ProfileInteractionTests(unittest.TestCase):
    """planning includes TodoWrite; coding/readonly exclude it (unavailable)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_planning_profile_default_includes_todowrite(self):
        # 9: planning profile + default extension exposes TodoWrite.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="planning"
        )
        self.assertIn("TodoWrite", captured[0])

    def test_coding_profile_excludes_todowrite(self):
        # 10: coding profile does not whitelist TodoWrite.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="coding"
        )
        self.assertNotIn("TodoWrite", captured[0])

    def test_readonly_profile_excludes_todowrite(self):
        # 10: readonly profile does not whitelist TodoWrite.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="readonly"
        )
        self.assertNotIn("TodoWrite", captured[0])

    def test_coding_profile_todowrite_call_returns_unavailable(self):
        # 10 + error semantics: TodoExtension installed (default) but coding
        # profile hides TodoWrite → "unavailable", NOT "unknown".
        responses = [
            _resp_tool_use(name="TodoWrite", input_={"items": []}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "x"}]
        self.module.agent_loop(messages, tool_profile="coding")
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("unavailable", result.lower())
        self.assertNotIn("Unknown tool", result)


# ---------------------------------------------------------------------------
# 12, 13: concurrency isolation
# ---------------------------------------------------------------------------


class ConcurrencyIsolationTests(unittest.TestCase):
    """Per-call overlay isolation: default/disabled and different contributors."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_and_disabled_calls_do_not_interfere(self):
        # 12: default (TodoWrite on) and explicit-() (TodoWrite off) calls
        # do not pollute each other; Base stays at 19 (no TodoWrite/task tools).
        cap_default1 = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_default1
        )
        self.module.agent_loop([{"role": "user", "content": "a"}])

        cap_disabled = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_disabled
        )
        self.module.agent_loop(
            [{"role": "user", "content": "b"}], tool_contributors=()
        )

        cap_default2 = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_default2
        )
        self.module.agent_loop([{"role": "user", "content": "c"}])

        self.assertIn("TodoWrite", cap_default1[0])
        self.assertNotIn("TodoWrite", cap_disabled[0])
        self.assertIn("TodoWrite", cap_default2[0])
        # Base never polluted by any per-call overlay.
        self.assertNotIn(
            "TodoWrite", set(self.module.BASE_TOOL_REGISTRY.all_names())
        )

    def test_different_contributors_do_not_interfere(self):
        # 13: one agent uses a custom ext_todo contributor (no TodoWrite);
        # another uses the default (TodoWrite). Neither leaks into the other.
        cap_custom = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_custom
        )
        self.module.agent_loop(
            [{"role": "user", "content": "a"}],
            tool_contributors=[_ExtTodoContributor()],
        )

        cap_default = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_default
        )
        self.module.agent_loop([{"role": "user", "content": "b"}])

        self.assertIn("ext_todo", cap_custom[0])
        self.assertNotIn("TodoWrite", cap_custom[0])
        self.assertIn("TodoWrite", cap_default[0])
        self.assertNotIn("ext_todo", cap_default[0])


# ---------------------------------------------------------------------------
# 14, 15: conflict detection
# ---------------------------------------------------------------------------


class ConflictDetectionTests(unittest.TestCase):
    """Duplicate TodoExtension and impostor TodoWrite fail fast."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_duplicate_todo_extension_fails_fast(self):
        # 14: installing TodoExtension twice → second register of TodoWrite
        # conflicts with the first; error names both owners.
        with self.assertRaises(ValueError) as ctx:
            self.module.agent_loop(
                [{"role": "user", "content": "x"}],
                tool_contributors=(
                    self.module.TodoExtension(),
                    self.module.TodoExtension(),
                ),
            )
        msg = str(ctx.exception)
        self.assertIn("TodoWrite", msg)
        self.assertIn("todo-extension", msg)

    def test_impostor_todowrite_fails_fast(self):
        # 15: a custom extension registering TodoWrite alongside the real
        # TodoExtension → conflict, error names both owners.
        with self.assertRaises(ValueError) as ctx:
            self.module.agent_loop(
                [{"role": "user", "content": "x"}],
                tool_contributors=(
                    self.module.TodoExtension(),
                    _EvilTodoContributor(),
                ),
            )
        msg = str(ctx.exception)
        self.assertIn("TodoWrite", msg)
        self.assertIn("todo-extension", msg)
        self.assertIn("evil-todo", msg)


# ---------------------------------------------------------------------------
# 16: contributor mid-registration failure
# ---------------------------------------------------------------------------


class ContributorFailureTests(unittest.TestCase):
    """A contributor that raises mid-registration fails before the model
    request and does not pollute Base."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_failing_contributor_raises_before_model_request(self):
        # 16: the RuntimeError propagates (no model request), and the
        # partially-registered ok_tool is NOT in Base (per-call overlay
        # discarded).
        create_called = []

        def _create(**kwargs):
            create_called.append(True)
            return _resp_text()

        self.module.client.messages.create = _create
        with self.assertRaises(RuntimeError) as ctx:
            self.module.agent_loop(
                [{"role": "user", "content": "x"}],
                tool_contributors=[_FailingContributor()],
            )
        self.assertIn("exploded", str(ctx.exception))
        # Model never called.
        self.assertEqual(create_called, [])
        # Base not polluted.
        self.assertFalse(self.module.BASE_TOOL_REGISTRY.has("ok_tool"))
        self.assertNotIn("ok_tool", set(self.module.BASE_TOOL_REGISTRY.all_names()))


# ---------------------------------------------------------------------------
# Snapshot semantics unchanged
# ---------------------------------------------------------------------------


class SnapshotSemanticsTests(unittest.TestCase):
    """agent_loop active-tools snapshot is stable across calls and is a
    fresh per-call overlay (not the module-level _DEFAULT_COMPOSED_REGISTRY)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_stable_across_calls(self):
        cap1 = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap1
        )
        self.module.agent_loop([{"role": "user", "content": "a"}])
        cap2 = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap2
        )
        self.module.agent_loop([{"role": "user", "content": "b"}])
        self.assertEqual(cap1[0], cap2[0])
        self.assertEqual(cap1[0], EXPECTED_TOOL_NAMES)

    def test_per_call_overlay_distinct_from_module_default(self):
        # build_default_tool_registry() returns a FRESH overlay each call,
        # distinct from _DEFAULT_COMPOSED_REGISTRY and from BASE_TOOL_REGISTRY.
        fresh = self.module.build_default_tool_registry()
        self.assertIsNot(fresh, self.module._DEFAULT_COMPOSED_REGISTRY)
        self.assertIsNot(fresh, self.module.BASE_TOOL_REGISTRY)
        # But it resolves to the same 25-tool legacy set.
        self.assertEqual(
            [t["name"] for t in fresh.resolve(profile=None)], EXPECTED_TOOL_NAMES
        )


if __name__ == "__main__":
    unittest.main()
