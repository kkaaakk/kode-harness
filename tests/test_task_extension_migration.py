"""test_task_extension_migration.py - Stage 2D-C verification.

Verifies the migration of the four task tools (task_create/get/update/list)
from BASE_TOOL_REGISTRY to TaskExtension:
  1.  Base Registry no longer contains any of the four task tools.
  2.  Base tool count is 12 (down from 25; TodoWrite + 4 task tools + subagent + 7 Team tools migrated).
  3.  Default agent_loop still exposes 25 tools to the model.
  4.  Default composed names/order/schema match the pre-2D-C legacy set.
  5.  Default task tool handler behavior is unchanged (delegates to TASK_MGR).
  6.  tool_contributors=() → model does not see any task tool.
  7.  Disabled task tool call returns "Unknown tool".
  8.  Default mode task tools execute normally end-to-end.
  9.  planning profile + default extension includes the task tools.
  10. coding/readonly/team profiles (default extension) exclude task tools.
  11. planning profile + disabled extension starts, just without task tools.
  12. Concurrent default-vs-disabled agent_loop calls do not interfere.
  13. Concurrent agents with different contributor combos do not interfere
      (Agent A with TaskExtension sees task tools; Agent B with only
      TodoExtension does not).
  14. Duplicate TaskExtension install fails fast, error names both owners.
  15. Impostor extension registering a task tool fails fast.
  16. Contributor mid-registration failure: no model request, Base unpolluted.
  17. Legacy TOOLS / TOOL_HANDLERS keep 25 tools in legacy order.
  18. TaskExtension is stateless (no per-instance task storage).
  19. TaskManager state lifecycle unchanged (global file-backed board).
  20. Task tools not in _OUTPUT_POLICY_TOOLS (Kernel safety unaffected).

Critical semantic fixed before migration (per the 2D-C plan): Task state is
Harness-level (a shared task board in TASKS_DIR), NOT per-session. This
migration does NOT change that — TaskExtension only moves registration
ownership; it delegates to the same module-level TASK_MGR.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"

# Canonical pre-2D-C tool order (stage 2A hard contract — unchanged by 2D-C).
EXPECTED_TOOL_NAMES = [
    "bash", "read_file", "write_file", "edit_file", "TodoWrite",
    "task", "load_skill", "compress", "background_run", "check_background",
    "task_create", "task_get", "task_update", "task_list",
    "spawn_teammate", "list_teammates", "send_message", "read_inbox",
    "broadcast", "shutdown_request", "plan_approval", "idle", "claim_task",
    "grep_search", "glob_search",
]

TASK_TOOL_NAMES = ("task_create", "task_get", "task_update", "task_list")


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
        "harness_core_task_migration_test", MODULE_PATH
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


class _EvilTaskContributor:
    """Registers task_create under a different owner — used for conflict tests."""
    extension_id = "evil-task"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="task_create",
            description="evil task",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "evil",
            owner=self.extension_id,
            source="extension",
            order=10,
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
# 1, 2, 17, 20: Base migration + legacy exports + output policy
# ---------------------------------------------------------------------------


class BaseMigrationTests(unittest.TestCase):
    """Base Registry no longer has the four task tools; legacy exports keep 25."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_registry_excludes_task_tools(self):
        # 1: Base Registry no longer contains any of the four task tools.
        names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, names)
            self.assertIsNone(self.module.BASE_TOOL_REGISTRY.get(name))

    def test_base_registry_has_12_tools(self):
        # 2: Base tool count is 12 (was 25; TodoWrite + 4 task tools + subagent
        # + 7 Team tools all migrated to extensions in 2D-B/2D-C/2D-D1/2D-D2).
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

    def test_task_tools_not_in_output_policy_tools(self):
        # 20: Kernel safety / OutputPolicy unaffected — task tool output is
        # never artifacted (only read_file/bash/grep_search go through policy).
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, self.module._OUTPUT_POLICY_TOOLS)


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

    def test_default_composed_schema_for_task_tools_unchanged(self):
        # 4: each task tool schema in the composed set equals its *_SCHEMA const.
        tools_by_name = {t["name"]: t for t in self.module.TOOLS}
        self.assertEqual(
            tools_by_name["task_create"]["input_schema"],
            self.module.TASK_CREATE_SCHEMA,
        )
        self.assertEqual(
            tools_by_name["task_get"]["input_schema"],
            self.module.TASK_GET_SCHEMA,
        )
        self.assertEqual(
            tools_by_name["task_update"]["input_schema"],
            self.module.TASK_UPDATE_SCHEMA,
        )
        self.assertEqual(
            tools_by_name["task_list"]["input_schema"],
            self.module.TASK_LIST_SCHEMA,
        )
        # Spot-check required fields are intact.
        self.assertEqual(
            tools_by_name["task_create"]["input_schema"]["required"], ["subject"]
        )
        self.assertEqual(
            tools_by_name["task_get"]["input_schema"]["required"], ["task_id"]
        )

    def test_default_composed_schemas_all_well_formed(self):
        # 4: every composed tool exposes name/description/input_schema.
        for tool in self.module.TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)
            self.assertIsInstance(tool["input_schema"], dict)

    def test_default_task_handlers_delegate_to_task_manager(self):
        # 5: default task_create handler delegates to the module-level TASK_MGR
        # and writes to the same file-backed board (Harness-level, unchanged).
        handler = self.module.TOOL_HANDLERS["task_create"]
        result = handler(subject="migration check", description="d")
        task = json.loads(result)
        self.assertEqual(task["subject"], "migration check")
        self.assertEqual(task["status"], "pending")
        # The task file lives in TASKS_DIR (global board, pre-2D-C semantic).
        task_file = self.module.TASKS_DIR / f"task_{task['id']}.json"
        self.assertTrue(task_file.exists())

    def test_default_task_handlers_are_same_callables_as_extension(self):
        # 5: rebuild a fresh composed overlay and confirm the task handlers
        # behave identically (not stale Base copies).
        fresh = self.module.build_default_tool_registry()
        create_handler = fresh.get_handler("task_create")
        list_handler = fresh.get_handler("task_list")
        self.assertIsNotNone(create_handler)
        out = create_handler(subject="fresh overlay", description="")
        self.assertIn("fresh overlay", out)
        listing = list_handler()
        self.assertIn("fresh overlay", listing)


# ---------------------------------------------------------------------------
# 6, 7, 11: disable semantics — tool_contributors=()
# ---------------------------------------------------------------------------


class DisableSemanticsTests(unittest.TestCase):
    """tool_contributors=() disables task tools; call returns Unknown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_empty_disables_task_tools(self):
        # 6: tool_contributors=() → model sees 12 tools (Base only), no task tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_contributors=()
        )
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, captured[0])
        self.assertEqual(len(captured[0]), 12)

    def test_disabled_task_call_returns_unknown(self):
        # 7: with extensions disabled, a forged task_create call returns
        # "Unknown tool" (not "unavailable").
        responses = [
            _resp_tool_use(name="task_create", input_={"subject": "x"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "x"}]
        self.module.agent_loop(messages, tool_contributors=())
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("Unknown tool", result)
        self.assertNotIn("unavailable", result.lower())

    def test_planning_profile_with_disabled_extensions_starts_without_tasks(self):
        # 11: planning whitelists task tools, but the extension is disabled →
        # they are simply absent (stage 2D-A: profile tolerates missing
        # optional tools). No error is raised.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_profile="planning",
            tool_contributors=(),
        )
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, captured[0])
        # Other planning tools still present.
        self.assertIn("read_file", captured[0])
        self.assertIn("grep_search", captured[0])


# ---------------------------------------------------------------------------
# 8: default execution
# ---------------------------------------------------------------------------


class DefaultExecutionTests(unittest.TestCase):
    """Default mode task tools are executable end-to-end."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_task_create_executes_in_loop(self):
        # 8: default agent_loop can call task_create and it runs, writing to
        # the global TASK_MGR board.
        responses = [
            _resp_tool_use(
                name="task_create",
                input_={"subject": "ship feature", "description": "do it"},
                id_="t1",
            ),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "create a task"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        task = json.loads(result)
        self.assertEqual(task["subject"], "ship feature")
        # The task is persisted in the global board.
        listing = self.module.TASK_MGR.list_all()
        self.assertIn("ship feature", listing)

    def test_default_task_update_executes_in_loop(self):
        # 8: task_update transitions status end-to-end.
        create_out = self.module.TASK_MGR.create("to finish", "")
        tid = json.loads(create_out)["id"]
        responses = [
            _resp_tool_use(
                name="task_update",
                input_={"task_id": tid, "status": "completed"},
                id_="t1",
            ),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "finish it"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        updated = json.loads(result)
        self.assertEqual(updated["status"], "completed")


# ---------------------------------------------------------------------------
# 9, 10: profile interaction with default extension
# ---------------------------------------------------------------------------


class ProfileInteractionTests(unittest.TestCase):
    """planning includes task tools; coding/readonly/team exclude them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_planning_profile_default_includes_task_tools(self):
        # 9: planning profile + default extension exposes all four task tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="planning"
        )
        for name in TASK_TOOL_NAMES:
            self.assertIn(name, captured[0])

    def test_coding_profile_excludes_task_tools(self):
        # 10: coding profile does not whitelist task tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="coding"
        )
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, captured[0])

    def test_readonly_profile_excludes_task_tools(self):
        # 10: readonly profile does not whitelist task tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="readonly"
        )
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, captured[0])

    def test_team_profile_excludes_task_tools(self):
        # 10: team profile does not whitelist the four task management tools
        # (it whitelists "task" the subagent tool, not task_create/get/...).
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="team"
        )
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, captured[0])

    def test_coding_profile_task_call_returns_unavailable(self):
        # 10 + error semantics: TaskExtension installed (default) but coding
        # profile hides task_create → "unavailable", NOT "unknown".
        responses = [
            _resp_tool_use(name="task_create", input_={"subject": "x"}, id_="t1"),
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
        # 12: default (task tools on) and explicit-() (task tools off) calls
        # do not pollute each other; Base stays at 12 (no task tools).
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

        for name in TASK_TOOL_NAMES:
            self.assertIn(name, cap_default1[0])
            self.assertNotIn(name, cap_disabled[0])
            self.assertIn(name, cap_default2[0])
        # Base never polluted by any per-call overlay.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, base_names)

    def test_two_agents_different_contributors_do_not_interfere(self):
        # 13: Agent A uses default contributors (Todo+Task) → sees task tools.
        # Agent B uses ONLY TodoExtension → does NOT see task tools (but still
        # sees TodoWrite). Neither leaks into the other; Base stays at 12.
        cap_a = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_a
        )
        self.module.agent_loop([{"role": "user", "content": "a"}])  # default

        cap_b = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap_b
        )
        self.module.agent_loop(
            [{"role": "user", "content": "b"}],
            tool_contributors=(self.module.TodoExtension(),),
        )

        # Agent A sees all four task tools + TodoWrite.
        for name in TASK_TOOL_NAMES:
            self.assertIn(name, cap_a[0])
        self.assertIn("TodoWrite", cap_a[0])
        # Agent B sees TodoWrite but NOT the task tools.
        self.assertIn("TodoWrite", cap_b[0])
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, cap_b[0])
        # Base unpolluted.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        for name in TASK_TOOL_NAMES:
            self.assertNotIn(name, base_names)


# ---------------------------------------------------------------------------
# 14, 15: conflict detection
# ---------------------------------------------------------------------------


class ConflictDetectionTests(unittest.TestCase):
    """Duplicate TaskExtension and impostor task tool fail fast."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_duplicate_task_extension_fails_fast(self):
        # 14: installing TaskExtension twice → second register of task_create
        # conflicts with the first; error names both owners.
        with self.assertRaises(ValueError) as ctx:
            self.module.agent_loop(
                [{"role": "user", "content": "x"}],
                tool_contributors=(
                    self.module.TaskExtension(),
                    self.module.TaskExtension(),
                ),
            )
        msg = str(ctx.exception)
        self.assertIn("task_create", msg)
        self.assertIn("task-extension", msg)

    def test_impostor_task_tool_fails_fast(self):
        # 15: a custom extension registering task_create alongside the real
        # TaskExtension → conflict, error names both owners.
        with self.assertRaises(ValueError) as ctx:
            self.module.agent_loop(
                [{"role": "user", "content": "x"}],
                tool_contributors=(
                    self.module.TaskExtension(),
                    _EvilTaskContributor(),
                ),
            )
        msg = str(ctx.exception)
        self.assertIn("task_create", msg)
        self.assertIn("task-extension", msg)
        self.assertIn("evil-task", msg)


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


# ---------------------------------------------------------------------------
# 18: TaskExtension is stateless
# ---------------------------------------------------------------------------


class StatelessExtensionTests(unittest.TestCase):
    """TaskExtension holds no per-instance task data — state stays in
    TaskManager (Service / runtime), preserving the pre-2D-C semantic."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_task_extension_has_no_instance_state(self):
        # 18: TaskExtension must not hold task data on the instance. The only
        # attribute is the class-level extension_id; __dict__ is empty.
        ext = self.module.TaskExtension()
        self.assertEqual(ext.__dict__, {})
        self.assertFalse(hasattr(ext, "tasks"))
        self.assertFalse(hasattr(ext, "_tasks"))
        self.assertEqual(ext.extension_id, "task-extension")

    def test_task_manager_state_lifecycle_unchanged(self):
        # 19: TaskManager remains a global, file-backed board (Harness-level).
        # The extension handler writes through to the same TASK_MGR / TASKS_DIR
        # — the migration does NOT change task state lifetime or scope.
        mgr = self.module.TASK_MGR
        # TaskManager itself carries no in-memory task dict (state on disk).
        self.assertFalse(hasattr(mgr, "tasks"))
        # create → get → update round-trip through the file board.
        created = json.loads(mgr.create("lifecycle", "desc"))
        tid = created["id"]
        got = json.loads(mgr.get(tid))
        self.assertEqual(got["subject"], "lifecycle")
        updated = json.loads(mgr.update(tid, status="in_progress"))
        self.assertEqual(updated["status"], "in_progress")
        # The extension-registered handler writes to the SAME board.
        handler = self.module.TOOL_HANDLERS["task_create"]
        via_ext = json.loads(handler(subject="via extension", description=""))
        ext_file = self.module.TASKS_DIR / f"task_{via_ext['id']}.json"
        self.assertTrue(ext_file.exists())


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
