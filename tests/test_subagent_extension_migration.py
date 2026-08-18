"""test_subagent_extension_migration.py - Stage 2D-D1 verification.

Verifies the migration of the ``task`` (subagent) tool from
BASE_TOOL_REGISTRY to SubagentExtension:
  1.  Base Registry no longer contains the ``task`` tool.
  2.  Base tool count is 12 (down from 25; TodoWrite + 4 task tools +
      subagent + 7 Team tools migrated).
  3.  Default agent_loop still exposes 25 tools to the model.
  4.  Default composed names/order/schema match the pre-2D-D1 legacy set.
  5.  Default ``task`` handler behavior is unchanged (delegates to
      run_subagent).
  6.  tool_contributors=() → model does not see the ``task`` tool.
  7.  Disabled ``task`` call returns "Unknown tool".
  8.  Default mode ``task`` executes normally end-to-end.
  9.  team profile + default extension includes the ``task`` tool.
  10. coding/readonly/planning profiles (default extension) exclude ``task``.
  11. team profile + disabled extension starts, just without ``task``.
  12. Concurrent default-vs-disabled agent_loop calls do not interfere.
  13. Concurrent agents with different contributor combos do not interfere
      (Agent A with SubagentExtension sees ``task``; Agent B with only
      TodoExtension + TaskExtension does not).
  14. Duplicate SubagentExtension install fails fast, error names both owners.
  15. Impostor extension registering ``task`` fails fast.
  16. Contributor mid-registration failure: no model request, Base unpolluted.
  17. Legacy TOOLS / TOOL_HANDLERS keep 25 tools in legacy order.
  18. SubagentExtension is stateless (no per-instance child state).
  19. run_subagent internal behavior unchanged (Explore=2 / General=4 tools,
      no ToolRegistry, no agent_loop, 30 rounds, max_tokens=8000, shares
      Client/Model/Sandbox).
  20. Secure Bash Parent→Subagent reuse semantics preserved:
        a. Parent grant continued to be reused by Subagent.
        b. No Parent grant → Subagent bash rejected (in secure mode).
        c. Subagent end does NOT revoke Parent grant.
  21. ``task`` tool not in _OUTPUT_POLICY_TOOLS (Kernel safety unaffected).

Critical D1 principle (per the plan): "把 task 从 Base 拔下来，插进
SubagentExtension，除此之外什么都不变。" This migration is PURELY a
registration-ownership move. run_subagent()'s internal execution model
(synchronous same-Task, own 30-round loop, hardcoded child tool set, no
ToolRegistry, no agent_loop, shared Client/Model/Sandbox, reused
SecureBashContext) is NOT changed.
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

# Canonical pre-2D-D1 tool order (stage 2A hard contract — unchanged by 2D-D1).
EXPECTED_TOOL_NAMES = [
    "bash", "read_file", "write_file", "edit_file", "TodoWrite",
    "task", "load_skill", "compress", "background_run", "check_background",
    "task_create", "task_get", "task_update", "task_list",
    "spawn_teammate", "list_teammates", "send_message", "read_inbox",
    "broadcast", "shutdown_request", "plan_approval", "idle", "claim_task",
    "grep_search", "glob_search",
]

SUBAGENT_TOOL_NAME = "task"


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
        "harness_core_subagent_migration_test", MODULE_PATH
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


class _EvilSubagentContributor:
    """Registers ``task`` under a different owner — used for conflict tests."""
    extension_id = "evil-subagent"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="task",
            description="evil subagent",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "evil",
            owner=self.extension_id,
            source="extension",
            order=5,
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


class _FakeSandbox:
    """Fake sandbox with filesystem-isolation capabilities for secure-mode tests."""
    class _Caps:
        supports_filesystem_isolation = True

    capabilities = _Caps()

    def execute(self, command, **kwargs):
        return f"executed: {command}"

    def execute_structured(self, command):
        return (f"executed: {command}", "", 0)


# ---------------------------------------------------------------------------
# 1, 2, 17, 21: Base migration + legacy exports + output policy
# ---------------------------------------------------------------------------


class BaseMigrationTests(unittest.TestCase):
    """Base Registry no longer has the ``task`` tool; legacy exports keep 25."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_registry_excludes_subagent_task(self):
        # 1: Base Registry no longer contains the ``task`` (subagent) tool.
        names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        self.assertNotIn(SUBAGENT_TOOL_NAME, names)
        self.assertIsNone(self.module.BASE_TOOL_REGISTRY.get(SUBAGENT_TOOL_NAME))

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

    def test_subagent_task_not_in_output_policy_tools(self):
        # 21: Kernel safety / OutputPolicy unaffected — task tool output is
        # never artifacted (only read_file/bash/grep_search go through policy).
        self.assertNotIn(SUBAGENT_TOOL_NAME, self.module._OUTPUT_POLICY_TOOLS)

    def test_tool_registry_read_only_view_sees_task(self):
        # TOOL_REGISTRY (read-only view over composed) still sees ``task``
        # because SubagentExtension contributes it back.
        self.assertIn(SUBAGENT_TOOL_NAME, set(self.module.TOOL_REGISTRY.all_names()))
        entry = self.module.TOOL_REGISTRY[SUBAGENT_TOOL_NAME]
        self.assertEqual(entry.owner, "subagent-extension")


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

    def test_default_composed_schema_for_subagent_task_unchanged(self):
        # 4: ``task`` schema in the composed set equals TASK_SUBAGENT_SCHEMA.
        tools_by_name = {t["name"]: t for t in self.module.TOOLS}
        self.assertEqual(
            tools_by_name[SUBAGENT_TOOL_NAME]["input_schema"],
            self.module.TASK_SUBAGENT_SCHEMA,
        )
        self.assertEqual(
            tools_by_name[SUBAGENT_TOOL_NAME]["input_schema"]["required"],
            ["prompt"],
        )

    def test_default_composed_schemas_all_well_formed(self):
        # 4: every composed tool exposes name/description/input_schema.
        for tool in self.module.TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)
            self.assertIsInstance(tool["input_schema"], dict)

    def test_default_subagent_handler_delegates_to_run_subagent(self):
        # 5: default ``task`` handler delegates to run_subagent. We verify by
        # patching run_subagent in the extension's handler closure — but since
        # the handler is a lambda closing over run_subagent (module global),
        # we patch the module global and rebuild the overlay to confirm the
        # handler picks up the patched function.
        called = {"count": 0, "args": None}

        def fake_run_subagent(prompt, agent_type="Explore"):
            called["count"] += 1
            called["args"] = (prompt, agent_type)
            return "subagent result"

        original = self.module.run_subagent
        self.module.run_subagent = fake_run_subagent
        try:
            fresh = self.module.build_default_tool_registry()
            handler = fresh.get_handler(SUBAGENT_TOOL_NAME)
            self.assertIsNotNone(handler)
            result = handler(prompt="explore this", agent_type="Explore")
            self.assertEqual(result, "subagent result")
            self.assertEqual(called["count"], 1)
            self.assertEqual(called["args"], ("explore this", "Explore"))
        finally:
            self.module.run_subagent = original

    def test_subagent_task_at_correct_legacy_position(self):
        # 4: ``task`` is at index 5 in the composed TOOLS list (its legacy slot).
        names = [t["name"] for t in self.module.TOOLS]
        self.assertEqual(names.index(SUBAGENT_TOOL_NAME), 5)
        self.assertEqual(
            self.module.LEGACY_SUBAGENT_ORDER,
            self.module.LEGACY_25_TOOL_NAMES.index(SUBAGENT_TOOL_NAME),
        )


# ---------------------------------------------------------------------------
# 6, 7, 11: disable semantics — tool_contributors=()
# ---------------------------------------------------------------------------


class DisableSemanticsTests(unittest.TestCase):
    """tool_contributors=() disables the ``task`` tool; call returns Unknown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_empty_disables_subagent_task(self):
        # 6: tool_contributors=() → model sees 12 tools (Base only), no ``task``.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_contributors=()
        )
        self.assertNotIn(SUBAGENT_TOOL_NAME, captured[0])
        self.assertEqual(len(captured[0]), 12)

    def test_disabled_subagent_task_call_returns_unknown(self):
        # 7: with extensions disabled, a forged ``task`` call returns
        # "Unknown tool" (not "unavailable") — because ``task`` is not
        # registered in the per-call overlay at all.
        responses = [
            _resp_tool_use(
                name=SUBAGENT_TOOL_NAME,
                input_={"prompt": "do something"},
                id_="t1",
            ),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "x"}]
        self.module.agent_loop(messages, tool_contributors=())
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("Unknown tool", result)
        self.assertNotIn("unavailable", result.lower())

    def test_team_profile_with_disabled_extensions_starts_without_task(self):
        # 11: team whitelists ``task``, but the extension is disabled →
        # ``task`` is simply absent (stage 2D-A: profile tolerates missing
        # optional tools). No error is raised.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_profile="team",
            tool_contributors=(),
        )
        self.assertNotIn(SUBAGENT_TOOL_NAME, captured[0])
        # Other team tools still present.
        self.assertIn("read_file", captured[0])
        self.assertIn("bash", captured[0])


# ---------------------------------------------------------------------------
# 8: default execution
# ---------------------------------------------------------------------------


class DefaultExecutionTests(unittest.TestCase):
    """Default mode ``task`` tool is executable end-to-end."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_task_call_executes_run_subagent(self):
        # 8: default agent_loop can call ``task`` and it runs run_subagent.
        # We patch run_subagent to return a known string and verify the tool
        # result contains it.
        original = self.module.run_subagent
        self.module.run_subagent = lambda prompt, agent_type="Explore": (
            f"subagent ran: {prompt}"
        )
        try:
            fresh = self.module.build_default_tool_registry()
            handler = fresh.get_handler(SUBAGENT_TOOL_NAME)
            result = handler(prompt="test prompt", agent_type="Explore")
            self.assertEqual(result, "subagent ran: test prompt")
        finally:
            self.module.run_subagent = original

    def test_default_task_call_in_loop_with_mock_client(self):
        # 8: full end-to-end — agent_loop calls ``task``, run_subagent is
        # patched to avoid real LLM calls, and the result flows back.
        original = self.module.run_subagent
        self.module.run_subagent = lambda prompt, agent_type="Explore": (
            "subagent explored successfully"
        )
        try:
            responses = [
                _resp_tool_use(
                    name=SUBAGENT_TOOL_NAME,
                    input_={"prompt": "explore the codebase", "agent_type": "Explore"},
                    id_="t1",
                ),
                _resp_text(),
            ]
            self.module.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "spawn a subagent"}]
            self.module.agent_loop(messages)
            result = _find_tool_result(messages, "t1")
            self.assertIsNotNone(result)
            self.assertIn("subagent explored successfully", result)
        finally:
            self.module.run_subagent = original


# ---------------------------------------------------------------------------
# 9, 10: profile interaction with default extension
# ---------------------------------------------------------------------------


class ProfileInteractionTests(unittest.TestCase):
    """team includes ``task``; coding/readonly/planning exclude it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_team_profile_default_includes_subagent_task(self):
        # 9: team profile + default extension exposes the ``task`` tool.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="team"
        )
        self.assertIn(SUBAGENT_TOOL_NAME, captured[0])

    def test_coding_profile_excludes_subagent_task(self):
        # 10: coding profile does not whitelist ``task``.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="coding"
        )
        self.assertNotIn(SUBAGENT_TOOL_NAME, captured[0])

    def test_readonly_profile_excludes_subagent_task(self):
        # 10: readonly profile does not whitelist ``task``.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="readonly"
        )
        self.assertNotIn(SUBAGENT_TOOL_NAME, captured[0])

    def test_planning_profile_excludes_subagent_task(self):
        # 10: planning profile does not whitelist ``task`` (it has task_* mgmt
        # tools, NOT the subagent ``task`` tool).
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="planning"
        )
        self.assertNotIn(SUBAGENT_TOOL_NAME, captured[0])

    def test_coding_profile_task_call_returns_unavailable(self):
        # 10 + error semantics: SubagentExtension installed (default) but
        # coding profile hides ``task`` → "unavailable", NOT "unknown".
        responses = [
            _resp_tool_use(
                name=SUBAGENT_TOOL_NAME,
                input_={"prompt": "x"},
                id_="t1",
            ),
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
        # 12: default (task on) and explicit-() (task off) calls do not
        # pollute each other; Base stays at 12 (no task).
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

        self.assertIn(SUBAGENT_TOOL_NAME, cap_default1[0])
        self.assertNotIn(SUBAGENT_TOOL_NAME, cap_disabled[0])
        self.assertIn(SUBAGENT_TOOL_NAME, cap_default2[0])
        # Base never polluted by any per-call overlay.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        self.assertNotIn(SUBAGENT_TOOL_NAME, base_names)

    def test_two_agents_different_contributors_do_not_interfere(self):
        # 13: Agent A uses default contributors (Todo+Task+Subagent) → sees
        # ``task``. Agent B uses ONLY TodoExtension + TaskExtension → does
        # NOT see ``task`` (but still sees TodoWrite + task_* tools).
        # Neither leaks into the other; Base stays at 12.
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
            tool_contributors=(self.module.TodoExtension(), self.module.TaskExtension()),
        )

        self.assertIn(SUBAGENT_TOOL_NAME, cap_a[0])
        self.assertNotIn(SUBAGENT_TOOL_NAME, cap_b[0])
        # Agent B still has TodoWrite + task tools (just not subagent task).
        self.assertIn("TodoWrite", cap_b[0])
        self.assertIn("task_create", cap_b[0])
        # Base stays clean.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        self.assertNotIn(SUBAGENT_TOOL_NAME, base_names)
        self.assertEqual(len(base_names), 12)

    def test_two_custom_subagent_contributors_do_not_pollute(self):
        # 13b: two sequential calls each using a fresh SubagentExtension
        # instance — neither pollutes the other's overlay. Base stays 12..
        ext1 = self.module.SubagentExtension()
        ext2 = self.module.SubagentExtension()

        cap1 = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap1
        )
        self.module.agent_loop(
            [{"role": "user", "content": "a"}],
            tool_contributors=(ext1,),
        )
        cap2 = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], cap2
        )
        self.module.agent_loop(
            [{"role": "user", "content": "b"}],
            tool_contributors=(ext2,),
        )
        self.assertIn(SUBAGENT_TOOL_NAME, cap1[0])
        self.assertIn(SUBAGENT_TOOL_NAME, cap2[0])
        # Each overlay had exactly 1 task tool (not duplicated).
        self.assertEqual(cap1[0].count(SUBAGENT_TOOL_NAME), 1)
        self.assertEqual(cap2[0].count(SUBAGENT_TOOL_NAME), 1)
        # Base stays clean.
        self.assertNotIn(
            SUBAGENT_TOOL_NAME,
            set(self.module.BASE_TOOL_REGISTRY.all_names()),
        )


# ---------------------------------------------------------------------------
# 14, 15, 16: conflict / impostor / mid-registration failure
# ---------------------------------------------------------------------------


class ConflictTests(unittest.TestCase):
    """Duplicate SubagentExtension / impostor / failing contributor all fail fast."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_duplicate_subagent_extension_install_fails_fast(self):
        # 14: installing SubagentExtension twice on the same overlay raises
        # ValueError, and the error names both owners.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        ext = self.module.SubagentExtension()
        ext.contribute_tools(overlay)
        with self.assertRaises(ValueError) as ctx:
            ext.contribute_tools(overlay)
        msg = str(ctx.exception)
        self.assertIn("task", msg)
        self.assertIn("subagent-extension", msg)

    def test_impostor_registering_task_fails_fast(self):
        # 15: a different extension registering ``task`` fails with ValueError
        # naming both owners (evil-subagent vs subagent-extension).
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        self.module.SubagentExtension().contribute_tools(overlay)
        evil = _EvilSubagentContributor()
        with self.assertRaises(ValueError) as ctx:
            evil.contribute_tools(overlay)
        msg = str(ctx.exception)
        self.assertIn("task", msg)
        self.assertIn("evil-subagent", msg)
        self.assertIn("subagent-extension", msg)

    def test_mid_registration_failure_no_model_request_base_unpolluted(self):
        # 16: if a contributor fails mid-registration, agent_loop must not
        # make any model request, and Base must stay at 12.
        model_called = {"count": 0}
        original_create = self.module.client.messages.create

        def counting_create(**kwargs):
            model_called["count"] += 1
            return _resp_text()

        self.module.client.messages.create = counting_create
        try:
            with self.assertRaises(RuntimeError):
                self.module.agent_loop(
                    [{"role": "user", "content": "x"}],
                    tool_contributors=(self.module.SubagentExtension(), _FailingContributor()),
                )
        finally:
            self.module.client.messages.create = original_create
        self.assertEqual(model_called["count"], 0)
        self.assertNotIn(
            SUBAGENT_TOOL_NAME,
            set(self.module.BASE_TOOL_REGISTRY.all_names()),
        )
        self.assertEqual(
            len(self.module.BASE_TOOL_REGISTRY.resolve(profile=None)), 12
        )


# ---------------------------------------------------------------------------
# 18: SubagentExtension is stateless
# ---------------------------------------------------------------------------


class SubagentExtensionStatelessnessTests(unittest.TestCase):
    """SubagentExtension must NOT hold child state, messages, Sandbox, Client,
    Model, Artifact, or Token Budget. It is a thin registration shim."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_extension_has_no_instance_state_attributes(self):
        # 18: the extension instance has no per-instance state beyond its
        # class-level extension_id. No child messages, Sandbox, Client, etc.
        ext = self.module.SubagentExtension()
        # Class-level extension_id is the only attribute.
        self.assertEqual(ext.extension_id, "subagent-extension")
        # Instance __dict__ should be empty (no per-instance state).
        self.assertEqual(vars(ext), {})

    def test_extension_does_not_hold_child_references(self):
        # 18: the extension class must NOT reference child state, Sandbox,
        # Client, Model, Artifact, or Token Budget as attributes.
        ext_cls = self.module.SubagentExtension
        forbidden_attrs = {
            "child_state", "child_messages", "sandbox", "client", "model",
            "artifact", "token_budget", "running_agents", "messages",
            "session", "trace",
        }
        for attr in forbidden_attrs:
            self.assertFalse(
                hasattr(ext_cls, attr),
                f"SubagentExtension must not have attribute {attr!r}",
            )

    def test_two_extension_instances_share_no_state(self):
        # 18: two SubagentExtension instances are independent — mutating one
        # (if it had state) would not affect the other. Since it's stateless,
        # both are identical shims.
        ext1 = self.module.SubagentExtension()
        ext2 = self.module.SubagentExtension()
        self.assertEqual(ext1.extension_id, ext2.extension_id)
        # Both contribute the same tool with the same owner.
        overlay1 = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        overlay2 = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        ext1.contribute_tools(overlay1)
        ext2.contribute_tools(overlay2)
        e1 = overlay1.get(SUBAGENT_TOOL_NAME)
        e2 = overlay2.get(SUBAGENT_TOOL_NAME)
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)
        self.assertEqual(e1.owner, e2.owner)
        self.assertEqual(e1.name, e2.name)


# ---------------------------------------------------------------------------
# 19: run_subagent internal behavior unchanged
# ---------------------------------------------------------------------------


class SubagentRuntimeUnchangedTests(unittest.TestCase):
    """run_subagent's internal execution model is NOT changed by D1.

    Locks:
      - Explore = bash + read_file (2 tools)
      - general-purpose = bash + read_file + write_file + edit_file (4 tools)
      - Does NOT use ToolRegistry / TOOLS / TOOL_HANDLERS / BASE_TOOL_REGISTRY
      - Does NOT call agent_loop (own 30-round loop)
      - max_tokens=8000 (hardcoded, not parent budget)
      - Shares global Client / Model / Sandbox (via agents.config import)
      - Synchronous (no new thread/Task)
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _capture_subagent_call(self, agent_type, sub_responses=None):
        """Run run_subagent with a capturing client; return captured kwargs
        (tools, model, max_tokens) and the result string.

        Patches via ``run_subagent.__globals__`` so we hit the EXACT module
        namespace the loaded harness uses."""
        captured = {}

        class _FakeMsgs:
            def __init__(self, responses):
                self._responses = responses or [_resp_text("subagent done")]

            def create(self, **kwargs):
                captured["tools"] = [t["name"] for t in kwargs.get("tools", [])]
                captured["model"] = kwargs.get("model")
                captured["max_tokens"] = kwargs.get("max_tokens")
                resp = self._responses.pop(0) if self._responses else _resp_text()
                return resp

        run_subagent = self.module.run_subagent
        g = run_subagent.__globals__
        original_client = g.get("client")
        g["client"] = types.SimpleNamespace(messages=_FakeMsgs(sub_responses))
        try:
            result = run_subagent("explore this", agent_type)
        finally:
            g["client"] = original_client
        return captured, result

    def test_explore_exposes_only_bash_and_read(self):
        # 19: Explore = bash + read_file (2 tools).
        captured, _ = self._capture_subagent_call("Explore")
        self.assertEqual(captured["tools"], ["bash", "read_file"])

    def test_general_purpose_adds_write_and_edit(self):
        # 19: general-purpose = bash + read_file + write_file + edit_file (4).
        captured, _ = self._capture_subagent_call("general-purpose")
        self.assertEqual(
            captured["tools"],
            ["bash", "read_file", "write_file", "edit_file"],
        )

    def test_does_not_inherit_parent_tools(self):
        # 19: child tool set is the hardcoded 2-4 tools — never TodoWrite,
        # task_*, team tools, or the subagent ``task`` tool (no recursion).
        captured, _ = self._capture_subagent_call("general-purpose")
        forbidden = {
            "TodoWrite", "task_create", "task_get", "task_update", "task_list",
            "task", "spawn_teammate", "list_teammates", "send_message",
            "read_inbox", "broadcast", "shutdown_request", "plan_approval",
            "claim_task", "idle", "load_skill", "compress",
            "background_run", "check_background", "grep_search", "glob_search",
        }
        self.assertEqual(set(captured["tools"]) & forbidden, set())

    def test_does_not_use_tool_registry(self):
        # 19: run_subagent builds tools inline; the tool names passed to the
        # model are NOT derived from BASE_TOOL_REGISTRY / DEFAULT_TOOL_CONTRIBUTORS
        # (which would be 25 tools). This is the key independence contract.
        captured, _ = self._capture_subagent_call("Explore")
        self.assertNotEqual(len(captured["tools"]), 25)
        self.assertEqual(len(captured["tools"]), 2)

    def test_hardcodes_max_tokens_not_parent_budget(self):
        # 19: child does not inherit a parent token budget; it uses 8000.
        captured, _ = self._capture_subagent_call("Explore")
        self.assertEqual(captured["max_tokens"], 8000)

    def test_does_not_call_agent_loop(self):
        # 19: run_subagent has its own loop; it must not recurse into
        # agent_loop. Structural proof: agent_loop is not even in
        # run_subagent's module namespace.
        run_subagent = self.module.run_subagent
        self.assertNotIn("agent_loop", run_subagent.__globals__)
        # And it does not import ToolRegistry-derived names either.
        for name in ("TOOLS", "TOOL_HANDLERS", "BASE_TOOL_REGISTRY"):
            self.assertNotIn(name, run_subagent.__globals__)

    def test_shares_global_client_and_model(self):
        # 19: run_subagent imports client/MODEL from agents.config — the SAME
        # globals the parent agent_loop uses. This proves it shares Client/Model.
        run_subagent = self.module.run_subagent
        self.assertIn("client", run_subagent.__globals__)
        self.assertIn("MODEL", run_subagent.__globals__)

    def test_30_round_loop_executes_multiple_tool_calls(self):
        # 19: run_subagent's loop runs up to 30 rounds. We verify by feeding
        # multiple tool_use responses and confirming all are processed.
        responses = []
        for i in range(5):
            responses.append(types.SimpleNamespace(
                stop_reason="tool_use",
                content=[_Block(name="bash", input_={"command": f"echo {i}"}, id_=f"t{i}")],
                usage=None,
            ))
        responses.append(_resp_text("all done"))
        captured, result = self._capture_subagent_call("Explore", responses)
        self.assertIn("all done", result)
        # The loop processed 6 model calls (5 tool_use + 1 end_turn).
        self.assertEqual(captured["max_tokens"], 8000)

    def test_synchronous_no_new_thread(self):
        # 19: run_subagent is synchronous — it runs in the caller's thread.
        # Structural proof: it does not import threading or asyncio.
        run_subagent = self.module.run_subagent
        self.assertNotIn("threading", run_subagent.__globals__)
        self.assertNotIn("asyncio", run_subagent.__globals__)


# ---------------------------------------------------------------------------
# 20: Secure Bash Parent→Subagent reuse semantics
# ---------------------------------------------------------------------------


class SecureBashReuseTests(unittest.TestCase):
    """D1 must NOT change SecureBash semantics. Three scenarios:

    a. Parent grant continued to be reused by Subagent (same Task → same
       ContextVar + live nonce).
    b. No Parent grant → Subagent bash rejected (in secure mode).
    c. Subagent end does NOT revoke Parent grant (run_subagent does not call
       set/reset_secure_bash_context).

    These tests patch run_bash's module globals to simulate secure_multi_session
    mode, then verify the ContextVar-based reuse semantics.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))
        # Access the base_tools module namespace via run_bash.__globals__.
        self.run_bash = self.module.run_bash
        self.base_tools_globals = self.run_bash.__globals__
        # Save originals for restoration.
        self._orig_run_mode = self.base_tools_globals.get("RUN_MODE")
        self._orig_sandbox = self.base_tools_globals.get("SANDBOX")
        # The secure-context functions (same module as run_bash).
        self.set_secure_bash_context = self.base_tools_globals["set_secure_bash_context"]
        self.reset_secure_bash_context = self.base_tools_globals["reset_secure_bash_context"]
        self.has_valid_secure_bash_context = self.base_tools_globals["has_valid_secure_bash_context"]
        self.fake_sandbox = _FakeSandbox()

    def tearDown(self):
        self.base_tools_globals["RUN_MODE"] = self._orig_run_mode
        self.base_tools_globals["SANDBOX"] = self._orig_sandbox
        self._tmp.cleanup()

    def _enable_secure_mode(self):
        """Patch run_bash's module to simulate secure_multi_session mode."""
        self.base_tools_globals["RUN_MODE"] = "secure_multi_session"
        self.base_tools_globals["SANDBOX"] = self.fake_sandbox

    def _make_subagent_bash_response(self, command="echo hello"):
        """Build a subagent response sequence: one bash tool_use then end_turn."""
        return [
            types.SimpleNamespace(
                stop_reason="tool_use",
                content=[_Block(name="bash", input_={"command": command}, id_="sub1")],
                usage=None,
            ),
            _resp_text("subagent finished"),
        ]

    def _run_subagent_with_bash(self, prompt="do work"):
        """Call run_subagent with a mock client that triggers one bash call.
        Returns the bash tool_result content (or None if bash wasn't called)."""
        bash_result = {"content": None}

        class _FakeMsgs:
            def __init__(self, responses):
                self._responses = list(responses)

            def create(self, **kwargs):
                if not self._responses:
                    return _resp_text("no more responses")
                resp = self._responses.pop(0)
                return resp

        run_subagent = self.module.run_subagent
        g = run_subagent.__globals__
        original_client = g.get("client")
        original_run_bash = g.get("run_bash")

        # Wrap run_bash to capture its output.
        def capturing_run_bash(command):
            out = original_run_bash(command)
            bash_result["content"] = str(out)
            return out

        g["client"] = types.SimpleNamespace(
            messages=_FakeMsgs(self._make_subagent_bash_response())
        )
        g["run_bash"] = capturing_run_bash
        try:
            result = run_subagent(prompt, "Explore")
        finally:
            g["client"] = original_client
            g["run_bash"] = original_run_bash
        return bash_result["content"], result

    def test_parent_grant_reused_by_subagent(self):
        # 20a: Parent secure context is established → run_subagent's bash
        # calls succeed (same Task shares the ContextVar).
        self._enable_secure_mode()
        # Establish parent grant.
        token = self.set_secure_bash_context(
            run_id="parent-run", sandbox=self.fake_sandbox
        )
        try:
            self.assertTrue(
                self.has_valid_secure_bash_context(self.fake_sandbox)
            )
            bash_output, _ = self._run_subagent_with_bash()
            # bash was NOT blocked — output is the execution result, not an error.
            self.assertIsNotNone(bash_output)
            self.assertNotIn("Error:", bash_output)
            self.assertIn("executed:", bash_output)
        finally:
            self.reset_secure_bash_context(token)

    def test_no_parent_grant_blocks_subagent_bash(self):
        # 20b: No SecureBashContext established → run_subagent's bash calls
        # are rejected in secure_multi_session mode. The extension does NOT
        # create its own grant — it merely reuses what the parent set up.
        self._enable_secure_mode()
        # Do NOT set a secure context — simulate a direct run_subagent call
        # from outside agent_loop (no parent grant).
        self.assertFalse(
            self.has_valid_secure_bash_context(self.fake_sandbox)
        )
        bash_output, _ = self._run_subagent_with_bash()
        # bash WAS blocked — output starts with "Error:".
        self.assertIsNotNone(bash_output)
        self.assertIn("Error:", bash_output)
        self.assertIn("secure_multi_session", bash_output)

    def test_subagent_end_does_not_revoke_parent_grant(self):
        # 20c: After run_subagent returns, the parent's secure context is
        # STILL valid — because run_subagent does NOT call
        # reset_secure_bash_context (it has no token to reset).
        self._enable_secure_mode()
        token = self.set_secure_bash_context(
            run_id="parent-run", sandbox=self.fake_sandbox
        )
        try:
            self.assertTrue(
                self.has_valid_secure_bash_context(self.fake_sandbox)
            )
            self._run_subagent_with_bash()
            # Parent grant still valid after subagent returned.
            self.assertTrue(
                self.has_valid_secure_bash_context(self.fake_sandbox)
            )
            # Parent can still run bash directly.
            parent_bash = self.run_bash("echo parent still works")
            self.assertNotIn("Error:", str(parent_bash))
        finally:
            self.reset_secure_bash_context(token)

    def test_run_subagent_does_not_import_secure_context_functions(self):
        # 20 (structural): run_subagent must NOT import or call
        # set_secure_bash_context / reset_secure_bash_context. This is the
        # structural guarantee that D1 did not add independent grant logic.
        run_subagent = self.module.run_subagent
        g = run_subagent.__globals__
        self.assertNotIn("set_secure_bash_context", g)
        self.assertNotIn("reset_secure_bash_context", g)
        self.assertNotIn("has_valid_secure_bash_context", g)
        self.assertNotIn("SecureBashContext", g)


# ---------------------------------------------------------------------------
# Legacy order hard contract
# ---------------------------------------------------------------------------


class LegacyOrderTests(unittest.TestCase):
    """LEGACY_SUBAGENT_ORDER pins ``task`` to its original slot (index 5)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_legacy_subagent_order_is_5(self):
        # ``task`` was at index 5 in the legacy 25-tool list.
        self.assertEqual(self.module.LEGACY_SUBAGENT_ORDER, 5)

    def test_composed_order_matches_legacy_25(self):
        # The default composed registry resolves to EXACTLY LEGACY_25_TOOL_NAMES.
        composed = self.module.build_default_tool_registry()
        names = [t["name"] for t in composed.resolve(profile=None)]
        self.assertEqual(names, self.module.LEGACY_25_TOOL_NAMES)

    def test_task_not_appended_at_end(self):
        # Critical: ``task`` must NOT be appended at the end of the tool list
        # (which would happen if order defaulted to 0 or was omitted). It must
        # stay at its legacy position (index 5).
        names = [t["name"] for t in self.module.TOOLS]
        self.assertEqual(names[5], SUBAGENT_TOOL_NAME)
        self.assertNotEqual(names[-1], SUBAGENT_TOOL_NAME)


if __name__ == "__main__":
    unittest.main()
