"""test_team_extension_migration.py - Stage 2D-D2 verification.

Verifies the migration of the seven Parent-visible Team management tools
(spawn_teammate, list_teammates, send_message, read_inbox, broadcast,
shutdown_request, plan_approval) from BASE_TOOL_REGISTRY to TeamExtension:

  1.  Base Registry no longer contains any of the seven Team tools.
  2.  Base tool count is 12 (down from 25; TodoWrite + 4 task tools +
      subagent + 7 Team tools all migrated to extensions).
  3.  Default agent_loop still exposes 25 tools to the model.
  4.  Default composed names/order/schema match the pre-2D-D2 legacy set.
  5.  Default Team tool handlers delegate to the same module-level helpers
      (TEAM, BUS, handle_shutdown_request, handle_plan_review) — behavior
      unchanged.
  6.  tool_contributors=() → model does not see any Team tool.
  7.  Disabled Team tool call returns "Unknown tool".
  8.  (covered by default composition tests — Team tools execute via the
      delegated handlers.)
  9.  team profile + default extension includes all seven Team tools.
  10. coding/readonly/planning profiles (default extension) exclude Team
      tools — installing TeamExtension does NOT leak Team tools into
      non-team profiles.
  11. team profile + disabled extension starts, just without Team tools.
  12. Concurrent default-vs-disabled agent_loop calls do not interfere.
  13. Concurrent agents with different contributor combos do not interfere
      (Agent A with TeamExtension sees Team tools; Agent B without does not).
  14. Duplicate TeamExtension install fails fast, error names both owners.
  15. Impostor extension registering a Team tool fails fast.
  16. Contributor mid-registration failure: no model request, Base unpolluted.
  17. Legacy TOOLS / TOOL_HANDLERS keep 25 tools in legacy order.
  18. TeamExtension is stateless (no per-instance team state).
  19. Team member _loop internal behavior unchanged:
        - 7 hardcoded tools (bash/read_file/write_file/edit_file/
          send_message/idle/claim_task)
        - Does NOT use ToolRegistry / TOOLS / TOOL_HANDLERS
        - Does NOT call agent_loop (own 50-round loop)
        - max_tokens=8000 (hardcoded, not parent budget)
        - Shares global Client / Model / Sandbox / BUS / TASK_MGR
        - Runs in a daemon thread (spawn → threading.Thread(daemon=True))
        - Does NOT inherit Parent Profile / Contributors
  20. Secure Bash Parent→Team-member asymmetry preserved:
        a. Parent grant is NOT inherited by teammate thread (ContextVar is
           per-thread → fresh None in the new thread).
        b. No Parent grant → teammate bash rejected (in secure mode).
        c. Teammate lifecycle does NOT revoke Parent grant.
        d. TeamExtension itself does NOT create a secure context.
  21. Team tools not in _OUTPUT_POLICY_TOOLS (Kernel safety unaffected).
  22. Two-Team name-scoped isolation:
        a. Different teammate names → separate inboxes.
        b. shutdown_request is name-scoped (does not shut down others).
        c. Task claim is name-scoped.
        d. BUS / TASK_MGR are globally shared (no Team namespace) —
           documented as Runtime tech debt, NOT fixed by D2.
  23. Legacy order constants pin each Team tool to its original slot.

Critical D2 principle (per the plan): "只迁 Parent Agent 可见的 Team 工具
注册归属，不改 Team member 内部循环、线程模型、消息总线和状态生命
周期。" This migration is PURELY a registration-ownership move. The Team
member _loop()'s internal execution model (separate daemon thread, own
50-round loop, hardcoded 7-tool set, no ToolRegistry, no agent_loop,
shared Client/Model/Sandbox/BUS/TASK_MGR, NO inherited SecureBashContext)
is NOT changed.

IMPORTANT — two sets of "Team tools" are NOT the same:
  - Parent Agent's Team management tools (7, migrated by D2):
    spawn_teammate, list_teammates, send_message, read_inbox, broadcast,
    shutdown_request, plan_approval
  - Team member _loop's internal hardcoded tools (7, NOT migrated):
    bash, read_file, write_file, edit_file, send_message, idle, claim_task
D2 does NOT touch the second set. See the D0 contract snapshot for the
full Parent→Child inheritance matrix.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"

# Canonical pre-2D-D2 tool order (stage 2A hard contract — unchanged by 2D-D2).
EXPECTED_TOOL_NAMES = [
    "bash", "read_file", "write_file", "edit_file", "TodoWrite",
    "task", "load_skill", "compress", "background_run", "check_background",
    "task_create", "task_get", "task_update", "task_list",
    "spawn_teammate", "list_teammates", "send_message", "read_inbox",
    "broadcast", "shutdown_request", "plan_approval", "idle", "claim_task",
    "grep_search", "glob_search",
]

# The seven Parent-visible Team management tools migrated by D2.
TEAM_TOOL_NAMES = (
    "spawn_teammate", "list_teammates", "send_message", "read_inbox",
    "broadcast", "shutdown_request", "plan_approval",
)

# The seven Team member _loop internal hardcoded tools (NOT migrated).
TEAM_MEMBER_LOOP_TOOLS = (
    "bash", "read_file", "write_file", "edit_file",
    "send_message", "idle", "claim_task",
)


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
        "harness_core_team_migration_test", MODULE_PATH
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


class _EvilTeamContributor:
    """Registers ``spawn_teammate`` under a different owner — conflict test."""

    extension_id = "evil-team"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="spawn_teammate",
            description="evil spawn",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "evil",
            owner=self.extension_id,
            source="extension",
            order=14,
        )


class _FailingContributor:
    """Registers one tool, then raises — mid-registration failure."""

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
    """Base Registry no longer has any of the seven Team tools."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_registry_excludes_all_team_tools(self):
        # 1: Base Registry no longer contains any of the seven Team tools.
        names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, names)
            self.assertIsNone(self.module.BASE_TOOL_REGISTRY.get(tool_name))

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

    def test_team_tools_not_in_output_policy_tools(self):
        # 21: Kernel safety / OutputPolicy unaffected — Team tool outputs are
        # never artifacted (only read_file/bash/grep_search go through policy).
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, self.module._OUTPUT_POLICY_TOOLS)

    def test_tool_registry_read_only_view_sees_team_tools(self):
        # TOOL_REGISTRY (read-only view over composed) still sees all seven
        # Team tools because TeamExtension contributes them back.
        for tool_name in TEAM_TOOL_NAMES:
            self.assertIn(tool_name, set(self.module.TOOL_REGISTRY.all_names()))
            entry = self.module.TOOL_REGISTRY[tool_name]
            self.assertEqual(entry.owner, "team-extension")

    def test_default_contributors_include_team_extension(self):
        # DEFAULT_TOOL_CONTRIBUTORS must include TeamExtension as the 4th entry.
        names = [type(c).__name__ for c in self.module.DEFAULT_TOOL_CONTRIBUTORS]
        self.assertEqual(
            names,
            ["TodoExtension", "TaskExtension", "SubagentExtension", "TeamExtension"],
        )


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

    def test_default_composed_schema_for_team_tools_unchanged(self):
        # 4: each Team tool's schema in the composed set equals its constant.
        tools_by_name = {t["name"]: t for t in self.module.TOOLS}
        schema_map = {
            "spawn_teammate": self.module.SPAWN_TEAMMATE_SCHEMA,
            "list_teammates": self.module.LIST_TEAMMATES_SCHEMA,
            "send_message": self.module.SEND_MESSAGE_SCHEMA,
            "read_inbox": self.module.READ_INBOX_SCHEMA,
            "broadcast": self.module.BROADCAST_SCHEMA,
            "shutdown_request": self.module.SHUTDOWN_REQUEST_SCHEMA,
            "plan_approval": self.module.PLAN_APPROVAL_SCHEMA,
        }
        for tool_name, expected_schema in schema_map.items():
            self.assertEqual(
                tools_by_name[tool_name]["input_schema"],
                expected_schema,
                f"Schema mismatch for {tool_name}",
            )

    def test_default_composed_schemas_all_well_formed(self):
        # 4: every composed tool exposes name/description/input_schema.
        for tool in self.module.TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)
            self.assertIsInstance(tool["input_schema"], dict)

    def test_default_team_handlers_delegate_to_module_helpers(self):
        # 5: Team tool handlers delegate to the same module-level helpers
        # (TEAM, BUS, handle_shutdown_request, handle_plan_review). We verify
        # by patching the module globals and confirming the handlers pick up
        # the patched objects.
        fresh = self.module.build_default_tool_registry()

        # spawn_teammate → TEAM.spawn
        called = {"spawn": False}

        def fake_spawn(name, role, prompt):
            called["spawn"] = True
            return f"spawned {name}"
        original_spawn = self.module.TEAM.spawn
        self.module.TEAM.spawn = fake_spawn
        try:
            handler = fresh.get_handler("spawn_teammate")
            result = handler(name="alice", role="coder", prompt="go")
            self.assertTrue(called["spawn"])
            self.assertIn("spawned alice", result)
        finally:
            self.module.TEAM.spawn = original_spawn

        # list_teammates → TEAM.list_all
        original_list = self.module.TEAM.list_all
        self.module.TEAM.list_all = lambda: "team listing"
        try:
            handler = fresh.get_handler("list_teammates")
            self.assertEqual(handler(), "team listing")
        finally:
            self.module.TEAM.list_all = original_list

        # send_message → BUS.send
        called["send"] = {}

        def fake_send(sender, to, content, msg_type="message"):
            called["send"] = (sender, to, content, msg_type)
            return "sent"
        original_send = self.module.BUS.send
        self.module.BUS.send = fake_send
        try:
            handler = fresh.get_handler("send_message")
            handler(to="bob", content="hello", msg_type="message")
            self.assertEqual(called["send"], ("lead", "bob", "hello", "message"))
        finally:
            self.module.BUS.send = original_send

        # read_inbox → BUS.read_inbox("lead")
        original_read = self.module.BUS.read_inbox
        self.module.BUS.read_inbox = lambda name: [{"from": "bob", "content": "hi"}]
        try:
            handler = fresh.get_handler("read_inbox")
            import json as _json
            result = _json.loads(handler())
            self.assertEqual(result[0]["content"], "hi")
        finally:
            self.module.BUS.read_inbox = original_read

        # broadcast → BUS.broadcast
        called["broadcast"] = {}

        def fake_broadcast(sender, content, names):
            called["broadcast"] = (sender, content, names)
            return "broadcast done"
        original_broadcast = self.module.BUS.broadcast
        self.module.BUS.broadcast = fake_broadcast
        try:
            handler = fresh.get_handler("broadcast")
            handler(content="team update")
            self.assertEqual(called["broadcast"][0], "lead")
            self.assertEqual(called["broadcast"][1], "team update")
        finally:
            self.module.BUS.broadcast = original_broadcast

        # shutdown_request → handle_shutdown_request
        called["shutdown"] = {}

        def fake_shutdown(bus, teammate):
            called["shutdown"] = teammate
            return "shutdown sent"
        original_shutdown = self.module.handle_shutdown_request
        self.module.handle_shutdown_request = fake_shutdown
        try:
            handler = fresh.get_handler("shutdown_request")
            handler(teammate="alice")
            self.assertEqual(called["shutdown"], "alice")
        finally:
            self.module.handle_shutdown_request = original_shutdown

        # plan_approval → handle_plan_review
        called["plan"] = {}

        def fake_plan(bus, request_id, approve, feedback=""):
            called["plan"] = (request_id, approve, feedback)
            return "plan reviewed"
        original_plan = self.module.handle_plan_review
        self.module.handle_plan_review = fake_plan
        try:
            handler = fresh.get_handler("plan_approval")
            handler(request_id="r1", approve=True, feedback="good")
            self.assertEqual(called["plan"], ("r1", True, "good"))
        finally:
            self.module.handle_plan_review = original_plan

    def test_team_tools_at_correct_legacy_positions(self):
        # 4: each Team tool is at its original legacy slot in the composed list.
        names = [t["name"] for t in self.module.TOOLS]
        expected_positions = {
            "spawn_teammate": 14,
            "list_teammates": 15,
            "send_message": 16,
            "read_inbox": 17,
            "broadcast": 18,
            "shutdown_request": 19,
            "plan_approval": 20,
        }
        for tool_name, pos in expected_positions.items():
            self.assertEqual(names.index(tool_name), pos)


# ---------------------------------------------------------------------------
# 6, 7, 11: disable semantics — tool_contributors=()
# ---------------------------------------------------------------------------


class DisableSemanticsTests(unittest.TestCase):
    """tool_contributors=() disables all Team tools; calls return Unknown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_empty_disables_all_team_tools(self):
        # 6: tool_contributors=() → model sees 12 tools (Base only), no Team tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_contributors=()
        )
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, captured[0])
        self.assertEqual(len(captured[0]), 12)

    def test_disabled_team_tool_call_returns_unknown(self):
        # 7: with extensions disabled, a forged Team tool call returns
        # "Unknown tool" (not "unavailable") — because the tool is not
        # registered in the per-call overlay at all.
        responses = [
            _resp_tool_use(
                name="spawn_teammate",
                input_={"name": "x", "role": "y", "prompt": "z"},
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

    def test_team_profile_with_disabled_extensions_starts_without_team_tools(self):
        # 11: team whitelists Team tools, but extensions are disabled →
        # Team tools are simply absent (stage 2D-A: profile tolerates missing
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
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, captured[0])
        # Other Base team-profile tools still present.
        self.assertIn("read_file", captured[0])
        self.assertIn("bash", captured[0])


# ---------------------------------------------------------------------------
# Partial contributor tests — TeamExtension only / without TeamExtension
# ---------------------------------------------------------------------------


class PartialContributorTests(unittest.TestCase):
    """Selective contributor combos: Team tools appear only with TeamExtension."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_todo_task_subagent_without_team_extension(self):
        # Todo + Task + Subagent (no TeamExtension) → Team tools Unknown.
        # The model should see Base(12) + TodoWrite + task_* + task = 18,
        # with NO Team tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_contributors=(
                self.module.TodoExtension(),
                self.module.TaskExtension(),
                self.module.SubagentExtension(),
            ),
        )
        sent = captured[0]
        self.assertEqual(len(sent), 18)  # 12 Base + 1 Todo + 4 task + 1 subagent
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, sent)
        # But TodoWrite, task_*, and task ARE present.
        self.assertIn("TodoWrite", sent)
        self.assertIn("task_create", sent)
        self.assertIn("task", sent)

    def test_team_extension_only(self):
        # Only TeamExtension → Team tools present, Todo/task/subagent absent.
        # The model should see Base(12) + 7 Team = 19.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_contributors=(self.module.TeamExtension(),),
        )
        sent = captured[0]
        self.assertEqual(len(sent), 19)  # 12 Base + 7 Team
        for tool_name in TEAM_TOOL_NAMES:
            self.assertIn(tool_name, sent)
        # But TodoWrite, task_*, and task are NOT present.
        self.assertNotIn("TodoWrite", sent)
        self.assertNotIn("task_create", sent)
        self.assertNotIn("task", sent)


# ---------------------------------------------------------------------------
# 9, 10: profile interaction with default extension
# ---------------------------------------------------------------------------


class ProfileInteractionTests(unittest.TestCase):
    """team includes all Team tools; coding/readonly/planning exclude them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_team_profile_default_includes_all_team_tools(self):
        # 9: team profile + default extension exposes all seven Team tools.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="team"
        )
        for tool_name in TEAM_TOOL_NAMES:
            self.assertIn(tool_name, captured[0])

    def test_coding_profile_excludes_team_tools(self):
        # 10: coding profile does not whitelist any Team tool — installing
        # TeamExtension must NOT leak Team tools into non-team profiles.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="coding"
        )
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, captured[0])

    def test_readonly_profile_excludes_team_tools(self):
        # 10: readonly profile does not whitelist any Team tool.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="readonly"
        )
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, captured[0])

    def test_planning_profile_excludes_team_tools(self):
        # 10: planning profile does not whitelist any Team tool.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="planning"
        )
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, captured[0])

    def test_coding_profile_team_tool_call_returns_unavailable(self):
        # 10 + error semantics: TeamExtension installed (default) but coding
        # profile hides Team tools → "unavailable", NOT "unknown".
        responses = [
            _resp_tool_use(
                name="spawn_teammate",
                input_={"name": "x", "role": "y", "prompt": "z"},
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

    def test_team_profile_with_disabled_team_extension_starts(self):
        # 11: team profile + tool_contributors=() (all extensions disabled) →
        # team profile still resolves (tolerates missing optional tools),
        # just without any Team tools. No profile configuration error.
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_profile="team",
            tool_contributors=(),
        )
        # No error raised; Base team-profile tools are present.
        self.assertIn("read_file", captured[0])
        self.assertIn("bash", captured[0])
        # Team tools absent.
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, captured[0])


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
        # 12: default (Team on) and explicit-() (Team off) calls do not
        # pollute each other; Base stays at 12 (no Team tools).
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

        # Default calls see Team tools; disabled does not.
        for tool_name in TEAM_TOOL_NAMES:
            self.assertIn(tool_name, cap_default1[0])
            self.assertNotIn(tool_name, cap_disabled[0])
            self.assertIn(tool_name, cap_default2[0])
        # Base never polluted by any per-call overlay.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, base_names)

    def test_two_agents_different_contributors_do_not_interfere(self):
        # 13: Agent A uses default contributors (Todo+Task+Subagent+Team) →
        # sees Team tools. Agent B uses ONLY TodoExtension + TaskExtension →
        # does NOT see Team tools (but still sees TodoWrite + task_* tools).
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

        for tool_name in TEAM_TOOL_NAMES:
            self.assertIn(tool_name, cap_a[0])
            self.assertNotIn(tool_name, cap_b[0])
        # Agent B still has TodoWrite + task tools.
        self.assertIn("TodoWrite", cap_b[0])
        self.assertIn("task_create", cap_b[0])
        # Base stays clean.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(tool_name, base_names)
        self.assertEqual(len(base_names), 12)

    def test_two_custom_team_contributors_do_not_pollute(self):
        # 13b: two sequential calls each using a fresh TeamExtension instance
        # — neither pollutes the other's overlay. Base stays 12.
        ext1 = self.module.TeamExtension()
        ext2 = self.module.TeamExtension()

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
        for tool_name in TEAM_TOOL_NAMES:
            self.assertIn(tool_name, cap1[0])
            self.assertIn(tool_name, cap2[0])
            # Each overlay had exactly 1 of each Team tool (not duplicated).
            self.assertEqual(cap1[0].count(tool_name), 1)
            self.assertEqual(cap2[0].count(tool_name), 1)
        # Base stays clean.
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(
                tool_name,
                set(self.module.BASE_TOOL_REGISTRY.all_names()),
            )


# ---------------------------------------------------------------------------
# 14, 15, 16: conflict / impostor / mid-registration failure
# ---------------------------------------------------------------------------


class ConflictTests(unittest.TestCase):
    """Duplicate TeamExtension / impostor / failing contributor all fail fast."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_duplicate_team_extension_install_fails_fast(self):
        # 14: installing TeamExtension twice on the same overlay raises
        # ValueError, and the error names both owners.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        ext = self.module.TeamExtension()
        ext.contribute_tools(overlay)
        with self.assertRaises(ValueError) as ctx:
            ext.contribute_tools(overlay)
        msg = str(ctx.exception)
        self.assertIn("spawn_teammate", msg)
        self.assertIn("team-extension", msg)

    def test_impostor_registering_team_tool_fails_fast(self):
        # 15: a different extension registering a Team tool fails with
        # ValueError naming both owners (evil-team vs team-extension).
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        self.module.TeamExtension().contribute_tools(overlay)
        evil = _EvilTeamContributor()
        with self.assertRaises(ValueError) as ctx:
            evil.contribute_tools(overlay)
        msg = str(ctx.exception)
        self.assertIn("spawn_teammate", msg)
        self.assertIn("evil-team", msg)
        self.assertIn("team-extension", msg)

    def test_mid_registration_failure_no_model_request_base_unpolluted(self):
        # 16: if a contributor fails mid-registration, agent_loop must not
        # make any model request, and Base must stay at 12.
        model_called = {"count": 0}

        def counting_create(**kwargs):
            model_called["count"] += 1
            return _resp_text()

        self.module.client.messages.create = counting_create
        with self.assertRaises(RuntimeError):
            self.module.agent_loop(
                [{"role": "user", "content": "x"}],
                tool_contributors=(self.module.TeamExtension(), _FailingContributor()),
            )
        self.assertEqual(model_called["count"], 0)
        for tool_name in TEAM_TOOL_NAMES:
            self.assertNotIn(
                tool_name,
                set(self.module.BASE_TOOL_REGISTRY.all_names()),
            )
        self.assertEqual(
            len(self.module.BASE_TOOL_REGISTRY.resolve(profile=None)), 12
        )


# ---------------------------------------------------------------------------
# 18: TeamExtension is stateless
# ---------------------------------------------------------------------------


class TeamExtensionStatelessnessTests(unittest.TestCase):
    """TeamExtension must NOT hold team state, members, inbox, tasks, Sandbox,
    Client, Model, Artifact, or Token Budget. It is a thin registration shim."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_extension_has_no_instance_state_attributes(self):
        # 18: the extension instance has no per-instance state beyond its
        # class-level extension_id. No members, inbox, tasks, Sandbox, etc.
        ext = self.module.TeamExtension()
        self.assertEqual(ext.extension_id, "team-extension")
        # Instance __dict__ should be empty (no per-instance state).
        self.assertEqual(vars(ext), {})

    def test_extension_does_not_hold_team_state(self):
        # 18: the extension class must NOT reference team state, members,
        # inbox, tasks, Sandbox, Client, Model, Artifact, or Token Budget
        # as attributes.
        ext_cls = self.module.TeamExtension
        forbidden_attrs = {
            "members", "inbox", "tasks", "member_state", "messages",
            "sandbox", "client", "model", "artifact", "token_budget",
            "session", "trace", "bus", "task_mgr", "team_mgr",
        }
        for attr in forbidden_attrs:
            self.assertFalse(
                hasattr(ext_cls, attr),
                f"TeamExtension must not have attribute {attr!r}",
            )

    def test_two_extension_instances_share_no_state(self):
        # 18: two TeamExtension instances are independent — since stateless,
        # both are identical shims.
        ext1 = self.module.TeamExtension()
        ext2 = self.module.TeamExtension()
        self.assertEqual(ext1.extension_id, ext2.extension_id)
        # Both contribute the same tools with the same owner.
        overlay1 = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        overlay2 = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        ext1.contribute_tools(overlay1)
        ext2.contribute_tools(overlay2)
        for tool_name in TEAM_TOOL_NAMES:
            e1 = overlay1.get(tool_name)
            e2 = overlay2.get(tool_name)
            self.assertIsNotNone(e1)
            self.assertIsNotNone(e2)
            self.assertEqual(e1.owner, e2.owner)
            self.assertEqual(e1.name, e2.name)


# ---------------------------------------------------------------------------
# 19: Team member _loop internal behavior unchanged
# ---------------------------------------------------------------------------


class TeamRuntimeUnchangedTests(unittest.TestCase):
    """Team member _loop's internal execution model is NOT changed by D2.

    Locks:
      - _loop builds 7 hardcoded tools inline (bash/read_file/write_file/
        edit_file/send_message/idle/claim_task) — NOT from ToolRegistry.
      - Does NOT use ToolRegistry / TOOLS / TOOL_HANDLERS / BASE_TOOL_REGISTRY.
      - Does NOT call agent_loop (own 50-round work phase loop).
      - max_tokens=8000 (hardcoded, not parent budget).
      - Shares global Client / Model / Sandbox / BUS / TASK_MGR.
      - Runs in a daemon thread (spawn → threading.Thread(daemon=True)).
      - Does NOT inherit Parent Profile / Contributors.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _get_tm_globals(self):
        """Access the agents.team_manager module namespace.

        After load_harness_module returns, the agents.* modules are removed
        from sys.modules (the test loader restores cached versions). But the
        objects imported by the harness (TEAM, handle_shutdown_request, etc.)
        still reference the team_manager module namespace via their
        __globals__. We use that to inspect team_manager's module-level names.
        """
        return self.module.handle_shutdown_request.__globals__

    def _capture_loop_first_client_call(self, member_name="test-member"):
        """Run _loop in a thread; capture the first client.messages.create
        kwargs (tools, model, max_tokens). The _loop exits immediately
        because the client raises after capturing."""
        captured = {}

        class _CapturingMsgs:
            def create(self, **kwargs):
                captured["tools"] = [t["name"] for t in kwargs.get("tools", [])]
                captured["model"] = kwargs.get("model")
                captured["max_tokens"] = kwargs.get("max_tokens")
                captured["system"] = kwargs.get("system")
                # Raise to make _loop exit via the except branch.
                raise Exception("capture-done")

        # Patch the client object shared by both harness_core and team_manager.
        original_messages = self.module.client.messages
        self.module.client.messages = _CapturingMsgs()

        # Register the member so _set_status works (no-op if not found, but
        # we register for correctness).
        team_mgr = self.module.TEAM
        # Clean any pre-existing config.
        team_mgr.config["members"] = []
        team_mgr.config["members"].append(
            {"name": member_name, "role": "coder", "status": "working"}
        )
        team_mgr._save()

        # Run _loop in a thread (mirrors real spawn behavior).
        t = threading.Thread(
            target=team_mgr._loop,
            args=(member_name, "coder", "do work"),
            daemon=True,
        )
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "_loop thread should have exited")

        # Restore client.
        self.module.client.messages = original_messages
        return captured

    def test_member_loop_exposes_exactly_7_hardcoded_tools(self):
        # 19: _loop builds exactly 7 hardcoded tools inline.
        captured = self._capture_loop_first_client_call()
        self.assertEqual(captured["tools"], list(TEAM_MEMBER_LOOP_TOOLS))

    def test_member_loop_does_not_inherit_parent_tools(self):
        # 19: child tool set is the hardcoded 7 — never TodoWrite, task_*,
        # Team management tools, load_skill, compress, etc.
        captured = self._capture_loop_first_client_call()
        forbidden = {
            "TodoWrite", "task_create", "task_get", "task_update", "task_list",
            "task", "spawn_teammate", "list_teammates", "read_inbox",
            "broadcast", "shutdown_request", "plan_approval",
            "load_skill", "compress", "background_run", "check_background",
            "grep_search", "glob_search",
        }
        self.assertEqual(set(captured["tools"]) & forbidden, set())

    def test_member_loop_does_not_use_tool_registry(self):
        # 19: _loop builds tools inline; the tool names are NOT derived from
        # BASE_TOOL_REGISTRY / DEFAULT_TOOL_CONTRIBUTORS (which would be 25).
        captured = self._capture_loop_first_client_call()
        self.assertNotEqual(len(captured["tools"]), 25)
        self.assertEqual(len(captured["tools"]), 7)

    def test_member_loop_hardcodes_max_tokens_8000(self):
        # 19: child does not inherit a parent token budget; it uses 8000.
        captured = self._capture_loop_first_client_call()
        self.assertEqual(captured["max_tokens"], 8000)

    def test_member_loop_does_not_call_agent_loop(self):
        # 19 (structural): _loop has its own loop; it must not recurse into
        # agent_loop. Structural proof: agent_loop is not in team_manager's
        # module namespace.
        tm_globals = self._get_tm_globals()
        self.assertNotIn("agent_loop", tm_globals)
        # And it does not import ToolRegistry-derived names either.
        for name in ("TOOLS", "TOOL_HANDLERS", "BASE_TOOL_REGISTRY",
                      "ToolRegistry", "ToolRegistryOverlay"):
            self.assertNotIn(name, tm_globals)

    def test_member_loop_does_not_import_secure_context_functions(self):
        # 19 (structural): _loop must NOT import or call
        # set_secure_bash_context / reset_secure_bash_context. This is the
        # structural guarantee that D2 did not add independent grant logic.
        tm_globals = self._get_tm_globals()
        self.assertNotIn("set_secure_bash_context", tm_globals)
        self.assertNotIn("reset_secure_bash_context", tm_globals)
        self.assertNotIn("has_valid_secure_bash_context", tm_globals)
        self.assertNotIn("SecureBashContext", tm_globals)

    def test_spawn_starts_daemon_thread(self):
        # 19 (structural): spawn() starts a daemon thread. We verify by
        # inspecting the source of the spawn method.
        import inspect
        source = inspect.getsource(self.module.TEAM.spawn)
        self.assertIn("threading.Thread", source)
        self.assertIn("daemon=True", source)

    def test_shares_global_client_and_model(self):
        # 19: team_manager imports client/MODEL from agents.config — the SAME
        # globals the parent agent_loop uses. This proves it shares Client/Model.
        tm_globals = self._get_tm_globals()
        self.assertIn("client", tm_globals)
        self.assertIn("MODEL", tm_globals)
        # Same object as the harness module's client/MODEL.
        self.assertIs(tm_globals["client"], self.module.client)
        self.assertIs(tm_globals["MODEL"], self.module.MODEL)

    def test_shares_global_bus_and_task_mgr(self):
        # 19: the harness module's TEAM shares the same BUS and TASK_MGR
        # that agent_loop uses.
        self.assertIs(self.module.TEAM.bus, self.module.BUS)
        self.assertIs(self.module.TEAM.task_mgr, self.module.TASK_MGR)

    def test_member_loop_uses_50_round_work_phase(self):
        # 19 (structural): _loop's work phase runs ``for _ in range(50)``.
        # We verify by inspecting the source.
        import inspect
        source = inspect.getsource(self.module.TEAM._loop)
        self.assertIn("range(50)", source)

    def test_member_loop_uses_threading_not_asyncio(self):
        # 19 (structural): team_manager uses threading (not asyncio) for the
        # teammate daemon thread.
        tm_globals = self._get_tm_globals()
        self.assertIn("threading", tm_globals)
        self.assertNotIn("asyncio", tm_globals)


# ---------------------------------------------------------------------------
# 20: Secure Bash Parent→Team-member asymmetry
# ---------------------------------------------------------------------------


class SecureBashAsymmetryTests(unittest.TestCase):
    """D2 must NOT change the Secure Bash asymmetry. Four scenarios:

    a. Parent grant is NOT inherited by teammate thread (ContextVar is
       per-thread → fresh None in the new thread → bash rejected).
    b. No Parent grant → teammate bash rejected (in secure mode).
    c. Teammate lifecycle does NOT revoke Parent grant.
    d. TeamExtension itself does NOT create a secure context.

    Key difference from Subagent (D1):
      - Subagent runs in the SAME asyncio.Task → REUSES Parent's grant.
      - Team member runs in a NEW daemon thread → does NOT inherit the
        ContextVar → has NO grant → bash rejected in secure mode.
    This asymmetry is intentional and must be preserved by D2.
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
        self.has_valid_secure_bash_context = self.base_tools_globals[
            "has_valid_secure_bash_context"
        ]
        self.fake_sandbox = _FakeSandbox()

    def tearDown(self):
        self.base_tools_globals["RUN_MODE"] = self._orig_run_mode
        self.base_tools_globals["SANDBOX"] = self._orig_sandbox
        self._tmp.cleanup()

    def _enable_secure_mode(self):
        """Patch run_bash's module to simulate secure_multi_session mode."""
        self.base_tools_globals["RUN_MODE"] = "secure_multi_session"
        self.base_tools_globals["SANDBOX"] = self.fake_sandbox

    def _run_bash_in_thread(self, command="echo hello"):
        """Run run_bash in a NEW thread (simulating team member execution)
        and return the output. This mirrors how _loop calls run_bash from
        a daemon thread."""
        result = {"output": None}
        error = {"exc": None}

        def thread_target():
            try:
                result["output"] = str(self.run_bash(command))
            except Exception as exc:
                error["exc"] = exc

        t = threading.Thread(target=thread_target, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "bash thread should have finished")
        if error["exc"]:
            raise error["exc"]
        return result["output"]

    def test_parent_grant_not_inherited_by_teammate_thread(self):
        # 20a: Parent secure context is established in the MAIN thread →
        # a NEW thread (team member) does NOT inherit the ContextVar →
        # bash is rejected in the new thread.
        self._enable_secure_mode()
        token = self.set_secure_bash_context(
            run_id="parent-run", sandbox=self.fake_sandbox
        )
        try:
            # Parent (main thread) has a valid grant.
            self.assertTrue(
                self.has_valid_secure_bash_context(self.fake_sandbox)
            )
            # Team member (new thread) does NOT inherit the grant.
            output = self._run_bash_in_thread()
            self.assertIsNotNone(output)
            self.assertIn("Error:", output)
            self.assertIn("secure_multi_session", output)
        finally:
            self.reset_secure_bash_context(token)

    def test_no_parent_grant_blocks_teammate_bash(self):
        # 20b: No SecureBashContext established → team member's bash calls
        # are rejected in secure_multi_session mode. The extension does NOT
        # create its own grant.
        self._enable_secure_mode()
        self.assertFalse(
            self.has_valid_secure_bash_context(self.fake_sandbox)
        )
        output = self._run_bash_in_thread()
        self.assertIsNotNone(output)
        self.assertIn("Error:", output)
        self.assertIn("secure_multi_session", output)

    def test_teammate_lifecycle_does_not_revoke_parent_grant(self):
        # 20c: After the team member thread runs (and its bash is rejected),
        # the parent's secure context is STILL valid — because the team
        # member thread does NOT call reset_secure_bash_context.
        self._enable_secure_mode()
        token = self.set_secure_bash_context(
            run_id="parent-run", sandbox=self.fake_sandbox
        )
        try:
            self.assertTrue(
                self.has_valid_secure_bash_context(self.fake_sandbox)
            )
            # Team member thread runs and fails (no inherited grant).
            self._run_bash_in_thread()
            # Parent grant still valid after team member thread finished.
            self.assertTrue(
                self.has_valid_secure_bash_context(self.fake_sandbox)
            )
            # Parent can still run bash directly.
            parent_bash = str(self.run_bash("echo parent still works"))
            self.assertNotIn("Error:", parent_bash)
            self.assertIn("executed:", parent_bash)
        finally:
            self.reset_secure_bash_context(token)

    def test_team_extension_does_not_create_secure_context(self):
        # 20d (structural): TeamExtension must NOT import or call
        # set_secure_bash_context / reset_secure_bash_context. This is the
        # structural guarantee that D2 did not add independent grant logic.
        ext_globals = vars(self.module.TeamExtension)
        # The class itself should not reference secure-context functions.
        for name in ("set_secure_bash_context", "reset_secure_bash_context",
                      "has_valid_secure_bash_context", "SecureBashContext"):
            self.assertNotIn(name, ext_globals)
        # The contribute_tools method's closure should not capture them.
        import inspect
        source = inspect.getsource(self.module.TeamExtension.contribute_tools)
        for name in ("set_secure_bash_context", "reset_secure_bash_context",
                      "SecureBashContext"):
            self.assertNotIn(name, source)

    def test_team_loop_does_not_import_secure_context_functions(self):
        # 20 (structural): team_manager module (where _loop lives) must NOT
        # import set_secure_bash_context / reset_secure_bash_context.
        tm_globals = self.module.handle_shutdown_request.__globals__
        for name in ("set_secure_bash_context", "reset_secure_bash_context",
                      "has_valid_secure_bash_context", "SecureBashContext"):
            self.assertNotIn(name, tm_globals)


# ---------------------------------------------------------------------------
# 22: Two-Team name-scoped isolation
# ---------------------------------------------------------------------------


class TwoTeamConcurrencyTests(unittest.TestCase):
    """Name-scoped isolation between teammates.

    IMPORTANT: BUS and TASK_MGR are Harness-level GLOBAL singletons with NO
    Team namespace. Isolation is by RECIPIENT NAME, not by Team. If two
    teams both have a member named "alice", they would share the same inbox
    — that is a known Runtime tech debt, NOT fixed by D2.

    These tests verify the CURRENT name-scoped behavior:
      a. Different names → separate inboxes (A1 ≠ A2 ≠ B1 ≠ B2).
      b. shutdown_request to A1 does not appear in A2/B1/B2 inbox.
      c. Task claim is name-scoped (A1's claim doesn't affect B1).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_different_teammates_have_separate_inboxes(self):
        # 22a: messages sent to A1 go to A1's inbox only, not A2 or B1.
        bus = self.module.BUS
        bus.send("lead", "A1", "message for A1")
        bus.send("lead", "A2", "message for A2")
        bus.send("lead", "B1", "message for B1")

        a1_inbox = bus.read_inbox("A1")
        a2_inbox = bus.read_inbox("A2")
        b1_inbox = bus.read_inbox("B1")
        # Also check B2 (never sent anything) is empty.
        b2_inbox = bus.read_inbox("B2")

        self.assertEqual(len(a1_inbox), 1)
        self.assertEqual(a1_inbox[0]["content"], "message for A1")
        self.assertEqual(len(a2_inbox), 1)
        self.assertEqual(a2_inbox[0]["content"], "message for A2")
        self.assertEqual(len(b1_inbox), 1)
        self.assertEqual(b1_inbox[0]["content"], "message for B1")
        self.assertEqual(len(b2_inbox), 0)

    def test_shutdown_request_is_name_scoped(self):
        # 22b: shutdown_request to A1 goes to A1's inbox only.
        bus = self.module.BUS
        self.module.handle_shutdown_request(bus, "A1")

        a1_inbox = bus.read_inbox("A1")
        a2_inbox = bus.read_inbox("A2")
        b1_inbox = bus.read_inbox("B1")

        self.assertEqual(len(a1_inbox), 1)
        self.assertEqual(a1_inbox[0]["type"], "shutdown_request")
        self.assertEqual(len(a2_inbox), 0)
        self.assertEqual(len(b1_inbox), 0)

    def test_broadcast_uses_team_member_names(self):
        # 22: broadcast sends to all names returned by TEAM.member_names().
        # If Team A has [A1, A2] and Team B has [B1, B2], but TEAM is a
        # global singleton, member_names() returns ALL members across both
        # "teams". This is the known tech debt — broadcast is global, not
        # team-scoped.
        team_mgr = self.module.TEAM
        team_mgr.config["members"] = []
        team_mgr.config["members"].extend([
            {"name": "A1", "role": "coder", "status": "working"},
            {"name": "A2", "role": "coder", "status": "working"},
            {"name": "B1", "role": "coder", "status": "working"},
            {"name": "B2", "role": "coder", "status": "working"},
        ])
        team_mgr._save()

        bus = self.module.BUS
        bus.broadcast("lead", "global update", team_mgr.member_names())

        # ALL members get the broadcast (global, not team-scoped).
        for name in ("A1", "A2", "B1", "B2"):
            inbox = bus.read_inbox(name)
            self.assertEqual(len(inbox), 1)
            self.assertEqual(inbox[0]["type"], "broadcast")
            self.assertEqual(inbox[0]["content"], "global update")

    def test_global_bus_and_task_mgr_are_shared_no_team_namespace(self):
        # 22d (tech debt documentation): BUS and TASK_MGR are global
        # singletons. There is no Team namespace. This test DOCUMENTS the
        # current behavior so that a future Runtime Unification phase can
        # deliberately change it.
        # The harness module's TEAM, BUS, TASK_MGR are all module-level
        # globals — there is exactly one instance per harness load.
        self.assertIs(self.module.TEAM.bus, self.module.BUS)
        self.assertIs(self.module.TEAM.task_mgr, self.module.TASK_MGR)
        # The MessageBus has no "team_id" or namespace concept. We verify
        # by inspecting the source of its send method (source inspection on
        # the class itself fails because the module is not in sys.modules
        # after the test loader restores cached modules).
        import inspect
        send_source = inspect.getsource(type(self.module.BUS).send)
        self.assertNotIn("team_id", send_source.lower())
        self.assertNotIn("namespace", send_source.lower())
        # And the class has no team_id / namespace attributes.
        bus_attrs = set(dir(type(self.module.BUS)))
        self.assertNotIn("team_id", bus_attrs)
        self.assertNotIn("namespace", bus_attrs)


# ---------------------------------------------------------------------------
# 23: Legacy order hard contract
# ---------------------------------------------------------------------------


class LegacyOrderTests(unittest.TestCase):
    """LEGACY_*_ORDER constants pin each Team tool to its original slot."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_legacy_team_order_constants(self):
        # 23: each LEGACY_*_ORDER constant equals the tool's index in
        # LEGACY_25_TOOL_NAMES.
        expected = {
            "LEGACY_SPAWN_TEAMMATE_ORDER": "spawn_teammate",
            "LEGACY_LIST_TEAMMATES_ORDER": "list_teammates",
            "LEGACY_SEND_MESSAGE_ORDER": "send_message",
            "LEGACY_READ_INBOX_ORDER": "read_inbox",
            "LEGACY_BROADCAST_ORDER": "broadcast",
            "LEGACY_SHUTDOWN_REQUEST_ORDER": "shutdown_request",
            "LEGACY_PLAN_APPROVAL_ORDER": "plan_approval",
        }
        for const_name, tool_name in expected.items():
            value = getattr(self.module, const_name)
            self.assertEqual(
                value,
                self.module.LEGACY_25_TOOL_NAMES.index(tool_name),
                f"{const_name} should equal index of '{tool_name}'",
            )

    def test_composed_order_matches_legacy_25(self):
        # 23: the default composed registry resolves to EXACTLY
        # LEGACY_25_TOOL_NAMES.
        composed = self.module.build_default_tool_registry()
        names = [t["name"] for t in composed.resolve(profile=None)]
        self.assertEqual(names, self.module.LEGACY_25_TOOL_NAMES)

    def test_team_tools_not_appended_at_end(self):
        # 23: Team tools must NOT be appended at the end of the tool list
        # (which would happen if order defaulted to 0 or was omitted). They
        # must stay at their legacy positions (indices 14-20).
        names = [t["name"] for t in self.module.TOOLS]
        for tool_name in TEAM_TOOL_NAMES:
            idx = names.index(tool_name)
            self.assertGreaterEqual(idx, 14)
            self.assertLessEqual(idx, 20)
            self.assertNotEqual(names[-1], tool_name)


if __name__ == "__main__":
    unittest.main()
