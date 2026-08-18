"""test_subagent_team_contract_snapshot.py - Stage 2D-D0 contract pinning.

D0 does NOT change production logic. It investigates how subagent and team
members actually run today, then locks that behavior into tests so the
upcoming 2D-D1 (subagent migration) and 2D-D2 (team migration) can detect
any accidental semantic shift.

================================================================================
PARENT → CHILD INHERITANCE MATRIX  (the D0 deliverable — pinned from code)
================================================================================
Source of truth at D0 time:
  - agents/subagent.py :: run_subagent  (the ``task`` tool handler)
  - agents/team_manager.py :: TeammateManager._loop  (runs in a daemon thread)
  - agents/base_tools.py :: SecureBashContext / ContextVar / nonce live-set
  - agents/harness_core.py :: agent_loop (secure-context setup + tool dispatch)

Legend:  ✅ = does it   ❌ = does not   ⚠️ = conditional / gap

                              │ Subagent (run_subagent)        │ Team member (_loop)
──────────────────────────────┼────────────────────────────────┼────────────────────────────────
Execution site                │ Sync, in Parent's asyncio.Task │ Separate daemon thread
Calls agent_loop?             │ ❌ own 30-round loop            │ ❌ own 50-round loop
Tool set source               │ Hardcoded in subagent.py       │ Hardcoded in team_manager.py
  → bash                      │ ✅                              │ ✅
  → read_file                 │ ✅                              │ ✅
  → write_file / edit_file    │ ✅ (general-purpose only)      │ ✅
  → TodoWrite                 │ ❌                              │ ❌
  → task_create/get/update    │ ❌                              │ ❌
  → task (subagent)           │ ❌ (cannot recurse)            │ ❌
  → claim_task                │ ❌                              │ ✅ (calls TASK_MGR.claim)
  → send_message              │ ❌                              │ ✅ (calls BUS.send)
  → spawn_teammate / list_…   │ ❌                              │ ❌
  → read_inbox / broadcast    │ ❌                              │ ❌
  → shutdown/plan_approval    │ ❌                              │ ❌
  → idle                      │ ❌                              │ ✅ (own handler)
Tool Profile inherited?       │ ❌ no profile concept           │ ❌ no profile concept
Tool Contributors inherited?  │ ❌                              │ ❌
ToolRegistry used?            │ ❌ builds tools inline          │ ❌ builds tools inline
──────────────────────────────┼────────────────────────────────┼────────────────────────────────
Memory context injected?      │ ❌                              │ ❌
ArtifactStore / session?      │ ❌ (no agent_loop → no store)   │ ❌
Tracing?                      │ ❌ (no tracer calls)           │ ❌
Token budget?                 │ ❌ hardcoded max_tokens=8000   │ ❌ hardcoded max_tokens=8000
Sandbox backend               │ Shared global SANDBOX          │ Shared global SANDBOX
Client / Model                │ Shared global client / MODEL   │ Shared global client / MODEL
Task board (TASK_MGR)         │ Shared (but no task tools)     │ Shared (can claim)
Message bus (BUS)             │ N/A                             │ Shared global BUS
──────────────────────────────┼────────────────────────────────┼────────────────────────────────
SECURE BASH CONTEXT (critical)│                                │
  Parent validated?           │ (Parent ran _validate_secure)  │ (Parent ran _validate_secure)
  Child re-validates?         │ ❌ no _validate_secure call     │ ❌ no _validate_secure call
  Child secure context        │ ⚠️ REUSES Parent's: same Task  │ ⚠️ NONE: new thread → fresh
                              │   shares ContextVar + live     │   ContextVar=None → run_bash
                              │   nonce (no independent grant) │   FAILS in secure_multi_session
                              │   → bash works while Parent    │   mode (no inherited grant,
                              │     agent_loop is running       │   no re-validation)
──────────────────────────────┴────────────────────────────────┴────────────────────────────────

KEY ASYMMETRY (must be settled before D1/D2):
  - Subagent REUSES Parent's secure-bash grant (sync in same Task).
  - Team member has NO secure-bash grant at all (separate thread, fresh ctx),
    so in secure_multi_session mode team-member bash would be rejected.
  Neither child independently re-validates today. D1 (subagent) and D2 (team)
  must FIRST preserve this exact behavior; any change to child re-validation
  is a separate, deliberate decision — never a side-effect of migration.

PROFILE NOTE (resolves the 2D-C wording conflict):
  STANDARD_PROFILES["team"] includes ``task`` (the subagent tool) but NOT
  ``task_create``/``task_get``/``task_update``/``task_list`` (the management
  tools migrated in 2D-C). ``task`` and ``task_*`` are different tools that
  share a prefix — do not conflate them.
================================================================================
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

# ---------------------------------------------------------------------------
# Expected snapshots — pinned from the current source. Any change here is a
# deliberate contract change that D1/D2 must acknowledge.
# ---------------------------------------------------------------------------

# STANDARD_PROFILES exact whitelists (agents/tool_registry.py).
EXPECTED_PROFILES = {
    "coding": (
        "read_file", "write_file", "edit_file", "bash",
        "grep_search", "glob_search",
    ),
    "planning": (
        "read_file", "grep_search", "glob_search", "TodoWrite",
        "task_create", "task_get", "task_update", "task_list",
    ),
    "readonly": (
        "read_file", "grep_search", "glob_search",
    ),
    "team": (
        "read_file", "write_file", "edit_file", "bash",
        "grep_search", "glob_search",
        "task",                       # subagent tool (NOT task_* management)
        "spawn_teammate", "list_teammates", "send_message", "read_inbox",
        "broadcast", "shutdown_request", "plan_approval",
    ),
}

# run_subagent tool exposure (agents/subagent.py).
EXPECTED_SUBAGENT_TOOLS_EXPLORE = ["bash", "read_file"]
EXPECTED_SUBAGENT_TOOLS_GENERAL = ["bash", "read_file", "write_file", "edit_file"]

# TeammateManager._loop tool exposure (agents/team_manager.py).
EXPECTED_TEAM_MEMBER_TOOLS = [
    "bash", "read_file", "write_file", "edit_file",
    "send_message", "idle", "claim_task",
]


def load_harness_module(temp_cwd: Path):
    """Load harness_core.py with mocked Anthropic/dotenv (shared harness)."""
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
        "harness_core_d0_snapshot_test", MODULE_PATH
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

class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


def _resp_text(text="done"):
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[_Text(text)],
        usage=None,
    )


# ---------------------------------------------------------------------------
# 1. Profile snapshot — lock STANDARD_PROFILES exact content
# ---------------------------------------------------------------------------


class ProfileSnapshotTests(unittest.TestCase):
    """Lock the four STANDARD_PROFILES whitelists so any change during D1/D2
    is detected. Resolves the task vs task_* naming conflation."""

    def test_standard_profiles_snapshot(self):
        from agents.tool_registry import STANDARD_PROFILES
        self.assertEqual(set(STANDARD_PROFILES.keys()), set(EXPECTED_PROFILES))
        for name, expected in EXPECTED_PROFILES.items():
            self.assertEqual(
                STANDARD_PROFILES[name], expected,
                f"Profile {name!r} whitelist changed",
            )

    def test_team_profile_contains_subagent_task_not_management_tools(self):
        # The 2D-C wording conflict, resolved: "task" in team is the subagent
        # tool; the four task_* management tools are NOT in team.
        from agents.tool_registry import STANDARD_PROFILES
        team = STANDARD_PROFILES["team"]
        self.assertIn("task", team)
        for mgmt in ("task_create", "task_get", "task_update", "task_list"):
            self.assertNotIn(mgmt, team)

    def test_planning_profile_contains_task_management_tools(self):
        from agents.tool_registry import STANDARD_PROFILES
        planning = STANDARD_PROFILES["planning"]
        for mgmt in ("task_create", "task_get", "task_update", "task_list"):
            self.assertIn(mgmt, planning)
        # planning does NOT include the subagent "task" tool.
        self.assertNotIn("task", planning)

    def test_loaded_registry_uses_these_profiles(self):
        # The loaded harness module's Base registry knows these profiles and
        # resolves them consistently with the snapshot.
        #
        # Stage 2D-D1/2D-D2 update: "task" (subagent) and the seven Team
        # management tools migrated from Base to SubagentExtension and
        # TeamExtension respectively. So:
        #   - BASE_TOOL_REGISTRY.resolve("team") NO LONGER includes "task"
        #     or any Team management tool (they're not Base tools anymore).
        #   - The default COMPOSED registry (Base + TodoExtension +
        #     TaskExtension + SubagentExtension + TeamExtension) resolves
        #     "team" WITH all of them visible — because the extensions
        #     contribute them back.
        # This is exactly the "Profile whitelist names an optional tool that
        # is contributed by an extension" semantic: the profile still works,
        # the tool appears only when the extension is installed.
        tmp = tempfile.TemporaryDirectory()
        try:
            module = load_harness_module(Path(tmp.name))
            for name in EXPECTED_PROFILES:
                self.assertTrue(module.BASE_TOOL_REGISTRY.known_profile(name))
            # Base-only resolve: "task" and Team tools are NOT in Base.
            base_team_tools = [
                t["name"] for t in module.BASE_TOOL_REGISTRY.resolve("team")
            ]
            self.assertNotIn("task", base_team_tools)
            for team_tool in (
                "spawn_teammate", "list_teammates", "send_message",
                "read_inbox", "broadcast", "shutdown_request", "plan_approval",
            ):
                self.assertNotIn(team_tool, base_team_tools)
            # Composed resolve: extensions contribute everything back, so
            # the team profile sees all whitelisted tools again.
            composed = module.build_default_tool_registry()
            composed_team_tools = [t["name"] for t in composed.resolve("team")]
            self.assertIn("task", composed_team_tools)
            for team_tool in (
                "spawn_teammate", "list_teammates", "send_message",
                "read_inbox", "broadcast", "shutdown_request", "plan_approval",
            ):
                self.assertIn(team_tool, composed_team_tools)
        finally:
            tmp.cleanup()


# ---------------------------------------------------------------------------
# 2. Subagent contract snapshot — lock run_subagent's behavior
# ---------------------------------------------------------------------------


class SubagentContractSnapshotTests(unittest.TestCase):
    """Lock run_subagent's tool exposure and independence from ToolRegistry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _capture_subagent_tools(self, agent_type):
        """Run run_subagent with a capturing client; return the tool names
        passed to client.messages.create on the first call.

        Patches via ``run_subagent.__globals__`` so we hit the EXACT module
        namespace the loaded harness uses (load_harness_module restores the
        agents.* sys.modules snapshot in finally, so a fresh ``import
        agents.subagent`` would be a different module object)."""
        captured = {}

        class _FakeMsgs:
            def create(self, **kwargs):
                captured["tools"] = [t["name"] for t in kwargs.get("tools", [])]
                captured["model"] = kwargs.get("model")
                captured["max_tokens"] = kwargs.get("max_tokens")
                return _resp_text("subagent done")

        run_subagent = self.module.run_subagent
        g = run_subagent.__globals__
        original_client = g.get("client")
        g["client"] = types.SimpleNamespace(messages=_FakeMsgs())
        try:
            result = run_subagent("explore this", agent_type)
        finally:
            g["client"] = original_client
        return captured, result

    def test_subagent_explore_exposes_only_bash_and_read(self):
        captured, _ = self._capture_subagent_tools("Explore")
        self.assertEqual(captured["tools"], EXPECTED_SUBAGENT_TOOLS_EXPLORE)

    def test_subagent_general_purpose_adds_write_and_edit(self):
        captured, _ = self._capture_subagent_tools("general-purpose")
        self.assertEqual(captured["tools"], EXPECTED_SUBAGENT_TOOLS_GENERAL)

    def test_subagent_does_not_inherit_todo_task_team_tools(self):
        # The child tool set is the hardcoded 2-4 tools — never TodoWrite,
        # task_*, team tools, or the subagent "task" tool (no recursion).
        captured, _ = self._capture_subagent_tools("general-purpose")
        forbidden = {
            "TodoWrite", "task_create", "task_get", "task_update", "task_list",
            "task", "spawn_teammate", "list_teammates", "send_message",
            "read_inbox", "broadcast", "shutdown_request", "plan_approval",
            "claim_task", "idle", "load_skill", "compress",
            "background_run", "check_background", "grep_search", "glob_search",
        }
        self.assertEqual(set(captured["tools"]) & forbidden, set())

    def test_subagent_does_not_use_tool_registry_or_profile(self):
        # run_subagent builds tools inline; the tool names passed to the model
        # are NOT derived from BASE_TOOL_REGISTRY / DEFAULT_TOOL_CONTRIBUTORS
        # (which would be 25 tools). This is the key independence contract.
        captured, _ = self._capture_subagent_tools("Explore")
        self.assertNotEqual(len(captured["tools"]), 25)
        self.assertEqual(len(captured["tools"]), 2)

    def test_subagent_hardcodes_max_tokens_not_parent_budget(self):
        # Child does not inherit a parent token budget; it uses a fixed cap.
        captured, _ = self._capture_subagent_tools("Explore")
        self.assertEqual(captured["max_tokens"], 8000)

    def test_subagent_does_not_call_agent_loop(self):
        # run_subagent has its own loop; it must not recurse into agent_loop.
        # Structural proof: agent_loop is not even in run_subagent's module
        # namespace, so the function has no way to call it. (run_subagent
        # imports only client/MODEL/base_tools — not harness_core.agent_loop.)
        run_subagent = self.module.run_subagent
        self.assertNotIn("agent_loop", run_subagent.__globals__)
        # And it does not import ToolRegistry-derived names either.
        for name in ("TOOLS", "TOOL_HANDLERS", "BASE_TOOL_REGISTRY"):
            self.assertNotIn(name, run_subagent.__globals__)


# ---------------------------------------------------------------------------
# 3. Team member contract snapshot — lock TeammateManager._loop tool exposure
# ---------------------------------------------------------------------------


class TeamMemberContractSnapshotTests(unittest.TestCase):
    """Lock the team-member tool set and its independence from ToolRegistry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _capture_team_member_tools(self):
        """Call TEAM._loop directly (sync) with a capturing client that
        raises after the first call, so _loop exits via its except-branch.

        Patches via ``type(TEAM).__globals__`` (the TeammateManager module
        namespace) so the method's global ``client`` lookup hits our fake —
        the same module the loaded harness's TEAM instance belongs to."""
        captured = {}

        class _FakeMsgs:
            def create(self, **kwargs):
                captured["tools"] = [t["name"] for t in kwargs.get("tools", [])]
                captured["system"] = kwargs.get("system")
                captured["max_tokens"] = kwargs.get("max_tokens")
                raise RuntimeError("capture-and-stop")

        team = self.module.TEAM
        g = team._loop.__func__.__globals__  # method's module namespace
        original_client = g.get("client")
        g["client"] = types.SimpleNamespace(messages=_FakeMsgs())
        try:
            # Call _loop directly (not via spawn/thread). _loop reads inbox
            # (empty for a fresh name), then calls client → captures + raises
            # → _loop's except-branch sets status shutdown and returns.
            team._loop("snap-member", "tester", "do something")
        finally:
            g["client"] = original_client
        return captured

    def test_team_member_tool_set_snapshot(self):
        captured = self._capture_team_member_tools()
        self.assertEqual(captured["tools"], EXPECTED_TEAM_MEMBER_TOOLS)

    def test_team_member_does_not_inherit_todo_task_team_admin_tools(self):
        # Team members get a fixed 7-tool set — no TodoWrite, no task_*,
        # no subagent "task", no team-admin tools (spawn/list/broadcast/...).
        captured = self._capture_team_member_tools()
        forbidden = {
            "TodoWrite", "task_create", "task_get", "task_update", "task_list",
            "task", "spawn_teammate", "list_teammates", "read_inbox",
            "broadcast", "shutdown_request", "plan_approval",
            "load_skill", "compress", "background_run", "check_background",
            "grep_search", "glob_search",
        }
        self.assertEqual(set(captured["tools"]) & forbidden, set())

    def test_team_member_does_not_use_tool_registry(self):
        # The member tool set is hardcoded (7 tools), NOT the 25-tool
        # composed registry. Independence contract for D2.
        captured = self._capture_team_member_tools()
        self.assertNotEqual(len(captured["tools"]), 25)
        self.assertEqual(len(captured["tools"]), 7)

    def test_team_member_hardcodes_max_tokens(self):
        captured = self._capture_team_member_tools()
        self.assertEqual(captured["max_tokens"], 8000)

    def test_team_member_system_prompt_includes_identity(self):
        # The member system prompt carries name/role/team identity (this is
        # the only "inherited" context — identity strings, not capabilities).
        captured = self._capture_team_member_tools()
        self.assertIsNotNone(captured["system"])
        self.assertIn("snap-member", captured["system"])
        self.assertIn("tester", captured["system"])


# ---------------------------------------------------------------------------
# 4. Default composition still 25 (cross-reference — D1 updated this lock)
# ---------------------------------------------------------------------------


class DefaultCompositionStillLockedTests(unittest.TestCase):
    """D2 update: the seven Team management tools migrated from Base to
    TeamExtension. So Base = 12 (was 19), composed = 25 (unchanged),
    and DEFAULT_TOOL_CONTRIBUTORS now includes TeamExtension."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_is_12_composed_is_25(self):
        # Stage 2D-D2: Base = 12 (TodoWrite + 4 task tools + subagent + 7 Team
        # tools all migrated out). Default composed = 25 (all contributed back).
        self.assertEqual(
            len(self.module.BASE_TOOL_REGISTRY.resolve(profile=None)), 12
        )
        self.assertEqual(len(self.module.TOOLS), 25)

    def test_default_contributors_are_todo_task_subagent_and_team(self):
        # Stage 2D-D2: TeamExtension joined DEFAULT_TOOL_CONTRIBUTORS.
        names = [type(c).__name__ for c in self.module.DEFAULT_TOOL_CONTRIBUTORS]
        self.assertEqual(
            names,
            ["TodoExtension", "TaskExtension", "SubagentExtension", "TeamExtension"],
        )

    def test_subagent_task_tool_migrated_to_extension(self):
        # Stage 2D-D1: "task" (subagent) is NO LONGER in Base — it is
        # contributed by SubagentExtension. The composed registry sees it.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        self.assertNotIn("task", base_names)
        composed_names = set(self.module.TOOL_REGISTRY.all_names())
        self.assertIn("task", composed_names)
        # The owner is the extension, not kernel.
        entry = self.module.TOOL_REGISTRY["task"]
        self.assertEqual(entry.owner, "subagent-extension")

    def test_team_tools_migrated_to_extension(self):
        # Stage 2D-D2: the seven Team management tools are NO LONGER in Base —
        # they are contributed by TeamExtension. The composed registry sees them.
        team_tool_names = (
            "spawn_teammate", "list_teammates", "send_message", "read_inbox",
            "broadcast", "shutdown_request", "plan_approval",
        )
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        composed_names = set(self.module.TOOL_REGISTRY.all_names())
        for name in team_tool_names:
            self.assertNotIn(name, base_names)
            self.assertIn(name, composed_names)
            entry = self.module.TOOL_REGISTRY[name]
            self.assertEqual(entry.owner, "team-extension")


if __name__ == "__main__":
    unittest.main()
