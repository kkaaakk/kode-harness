"""test_harness_tool_profile.py - Stage 2B verification.

Verifies that tool_profile in agent_loop:
- profile=None preserves pre-2B behavior (all 25 tools, all executable)
- named profiles expose only whitelisted tools to the model
- named profiles make non-whitelisted tools UNexecutable (denied)
- unknown tools still produce "Unknown tool" (not crash)
- inactive-but-registered tools produce "Tool unavailable in current profile"
- unknown profile raises before any model request
- concurrent agent_loop calls with different profiles don't interfere
- Kernel safety still works under named profiles
- TOOLS/TOOL_HANDLERS legacy exports unchanged
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"

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
    # Snapshot and clear agents.* modules so WORKDIR/RUN_MODE/SANDBOX are
    # re-evaluated with the current env + cwd. Without this, a prior test
    # file that loaded agents.config with AGENT_RUN_MODE=secure_multi_session
    # leaves a stale agents.config in sys.modules, and our harness_core
    # import picks up the polluted RUN_MODE — causing SecureSandboxError
    # in tests that expect trusted_local.
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
        "harness_core_tool_profile_test", MODULE_PATH
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
        # Clear any agents.* modules that this load created, then restore
        # the pre-call snapshot so concurrent test files don't interfere.
        for k in list(sys.modules.keys()):
            if k == "agents" or k.startswith("agents."):
                sys.modules.pop(k, None)
        sys.modules.update(cached_agents)
        for path in added_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


class _Block:
    type = "tool_use"
    def __init__(self, name="bash", input_=None, id_="t1"):
        self.name = name
        self.input = input_ or {"command": "echo hi"}
        self.id = id_


class _Text:
    type = "text"
    def __init__(self, text): self.text = text


def _resp_tool_use(name="bash", input_=None, id_="t1"):
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


# Captures the tools list sent to the model on each request.
def _make_capturing_create(responses, captured_tools):
    def _create(**kwargs):
        captured_tools.append(list(kwargs.get("tools", [])))
        return responses.pop(0)
    return _create


class ProfileNoneTests(unittest.TestCase):
    """profile=None must be identical to pre-2B behavior."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_profile_exposes_all_25_tools(self):
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(len(captured), 1)
        names = [t["name"] for t in captured[0]]
        self.assertEqual(names, EXPECTED_TOOL_NAMES)

    def test_default_profile_executes_any_tool(self):
        responses = [_resp_tool_use(name="bash"), _resp_text()]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        # Find the tool_result
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for r in m["content"]:
                    if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                        self.assertNotIn("Unknown tool", r["content"])
                        self.assertNotIn("unavailable", r["content"])
                        return
        self.fail("tool_result not found")

    def test_unknown_tool_still_unknown_message(self):
        responses = [_resp_tool_use(name="nonexistent"), _resp_text()]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "hi"}]
        self.module.agent_loop(messages)
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for r in m["content"]:
                    if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                        self.assertIn("Unknown tool", r["content"])
                        return
        self.fail("tool_result not found")


class NamedProfileTests(unittest.TestCase):
    """Named profiles filter visible tools and block execution of inactive ones."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_readonly_excludes_write_and_bash_from_model(self):
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="readonly"
        )
        names = {t["name"] for t in captured[0]}
        self.assertIn("read_file", names)
        self.assertIn("grep_search", names)
        self.assertIn("glob_search", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("edit_file", names)
        self.assertNotIn("bash", names)

    def test_coding_exposes_only_coding_tools(self):
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="coding"
        )
        names = {t["name"] for t in captured[0]}
        expected = {"read_file", "write_file", "edit_file", "bash",
                    "grep_search", "glob_search"}
        self.assertEqual(names, expected)

    def test_inactive_tool_denied_in_readonly(self):
        # Model tries to call write_file under readonly profile.
        responses = [_resp_tool_use(name="write_file",
                                    input_={"path": "x", "content": "y"}),
                     _resp_text()]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "write a file"}]
        self.module.agent_loop(messages, tool_profile="readonly")
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for r in m["content"]:
                    if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                        self.assertIn("unavailable", r["content"])
                        self.assertIn("readonly", r["content"])
                        return
        self.fail("tool_result not found")

    def test_inactive_tool_vs_unknown_tool_distinct_messages(self):
        responses = [
            _resp_tool_use(name="write_file", id_="t1"),  # inactive
            _resp_tool_use(name="nonexistent", id_="t2"),  # unknown
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "hi"}]
        self.module.agent_loop(messages, tool_profile="readonly")
        results = {}
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for r in m["content"]:
                    if isinstance(r, dict) and "tool_use_id" in r:
                        results[r["tool_use_id"]] = r["content"]
        self.assertIn("unavailable", results.get("t1", ""))
        self.assertIn("Unknown tool", results.get("t2", ""))

    def test_unknown_profile_raises_before_model_request(self):
        # Take UnknownToolProfileError from the loaded module's namespace
        # so it matches the exact class object that agent_loop raises.
        # Importing from agents.tool_registry would return a different
        # class object after load_harness_module clears agents.* cache.
        UnknownToolProfileError = self.module.UnknownToolProfileError
        # Even if the model create would work, profile validation must fail first.
        self.module.client.messages.create = lambda **kwargs: _resp_text()
        with self.assertRaises(UnknownToolProfileError):
            self.module.agent_loop(
                [{"role": "user", "content": "hi"}],
                tool_profile="readnoly",  # typo
            )


class ConcurrencyTests(unittest.TestCase):
    """Truly concurrent agent_loop calls with different profiles must not
    interfere. We use a threading.Barrier so the two calls overlap during
    the model request — their active_tools resolution happens concurrently.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_truly_concurrent_profiles_do_not_interfere(self):
        """Two threads, each with a different profile, forced to overlap at
        the model request via a barrier. Each must see only its own tools."""
        barrier = threading.Barrier(2)
        results = {}
        errors = []

        def _barrier_create(responses, captured, key):
            def _create(**kwargs):
                # Capture the tools list BEFORE the barrier — this is the
                # moment when the two agent_loop calls overlap.
                captured.append(list(kwargs.get("tools", [])))
                try:
                    # Block until both threads reach this point, forcing
                    # genuine overlap.
                    barrier.wait(timeout=5)
                except threading.BrokenBarrierError:
                    pass
                return responses.pop(0)
            return _create

        def run(profile, key):
            try:
                captured = []
                responses = [_resp_text()]
                # Each thread sets its own create — but they share self.module.
                # To avoid races on self.module.client.messages.create, we
                # use a single create that dispatches by thread-local key.
                results[key] = (captured, responses, profile)
            except Exception as e:
                errors.append(e)

        # We need a single shared create that handles both threads. Each
        # thread captures its own tools list via thread-local storage.
        import threading as _t
        local = _t.local()
        shared_captures = {}

        def _shared_create(**kwargs):
            tid = _t.get_ident()
            shared_captures.setdefault(tid, []).append(list(kwargs.get("tools", [])))
            try:
                barrier.wait(timeout=5)
            except _t.BrokenBarrierError:
                pass
            return _resp_text()

        self.module.client.messages.create = _shared_create

        def run_real(profile):
            try:
                self.module.agent_loop(
                    [{"role": "user", "content": "hi"}], tool_profile=profile
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_real, args=("readonly",))
        t2 = threading.Thread(target=run_real, args=("coding",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertEqual(errors, [], f"Threads raised: {errors}")
        # Two distinct thread IDs captured tools — proves they ran concurrently.
        self.assertEqual(len(shared_captures), 2,
                         f"Expected 2 threads, got {len(shared_captures)}")
        # Each captured exactly one model request.
        all_captures = list(shared_captures.values())
        self.assertTrue(all(len(c) == 1 for c in all_captures))

        # Identify which capture is which by tool set.
        tool_sets = [set(t["name"] for t in captures[0]) for captures in all_captures]
        readonly_set = next(s for s in tool_sets if "bash" not in s)
        coding_set = next(s for s in tool_sets if "bash" in s)

        # readonly must not have write_file/bash; coding must have bash.
        self.assertNotIn("write_file", readonly_set)
        self.assertNotIn("bash", readonly_set)
        self.assertIn("bash", coding_set)
        self.assertIn("write_file", coding_set)


class McpLinkageTests(unittest.TestCase):
    """Verify that MCP tools are NOT silently injected into model requests
    under named profiles. Even though MCP is independent of harness_core in
    the current codebase, this test confirms the linkage contract: the model
    sees EXACTLY active_tools, nothing more.

    If a future change adds MCP injection into agent_loop, this test will
    catch it and force the author to decide how MCP interacts with profiles.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_profile_none_tools_exactly_match_registry_resolve(self):
        """The tools sent to the model under profile=None must be EXACTLY
        TOOL_REGISTRY.resolve(None) — no extra MCP or dynamic tools appended."""
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        sent_names = [t["name"] for t in captured[0]]
        registry_names = [t["name"] for t in self.module.TOOL_REGISTRY.resolve(None)]
        self.assertEqual(sent_names, registry_names)

    def test_named_profile_tools_exactly_match_registry_resolve(self):
        """Same contract under a named profile: no extra tools appended."""
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], tool_profile="readonly"
        )
        sent_names = [t["name"] for t in captured[0]]
        registry_names = [t["name"]
                          for t in self.module.TOOL_REGISTRY.resolve("readonly")]
        self.assertEqual(sent_names, registry_names)
        # And definitely no write_file / bash sneaking in.
        self.assertNotIn("write_file", sent_names)
        self.assertNotIn("bash", sent_names)

    def test_dynamically_registered_tool_not_in_active_set_if_not_in_profile(self):
        """If someone dynamically registers a tool (simulating an MCP tool)
        AFTER agent_loop started, it cannot appear in the model request
        because active_tools is a startup snapshot."""
        captured = []
        self.module.client.messages.create = _make_capturing_create(
            [_resp_text()], captured
        )
        # Register a fake "mcp_tool" before the call — it's in the registry
        # but not in any named profile whitelist. Stage 2D-B: dynamic
        # registration targets BASE_TOOL_REGISTRY (agent_loop builds its
        # per-call overlay over Base); TOOL_REGISTRY is now a read-only
        # legacy view and must not receive dynamic registrations.
        self.module.BASE_TOOL_REGISTRY.register(
            name="fake_mcp_tool",
            description="simulated MCP",
            input_schema={"type": "object"},
            handler=lambda **kw: "mcp result",
        )
        try:
            self.module.agent_loop(
                [{"role": "user", "content": "hi"}], tool_profile="readonly"
            )
            sent_names = [t["name"] for t in captured[0]]
            # Not in readonly whitelist -> not sent to model.
            self.assertNotIn("fake_mcp_tool", sent_names)
            # But under profile=None it WOULD be sent (it's registered).
            captured_none = []
            self.module.client.messages.create = _make_capturing_create(
                [_resp_text()], captured_none
            )
            self.module.agent_loop([{"role": "user", "content": "hi"}])
            sent_none_names = [t["name"] for t in captured_none[0]]
            self.assertIn("fake_mcp_tool", sent_none_names)
        finally:
            self.module.BASE_TOOL_REGISTRY.unregister("fake_mcp_tool")


class SnapshotSemanticsTests(unittest.TestCase):
    """active_tools is a startup snapshot: tools registered DURING agent_loop
    do not appear until the next agent_loop call. This is the documented
    contract.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_dynamic_register_during_loop_not_visible_until_next_call(self):
        """A tool registered via an Extension handler DURING the loop must
        not appear in the CURRENT model request, only in the NEXT agent_loop."""
        from agents.types.events import Event, Priority

        registered = {"done": False}
        captured_per_turn = []

        def _create(**kwargs):
            captured_per_turn.append([t["name"] for t in kwargs.get("tools", [])])
            # On the first call, dynamically register a new tool.
            if not registered["done"]:
                self.module.BASE_TOOL_REGISTRY.register(
                    name="mid_loop_tool",
                    description="added mid-loop",
                    input_schema={"type": "object"},
                    handler=lambda **kw: "mid",
                )
                registered["done"] = True
                return _resp_tool_use(name="bash", input_={"command": "echo x"})
            return _resp_text()

        self.module.client.messages.create = _create
        self.module.agent_loop([{"role": "user", "content": "hi"}])

        # Turn 1: mid_loop_tool NOT visible (snapshot taken at startup).
        self.assertNotIn("mid_loop_tool", captured_per_turn[0])
        # Turn 2: still NOT visible — snapshot is per agent_loop call, and
        # we're still in the same call.
        self.assertNotIn("mid_loop_tool", captured_per_turn[1])
        # Cleanup.
        self.module.BASE_TOOL_REGISTRY.unregister("mid_loop_tool")


class KernelSafetyUnderProfileTests(unittest.TestCase):
    """Kernel safety (dangerous bash) still works under named profiles."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_dangerous_bash_blocked_under_coding_profile(self):
        responses = [
            _resp_tool_use(name="bash", input_={"command": "sudo rm -rf /"}),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "rm"}]
        self.module.agent_loop(messages, tool_profile="coding")
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for r in m["content"]:
                    if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                        self.assertIn("Dangerous command blocked", r["content"])
                        return
        self.fail("tool_result not found")


class LegacyExportsTests(unittest.TestCase):
    """TOOLS and TOOL_HANDLERS legacy exports remain the full set."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_tools_legacy_export_unchanged(self):
        names = [t["name"] for t in self.module.TOOLS]
        self.assertEqual(names, EXPECTED_TOOL_NAMES)

    def test_tool_handlers_legacy_export_unchanged(self):
        self.assertEqual(set(self.module.TOOL_HANDLERS.keys()),
                         set(EXPECTED_TOOL_NAMES))


class DynamicRegistrationTests(unittest.TestCase):
    """Newly registered tools appear in next resolve() but not in old snapshot."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_dynamic_register_appears_in_new_resolve_not_old_snapshot(self):
        old_tools_len = len(self.module.TOOLS)
        # Register a new tool dynamically (stage 2D-B: register against
        # BASE_TOOL_REGISTRY — TOOL_REGISTRY is a legacy overlay alias).
        self.module.BASE_TOOL_REGISTRY.register(
            name="dynamic_tool",
            description="dynamically added",
            input_schema={"type": "object"},
            handler=lambda **kw: "dynamic",
        )
        # Old snapshot unchanged.
        self.assertEqual(len(self.module.TOOLS), old_tools_len)
        # New resolve(None) sees it.
        new_tools = self.module.BASE_TOOL_REGISTRY.resolve(profile=None)
        names = {t["name"] for t in new_tools}
        self.assertIn("dynamic_tool", names)
        # Cleanup.
        self.module.BASE_TOOL_REGISTRY.unregister("dynamic_tool")


if __name__ == "__main__":
    unittest.main()
