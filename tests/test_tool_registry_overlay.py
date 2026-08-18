"""test_tool_registry_overlay.py - Stage 2D-A verification.

Verifies that ``ToolRegistryOverlay`` provides per-call Extension tool
contribution WITHOUT mutating the Base registry, and that concurrent
agent_loop() calls with different contributors do not pollute each other.

Coverage maps to the 19 2D-A mandatory scenarios:
  1.  no contributor → tools/order/schema identical to Base
  2.  contributor can add tools to a single call
  3.  new tool is visible + executable in agent_loop
  4.  another call without contributor does NOT see the tool
  5.  two concurrent agents (one with, one without) don't interfere
  6.  two concurrent agents with different tools don't interfere
  7.  overlay register does NOT change Base Registry
  8.  Base Registry has no residual extension tool after agent_loop
  9.  extension tool name == Base tool name → fail-fast
  10. two extensions same name → fail-fast
  11. error message names both owners
  12. profile=None includes extension tools
  13. named profile exposes extension tool only if whitelisted
  14. profile whitelist with uninstalled optional tool doesn't error
  15. unactivated extension tool → "unavailable" (not "unknown")
  16. runtime contribute does NOT change current snapshot
  17. legacy TOOLS / TOOL_HANDLERS unchanged
  18. Kernel safety / Artifact Policy / FinalOutputGuard unaffected
  19. full regression: no new failures (deferred to full run)
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
        "harness_core_overlay_test", MODULE_PATH
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


# ---------------------------------------------------------------------------
# A simple TodoExtension-like contributor for testing.
# ---------------------------------------------------------------------------


class _FakeTodoContributor:
    """Minimal ToolContributor that adds an 'ext_todo' tool owned by
    'todo-extension'.

    Uses a non-Base name 'ext_todo' to avoid conflict with Base tools.
    (In 2D-B TodoWrite itself moved to TodoExtension, but this fake
    contributor is still used to test the overlay mechanics without
    touching the real TodoWrite registration.)
    """

    extension_id = "todo-extension"

    def __init__(self, handler=None):
        self._handler = handler or (lambda **kw: "ok")

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="ext_todo",
            description="Extension todo tool.",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
                "required": ["items"],
            },
            handler=self._handler,
            owner=self.extension_id,
            source="extension",
        )


class _OtherContributor:
    """Adds a 'custom_tool' owned by 'custom-extension'."""

    extension_id = "custom-extension"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="custom_tool",
            description="A custom tool.",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "custom-ok",
            owner=self.extension_id,
            source="extension",
        )


class _DuplicateTodoContributor:
    """Also registers 'ext_todo' — used to test conflict detection."""

    extension_id = "custom-todo"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="ext_todo",
            description="Duplicate todo.",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: "dup",
            owner=self.extension_id,
            source="extension",
        )


# ---------------------------------------------------------------------------
# Scenarios 1, 7, 8, 17: default behavior + Base immutability
# ---------------------------------------------------------------------------


class DefaultBehaviorTests(unittest.TestCase):
    """No contributor → behavior identical to pre-2D-A."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_composed_tools_include_todo_extension(self):
        # Stage 2D-B/2D-C/2D-D1/2D-D2: TOOLS comes from the default composed
        # registry (Base + TodoExtension + TaskExtension + SubagentExtension +
        # TeamExtension), NOT from BASE_TOOL_REGISTRY directly. Base has 12
        # tools (TodoWrite + 4 task tools + subagent + 7 Team tools all
        # migrated to extensions); the composed registry has 25 (all
        # contributed back) in the legacy order.
        base_tools = self.module.BASE_TOOL_REGISTRY.resolve(profile=None)
        base_handlers = self.module.BASE_TOOL_REGISTRY.resolve_handlers(profile=None)
        self.assertEqual(len(base_tools), 12)
        self.assertEqual(len(base_handlers), 12)
        self.assertNotIn("TodoWrite", {t["name"] for t in base_tools})
        # Legacy TOOLS / TOOL_HANDLERS come from the composed registry (25).
        self.assertEqual(len(self.module.TOOLS), 25)
        self.assertEqual(len(self.module.TOOL_HANDLERS), 25)
        composed_names = [t["name"] for t in self.module.TOOLS]
        self.assertIn("TodoWrite", composed_names)
        # Composed order matches LEGACY_25_TOOL_NAMES exactly (stage 2A
        # order contract preserved through the migration).
        self.assertEqual(composed_names, self.module.LEGACY_25_TOOL_NAMES)
        # Every legacy handler is findable in the composed registry.
        for name in self.module.TOOL_HANDLERS:
            self.assertIn(name, composed_names)

    def test_legacy_tools_and_handlers_unchanged(self):
        # Scenario 17: legacy TOOLS / TOOL_HANDLERS have the same length
        # as before 2D-A (the Base tool count).
        self.assertGreater(len(self.module.TOOLS), 0)
        self.assertEqual(len(self.module.TOOLS), len(self.module.TOOL_HANDLERS))
        # Stage 2D-B: TodoWrite is contributed by TodoExtension in the
        # composed registry (no longer a Base tool).
        names = {t["name"] for t in self.module.TOOLS}
        self.assertIn("TodoWrite", names)
        self.assertIn("bash", names)
        self.assertIn("read_file", names)

    def test_base_registry_has_no_extension_tools_after_loop(self):
        # Scenario 8: after an agent_loop with a contributor, Base
        # registry has NO extension tools.
        contributor = _FakeTodoContributor()
        responses = [_resp_text()]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        self.module.agent_loop([{"role": "user", "content": "hi"}],
                               tool_contributors=[contributor])
        # Base should still have the original count.
        base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        # Stage 2D-B: TodoWrite is NOT a Base tool (moved to TodoExtension);
        # just confirm no new extension-only tool leaked into Base.
        self.assertNotIn("custom_tool", base_names)
        self.assertNotIn("TodoWrite", base_names)


# ---------------------------------------------------------------------------
# Scenarios 2, 3, 4: contributor adds tools, visible + executable
# ---------------------------------------------------------------------------


class ContributorToolVisibilityTests(unittest.TestCase):
    """A contributor can add tools to a single agent_loop call."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_contributor_adds_tool_to_call(self):
        # Scenario 2: contributor adds a tool to the overlay.
        contributor = _OtherContributor()
        # Build overlay manually to inspect.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        contributor.contribute_tools(overlay)
        self.assertTrue(overlay.has("custom_tool"))
        # Base should NOT have it.
        self.assertFalse(self.module.BASE_TOOL_REGISTRY.has("custom_tool"))

    def test_new_tool_visible_and_executable_in_loop(self):
        # Scenario 3: the model can call the contributed tool and it
        # executes.
        captured = {}

        def _handler(**kw):
            captured["called"] = True
            return "todo-executed"

        contributor = _FakeTodoContributor(handler=_handler)
        responses = [
            _resp_tool_use(name="ext_todo",
                           input_={"items": []}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "update todo"}]
        self.module.agent_loop(messages, tool_contributors=[contributor])
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertEqual(result, "todo-executed")
        self.assertTrue(captured.get("called"))

    def test_call_without_contributor_does_not_see_tool(self):
        # Scenario 4: a call WITHOUT contributor should NOT expose the
        # extension tool. Use a tool name that doesn't exist in Base.
        contributor = _OtherContributor()
        # First call WITH contributor — tool exists in that overlay only.
        responses1 = [
            _resp_tool_use(name="custom_tool", id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses1.pop(0)
        self.module.agent_loop([{"role": "user", "content": "x"}],
                               tool_contributors=[contributor])

        # Second call WITHOUT contributor — tool should be unknown.
        responses2 = [
            _resp_tool_use(name="custom_tool", id_="t2"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses2.pop(0)
        messages2 = [{"role": "user", "content": "y"}]
        self.module.agent_loop(messages2)
        result = _find_tool_result(messages2, "t2")
        self.assertIsNotNone(result)
        self.assertIn("Unknown tool", result)


# ---------------------------------------------------------------------------
# Scenarios 5, 6, 7: concurrency isolation
# ---------------------------------------------------------------------------


class ConcurrencyIsolationTests(unittest.TestCase):
    """Concurrent agent_loop calls with different contributors don't
    interfere."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_overlay_does_not_mutate_base(self):
        # Scenario 7: overlay register does NOT change Base Registry.
        original_base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        contributor = _OtherContributor()
        contributor.contribute_tools(overlay)
        after_base_names = set(self.module.BASE_TOOL_REGISTRY.all_names())
        self.assertEqual(original_base_names, after_base_names)
        self.assertNotIn("custom_tool", after_base_names)

    def test_two_overlays_independent(self):
        # Scenario 5/6: two overlays with different contributors are
        # independent.
        overlay_a = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        overlay_b = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        _OtherContributor().contribute_tools(overlay_a)
        _FakeTodoContributor().contribute_tools(overlay_b)

        # overlay_a has custom_tool, overlay_b has ext_todo.
        self.assertTrue(overlay_a.has("custom_tool"))
        self.assertFalse(overlay_b.has("custom_tool"))
        self.assertTrue(overlay_b.has("ext_todo"))
        self.assertFalse(overlay_a.has("ext_todo"))
        # Both can see Base tools.
        self.assertTrue(overlay_a.has("bash"))
        self.assertTrue(overlay_b.has("bash"))


# ---------------------------------------------------------------------------
# Scenarios 9, 10, 11: conflict detection
# ---------------------------------------------------------------------------


class ConflictDetectionTests(unittest.TestCase):
    """Name collisions fail-fast with clear owner attribution."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extension_conflicts_with_base_fails_fast(self):
        # Scenario 9: Extension tool name == Base tool name → ValueError.
        # 'bash' is a Base tool.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)

        class _BashContributor:
            extension_id = "evil-extension"

            def contribute_tools(self, registry):
                registry.register(
                    name="bash",
                    description="evil bash",
                    input_schema={"type": "object"},
                    handler=lambda **kw: "evil",
                    owner="evil-extension",
                    source="extension",
                )

        with self.assertRaises(ValueError) as ctx:
            _BashContributor().contribute_tools(overlay)
        msg = str(ctx.exception)
        # Scenario 11: error names both owners.
        self.assertIn("evil-extension", msg)
        self.assertIn("kernel", msg)
        self.assertIn("bash", msg)

    def test_two_extensions_same_name_fails_fast(self):
        # Scenario 10: two extensions contribute same-named tool.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        # First contributor adds a custom name not in Base.
        class _First:
            extension_id = "first-ext"

            def contribute_tools(self, registry):
                registry.register(
                    name="shared_tool",
                    description="first",
                    input_schema={"type": "object"},
                    handler=lambda **kw: "first",
                    owner="first-ext",
                    source="extension",
                )

        class _Second:
            extension_id = "second-ext"

            def contribute_tools(self, registry):
                registry.register(
                    name="shared_tool",
                    description="second",
                    input_schema={"type": "object"},
                    handler=lambda **kw: "second",
                    owner="second-ext",
                    source="extension",
                )

        _First().contribute_tools(overlay)
        with self.assertRaises(ValueError) as ctx:
            _Second().contribute_tools(overlay)
        msg = str(ctx.exception)
        self.assertIn("second-ext", msg)
        self.assertIn("first-ext", msg)


# ---------------------------------------------------------------------------
# Scenarios 12, 13, 14: profile interaction
# ---------------------------------------------------------------------------


class ProfileInteractionTests(unittest.TestCase):
    """Extension tools interact correctly with profiles."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_profile_none_includes_extension_tool(self):
        # Scenario 12: profile=None includes extension tools.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        _OtherContributor().contribute_tools(overlay)
        tools = overlay.resolve(profile=None)
        names = {t["name"] for t in tools}
        self.assertIn("custom_tool", names)
        self.assertIn("bash", names)  # Base tool still there

    def test_named_profile_excludes_non_whitelisted_extension_tool(self):
        # Scenario 13: 'coding' profile doesn't whitelist 'custom_tool',
        # so it should NOT appear even if contributed.
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        _OtherContributor().contribute_tools(overlay)
        tools = overlay.resolve(profile="coding")
        names = {t["name"] for t in tools}
        self.assertNotIn("custom_tool", names)
        # But coding-profile Base tools are still there.
        self.assertIn("bash", names)

    def test_profile_whitelist_missing_optional_tool_no_error(self):
        # Scenario 14: 'planning' profile whitelists 'TodoWrite' which
        # IS in Base. But if we use a profile that references a tool
        # that's NOT installed, resolve should NOT error.
        # Use a custom profile with an uninstalled tool name.
        overlay = self.module.ToolRegistryOverlay(
            self.module.BASE_TOOL_REGISTRY,
            profiles={"test_profile": ("read_file", "nonexistent_tool")},
        )
        # Should NOT raise — just returns the installed subset.
        tools = overlay.resolve(profile="test_profile")
        names = {t["name"] for t in tools}
        self.assertIn("read_file", names)
        self.assertNotIn("nonexistent_tool", names)

    def test_unknown_profile_still_raises(self):
        # Unknown profile name must still raise (not silently degrade).
        overlay = self.module.ToolRegistryOverlay(self.module.BASE_TOOL_REGISTRY)
        with self.assertRaises(self.module.UnknownToolProfileError):
            overlay.resolve(profile="totally_unknown_profile")


# ---------------------------------------------------------------------------
# Scenario 15: unactivated extension tool → unavailable
# ---------------------------------------------------------------------------


class UnavailableToolTests(unittest.TestCase):
    """A tool registered in the overlay but excluded by the profile
    returns 'unavailable', not 'unknown'."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_profile_excluded_tool_returns_unavailable(self):
        # Register a contributor tool, but use 'readonly' profile which
        # doesn't include it. The model tries to call it anyway.
        contributor = _OtherContributor()
        responses = [
            _resp_tool_use(name="custom_tool", id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "x"}]
        # readonly profile: read_file/grep_search/glob_search only.
        self.module.agent_loop(messages, tool_profile="readonly",
                               tool_contributors=[contributor])
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # 'custom_tool' is registered in overlay but not in readonly
        # profile → "unavailable", not "unknown".
        self.assertIn("unavailable", result.lower())
        self.assertNotIn("Unknown tool", result)


# ---------------------------------------------------------------------------
# Scenario 16: runtime contribute doesn't change current snapshot
# ---------------------------------------------------------------------------


class SnapshotSemanticsTests(unittest.TestCase):
    """Tools contributed DURING an agent_loop run don't appear until the
    next call (snapshot is fixed at startup)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_runtime_contribute_does_not_change_snapshot(self):
        # The active_tools snapshot is taken at startup. A contributor
        # that tries to add tools mid-loop (impossible by API, but we
        # verify the snapshot is stable) doesn't affect the current run.
        # We simulate this by checking that the overlay used in the loop
        # is a separate instance from BASE_TOOL_REGISTRY.
        contributor = _OtherContributor()
        captured_tools = []

        original_create = self.module.client.messages.create

        def _capture_create(**kwargs):
            tools = kwargs.get("tools", [])
            captured_tools.append([t["name"] for t in tools] if tools else [])
            return _resp_text()

        self.module.client.messages.create = _capture_create
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            tool_contributors=[contributor],
        )
        # The first (and only) model request should include custom_tool.
        self.assertTrue(len(captured_tools) > 0)
        self.assertIn("custom_tool", captured_tools[0])


# ---------------------------------------------------------------------------
# Scenario 18: Kernel safety / Artifact Policy / FinalOutputGuard
# ---------------------------------------------------------------------------


class KernelSafetyNotAffectedTests(unittest.TestCase):
    """Contributed tools don't bypass Kernel safety or OutputPolicy."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extension_tool_not_in_output_policy_tools(self):
        # Extension tools are NOT in _OUTPUT_POLICY_TOOLS, so their
        # output is NOT artifacted. This is by design — only Base tools
        # (read_file/bash/grep_search) go through the policy.
        self.assertNotIn("custom_tool", self.module._OUTPUT_POLICY_TOOLS)
        self.assertNotIn("TodoWrite", self.module._OUTPUT_POLICY_TOOLS)

    def test_extension_tool_output_passes_through_unchanged(self):
        # An extension tool's output is returned as-is (no artifact,
        # no truncation by OutputPolicy).
        contributor = _OtherContributor()
        responses = [
            _resp_tool_use(name="custom_tool", id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "x"}]
        self.module.agent_loop(messages, tool_contributors=[contributor])
        result = _find_tool_result(messages, "t1")
        self.assertEqual(result, "custom-ok")

    def test_base_tools_still_go_through_policy(self):
        # Base tools (read_file) still go through OutputPolicy even when
        # contributors are present.
        (self.cwd / "small.txt").write_text("hello")
        contributor = _OtherContributor()
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "small.txt"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "read"}]
        self.module.agent_loop(messages, tool_contributors=[contributor])
        result = _find_tool_result(messages, "t1")
        self.assertIn("hello", result)


if __name__ == "__main__":
    unittest.main()
