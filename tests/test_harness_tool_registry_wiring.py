"""test_harness_tool_registry_wiring.py - Stage 2A verification.

Verifies that wiring ToolRegistry into harness_core preserves all pre-2A
behavior: tool count, names, order, schemas, handlers, unknown-tool error,
Kernel safety, and full agent_loop execution.

Backward-compat: TOOLS and TOOL_HANDLERS still exist and are derived from
TOOL_REGISTRY.resolve(profile=None).
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

# The canonical pre-2A tool names in order. Any change here is a behavior
# change that must be deliberate.
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
    # file that loaded agents.config with a different AGENT_RUN_MODE leaves
    # a stale agents.config in sys.modules, and our harness_core import
    # picks up the polluted RUN_MODE.
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
        "harness_core_tool_registry_wiring_test", MODULE_PATH
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


class ToolRegistryWiringTests(unittest.TestCase):
    """Stage 2A: TOOLS and TOOL_HANDLERS are derived from TOOL_REGISTRY."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.module = load_harness_module(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    # 1. Tool count unchanged
    def test_tool_count_unchanged(self):
        self.assertEqual(len(self.module.TOOLS), 25)
        self.assertEqual(len(self.module.TOOL_HANDLERS), 25)
        self.assertEqual(len(self.module.TOOL_REGISTRY.all_names()), 25)

    # 2. Tool names and order unchanged
    def test_tool_names_and_order_unchanged(self):
        names = [t["name"] for t in self.module.TOOLS]
        self.assertEqual(names, EXPECTED_TOOL_NAMES)
        # TOOL_HANDLERS keys match (dict order also preserved in Python 3.7+)
        self.assertEqual(list(self.module.TOOL_HANDLERS.keys()), EXPECTED_TOOL_NAMES)

    # 3. Tool schemas unchanged (each schema is a dict with name/description/input_schema)
    def test_tool_schemas_have_required_fields(self):
        for tool in self.module.TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)
            self.assertIsInstance(tool["input_schema"], dict)

    def test_specific_tool_schemas_unchanged(self):
        """Spot-check a few critical tool schemas for exact shape."""
        tools_by_name = {t["name"]: t for t in self.module.TOOLS}

        # bash
        self.assertEqual(tools_by_name["bash"]["input_schema"]["required"], ["command"])
        # write_file
        self.assertEqual(
            sorted(tools_by_name["write_file"]["input_schema"]["required"]),
            ["content", "path"],
        )
        # grep_search has a description on pattern field
        grep_schema = tools_by_name["grep_search"]["input_schema"]
        self.assertIn("pattern", grep_schema["properties"])
        self.assertEqual(grep_schema["required"], ["pattern"])

    # 4. Every old tool handler is findable via Registry
    def test_every_tool_handler_findable_via_registry(self):
        for name in EXPECTED_TOOL_NAMES:
            handler = self.module.TOOL_REGISTRY.get_handler(name)
            self.assertIsNotNone(handler, f"Handler for {name} not found in registry")
            # Same handler object as TOOL_HANDLERS
            self.assertIs(handler, self.module.TOOL_HANDLERS[name])

    # 5. Unknown tool returns None (not raising) — caller produces "Unknown tool: xxx"
    def test_unknown_tool_handler_returns_none(self):
        self.assertIsNone(self.module.TOOL_REGISTRY.get_handler("nonexistent_tool"))
        self.assertIsNone(self.module.TOOL_HANDLERS.get("nonexistent_tool"))

    # 6. profile=None returns all tools (no filtering)
    def test_profile_none_returns_all_tools(self):
        all_tools = self.module.TOOL_REGISTRY.resolve(profile=None)
        self.assertEqual(len(all_tools), 25)
        self.assertEqual([t["name"] for t in all_tools], EXPECTED_TOOL_NAMES)

    # 7. Duplicate registration rejected by default
    def test_duplicate_registration_rejected(self):
        from agents.tool_registry import ToolRegistry
        reg = ToolRegistry()
        reg.register("bash", "v1", {"type": "object"}, lambda **kw: None)
        with self.assertRaises(ValueError):
            reg.register("bash", "v2", {"type": "object"}, lambda **kw: None)

    def test_duplicate_registration_allowed_with_overwrite(self):
        from agents.tool_registry import ToolRegistry
        reg = ToolRegistry()
        reg.register("bash", "v1", {"type": "object"}, lambda **kw: "v1")
        reg.register("bash", "v2", {"type": "object"}, lambda **kw: "v2",
                     overwrite=True)
        entry = reg.get("bash")
        self.assertEqual(entry.description, "v2")

    # 8. Registry clear does NOT affect the already-derived TOOLS list
    def test_registry_clear_does_not_mutate_derived_tools(self):
        # Stage 2D-B.1: TOOL_REGISTRY is a read-only legacy view — clear()
        # raises TypeError to direct mutation to BASE_TOOL_REGISTRY.
        with self.assertRaises(TypeError):
            self.module.TOOL_REGISTRY.clear()
        # TOOLS was derived as a snapshot list at import; clearing the Base
        # registry doesn't retroactively change the list agent_loop closes over.
        original_len = len(self.module.TOOLS)
        self.module.BASE_TOOL_REGISTRY.clear()
        self.assertEqual(len(self.module.TOOLS), original_len)

    # 9. Empty Extension + default Registry runs full agent_loop
    def test_empty_extension_default_registry_runs_full_loop(self):
        class TextBlock:
            type = "text"
            def __init__(self, text): self.text = text

        class ToolBlock:
            type = "tool_use"
            def __init__(self, name="bash", input_=None, id_="t1"):
                self.name = name
                self.input = input_ or {"command": "echo hi"}
                self.id = id_

        responses = [
            types.SimpleNamespace(
                stop_reason="tool_use",
                content=[ToolBlock()],
                usage=None,
            ),
            types.SimpleNamespace(
                stop_reason="end_turn",
                content=[TextBlock("done")],
                usage=None,
            ),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run echo"}]
        # Should complete without raising.
        self.module.agent_loop(messages)
        self.assertEqual(messages[-1]["role"], "assistant")

    # 10. Unknown tool in agent_loop produces "Unknown tool" message (not crash)
    def test_unknown_tool_in_loop_produces_unknown_message(self):
        class ToolBlock:
            type = "tool_use"
            def __init__(self, name, input_, id_):
                self.name = name
                self.input = input_
                self.id = id_

        class TextBlock:
            type = "text"
            def __init__(self, text): self.text = text

        responses = [
            types.SimpleNamespace(
                stop_reason="tool_use",
                content=[ToolBlock("nonexistent_tool", {}, "t1")],
                usage=None,
            ),
            types.SimpleNamespace(
                stop_reason="end_turn",
                content=[TextBlock("ok")],
                usage=None,
            ),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "hi"}]
        self.module.agent_loop(messages)
        # Find the tool_result for t1
        user_msgs = [m for m in messages
                     if m.get("role") == "user" and isinstance(m.get("content"), list)]
        result = next(r for r in user_msgs[0]["content"]
                      if isinstance(r, dict) and r.get("tool_use_id") == "t1")
        self.assertIn("Unknown tool", result["content"])

    # 11. Kernel safety still blocks dangerous bash after Registry wiring
    def test_dangerous_bash_still_blocked(self):
        handler = self.module.TOOL_HANDLERS["bash"]
        result = handler(command="sudo rm -rf /")
        self.assertIn("Dangerous command blocked", result)

    # 12. Tool call args and result shape (stage 2C-B2A: bash now returns
    # BashExecutionResult, not str; dangerous/timeout still return str).
    def test_bash_handler_args_and_result_shape(self):
        handler = self.module.TOOL_HANDLERS["bash"]
        result = handler(command="echo hello")
        # Normal execution: structured result.
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)
        # Backward-compatible str() includes the exit code line.
        self.assertIn("[exit_code: 0]", str(result))


if __name__ == "__main__":
    unittest.main()
