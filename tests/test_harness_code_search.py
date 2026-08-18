"""Tests verifying harness_core.py exposes grep_search and glob_search tools."""

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
    """Load harness_core.py in isolation with mocked dependencies."""
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
    # Snapshot and clear agents.* modules so WORKDIR is re-evaluated with
    # the temp cwd. Without this, a prior test file that loaded
    # agents.config leaves a stale agents.config in sys.modules, and our
    # harness_core import picks up the polluted WORKDIR — causing the
    # grep_search handler to search the WRONG directory (flaky failure
    # in test_grep_search_handler_calls_with_workdir).
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
    spec = importlib.util.spec_from_file_location("harness_core_code_search_under_test", MODULE_PATH)
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


class HarnessCodeSearchTests(unittest.TestCase):
    """Verify grep_search and glob_search are properly registered in harness_core."""

    def test_harness_exposes_grep_search_tool_and_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            tool_names = {tool["name"] for tool in module.TOOLS}
            handler_names = set(module.TOOL_HANDLERS)

            # grep_search must be in both TOOLS and TOOL_HANDLERS
            self.assertIn("grep_search", tool_names)
            self.assertIn("grep_search", handler_names)

    def test_harness_exposes_glob_search_tool_and_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            tool_names = {tool["name"] for tool in module.TOOLS}
            handler_names = set(module.TOOL_HANDLERS)

            # glob_search must be in both TOOLS and TOOL_HANDLERS
            self.assertIn("glob_search", tool_names)
            self.assertIn("glob_search", handler_names)

    def test_grep_search_handler_calls_with_workdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            # Create a test file in the temp workspace
            test_file = Path(tmp) / "test.py"
            test_file.write_text("hello world\n", encoding="utf-8")

            # Call the handler — it should search within WORKDIR (which is tmp).
            # Stage 2C-B3A: handler returns a GrepSearchResult; str() reproduces
            # the legacy text format the model sees.
            result = module.TOOL_HANDLERS["grep_search"](pattern="hello")
            text = str(result)
            self.assertIn("hello world", text)
            self.assertIn("test.py", text)

    def test_glob_search_handler_calls_with_workdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            # Create a test file
            test_file = Path(tmp) / "example.py"
            test_file.write_text("# example\n", encoding="utf-8")

            result = module.TOOL_HANDLERS["glob_search"](pattern="*.py")
            self.assertIn("example.py", result)

    def test_grep_search_tool_schema_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            grep_tool = next(t for t in module.TOOLS if t["name"] == "grep_search")
            schema = grep_tool["input_schema"]
            self.assertIn("pattern", schema["properties"])
            self.assertEqual(schema["required"], ["pattern"])

    def test_glob_search_tool_schema_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            glob_tool = next(t for t in module.TOOLS if t["name"] == "glob_search")
            schema = glob_tool["input_schema"]
            self.assertIn("pattern", schema["properties"])
            self.assertEqual(schema["required"], ["pattern"])


if __name__ == "__main__":
    unittest.main()
