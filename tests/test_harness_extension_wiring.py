"""test_harness_extension_wiring.py - Stage 1 verification that harness_core's
agent_loop correctly invokes all 8 extension hook points, and that an empty
registry preserves pre-stage-1 behavior.

Uses the same isolated-module loading pattern as test_harness_background.py.
"""

from __future__ import annotations

import asyncio
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
    added_paths = []
    for path in (REPO_ROOT, REPO_ROOT / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_paths.append(text)
    spec = importlib.util.spec_from_file_location(
        "harness_core_extension_wiring_test", MODULE_PATH
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
        for path in added_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


class TextBlock:
    type = "text"
    def __init__(self, text):
        self.text = text


class ToolBlock:
    type = "tool_use"
    def __init__(self, name="bash", input_=None, id_="t1"):
        self.name = name
        self.input = input_ or {"command": "echo hi"}
        self.id = id_


class EmptyRegistryPreservesBehaviorTests(unittest.TestCase):
    """With no extensions registered, agent_loop behaves exactly as before."""

    def test_empty_registry_runs_full_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

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
            module.client.messages.create = lambda **_: responses.pop(0)

            messages = [{"role": "user", "content": "run echo"}]
            # Should complete without raising, exactly like pre-stage-1.
            module.agent_loop(messages)
            # Final assistant message present
            self.assertEqual(messages[-1]["role"], "assistant")


class AllEightHooksInvokedTests(unittest.TestCase):
    """Verify all 8 hook points fire during a typical agent_loop run."""

    def test_all_eight_hooks_fire(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            # Record every hook invocation
            fired = []

            def make_handler(hook_name):
                def h(ctx):
                    fired.append(hook_name)
                    return None
                return h

            from agents.types.events import Event
            for hook in [
                Event.BEFORE_AGENT_START,
                Event.BEFORE_MODEL_REQUEST,
                Event.AFTER_MODEL_RESPONSE,
                Event.BEFORE_TOOL_CALL,
                Event.AFTER_TOOL_RESULT,
                Event.TURN_END,
                Event.AGENT_END,
            ]:
                module.EXTENSIONS.on(hook, make_handler(hook),
                                     extension_id=f"rec_{hook}")

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
            module.client.messages.create = lambda **_: responses.pop(0)

            module.agent_loop([{"role": "user", "content": "run echo"}])

            # All 7 hooks should have fired (BEFORE_COMPACTION only fires on
            # actual compaction, which doesn't happen in this short test).
            for hook in [
                Event.BEFORE_AGENT_START,
                Event.BEFORE_MODEL_REQUEST,
                Event.AFTER_MODEL_RESPONSE,
                Event.BEFORE_TOOL_CALL,
                Event.AFTER_TOOL_RESULT,
                Event.TURN_END,
                Event.AGENT_END,
            ]:
                self.assertIn(hook, fired,
                              f"Hook {hook} did not fire. Fired: {fired}")

    def test_before_compaction_fires_on_manual_compress(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            fired = []

            from agents.types.events import Event
            module.EXTENSIONS.on(
                Event.BEFORE_COMPACTION,
                lambda ctx: fired.append(Event.BEFORE_COMPACTION),
                extension_id="comp_rec",
            )

            # First response: model requests compress tool -> manual compact path
            responses = [
                types.SimpleNamespace(
                    stop_reason="tool_use",
                    content=[ToolBlock(name="compress", input_={}, id_="c1")],
                    usage=None,
                ),
            ]
            module.client.messages.create = lambda **_: responses.pop(0)
            # Stub auto_compact so it doesn't try to call the LLM
            module.auto_compact = lambda msgs: []

            module.agent_loop([{"role": "user", "content": "compact please"}])

            self.assertIn(Event.BEFORE_COMPACTION, fired)


class ExtensionCanBlockToolCallTests(unittest.TestCase):
    """An extension can block a tool call (additional restriction on top of
    kernel safety). The blocked call is recorded as a denied result."""

    def test_block_tool_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            from agents.types.events import Event, HookResult, Priority

            def block_bash(ctx):
                if ctx.get("tool_name") == "bash":
                    return HookResult(block=True, reason="bash disabled in test")
                return None

            module.EXTENSIONS.on(Event.BEFORE_TOOL_CALL, block_bash,
                                 extension_id="block_bash")

            responses = [
                types.SimpleNamespace(
                    stop_reason="tool_use",
                    content=[ToolBlock()],
                    usage=None,
                ),
                types.SimpleNamespace(
                    stop_reason="end_turn",
                    content=[TextBlock("ok")],
                    usage=None,
                ),
            ]
            module.client.messages.create = lambda **_: responses.pop(0)

            messages = [{"role": "user", "content": "run bash"}]
            module.agent_loop(messages)

            # The tool result for the blocked bash call should mention "Blocked"
            user_msgs_with_results = [
                m for m in messages
                if m.get("role") == "user" and isinstance(m.get("content"), list)
            ]
            self.assertTrue(len(user_msgs_with_results) > 0)
            results = user_msgs_with_results[0]["content"]
            blocked_result = next(
                r for r in results
                if isinstance(r, dict) and r.get("tool_use_id") == "t1"
            )
            self.assertIn("Blocked by extension", blocked_result["content"])


class ExtensionCanPatchModelRequestTests(unittest.TestCase):
    """An extension can patch model request kwargs (e.g. swap tools list)."""

    def test_patch_model_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            from agents.types.events import Event, HookResult

            captured = {}

            def patcher(ctx):
                # Reduce max_tokens via patch
                return HookResult(model_request_patch={"max_tokens": 1234})

            module.EXTENSIONS.on(Event.BEFORE_MODEL_REQUEST, patcher,
                                 extension_id="patcher")

            def fake_create(**kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    stop_reason="end_turn",
                    content=[TextBlock("done")],
                    usage=None,
                )

            module.client.messages.create = fake_create
            module.agent_loop([{"role": "user", "content": "hi"}])
            self.assertEqual(captured["max_tokens"], 1234)


# ---------------------------------------------------------------------------
# Stage 1.1: AGENT_END contract — fires exactly once on every exit path
# ---------------------------------------------------------------------------

class AgentEndContractTests(unittest.TestCase):
    """AGENT_END must fire EXACTLY ONCE per agent_loop() call, on every exit
    path: normal completion, exception, cancellation, and extension block.

    The context includes ``status`` ("completed" | "failed" | "blocked" |
    "cancelled") and ``error`` (the exception, if any).
    """

    def _record_agent_end(self, module):
        records = []
        from agents.types.events import Event
        module.EXTENSIONS.on(
            Event.AGENT_END,
            lambda ctx: records.append(ctx),
            extension_id="end_recorder",
        )
        return records

    def test_agent_end_fires_once_on_normal_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            module.client.messages.create = lambda **_: types.SimpleNamespace(
                stop_reason="end_turn",
                content=[TextBlock("done")],
                usage=None,
            )
            module.agent_loop([{"role": "user", "content": "hi"}])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "completed")
            self.assertIsNone(records[0]["error"])

    def test_agent_end_fires_once_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            def boom(**kwargs):
                raise RuntimeError("provider exploded")
            module.client.messages.create = boom

            with self.assertRaises(RuntimeError):
                module.agent_loop([{"role": "user", "content": "hi"}])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "failed")
            self.assertIn("provider exploded", records[0]["error"])

    def test_agent_end_fires_once_on_cancellation(self):
        """Cancellation via asyncio.CancelledError (not just KeyboardInterrupt).

        CancelledError inherits from BaseException in Python 3.8+, so a bare
        `except Exception` would NOT catch it. The harness must handle it
        explicitly. This test raises a REAL CancelledError, not a stand-in.
        """
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            def boom(**kwargs):
                raise asyncio.CancelledError()
            module.client.messages.create = boom

            with self.assertRaises(asyncio.CancelledError):
                module.agent_loop([{"role": "user", "content": "hi"}])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "cancelled")
            self.assertEqual(records[0]["error"], "asyncio.CancelledError")

    def test_agent_end_fires_once_on_keyboard_interrupt(self):
        """KeyboardInterrupt (sync cancellation) also maps to cancelled."""
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            def boom(**kwargs):
                raise KeyboardInterrupt()
            module.client.messages.create = boom

            with self.assertRaises(KeyboardInterrupt):
                module.agent_loop([{"role": "user", "content": "hi"}])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "cancelled")
            self.assertEqual(records[0]["error"], "KeyboardInterrupt")

    def test_agent_end_fires_once_on_agent_level_block(self):
        """Agent-level block: BEFORE_MODEL_REQUEST returns block=True.
        This stops the whole loop -> status="blocked"."""
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            from agents.types.events import Event, HookResult, Priority

            def block_model(ctx):
                return HookResult(block=True, reason="rate limited")
            module.EXTENSIONS.on(Event.BEFORE_MODEL_REQUEST, block_model,
                                 priority=Priority.NORMAL, extension_id="blocker")

            module.agent_loop([{"role": "user", "content": "hi"}])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "blocked")
            self.assertEqual(records[0]["error"], "rate limited")

    def test_tool_level_block_does_not_set_agent_blocked(self):
        """Tool-level block: BEFORE_TOOL_CALL returns block=True. The single
        tool call is denied, but agent_status stays "completed" because the
        model can still finish with a text answer."""
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            from agents.types.events import Event, HookResult, Priority

            def block_bash(ctx):
                if ctx.get("tool_name") == "bash":
                    return HookResult(block=True, reason="readonly mode")
                return None
            module.EXTENSIONS.on(Event.BEFORE_TOOL_CALL, block_bash,
                                 priority=Priority.NORMAL, extension_id="blocker")

            # Turn 1: bash blocked. Turn 2: model gives final text answer.
            responses = [
                types.SimpleNamespace(
                    stop_reason="tool_use",
                    content=[ToolBlock()],
                    usage=None,
                ),
                types.SimpleNamespace(
                    stop_reason="end_turn",
                    content=[TextBlock("I couldn't run bash, but here's my answer")],
                    usage=None,
                ),
            ]
            module.client.messages.create = lambda **_: responses.pop(0)

            module.agent_loop([{"role": "user", "content": "run bash"}])

            self.assertEqual(len(records), 1)
            # Tool block did NOT make agent_status="blocked"; it completed.
            self.assertEqual(records[0]["status"], "completed")
            self.assertIsNone(records[0]["error"])

    def test_agent_end_handler_failure_does_not_mask_original_exception(self):
        """If the main loop raises AND an AGENT_END handler also raises,
        the original exception must be the one propagated, not the handler's."""
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            from agents.types.events import Event, Priority

            def boom_agent_end(ctx):
                raise RuntimeError("AGENT_END cleanup failed")
            module.EXTENSIONS.on(Event.AGENT_END, boom_agent_end,
                                 priority=Priority.NORMAL,
                                 fail_policy="fail_closed",
                                 extension_id="bad_cleanup")

            def boom_provider(**kwargs):
                raise ConnectionError("provider down")
            module.client.messages.create = boom_provider

            # The ORIGINAL exception (ConnectionError) must propagate, not
            # the AGENT_END handler's RuntimeError.
            with self.assertRaises(ConnectionError) as cm:
                module.agent_loop([{"role": "user", "content": "hi"}])
            self.assertIn("provider down", str(cm.exception))

    def test_agent_end_does_not_fire_twice_on_manual_compress(self):
        """Manual compress path returns early; AGENT_END must still fire
        exactly once (from the finally block), not twice."""
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            records = self._record_agent_end(module)

            responses = [
                types.SimpleNamespace(
                    stop_reason="tool_use",
                    content=[ToolBlock(name="compress", input_={}, id_="c1")],
                    usage=None,
                ),
            ]
            module.client.messages.create = lambda **_: responses.pop(0)
            module.auto_compact = lambda msgs: []

            module.agent_loop([{"role": "user", "content": "compact"}])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
