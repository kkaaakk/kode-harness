"""test_phase3a0_golden_baseline.py - Phase 3A-0 golden behavior snapshots.

Purpose
-------
Before the 3A-1 AnthropicAdapter lands, lock the OBSERVABLE behavior of
the Agent Loop's model-call path with scripted fake responses. After 3A-1
replaces the direct ``client.messages.create`` call inside agent_loop with
``provider.complete()``, these same scripts MUST produce identical
results - the tests are written against public behavior (events, message
history, tool execution, request shapes), not implementation internals.

Golden scenarios locked here (per docs/phase3a0-provider-contract-audit.md
section 5):

  1. plain text answer (end_turn, single text block)
  2. single tool call round (tool_use -> tool_result -> end_turn)
  3. multiple tool calls in one response
  4. text + tool_call mixed in one response
  5. token usage event emission (input/output)
  6. request shape sent to the model (model/system/messages/tools/
     max_tokens) - captured verbatim per call
  7. exception propagation from the model call (no swallowing)
  8. empty content / end_turn without text

The fake Anthropic response objects mirror the attribute surface the
loop actually touches: ``content`` (iterable of blocks with .type/.name/
.id/.input/.text), ``stop_reason`` (str), ``usage`` (input_tokens/
output_tokens attributes).

These tests double as the equivalence oracle for 3A-1: the adapter path
must reproduce every assertion in this file without modification.
"""

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


def load_harness_module(temp_cwd: Path):
    """Load harness_core under a fake anthropic SDK (no network)."""
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
        "harness_core_golden_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)

    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv
    # Force a deterministic MODEL_ID (other suites may leave one behind),
    # restoring whatever was there afterwards.
    previous_model_id = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "test-model"
    # Purge any cached real agents.config: harness_core does
    # ``from agents.config import *`` and a previously-imported REAL config
    # (built from the developer's .env, e.g. MODEL_ID=deepseek-...) would
    # otherwise be reused, ignoring our fakes above.
    previous_config = sys.modules.get("agents.config")
    sys.modules.pop("agents.config", None)
    try:
        os.chdir(temp_cwd)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_model_id is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model_id
        # Drop the test-built config; restore the real one if there was one.
        sys.modules.pop("agents.config", None)
        if previous_config is not None:
            sys.modules["agents.config"] = previous_config
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


class FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeToolBlock:
    def __init__(self, name, tool_id, tool_input):
        self.type = "tool_use"
        self.name = name
        self.id = tool_id
        self.input = tool_input


class FakeResponse:
    def __init__(self, content, stop_reason="end_turn", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class RecordingClient:
    """Replaces module.client.messages.create; records kwargs per call and
    pops scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        # Snapshot mutable args (messages is passed by reference and the
        # loop appends to it AFTER the call) so the recorded shape reflects
        # what the model was actually sent at call time.
        kwargs = dict(kwargs)
        if "messages" in kwargs:
            kwargs["messages"] = list(kwargs["messages"])
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No scripted response left")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class GoldenBaselineTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))

    def _install(self, responses):
        recorder = RecordingClient(responses)
        self.module.client.messages.create = recorder
        return recorder

    def _events(self, callback_calls, etype):
        return [e for e in callback_calls if e["type"] == etype]

    # ------------------------------------------------------------------
    # 1. Plain text answer
    # ------------------------------------------------------------------

    def test_golden_plain_text_answer(self):
        events = []
        self._install([
            FakeResponse([FakeTextBlock("hello world")], "end_turn",
                         FakeUsage(10, 7)),
        ])
        messages = [{"role": "user", "content": "hi"}]
        self.module.agent_loop(messages, event_callback=events.append)

        text_events = self._events(events, "text")
        self.assertEqual(
            [e["text"] for e in text_events], ["hello world"]
        )
        # Final assistant message was appended verbatim to history.
        assistant = messages[-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0].text, "hello world")

    def test_golden_plain_text_token_event(self):
        events = []
        self._install([
            FakeResponse([FakeTextBlock("x")], "end_turn",
                         FakeUsage(input_tokens=11, output_tokens=3)),
        ])
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], event_callback=events.append
        )
        token_events = self._events(events, "tokens")
        self.assertEqual(len(token_events), 1)
        self.assertEqual(token_events[0]["input"], 11)
        self.assertEqual(token_events[0]["output"], 3)
        self.assertEqual(token_events[0]["tokens"], 14)

    # ------------------------------------------------------------------
    # 2. Single tool call round
    # ------------------------------------------------------------------

    def test_golden_single_tool_call_round(self):
        events = []
        tool_block = FakeToolBlock(
            "bash", "toolu_1", {"command": "echo golden"}
        )
        self._install([
            FakeResponse([tool_block], "tool_use", FakeUsage(5, 5)),
            FakeResponse([FakeTextBlock("after tool")], "end_turn",
                         FakeUsage(9, 2)),
        ])
        messages = [{"role": "user", "content": "run it"}]
        self.module.agent_loop(messages, event_callback=events.append)

        call_events = self._events(events, "tool_call")
        self.assertEqual(len(call_events), 1)
        self.assertEqual(call_events[0]["name"], "bash")
        self.assertEqual(call_events[0]["id"], "toolu_1")
        self.assertEqual(call_events[0]["input"], {"command": "echo golden"})

        result_events = self._events(events, "tool_result")
        self.assertEqual(len(result_events), 1)
        self.assertEqual(result_events[0]["name"], "bash")
        self.assertEqual(result_events[0]["id"], "toolu_1")
        self.assertIn("golden", result_events[0]["output"])

        # History shape: user, assistant(tool), user(tool_result),
        # assistant(text).
        self.assertEqual(
            [m["role"] for m in messages],
            ["user", "assistant", "user", "assistant"],
        )
        tool_result_msg = messages[2]
        first_result = tool_result_msg["content"][0]
        self.assertEqual(first_result["type"], "tool_result")
        self.assertEqual(first_result["tool_use_id"], "toolu_1")
        self.assertIn("golden", first_result["content"])

    # ------------------------------------------------------------------
    # 3. Multiple tool calls in one response
    # ------------------------------------------------------------------

    def test_golden_multiple_tool_calls(self):
        events = []
        blocks = [
            FakeToolBlock("read_file", "toolu_a", {"path": "x.py"}),
            FakeToolBlock("glob_search", "toolu_b", {"pattern": "*.py"}),
        ]
        self._install([
            FakeResponse(blocks, "tool_use", FakeUsage(5, 5)),
            FakeResponse([FakeTextBlock("both done")], "end_turn",
                         FakeUsage(9, 2)),
        ])
        messages = [{"role": "user", "content": "two things"}]
        self.module.agent_loop(messages, event_callback=events.append)

        call_events = self._events(events, "tool_call")
        self.assertEqual(
            [(e["name"], e["id"]) for e in call_events],
            [("read_file", "toolu_a"), ("glob_search", "toolu_b")],
        )
        result_msg = messages[2]
        result_ids = [b["tool_use_id"] for b in result_msg["content"]]
        self.assertEqual(result_ids, ["toolu_a", "toolu_b"])

    # ------------------------------------------------------------------
    # 4. text + tool_call mixed
    # ------------------------------------------------------------------

    def test_golden_text_plus_tool_call_mixed(self):
        events = []
        blocks = [
            FakeTextBlock("I will check the file."),
            FakeToolBlock("read_file", "toolu_m", {"path": "m.py"}),
        ]
        self._install([
            FakeResponse(blocks, "tool_use", FakeUsage(5, 5)),
            FakeResponse([FakeTextBlock("done")], "end_turn", FakeUsage(9, 2)),
        ])
        messages = [{"role": "user", "content": "mixed"}]
        self.module.agent_loop(messages, event_callback=events.append)

        # Text BEFORE the tool call was streamed, then tool call.
        text_events = self._events(events, "text")
        call_events = self._events(events, "tool_call")
        # Text events stream from EVERY round: the pre-tool text here, and
        # the final round's text ("done") in the second response.
        self.assertEqual(
            [e["text"] for e in text_events],
            ["I will check the file.", "done"],
        )
        self.assertEqual(len(call_events), 1)
        # Mixed assistant content preserved verbatim in history.
        assistant = messages[1]["content"]
        self.assertEqual(assistant[0].type, "text")
        self.assertEqual(assistant[1].type, "tool_use")

    # ------------------------------------------------------------------
    # 5. Request shape (captured verbatim)
    # ------------------------------------------------------------------

    def test_golden_request_shape(self):
        recorder = self._install([
            FakeResponse([FakeTextBlock("ok")], "end_turn", FakeUsage(1, 1)),
        ])
        request_messages = [{"role": "user", "content": "shape"}]
        # agent_loop mutates the passed list in place; snapshot what the
        # FIRST model call received (captured before any mutation matters).
        self.module.agent_loop(request_messages)
        # The history arg was forwarded by reference, so after the loop it
        # contains the assistant reply too; the first REQUEST's messages
        # kwargs must be exactly the original single user message.
        self.assertEqual(len(recorder.calls), 1)
        kwargs = recorder.calls[0]
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["max_tokens"], 8000)
        self.assertIn("system", kwargs)
        self.assertIn("tools", kwargs)
        self.assertEqual(
            kwargs["messages"], [{"role": "user", "content": "shape"}]
        )
        tool_names = [t["name"] for t in kwargs["tools"]]
        self.assertIn("bash", tool_names)
        self.assertIn("TodoWrite", tool_names)
        self.assertIn("spawn_teammate", tool_names)

    # ------------------------------------------------------------------
    # 6. Exception propagation (no swallowing)
    # ------------------------------------------------------------------

    def test_golden_exception_propagates(self):
        class BoomError(Exception):
            pass

        self._install([BoomError("api down")])
        with self.assertRaises(BoomError):
            self.module.agent_loop([{"role": "user", "content": "boom"}])

    # ------------------------------------------------------------------
    # 7. Empty content end_turn
    # ------------------------------------------------------------------

    def test_golden_empty_content_end_turn(self):
        events = []
        self._install([
            FakeResponse([], "end_turn", FakeUsage(1, 0)),
        ])
        messages = [{"role": "user", "content": "quiet"}]
        # Must not raise despite zero content blocks.
        self.module.agent_loop(messages, event_callback=events.append)
        self.assertEqual(self._events(events, "text"), [])
        self.assertEqual(messages[-1]["role"], "assistant")

    # ------------------------------------------------------------------
    # 8. stop_reason=max_tokens ends loop without tool execution
    # ------------------------------------------------------------------

    def test_golden_max_tokens_stops_without_tool_round(self):
        events = []
        self._install([
            FakeResponse([FakeTextBlock("partial")], "max_tokens",
                         FakeUsage(3, 8)),
        ])
        messages = [{"role": "user", "content": "long"}]
        self.module.agent_loop(messages, event_callback=events.append)
        self.assertEqual(self._events(events, "tool_result"), [])
        self.assertEqual(len(messages), 2)

    # ------------------------------------------------------------------
    # 9. usage=None is tolerated (no token event)
    # ------------------------------------------------------------------

    def test_golden_usage_none_tolerated(self):
        events = []
        self._install([
            FakeResponse([FakeTextBlock("no usage")], "end_turn", usage=None),
        ])
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}], event_callback=events.append
        )
        self.assertEqual(self._events(events, "tokens"), [])

    # ------------------------------------------------------------------
    # 10. Golden equivalence oracle for 3A-1
    # ------------------------------------------------------------------

    def test_golden_event_sequence_snapshot(self):
        """Full ordered event sequence for the canonical tool round - the
        single most important snapshot for adapter equivalence."""
        events = []
        tool_block = FakeToolBlock(
            "bash", "toolu_seq", {"command": "echo seq"}
        )
        self._install([
            FakeResponse([tool_block], "tool_use", FakeUsage(2, 2)),
            FakeResponse([FakeTextBlock("final")], "end_turn", FakeUsage(4, 1)),
        ])
        self.module.agent_loop(
            [{"role": "user", "content": "seq"}], event_callback=events.append
        )
        # Exact ordered event-type sequence observed today.
        self.assertEqual(
            [e["type"] for e in events],
            ["tokens", "tool_call", "tool_result", "tokens", "text"],
        )
        # Event payloads (stable fields only) for byte-level comparison.
        self.assertEqual(events[1]["name"], "bash")
        self.assertEqual(events[1]["id"], "toolu_seq")
        self.assertEqual(events[1]["input"], {"command": "echo seq"})
        self.assertIn("seq", events[2]["output"])
        self.assertEqual(events[4]["text"], "final")


if __name__ == "__main__":
    unittest.main()
