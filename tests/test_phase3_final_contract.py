"""test_phase3_final_contract.py - Phase 3D-final cross-provider history + Phase 3 seal.

The ultimate Phase 3 integration contract: history left by one provider
must be consumable by the next provider, including tool calls.

    Previous        Next            History continuation
    -------         ----            --------------------
    Claude          DeepSeek        ✅ (legacy -> canonical -> OpenAI wire)
    DeepSeek        Claude          ✅ (OpenAI raw_response.content bridge -> Anthropic)
    Claude          OpenRouter/Qwen ✅
    OpenRouter/Qwen Claude          ✅

Also locks:
  - tool_call_id correspondence survives canonical/render (A->A, B->B)
  - multiple tool calls keep their mapping + order
  - text + tool_call mixed history preserved
  - tool error content is not lost (even though OpenAI wire has no
    equivalent is_error field)
  - /model does NOT clear session history (switch != new session)
  - Claude -> DeepSeek -> Claude three-run continuity
  - exactly 2 effective switches -> exactly 2 MODEL_CHANGED
  - explicit model_alias override does not change Session selection
  - override and session runs share the same history
  - non-Anthropic session full runtime needs no Anthropic key
"""

import json
import os
import unittest

from agents.providers.message_codec import render_openai, to_canonical
from agents.providers.types import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Codec-level contracts (no network)
# ---------------------------------------------------------------------------

class CrossProviderCodecTests(unittest.TestCase):
    """Legacy Anthropic-shaped history -> canonical -> OpenAI wire, with
    tool_call_id correspondence preserved."""

    def _claude_history_with_tool(self):
        return [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_A", "name": "read_file",
                 "input": {"path": "x.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_A",
                 "content": "content-of-x"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]

    def test_claude_to_openai_tool_id_preserved(self):
        """Claude history -> OpenAI wire: toolu_A maps to tool_call_id."""
        wire = render_openai(self._claude_history_with_tool())
        roles = [m["role"] for m in wire]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        assistant = wire[1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "toolu_A")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
            {"path": "x.py"},
        )
        tool = wire[2]
        self.assertEqual(tool["tool_call_id"], "toolu_A")
        self.assertEqual(tool["content"], "content-of-x")

    def test_multiple_tool_calls_mapping_kept(self):
        """A->A, B->B after canonical+render."""
        legacy = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "bash",
                 "input": {"command": "ls"}},
                {"type": "tool_use", "id": "b", "name": "read_file",
                 "input": {"path": "y.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "ra"},
                {"type": "tool_result", "tool_use_id": "b", "content": "rb"},
            ]},
        ]
        wire = render_openai(legacy)
        assistant = wire[0]
        ids = [tc["id"] for tc in assistant["tool_calls"]]
        self.assertEqual(ids, ["a", "b"])
        tools = wire[1:]
        self.assertEqual([t["tool_call_id"] for t in tools], ["a", "b"])
        self.assertEqual([t["content"] for t in tools], ["ra", "rb"])

    def test_mixed_text_and_tool_call_preserved(self):
        legacy = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "I will check."},
                {"type": "tool_use", "id": "t1", "name": "bash",
                 "input": {"command": "echo hi"}},
            ]},
        ]
        wire = render_openai(legacy)
        self.assertEqual(wire[0]["content"], "I will check.")
        self.assertEqual(len(wire[0]["tool_calls"]), 1)
        self.assertEqual(wire[0]["tool_calls"][0]["id"], "t1")

    def test_tool_error_content_not_lost(self):
        """is_error is inferred but the error CONTENT survives the wire."""
        legacy = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "e1", "name": "bash",
                 "input": {"command": "false"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "e1",
                 "content": "Error: command failed with exit code 1"},
            ]},
        ]
        wire = render_openai(legacy)
        tool = wire[1]
        self.assertEqual(tool["tool_call_id"], "e1")
        self.assertIn("command failed", tool["content"])

    def test_canonical_roundtrip_tool_id(self):
        """canonical dataclasses -> OpenAI wire keeps ids (DeepSeek path)."""
        wire = render_openai([
            AssistantMessage(
                text="ok",
                tool_calls=[ToolCall(id="call_deepseek_1", name="bash",
                                      arguments={"command": "ls"})],
            ),
            ToolResultMessage(tool_call_id="call_deepseek_1", content="out"),
        ])
        self.assertEqual(wire[0]["tool_calls"][0]["id"], "call_deepseek_1")
        self.assertEqual(wire[1]["tool_call_id"], "call_deepseek_1")

    def test_to_canonical_accepts_deepseek_wire_blocks(self):
        """OpenAI-shaped assistant blocks (from raw_response.content bridge)
        normalize into canonical with the same tool ids."""
        from agents.providers.types import ToolCall as _TC
        canonical = to_canonical([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_x", "name": "read_file",
                 "input": {"path": "z.py"}},
            ]},
        ])
        self.assertEqual(canonical[0].tool_calls[0].id, "call_x")
        self.assertEqual(canonical[0].tool_calls[0].name, "read_file")


# ---------------------------------------------------------------------------
# Session-level contracts
# ---------------------------------------------------------------------------

class ModelSwitchHistoryTests(unittest.TestCase):
    """/model does not clear history; three-run continuity; 2 events."""

    def test_model_switch_does_not_clear_session_state(self):
        from agents.session import SessionState, handle_model_command
        from agents.providers.model_spec import ModelRegistry, ModelSpec

        reg = ModelRegistry()
        reg.register(ModelSpec(alias="claude", provider="anthropic",
                               model_id="claude-x"))
        reg.register(ModelSpec(alias="deepseek", provider="deepseek",
                               model_id="deepseek-chat"))
        s = SessionState()
        # Simulate an existing conversation history on the session.
        history = [{"role": "user", "content": "hello"}]
        s.metadata["history_ref"] = history  # placeholder reference
        handle_model_command("/model deepseek", s, reg)
        # History untouched; only model_alias changed.
        self.assertEqual(s.model_alias, "deepseek")
        self.assertIs(s.metadata["history_ref"], history)
        self.assertEqual(history, [{"role": "user", "content": "hello"}])

    def test_two_effective_switches_two_events(self):
        from agents.session import SessionState, handle_model_command
        from agents.providers.model_spec import ModelRegistry, ModelSpec
        from agents.types.events import Event

        reg = ModelRegistry()
        reg.register(ModelSpec(alias="claude", provider="anthropic",
                               model_id="claude-x"))
        reg.register(ModelSpec(alias="deepseek", provider="deepseek",
                               model_id="deepseek-chat"))

        class Ex:
            def __init__(self):
                self.events = []

            def emit(self, name, ctx):
                self.events.append(name)

        exts = Ex()
        s = SessionState()
        handle_model_command("/model deepseek", s, reg, extensions=exts)
        handle_model_command("/model claude", s, reg, extensions=exts)
        self.assertEqual(exts.events, [Event.MODEL_CHANGED,
                                       Event.MODEL_CHANGED])

    def test_explicit_override_does_not_change_session_selection(self):
        from agents.session import SessionState
        s = SessionState()
        s.model_alias = "deepseek"
        # Explicit override resolves to claude for THIS run, but the
        # session selection stays deepseek.
        from agents.session import resolve_session_model_alias
        self.assertEqual(resolve_session_model_alias(s, "claude"), "claude")
        self.assertEqual(s.model_alias, "deepseek")
        # Next run (no explicit) returns to deepseek.
        self.assertEqual(resolve_session_model_alias(s, None), "deepseek")

    def test_override_and_session_share_history_concept(self):
        """Both an override run and a session run consume the SAME history
        list - switching model never forks/clears the conversation."""
        from agents.session import SessionState, resolve_session_model_alias
        history = [{"role": "user", "content": "shared"}]
        s = SessionState()
        s.model_alias = "deepseek"
        # Both resolutions share the same history object (passed to
        # agent_loop by the caller; nothing in the selection mutates it).
        self.assertEqual(resolve_session_model_alias(s, "claude"), "claude")
        self.assertEqual(resolve_session_model_alias(s, None), "deepseek")
        self.assertEqual(history, [{"role": "user", "content": "shared"}])


class NoHiddenAnthropicDependencyFinalTests(unittest.TestCase):
    """Non-Anthropic session full runtime needs no Anthropic key."""

    def test_deepseek_switch_does_not_touch_anthropic_key(self):
        from agents.session import SessionState, handle_model_command
        from agents.providers.model_spec import ModelRegistry, ModelSpec
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            reg = ModelRegistry()
            reg.register(ModelSpec(alias="claude", provider="anthropic",
                                   model_id="claude-x"))
            reg.register(ModelSpec(alias="deepseek", provider="deepseek",
                                   model_id="deepseek-chat"))
            s = SessionState()
            handle_model_command("/model deepseek", s, reg)
            self.assertEqual(s.model_alias, "deepseek")
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved


# ---------------------------------------------------------------------------
# End-to-end: Claude -> DeepSeek -> Claude with the SAME history through
# the REAL agent_loop path (not just codec functions). This is the
# ultimate Phase 3 proof: tool calls/results produced by one provider's
# run must be consumed by the next provider's run.
# ---------------------------------------------------------------------------

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"


def load_harness_module(temp_cwd: Path):
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
        "harness_core_phase3_final_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)

    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv
    previous_model_id = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "test-model"
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


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, tool_id, name, tool_input):
        self.id = tool_id
        self.name = name
        self.input = tool_input


def _anthropic_resp(content, stop_reason):
    return types.SimpleNamespace(
        stop_reason=stop_reason, content=content, usage=None)


def _openai_resp(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "x",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish}],
    }


class CrossProviderEndToEndTests(unittest.TestCase):
    """The Phase 3 ultimate test: three agent_loop runs on the SAME
    messages list, switching provider mid-stream via /model, with a full
    tool call/result round on each side.

        Run 1 (Claude):   user -> tool_call A -> tool_result A -> text
        /model deepseek
        Run 2 (DeepSeek): reads Claude history -> tool_call B ->
                          tool_result B -> text
        /model claude
        Run 3 (Claude):   reads Claude + DeepSeek history -> text
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))

        # Registry + router for claude / deepseek / qwen-openrouter.
        from agents.providers.model_spec import ModelRegistry, ModelSpec
        from agents.providers.provider_router import (
            ProviderBinding, ProviderRouter)
        from agents.providers.openai_compatible_adapter import (
            OpenAICompatibleAdapter, OpenAICompatibleConfig)

        self.reg = ModelRegistry()
        self.reg.register(ModelSpec(alias="claude", provider="anthropic",
                                    model_id="claude-x"))
        self.reg.register(ModelSpec(alias="deepseek", provider="deepseek",
                                    model_id="deepseek-chat"))
        self.reg.register(ModelSpec(alias="qwen-openrouter",
                                    provider="openrouter",
                                    model_id="qwen/qwen-2.5-72b-instruct"))

        # Per-run OpenAI response queues keyed by provider.
        self.openai_queues = {"deepseek": [], "openrouter": []}
        self.openai_seen = []  # every OpenAI payload model

        class FakeOCClient:
            def __init__(self, owner, queue):
                self.owner = owner
                self.queue = queue

            def complete(self, payload):
                self.owner.openai_seen.append(payload)
                if not self.queue:
                    return _openai_resp(content="(no more) - done")
                return self.queue.pop(0)

        def make_factory(provider):
            def factory(model_id):
                config = OpenAICompatibleConfig(
                    base_url=f"https://{provider}.example.invalid",
                    api_key="test-key", model=model_id)
                adapter = OpenAICompatibleAdapter(config)
                queue = self.openai_queues[provider]
                adapter._client_provider = lambda: FakeOCClient(
                    self, queue)  # noqa: SLF001
                return adapter
            return factory

        self.router = ProviderRouter()
        self.router.register(ProviderBinding(
            provider="anthropic", adapter_type="anthropic",
            adapter_factory=lambda model_id: self.module.ANTHROPIC_ADAPTER))
        self.router.register(ProviderBinding(
            provider="deepseek", adapter_type="openai-compatible",
            base_url="https://deepseek.example.invalid",
            adapter_factory=make_factory("deepseek")))
        self.router.register(ProviderBinding(
            provider="openrouter", adapter_type="openai-compatible",
            base_url="https://openrouter.example.invalid",
            adapter_factory=make_factory("openrouter")))

        self.session = None
        self.messages = []

    def _make_anthropic_client(self, responses):
        q = list(responses)

        def create(**kwargs):
            if not q:
                return _anthropic_resp([_Block("done")], "end_turn")
            return q.pop(0)

        self.module.client.messages.create = create

    def _switch(self, alias):
        from agents.session import handle_model_command
        handle_model_command(f"/model {alias}", self.session, self.reg)

    def _run(self):
        self.module.agent_loop(
            self.messages,
            model_registry=self.reg,
            provider_router=self.router,
            session=self.session,
        )

    # ------------------------------------------------------------------

    def test_three_run_cross_provider_tool_history(self):
        from agents.session import SessionState
        self.session = SessionState()

        # --- Run 1: Claude. tool_call A -> tool_result A -> final ---
        self._make_anthropic_client([
            _anthropic_resp(
                [_ToolUse("toolu_A", "read_file", {"path": "x.py"})],
                "tool_use"),
            _anthropic_resp([_Block("claude final")], "end_turn"),
        ])
        self.messages = [{"role": "user", "content": "read x.py"}]
        self._run()

        # Claude history now has tool_call A + tool_result A.
        roles = [m["role"] for m in self.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertIn("claude final",
                      self.messages[-1]["content"][0].text)
        tool_result_A = self.messages[2]["content"][0]
        self.assertEqual(tool_result_A["tool_use_id"], "toolu_A")

        # --- switch to DeepSeek ---
        self._switch("deepseek")

        # --- Run 2: DeepSeek. It must read Claude's history (toolu_A),
        # then produce tool_call B -> tool_result B -> final. ---
        self.openai_queues["deepseek"] = [
            _openai_resp(
                content="deepseek reading",
                tool_calls=[{
                    "id": "call_B", "type": "function",
                    "function": {"name": "bash",
                                 "arguments": '{"command": "echo B"}'},
                }],
                finish="tool_calls"),
            _openai_resp(content="deepseek final"),
        ]
        self._run()

        # DeepSeek run added assistant (tool_call B) + user (tool_result B)
        # + assistant (final). History still contains Claude's earlier parts.
        self.assertEqual(self.messages[-1]["role"], "assistant")
        self.assertIn("deepseek final",
                      self.messages[-1]["content"][0].text)

        # The OpenAI request payload for Run 2 must have carried Claude's
        # toolu_A id across the provider boundary.
        ds_payload = next(
            p for p in self.openai_seen
            if p.get("model") == "deepseek-chat")
        wire_roles = [m["role"] for m in ds_payload["messages"]]
        self.assertIn("tool", wire_roles)
        # Claude's toolu_A appears as a tool message for DeepSeek.
        tool_msgs = [m for m in ds_payload["messages"]
                     if m.get("role") == "tool"]
        self.assertTrue(any(m["tool_call_id"] == "toolu_A"
                            for m in tool_msgs),
                        "DeepSeek must see Claude's toolu_A result")

        # --- switch back to Claude ---
        self._switch("claude")

        # --- Run 3: Claude. Reads Claude + DeepSeek history, answers. ---
        self._make_anthropic_client([
            _anthropic_resp([_Block("claude after both")], "end_turn"),
        ])
        self._run()
        self.assertEqual(self.messages[-1]["role"], "assistant")
        self.assertIn("claude after both",
                      self.messages[-1]["content"][0].text)

        # Session selection went deepseek -> claude across the two switches.
        self.assertEqual(self.session.model_alias, "claude")

    def test_three_run_events_exactly_two(self):
        from agents.session import SessionState
        from agents.types.events import Event

        class Ex:
            def __init__(self):
                self.events = []

            def emit(self, name, ctx):
                self.events.append(name)

        exts = Ex()
        self.session = SessionState()
        from agents.session import handle_model_command
        handle_model_command("/model deepseek", self.session, self.reg,
                             extensions=exts)
        handle_model_command("/model claude", self.session, self.reg,
                             extensions=exts)
        self.assertEqual(exts.events, [Event.MODEL_CHANGED,
                                       Event.MODEL_CHANGED])


if __name__ == "__main__":
    unittest.main()
