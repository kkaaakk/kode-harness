"""test_subagent.py — Subagent tool selection, turn limits, result summarisation.

subagent.py depends on agents.config (client, MODEL) + agents.base_tools.
We mock LLM calls and tool handlers to test the control flow in isolation.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixture: set up mocked subagent module
# ---------------------------------------------------------------------------


@pytest.fixture()
def sa_env():
    """Provide a freshly-loaded subagent module with all deps mocked."""
    # --- fake agents.config ---
    fake_config = types.ModuleType("agents.config")
    fake_config.client = MagicMock()
    fake_config.MODEL = "test-model"

    # --- fake agents.base_tools ---
    fake_bt = types.ModuleType("agents.base_tools")
    fake_bt.run_bash = MagicMock(return_value="bash-result")
    fake_bt.run_read = MagicMock(return_value="read-result")
    fake_bt.run_write = MagicMock(return_value="write-result")
    fake_bt.run_edit = MagicMock(return_value="edit-result")

    saved = {}
    for mod_name in ("agents.config", "agents.base_tools", "agents.subagent"):
        saved[mod_name] = sys.modules.get(mod_name)

    sys.modules["agents.config"] = fake_config
    sys.modules["agents.base_tools"] = fake_bt

    if "agents.subagent" in sys.modules:
        del sys.modules["agents.subagent"]
    from agents.subagent import run_subagent

    yield types.SimpleNamespace(
        run_subagent=run_subagent,
        client=fake_config.client,
        bt=fake_bt,
    )

    for mod_name, prev in saved.items():
        if prev is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = prev


# ---------------------------------------------------------------------------
# Helpers: build fake LLM responses
# ---------------------------------------------------------------------------


class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, name: str, tool_id: str, input_data: dict):
        self.type = "tool_use"
        self.name = name
        self.id = tool_id
        self.input = input_data


def _end_turn_response(text: str):
    """LLM returns a final text answer (no tool use)."""
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [FakeTextBlock(text)]
    return resp


def _tool_use_response(tools: list[tuple[str, str, dict]]):
    """LLM requests tool use. Each tuple: (name, id, input)."""
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [FakeToolUseBlock(n, tid, inp) for n, tid, inp in tools]
    return resp


def _empty_response():
    """LLM returns end_turn with no text blocks."""
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = []
    return resp


# ===================================================================
# Tool selection — Explore vs general-purpose
# ===================================================================


class TestSubagentToolSelection:

    def test_explore_mode_only_has_read_tools(self, sa_env):
        """Explore subagent should only receive bash + read_file."""
        sa_env.client.messages.create.return_value = _end_turn_response("done")
        sa_env.run_subagent("Explore the codebase", agent_type="Explore")

        # Inspect the tools argument passed to the LLM
        call_kwargs = sa_env.client.messages.create.call_args
        tool_names = {t["name"] for t in call_kwargs.kwargs["tools"]}
        assert tool_names == {"bash", "read_file"}

    def test_general_purpose_has_write_tools(self, sa_env):
        """General-purpose subagent should also get write_file + edit_file."""
        sa_env.client.messages.create.return_value = _end_turn_response("done")
        sa_env.run_subagent("Build something", agent_type="general-purpose")

        call_kwargs = sa_env.client.messages.create.call_args
        tool_names = {t["name"] for t in call_kwargs.kwargs["tools"]}
        assert tool_names == {"bash", "read_file", "write_file", "edit_file"}

    def test_default_agent_type_is_explore(self, sa_env):
        sa_env.client.messages.create.return_value = _end_turn_response("ok")
        sa_env.run_subagent("Look around")

        call_kwargs = sa_env.client.messages.create.call_args
        tool_names = {t["name"] for t in call_kwargs.kwargs["tools"]}
        assert tool_names == {"bash", "read_file"}


# ===================================================================
# Turn limit — max 30 iterations
# ===================================================================


class TestSubagentTurnLimit:

    def test_stops_on_end_turn_immediately(self, sa_env):
        """If LLM returns end_turn on first call, only one API call is made."""
        sa_env.client.messages.create.return_value = _end_turn_response("quick answer")
        result = sa_env.run_subagent("What?")
        assert sa_env.client.messages.create.call_count == 1
        assert result == "quick answer"

    def test_max_turns_capped_at_30(self, sa_env):
        """Even if LLM keeps requesting tool_use, loop stops at 30 iterations."""
        sa_env.client.messages.create.return_value = _tool_use_response(
            [("bash", "t1", {"command": "ls"})]
        )
        sa_env.run_subagent("Loop forever")
        assert sa_env.client.messages.create.call_count == 30

    def test_tool_use_then_end_turn(self, sa_env):
        """LLM uses a tool, then returns a final answer on the second turn."""
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("bash", "t1", {"command": "ls"})]),
            _end_turn_response("Found 5 files"),
        ]
        result = sa_env.run_subagent("List files")
        assert sa_env.client.messages.create.call_count == 2
        assert result == "Found 5 files"


# ===================================================================
# Tool execution flow
# ===================================================================


class TestSubagentToolExecution:

    def test_bash_tool_dispatched(self, sa_env):
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("bash", "t1", {"command": "echo hi"})]),
            _end_turn_response("done"),
        ]
        sa_env.run_subagent("Run echo")
        sa_env.bt.run_bash.assert_called_once_with("echo hi")

    def test_read_file_tool_dispatched(self, sa_env):
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("read_file", "t1", {"path": "README.md"})]),
            _end_turn_response("done"),
        ]
        sa_env.run_subagent("Read README")
        sa_env.bt.run_read.assert_called_once_with("README.md")

    def test_write_file_tool_dispatched(self, sa_env):
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("write_file", "t1", {"path": "out.txt", "content": "data"})]),
            _end_turn_response("done"),
        ]
        sa_env.run_subagent("Write file", agent_type="general-purpose")
        sa_env.bt.run_write.assert_called_once_with("out.txt", "data")

    def test_edit_file_tool_dispatched(self, sa_env):
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([
                ("edit_file", "t1", {"path": "a.py", "old_text": "x", "new_text": "y"})
            ]),
            _end_turn_response("done"),
        ]
        sa_env.run_subagent("Edit file", agent_type="general-purpose")
        sa_env.bt.run_edit.assert_called_once_with("a.py", "x", "y")

    def test_unknown_tool_returns_unknown(self, sa_env):
        """If the LLM requests a tool we don't support, handler returns 'Unknown tool'."""
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("magic_tool", "t1", {"spell": "abracadabra"})]),
            _end_turn_response("done"),
        ]
        # Should not crash
        result = sa_env.run_subagent("Cast spell")
        assert result == "done"

    def test_multiple_tools_in_one_turn(self, sa_env):
        """LLM can call multiple tools in a single response."""
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([
                ("bash", "t1", {"command": "ls"}),
                ("read_file", "t2", {"path": "main.py"}),
            ]),
            _end_turn_response("analyzed"),
        ]
        result = sa_env.run_subagent("Analyze project")
        assert result == "analyzed"
        sa_env.bt.run_bash.assert_called_once()
        sa_env.bt.run_read.assert_called_once()

    def test_tool_result_truncated_at_50000(self, sa_env):
        """Tool result content is truncated to 50000 chars."""
        sa_env.bt.run_bash.return_value = "x" * 60000
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("bash", "t1", {"command": "big_output"})]),
            _end_turn_response("done"),
        ]
        sa_env.run_subagent("Get big output")

        # The second API call carries the tool_result in its message history
        second_call = sa_env.client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        # Find the tool_result message (role=user, content is a list)
        tool_result_msg = next(
            m for m in messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        )
        result_content = tool_result_msg["content"][0]["content"]
        assert len(result_content) == 50000


# ===================================================================
# Result summarisation
# ===================================================================


class TestSubagentResult:

    def test_returns_text_from_final_response(self, sa_env):
        sa_env.client.messages.create.return_value = _end_turn_response("The answer is 42")
        result = sa_env.run_subagent("What is the answer?")
        assert result == "The answer is 42"

    def test_returns_no_summary_when_empty_content(self, sa_env):
        sa_env.client.messages.create.return_value = _empty_response()
        result = sa_env.run_subagent("Silent query")
        assert result == "(no summary)"

    def test_multiple_text_blocks_concatenated(self, sa_env):
        resp = MagicMock()
        resp.stop_reason = "end_turn"
        resp.content = [FakeTextBlock("part1 "), FakeTextBlock("part2")]
        sa_env.client.messages.create.return_value = resp
        result = sa_env.run_subagent("Multi-part")
        assert result == "part1 part2"

    def test_messages_accumulate_across_turns(self, sa_env):
        """Verify conversation history grows with each turn."""
        sa_env.client.messages.create.side_effect = [
            _tool_use_response([("bash", "t1", {"command": "step1"})]),
            _tool_use_response([("bash", "t2", {"command": "step2"})]),
            _end_turn_response("final"),
        ]
        sa_env.run_subagent("Multi-step")

        assert sa_env.client.messages.create.call_count == 3
        # Third call should have accumulated history:
        # user(prompt) + assistant(tool_use) + user(tool_result)
        # + assistant(tool_use) + user(tool_result)
        # Note: the 3rd call's response hasn't been appended yet
        third_call = sa_env.client.messages.create.call_args_list[2]
        msgs = third_call.kwargs["messages"]
        assert len(msgs) == 6  # prompt + (assistant+user) × 2 + final assistant
