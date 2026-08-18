"""Tool-selection evaluation scenarios.

Each test verifies that the agent picks the correct tool for a given task.
Primary assessment: structural assertions (tool was/was-not called).
Optional secondary: LLM-as-a-Judge scoring.
"""

from __future__ import annotations

import pytest

from tests.eval.lib.assertions import (
    assert_file_exists,
    assert_output_contains,
    assert_tool_called,
    assert_tool_not_called,
)


@pytest.mark.eval
class TestToolSelection:
    """Evaluate whether the agent picks the right tool for common tasks."""

    def test_search_content_uses_grep(self, agent_runner):
        (agent_runner.workdir / "code.py").write_text("# TODO: fix this\nprint('hi')")
        result = agent_runner.run("Find all Python files that contain TODO")
        assert_tool_called(result, "grep_search")

    def test_list_files_uses_glob(self, agent_runner):
        (agent_runner.workdir / "README.md").write_text("# Hello")
        (agent_runner.workdir / "notes.md").write_text("# Notes")
        result = agent_runner.run("List all .md files in the workspace")
        assert_tool_called(result, "glob_search")

    def test_read_file_uses_read(self, agent_runner):
        (agent_runner.workdir / "target.txt").write_text("secret content")
        result = agent_runner.run("Read the file target.txt and tell me its content")
        assert_tool_called(result, "read_file")
        assert_output_contains(result, "secret content")

    def test_write_creates_file(self, agent_runner):
        result = agent_runner.run(
            "Create a file called hello.py containing: print('hello world')"
        )
        assert_tool_called(result, "write_file")
        assert_file_exists(agent_runner.workdir, "hello.py")

    def test_edit_modifies_file(self, agent_runner):
        (agent_runner.workdir / "main.py").write_text("x = 'old_value'\n")
        result = agent_runner.run(
            "In main.py, replace 'old_value' with 'new_value'"
        )
        assert_tool_called(result, "edit_file")
        content = (agent_runner.workdir / "main.py").read_text()
        assert "new_value" in content

    def test_computation_uses_bash(self, agent_runner):
        result = agent_runner.run("Calculate the result of 123 * 456 using Python")
        assert_tool_called(result, "bash")
        assert_output_contains(result, "56088")

    def test_search_prefers_grep_over_bash(self, agent_runner):
        (agent_runner.workdir / "config.json").write_text('{"key": "value"}')
        result = agent_runner.run("Search for configuration files in the workspace")
        # Agent should use grep_search or glob_search, not bash
        search_tools = [
            tc.get("name") for tc in result.tool_calls
            if tc.get("name") in ("grep_search", "glob_search")
        ]
        assert len(search_tools) > 0, (
            "Agent should use grep_search or glob_search for file search, "
            f"but called: {[tc.get('name') for tc in result.tool_calls]}"
        )
