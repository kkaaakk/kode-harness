"""Multi-step task evaluation scenarios.

Each scenario requires the agent to plan and execute multiple tool calls
in the correct order.  Primary assessment: structural (files created,
status transitions).  Secondary: LLM judge on planning quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval.lib.assertions import (
    assert_file_exists,
    assert_output_contains,
    assert_tool_called,
    assert_tool_call_order,
)


@pytest.mark.eval
class TestMultistep:
    """Evaluate multi-step task completion and planning."""

    def test_create_project_structure(self, agent_runner):
        """Agent should create a multi-file project from a single prompt."""
        result = agent_runner.run(
            "Create a minimal Python project with three files: "
            "src/app.py (with a main function), "
            "src/utils.py (with a helper function), "
            "and a README.md describing the project."
        )
        assert_tool_called(result, "write_file", min_count=3)
        assert_file_exists(agent_runner.workdir, "src/app.py")
        assert_file_exists(agent_runner.workdir, "src/utils.py")
        assert_file_exists(agent_runner.workdir, "README.md")

    def test_find_and_modify(self, agent_runner):
        """Agent should search for content and then edit the matching file."""
        (agent_runner.workdir / "a.py").write_text("status = 'draft'\n")
        (agent_runner.workdir / "b.py").write_text("status = 'draft'\n")

        result = agent_runner.run(
            "Find which .py files contain 'draft' and change them all to 'published'"
        )
        assert_tool_called(result, "grep_search")
        assert_tool_called(result, "edit_file", min_count=1)

        for name in ("a.py", "b.py"):
            content = (agent_runner.workdir / name).read_text()
            assert "published" in content, f"{name} should contain 'published'"

    def test_background_task_lifecycle(self, agent_runner):
        """Agent should start a background task and then check its result."""
        result = agent_runner.run(
            "Run 'echo hello-from-bg' as a background task, "
            "wait a moment, then check its output."
        )
        assert_tool_called(result, "background_run")
        assert_tool_called(result, "check_background")
        assert_output_contains(result, "hello-from-bg")

    def test_task_management_workflow(self, agent_runner):
        """Agent should create, update, and complete a persistent task."""
        result = agent_runner.run(
            "Create a task called 'Setup CI' with description 'configure GitHub Actions'. "
            "Then mark it as in_progress. "
            "Then mark it as completed."
        )
        assert_tool_called(result, "task_create")
        assert_tool_called(result, "task_update", min_count=2)
