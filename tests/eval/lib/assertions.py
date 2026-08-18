"""Structured assertions for eval RunResult objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.eval.lib.runner import RunResult


def assert_tool_called(result: RunResult, tool_name: str, min_count: int = 1) -> None:
    """Assert that *tool_name* was invoked at least *min_count* times."""
    count = sum(1 for tc in result.tool_calls if tc.get("name") == tool_name)
    assert count >= min_count, (
        f"Expected {tool_name} called >= {min_count} time(s), got {count}. "
        f"Tools called: {[tc.get('name') for tc in result.tool_calls]}"
    )


def assert_tool_not_called(result: RunResult, tool_name: str) -> None:
    """Assert that *tool_name* was never invoked."""
    count = sum(1 for tc in result.tool_calls if tc.get("name") == tool_name)
    assert count == 0, (
        f"Expected {tool_name} NOT called, but it was called {count} time(s)"
    )


def assert_tool_call_order(result: RunResult, expected_order: list[str]) -> None:
    """Assert that *expected_order* appears (in order) within the tool call sequence."""
    actual = [tc.get("name") for tc in result.tool_calls]
    idx = 0
    for name in expected_order:
        while idx < len(actual) and actual[idx] != name:
            idx += 1
        assert idx < len(actual), (
            f"Expected order {expected_order} not found in {actual}"
        )
        idx += 1


def assert_output_contains(result: RunResult, text: str) -> None:
    """Assert the agent's final text output contains *text*."""
    assert text in result.final_text, (
        f"Expected {text!r} in final output, got: {result.final_text[:200]!r}"
    )


def assert_file_exists(workdir: Path, relative_path: str) -> None:
    """Assert that *relative_path* was created inside *workdir*."""
    full = workdir / relative_path
    assert full.exists(), f"Expected file {relative_path} to exist at {full}"
