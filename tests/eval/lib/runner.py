"""AgentLoopRunner — drive agent_loop for evaluation scenarios.

Uses ``harness_core.agent_loop(messages, event_callback=cb)`` to collect
a full execution trace without modifying production code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Ensure repo root is on path so agents.* imports work
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BenchmarkResult:
    """Benchmark metrics for a single agent execution."""

    case_id: str
    tool_correct: bool
    turns: int
    tokens: int
    expected_tool: str
    actual_tools: list[str]


@dataclass
class RunResult:
    """Outcome of a single agent execution."""

    messages: list[dict]
    tool_calls: list[dict]
    final_text: str
    rounds: int
    events: list[dict]
    tokens: int = 0

    def to_benchmark(
        self,
        case_id: str,
        expected_tool: str,
    ) -> BenchmarkResult:
        """Convert to a benchmark metric record."""
        actual_tools = [tc.get("name", "") for tc in self.tool_calls]
        tool_correct = expected_tool in actual_tools
        return BenchmarkResult(
            case_id=case_id,
            tool_correct=tool_correct,
            turns=self.rounds,
            tokens=self.tokens,
            expected_tool=expected_tool,
            actual_tools=actual_tools,
        )


class AgentLoopRunner:
    """Thin wrapper around ``agent_loop`` that collects event traces."""

    def __init__(self, workdir: Path, client: Any):
        self.workdir = workdir
        self.client = client

    def run(self, prompt: str, *, max_turns: int = 10) -> RunResult:
        """Execute *prompt* through the agent loop and return structured results."""
        from agents import harness_core  # type: ignore[import-untyped]

        # Override globals for this run
        harness_core.WORKDIR = self.workdir
        harness_core.client = self.client

        events: list[dict] = []
        total_tokens = 0

        def callback(event: dict) -> None:
            nonlocal total_tokens
            events.append(event)
            # Track token usage from events
            if "tokens" in event:
                total_tokens += event["tokens"]
            # Hard-stop after max_turns tool-call rounds
            tool_rounds = sum(
                1 for e in events if e.get("type") == "tool_result"
            )
            if tool_rounds >= max_turns:
                raise _MaxTurnsReached()

        messages: list[dict] = [{"role": "user", "content": prompt}]
        try:
            harness_core.agent_loop(messages, event_callback=callback)
        except _MaxTurnsReached:
            pass

        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        text_events = [e for e in events if e.get("type") == "text"]
        final_text = text_events[-1]["text"] if text_events else ""

        return RunResult(
            messages=messages,
            tool_calls=tool_calls,
            final_text=final_text,
            rounds=len([e for e in events if e.get("type") == "tool_result"]),
            events=events,
            tokens=total_tokens,
        )


class _MaxTurnsReached(Exception):
    """Internal: raised by the callback when max_turns is hit."""
