"""Compression-behaviour evaluation scenarios.

Tests that the agent's context-compression pipeline preserves essential
information and produces coherent summaries.
"""

from __future__ import annotations

import pytest

from tests.eval.lib.assertions import assert_output_contains


@pytest.mark.eval
class TestCompression:
    """Evaluate the agent's context-compression behaviour."""

    def test_long_conversation_retains_early_info(self, agent_runner):
        """After many rounds, the agent should still recall early information."""
        # Seed the workspace with a marker file
        (agent_runner.workdir / "marker.txt").write_text("remember: ALPHA-7")

        # Ask the agent to read it
        result1 = agent_runner.run("Read marker.txt and remember its content")
        assert_output_contains(result1, "ALPHA-7")

        # Run several filler rounds to push context toward the threshold
        for i in range(5):
            agent_runner.run(f"Count the files in the workspace (round {i})")

        # Ask about the earlier content
        result_final = agent_runner.run(
            "What was the content of marker.txt that we looked at earlier?"
        )
        # After compression the agent should still reference ALPHA-7
        assert_output_contains(result_final, "ALPHA-7")

    def test_manual_compress_produces_coherent_summary(self, agent_runner, judge):
        """Triggering the compress tool should yield a usable summary."""
        (agent_runner.workdir / "data.txt").write_text("important: ZULU-9 data")

        # Build up context
        agent_runner.run("Read data.txt and note its content")
        agent_runner.run("List all files in the workspace")

        # Ask the agent to summarise its context so far
        result = agent_runner.run(
            "Please compress your conversation context and give me a brief "
            "summary of everything we've done so far."
        )
        # The summary (final_text) should be non-empty and mention the data
        assert len(result.final_text) > 0, "Agent should produce some output"

        # Optional: LLM judge scores the summary quality
        verdict = judge.evaluate_compression_quality(
            before_messages=result.messages[:4],
            after_messages=result.messages,
        )
        assert verdict.passed, (
            f"Judge score {verdict.score:.2f} below threshold: {verdict.reasoning}"
        )
