"""test_compression.py — Non-LLM unit tests for compression & token budget.

Covers two modules:
1. agents/compression.py  — estimate_tokens, microcompact
2. agents/token_budget.py — estimate_tokens, MicroCompactState,
   micro_compact_tool_results, is_protected_memory_message,
   compact_messages, _extract_memory_layers, _split_recent_steps

compression.py depends on agents.config (TRANSCRIPT_DIR, client, MODEL)
which requires mocking. token_budget.py is self-contained.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: load compression.py with mocked config
# ---------------------------------------------------------------------------


@pytest.fixture()
def comp_module(tmp_path: Path):
    """Load agents.compression with mocked config dependencies."""
    fake_config = types.ModuleType("agents.config")
    fake_config.TRANSCRIPT_DIR = tmp_path / "transcripts"
    fake_config.client = None
    fake_config.MODEL = "test-model"

    prev_config = sys.modules.get("agents.config")
    prev_comp = sys.modules.get("agents.compression")
    sys.modules["agents.config"] = fake_config

    if "agents.compression" in sys.modules:
        del sys.modules["agents.compression"]
    from agents.compression import estimate_tokens, microcompact

    with patch("agents.compression.TRANSCRIPT_DIR", fake_config.TRANSCRIPT_DIR):
        yield types.SimpleNamespace(
            estimate_tokens=estimate_tokens,
            microcompact=microcompact,
            TRANSCRIPT_DIR=fake_config.TRANSCRIPT_DIR,
        )

    if prev_config is None:
        sys.modules.pop("agents.config", None)
    else:
        sys.modules["agents.config"] = prev_config
    if prev_comp is not None:
        sys.modules["agents.compression"] = prev_comp
    else:
        sys.modules.pop("agents.compression", None)


# ===================================================================
# compression.py — estimate_tokens
# ===================================================================


class TestCompressionEstimateTokens:

    def test_empty_list_returns_zero_or_small(self, comp_module):
        result = comp_module.estimate_tokens([])
        assert result >= 0

    def test_known_string_approximate(self, comp_module):
        """json.dumps('hello world') = '"hello world"' → 14 chars // 4 = 3."""
        messages = [{"role": "user", "content": "hello world"}]
        result = comp_module.estimate_tokens(messages)
        # Rough: json.dumps gives ~50 chars → ~12 tokens
        assert result > 0
        assert result < 100

    def test_larger_payload_more_tokens(self, comp_module):
        small = comp_module.estimate_tokens([{"x": "a"}])
        big = comp_module.estimate_tokens([{"x": "a" * 1000}])
        assert big > small


# ===================================================================
# compression.py — microcompact
# ===================================================================


class TestCompressionMicrocompact:

    def _make_tool_result_msg(self, tool_id: str, content: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": content}
            ],
        }

    def test_keeps_all_when_three_or_fewer(self, comp_module):
        messages = [
            self._make_tool_result_msg("t1", "short"),
            self._make_tool_result_msg("t2", "short"),
            self._make_tool_result_msg("t3", "short"),
        ]
        comp_module.microcompact(messages)
        # None should be cleared
        for msg in messages:
            assert msg["content"][0]["content"] == "short"

    def test_clears_old_keeps_recent_three(self, comp_module):
        messages = [
            self._make_tool_result_msg(f"t{i}", "x" * 200)
            for i in range(5)
        ]
        comp_module.microcompact(messages)
        # First 2 should be cleared (>100 chars)
        assert messages[0]["content"][0]["content"] == "[cleared]"
        assert messages[1]["content"][0]["content"] == "[cleared]"
        # Last 3 should be preserved
        for msg in messages[2:]:
            assert "x" * 200 in msg["content"][0]["content"]

    def test_short_content_not_cleared(self, comp_module):
        """Content ≤ 100 chars is not cleared even if old."""
        messages = [
            self._make_tool_result_msg(f"t{i}", "short")
            for i in range(5)
        ]
        comp_module.microcompact(messages)
        # "short" is only 5 chars, should NOT be cleared
        for msg in messages:
            assert msg["content"][0]["content"] == "short"

    def test_ignores_non_tool_result_messages(self, comp_module):
        messages = [
            {"role": "user", "content": "plain text"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "another plain"},
        ]
        comp_module.microcompact(messages)
        assert messages[0]["content"] == "plain text"
        assert messages[1]["content"] == "reply"

    def test_exactly_four_clears_one(self, comp_module):
        messages = [
            self._make_tool_result_msg(f"t{i}", "y" * 150)
            for i in range(4)
        ]
        comp_module.microcompact(messages)
        assert messages[0]["content"][0]["content"] == "[cleared]"
        for msg in messages[1:]:
            assert "y" * 150 in msg["content"][0]["content"]


# ===================================================================
# token_budget.py — estimate_tokens
# ===================================================================

from agents.token_budget import (
    MicroCompactState,
    TokenBudgetConfig,
    BudgetCheckReport,
    CompactionReport,
    estimate_tokens as budget_estimate_tokens,
    micro_compact_tool_results,
    compact_messages,
    is_protected_memory_message,
    is_summary_message,
    make_summary_message,
    extract_summary_text,
    save_transcript,
)


class TestTokenBudgetEstimateTokens:

    def test_string_input(self):
        result = budget_estimate_tokens("hello world")
        assert result >= 1

    def test_list_input(self):
        result = budget_estimate_tokens([{"role": "user", "content": "hi"}])
        assert result >= 1

    def test_larger_input_more_tokens(self):
        small = budget_estimate_tokens("hi")
        large = budget_estimate_tokens("word " * 500)
        assert large > small

    def test_empty_string_returns_zero_or_positive(self):
        """tiktoken returns 0 for empty string; fallback returns max(1, ...)."""
        assert budget_estimate_tokens("") >= 0


# ===================================================================
# token_budget.py — is_protected_memory_message
# ===================================================================


class TestIsProtectedMemoryMessage:

    def test_user_preference_is_protected(self):
        msg = {"role": "user", "content": "x", "metadata": {"memory_type": "user_preference"}}
        assert is_protected_memory_message(msg) is True

    def test_long_term_is_protected(self):
        msg = {"role": "user", "content": "x", "metadata": {"memory_type": "long_term"}}
        assert is_protected_memory_message(msg) is True

    def test_protected_type_is_protected(self):
        msg = {"role": "user", "content": "x", "metadata": {"memory_type": "protected"}}
        assert is_protected_memory_message(msg) is True

    def test_preference_alias_is_protected(self):
        msg = {"role": "user", "content": "x", "metadata": {"memory_type": "preference"}}
        assert is_protected_memory_message(msg) is True

    def test_no_metadata_not_protected(self):
        msg = {"role": "user", "content": "x"}
        assert is_protected_memory_message(msg) is False

    def test_unrelated_metadata_not_protected(self):
        msg = {"role": "user", "content": "x", "metadata": {"memory_type": "ephemeral"}}
        assert is_protected_memory_message(msg) is False

    def test_case_insensitive(self):
        msg = {"role": "user", "content": "x", "metadata": {"memory_type": "LONG_TERM"}}
        assert is_protected_memory_message(msg) is True


# ===================================================================
# token_budget.py — summary message helpers
# ===================================================================


class TestSummaryHelpers:

    def test_is_summary_message_by_tag(self):
        msg = make_summary_message("some summary", Path("/tmp/t.jsonl"))
        assert is_summary_message(msg) is True

    def test_is_summary_message_by_metadata(self):
        msg = {"role": "user", "content": "x", "metadata": {"kind": "conversation_summary"}}
        assert is_summary_message(msg) is True

    def test_regular_message_is_not_summary(self):
        msg = {"role": "user", "content": "hello"}
        assert is_summary_message(msg) is False

    def test_extract_summary_text(self):
        msg = make_summary_message("important context here", Path("/tmp/t.jsonl"))
        text = extract_summary_text(msg)
        assert "important context here" in text


# ===================================================================
# token_budget.py — micro_compact_tool_results
# ===================================================================


class TestMicroCompactToolResults:

    def _make_pair(self, tool_id: str, tool_name: str, result_content: str):
        return [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": tool_name,
                             "input": {"cmd": "x"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_id,
                             "content": result_content}],
            },
        ]

    def test_keeps_recent_results(self):
        state = MicroCompactState()
        messages = []
        for i in range(3):
            messages.extend(self._make_pair(f"t{i}", "bash", f"result_{i}" + "x" * 150))

        micro_compact_tool_results(messages, state=state, keep_recent=3)
        # All 3 should be kept
        for i in range(3):
            idx = i * 2 + 1
            assert f"result_{i}" in messages[idx]["content"][0]["content"]

    def test_clears_old_beyond_keep_recent(self):
        state = MicroCompactState()
        messages = []
        for i in range(5):
            messages.extend(self._make_pair(f"t{i}", "bash", f"result_{i}" + "x" * 150))

        micro_compact_tool_results(messages, state=state, keep_recent=2)
        # First 3 should be cleared
        assert messages[1]["content"][0]["content"] == "[cleared]"
        assert messages[3]["content"][0]["content"] == "[cleared]"
        assert messages[5]["content"][0]["content"] == "[cleared]"
        # Last 2 preserved
        assert "result_3" in messages[7]["content"][0]["content"]
        assert "result_4" in messages[9]["content"][0]["content"]

    def test_preserve_result_tools_not_cleared(self):
        state = MicroCompactState()
        messages = []
        for i in range(4):
            name = "grep_search" if i == 0 else "bash"
            messages.extend(self._make_pair(f"t{i}", name, f"result_{i}" + "x" * 150))

        micro_compact_tool_results(
            messages, state=state, keep_recent=1,
            preserve_result_tools={"grep_search"},
        )
        # grep_search result should survive even though it's old
        assert "result_0" in messages[1]["content"][0]["content"]

    def test_short_content_not_cleared(self):
        state = MicroCompactState()
        messages = []
        for i in range(5):
            messages.extend(self._make_pair(f"t{i}", "bash", "short"))

        micro_compact_tool_results(messages, state=state, keep_recent=1)
        # "short" is < 100 chars, should NOT be cleared
        for i in range(5):
            assert messages[i * 2 + 1]["content"][0]["content"] == "short"

    def test_cursor_advances(self):
        state = MicroCompactState()
        messages = self._make_pair("t0", "bash", "r0" + "x" * 150)
        micro_compact_tool_results(messages, state=state, keep_recent=1)
        assert state.last_message_index == 2

        # Add more messages
        messages.extend(self._make_pair("t1", "bash", "r1" + "y" * 150))
        micro_compact_tool_results(messages, state=state, keep_recent=1)
        assert state.last_message_index == 4

    def test_stale_state_triggers_reset(self):
        """If messages list is replaced (different id), state resets."""
        state = MicroCompactState()
        messages = self._make_pair("t0", "bash", "r0" + "x" * 150)
        micro_compact_tool_results(messages, state=state, keep_recent=1)

        # Replace messages entirely
        new_messages = self._make_pair("t1", "bash", "r1" + "y" * 150)
        micro_compact_tool_results(new_messages, state=state, keep_recent=1)
        # Should work without error, state was reset
        assert state.last_message_index == 2


# ===================================================================
# token_budget.py — compact_messages
# ===================================================================


class TestCompactMessages:

    def test_protected_messages_survive(self):
        messages = [
            {"role": "user", "content": "prefer dark mode",
             "metadata": {"memory_type": "user_preference"}},
        ]
        for i in range(6):
            messages.append({"role": "user", "content": f"old msg {i}" + "x" * 100})
            messages.append({"role": "assistant", "content": f"old reply {i}" + "y" * 100})
        messages.append({"role": "user", "content": "recent prompt"})

        def fake_summarize(*, previous_summary, history_text, summary_max_tokens):
            return "rolled up summary"

        with tempfile.TemporaryDirectory() as tmp:
            report = compact_messages(
                messages,
                config=TokenBudgetConfig(
                    max_context_tokens=200,
                    compress_threshold_ratio=0.5,
                    recent_steps_keep_count=1,
                    summary_max_tokens=64,
                ),
                summarize=fake_summarize,
                transcript_dir=Path(tmp),
            )

        compacted = json.dumps(messages, default=str)
        # Protected memory survives
        assert "prefer dark mode" in compacted
        # Recent step survives
        assert "recent prompt" in compacted
        # Summary was inserted
        assert "rolled up summary" in compacted
        # Old messages removed
        assert "old msg 0" not in compacted
        # Report is correct
        assert report.protected_message_count == 1
        assert report.before_tokens > report.after_tokens

    def test_previous_summary_rolled_forward(self):
        messages = [
            {"role": "user",
             "content": "<conversation_summary>prior context</conversation_summary>"},
        ]
        for i in range(6):
            messages.append({"role": "user", "content": f"step {i}" + "x" * 100})
            messages.append({"role": "assistant", "content": f"reply {i}" + "y" * 100})
        messages.append({"role": "user", "content": "latest"})

        captured = {}

        def fake_summarize(*, previous_summary, history_text, summary_max_tokens):
            captured["previous_summary"] = previous_summary
            return "new summary"

        with tempfile.TemporaryDirectory() as tmp:
            compact_messages(
                messages,
                config=TokenBudgetConfig(
                    max_context_tokens=200,
                    compress_threshold_ratio=0.5,
                    recent_steps_keep_count=1,
                ),
                summarize=fake_summarize,
                transcript_dir=Path(tmp),
            )

        assert "prior context" in captured["previous_summary"]

    def test_transcript_saved(self):
        messages = [{"role": "user", "content": "hello"}]
        for i in range(6):
            messages.append({"role": "user", "content": f"s{i}" + "x" * 200})
            messages.append({"role": "assistant", "content": f"r{i}" + "y" * 200})

        with tempfile.TemporaryDirectory() as tmp:
            report = compact_messages(
                messages,
                config=TokenBudgetConfig(max_context_tokens=200,
                                         compress_threshold_ratio=0.5,
                                         recent_steps_keep_count=1),
                summarize=lambda **_: "summary",
                transcript_dir=Path(tmp),
            )
            assert report.transcript_path.exists()
            raw = report.transcript_path.read_text()
            assert "hello" in raw


# ===================================================================
# token_budget.py — TokenBudgetConfig
# ===================================================================


class TestTokenBudgetConfig:

    def test_defaults(self):
        cfg = TokenBudgetConfig()
        assert cfg.max_context_tokens == 100000
        assert cfg.compress_threshold_ratio == 0.85
        assert cfg.recent_steps_keep_count == 3
        assert cfg.summary_max_tokens == 2000

    def test_threshold_tokens(self):
        cfg = TokenBudgetConfig(max_context_tokens=1000, compress_threshold_ratio=0.5)
        assert cfg.threshold_tokens == 500

    def test_frozen(self):
        cfg = TokenBudgetConfig()
        with pytest.raises(AttributeError):
            cfg.max_context_tokens = 999
