from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agents.token_budget import (
    MicroCompactState,
    TokenBudgetConfig,
    micro_compact_tool_results,
    prepare_messages_for_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "agents" / "harness_core.py"


def _load_harness_module(temp_cwd: Path):
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
    spec = importlib.util.spec_from_file_location("harness_core_budget_under_test", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {HARNESS_PATH}")
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


class TokenBudgetCompactionTests(unittest.TestCase):
    def test_micro_compact_uses_cursor_and_recent_result_buffer(self):
        state = MicroCompactState()
        messages = []

        for index in range(2):
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": f"tool_{index}",
                                "name": "bash",
                                "input": {"command": "echo hi"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"tool_{index}",
                                "content": f"result {index} " + ("x" * 160),
                            }
                        ],
                    },
                ]
            )

        micro_compact_tool_results(messages, state=state, keep_recent=2)
        self.assertIn("result 0", messages[1]["content"][0]["content"])
        self.assertIn("result 1", messages[3]["content"][0]["content"])
        self.assertEqual(state.last_message_index, len(messages))

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_2",
                            "name": "bash",
                            "input": {"command": "echo later"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_2",
                            "content": "result 2 " + ("y" * 160),
                        }
                    ],
                },
            ]
        )

        micro_compact_tool_results(messages, state=state, keep_recent=2)
        self.assertEqual(messages[1]["content"][0]["content"], "[cleared]")
        self.assertIn("result 1", messages[3]["content"][0]["content"])
        self.assertIn("result 2", messages[5]["content"][0]["content"])
        self.assertEqual(state.last_message_index, len(messages))

        micro_compact_tool_results(messages, state=state, keep_recent=2)
        self.assertIn("result 1", messages[3]["content"][0]["content"])
        self.assertIn("result 2", messages[5]["content"][0]["content"])

    def test_micro_compact_resets_cursor_when_messages_are_replaced(self):
        state = MicroCompactState()
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "old", "name": "bash"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "old",
                        "content": "old result " + ("x" * 160),
                    }
                ],
            },
        ]
        micro_compact_tool_results(messages, state=state, keep_recent=1)

        messages[:] = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "new", "name": "bash"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "new",
                        "content": "new result " + ("y" * 160),
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "newer", "name": "bash"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "newer",
                        "content": "newer result " + ("z" * 160),
                    }
                ],
            },
        ]

        micro_compact_tool_results(messages, state=state, keep_recent=1)
        self.assertEqual(messages[1]["content"][0]["content"], "[cleared]")
        self.assertIn("newer result", messages[3]["content"][0]["content"])

    def test_budget_compaction_rolls_summary_and_preserves_recent_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            messages = [
                {
                    "role": "user",
                    "content": "User prefers Chinese responses.",
                    "metadata": {"memory_type": "long_term"},
                },
                {
                    "role": "user",
                    "content": (
                        "<conversation_summary>\n"
                        "Previous summary: existing s06 compact flow found.\n"
                        "</conversation_summary>"
                    ),
                },
            ]
            for index in range(8):
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": f"old user step {index} " + ("x" * 160),
                        },
                        {
                            "role": "assistant",
                            "content": f"old assistant step {index} " + ("y" * 160),
                        },
                    ]
                )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": "recent step alpha should remain raw",
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_recent",
                                "name": "bash",
                                "input": {"command": "pwd"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_recent",
                                "content": "recent tool result should survive",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "recent assistant response should survive",
                    },
                    {
                        "role": "user",
                        "content": "recent step beta should also remain raw",
                    },
                ]
            )

            summarize_call = {}

            def fake_summarize(*, previous_summary, history_text, summary_max_tokens):
                summarize_call["previous_summary"] = previous_summary
                summarize_call["history_text"] = history_text
                summarize_call["summary_max_tokens"] = summary_max_tokens
                return f"rolled summary includes prior: {previous_summary}"

            report = prepare_messages_for_model(
                messages,
                config=TokenBudgetConfig(
                    max_context_tokens=220,
                    compress_threshold_ratio=0.5,
                    recent_steps_keep_count=2,
                    summary_max_tokens=64,
                ),
                summarize=fake_summarize,
                transcript_dir=Path(tmp),
                logger=lambda _: None,
            )

            compacted = json.dumps(messages, default=str)
            self.assertTrue(report.triggered)
            self.assertGreater(report.before_tokens, report.after_tokens)
            self.assertIn("Previous summary: existing s06 compact flow found.", compacted)
            self.assertIn("User prefers Chinese responses.", compacted)
            self.assertIn("recent step alpha should remain raw", compacted)
            self.assertIn("recent tool result should survive", compacted)
            self.assertIn("recent step beta should also remain raw", compacted)
            self.assertNotIn("old user step 0", compacted)
            self.assertIn("old user step 0", summarize_call["history_text"])
            self.assertEqual(summarize_call["summary_max_tokens"], 64)

    def test_harness_sends_summary_to_next_model_call_after_budget_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = _load_harness_module(Path(tmp))
            module.TOKEN_BUDGET = module.TokenBudgetConfig(
                max_context_tokens=180,
                compress_threshold_ratio=0.5,
                recent_steps_keep_count=1,
                summary_max_tokens=80,
            )
            module.TRANSCRIPT_DIR = Path(tmp) / ".transcripts"

            captured = {}

            class TextBlock:
                type = "text"

                def __init__(self, text: str):
                    self.text = text

            class Response:
                def __init__(self, text: str):
                    self.content = [TextBlock(text)]
                    self.stop_reason = "end_turn"

            def fake_create(**kwargs):
                if "tools" in kwargs:
                    captured["model_messages"] = kwargs["messages"]
                    return Response("done")
                return Response("summary from fake budget compactor")

            module.client.messages.create = fake_create
            messages = [
                {"role": "user", "content": "older turn " + ("x" * 400)},
                {"role": "assistant", "content": "older answer " + ("y" * 400)},
                {"role": "user", "content": "recent live user prompt"},
            ]

            module.agent_loop(messages)

            sent = json.dumps(captured["model_messages"], default=str)
            self.assertIn("<conversation_summary>", sent)
            self.assertIn("summary from fake budget compactor", sent)
            self.assertIn("recent live user prompt", sent)

            trace_dir = Path(tmp) / ".team" / "traces"
            events = []
            for path in sorted(trace_dir.glob("trace_*.jsonl")):
                events.extend(json.loads(line) for line in path.read_text().splitlines())
            compaction_event = next(event for event in events if event["event"] == "compaction_report")
            self.assertEqual(compaction_event["compaction_kind"], "auto")
            self.assertIn("before_tokens", compaction_event["output_summary"])


if __name__ == "__main__":
    unittest.main()
