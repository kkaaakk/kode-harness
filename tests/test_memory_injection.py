"""Integration tests — memory injection + compaction survival in harness_core agent loop."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agents.memory_manager import MemoryManager
from agents.token_budget import (
    TokenBudgetConfig,
    compact_messages,
    is_protected_memory_message,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "agents" / "harness_core.py"


def _load_harness_module(temp_cwd: Path):
    """Load harness_core.py with fake Anthropic/dotenv, pointing at temp_cwd."""
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
    spec = importlib.util.spec_from_file_location(
        "harness_core_integration_test", HARNESS_PATH
    )
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


class MemoryInjectionSmokeTests(unittest.TestCase):
    """Test that MemoryManager is importable and wired in harness_core.py."""

    def test_module_imports_memory_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = _load_harness_module(Path(tmp))
            self.assertTrue(hasattr(module, "MEMORY"))
            self.assertIsInstance(module.MEMORY, MemoryManager)

    def test_memory_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = _load_harness_module(Path(tmp))
            memory_dir = Path(tmp) / ".memory"
            self.assertTrue(memory_dir.is_dir())
            self.assertTrue((memory_dir / "MEMORY.md").exists())


class MemoryCompactionSurvivalTests(unittest.TestCase):
    """Verify that injected memory messages survive token budget compaction."""

    def test_injected_memory_survives_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            memory_dir = tmp_path / ".memory"
            mgr = MemoryManager(memory_dir)

            # Create memories of different types
            mgr.create("User prefers Chinese responses.", name="chinese-pref",
                       description="Language preference", memory_type="user_preference")
            mgr.create("Project uses FastAPI for backend.", name="fastapi-backend",
                       description="Tech stack", memory_type="long_term")

            # Build a conversation that would trigger compaction
            memory_msgs = mgr.load_all_as_messages()

            messages = list(memory_msgs)
            # Add a lot of content to trigger compaction
            for i in range(8):
                messages.append({
                    "role": "user",
                    "content": f"user message {i} " + ("x" * 160),
                })
                messages.append({
                    "role": "assistant",
                    "content": f"assistant message {i} " + ("y" * 160),
                })
            # Recent steps that should survive
            messages.append({"role": "user", "content": "recent user prompt"})
            messages.append({"role": "assistant", "content": "recent assistant reply"})

            def fake_summarize(*, previous_summary, history_text, summary_max_tokens):
                return "rolled summary"

            report = compact_messages(
                messages,
                config=TokenBudgetConfig(
                    max_context_tokens=220,
                    compress_threshold_ratio=0.5,
                    recent_steps_keep_count=2,
                    summary_max_tokens=64,
                ),
                summarize=fake_summarize,
                transcript_dir=tmp_path / ".transcripts",
            )

            compacted = json.dumps(messages, default=str)

            # Memory messages must survive
            self.assertIn("User prefers Chinese responses.", compacted)
            self.assertIn("Project uses FastAPI", compacted)

            # Recent messages must survive
            self.assertIn("recent user prompt", compacted)

            # Old messages must be gone
            self.assertNotIn("user message 0", compacted)

            # Compaction report should count protected messages
            self.assertGreater(report.protected_message_count, 0)

    def test_is_protected_memory_message_recognizes_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(Path(tmp) / ".memory")
            mgr.create("Test memory.", name="test", memory_type="long_term")

            messages = mgr.load_all_as_messages()
            self.assertGreater(len(messages), 0)

            for msg in messages:
                self.assertTrue(
                    is_protected_memory_message(msg),
                    f"Message should be recognized as protected: {msg}",
                )


class MemoryAgentLoopIntegrationTests(unittest.TestCase):
    """Integration test — agent_loop with memory injection."""

    def test_agent_loop_injects_memory_before_first_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            module = _load_harness_module(tmp_path)

            # Pre-populate a memory
            memory_dir = tmp_path / ".memory"
            mgr = MemoryManager(memory_dir)
            mgr.create("The answer is 42.", name="answer",
                       memory_type="long_term")

            module.TOKEN_BUDGET = TokenBudgetConfig(
                max_context_tokens=100000,
            )
            module.TRANSCRIPT_DIR = tmp_path / ".transcripts"

            captured_messages = []

            def fake_create(**kwargs):
                captured_messages.extend(kwargs.get("messages", []))
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(
                        type="text", text="I remember."
                    )],
                    stop_reason="end_turn",
                )

            module.client.messages.create = fake_create
            messages = [{"role": "user", "content": "What is the answer?"}]
            module.agent_loop(messages)

            # The model should have received memory-injected messages
            memory_contents = [
                m.get("content", "")
                for m in captured_messages
                if isinstance(m.get("metadata"), dict)
                and "memory_name" in m["metadata"]
            ]
            self.assertGreater(len(memory_contents), 0)
            self.assertTrue(any("The answer is 42." in c for c in memory_contents))


if __name__ == "__main__":
    unittest.main()
