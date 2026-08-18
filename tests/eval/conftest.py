"""Eval-specific pytest configuration.

All tests under ``tests/eval/`` are automatically skipped when
``ANTHROPIC_API_KEY`` is not set in the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    """Auto-skip every test in this directory when no API key is available."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def api_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def llm_client(api_key):
    from anthropic import Anthropic
    return Anthropic()


@pytest.fixture
def agent_runner(tmp_path, llm_client):
    """Provide an AgentLoopRunner backed by a fresh tmp workdir."""
    from tests.eval.lib.runner import AgentLoopRunner
    return AgentLoopRunner(workdir=tmp_path, client=llm_client)


@pytest.fixture
def judge(llm_client):
    """Provide an LLMJudge instance."""
    from tests.eval.lib.judge import LLMJudge
    model = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-4-20250514")
    return LLMJudge(client=llm_client, model=model)
