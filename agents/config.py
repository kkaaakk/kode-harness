"""
config.py - Global configuration, environment setup, and shared constants.

This module is the single source of truth for all configuration values.
It has NO internal agents/ dependencies (only stdlib + external packages).
"""

import atexit
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

try:
    from agents.sandbox import create_sandbox
except ImportError:
    from sandbox import create_sandbox  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKDIR = Path.cwd()

RUNTIME_DIR = WORKDIR / ".tmp" / "runtime"
TEAM_DIR = RUNTIME_DIR / "team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = RUNTIME_DIR / "tasks"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = RUNTIME_DIR / "transcripts"

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# ---------------------------------------------------------------------------
# Sandbox (initialised once, cleaned up at exit)
# ---------------------------------------------------------------------------
SANDBOX = create_sandbox(str(WORKDIR))
atexit.register(SANDBOX.cleanup)

# ---------------------------------------------------------------------------
# Run mode — controls whether bash requires a filesystem-isolated sandbox.
# ---------------------------------------------------------------------------
# Stage 2C-B2B-2: declares the trust level of the environment.
#
#   trusted_local        — single-user, single-session dev. NoOpSandbox is
#                          allowed. Bash runs on the host with the Harness
#                          user's privileges. No cross-session filesystem
#                          isolation is provided; do not run untrusted
#                          sessions concurrently.
#
#   secure_multi_session — multi-session / multi-tenant. Bash REQUIRES a
#                          backend with capabilities.filesystem_isolation=True
#                          (e.g. DockerSandbox). NoOpSandbox is rejected at
#                          run_bash entry; the model sees an Error: string.
#                          This is the only mode that can safely host
#                          concurrent sessions with artifact isolation.
#
# Configured via AGENT_RUN_MODE env var. Defaults to trusted_local so that
# existing single-user dev workflows keep working unchanged.
_RUN_MODE_ENV = os.getenv("AGENT_RUN_MODE", "trusted_local").strip().lower()
if _RUN_MODE_ENV not in ("trusted_local", "secure_multi_session"):
    raise ValueError(
        f"AGENT_RUN_MODE must be 'trusted_local' or 'secure_multi_session', "
        f"got: {_RUN_MODE_ENV!r}"
    )
RUN_MODE = _RUN_MODE_ENV

# ---------------------------------------------------------------------------
# Public API (for ``from agents.config import *``)
# ---------------------------------------------------------------------------
__all__ = [
    "WORKDIR",
    "RUNTIME_DIR",
    "TEAM_DIR",
    "INBOX_DIR",
    "TASKS_DIR",
    "SKILLS_DIR",
    "TRANSCRIPT_DIR",
    "client",
    "MODEL",
    "TOKEN_THRESHOLD",
    "POLL_INTERVAL",
    "IDLE_TIMEOUT",
    "VALID_MSG_TYPES",
    "SANDBOX",
    "RUN_MODE",
]
