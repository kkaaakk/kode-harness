from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``docker`` marker used by the real-container
    integration tests in ``tests/integration/``. Without this, pytest
    emits a warning on every collection."""
    config.addinivalue_line(
        "markers",
        "docker: marks tests that require a real Docker daemon "
        "(skipped unless Docker is available; select with '-m docker')",
    )
