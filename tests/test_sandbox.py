"""test_sandbox.py — Tests for sandbox backends.

Tests are split:

1. Always-run — NoOpSandbox, factory, DockerSandbox path helpers (no Docker needed).
2. Docker-gated — Integration tests against a real Docker daemon. Skipped when
   Docker is not available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agents"))

from agents.sandbox import (  # noqa: E402
    DockerSandbox,
    NoOpSandbox,
    create_sandbox,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def has_docker() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


docker_required = pytest.mark.skipif(
    not has_docker(), reason="Docker not available"
)


# ===================================================================
# NoOpSandbox — always run
# ===================================================================

@pytest.fixture
def tmp_workdir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def noop(tmp_workdir: str) -> NoOpSandbox:
    return NoOpSandbox(tmp_workdir)


class TestNoOpSandboxExecute:
    """Shell command execution on host."""

    def test_returns_stdout(self, noop: NoOpSandbox) -> None:
        result = noop.execute("echo hello")
        assert "hello" in result

    def test_returns_stderr(self, noop: NoOpSandbox) -> None:
        result = noop.execute("echo error >&2")
        assert "error" in result

    def test_respects_cwd(self, tmp_workdir: str) -> None:
        sub = Path(tmp_workdir) / "subdir"
        sub.mkdir()
        s = NoOpSandbox(tmp_workdir)
        result = s.execute("pwd", cwd=str(sub))
        assert str(sub) in result or sub.name in result

    def test_timeout(self, noop: NoOpSandbox) -> None:
        result = noop.execute("python -c \"import time; time.sleep(5)\"", timeout=1)
        assert "Timeout" in result

    def test_empty_output(self, noop: NoOpSandbox) -> None:
        result = noop.execute("python -c \"\"")
        assert result == ""


class TestNoOpSandboxFile:
    """File operations on host."""

    def test_write_and_read(self, tmp_workdir: str) -> None:
        s = NoOpSandbox(tmp_workdir)
        path = str(Path(tmp_workdir) / "test.txt")
        s.write_file(path, "hello")
        assert "hello" == Path(path).read_text()

    def test_write_creates_parent_dirs(self, tmp_workdir: str) -> None:
        s = NoOpSandbox(tmp_workdir)
        path = str(Path(tmp_workdir) / "a" / "b" / "c.txt")
        result = s.write_file(path, "nested")
        assert "Wrote" in result
        assert Path(path).exists()
        assert Path(path).read_text() == "nested"

    def test_edit(self, tmp_workdir: str) -> None:
        s = NoOpSandbox(tmp_workdir)
        path = str(Path(tmp_workdir) / "edit.txt")
        Path(path).write_text("before after before")
        result = s.edit_file(path, "after", "AFTER")
        assert "Edited" in result
        assert Path(path).read_text() == "before AFTER before"

    def test_edit_text_not_found(self, tmp_workdir: str) -> None:
        s = NoOpSandbox(tmp_workdir)
        path = str(Path(tmp_workdir) / "nope.txt")
        Path(path).write_text("something")
        result = s.edit_file(path, "nonexistent", "replacement")
        assert "not found" in result.lower()

    def test_is_ready(self, noop: NoOpSandbox) -> None:
        assert noop.is_ready()

    def test_cleanup_noop(self, noop: NoOpSandbox) -> None:
        noop.cleanup()
        assert noop.is_ready()


# ===================================================================
# Factory
# ===================================================================

class TestFactory:
    def teardown_method(self) -> None:
        for key in (
            "AGENT_SANDBOX_BACKEND", "AGENT_SANDBOX_IMAGE",
            "AGENT_SANDBOX_MEMORY", "AGENT_SANDBOX_CPUS", "AGENT_SANDBOX_NETWORK",
        ):
            os.environ.pop(key, None)

    def test_default_is_docker_if_available(self) -> None:
        """When AGENT_SANDBOX_BACKEND is unset, Docker is used if available."""
        os.environ.pop("AGENT_SANDBOX_BACKEND", None)
        s = create_sandbox()
        if has_docker():
            assert isinstance(s, DockerSandbox)
            s.cleanup()
        else:
            assert isinstance(s, NoOpSandbox)

    def test_empty_uses_default(self) -> None:
        """Empty string behaves same as unset."""
        os.environ["AGENT_SANDBOX_BACKEND"] = ""
        s = create_sandbox()
        if has_docker():
            assert isinstance(s, DockerSandbox)
            s.cleanup()
        else:
            assert isinstance(s, NoOpSandbox)

    def test_none_disables_sandbox(self) -> None:
        os.environ["AGENT_SANDBOX_BACKEND"] = "none"
        assert isinstance(create_sandbox(), NoOpSandbox)

    def test_off_disables_sandbox(self) -> None:
        os.environ["AGENT_SANDBOX_BACKEND"] = "off"
        assert isinstance(create_sandbox(), NoOpSandbox)

    def test_unknown_uses_default(self) -> None:
        os.environ["AGENT_SANDBOX_BACKEND"] = "kubernetes"
        s = create_sandbox()
        if has_docker():
            assert isinstance(s, DockerSandbox)
            s.cleanup()
        else:
            assert isinstance(s, NoOpSandbox)

    def test_docker_explicit(self) -> None:
        os.environ["AGENT_SANDBOX_BACKEND"] = "docker"
        s = create_sandbox("/tmp/test")
        if has_docker():
            assert isinstance(s, DockerSandbox)
        else:
            # Falls back to NoOpSandbox with warning when Docker unavailable
            assert isinstance(s, NoOpSandbox)
        s.cleanup()


# ===================================================================
# DockerSandbox path helpers — always run
# ===================================================================

class TestDockerSandboxPathResolution:

    @pytest.fixture
    def win_sandbox(self) -> DockerSandbox:
        return DockerSandbox(workdir="C:\\Users\\test\\project")

    @pytest.fixture
    def unix_sandbox(self) -> DockerSandbox:
        return DockerSandbox(workdir="/home/user/project")

    def test_container_workdir_windows(self, win_sandbox: DockerSandbox) -> None:
        if os.name != "nt":
            pytest.skip("Windows-only")
        assert win_sandbox._container_workdir == "/workspace"

    def test_container_workdir_unix(self, unix_sandbox: DockerSandbox) -> None:
        if os.name == "nt":
            pytest.skip("Unix-only")
        assert unix_sandbox._container_workdir == "/home/user/project"

    def test_mount_windows(self, win_sandbox: DockerSandbox) -> None:
        if os.name != "nt":
            pytest.skip("Windows-only")
        assert win_sandbox._resolve_mount() == "/c/users/test/project:/workspace"

    def test_mount_unix(self, unix_sandbox: DockerSandbox) -> None:
        if os.name == "nt":
            pytest.skip("Unix-only")
        assert unix_sandbox._resolve_mount() == "/home/user/project:/home/user/project"

    def test_translate_path_windows(self, win_sandbox: DockerSandbox) -> None:
        if os.name != "nt":
            pytest.skip("Windows-only")
        result = win_sandbox._host_path_to_container("C:\\Users\\test\\project\\src\\main.py")
        assert result == "/workspace/src/main.py"

    def test_translate_path_windows_workdir(self, win_sandbox: DockerSandbox) -> None:
        if os.name != "nt":
            pytest.skip("Windows-only")
        result = win_sandbox._host_path_to_container("C:\\Users\\test\\project")
        assert result == "/workspace"

    def test_translate_path_windows_outside(self, win_sandbox: DockerSandbox) -> None:
        if os.name != "nt":
            pytest.skip("Windows-only")
        result = win_sandbox._host_path_to_container("C:\\Windows\\System32")
        assert result == "/workspace"

    def test_translate_path_unix(self, unix_sandbox: DockerSandbox) -> None:
        if os.name == "nt":
            pytest.skip("Unix-only")
        result = unix_sandbox._host_path_to_container("/home/user/project/src")
        assert result == "/home/user/project/src"


# ===================================================================
# DockerSandbox integration — require Docker daemon
# ===================================================================

@docker_required
class TestDockerSandboxIntegration:

    @pytest.fixture
    def docker_workdir(self) -> str:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "host_file.txt").write_text("hello from host")
            yield d

    @pytest.fixture
    def sandbox(self, docker_workdir: str) -> DockerSandbox:
        s = DockerSandbox(workdir=docker_workdir)
        yield s
        s.cleanup()

    # --- shell ---

    def test_execute_echo(self, sandbox: DockerSandbox) -> None:
        assert "sandboxed" in sandbox.execute("echo sandboxed")

    def test_timeout(self, sandbox: DockerSandbox) -> None:
        assert "Timeout" in sandbox.execute("sleep 10", timeout=1)

    # --- file ops ---

    def test_write_file(self, sandbox: DockerSandbox, docker_workdir: str) -> None:
        path = str(Path(docker_workdir) / "out.txt")
        result = sandbox.write_file(path, "docker-written")
        assert "Wrote" in result
        assert Path(path).read_text() == "docker-written"

    def test_write_creates_dirs(self, sandbox: DockerSandbox, docker_workdir: str) -> None:
        path = str(Path(docker_workdir) / "deep" / "nested" / "file.txt")
        sandbox.write_file(path, "deep")
        assert Path(path).read_text() == "deep"

    def test_edit_file(self, sandbox: DockerSandbox, docker_workdir: str) -> None:
        path = str(Path(docker_workdir) / "before.txt")
        Path(path).write_text("alpha beta gamma")
        sandbox.edit_file(path, "beta", "BETA")
        assert Path(path).read_text() == "alpha BETA gamma"

    def test_edit_not_found(self, sandbox: DockerSandbox, docker_workdir: str) -> None:
        path = str(Path(docker_workdir) / "x.txt")
        Path(path).write_text("abc")
        result = sandbox.edit_file(path, "zzz", "yyy")
        assert "not found" in result.lower()

    # --- lifecycle ---

    def test_is_ready(self, sandbox: DockerSandbox) -> None:
        assert not sandbox.is_ready()
        sandbox.ensure_running()
        assert sandbox.is_ready()

    def test_cleanup_idempotent(self, sandbox: DockerSandbox) -> None:
        sandbox.cleanup()
        sandbox.cleanup()
