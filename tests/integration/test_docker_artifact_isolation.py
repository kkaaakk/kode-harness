"""test_docker_artifact_isolation.py - Real-Docker integration tests
for Stage 2C-B2B-4.1.

These tests verify that a REAL DockerSandbox container cannot read,
enumerate, or symlink its way to the host artifact root. They are the
empirical counterpart to the unit-level ``assess_isolation()`` tests
in ``test_harness_output_policy_wiring.py``, which only prove that the
mount PLAN excludes the artifact root — not that the running container
actually enforces it.

Why a separate file, why pytest, why skipif:
  - Unit tests must run on every platform (Windows, macOS, Linux) without
    Docker. Keeping these assertions in the unit suite would either skip
    noisily on every non-Linux dev machine or, worse, get deleted.
  - This file is pytest-only (the rest of the suite is unittest) so the
    ``@pytest.mark.docker`` marker can be registered in ``conftest.py``
    and selected/deselected explicitly in CI:
        pytest tests/integration/test_docker_artifact_isolation.py -m docker
  - ``skipif`` makes the file safe to collect in any environment. The
    skip is silent-by-default; CI jobs that want to enforce the contract
    set ``AGENT_DOCKER_IT=1`` (or rely on ``-m docker`` selecting them).

What the tests assert (the 5 vectors from B2B-4.1):
  1. ``cat <artifact_path>`` inside the container fails (no read).
  2. ``python -c "open(<artifact_path>).read()"`` inside the container
     raises (no Python-level bypass).
  3. ``find /workspace -name <secret_marker>`` returns nothing (no
     enumeration of the artifact root via the mounted workdir).
  4. A symlink inside WORKDIR pointing to ``<artifact_root>`` cannot
     be followed (no symlink escape).
  5. The actual Docker Mount set matches what ``_mount_host_sources()``
     declared — i.e., the unit-test fixture and the real container agree.

These tests MUST NOT be replaced by ordinary unit tests. If Docker is
unavailable, they skip. They turn green only on a Linux + Docker CI job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------
def _docker_available() -> bool:
    """Return True iff ``docker run hello-world`` would plausibly work.

    We check the binary first (fast) then probe the daemon. Both must
    succeed for the integration tests to run.
    """
    if shutil.which("docker") is None:
        return False
    if os.environ.get("AGENT_DOCKER_IT") == "0":
        # Explicit opt-out — useful for dev machines that have Docker
        # installed but where the daemon is stopped.
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


DOCKER_AVAILABLE = _docker_available()

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        not DOCKER_AVAILABLE,
        reason=(
            "Docker is unavailable — real-container artifact isolation "
            "tests are skipped. Run on a Linux + Docker CI job to "
            "enforce the contract."
        ),
    ),
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Docker integration tests run on Linux CI only.",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def docker_sandbox(tmp_path: Path):
    """Build a real DockerSandbox against a temp workdir.

    Imported lazily so collection on Docker-less machines does not fail
    on missing optional deps.
    """
    # Ensure agents.* is importable from the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from agents.sandbox import DockerSandbox  # noqa: WPS433

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    sandbox = DockerSandbox(workdir=str(workdir))
    yield sandbox


@pytest.fixture
def artifact_root_outside_workdir(tmp_path: Path):
    """A private artifact root that lives OUTSIDE the sandbox workdir,
    simulating the B2B-4 AGENT_ARTIFACT_ROOT layout."""
    artifact_root = tmp_path / "private-artifacts"
    artifact_root.mkdir(mode=0o700)
    # Drop a unique secret marker so we can grep for it.
    secret_file = artifact_root / "secret.txt"
    secret_file.write_text("DOCKER_IT_SECRET_TOKEN_42\n")
    yield artifact_root
    # Cleanup: shutil.rmtree ignores errors so a permission glitch in
    # one test does not cascade.
    shutil.rmtree(artifact_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Vector 1: cat inside the container must fail
# ---------------------------------------------------------------------------
def test_container_cannot_cat_artifact(
    docker_sandbox, artifact_root_outside_workdir
):
    """``cat <artifact_root>/secret.txt`` inside the container must fail
    with a No such file / Permission denied style error. The secret
    token MUST NOT appear in stdout or stderr."""
    secret_path = artifact_root_outside_workdir / "secret.txt"
    # ``execute_structured`` returns (stdout, stderr, exit_code).
    stdout, stderr, exit_code = docker_sandbox.execute_structured(
        f"cat '{secret_path}'"
    )
    assert exit_code != 0, (
        f"container cat succeeded on host artifact root — isolation broken.\n"
        f"stdout={stdout!r}\nstderr={stderr!r}"
    )
    combined = (stdout or "") + (stderr or "")
    assert "DOCKER_IT_SECRET_TOKEN_42" not in combined


# ---------------------------------------------------------------------------
# Vector 2: Python file-read inside the container must raise
# ---------------------------------------------------------------------------
def test_container_cannot_python_open_artifact(
    docker_sandbox, artifact_root_outside_workdir
):
    """Python-level read must also fail. This blocks the bypass where
    the sandbox blocks ``cat`` but a Python one-liner reads the file."""
    secret_path = artifact_root_outside_workdir / "secret.txt"
    # Use a Python snippet that prints READ_OK on success.
    cmd = (
        f'python3 -c "'
        f"import sys; "
        f"try: print('READ_OK:' + open('{secret_path}').read().strip())"
        f"except Exception as e: print('FAIL:' + str(e))"
        f'"'
    )
    stdout, stderr, exit_code = docker_sandbox.execute_structured(cmd)
    combined = (stdout or "") + (stderr or "")
    assert "READ_OK" not in combined, (
        f"Python open() succeeded inside the container — isolation broken.\n"
        f"stdout={stdout!r}\nstderr={stderr!r}"
    )
    assert "DOCKER_IT_SECRET_TOKEN_42" not in combined


# ---------------------------------------------------------------------------
# Vector 3: find / grep cannot enumerate the artifact root from the mount
# ---------------------------------------------------------------------------
def test_container_cannot_enumerate_artifact_via_workdir(
    docker_sandbox, artifact_root_outside_workdir
):
    """``find /workspace -name secret.txt`` must return nothing. The
    artifact root is NOT under the mounted workdir, so the secret file
    must not be reachable from the mount point."""
    stdout, stderr, exit_code = docker_sandbox.execute_structured(
        "find /workspace -name 'secret.txt' 2>/dev/null"
    )
    assert exit_code == 0, f"find failed unexpectedly: {stderr!r}"
    assert "secret.txt" not in (stdout or ""), (
        f"find reached the artifact root via the mount — isolation broken.\n"
        f"stdout={stdout!r}"
    )
    # Also grep for the secret token across the mounted tree.
    stdout2, _, _ = docker_sandbox.execute_structured(
        "grep -r DOCKER_IT_SECRET_TOKEN_42 /workspace 2>/dev/null || true"
    )
    assert "DOCKER_IT_SECRET_TOKEN_42" not in (stdout2 or "")


# ---------------------------------------------------------------------------
# Vector 4: symlink escape from WORKDIR to artifact root must fail
# ---------------------------------------------------------------------------
def test_container_cannot_symlink_to_artifact(
    docker_sandbox, artifact_root_outside_workdir
):
    """A symlink inside WORKDIR pointing at the host artifact root must
    NOT let the container read the secret. Even though symlink creation
    may succeed, following it must fail because the target is outside
    the mount set."""
    secret_path = artifact_root_outside_workdir / "secret.txt"
    # Create the symlink and try to read through it in one command.
    cmd = (
        f"ln -s '{artifact_root_outside_workdir}' /workspace/escape_link "
        f"&& cat /workspace/escape_link/secret.txt"
    )
    stdout, stderr, exit_code = docker_sandbox.execute_structured(cmd)
    combined = (stdout or "") + (stderr or "")
    assert "DOCKER_IT_SECRET_TOKEN_42" not in combined, (
        f"symlink escape succeeded — isolation broken.\n"
        f"stdout={stdout!r}\nstderr={stderr!r}"
    )


# ---------------------------------------------------------------------------
# Vector 5: actual Mount set matches _mount_host_sources()
# ---------------------------------------------------------------------------
def test_real_mount_set_matches_declared_plan(docker_sandbox, tmp_path: Path):
    """The Docker container's actual mount set must match what
    ``DockerSandbox._mount_host_sources()`` declares. This is the
    empirical check that the unit-level assess_isolation() tests are
    reasoning about the SAME mounts the runtime uses — no hidden mounts
    leaking the artifact root."""
    # Inspect the container's /proc/mounts (Linux only).
    stdout, stderr, exit_code = docker_sandbox.execute_structured(
        "cat /proc/mounts"
    )
    assert exit_code == 0, f"could not read /proc/mounts: {stderr!r}"

    declared = docker_sandbox._mount_host_sources()
    declared_host_paths = {
        Path(host_native).resolve()
        for host_native, _container in declared
    }

    # Every host path that Docker actually bind-mounted must be in the
    # declared set. We filter to bind mounts only (the ``rw``/``ro``
    # entries with a host source).
    leaked = []
    for line in (stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        source, _target, fstype, opts = parts[0], parts[1], parts[2], parts[3]
        if fstype not in {"none", "bind"}:
            continue
        # Only consider sources that look like absolute host paths
        # (skip tmpfs, proc, sysfs, cgroup, overlay, etc.).
        if not source.startswith("/"):
            continue
        if any(
            source.startswith(p) for p in (
                "/proc", "/sys", "/dev", "/etc/", "/var/lib/docker",
            )
        ):
            continue
        try:
            resolved_source = Path(source).resolve()
        except OSError:
            continue
        if resolved_source not in declared_host_paths:
            leaked.append(source)

    assert not leaked, (
        f"container has bind mounts not declared in _mount_host_sources(): "
        f"{leaked}. The unit-level assess_isolation() tests are reasoning "
        f"about an incomplete mount set."
    )
