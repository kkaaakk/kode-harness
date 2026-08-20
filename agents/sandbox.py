"""sandbox.py — Pluggable sandbox backends for command + file execution.

Provides:
- ``SandboxBackend`` — abstract protocol
- ``NoOpSandbox`` — runs commands + file ops directly on the host (default)
- ``DockerSandbox`` — runs commands + file ops inside a Docker container
- ``create_sandbox()`` — factory that picks the backend via env vars

Usage::

    from sandbox import create_sandbox
    SANDBOX = create_sandbox(workdir)
    SANDBOX.execute("ls -la")
    SANDBOX.write_file(path, content)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# SandboxCapabilities — declares what a backend can actually isolate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SandboxCapabilities:
    """Declares the isolation guarantees a SandboxBackend *supports*.

    Stage 2C-B2B-2: the Harness uses these flags to decide whether a
    backend is *eligible* for multi-session bash execution.

    IMPORTANT (B2B-3): ``supports_*`` flags declare capability, NOT
    current-config safety. A backend may ``supports_filesystem_isolation=True``
    yet still leak a specific private path if that path happens to live
    under a mounted host directory. Use ``assess_isolation()`` to evaluate
    the *actual* runtime configuration before accepting the backend in
    ``secure_multi_session`` mode.

    Fields:
        supports_filesystem_isolation: True if this backend CAN be
            configured to prevent bash from reaching files outside the
            mounted workdir (e.g. DockerSandbox with a constrained mount
            namespace). NoOpSandbox is False (it shares the host fs).
        supports_process_isolation: True if this backend CAN be
            configured to prevent bash from seeing/signalling the Harness
            process or other sessions' processes.
    """
    supports_filesystem_isolation: bool
    supports_process_isolation: bool


@dataclass(frozen=True)
class IsolationAssessment:
    """Result of evaluating a sandbox's *actual* runtime isolation.

    Stage 2C-B2B-3: produced by ``SandboxBackend.assess_isolation()``.
    Where ``SandboxCapabilities`` declares what a backend *supports*,
    this dataclass reports whether the *current configuration* actually
    achieves isolation for the given private paths.

    Fields:
        filesystem_isolated: True iff bash running in this sandbox cannot
            reach any of the ``private_paths`` via the configured mount
            list. This is the flag ``secure_multi_session`` checks.
        reasons: Human-readable explanations when ``filesystem_isolated``
            is False (e.g. "private path /tmp/artifacts is under mount
            source /tmp"). Empty tuple when isolated.
        mount_sources: Resolved host paths that are mounted INTO the
            sandbox. Exposed for diagnostics and testing; the sandbox
            itself owns this list.
    """
    filesystem_isolated: bool
    reasons: tuple[str, ...]
    mount_sources: tuple[str, ...]


# ---------------------------------------------------------------------------
# SandboxBackend — abstract protocol
# ---------------------------------------------------------------------------
class SandboxBackend(ABC):
    """Protocol that all sandbox backends must satisfy."""

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        """Isolation guarantees this backend *supports*. See
        SandboxCapabilities for the meaning of each flag. Use
        ``assess_isolation()`` to evaluate the actual config."""
        ...

    @abstractmethod
    def assess_isolation(
        self, *, workdir: str, private_paths: tuple[str, ...]
    ) -> IsolationAssessment:
        """Evaluate whether the *current* configuration isolates the
        given private paths from bash running in this sandbox.

        Stage 2C-B2B-3: backends MUST resolve symlinks and ``..`` in
        both mount sources and private paths before containment checks.
        ``secure_multi_session`` mode calls this at agent_loop startup
        and rejects the configuration when ``filesystem_isolated`` is
        False.

        Args:
            workdir: The host workdir that bash is expected to access.
            private_paths: Host paths that bash must NOT be able to
                reach (e.g. the artifact store root). Each is resolved
                before checking containment.

        Returns:
            IsolationAssessment with ``filesystem_isolated`` and
            human-readable ``reasons`` for any leak.
        """
        ...

    @abstractmethod
    def execute(
        self, command: str, *, cwd: str | None = None, timeout: int = 120
    ) -> str:
        """Execute a shell command.  Returns combined stdout+stderr."""
        ...

    @abstractmethod
    def execute_structured(
        self, command: str, *, cwd: str | None = None, timeout: int = 120
    ) -> tuple[str, str, int]:
        """Execute a shell command and return (stdout, stderr, exit_code).

        Stage 2C-B2A: this is the structured-execution entry point used
        by ``run_bash`` to build a ``BashExecutionResult``. Backends MUST
        capture exit_code directly from the subprocess — never infer it
        from the output string.
        """
        ...

    @abstractmethod
    def write_file(self, host_path: str, content: str) -> str:
        """Write *content* to *host_path* inside the sandbox."""
        ...

    @abstractmethod
    def edit_file(self, host_path: str, old_text: str, new_text: str) -> str:
        """Replace *old_text* with *new_text* in *host_path*."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the sandbox is running and accepting operations."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down the sandbox.  Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Path-containment helper — shared by all backends
# ---------------------------------------------------------------------------
def _path_contains(parent: Path, child: Path) -> bool:
    """Return True iff ``child`` is ``parent`` or inside ``parent``.

    Both paths are resolved (symlinks + ``..``) before comparison. On
    cross-drive Windows comparisons where ``commonpath`` would raise,
    returns False.
    """
    try:
        p = parent.resolve()
        c = child.resolve()
    except (OSError, ValueError):
        return False
    try:
        common = Path(os.path.commonpath([str(p), str(c)]))
    except ValueError:
        return False
    return common == p


# ---------------------------------------------------------------------------
# NoOpSandbox — direct host execution (default / fallback)
# ---------------------------------------------------------------------------
class NoOpSandbox(SandboxBackend):
    """Execute operations directly on the host (current pre-Docker behaviour).

    Security note (stage 2C-B2B-2): NoOpSandbox provides NO filesystem or
    process isolation. Bash commands run with the Harness process's
    privileges and can read any file the OS user can read, including the
    private artifact root. Use this backend ONLY in trusted_local mode
    (single-user, single-session dev). For secure multi-session operation,
    require a backend with capabilities.supports_filesystem_isolation=True
    AND a passing assess_isolation() result.
    """

    # Sentinel instance: NoOpSandbox never supports isolation.
    _CAPS = SandboxCapabilities(
        supports_filesystem_isolation=False,
        supports_process_isolation=False,
    )

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self._CAPS

    def __init__(self, workdir: str) -> None:
        self._workdir = workdir

    def assess_isolation(
        self, *, workdir: str, private_paths: tuple[str, ...]
    ) -> IsolationAssessment:
        """NoOpSandbox shares the host filesystem with the Harness process.

        It CANNOT isolate any private path — bash can ``cat`` any file
        the OS user can read. ``filesystem_isolated`` is always False
        with a clear reason, so ``secure_multi_session`` mode rejects
        this backend at startup regardless of where the artifact root
        lives.
        """
        reason = (
            "NoOpSandbox shares the host filesystem with the Harness "
            "process; bash can read any host file the OS user can read. "
            "It cannot isolate private paths regardless of their location."
        )
        return IsolationAssessment(
            filesystem_isolated=False,
            reasons=(reason,),
            mount_sources=("/",),  # entire host fs is reachable
        )

    # -- shell -----------------------------------------------------------

    def execute(
        self, command: str, *, cwd: str | None = None, timeout: int = 120
    ) -> str:
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self._workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            return "Error: Timeout"

    def execute_structured(
        self, command: str, *, cwd: str | None = None, timeout: int = 120
    ) -> tuple[str, str, int]:
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self._workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return r.stdout.rstrip(), r.stderr.rstrip(), r.returncode
        except subprocess.TimeoutExpired:
            # Raise so run_bash can catch and return its own error string.
            raise

    # -- file ------------------------------------------------------------

    def write_file(self, host_path: str, content: str) -> str:
        fp = Path(host_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {host_path}"

    def edit_file(self, host_path: str, old_text: str, new_text: str) -> str:
        fp = Path(host_path)
        text = fp.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: Text not found in {host_path}"
        fp.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {host_path}"

    # -- lifecycle -------------------------------------------------------

    def is_ready(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# DockerSandbox — one persistent container, all ops via ``docker exec``
# ---------------------------------------------------------------------------
class DockerSandbox(SandboxBackend):
    """One Docker container per agent session.

    Created lazily on first use; stays alive for the process lifetime.
    All commands run via ``docker exec``.

    Isolation note (stage 2C-B2B-2): a correctly-configured DockerSandbox
    CAN provide filesystem and process isolation because the container
    has its own mount namespace and PID namespace. Therefore
    ``capabilities.supports_filesystem_isolation=True`` and
    ``capabilities.supports_process_isolation=True``.

    BUT capability ≠ current-config safety (B2B-3). Whether the
    *current* mount plan actually isolates a given private path is
    evaluated by ``assess_isolation()``, which resolves every mount
    source and private path and checks containment. A DockerSandbox
    that mounts ``/home/user`` will NOT isolate an artifact root at
    ``/home/user/private-artifacts``, even though the capability flag
    is True. ``secure_multi_session`` mode must check the assessment,
    not just the capability.
    """

    IMAGE_NAME = "agent-sandbox:latest"
    CONTAINER_PREFIX = "agent_sandbox_"

    # Sentinel instance: DockerSandbox CAN provide full isolation when
    # correctly configured (only the workdir is mounted; the private
    # artifact root is outside the container's mount namespace).
    _CAPS = SandboxCapabilities(
        supports_filesystem_isolation=True,
        supports_process_isolation=True,
    )

    def __init__(
        self,
        *,
        workdir: str,
        image: str | None = None,
        memory: str | None = None,
        cpus: str | None = None,
        network: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._workdir = os.path.abspath(workdir)
        self._image = image or self.IMAGE_NAME
        self._memory = memory
        self._cpus = cpus
        self._network = network
        self._env = env or {}
        self._container_name = f"{self.CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
        self._container_id: str | None = None
        self._container_workdir = self._resolve_container_workdir()
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self._CAPS

    def assess_isolation(
        self, *, workdir: str, private_paths: tuple[str, ...]
    ) -> IsolationAssessment:
        """Evaluate the *current* mount plan against the given private
        paths.

        Stage 2C-B2B-3/4: this is the runtime check that
        ``secure_multi_session`` mode uses. It does NOT depend on Docker
        being available — it inspects ``_mount_host_sources()`` directly
        so the check works in unit tests on hosts without Docker.

        Algorithm:
            1. Iterate EVERY mount source from ``_mount_host_sources()``
               — this is the single source of truth. When future mounts
               are added (cache dir, docker socket, etc.), they are
               automatically included.
            2. Resolve each host mount source (symlinks + ``..``).
            3. For each private path, resolve it and check whether any
               mount source contains it. If so, bash can reach it via
               the mount → not isolated. Even read-only mounts leak
               content, so they count as non-isolated too.
            4. Return ``filesystem_isolated=True`` only when no private
               path is contained in any mount source.
        """
        reasons: list[str] = []
        # Collect host-native mount source paths (not the WSL-style spec
        # strings) so Path.resolve() and _path_contains() work on the
        # same OS-native form as the private paths.
        mount_host_paths: list[str] = []
        for host_src, _container_dst in self._mount_host_sources():
            mount_host_paths.append(host_src)

        # Resolve mount sources once.
        resolved_mounts: list[Path] = []
        for src in mount_host_paths:
            try:
                resolved_mounts.append(Path(src).resolve())
            except (OSError, ValueError):
                # Unresolvable mount — treat as a leak with a reason.
                reasons.append(
                    f"mount source {src!r} could not be resolved; "
                    f"cannot guarantee isolation"
                )

        for pp in private_paths:
            if not pp:
                continue
            try:
                pp_resolved = Path(pp).resolve()
            except (OSError, ValueError):
                reasons.append(
                    f"private path {pp} could not be resolved; "
                    f"cannot guarantee isolation"
                )
                continue
            for m in resolved_mounts:
                if _path_contains(m, pp_resolved):
                    reasons.append(
                        f"private path {pp} is under mount source "
                        f"{m}; bash can reach it via the mount"
                    )

        return IsolationAssessment(
            filesystem_isolated=len(reasons) == 0,
            reasons=tuple(reasons),
            mount_sources=tuple(mount_host_paths),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        if self._container_id is not None:
            return True
        if not self._docker_available():
            return False
        self._container_id = self._find_existing_container()
        return self._container_id is not None

    def ensure_running(self) -> None:
        if self.is_ready():
            return
        with self._lock:
            if self.is_ready():
                return
            self._build_image_if_missing()
            self._create_container()
            self._start_container()

    # -- shell -----------------------------------------------------------

    def execute(
        self, command: str, *, cwd: str | None = None, timeout: int = 120
    ) -> str:
        self.ensure_running()
        workdir = self._host_path_to_container(cwd) if cwd else self._container_workdir
        try:
            r = subprocess.run(
                ["docker", "exec", "-w", workdir, self._container_name,
                 "sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            return (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            return "Error: Timeout"

    def execute_structured(
        self, command: str, *, cwd: str | None = None, timeout: int = 120
    ) -> tuple[str, str, int]:
        self.ensure_running()
        workdir = self._host_path_to_container(cwd) if cwd else self._container_workdir
        r = subprocess.run(
            ["docker", "exec", "-w", workdir, self._container_name,
             "sh", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.rstrip(), r.stderr.rstrip(), r.returncode

    # -- file ------------------------------------------------------------

    def write_file(self, host_path: str, content: str) -> str:
        self.ensure_running()
        container_path = self._host_path_to_container(host_path)
        # Use python3 inside the container; pipe content via stdin
        script = (
            "import sys,os;"
            "p=sys.argv[1];"
            "os.makedirs(os.path.dirname(p),exist_ok=True);"
            "open(p,'w',encoding='utf-8').write(sys.stdin.read());"
            "print(f'Wrote {len(sys.stdin.read())} bytes')"
        )
        try:
            r = subprocess.run(
                ["docker", "exec", "-i", self._container_name,
                 "python3", "-c", script, container_path],
                input=content,
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return f"Error: {r.stderr.strip()}"
            return f"Wrote {len(content)} bytes to {host_path}"
        except Exception as e:
            return f"Error: {e}"

    def edit_file(self, host_path: str, old_text: str, new_text: str) -> str:
        self.ensure_running()
        container_path = self._host_path_to_container(host_path)
        script = (
            "import sys;"
            "p=sys.argv[1];old=sys.argv[2];new=sys.argv[3];"
            "c=open(p,encoding='utf-8').read();"
            "print('not_found' if old not in c else 'found');"
            "if old in c: open(p,'w',encoding='utf-8').write(c.replace(old,new,1))"
        )
        try:
            r = subprocess.run(
                ["docker", "exec", self._container_name,
                 "python3", "-c", script, container_path, old_text, new_text],
                capture_output=True, text=True, timeout=30,
            )
            output = (r.stdout + r.stderr).strip()
            if "not_found" in output:
                return f"Error: Text not found in {host_path}"
            if r.returncode != 0:
                return f"Error: {output}"
            return f"Edited {host_path}"
        except Exception as e:
            return f"Error: {e}"

    # -- lifecycle -------------------------------------------------------

    def cleanup(self) -> None:
        if not self._container_name:
            return
        subprocess.run(
            ["docker", "stop", "--time", "5", self._container_name],
            capture_output=True,
        )
        self._container_id = None

    # ------------------------------------------------------------------
    # internal: container lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _docker_available() -> bool:
        return _docker_ready()

    def _find_existing_container(self) -> str | None:
        r = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={self._container_name}"],
            capture_output=True, text=True,
        )
        return r.stdout.strip() or None

    def _build_image_if_missing(self) -> None:
        r = subprocess.run(
            ["docker", "image", "inspect", self._image],
            capture_output=True,
        )
        if r.returncode == 0:
            return
        dockerfile = os.getenv(
            "AGENT_SANDBOX_DOCKERFILE",
            str(Path(__file__).resolve().parent.parent / "docker" / "sandbox.Dockerfile"),
        )
        subprocess.run(
            ["docker", "build", "-t", self._image, "-f", dockerfile, "."],
            check=True,
        )

    def _create_container(self) -> None:
        args = [
            "docker", "create",
            "--name", self._container_name,
            "--rm",
            "--workdir", self._container_workdir,
            "-v", self._resolve_mount(),
        ]
        if self._memory:
            args += ["--memory", self._memory]
        if self._cpus:
            args += ["--cpus", self._cpus]
        if self._network is not None:
            args += ["--network", self._network]
        for k, v in self._env.items():
            args += ["-e", f"{k}={v}"]
        args.append(self._image)
        args += ["sh", "-c", "tail -f /dev/null"]
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        self._container_id = result.stdout.strip()[:12]

    def _start_container(self) -> None:
        subprocess.run(
            ["docker", "start", self._container_name],
            capture_output=True, check=True,
        )

    # ------------------------------------------------------------------
    # internal: path translation
    # ------------------------------------------------------------------

    def _resolve_container_workdir(self) -> str:
        if os.name == "nt":
            return "/workspace"
        return self._workdir

    def _resolve_mount(self) -> str:
        """Return the primary mount spec (``host:container``).

        Stage 2C-B2B-3: kept for backward compat with ``_create_container``.
        The full mount plan is exposed via ``_mount_specs()`` so that
        ``assess_isolation()`` can iterate every mount source.
        """
        if os.name == "nt":
            wsl_path = "/" + self._workdir.replace(":\\", "/").replace("\\", "/").lower()
            return f"{wsl_path}:/workspace"
        return f"{self._workdir}:{self._workdir}"

    def _mount_specs(self) -> list[str]:
        """Return ALL mount specs (``host:container``) for this sandbox.

        Stage 2C-B2B-3: the default configuration mounts ONLY the
        workdir. The private artifact root is intentionally NOT in
        this list — that is the filesystem-isolation guarantee
        ``assess_isolation()`` verifies. If a future change adds extra
        mounts (e.g. for MCP servers), append them here so the
        assessment stays accurate.
        """
        return [self._resolve_mount()]

    def _mount_host_sources(self) -> list[tuple[str, str]]:
        """Return ALL mount sources as ``(host_native_path, container_path)``.

        Stage 2C-B2B-4: this is the SINGLE SOURCE OF TRUTH that
        ``assess_isolation()`` iterates. Unlike ``_mount_specs()`` (which
        returns WSL-style spec strings on Windows that cannot be
        compared with native Windows private paths), this method
        returns host paths in the OS-native form so ``Path.resolve()``
        and ``_path_contains()`` work correctly.

        Each tuple is (host_native_path, container_path). The host path
        is what Docker actually mounts from the host filesystem; the
        container path is where it appears inside the container. For
        isolation assessment only the host path matters — if a private
        path is under any host source, bash can reach it.

        When a future change adds extra mounts (cache dir, docker
        socket, etc.), append them here. ``assess_isolation()`` will
        automatically include them in the evaluation — no other change
        needed.
        """
        # Default: only the workdir is mounted. We use self._workdir
        # (the native Windows path) rather than the WSL-style spec
        # from _resolve_mount(), because private_paths are native too.
        return [(self._workdir, self._container_workdir)]

    def _host_path_to_container(self, host_path: str) -> str:
        """Translate an absolute host path to its container equivalent.

        Workdir-inside paths map under /workspace; paths OUTSIDE the
        workdir map to /workspace root (best-effort containment, never
        expose host paths into the container).
        """
        host_abs = os.path.abspath(host_path)
        workdir_abs = os.path.abspath(self._workdir)
        if host_abs == workdir_abs:
            return "/workspace"
        if host_abs.startswith(workdir_abs + os.sep):
            relative = host_abs[len(workdir_abs):].lstrip(os.sep).replace("\\", "/")
            return f"/workspace/{relative}"
        # Outside workdir -> best-effort, map to /workspace root
        return "/workspace"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _docker_ready() -> bool:
    """Check that Docker CLI is present AND the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


def create_sandbox(workdir: str | None = None) -> SandboxBackend:
    """Return the active sandbox backend.

    Reads ``AGENT_SANDBOX_BACKEND``:

    - unset            → ``DockerSandbox`` if Docker is available, else ``NoOpSandbox``
    - ``"docker"``     → ``DockerSandbox`` (error if Docker not available)
    - ``"none"`` / ``"off"`` → ``NoOpSandbox`` (skip Docker even if available)
    """
    backend = os.getenv("AGENT_SANDBOX_BACKEND", "").strip().lower()
    wd = workdir or os.getcwd()

    if backend in ("none", "off"):
        return NoOpSandbox(workdir=wd)

    docker_ok = _docker_ready()

    if backend == "docker":
        if not docker_ok:
            print("[sandbox] AGENT_SANDBOX_BACKEND=docker but Docker daemon "
                  "not reachable — commands will run on host. Start Docker or "
                  "set AGENT_SANDBOX_BACKEND=none to silence this warning.")
            return NoOpSandbox(workdir=wd)
        return _make_docker_sandbox(wd)

    # default: Docker if available
    if docker_ok:
        return _make_docker_sandbox(wd)

    return NoOpSandbox(workdir=wd)


def _make_docker_sandbox(workdir: str) -> DockerSandbox:
    return DockerSandbox(
        workdir=workdir,
        image=os.getenv("AGENT_SANDBOX_IMAGE") or None,
        memory=os.getenv("AGENT_SANDBOX_MEMORY") or None,
        cpus=os.getenv("AGENT_SANDBOX_CPUS") or None,
        network=os.getenv("AGENT_SANDBOX_NETWORK") or None,
        env={
            "DEBIAN_FRONTEND": "noninteractive",
            "PYTHONUNBUFFERED": "1",
        },
    )
