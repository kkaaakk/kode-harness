"""artifacts.py - ArtifactStore for large tool outputs.

Stage 2C-A: infrastructure only, not yet wired into harness_core.

Purpose
-------
When a tool (read_file, bash, grep_search) produces output that is too large
to inline into the model context, the full output is written to an artifact
file on disk. The model receives a structured preview + an artifact_path
that it can read later via read_file if needed.

Security rules
--------------
1. Artifact filenames are server-generated (uuid4 hex). Tool outputs and user
   parameters NEVER control the physical filename.
2. Artifacts are stored under ``<root>/<session_id>/<artifact_id>.txt``.
   Path traversal (``../``) in session_id or artifact_id is rejected.
3. The path returned to the model is a LOGICAL path
   (``.harness/artifacts/<session>/<id>.txt``), never the host absolute path.
4. Writes are atomic: write to temp file -> fsync -> rename.
5. On write failure, the store raises ArtifactWriteError; the caller
   (ToolOutputPolicy) catches it and degrades to safe truncation.

Lifecycle
---------
- cleanup_expired(): remove artifacts past their expires_at.
- delete_session_artifacts(session_id): remove all artifacts for a session.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


class ArtifactError(Exception):
    """Base error for artifact store failures."""


class ArtifactWriteError(ArtifactError):
    """Failed to write artifact to disk."""


class PathTraversalError(ArtifactError):
    """session_id or artifact_id contains path traversal characters."""


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a stored artifact.

    Fields:
        artifact_id: Server-generated unique ID (uuid4 hex).
        artifact_uri: Virtual URI the model uses with read_file
                      (``artifact://<artifact_id>``). The model NEVER sees
                      the host filesystem path, so bash/grep/glob cannot
                      access other sessions' artifacts via shell.
        artifact_path: (Deprecated, kept for backward compat) logical path.
                       New code should use artifact_uri.
        total_bytes: Size of the stored content in bytes.
        total_lines: Number of lines in the content, or None if not counted.
        sha256: SHA-256 hex digest of the stored content.
        created_at: UTC timestamp when the artifact was created.
        expires_at: UTC timestamp when the artifact expires, or None.
    """
    artifact_id: str
    artifact_uri: str
    artifact_path: str
    total_bytes: int
    total_lines: int | None
    sha256: str
    created_at: datetime
    expires_at: datetime | None


_LOGICAL_ROOT = ".harness/artifacts"


def _validate_id(value: str, name: str) -> str:
    """Reject path traversal in session_id / artifact_id.

    Only allow alphanumeric, dash, underscore. No slashes, no dots.
    """
    if not value or not all(c.isalnum() or c in "-_" for c in value):
        raise PathTraversalError(
            f"Invalid {name}: {value!r}. Only alphanumeric, dash, underscore allowed."
        )
    return value


class ArtifactStore:
    """Stores large tool outputs as files on disk.

    Args:
        root_dir: Host filesystem root for artifact storage. Typically
                  ``<workdir>/.harness/artifacts``. Must exist or be creatable.
        session_id: Session identifier for isolation. Each session gets its
                    own subdirectory.
        default_ttl: Default time-to-live for artifacts. None = no expiry.
        max_bytes: Maximum bytes per artifact. Larger content is rejected
                   with ArtifactWriteError (caller degrades to truncation).
    """

    def __init__(
        self,
        root_dir: Path,
        session_id: str,
        *,
        default_ttl: timedelta | None = None,
        max_bytes: int = 50 * 1024 * 1024,  # 50 MB default
    ):
        _validate_id(session_id, "session_id")
        self._root = Path(root_dir)
        self._session_id = session_id
        self._session_dir = self._root / session_id
        self._default_ttl = default_ttl
        self._max_bytes = max_bytes
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        content: str | bytes,
        *,
        total_lines: int | None = None,
        ttl: timedelta | None = None,
    ) -> ArtifactRef:
        """Atomically write content to an artifact file and return a ref.

        Raises:
            ArtifactWriteError: if content exceeds max_bytes or write fails.
        """
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content

        if len(data) > self._max_bytes:
            raise ArtifactWriteError(
                f"Artifact content ({len(data)} bytes) exceeds max_bytes "
                f"({self._max_bytes}). Degrading to truncation."
            )

        artifact_id = uuid.uuid4().hex
        _validate_id(artifact_id, "artifact_id")
        filename = f"{artifact_id}.txt"
        target = self._session_dir / filename
        logical_path = f"{_LOGICAL_ROOT}/{self._session_id}/{filename}"

        # Atomic write: temp file in same dir, then rename.
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self._session_dir, prefix=".tmp_", suffix=".txt"
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, target)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            raise ArtifactWriteError(f"Failed to write artifact: {e}") from e

        sha = hashlib.sha256(data).hexdigest()
        now = datetime.now(timezone.utc)
        expires = now + (ttl or self._default_ttl) if (ttl or self._default_ttl) else None

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_uri=f"artifact://{artifact_id}",
            artifact_path=logical_path,
            total_bytes=len(data),
            total_lines=total_lines,
            sha256=sha,
            created_at=now,
            expires_at=expires,
        )

    def read_by_uri(self, uri: str) -> bytes:
        """Read artifact content by ``artifact://<id>`` URI.

        Security: this method ONLY accepts the virtual URI, never a
        filesystem path. The store resolves the URI to its own session
        directory, so cross-session access is impossible — the caller
        cannot craft a URI that escapes the current store's session.

        Raises:
            ArtifactError: if URI is malformed or artifact not found.
        """
        if not isinstance(uri, str) or not uri.startswith("artifact://"):
            raise ArtifactError(f"Not an artifact URI: {uri!r}")
        artifact_id = uri[len("artifact://"):]
        return self.read(artifact_id)

    def read(self, artifact_id: str) -> bytes:
        """Read artifact content by ID. Mainly for testing / read_file integration."""
        _validate_id(artifact_id, "artifact_id")
        target = self._session_dir / f"{artifact_id}.txt"
        if not target.exists():
            raise ArtifactError(f"Artifact not found: {artifact_id}")
        return target.read_bytes()

    def cleanup_expired(self) -> int:
        """Remove artifacts whose expires_at has passed. Returns count removed."""
        removed = 0
        now = datetime.now(timezone.utc)
        for path in self._session_dir.glob("*.txt"):
            # We don't store per-artifact metadata on disk; in a real impl
            # we'd read a sidecar. For now, this is a stub that could be
            # extended. We skip files newer than a heuristic.
            # Stage 2C-A: stub returns 0; full impl in later stage.
            pass
        return removed

    def delete_session_artifacts(self, session_id: str | None = None) -> int:
        """Delete all artifacts for a session (default: this store's session).

        Returns count of files removed.
        """
        sid = session_id or self._session_id
        _validate_id(sid, "session_id")
        target_dir = self._root / sid
        if not target_dir.exists():
            return 0
        count = 0
        for path in target_dir.glob("*.txt"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        # Remove empty session dir
        try:
            target_dir.rmdir()
        except OSError:
            pass
        return count

    def session_dir(self) -> Path:
        """Return the host session directory (for testing)."""
        return self._session_dir

    @property
    def root_dir(self) -> Path:
        """Host filesystem root for all sessions' artifacts.

        Stage 2C-B2B-3: exposed so ``harness_core._validate_secure_sandbox``
        can pass this path to ``SandboxBackend.assess_isolation()`` as a
        private path that bash must not reach.
        """
        return self._root
