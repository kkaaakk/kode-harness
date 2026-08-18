"""
base_tools.py - Core tool functions: file I/O, shell execution, code search.

Depends on: config (WORKDIR, SANDBOX)
"""

from __future__ import annotations

import re
import subprocess
import threading
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path

from agents.config import WORKDIR, SANDBOX, RUN_MODE

try:
    from agents.code_search import grep_search, glob_search, GrepSearchResult
except ImportError:
    from code_search import grep_search, glob_search, GrepSearchResult  # type: ignore[no-redef]


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# Pre-compiled dangerous-command patterns (regex, not substring).
_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\brm\b[^|&;]*\s-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(?:\s|$|\*)'), 'rm -rf /'),
    (re.compile(r'\brm\b[^|&;]*\s-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/(?:\s|$|\*)'), 'rm -fr /'),
    (re.compile(r'\bsudo\b'), 'sudo'),
    (re.compile(r'\b(shutdown|reboot|halt|poweroff|init\s+[06])\b'), 'system control'),
    (re.compile(r'\bgit\s+push\b[^|&;]*(-[a-zA-Z]*f|--(force|force-with-lease))\b'), 'force push'),
    (re.compile(r'\bchmod\s+(777|666|a\+rwx)\b'), 'chmod 777'),
    (re.compile(r'\bdd\b[^|&;]*\bof=/dev/'), 'dd → device'),
    (re.compile(r'\bmkfs\b'), 'mkfs'),
    (re.compile(r'>\s*/dev/sd[a-z]'), 'write to block device'),
    (re.compile(r'\b(curl|wget)\b[^|&;]*\|\s*(ba)?sh'), 'remote code exec (curl|sh)'),
    (re.compile(r':\(\)\s*\{\s*:\s*\\?\|\s*:\s*&\s*}\s*;\s*:'), 'fork bomb'),
]


def _is_dangerous(command: str) -> str | None:
    """Return the matched rule label if *command* is dangerous, else ``None``."""
    for pattern, label in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return label
    return None


@dataclass(frozen=True)
class BashExecutionResult:
    """Structured result of a bash command execution (stage 2C-B2A).

    Fields:
        stdout: Standard output of the command (already stripped of trailing
                whitespace by the sandbox). May be empty.
        stderr: Standard error of the command. May be empty.
        exit_code: Process exit code. 0 = success. Non-zero = failure.
        command: The original command string, optional (for logging/debug).

    Serialization contract:
        ``__str__`` produces a backward-compatible text representation so
        existing harness code that does ``str(output)`` keeps working. The
        format is:

            [exit_code: N]
            <stdout>
            [stderr]
            <stderr>

        The ``[exit_code: N]`` line is always present so downstream code
        can detect failure without parsing natural language. The
        ``[stderr]`` block is omitted when stderr is empty.

    Why a dataclass, not a dict:
        - Frozen dataclass is hashable and immutable.
        - ``isinstance(x, BashExecutionResult)`` is a clean type guard.
        - ToolOutputPolicy can pattern-match on the type and access
          stdout/stderr/exit_code directly, instead of regex-parsing
          the serialized text.
    """
    stdout: str
    stderr: str
    exit_code: int
    command: str | None = None

    def __str__(self) -> str:
        parts = [f"[exit_code: {self.exit_code}]"]
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append("[stderr]")
            parts.append(self.stderr)
        if not self.stdout and not self.stderr:
            parts.append("(no output)")
        return "\n".join(parts)

    @property
    def is_error(self) -> bool:
        """True if the command failed (non-zero exit code). Used by
        harness_core to decide whether to skip OutputPolicy processing
        (consistent with the pre-2C-B2A ``output.startswith('Error')``
        behavior for string outputs)."""
        return self.exit_code != 0


# ---------------------------------------------------------------------------
# Per-call secure-context token (Stage 2C-B2B-4.1, ContextVar-based)
# ---------------------------------------------------------------------------
# In secure_multi_session mode, bash may only run inside an agent_loop()
# call that has ALREADY passed _validate_secure_sandbox(). Direct calls
# to run_bash() from outside agent_loop (e.g. a stray import + call) must
# be rejected, because they bypass the mount-plan assessment.
#
# Stage 2C-B2B-4.1: switched from threading.local to contextvars.ContextVar.
# threading.local only isolates threads, NOT asyncio Tasks running on the
# same event-loop thread. With threading.local, Task A setting a token
# and awaiting would leak the token to Task B on the same thread — a
# bypass. ContextVar is copied per Task, so each Task gets its own
# independent view of the context.
#
# Stage 2C-B2B-4.2: ContextVar is COPIED into child Tasks created via
# asyncio.create_task / ensure_future. A child Task inherits the parent's
# SecureBashContext at creation time, and parent's reset(token) cannot
# revoke the copy already held by the child. If the parent agent_loop
# ends while a child Task (Extension / team / background) is still alive,
# the child could still call run_bash() on a stale credential — a bypass.
#
# Fix: each SecureBashContext carries a unique nonce; the nonce is also
# registered in a process-wide ``_ACTIVE_SECURE_BASH_RUNS`` set at set()
# time and discarded at reset() time. has_valid_secure_bash_context()
# requires BOTH (a) the ContextVar carries a matching context AND
# (b) the nonce is still in the live set. When the parent agent_loop
# resets, the nonce is discarded, so any child Task holding a copied
# context now fails the live-set check — closed.
#
# The context also binds to the specific sandbox instance (id(sandbox))
# so that if the global SANDBOX is swapped after validation, the token
# no longer matches and run_bash fails closed.
#
# Nested calls: agent_loop uses ContextVar.reset(token) (not set(None))
# so that a nested agent_loop that sets its own context restores the
# outer context on exit — clearing only its own frame, not clobbering
# a concurrently-running outer agent_loop. The live nonce set is
# orthogonal: inner reset discards only the inner nonce; the outer
# nonce remains live because it was added by the outer set() and is
# only discarded by the outer reset().
#
# The live-nonce set is guarded by a threading.Lock because the project
# mixes asyncio Tasks and real threads (team_manager, background_run).
@dataclass(frozen=True)
class SecureBashContext:
    """Per-agent_loop-call secure-bash validation context.

    Stored in a ContextVar so each asyncio.Task gets its own copy.
    Binds to the sandbox instance that was validated, so swapping the
    global SANDBOX after validation invalidates the context. Carries a
    nonce registered in ``_ACTIVE_SECURE_BASH_RUNS`` so that a child
    Task that inherited a copy of the context cannot use it after the
    owning agent_loop has reset.
    """
    run_id: str
    sandbox_identity: int
    nonce: str


_SECURE_BASH_CONTEXT: ContextVar[SecureBashContext | None] = ContextVar(
    "secure_bash_context",
    default=None,
)

# Process-wide live nonce registry. A nonce is added at set_secure_bash_context
# time and discarded at reset_secure_bash_context time. has_valid_secure_bash_context
# requires the nonce to still be present, so a child Task that inherited a copied
# ContextVar value cannot use it once the owning agent_loop has reset.
# Guarded by a lock because the project mixes asyncio Tasks and real threads.
_ACTIVE_SECURE_BASH_RUNS: set[str] = set()
_ACTIVE_SECURE_BASH_RUNS_LOCK = threading.Lock()


def set_secure_bash_context(
    *,
    run_id: str,
    sandbox: object,
) -> Token:
    """Set the per-call secure-bash validation context.

    Called by harness_core.agent_loop after _validate_secure_sandbox
    passes. Returns a Token that MUST be passed to
    reset_secure_bash_context() in the finally block — do NOT use
    set(None), which would clobber a concurrently-running outer call's
    context instead of restoring it.

    Also registers a fresh nonce in ``_ACTIVE_SECURE_BASH_RUNS`` so that
    child Tasks inheriting a copy of the context cannot use it after
    this agent_loop resets (B2B-4.2 child-Task revocation).
    """
    nonce = uuid.uuid4().hex
    with _ACTIVE_SECURE_BASH_RUNS_LOCK:
        _ACTIVE_SECURE_BASH_RUNS.add(nonce)
    return _SECURE_BASH_CONTEXT.set(
        SecureBashContext(
            run_id=run_id,
            sandbox_identity=id(sandbox),
            nonce=nonce,
        )
    )


def reset_secure_bash_context(token: Token) -> None:
    """Reset the secure-bash context to its pre-call value.

    Uses ContextVar.reset(token) so nested agent_loop() calls restore
    the outer context rather than clearing it to None. Called in
    agent_loop's finally block.

    ALSO discards the nonce from ``_ACTIVE_SECURE_BASH_RUNS`` so that
    any child Task still holding a copied context (B2B-4.2) cannot use
    it after this point. The nonce is read from the ContextVar BEFORE
    reset (after reset the current Task's context is the outer one or
    None, and we must not discard the outer nonce).
    """
    # Capture the context we are about to reset, so we can revoke its
    # nonce. ``token`` was returned by set(); the Var's current value
    # in THIS Task is the context set by that set() call (unless a
    # nested set() overwrote it — but then the nested call would have
    # already reset to ours, restoring our value here).
    current = _SECURE_BASH_CONTEXT.get()
    if current is not None:
        with _ACTIVE_SECURE_BASH_RUNS_LOCK:
            _ACTIVE_SECURE_BASH_RUNS.discard(current.nonce)
    _SECURE_BASH_CONTEXT.reset(token)


def has_valid_secure_bash_context(sandbox: object) -> bool:
    """Return True iff the current async context holds a valid
    secure-bash context bound to the given sandbox instance AND the
    context's nonce is still live (owning agent_loop has not reset).

    This means an agent_loop() in the current Task has passed secure-mode
    startup validation for the given sandbox. Direct run_bash() calls
    without a context (or with a context bound to a different sandbox,
    or with a nonce whose owning agent_loop has ended) are rejected in
    secure_multi_session mode.
    """
    context = _SECURE_BASH_CONTEXT.get()
    if context is None:
        return False
    if context.sandbox_identity != id(sandbox):
        return False
    with _ACTIVE_SECURE_BASH_RUNS_LOCK:
        return context.nonce in _ACTIVE_SECURE_BASH_RUNS


def _is_bash_blocked_by_run_mode() -> str | None:
    """Stage 2C-B2B-2/4.1: in secure_multi_session mode, bash requires
    (a) a sandbox with supports_filesystem_isolation=True AND
    (b) a per-Task secure-bash context (ContextVar) proving agent_loop
        already passed _validate_secure_sandbox (which includes the
        mount-plan assessment), bound to the CURRENT sandbox instance.
    Returns a denial reason string if bash must be blocked, else None.

    This is the RUNTIME guard inside run_bash. The primary check is
    the STARTUP-TIME validation in agent_loop. This guard stays as
    defence-in-depth:
      - catches direct run_bash() calls that bypass agent_loop
      - catches tool profiles that somehow add bash after startup
      - catches sandbox-swap-after-validation (context binds to id)
      - fails closed: no context / wrong sandbox → no bash

    trusted_local mode never blocks here — NoOpSandbox is explicitly
    allowed and the user accepts that there is no cross-session
    filesystem isolation.
    """
    if RUN_MODE != "secure_multi_session":
        return None
    caps = getattr(SANDBOX, "capabilities", None)
    if caps is None:
        # Backend does not declare capabilities — treat as unisolated.
        return ("secure_multi_session mode requires a sandbox with "
                "supports_filesystem_isolation=True, but the active "
                "backend does not declare capabilities")
    if not caps.supports_filesystem_isolation:
        return ("secure_multi_session mode requires a sandbox with "
                "supports_filesystem_isolation=True, but the active "
                f"backend ({type(SANDBOX).__name__}) does not support it")
    # Capability is fine — but did an agent_loop in THIS async Task
    # actually validate the current mount plan, AND is the context
    # bound to the sandbox we're about to execute on? Without a valid
    # context, this is a direct call that bypassed the startup check
    # (or the sandbox was swapped after validation). Fail closed.
    if not has_valid_secure_bash_context(SANDBOX):
        return ("secure_multi_session mode requires bash to run inside "
                "an agent_loop() call that has passed startup validation "
                "for the current sandbox; this direct call bypassed the "
                "mount-plan assessment or the sandbox was swapped")
    return None


def run_bash(command: str) -> BashExecutionResult | str:
    """Execute a shell command and return a structured result.

    Returns:
        BashExecutionResult on normal execution (including non-zero exit
        codes — those are NOT errors at the transport level, they are
        command-level failures and the model should see them).

        str starting with "Error:" only when the command is blocked by
        the dangerous-command checker, the run-mode guard, or the
        sandbox times out. These are transport-level failures (the
        command never ran), so they bypass OutputPolicy and are
        recorded as tool errors.
    """
    blocked = _is_dangerous(command)
    if blocked:
        return f"Error: Dangerous command blocked ({blocked})"
    mode_block = _is_bash_blocked_by_run_mode()
    if mode_block:
        return f"Error: {mode_block}"
    try:
        # Use the sandbox's structured execution if available; fall back
        # to the legacy string-returning execute() for backends that
        # haven't been upgraded yet.
        if hasattr(SANDBOX, "execute_structured"):
            stdout, stderr, exit_code = SANDBOX.execute_structured(command)
        else:
            combined = SANDBOX.execute(command)
            stdout, stderr, exit_code = combined, "", 0
        return BashExecutionResult(
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
            command=command,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        return SANDBOX.write_file(str(fp), content)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        return SANDBOX.edit_file(str(fp), old_text, new_text)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def run_grep_search(
    pattern: str,
    path: str = ".",
    include: str = "",
    ignore_case: bool = False,
    max_results: int = 50,
) -> GrepSearchResult:
    """Run grep_search and return a structured GrepSearchResult.

    Stage 2C-B3A: returns GrepSearchResult (not str) so ToolOutputPolicy
    can read total_matches / matched_files directly from the structured
    fields instead of parsing natural-language summary lines. The
    model-facing content is still the legacy text format produced by
    ``str(GrepSearchResult)`` — agent_loop's ``str(output)`` at the
    message-boundary handles the conversion.
    """
    try:
        return grep_search(
            pattern,
            path=path,
            include=include,
            ignore_case=ignore_case,
            max_results=max_results,
            workdir=WORKDIR,
        )
    except Exception as e:
        # Transport error: structured error result so the policy can
        # distinguish "search aborted" from "zero hits".
        return GrepSearchResult(
            query=pattern,
            matches=(),
            total_matches=None,
            matched_files=None,
            errors=(str(e),),
            metadata_complete=False,
        )


def run_glob_search(
    pattern: str, path: str = ".", max_results: int = 100
) -> str:
    try:
        return glob_search(
            pattern, path=path, max_results=max_results, workdir=WORKDIR
        )
    except Exception as e:
        return f"Error: {e}"
