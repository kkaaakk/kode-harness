"""tool_output_policy.py - Unified tool output processing + artifact offloading.

Stage 2C-A: infrastructure only, not yet wired into harness_core.

Purpose
-------
When a tool returns a large result, we don't want to dump 50KB of text into
the model context. Instead:

    raw_result
      -> ToolOutputPolicy.process(tool_name, raw_result, context)
      -> ProcessedToolOutput(content, truncated, artifact, metadata)

If the result is small enough, ``content`` is the original result unchanged
and ``truncated=False`` / ``artifact=None``.

If the result exceeds thresholds, the full content is written to an
ArtifactStore, and ``content`` becomes a structured preview with a reference
to the artifact_path. The model can later read the full content via read_file.

Tool formatters
---------------
Different tools need different preview formats:
  - read_file: head N lines + tail M lines with original line numbers
  - bash: separate stdout/stderr/exit_code (must not lose exit code)
  - grep_search: total_matches, matched_files, shown_matches

Unknown tools fall back to a generic text formatter (head + tail).

Order with extensions
---------------------
The core Output Policy runs BEFORE AFTER_TOOL_RESULT extensions. This means
extensions see the already-truncated, artifact-offloaded result, never the
raw huge output. This prevents an extension from re-injecting huge content
into the context.

Failure handling
----------------
If the ArtifactStore write fails, we degrade to safe truncation
(inline_max_bytes) and set artifact=None. The agent loop must NOT crash.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from agents.artifacts import ArtifactRef, ArtifactStore, ArtifactWriteError


_logger = logging.getLogger("agents.tool_output_policy")


def _is_bash_result(obj: Any) -> bool:
    """Duck-typed check for BashExecutionResult without importing it.

    We avoid ``from agents.base_tools import BashExecutionResult`` here to
    keep the import graph clean (base_tools pulls in config → sandbox,
    which is heavy and can cause cycles in test reloads). The dataclass
    is frozen and has a stable shape, so a duck-type check is safe.
    """
    return (
        hasattr(obj, "stdout")
        and hasattr(obj, "stderr")
        and hasattr(obj, "exit_code")
        and hasattr(obj, "is_error")
    )


def _is_grep_result(obj: Any) -> bool:
    """Duck-typed check for GrepSearchResult without importing it.

    Stage 2C-B3A: grep_search now returns a structured GrepSearchResult
    with ``query``, ``matches``, ``total_matches``, ``matched_files``,
    ``errors``, and ``metadata_complete``. We duck-type so the policy
    module does not pull in base_tools → config → sandbox (heavy import
    graph, cycle risk in test reloads).
    """
    return (
        hasattr(obj, "query")
        and hasattr(obj, "matches")
        and hasattr(obj, "total_matches")
        and hasattr(obj, "matched_files")
        and hasattr(obj, "errors")
        and hasattr(obj, "metadata_complete")
    )


def resolve_existing_artifact(
    source_path: str,
    *,
    workdir: Path,
    artifact_root: Path,
    session_id: str,
) -> Path | None:
    """Resolve whether source_path points to a real artifact in the current
    session's artifact directory.

    Returns the resolved host Path if it is a file inside
    ``<artifact_root>/<session_id>/``, otherwise None.

    Security:
    - Resolves symlinks and ``..`` via Path.resolve().
    - Uses os.path.commonpath to verify containment.
    - Rejects cross-session reads.
    - Rejects non-existent files.
    - Rejects path-traversal escapes.
    """
    try:
        candidate = (workdir / source_path).resolve()
    except (OSError, ValueError):
        return None
    try:
        session_root = (artifact_root / session_id).resolve()
    except (OSError, ValueError):
        return None
    if not session_root.exists():
        return None
    try:
        common = Path(os.path.commonpath([candidate, session_root]))
    except ValueError:
        # commonpath raises ValueError if paths are on different drives
        # (Windows) or otherwise incomparable.
        return None
    if common != session_root:
        return None
    if not candidate.is_file():
        return None
    return candidate


@dataclass(frozen=True)
class OutputPolicyConfig:
    """Thresholds for inline vs artifact offloading.

    If a result exceeds ANY of these thresholds, it gets artifacted.
    """
    inline_max_bytes: int = 8000
    inline_max_lines: int = 200
    preview_head_lines: int = 50
    preview_tail_lines: int = 20
    artifact_max_bytes: int = 50 * 1024 * 1024  # 50 MB


@dataclass
class ProcessedToolOutput:
    """The result of processing a tool output.

    Fields:
        content: What goes into the model context. Either the original
                 result (small) or a structured preview string (large).
        truncated: True if the inline content is a preview, not the full output.
        artifact: Reference to the full stored output, or None if not offloaded.
        metadata: Tool-specific metadata (e.g. exit_code, total_matches).
    """
    content: str
    truncated: bool
    artifact: ArtifactRef | None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolOutputPolicy:
    """Processes tool outputs, offloading large results to ArtifactStore.

    Args:
        store: The ArtifactStore to use for large outputs. May be None for
               testing (in which case large outputs are truncated but not
               artifacted).
        config: Threshold configuration.
    """

    def __init__(
        self,
        store: ArtifactStore | None,
        config: OutputPolicyConfig | None = None,
        *,
        workdir: Path | None = None,
        artifact_root: Path | None = None,
        session_id: str | None = None,
    ):
        self._store = store
        self._config = config or OutputPolicyConfig()
        self._workdir = workdir
        self._artifact_root = artifact_root
        self._session_id = session_id

    def process(
        self,
        tool_name: str,
        raw_result: Any,
        context: dict | None = None,
    ) -> ProcessedToolOutput:
        """Process a tool result.

        Args:
            tool_name: Name of the tool that produced this result.
            raw_result: The tool's return value. Usually a string; for
                bash (stage 2C-B2A) this may be a BashExecutionResult
                with structured stdout/stderr/exit_code.
            context: Optional execution context (tool args, etc.).

        Returns:
            ProcessedToolOutput with content for the model.
        """
        context = context or {}

        # Stage 2C-B2A: bash structured output. When raw_result is a
        # BashExecutionResult, handle it via the dedicated bash path so
        # we can separate stdout/stderr and preserve exit_code in metadata
        # without parsing natural language.
        if tool_name == "bash" and _is_bash_result(raw_result):
            return self._process_bash_structured(raw_result)

        # Stage 2C-B3A: grep structured output. When raw_result is a
        # GrepSearchResult, handle it via the dedicated grep path so
        # total_matches / matched_files come from the structured fields,
        # not from parsing natural-language summary lines. Zero-hit and
        # execution-error are distinguished by total_matches (0 vs None).
        if tool_name == "grep_search" and _is_grep_result(raw_result):
            return self._process_grep_structured(raw_result)

        # Normalize to string for size measurement.
        if not isinstance(raw_result, str):
            raw_str = str(raw_result)
        else:
            raw_str = raw_result

        # Small output: pass through unchanged.
        if not self._exceeds_threshold(raw_str):
            return ProcessedToolOutput(
                content=raw_str,
                truncated=False,
                artifact=None,
                metadata={},
            )

        # Artifact re-read protection: if the tool is read_file and the source
        # is an artifact:// URI, do NOT create a new artifact (copy chain
        # A -> B -> C). Instead return a preview. The model should use range
        # reads (start_line) to access specific sections of the original
        # artifact:// URI.
        if tool_name == "read_file":
            source_path = context.get("tool_args", {}).get("path", "")
            if isinstance(source_path, str) and source_path.startswith("artifact://"):
                formatter = self._get_formatter(tool_name)
                preview, metadata, total_lines = formatter(raw_str, context)
                metadata["is_artifact_reread"] = True
                metadata["original_artifact_uri"] = source_path
                # Hard-truncate the preview to size limit, no new artifact.
                content = self._hard_truncate(preview)
                content += (
                    f"\n\n[Note: This is an artifact re-read. Use read_file"
                    f" with the artifact:// URI and start_line/limit to access"
                    f" specific sections of {source_path}]"
                )
                return ProcessedToolOutput(
                    content=content,
                    truncated=True,
                    artifact=None,  # No new artifact created
                    metadata=metadata,
                )

        # Large output: format a preview and try to artifact.
        formatter = self._get_formatter(tool_name)
        preview, metadata, total_lines = formatter(raw_str, context)

        artifact = None
        if self._store is not None:
            try:
                artifact = self._store.store(
                    raw_str,
                    total_lines=total_lines,
                )
            except ArtifactWriteError as e:
                # Degrade: keep the preview, no artifact. Do NOT crash.
                _logger.warning("Artifact write failed for tool %s: %s. "
                                "Degrading to inline preview only.", tool_name, e)
                artifact = None

        # Build the final content for the model.
        if artifact is not None:
            content = self._format_with_artifact_ref(preview, artifact, metadata)
        else:
            # No artifact (store missing or write failed): hard-truncate preview.
            content = self._hard_truncate(preview)

        return ProcessedToolOutput(
            content=content,
            truncated=True,
            artifact=artifact,
            metadata=metadata,
        )

    def _process_bash_structured(self, result: Any) -> ProcessedToolOutput:
        """Stage 2C-B2A: process a BashExecutionResult.

        Rules:
          - Small combined output: return the backward-compatible str()
            representation, untruncated, with exit_code in metadata.
          - Large combined output: write a JSON artifact containing the
            full structured result (stdout/stderr/exit_code/command),
            and return a preview that separates stdout and stderr while
            always preserving the exit_code line.
          - Artifact write failure: degrade to hard-truncated preview,
            no crash, exit_code still preserved.
        """
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.exit_code
        combined = f"{stdout}\n{stderr}".strip()

        metadata: dict[str, Any] = {
            "tool": "bash",
            "exit_code": str(exit_code),
        }

        # Small output: pass through as text (backward compatible).
        # The str() representation always includes [exit_code: N] so the
        # model can see the exit code without parsing.
        if not self._exceeds_threshold(combined):
            return ProcessedToolOutput(
                content=str(result),
                truncated=False,
                artifact=None,
                metadata=metadata,
            )

        # Large output: build a structured preview.
        preview = self._format_bash_preview(stdout, stderr, exit_code)

        # Artifact: store the FULL structured data as JSON so the model
        # can recover stdout/stderr/exit_code exactly via read_file on
        # the artifact URI. We use indent=2 so a re-read preview shows
        # the structured fields (exit_code/stderr/command) near the top
        # instead of being buried after a huge stdout string.
        artifact = None
        if self._store is not None:
            payload = json.dumps({
                "exit_code": exit_code,
                "stderr": stderr,
                "command": result.command,
                "stdout": stdout,
            }, ensure_ascii=False, indent=2)
            try:
                artifact = self._store.store(
                    payload,
                    total_lines=stdout.count("\n") + 1 + stderr.count("\n") + 1,
                )
            except ArtifactWriteError as e:
                _logger.warning("Artifact write failed for bash: %s. "
                                "Degrading to inline preview only.", e)
                artifact = None

        if artifact is not None:
            content = self._format_with_artifact_ref(preview, artifact, metadata)
        else:
            content = self._hard_truncate(preview)

        return ProcessedToolOutput(
            content=content,
            truncated=True,
            artifact=artifact,
            metadata=metadata,
        )

    def _format_bash_preview(
        self, stdout: str, stderr: str, exit_code: int
    ) -> str:
        """Build a semantic preview for a large bash result.

        stdout and stderr are truncated independently (head + tail each)
        so a huge stderr cannot squeeze out the stdout the model needs,
        and vice versa. The exit_code line is always present at the top.
        """
        head = self._config.preview_head_lines
        tail = self._config.preview_tail_lines

        def _truncate(text: str) -> str:
            lines = text.split("\n") if text else []
            if not lines:
                return ""
            if len(lines) <= head + tail:
                return "\n".join(lines)
            omitted = len(lines) - head - tail
            return (
                "\n".join(lines[:head])
                + f"\n... ({omitted} lines omitted) ...\n"
                + "\n".join(lines[-tail:])
            )

        parts = [f"[exit_code: {exit_code}]"]
        if stdout:
            parts.append("--- stdout ---")
            parts.append(_truncate(stdout))
        if stderr:
            parts.append("--- stderr ---")
            parts.append(_truncate(stderr))
        if not stdout and not stderr:
            parts.append("(no output)")
        return "\n".join(parts)

    def _process_grep_structured(self, result: Any) -> ProcessedToolOutput:
        """Stage 2C-B3C: process a GrepSearchResult.

        Distinguishes two independent truncation states:

          - ``search_limited``: ``grep_search`` caps ``matches`` at
            ``max_results`` (default 50). When ``total_matches > len(matches)``,
            the remaining matches were NEVER materialized — they are not
            in the result, cannot be stored in an artifact, and cannot
            be recovered without re-running the search. The model must
            be told this explicitly so it doesn't mistake an artifact
            of the 50 returned matches for the complete 1200-match set.

          - ``output_truncated``: the 50 returned matches, rendered as
            legacy text, exceed the inline size threshold. Only THEN do
            we create an artifact — and that artifact contains ONLY the
            50 returned matches, marked ``artifact_complete=False``.

        Branches:
          1. Execution error (total_matches is None): error string.
          2. Zero-hit (total_matches == 0): "No matches found".
          3. Small output, no search limiting: legacy text, no artifact.
          4. Small output, search limited: legacy text + explicit
             "remaining N matches not materialized" notice, no artifact.
          5. Large output: JSONL artifact of the RETURNED matches only,
             with ``artifact_scope="returned_matches"`` and
             ``artifact_complete=False`` when ``search_limited``. Preview
             preserves path/line_number for the shown matches.
          6. Artifact write failure: degrade to hard-truncated preview.
        """
        query = getattr(result, "query", "")
        matches = getattr(result, "matches", ())
        total_matches = getattr(result, "total_matches", None)
        matched_files = getattr(result, "matched_files", None)
        errors = getattr(result, "errors", ())
        metadata_complete = getattr(result, "metadata_complete", True)

        returned_matches = len(matches)
        search_limited = (
            total_matches is not None and returned_matches < total_matches
        )

        metadata: dict[str, Any] = {
            "tool": "grep_search",
            "query": query,
            "total_matches": (
                str(total_matches) if total_matches is not None else None
            ),
            "matched_files": (
                str(matched_files) if matched_files is not None else None
            ),
            "returned_matches": str(returned_matches),
            "search_limited": search_limited,
            "metadata_complete": metadata_complete,
        }

        # Branch 1: execution error.
        if total_matches is None:
            metadata["errors"] = list(errors)
            metadata["metadata_complete"] = False
            if errors:
                content = f"Error: {errors[0]}"
            else:
                content = f"Error: search aborted for pattern: {query}"
            return ProcessedToolOutput(
                content=content,
                truncated=False,
                artifact=None,
                metadata=metadata,
            )

        # Branch 2: zero-hit.
        if total_matches == 0 or not matches:
            return ProcessedToolOutput(
                content=f"No matches found for pattern: {query}",
                truncated=False,
                artifact=None,
                metadata=metadata,
            )

        # Build the legacy text representation for size measurement.
        legacy_text = str(result)
        output_truncated = self._exceeds_threshold(legacy_text)

        # Branch 3 + 4: small output (inline-able as legacy text).
        if not output_truncated:
            if search_limited:
                # Branch 4: search was limited — append an explicit notice
                # so the model knows the remaining matches were not
                # materialized and cannot be read from any artifact.
                hidden = total_matches - returned_matches
                content = legacy_text + (
                    f"\n\n[search limited]\n"
                    f"Search found {total_matches} matches; "
                    f"{returned_matches} were returned.\n"
                    f"The remaining {hidden} matches were not materialized.\n"
                    f"Refine the query or use narrower include/path filters."
                )
            else:
                # Branch 3: complete result, small enough to inline.
                content = legacy_text
            return ProcessedToolOutput(
                content=content,
                truncated=False,
                artifact=None,
                metadata=metadata,
            )

        # Branch 5 + 6: large output — the RETURNED matches themselves
        # exceed the inline threshold. Build a semantic preview that
        # preserves path/line_number so the model can act on the shown
        # matches.
        preview = self._format_grep_preview(
            result, head=self._config.preview_head_lines
        )

        # Artifact: store the RETURNED matches as JSONL. The metadata
        # header row carries artifact_scope / artifact_complete so a
        # re-read preview makes clear whether this is the full match
        # set or just a slice.
        #
        # B3C.1: two distinct cases:
        #   - search_limited=False → artifact contains EVERY match from
        #     the search. artifact_scope="full_search_result",
        #     artifact_complete=True.
        #   - search_limited=True  → artifact contains only the returned
        #     matches, NOT the full set. artifact_scope="returned_matches",
        #     artifact_complete=False.
        artifact_complete = not search_limited
        artifact_scope = (
            "full_search_result" if artifact_complete else "returned_matches"
        )
        artifact = None
        if self._store is not None:
            payload_lines = [
                json.dumps({
                    "type": "metadata",
                    "query": query,
                    "total_matches": total_matches,
                    "matched_files": matched_files,
                    "returned_matches": returned_matches,
                    "artifact_matches": returned_matches,
                    "artifact_scope": artifact_scope,
                    "artifact_complete": artifact_complete,
                    "search_limited": search_limited,
                    "metadata_complete": metadata_complete,
                }, ensure_ascii=False)
            ]
            for m in matches:
                payload_lines.append(json.dumps({
                    "path": m.path,
                    "line_number": m.line_number,
                    "text": m.text,
                }, ensure_ascii=False))
            payload = "\n".join(payload_lines)
            try:
                artifact = self._store.store(
                    payload,
                    total_lines=len(payload_lines),
                )
            except ArtifactWriteError as e:
                _logger.warning(
                    "Artifact write failed for grep: %s. "
                    "Degrading to inline preview only.", e
                )
                artifact = None

        if artifact is not None:
            content = self._format_grep_artifact_ref(
                preview, artifact, result,
                search_limited=search_limited,
                artifact_complete=artifact_complete,
            )
        else:
            content = self._hard_truncate(preview)

        return ProcessedToolOutput(
            content=content,
            truncated=True,
            artifact=artifact,
            metadata=metadata,
        )

    def _format_grep_preview(self, result: Any, *, head: int) -> str:
        """Build a semantic preview for a large grep result.

        Preserves path + line_number for every shown match so the model
        can immediately call read_file / edit_file on them. The summary
        block (query / total_matches / matched_files / returned_matches /
        shown_matches / search_limited) is always at the top.

        Stage 2C-B3C: the ``truncated`` field was removed because it
        conflated two states. Instead the preview reports
        ``search_limited`` (true when total_matches > returned_matches)
        and ``shown_matches`` (the count actually in the [matches]
        block below). The artifact ref line (added by
        ``_format_grep_artifact_ref``) makes explicit that only the
        returned matches were saved, not the full match set.
        """
        query = getattr(result, "query", "")
        matches = getattr(result, "matches", ())
        total_matches = getattr(result, "total_matches", 0) or 0
        matched_files = getattr(result, "matched_files", 0) or 0
        returned_matches = len(matches)
        search_limited = returned_matches < total_matches

        shown = matches[:head]
        shown_lines = [
            f"{m.path}:{m.line_number}:{m.text}" for m in shown
            if m.line_number is not None
        ]

        parts = [
            "[grep summary]",
            f"query: {query}",
            f"total_matches: {total_matches}",
            f"matched_files: {matched_files}",
            f"returned_matches: {returned_matches}",
            f"shown_matches: {len(shown_lines)}",
            f"search_limited: {search_limited}",
            "",
            "[matches]",
        ]
        parts.extend(shown_lines)
        return "\n".join(parts)

    def _format_grep_artifact_ref(
        self, preview: str, artifact: ArtifactRef, result: Any,
        *, search_limited: bool, artifact_complete: bool,
    ) -> str:
        """Combine grep preview with artifact reference.

        B3C.1: two distinct wordings depending on artifact_complete.

        - artifact_complete=True (search_limited=False): the artifact
          contains EVERY match from the search. Wording is
          "[Full search result saved to artifact: ...]" and the model
          is told all N matches are included.

        - artifact_complete=False (search_limited=True): the artifact
          contains only the RETURNED matches. Wording is
          "[Returned matches saved to artifact: ...]" and the model is
          explicitly told how many matches are missing and that they
          are NOT in the artifact.
        """
        returned_matches = len(getattr(result, "matches", ()))
        total_matches = getattr(result, "total_matches", None)

        parts = [preview, ""]
        if artifact_complete:
            # Complete artifact — every match is included.
            parts.append(
                f"[Full search result saved to artifact: "
                f"{artifact.artifact_uri}]"
            )
            parts.append(
                f"[Size: {artifact.total_bytes} bytes, "
                f"Lines: {artifact.total_lines or 'unknown'}]"
            )
            parts.append(f"[SHA-256: {artifact.sha256[:16]}...]")
            parts.append(
                f"[artifact_scope: full_search_result]"
            )
            parts.append(
                f"[artifact_complete: True — all {total_matches} "
                f"matches are included]"
            )
        else:
            # Partial artifact — only returned matches are included.
            hidden = (total_matches or 0) - returned_matches
            parts.append(
                f"[Returned matches saved to artifact: "
                f"{artifact.artifact_uri}]"
            )
            parts.append(
                f"[Size: {artifact.total_bytes} bytes, "
                f"Lines: {artifact.total_lines or 'unknown'}]"
            )
            parts.append(f"[SHA-256: {artifact.sha256[:16]}...]")
            parts.append(
                f"[artifact_scope: returned_matches only "
                f"({returned_matches} of {total_matches} total)]"
            )
            parts.append(
                f"[artifact_complete: False — {returned_matches} of "
                f"{total_matches} matches are included]"
            )
            parts.append(
                f"[The remaining {hidden} matches were not materialized "
                f"and are NOT in this artifact]"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exceeds_threshold(self, text: str) -> bool:
        if len(text.encode("utf-8")) > self._config.inline_max_bytes:
            return True
        if text.count("\n") + 1 > self._config.inline_max_lines:
            return True
        return False

    def _get_formatter(self, tool_name: str):
        if tool_name == "read_file":
            return self._format_read_file
        if tool_name == "bash":
            return self._format_bash
        if tool_name == "grep_search":
            return self._format_grep
        return self._format_generic

    def _format_read_file(
        self, text: str, context: dict
    ) -> tuple[str, dict, int | None]:
        """Preview: head N lines + tail M lines with line numbers."""
        lines = text.split("\n")
        total = len(lines)
        head = self._config.preview_head_lines
        tail = self._config.preview_tail_lines

        if total <= head + tail:
            # Just number all lines.
            numbered = [f"{i+1:>6} | {ln}" for i, ln in enumerate(lines)]
            return "\n".join(numbered), {"total_lines": total}, total

        head_lines = [f"{i+1:>6} | {lines[i]}" for i in range(head)]
        tail_lines = [
            f"{total-tail+i+1:>6} | {lines[total-tail+i]}" for i in range(tail)
        ]
        omitted = total - head - tail
        preview = (
            "\n".join(head_lines)
            + f"\n... ({omitted} lines omitted) ...\n"
            + "\n".join(tail_lines)
        )
        return preview, {"total_lines": total}, total

    def _format_bash(
        self, text: str, context: dict
    ) -> tuple[str, dict, int | None]:
        """Preview: preserve stdout/stderr/exit_code semantics.

        Current bash returns a plain string. We try to preserve it as-is
        but note that exit code is not separately available in the current
        run_bash return. When 2C-B rewires bash to return a structured
        BashResult, this formatter will properly separate fields.
        """
        lines = text.split("\n")
        total = len(lines)
        head = self._config.preview_head_lines
        tail = self._config.preview_tail_lines

        metadata = {"tool": "bash"}
        # Try to detect exit code line if present.
        for ln in lines:
            if ln.strip().startswith("exit_code:") or ln.strip().startswith("[exit_code]"):
                metadata["exit_code"] = ln.strip()

        if total <= head + tail:
            return text, metadata, total

        head_lines = lines[:head]
        tail_lines = lines[-tail:]
        omitted = total - head - tail
        preview = (
            "\n".join(head_lines)
            + f"\n... ({omitted} lines omitted) ...\n"
            + "\n".join(tail_lines)
        )
        return preview, metadata, total

    def _format_grep(
        self, text: str, context: dict
    ) -> tuple[str, dict, int | None]:
        """Preview: preserve total_matches and matched_files count."""
        lines = text.split("\n")
        total = len(lines)
        head = self._config.preview_head_lines
        tail = self._config.preview_tail_lines

        metadata: dict[str, Any] = {"tool": "grep_search"}
        # Try to parse summary line like "(1834 matches in 92 files)"
        # or "... (1834 more matches in 92 files, showing first 50)"
        summary_line = lines[-1] if lines else ""
        if "matches in" in summary_line and "files" in summary_line:
            metadata["summary"] = summary_line.strip()

        if total <= head + tail:
            return text, metadata, total

        head_lines = lines[:head]
        tail_lines = lines[-tail:]
        omitted = total - head - tail
        preview = (
            "\n".join(head_lines)
            + f"\n... ({omitted} lines omitted) ...\n"
            + "\n".join(tail_lines)
        )
        return preview, metadata, total

    def _format_generic(
        self, text: str, context: dict
    ) -> tuple[str, dict, int | None]:
        """Generic preview: head + tail."""
        lines = text.split("\n")
        total = len(lines)
        head = self._config.preview_head_lines
        tail = self._config.preview_tail_lines

        if total <= head + tail:
            return text, {}, total

        head_lines = lines[:head]
        tail_lines = lines[-tail:]
        omitted = total - head - tail
        preview = (
            "\n".join(head_lines)
            + f"\n... ({omitted} lines omitted) ...\n"
            + "\n".join(tail_lines)
        )
        return preview, {}, total

    def _format_with_artifact_ref(
        self, preview: str, artifact: ArtifactRef, metadata: dict
    ) -> str:
        """Combine preview with artifact reference for the model.

        Uses the virtual ``artifact://<id>`` URI, NOT the filesystem path.
        This means bash/grep/glob cannot access other sessions' artifacts
        via shell — they'd need the artifact_id, and even then the
        ArtifactStore only resolves within the current session.
        """
        parts = [preview, ""]
        parts.append(f"[Full output saved to artifact: {artifact.artifact_uri}]")
        parts.append(f"[Size: {artifact.total_bytes} bytes, "
                     f"Lines: {artifact.total_lines or 'unknown'}]")
        parts.append(f"[SHA-256: {artifact.sha256[:16]}...]")
        return "\n".join(parts)

    def enforce_final(
        self,
        tool_name: str,
        content: Any,
        context: dict | None = None,
    ) -> str:
        """Final output guard: ensure content entering model context is
        within size limits, even if an Extension patched it.

        This runs AFTER AFTER_TOOL_RESULT patches are applied. If the patched
        content is small, it passes through. If it's large, it gets
        hard-truncated (no new artifact — the Extension's content is not
        trusted to be artifact-worthy). This prevents an Extension from
        re-injecting huge content into the context.

        Returns the final content string to put into messages.
        """
        if not isinstance(content, str):
            content = str(content)
        if not self._exceeds_threshold(content):
            return content
        # Hard-truncate the patched content. Do NOT create a new artifact:
        # the Extension-provided content is not a tool's raw output.
        _logger.warning(
            "Final output guard: Extension patch for %s exceeded size limit; "
            "hard-truncating.", tool_name
        )
        truncated = self._hard_truncate(content)
        # Append explicit guard marker so the model knows full content was
        # NOT persisted and cannot be retrieved from an artifact.
        truncated += (
            "\n\n[final output guard applied]\n"
            "Output exceeded the final context limit.\n"
            "Full extension-generated content was not persisted."
        )
        return truncated

    def _hard_truncate(self, preview: str) -> str:
        """When artifact store is unavailable, hard-truncate to inline_max_bytes."""
        data = preview.encode("utf-8")
        if len(data) <= self._config.inline_max_bytes:
            return preview
        truncated = data[:self._config.inline_max_bytes].decode("utf-8", errors="ignore")
        return truncated + f"\n... [hard-truncated at {self._config.inline_max_bytes} bytes]"
