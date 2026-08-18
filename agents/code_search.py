"""code_search.py — Native grep/glob code search for the workspace.

Replaces the external MCP RAG server with pure-Python file search:
- ``grep_search()``: regex-based content search (like ripgrep/grep)
- ``glob_search()``: filename pattern search (like find/glob)

No external dependencies. Cross-platform (Windows/Linux/macOS).

Stage 2C-B3A: ``grep_search`` returns a ``GrepSearchResult`` dataclass
with accurate ``total_matches`` / ``matched_files`` counts and a clear
distinction between zero-hit and execution-error. ``str(result)``
reproduces the legacy text format so existing callers are unaffected.
``ToolOutputPolicy`` reads the structured fields directly instead of
parsing natural-language summary lines.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Exclusion lists
# ---------------------------------------------------------------------------

SEARCH_EXCLUDE_DIRS: set[str] = {
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".idea",
    ".tmp",
    ".pytest_cache",
    ".benchmarks",
    ".tasks",
    ".qoder",
}

SEARCH_EXCLUDE_EXTS: set[str] = {
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
}

DEFAULT_MAX_RESULTS = 50
DEFAULT_MAX_GLOB_RESULTS = 100
MAX_OUTPUT_CHARS = 50_000
MAX_FILE_SIZE_BYTES = 1_000_000  # skip files > 1 MB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_workspace(path: Path, workdir: Path) -> Path:
    """Resolve *path* relative to *workdir* and verify it stays inside."""
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (workdir / path).resolve()
    if not resolved.is_relative_to(workdir.resolve()):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def _is_searchable(file_path: Path) -> bool:
    """Return True if *file_path* is a regular text file worth searching."""
    if not file_path.is_file():
        return False
    if file_path.suffix.lower() in SEARCH_EXCLUDE_EXTS:
        return False
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
    except OSError:
        return False
    return True


def _should_skip_dir(dir_name: str) -> bool:
    """Return True if this directory name should be excluded from traversal."""
    return dir_name in SEARCH_EXCLUDE_DIRS


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield all searchable files under *root*, skipping excluded dirs."""
    try:
        entries = sorted(root.iterdir())
    except PermissionError:
        return
    for entry in entries:
        if entry.is_dir():
            if not _should_skip_dir(entry.name):
                yield from _iter_files(entry)
        elif entry.is_file() and _is_searchable(entry):
            yield entry


def _match_glob(file_path: Path, pattern: str) -> bool:
    """Check if *file_path* matches a glob *pattern*.

    Supports:
    - simple globs: ``*.py``, ``test_*.py``
    - recursive globs: ``**/*.py``
    - brace alternatives are NOT supported (fnmatch limitation)
    """
    # Normalise to forward-slash relative path for matching
    rel = file_path.as_posix()

    # For "**/" patterns, match against the full relative path
    if "**" in pattern:
        # fnmatch doesn't understand "**", so we convert to a regex
        regex = _glob_to_regex(pattern)
        return bool(re.match(regex, rel))

    # Simple pattern: match against the filename only
    return fnmatch.fnmatch(file_path.name, pattern)


def _glob_to_regex(pattern: str) -> str:
    """Convert a glob pattern with ``**`` support to a regex string."""
    i = 0
    n = len(pattern)
    result: list[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # "**" — match any path segment(s)
                if i + 2 < n and pattern[i + 2] == "/":
                    result.append("(?:.+/)?")
                    i += 3
                    continue
                result.append(".*")
                i += 2
                continue
            result.append("[^/]*")
        elif c == "?":
            result.append("[^/]")
        elif c == "[":
            j = i + 1
            while j < n and pattern[j] != "]":
                j += 1
            result.append(pattern[i : j + 1])
            i = j
        elif c in r"\.+^${}|()":
            result.append("\\" + c)
        else:
            result.append(c)
        i += 1
    return "^" + "".join(result) + "$"


def _read_lines(file_path: Path) -> list[str] | None:
    """Read file as UTF-8 lines; return None on decode/IO errors."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="strict")
        return text.splitlines()
    except (UnicodeDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrepMatch:
    """A single grep match: file path + line number + line text.

    ``line_number`` is None when a match is reported without a line
    number (currently unused — every match has a concrete line number —
    but reserved for future streaming variants).
    """
    path: str
    line_number: int | None
    text: str


@dataclass(frozen=True)
class GrepSearchResult:
    """Structured result of ``grep_search``.

    Stage 2C-B3A: replaces the plain-string return so ``total_matches``
    and ``matched_files`` come from the actual search loop, NOT from
    parsing natural-language summary lines.

    Semantics:
        * ``total_matches`` — total number of matching lines across all
          files. ``0`` means the search ran successfully and matched
          nothing. ``None`` means the search could not run (e.g. invalid
          regex, path not found); check ``errors``.
        * ``matched_files`` — number of distinct files with at least one
          match. ``0`` for zero-hit or error cases.
        * ``errors`` — non-fatal execution errors (e.g. permission
          denied on one file). An error result still has
          ``total_matches=None`` and ``metadata_complete=False`` so the
          caller can distinguish "zero hits" from "search aborted".
        * ``metadata_complete`` — True iff ``total_matches`` and
          ``matched_files`` are exact (the search traversed all files).
          False when the search was aborted by an error or by an output
          size cap.
        * ``matches`` — the (possibly truncated) list of matches shown
          inline. ``len(matches)`` is what the model sees in the text
          preview; ``total_matches`` is the ground truth.

    ``__str__`` reproduces the legacy text format so existing callers
    (including the model-facing tool result) are unaffected.
    """
    query: str
    matches: tuple[GrepMatch, ...]
    total_matches: int | None
    matched_files: int | None
    errors: tuple[str, ...] = ()
    metadata_complete: bool = True

    def __str__(self) -> str:
        """Render the legacy text format.

        Zero-hit → "No matches found for pattern: <query>"
        Error    → "Error: <first error>"
        Hit      → "<path>:<line>:<text>" lines + summary line
        """
        if self.total_matches is None:
            # Error case: surface the first error (legacy behaviour).
            if self.errors:
                return f"Error: {self.errors[0]}"
            return f"Error: search aborted for pattern: {self.query}"
        if self.total_matches == 0 or not self.matches:
            return f"No matches found for pattern: {self.query}"

        lines = [f"{m.path}:{m.line_number}:{m.text}" for m in self.matches
                 if m.line_number is not None]
        shown = len(lines)
        # Use the ground-truth total/matched_files when metadata_complete,
        # otherwise fall back to what we actually have.
        total = self.total_matches if self.metadata_complete else shown
        files = self.matched_files if self.matched_files is not None else 0

        if total > shown:
            remaining = total - shown
            lines.append(
                f"\n... ({remaining} more matches in {files} files, "
                f"showing first {shown})"
            )
        else:
            lines.append(f"\n({total} matches in {files} files)")
        result = "\n".join(lines)
        return result[:MAX_OUTPUT_CHARS]


def grep_search(
    pattern: str,
    *,
    path: str = ".",
    include: str = "",
    ignore_case: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
    workdir: Path,
) -> GrepSearchResult:
    """Search file contents for a regex *pattern* across workspace files.

    Returns a :class:`GrepSearchResult` with accurate ``total_matches``
    and ``matched_files``. ``str(result)`` reproduces the legacy text
    format (``relative/path:linenum:content`` + summary line).

    Parameters
    ----------
    pattern:
        Regex pattern to search for (Python ``re`` syntax).
    path:
        Directory or file to search within, relative to *workdir*. Default ``"."``.
    include:
        Glob pattern to filter which files are searched (e.g. ``"*.py"``).
        Empty string means search all text files.
    ignore_case:
        Case-insensitive search. Default ``False``.
    max_results:
        Maximum number of matching lines to return inline. Default 50.
    workdir:
        Workspace root — all paths must stay inside this directory.
    """
    cleaned = pattern.strip()
    if not cleaned:
        return GrepSearchResult(
            query=pattern,
            matches=(),
            total_matches=None,
            matched_files=None,
            errors=("pattern is required",),
            metadata_complete=False,
        )

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(cleaned, flags)
    except re.error as exc:
        return GrepSearchResult(
            query=pattern,
            matches=(),
            total_matches=None,
            matched_files=None,
            errors=(f"invalid regex: {exc}",),
            metadata_complete=False,
        )

    search_root = _validate_workspace(Path(path), workdir)
    if not search_root.exists():
        return GrepSearchResult(
            query=pattern,
            matches=(),
            total_matches=None,
            matched_files=None,
            errors=(f"path not found: {path}",),
            metadata_complete=False,
        )

    workdir_resolved = workdir.resolve()
    matches: list[GrepMatch] = []
    total_matches = 0
    files_with_matches: set[str] = set()
    errors: list[str] = []

    if search_root.is_file():
        file_iter: Iterator[Path] = iter([search_root]) if _is_searchable(search_root) else iter([])
    else:
        file_iter = _iter_files(search_root)

    for file_path in file_iter:
        # Apply include filter
        if include and not _match_glob(file_path, include):
            continue

        lines = _read_lines(file_path)
        if lines is None:
            continue

        try:
            rel_path = file_path.resolve().relative_to(workdir_resolved).as_posix()
        except ValueError:
            rel_path = str(file_path)

        for line_num, line in enumerate(lines, start=1):
            if regex.search(line):
                total_matches += 1
                files_with_matches.add(rel_path)
                if len(matches) < max_results:
                    matches.append(GrepMatch(
                        path=rel_path,
                        line_number=line_num,
                        text=line,
                    ))

    # total_matches is the ground truth; matched_files is the set size.
    # metadata_complete stays True because we traversed the whole root
    # without an abort (the output-size cap only affects the inline
    # preview, not the counts).
    return GrepSearchResult(
        query=pattern,
        matches=tuple(matches),
        total_matches=total_matches,
        matched_files=len(files_with_matches),
        errors=tuple(errors),
        metadata_complete=True,
    )


def glob_search(
    pattern: str,
    *,
    path: str = ".",
    max_results: int = DEFAULT_MAX_GLOB_RESULTS,
    workdir: Path,
) -> str:
    """Find files matching a glob *pattern* within the workspace.

    Returns matching file paths relative to the workspace root.

    Parameters
    ----------
    pattern:
        Glob pattern for file names (e.g. ``"*.py"``, ``"test_*.py"``,
        ``"**/config/*.json"``).
    path:
        Base directory to search from, relative to *workdir*. Default ``"."``.
    max_results:
        Maximum number of paths to return. Default 100.
    workdir:
        Workspace root — all paths must stay inside this directory.

    Returns
    -------
    str
        Matching file paths, one per line, followed by a summary line.
    """
    cleaned = pattern.strip()
    if not cleaned:
        return "Error: pattern is required."

    search_root = _validate_workspace(Path(path), workdir)
    if not search_root.exists():
        return f"Error: path not found: {path}"

    workdir_resolved = workdir.resolve()
    matched: list[str] = []
    total = 0

    for file_path in _iter_files(search_root):
        if _match_glob(file_path, cleaned):
            total += 1
            if len(matched) < max_results:
                try:
                    rel = file_path.resolve().relative_to(workdir_resolved).as_posix()
                except ValueError:
                    rel = str(file_path)
                matched.append(rel)

    if not matched:
        return f"No files found matching pattern: {pattern}"

    output_lines = list(matched)
    if total > max_results:
        remaining = total - max_results
        output_lines.append(f"\n... ({remaining} more files, showing first {max_results})")
    else:
        output_lines.append(f"\n({total} files found)")

    result = "\n".join(output_lines)
    return result[:MAX_OUTPUT_CHARS]
