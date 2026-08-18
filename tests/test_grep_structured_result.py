"""test_grep_structured_result.py - Stage 2C-B3A verification.

Verifies that ``grep_search`` returns a structured ``GrepSearchResult``
with accurate counts and a clear zero-hit vs error distinction, and
that ``ToolOutputPolicy`` reads the structured fields directly instead
of parsing natural-language summary lines.

Scope (B3A — no harness wiring yet):
  - GrepSearchResult / GrepMatch dataclass shape
  - total_matches / matched_files accuracy
  - Zero-hit vs execution-error distinction
  - Legacy text format via str(result)
  - ToolOutputPolicy structured grep path (small / large / error / zero)
  - JSONL artifact payload shape
  - Workdir isolation (no process-level cwd dependency)
  - Windows-style paths and colons in match text
  - Concurrent runs with different workdirs

B3B (harness wiring) is a separate file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.code_search import (
    GrepMatch,
    GrepSearchResult,
    grep_search,
)
from agents.tool_output_policy import (
    OutputPolicyConfig,
    ToolOutputPolicy,
    _is_grep_result,
)
from agents.artifacts import ArtifactStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_workdir_with_files(files: dict[str, str]) -> Path:
    """Create a temp dir and write the given {relpath: content} files."""
    tmp = Path(tempfile.mkdtemp(prefix="grep_b3a_"))
    for rel, content in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------
class GrepSearchResultShapeTests(unittest.TestCase):
    """GrepSearchResult / GrepMatch are frozen dataclasses with the
    fields the OutputPolicy depends on."""

    def test_grep_match_frozen_fields(self):
        m = GrepMatch(path="a.py", line_number=3, text="hi")
        self.assertEqual(m.path, "a.py")
        self.assertEqual(m.line_number, 3)
        self.assertEqual(m.text, "hi")
        with self.assertRaises(Exception):
            m.path = "b.py"  # frozen

    def test_grep_search_result_frozen_fields(self):
        r = GrepSearchResult(
            query="x",
            matches=(),
            total_matches=0,
            matched_files=0,
        )
        self.assertEqual(r.query, "x")
        self.assertEqual(r.matches, ())
        self.assertEqual(r.total_matches, 0)
        self.assertEqual(r.matched_files, 0)
        self.assertEqual(r.errors, ())
        self.assertTrue(r.metadata_complete)
        with self.assertRaises(Exception):
            r.query = "y"  # frozen

    def test_is_grep_result_duck_type(self):
        r = GrepSearchResult(query="x", matches=(), total_matches=0,
                             matched_files=0)
        self.assertTrue(_is_grep_result(r))
        self.assertFalse(_is_grep_result("not a result"))
        self.assertFalse(_is_grep_result(None))
        self.assertFalse(_is_grep_result({"foo": "bar"}))


# ---------------------------------------------------------------------------
# Count accuracy
# ---------------------------------------------------------------------------
class GrepCountAccuracyTests(unittest.TestCase):
    """total_matches and matched_files must come from the real search."""

    def test_single_file_single_match(self):
        wd = _make_workdir_with_files({"a.py": "hello\nworld\n"})
        try:
            r = grep_search("hello", workdir=wd)
            self.assertEqual(r.total_matches, 1)
            self.assertEqual(r.matched_files, 1)
            self.assertEqual(len(r.matches), 1)
            self.assertEqual(r.matches[0].path, "a.py")
            self.assertEqual(r.matches[0].line_number, 1)
            self.assertEqual(r.matches[0].text, "hello")
            self.assertTrue(r.metadata_complete)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_multi_file_matches_counted(self):
        wd = _make_workdir_with_files({
            "a.py": "foo\nbar\nfoo\n",
            "b.py": "foo\nbaz\n",
            "c.txt": "nothing here\n",
        })
        try:
            r = grep_search("foo", workdir=wd)
            # 3 matches total (2 in a.py, 1 in b.py)
            self.assertEqual(r.total_matches, 3)
            self.assertEqual(r.matched_files, 2)
            self.assertTrue(r.metadata_complete)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_total_matches_exceeds_max_results(self):
        """When there are more matches than max_results, total_matches
        is the GROUND TRUTH (not just len(matches))."""
        wd = _make_workdir_with_files({
            "big.py": "\n".join(f"match{i}" for i in range(100)) + "\n",
        })
        try:
            r = grep_search("match", workdir=wd, max_results=10)
            self.assertEqual(r.total_matches, 100)
            self.assertEqual(len(r.matches), 10)
            self.assertEqual(r.matched_files, 1)
            self.assertTrue(r.metadata_complete)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Zero-hit vs error distinction
# ---------------------------------------------------------------------------
class GrepZeroHitVsErrorTests(unittest.TestCase):
    """Zero-hit: total_matches=0, errors=(), metadata_complete=True.
    Error: total_matches=None, errors non-empty, metadata_complete=False."""

    def test_zero_hit_is_not_error(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        try:
            r = grep_search("nonexistent_pattern", workdir=wd)
            self.assertEqual(r.total_matches, 0)
            self.assertEqual(r.matched_files, 0)
            self.assertEqual(r.matches, ())
            self.assertEqual(r.errors, ())
            self.assertTrue(r.metadata_complete)
            # str() produces the legacy zero-hit message
            self.assertIn("No matches found", str(r))
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_empty_pattern_is_error(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        try:
            r = grep_search("   ", workdir=wd)
            self.assertIsNone(r.total_matches)
            self.assertIsNone(r.matched_files)
            self.assertFalse(r.metadata_complete)
            self.assertGreater(len(r.errors), 0)
            self.assertIn("pattern is required", r.errors[0])
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_invalid_regex_is_error(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        try:
            r = grep_search("(unclosed", workdir=wd)
            self.assertIsNone(r.total_matches)
            self.assertFalse(r.metadata_complete)
            self.assertGreater(len(r.errors), 0)
            self.assertIn("invalid regex", r.errors[0])
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_path_not_found_is_error(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        try:
            r = grep_search("hello", path="nonexistent_dir", workdir=wd)
            self.assertIsNone(r.total_matches)
            self.assertFalse(r.metadata_complete)
            self.assertGreater(len(r.errors), 0)
            self.assertIn("path not found", r.errors[0])
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Legacy text format compatibility
# ---------------------------------------------------------------------------
class GrepLegacyTextFormatTests(unittest.TestCase):
    """str(GrepSearchResult) must reproduce the old text format so
    existing callers (and the model-facing tool result) are unaffected."""

    def test_str_hit_format(self):
        wd = _make_workdir_with_files({"a.py": "hello\nworld\n"})
        try:
            r = grep_search("hello", workdir=wd)
            s = str(r)
            self.assertIn("a.py:1:hello", s)
            self.assertIn("(1 matches in 1 files)", s)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_str_truncated_format(self):
        wd = _make_workdir_with_files({
            "big.py": "\n".join(f"match{i}" for i in range(100)) + "\n",
        })
        try:
            r = grep_search("match", workdir=wd, max_results=5)
            s = str(r)
            self.assertIn("... (95 more matches in 1 files, showing first 5)", s)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_str_zero_hit_format(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        try:
            r = grep_search("nope", workdir=wd)
            self.assertEqual(str(r), "No matches found for pattern: nope")
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_str_error_format(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        try:
            r = grep_search("(bad", workdir=wd)
            self.assertTrue(str(r).startswith("Error:"))
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Edge cases: colons in text, Windows paths
# ---------------------------------------------------------------------------
class GrepEdgeCaseTests(unittest.TestCase):
    """Match text containing colons must not break the file:line:text
    format. Windows-style paths must not break line-number parsing."""

    def test_colon_in_match_text(self):
        wd = _make_workdir_with_files({
            "a.py": "url: http://example.com\n",
        })
        try:
            r = grep_search("http", workdir=wd)
            self.assertEqual(r.total_matches, 1)
            self.assertEqual(r.matches[0].text, "url: http://example.com")
            # str() preserves the full text even with colons
            s = str(r)
            self.assertIn("url: http://example.com", s)
            # The FIRST colon after the path separates path from line,
            # the SECOND colon after the line number separates line from
            # text. Additional colons in text are preserved.
            self.assertIn("a.py:1:url: http://example.com", s)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_nested_subdir_path(self):
        """Matches in nested subdirectories report relative paths."""
        wd = _make_workdir_with_files({
            "src/deep/nested/file.py": "target line\n",
        })
        try:
            r = grep_search("target", workdir=wd)
            self.assertEqual(r.total_matches, 1)
            self.assertEqual(r.matches[0].path, "src/deep/nested/file.py")
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Workdir isolation (no process-level cwd dependency)
# ---------------------------------------------------------------------------
class GrepWorkdirIsolationTests(unittest.TestCase):
    """grep_search must use the passed-in workdir, NOT process cwd.
    Two concurrent runs with different workdirs must not interfere."""

    def test_uses_passed_workdir_not_process_cwd(self):
        wd_a = _make_workdir_with_files({"a.py": "alpha\n"})
        wd_b = _make_workdir_with_files({"b.py": "beta\n"})
        import os
        original_cwd = os.getcwd()
        try:
            # Change process cwd to somewhere unrelated
            os.chdir(wd_b)
            # Search wd_a — must find alpha, not beta
            r = grep_search("alpha", workdir=wd_a)
            self.assertEqual(r.total_matches, 1)
            self.assertEqual(r.matches[0].path, "a.py")
            self.assertEqual(r.matches[0].text, "alpha")
        finally:
            os.chdir(original_cwd)
            import shutil
            shutil.rmtree(wd_a, ignore_errors=True)
            shutil.rmtree(wd_b, ignore_errors=True)

    def test_concurrent_runs_different_workdirs(self):
        """Two runs with different workdirs must return results only
        from their own workdir."""
        wd_a = _make_workdir_with_files({"shared.py": "common\nunique_a\n"})
        wd_b = _make_workdir_with_files({"shared.py": "common\nunique_b\n"})
        try:
            r_a = grep_search("common", workdir=wd_a)
            r_b = grep_search("common", workdir=wd_b)
            self.assertEqual(r_a.total_matches, 1)
            self.assertEqual(r_b.total_matches, 1)
            # Both report their own file, no cross-contamination
            self.assertEqual(len(r_a.matches), 1)
            self.assertEqual(len(r_b.matches), 1)
        finally:
            import shutil
            shutil.rmtree(wd_a, ignore_errors=True)
            shutil.rmtree(wd_b, ignore_errors=True)


# ---------------------------------------------------------------------------
# ToolOutputPolicy structured grep path
# ---------------------------------------------------------------------------
class ToolOutputPolicyGrepTests(unittest.TestCase):
    """ToolOutputPolicy reads structured fields directly, not parsing
    natural-language summary lines."""

    def _make_store(self) -> ArtifactStore:
        return ArtifactStore(
            root_dir=Path(tempfile.mkdtemp(prefix="grep_store_")),
            session_id="b3a-test",
        )

    def test_small_grep_passes_through_as_legacy_text(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        store = self._make_store()
        try:
            r = grep_search("hello", workdir=wd)
            policy = ToolOutputPolicy(store=store)
            processed = policy.process("grep_search", r)
            self.assertFalse(processed.truncated)
            self.assertIsNone(processed.artifact)
            # Content is the legacy text format
            self.assertIn("a.py:1:hello", processed.content)
            # Metadata has structured fields
            self.assertEqual(processed.metadata["total_matches"], "1")
            self.assertEqual(processed.metadata["matched_files"], "1")
            self.assertTrue(processed.metadata["metadata_complete"])
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            shutil.rmtree(store.root_dir, ignore_errors=True)

    def test_error_grep_no_artifact(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        store = self._make_store()
        try:
            r = grep_search("(bad", workdir=wd)  # invalid regex
            policy = ToolOutputPolicy(store=store)
            processed = policy.process("grep_search", r)
            self.assertFalse(processed.truncated)
            self.assertIsNone(processed.artifact)
            self.assertTrue(processed.content.startswith("Error:"))
            self.assertIsNone(processed.metadata["total_matches"])
            self.assertFalse(processed.metadata["metadata_complete"])
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            shutil.rmtree(store.root_dir, ignore_errors=True)

    def test_zero_hit_no_artifact(self):
        wd = _make_workdir_with_files({"a.py": "hello\n"})
        store = self._make_store()
        try:
            r = grep_search("nope", workdir=wd)
            policy = ToolOutputPolicy(store=store)
            processed = policy.process("grep_search", r)
            self.assertFalse(processed.truncated)
            self.assertIsNone(processed.artifact)
            self.assertIn("No matches found", processed.content)
            self.assertEqual(processed.metadata["total_matches"], "0")
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            shutil.rmtree(store.root_dir, ignore_errors=True)

    def test_large_grep_produces_jsonl_artifact(self):
        """Large grep result produces an artifact with JSONL payload:
        first line is metadata, subsequent lines are one match per line."""
        # Generate enough matches to exceed the inline threshold.
        lines = [f"match_line_{i}" for i in range(500)]
        wd = _make_workdir_with_files({
            "big.py": "\n".join(lines) + "\n",
        })
        store = self._make_store()
        try:
            r = grep_search("match_line", workdir=wd, max_results=500)
            # Force the policy to treat this as large.
            policy = ToolOutputPolicy(
                store=store,
                config=OutputPolicyConfig(inline_max_bytes=100),
            )
            processed = policy.process("grep_search", r)
            self.assertTrue(processed.truncated)
            self.assertIsNotNone(processed.artifact)
            # Preview has the semantic summary block
            self.assertIn("[grep summary]", processed.content)
            self.assertIn("query: match_line", processed.content)
            self.assertIn("total_matches: 500", processed.content)
            self.assertIn("shown_matches:", processed.content)
            # Artifact is a JSONL file: first line is metadata
            artifact_data = store.read_by_uri(processed.artifact.artifact_uri)
            artifact_lines = artifact_data.decode("utf-8").splitlines()
            meta = json.loads(artifact_lines[0])
            self.assertEqual(meta["type"], "metadata")
            self.assertEqual(meta["total_matches"], 500)
            self.assertEqual(meta["matched_files"], 1)
            # Subsequent lines are matches
            first_match = json.loads(artifact_lines[1])
            self.assertEqual(first_match["path"], "big.py")
            self.assertIsInstance(first_match["line_number"], int)
            self.assertIn("match_line_", first_match["text"])
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            shutil.rmtree(store.root_dir, ignore_errors=True)

    def test_large_grep_preview_preserves_path_and_line(self):
        """The inline preview must retain path:line:text so the model
        can call read_file / edit_file on the shown matches."""
        wd = _make_workdir_with_files({
            "a.py": "\n".join(f"target line {i}" for i in range(300)) + "\n",
        })
        store = self._make_store()
        try:
            r = grep_search("target", workdir=wd, max_results=300)
            policy = ToolOutputPolicy(
                store=store,
                config=OutputPolicyConfig(inline_max_bytes=100),
            )
            processed = policy.process("grep_search", r)
            # Preview must contain at least one path:line:text entry
            self.assertIn("a.py:", processed.content)
            # Must contain the line number (not just the path)
            import re
            # Looking for pattern like "a.py:123:..."
            self.assertRegex(
                processed.content,
                r"a\.py:\d+:target line \d+",
            )
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            shutil.rmtree(store.root_dir, ignore_errors=True)

    def test_artifact_write_failure_degrades_safely(self):
        """If ArtifactStore write fails, the policy must degrade to a
        hard-truncated preview — no crash, metadata still preserved."""
        wd = _make_workdir_with_files({
            "big.py": "\n".join(f"match{i}" for i in range(300)) + "\n",
        })

        # A store that always fails on write.
        class _FailingStore:
            def store(self, *args, **kwargs):
                from agents.artifacts import ArtifactWriteError
                raise ArtifactWriteError("simulated failure")

        try:
            r = grep_search("match", workdir=wd, max_results=300)
            policy = ToolOutputPolicy(
                store=_FailingStore(),
                config=OutputPolicyConfig(inline_max_bytes=100),
            )
            processed = policy.process("grep_search", r)
            self.assertTrue(processed.truncated)
            self.assertIsNone(processed.artifact)
            # Still has metadata
            self.assertEqual(processed.metadata["total_matches"], "300")
            # Content is the hard-truncated preview
            self.assertIn("[grep summary]", processed.content)
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)

    def test_shown_matches_equals_preview_entries(self):
        """shown_matches in the summary must equal the number of
        path:line:text entries actually shown in the preview."""
        wd = _make_workdir_with_files({
            "a.py": "\n".join(f"hit {i}" for i in range(200)) + "\n",
        })
        store = self._make_store()
        try:
            r = grep_search("hit", workdir=wd, max_results=200)
            policy = ToolOutputPolicy(
                store=store,
                config=OutputPolicyConfig(
                    inline_max_bytes=100,
                    preview_head_lines=30,
                ),
            )
            processed = policy.process("grep_search", r)
            # Count actual match lines in the preview (lines after [matches])
            content = processed.content
            matches_section = content.split("[matches]", 1)[1]
            match_lines = [
                ln for ln in matches_section.split("\n")
                if ln.strip() and ln.startswith("a.py:")
            ]
            # Extract shown_matches from the summary
            import re
            m = re.search(r"shown_matches: (\d+)", content)
            self.assertIsNotNone(m, "shown_matches not found in summary")
            shown = int(m.group(1))
            self.assertEqual(shown, len(match_lines))
        finally:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            shutil.rmtree(store.root_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
