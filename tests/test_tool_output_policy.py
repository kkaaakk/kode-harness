"""test_tool_output_policy.py - Tests for ToolOutputPolicy.

Stage 2C-A: infrastructure tests, not wired into harness_core.

Covers:
- Small output passes through unchanged
- Large output triggers artifact
- read_file preview has head/tail with line numbers
- bash preserves exit code mention
- grep preserves match summary
- Unknown tool uses generic formatter
- No store: degrades to hard truncation
- Write failure: degrades gracefully
- Untruncated output: no artifact created
- AFTER_TOOL_RESULT sees processed (truncated) content, not raw
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.artifacts import ArtifactStore, ArtifactWriteError
from agents.tool_output_policy import (
    OutputPolicyConfig,
    ProcessedToolOutput,
    ToolOutputPolicy,
    resolve_existing_artifact,
)


def _make_policy(
    tmpdir: Path,
    inline_max_bytes: int = 100,
    inline_max_lines: int = 10,
    with_store: bool = True,
) -> ToolOutputPolicy:
    config = OutputPolicyConfig(
        inline_max_bytes=inline_max_bytes,
        inline_max_lines=inline_max_lines,
        preview_head_lines=3,
        preview_tail_lines=2,
        artifact_max_bytes=10 * 1024 * 1024,
    )
    store = None
    if with_store:
        store = ArtifactStore(tmpdir / "artifacts", session_id="test-sess")
    return ToolOutputPolicy(store=store, config=config)


class SmallOutputTests(unittest.TestCase):
    """Small outputs pass through unchanged."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_small_string_unchanged(self):
        result = self.policy.process("read_file", "hello\nworld\n")
        self.assertFalse(result.truncated)
        self.assertIsNone(result.artifact)
        self.assertEqual(result.content, "hello\nworld\n")

    def test_small_bash_unchanged(self):
        result = self.policy.process("bash", "output line\n")
        self.assertFalse(result.truncated)
        self.assertEqual(result.content, "output line\n")

    def test_small_grep_unchanged(self):
        result = self.policy.process("grep_search", "file.py:1:match\n")
        self.assertFalse(result.truncated)
        self.assertEqual(result.content, "file.py:1:match\n")


class LargeOutputTests(unittest.TestCase):
    """Large outputs get artifacted with structured preview."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_large_output_triggers_artifact(self):
        big = "line\n" * 500  # 2500 bytes, 500 lines
        result = self.policy.process("read_file", big)
        self.assertTrue(result.truncated)
        self.assertIsNotNone(result.artifact)
        self.assertIn("[Full output saved to artifact:", result.content)

    def test_large_output_by_bytes_triggers_artifact(self):
        # Few lines but huge bytes
        big = "x" * 500  # 500 bytes > 100 limit
        result = self.policy.process("read_file", big)
        self.assertTrue(result.truncated)
        self.assertIsNotNone(result.artifact)

    def test_artifact_content_byte_identical(self):
        big = "line\n" * 500
        result = self.policy.process("read_file", big)
        self.assertIsNotNone(result.artifact)
        data = self.policy._store.read(result.artifact.artifact_id)
        self.assertEqual(data.decode("utf-8"), big)

    def test_no_artifact_for_small_output(self):
        result = self.policy.process("read_file", "small")
        self.assertIsNone(result.artifact)


class ReadFileFormatterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_preview_has_head_and_tail_with_line_numbers(self):
        lines = [f"line {i}" for i in range(1, 101)]  # 100 lines
        big = "\n".join(lines)
        result = self.policy.process("read_file", big)
        self.assertTrue(result.truncated)
        # Preview should contain line 1 and line 100
        self.assertIn("1 | line 1", result.content)
        self.assertIn("100 | line 100", result.content)
        # Should contain omission marker
        self.assertIn("lines omitted", result.content)

    def test_preview_preserves_original_line_numbers(self):
        """Line numbers in preview must match the original file, not be
        renumbered from 1."""
        lines = [f"line {i}" for i in range(1, 201)]
        big = "\n".join(lines)
        result = self.policy.process("read_file", big)
        self.assertIn("1 | line 1", result.content)
        self.assertIn("200 | line 200", result.content)
        # Tail should start near line 198 (head=3, tail=2 -> last 2 lines)
        self.assertIn("199 | line 199", result.content)


class BashFormatterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_bash_preserves_exit_code_in_metadata(self):
        big = "line\nexit_code: 1\n" + "x\n" * 500
        result = self.policy.process("bash", big)
        self.assertTrue(result.truncated)
        self.assertIn("exit_code", result.metadata)
        self.assertIn("1", result.metadata["exit_code"])

    def test_bash_preview_has_head_and_tail(self):
        lines = [f"out {i}" for i in range(1, 101)]
        big = "\n".join(lines)
        result = self.policy.process("bash", big)
        self.assertTrue(result.truncated)
        self.assertIn("out 1", result.content)
        self.assertIn("out 100", result.content)


class GrepFormatterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_grep_preserves_match_summary_in_metadata(self):
        lines = [f"file.py:{i}:match" for i in range(1, 101)]
        lines.append("(1834 matches in 92 files)")
        big = "\n".join(lines)
        result = self.policy.process("grep_search", big)
        self.assertTrue(result.truncated)
        self.assertIn("summary", result.metadata)
        self.assertIn("1834", result.metadata["summary"])


class GenericFormatterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_unknown_tool_uses_generic_formatter(self):
        big = "\n".join(f"line {i}" for i in range(100))
        result = self.policy.process("some_unknown_tool", big)
        self.assertTrue(result.truncated)
        self.assertIn("line 0", result.content)
        self.assertIn("line 99", result.content)


class FailureDegradationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_store_degrades_to_hard_truncation(self):
        policy = _make_policy(Path(self._tmp.name), with_store=False)
        big = "x" * 500
        result = policy.process("read_file", big)
        self.assertTrue(result.truncated)
        self.assertIsNone(result.artifact)
        self.assertIn("hard-truncated", result.content)

    def test_write_failure_degrades_gracefully(self):
        """If the store raises ArtifactWriteError, the policy should
        degrade to hard truncation, not crash."""
        policy = _make_policy(Path(self._tmp.name))

        # Replace the store with a broken one.
        class BrokenStore:
            def store(self, *args, **kwargs):
                raise ArtifactWriteError("disk full")

        policy._store = BrokenStore()
        big = "x" * 500
        result = policy.process("read_file", big)
        self.assertTrue(result.truncated)
        self.assertIsNone(result.artifact)
        # Should still have preview content (hard-truncated)
        self.assertTrue(len(result.content) > 0)


class ExtensionOrderingTests(unittest.TestCase):
    """The core Output Policy runs BEFORE AFTER_TOOL_RESULT extensions.

    This means extensions see the already-truncated content, not the raw
    huge output. This test verifies the ProcessedToolOutput.content is what
    an extension would receive.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.policy = _make_policy(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_extension_sees_truncated_content_not_raw(self):
        # Use multi-line content so each line is short. The preview will
        # contain head + tail lines, not the full 500-line raw output.
        big = "\n".join(f"line {i}" for i in range(500))
        result = self.policy.process("read_file", big)
        # An extension receiving result.content should NOT see the raw 500
        # lines. It should see a preview + artifact reference instead.
        self.assertNotEqual(result.content, big)
        self.assertTrue(result.truncated)
        self.assertIsNotNone(result.artifact)
        self.assertIn("[Full output saved to artifact:", result.content)
        # The raw content has 500 lines; the preview should have far fewer.
        raw_line_count = big.count("\n") + 1
        content_line_count = result.content.count("\n") + 1
        self.assertLess(content_line_count, raw_line_count)
        # Specific middle lines should be omitted.
        self.assertNotIn("line 250", result.content)


class ResolveExistingArtifactTests(unittest.TestCase):
    """Verify artifact path resolution is strict: only real files inside
    the current session's artifact directory are treated as artifacts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tmp.name)
        self.artifact_root = self.workdir / ".harness" / "artifacts"
        self.session_id = "sess-1"
        # Create a real artifact
        self.store = ArtifactStore(self.artifact_root, session_id=self.session_id)
        self.ref = self.store.store("artifact content line1\nline2\n")
        self.artifact_logical = f".harness/artifacts/{self.session_id}/{self.ref.artifact_id}.txt"

    def tearDown(self):
        self._tmp.cleanup()

    def test_real_artifact_resolves(self):
        result = resolve_existing_artifact(
            self.artifact_logical,
            workdir=self.workdir,
            artifact_root=self.artifact_root,
            session_id=self.session_id,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.is_file())

    def test_normal_file_not_treated_as_artifact(self):
        # A normal file whose path HAPPENS to contain the string
        # ".harness/artifacts" but is NOT in the artifact dir.
        fake_path = "some/.harness/artifacts/fake.txt"
        result = resolve_existing_artifact(
            fake_path,
            workdir=self.workdir,
            artifact_root=self.artifact_root,
            session_id=self.session_id,
        )
        self.assertIsNone(result)

    def test_path_traversal_rejected(self):
        # Try to escape artifact dir via ..
        traversal = f".harness/artifacts/{self.session_id}/../../secret.txt"
        result = resolve_existing_artifact(
            traversal,
            workdir=self.workdir,
            artifact_root=self.artifact_root,
            session_id=self.session_id,
        )
        self.assertIsNone(result)

    def test_cross_session_rejected(self):
        # Artifact from session-2 should not be readable as session-1's artifact
        store2 = ArtifactStore(self.artifact_root, session_id="sess-2")
        ref2 = store2.store("other session")
        cross_path = f".harness/artifacts/sess-2/{ref2.artifact_id}.txt"
        result = resolve_existing_artifact(
            cross_path,
            workdir=self.workdir,
            artifact_root=self.artifact_root,
            session_id=self.session_id,  # sess-1
        )
        self.assertIsNone(result)

    def test_nonexistent_file_rejected(self):
        result = resolve_existing_artifact(
            f".harness/artifacts/{self.session_id}/nonexistent.txt",
            workdir=self.workdir,
            artifact_root=self.artifact_root,
            session_id=self.session_id,
        )
        self.assertIsNone(result)

    def test_symlink_escape_rejected(self):
        """A symlink inside the artifact dir pointing outside must not
        be treated as a valid artifact (resolve() follows symlinks)."""
        # Create a symlink in the session dir pointing to /etc/passwd or similar
        link_name = f"evil-{self.ref.artifact_id}-link.txt"
        link_path = self.artifact_root / self.session_id / link_name
        try:
            link_path.symlink_to(self.workdir / "secret.txt")
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation not supported on this platform")
        (self.workdir / "secret.txt").write_text("secret")
        link_logical = f".harness/artifacts/{self.session_id}/{link_name}"
        result = resolve_existing_artifact(
            link_logical,
            workdir=self.workdir,
            artifact_root=self.artifact_root,
            session_id=self.session_id,
        )
        # The symlink resolves to /workdir/secret.txt which is NOT inside
        # the session_root, so it must be rejected.
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
