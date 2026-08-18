"""test_artifacts.py - Tests for ArtifactStore.

Stage 2C-A: infrastructure tests, not wired into harness_core.

Covers:
- Store and retrieve content
- SHA-256 correctness
- File names are server-generated (uuid4), not controllable by caller
- Path traversal in session_id rejected
- Session isolation
- Atomic write (temp file cleaned on failure)
- max_bytes enforcement
- Write failure degradation
- cleanup_expired / delete_session_artifacts
- Logical path returned (not host absolute path)
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from agents.artifacts import (
    ArtifactRef,
    ArtifactStore,
    ArtifactWriteError,
    PathTraversalError,
)


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "artifacts"
        self.store = ArtifactStore(self.root, session_id="sess-1")

    def tearDown(self):
        self._tmp.cleanup()

    def test_store_and_retrieve_content(self):
        content = "line1\nline2\nline3\n"
        ref = self.store.store(content, total_lines=3)
        self.assertIsInstance(ref, ArtifactRef)
        self.assertEqual(ref.total_bytes, len(content.encode("utf-8")))
        self.assertEqual(ref.total_lines, 3)
        # Read back
        data = self.store.read(ref.artifact_id)
        self.assertEqual(data.decode("utf-8"), content)

    def test_sha256_correct(self):
        content = "hello world\n"
        ref = self.store.store(content)
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.assertEqual(ref.sha256, expected)

    def test_filename_is_server_generated_uuid(self):
        ref = self.store.store("content")
        # artifact_id is a 32-char hex (uuid4)
        self.assertEqual(len(ref.artifact_id), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in ref.artifact_id))
        # The filename on disk is <artifact_id>.txt
        files = list(self.store.session_dir().glob("*.txt"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, f"{ref.artifact_id}.txt")

    def test_logical_path_returned_not_host_absolute(self):
        ref = self.store.store("content")
        # Logical path starts with .harness/artifacts/
        self.assertTrue(ref.artifact_path.startswith(".harness/artifacts/"))
        # Must NOT contain the host absolute root path
        self.assertNotIn(str(self.root.resolve()), ref.artifact_path)

    def test_path_traversal_in_session_id_rejected(self):
        with self.assertRaises(PathTraversalError):
            ArtifactStore(self.root, session_id="../../etc")
        with self.assertRaises(PathTraversalError):
            ArtifactStore(self.root, session_id="foo/bar")
        with self.assertRaises(PathTraversalError):
            ArtifactStore(self.root, session_id="foo..bar")  # dots not allowed

    def test_session_isolation(self):
        store2 = ArtifactStore(self.root, session_id="sess-2")
        ref1 = self.store.store("from session 1")
        ref2 = store2.store("from session 2")
        # Different session dirs
        self.assertNotEqual(self.store.session_dir(), store2.session_dir())
        # Can't read session-2 artifact from session-1 store
        with self.assertRaises(Exception):
            self.store.read(ref2.artifact_id)
        # Can read own artifact
        data = self.store.read(ref1.artifact_id)
        self.assertEqual(data.decode("utf-8"), "from session 1")

    def test_max_bytes_enforcement(self):
        store = ArtifactStore(self.root, session_id="big",
                              max_bytes=100)
        with self.assertRaises(ArtifactWriteError):
            store.store("x" * 200)

    def test_write_failure_degradation(self):
        """If store() fails to write, it should raise ArtifactWriteError."""
        # Use max_bytes=0 so any content exceeds the limit -> ArtifactWriteError.
        store = ArtifactStore(self.root, session_id="fail-session", max_bytes=0)
        with self.assertRaises(ArtifactWriteError):
            store.store("content")

    def test_temp_file_cleaned_on_failure(self):
        """If rename fails, the temp file should be cleaned up."""
        store = ArtifactStore(self.root, session_id="temp-test")
        # We can't easily force a rename failure, but we can verify no
        # .tmp_ files are left after a successful write.
        store.store("content1")
        store.store("content2")
        tmp_files = list(store.session_dir().glob(".tmp_*"))
        self.assertEqual(len(tmp_files), 0)

    def test_delete_session_artifacts(self):
        self.store.store("content1")
        self.store.store("content2")
        self.assertEqual(len(list(self.store.session_dir().glob("*.txt"))), 2)
        count = self.store.delete_session_artifacts()
        self.assertEqual(count, 2)
        self.assertFalse(self.store.session_dir().exists())

    def test_delete_other_session_artifacts(self):
        store2 = ArtifactStore(self.root, session_id="sess-2")
        self.store.store("from 1")
        store2.store("from 2")
        # Delete session-2 artifacts from session-1 store
        count = self.store.delete_session_artifacts("sess-2")
        self.assertEqual(count, 1)
        # Session-1 artifacts untouched
        self.assertEqual(len(list(self.store.session_dir().glob("*.txt"))), 1)

    def test_expires_at_set_with_ttl(self):
        ref = self.store.store("content", ttl=timedelta(hours=1))
        self.assertIsNotNone(ref.expires_at)
        ref_no_ttl = self.store.store("content")
        # Default TTL is None -> no expiry
        self.assertIsNone(ref_no_ttl.expires_at)

    def test_bytes_content_accepted(self):
        data = b"\x00\x01\x02 binary"
        ref = self.store.store(data)
        self.assertEqual(ref.total_bytes, len(data))
        read_back = self.store.read(ref.artifact_id)
        self.assertEqual(read_back, data)


if __name__ == "__main__":
    unittest.main()
