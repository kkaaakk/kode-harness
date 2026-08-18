"""test_grep_harness_wiring.py - Stage 2C-B3B verification.

Verifies that ``grep_search`` is wired into the real agent_loop
ToolOutputPolicy chain:

  BEFORE_TOOL_CALL
  → Profile / path-based guard (artifact root + artifact:// URI)
  → run_grep_search returns GrepSearchResult
  → ToolOutputPolicy._process_grep_structured
  → JSONL Artifact (large) or legacy text (small)
  → AFTER_TOOL_RESULT (extension sees truncated preview only)
  → FinalOutputGuard (extension patch hard-truncated)
  → model context

Coverage maps to the 15 B3B mandatory scenarios:
  1.  small grep result == legacy text                SmallGrepWiredTests
  2.  zero-hit is not a tool error                    SmallGrepWiredTests
  3.  search error -> error path, no artifact         SmallGrepWiredTests
  4.  large result -> JSONL artifact                  LargeGrepArtifactTests
  5.  model receives preview only                     LargeGrepArtifactTests
  6.  artifact metadata total/matched_files accurate  LargeGrepArtifactTests
  7.  JSONL contains all matches, path/line/text kept LargeGrepArtifactTests
  8.  AFTER_TOOL_RESULT sees controlled preview       ExtensionSeesPreviewTests
  9.  extension huge patch -> FinalOutputGuard        ExtensionSeesPreviewTests
  10. artifact write failure -> agent loop continues  ArtifactFailureTests
  11. grep physical artifact root -> Guard denies     ArtifactRootGuardTests
  12. grep artifact:// URI -> explicit reject         ArtifactRootGuardTests
  13. readonly/planning/coding profile visibility     ProfileGrepVisibilityTests
  14. concurrent runs with different workdirs         ConcurrentWorkdirTests
  15. full regression: no new failures                (deferred to full run)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"


def load_harness_module(temp_cwd: Path):
    """Load harness_core.py with mocked Anthropic/dotenv.

    Clears cached ``agents.*`` modules so WORKDIR is re-evaluated against
    the temp cwd (same pattern as test_harness_output_policy_wiring).
    """
    fake_anthropic = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = types.SimpleNamespace(create=None)

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_anthropic, "Anthropic", FakeAnthropic)
    setattr(fake_dotenv, "load_dotenv", lambda override=True: None)

    previous_anthropic = sys.modules.get("anthropic")
    previous_dotenv = sys.modules.get("dotenv")
    previous_cwd = Path.cwd()
    cached_agents = {
        k: v for k, v in sys.modules.items()
        if k == "agents" or k.startswith("agents.")
    }
    for k in cached_agents:
        sys.modules.pop(k, None)

    added_paths = []
    for path in (REPO_ROOT, REPO_ROOT / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_paths.append(text)
    spec = importlib.util.spec_from_file_location(
        "harness_core_grep_wiring_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)

    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv
    try:
        os.chdir(temp_cwd)
        os.environ.setdefault("MODEL_ID", "test-model")
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_anthropic is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = previous_anthropic
        if previous_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = previous_dotenv
        for k in list(sys.modules.keys()):
            if k == "agents" or k.startswith("agents."):
                sys.modules.pop(k, None)
        sys.modules.update(cached_agents)
        for path in added_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


class _Block:
    type = "tool_use"

    def __init__(self, name="grep_search", input_=None, id_="t1"):
        self.name = name
        self.input = input_ or {"pattern": "x"}
        self.id = id_


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


def _resp_tool_use(name="grep_search", input_=None, id_="t1"):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[_Block(name=name, input_=input_, id_=id_)],
        usage=None,
    )


def _resp_text(text="done"):
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[_Text(text)],
        usage=None,
    )


def _find_tool_result(messages, tool_use_id):
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for r in m["content"]:
                if isinstance(r, dict) and r.get("tool_use_id") == tool_use_id:
                    return r["content"]
    return None


def _seed_workspace(cwd: Path) -> None:
    """Create a small workspace with a couple of searchable files."""
    src = cwd / "src"
    src.mkdir(exist_ok=True)
    (src / "main.py").write_text(
        "import os\n"
        "def main():\n"
        "    print('hello world')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "def helper(x):\n"
        "    return x * 2\n",
        encoding="utf-8",
    )


def _seed_large_workspace(cwd: Path, *, n_files: int = 20,
                          lines_per_file: int = 200) -> None:
    """Create a workspace where the RETURNED matches exceed the
    inline_max_bytes threshold (8000) when the model requests a large
    max_results.

    Each matching line is ~60 bytes. The test calls grep with
    max_results=200, so 200 lines × 60 bytes = ~12000 bytes > 8000.
    Total matches are 20 × 200 = 4000, so search_limited is also true.
    This exercises branch 5 of _process_grep_structured:
    output_truncated + search_limited.

    The preview only shows preview_head_lines=50 lines (~3000 bytes),
    leaving ample room for the summary block + artifact ref under the
    FinalOutputGuard's 8000-byte threshold.
    """
    src = cwd / "big_src"
    src.mkdir(exist_ok=True)
    for i in range(n_files):
        body = "\n".join(f"match_line_{i}_{j}" for j in range(lines_per_file))
        (src / f"f{i:03d}.py").write_text(body + "\n", encoding="utf-8")


def _seed_search_limited_small_workspace(cwd: Path) -> None:
    """Create a workspace where search_limited is true but the RETURNED
    matches are small enough to inline (branch 4).

    1 file × 200 short matches → total_matches=200, returned_matches=50
    (max_results default), legacy text ~50 short lines < 8000 bytes.
    """
    src = cwd / "lim_src"
    src.mkdir(exist_ok=True)
    body = "\n".join(f"m{i}" for i in range(200))
    (src / "f.py").write_text(body + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Scenarios 1-3: small / zero-hit / error paths
# ---------------------------------------------------------------------------


class SmallGrepWiredTests(unittest.TestCase):
    """Small grep results flow through the policy as legacy text.

    Scenario 1: small result == legacy text (str(GrepSearchResult)).
    Scenario 2: zero-hit is a normal result, NOT a tool_error.
    Scenario 3: search error returns error string, no artifact.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        _seed_workspace(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_small_grep_legacy_text(self):
        # Scenario 1: small result, model sees str(GrepSearchResult).
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "hello", "path": "."},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Legacy text format: path:line:text
        self.assertIn("src/main.py:3:", result)
        self.assertIn("hello world", result)
        # No artifact for small results.
        self.assertNotIn("saved to artifact:", result)
        # No dataclass/dict/JSON leaking to the model.
        self.assertNotIn("GrepSearchResult", result)
        self.assertNotIn('"matches"', result)

    def test_zero_hit_is_not_tool_error(self):
        # Scenario 2: zero-hit returns "No matches found..." which is a
        # normal tool result (NOT starting with "Error:"). The policy's
        # zero-hit branch returns this string, no artifact.
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "ZZZNOMATCH", "path": "."},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("No matches found", result)
        # Must NOT be flagged as a transport-level error.
        self.assertFalse(result.startswith("Error"))
        self.assertNotIn("saved to artifact:", result)

    def test_search_error_no_artifact(self):
        # Scenario 3: invalid regex → GrepSearchResult with
        # total_matches=None + errors. Policy returns "Error: ..." and
        # does NOT create an artifact.
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "[invalid", "path": "."},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("Error"))
        # Grep artifacts use "[Returned matches saved to artifact:" (B3C).
        self.assertNotIn("saved to artifact:", result)

    def test_path_not_found_is_error_not_zero_hit(self):
        # Bonus: path-not-found is an execution error (total_matches=None),
        # distinct from zero-hit. Policy must surface "Error:".
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "x",
                                   "path": "does_not_exist_dir"},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("Error"))


# ---------------------------------------------------------------------------
# Scenarios 4-7: large result -> JSONL artifact
# ---------------------------------------------------------------------------


class LargeGrepArtifactTests(unittest.TestCase):
    """Large grep results produce a JSONL artifact and a semantic preview.

    Scenario 4: large result -> JSONL artifact created.
    Scenario 5: model receives preview only (not all matches).
    Scenario 6: artifact metadata total_matches / matched_files accurate.
    Scenario 7: JSONL contains all matches, path/line/text preserved.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        _seed_large_workspace(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_large_grep(self, session_id="sess-large"):
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "match_line_",
                                   "path": ".",
                                   "max_results": 200},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages, session_id=session_id)
        return _find_tool_result(messages, "t1")

    def test_large_grep_produces_artifact(self):
        # Scenario 4: large result must be artifacted.
        result = self._run_large_grep()
        self.assertIsNotNone(result)
        self.assertIn("Returned matches saved to artifact: artifact://", result)

    def test_model_receives_preview_not_full_matches(self):
        # Scenario 5: the model sees the [grep summary] + a bounded
        # [matches] section, NOT all 1200 match lines.
        result = self._run_large_grep()
        self.assertIsNotNone(result)
        self.assertIn("[grep summary]", result)
        self.assertIn("[matches]", result)
        self.assertIn("total_matches:", result)
        self.assertIn("matched_files:", result)
        self.assertIn("returned_matches:", result)
        self.assertIn("shown_matches:", result)
        self.assertIn("search_limited: True", result)
        # The full result has 1200 matches; the preview must be far smaller.
        # Count actual match lines (path:line:text) in the [matches] block.
        matches_block = result.split("[matches]", 1)[1]
        match_lines = [l for l in matches_block.splitlines()
                       if l.strip() and ":" in l]
        self.assertLess(len(match_lines), 200)
        # And the model must NOT see all 4000 lines inline.
        self.assertNotIn("match_line_19_199", result)

    def test_artifact_metadata_counts_accurate(self):
        # Scenario 6: read the artifact back and verify the metadata row
        # carries accurate total_matches / matched_files, and the JSONL
        # match rows count equals returned_matches. B3C also verifies
        # artifact_scope / artifact_complete semantics.
        result = self._run_large_grep(session_id="sess-meta")
        self.assertIsNotNone(result)

        import re
        match = re.search(r"artifact: (artifact://[^\]]+)\]", result)
        self.assertIsNotNone(match, f"artifact URI not found in: {result!r}")
        artifact_uri = match.group(1).strip()

        # Read the artifact content via a fresh store bound to the same
        # session. The store resolves the URI within its own session dir.
        store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="sess-meta",
        )
        data = store.read_by_uri(artifact_uri).decode("utf-8")
        lines = data.splitlines()

        # First line is the metadata JSON.
        meta = json.loads(lines[0])
        self.assertEqual(meta["type"], "metadata")
        self.assertEqual(meta["query"], "match_line_")
        # 20 files × 200 lines = 4000 total matches (ground truth).
        self.assertEqual(meta["total_matches"], 4000)
        self.assertEqual(meta["matched_files"], 20)
        # returned_matches = 200 (model requested max_results=200).
        self.assertEqual(meta["returned_matches"], 200)
        # artifact_matches must equal returned_matches (we only stored
        # what grep_search returned, not the full 4000).
        self.assertEqual(meta["artifact_matches"], 200)

        # B3C: artifact_scope / artifact_complete semantics.
        self.assertEqual(meta["artifact_scope"], "returned_matches")
        self.assertFalse(meta["artifact_complete"])  # NOT all 4000
        self.assertTrue(meta["search_limited"])
        self.assertTrue(meta["metadata_complete"])

        # Remaining lines are match objects. The count MUST equal
        # returned_matches (metadata row ↔ actual match rows consistency).
        match_objs = [json.loads(l) for l in lines[1:]]
        self.assertEqual(len(match_objs), meta["returned_matches"])
        # Each match must have path / line_number / text.
        for m in match_objs:
            self.assertIn("path", m)
            self.assertIn("line_number", m)
            self.assertIn("text", m)
            self.assertTrue(m["path"].startswith("big_src/"))
            self.assertIsInstance(m["line_number"], int)
            self.assertIn("match_line_", m["text"])

    def test_jsonl_preserves_path_line_text(self):
        # Scenario 7: spot-check a few JSONL entries to confirm the
        # path/line_number/text triple is intact and parseable.
        result = self._run_large_grep(session_id="sess-jsonl")
        self.assertIsNotNone(result)

        import re
        match = re.search(r"artifact: (artifact://[^\]]+)\]", result)
        artifact_uri = match.group(1).strip()

        store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="sess-jsonl",
        )
        data = store.read_by_uri(artifact_uri).decode("utf-8")
        lines = data.splitlines()

        # Verify first and last match entries are well-formed.
        first_match = json.loads(lines[1])
        last_match = json.loads(lines[-1])
        for m in (first_match, last_match):
            self.assertTrue(m["path"].endswith(".py"))
            self.assertGreater(m["line_number"], 0)
            self.assertTrue(m["text"].startswith("match_line_"))


# ---------------------------------------------------------------------------
# Scenarios 8-9: extension sees preview, final guard limits patch
# ---------------------------------------------------------------------------


class ExtensionSeesPreviewTests(unittest.TestCase):
    """AFTER_TOOL_RESULT sees the truncated preview, not the raw result.
    A huge extension patch is hard-truncated by FinalOutputGuard."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        _seed_large_workspace(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_after_tool_result_sees_preview(self):
        # Scenario 8: register an extension that captures what it sees.
        from agents.types.events import Event, HookResult, Priority

        seen = {}

        def _capture(ctx):
            seen["tool_result"] = ctx.get("tool_result", "")
            return None  # no patch

        self.module.EXTENSIONS.on(
            Event.AFTER_TOOL_RESULT, _capture, priority=Priority.NORMAL
        )
        try:
            responses = [
                _resp_tool_use(name="grep_search",
                               input_={"pattern": "match_line_",
                                       "path": ".",
                                       "max_results": 200},
                               id_="t1"),
                _resp_text(),
            ]
            self.module.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "grep"}]
            self.module.agent_loop(messages)
        finally:
            self.module.EXTENSIONS._handlers[Event.AFTER_TOOL_RESULT] = []

        # The extension must have seen the truncated preview, not 4000 lines.
        seen_result = seen.get("tool_result", "")
        self.assertIn("[grep summary]", seen_result)
        self.assertIn("Returned matches saved to artifact:", seen_result)
        # Must NOT contain the full match set.
        self.assertNotIn("match_line_19_199", seen_result)

    def test_huge_extension_patch_hard_truncated(self):
        # Scenario 9: extension returns a 50KB patch; FinalOutputGuard
        # must hard-truncate it.
        from agents.types.events import Event, HookResult, Priority

        huge_patch = "Z" * 50000

        def _patch(ctx):
            return HookResult(tool_result_patch={"content": huge_patch})

        self.module.EXTENSIONS.on(
            Event.AFTER_TOOL_RESULT, _patch, priority=Priority.NORMAL
        )
        try:
            responses = [
                _resp_tool_use(name="grep_search",
                               input_={"pattern": "match_line_",
                                       "path": ".",
                                       "max_results": 200},
                               id_="t1"),
                _resp_text(),
            ]
            self.module.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "grep"}]
            self.module.agent_loop(messages)
            result = _find_tool_result(messages, "t1")
        finally:
            self.module.EXTENSIONS._handlers[Event.AFTER_TOOL_RESULT] = []

        self.assertIsNotNone(result)
        self.assertLess(len(result), 10000)
        self.assertIn("[final output guard applied]", result)


# ---------------------------------------------------------------------------
# Scenario 10: artifact write failure -> agent loop continues
# ---------------------------------------------------------------------------


class ArtifactFailureTests(unittest.TestCase):
    """If ArtifactStore.store raises, the agent loop must not crash and
    the model receives a hard-truncated preview (no artifact ref)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        _seed_large_workspace(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_artifact_write_failure_continues(self):
        AWE = self.module.ArtifactWriteError
        _real_store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="broken-grep-session",
        )

        class BrokenStore:
            _session_id = "broken-grep-session"

            def __init__(self, inner):
                self._inner = inner

            def store(self, *args, **kwargs):
                raise AWE("disk full simulated")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        broken = BrokenStore(_real_store)

        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "match_line_",
                                   "path": ".",
                                   "max_results": 200},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        # Must NOT raise.
        self.module.agent_loop(messages, artifact_store=broken)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # No artifact reference (write failed).
        self.assertNotIn("saved to artifact:", result)
        # Preview content still present (hard-truncated).
        self.assertIn("[grep summary]", result)


# ---------------------------------------------------------------------------
# Scenarios 11-12: path-based guard (artifact root + artifact:// URI)
# ---------------------------------------------------------------------------


class ArtifactRootGuardTests(unittest.TestCase):
    """grep_search cannot search the physical artifact root, and
    ``artifact://`` URIs are rejected as a search path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        self.artifact_root = self.module._ARTIFACT_ROOT

    def tearDown(self):
        self._tmp.cleanup()

    def _run_one(self, tool_input, session_id="sess-guard"):
        responses = [
            _resp_tool_use(name="grep_search", input_=tool_input, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages, session_id=session_id)
        return _find_tool_result(messages, "t1")

    def test_grep_physical_artifact_root_denied(self):
        # Scenario 11: grep with path = artifact root absolute path.
        # Seed an artifact first so the path genuinely exists.
        big = "\n".join(f"line {i}" for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big)
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="seed"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        seed_msgs = [{"role": "user", "content": "seed"}]
        self.module.agent_loop(seed_msgs, session_id="sess-guard")

        result = self._run_one(
            {"pattern": "line", "path": str(self.artifact_root)},
            session_id="sess-guard",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)
        self.assertNotIn("line 500", result)

    def test_grep_artifact_uri_rejected(self):
        # Scenario 12: artifact:// URI as grep path must be rejected
        # with a clear message pointing to read_file.
        result = self._run_one(
            {"pattern": "x", "path": "artifact://abc123"},
            session_id="sess-guard",
        )
        self.assertIsNotNone(result)
        self.assertIn("artifact://", result)
        self.assertIn("read_file", result)
        # Must NOT be treated as a normal search (no "No matches found",
        # no match lines like path:line:text).
        self.assertNotIn("No matches found", result)
        # The result should be an explicit rejection, not a search result.
        self.assertTrue(result.startswith("Error:"))


# ---------------------------------------------------------------------------
# Scenario 13: profile visibility
# ---------------------------------------------------------------------------


class ProfileGrepVisibilityTests(unittest.TestCase):
    """grep_search visibility is unchanged across readonly/planning/coding
    profiles. (This re-asserts the existing ToolRegistry contract; B3B
    must not alter it.)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def _profile_tool_names(self, profile):
        tools = self.module.TOOL_REGISTRY.resolve(profile=profile)
        return {t["name"] for t in tools if isinstance(t, dict)}

    def test_grep_visible_in_coding_profile(self):
        names = self._profile_tool_names("coding")
        self.assertIn("grep_search", names)

    def test_grep_visible_in_default_profile(self):
        names = self._profile_tool_names(None)
        self.assertIn("grep_search", names)

    def test_grep_visible_in_readonly_profile(self):
        # B3C: readonly profile must still expose grep_search.
        names = self._profile_tool_names("readonly")
        self.assertIn("grep_search", names)

    def test_grep_visible_in_planning_profile(self):
        # B3C: planning profile must still expose grep_search.
        names = self._profile_tool_names("planning")
        self.assertIn("grep_search", names)

    def test_grep_visibility_unchaged_across_profiles(self):
        # grep_search should be present in the same set of profiles as
        # before B3B — namely all profiles that expose path-based tools.
        default_names = self._profile_tool_names(None)
        coding_names = self._profile_tool_names("coding")
        readonly_names = self._profile_tool_names("readonly")
        planning_names = self._profile_tool_names("planning")
        # All four must contain grep_search.
        for label, names in [
            ("default", default_names),
            ("coding", coding_names),
            ("readonly", readonly_names),
            ("planning", planning_names),
        ]:
            self.assertIn("grep_search", names,
                          f"grep_search missing from {label} profile")


# ---------------------------------------------------------------------------
# Scenario 14: concurrent runs with different workdirs
# ---------------------------------------------------------------------------


class ConcurrentWorkdirTests(unittest.TestCase):
    """Two agent_loop calls with different workdirs must not pollute each
    other's grep results. This re-verifies the B3A workdir-isolation fix
    at the harness level."""

    def setUp(self):
        self._tmp_a = tempfile.TemporaryDirectory()
        self._tmp_b = tempfile.TemporaryDirectory()
        self.cwd_a = Path(self._tmp_a.name)
        self.cwd_b = Path(self._tmp_b.name)
        # Workspace A: contains "alpha"
        (self.cwd_a / "a.py").write_text("alpha_marker\n", encoding="utf-8")
        # Workspace B: contains "beta"
        (self.cwd_b / "b.py").write_text("beta_marker\n", encoding="utf-8")

    def tearDown(self):
        self._tmp_a.cleanup()
        self._tmp_b.cleanup()

    def test_concurrent_runs_do_not_cross_pollute(self):
        # Run A in cwd_a, then run B in cwd_b. Because load_harness_module
        # binds WORKDIR at import time, we load two separate module copies.
        module_a = load_harness_module(self.cwd_a)
        module_b = load_harness_module(self.cwd_b)

        # Run A: search for alpha_marker. Must find it in a.py.
        responses_a = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "alpha_marker", "path": "."},
                           id_="ta"),
            _resp_text(),
        ]
        module_a.client.messages.create = lambda **_: responses_a.pop(0)
        msgs_a = [{"role": "user", "content": "grep"}]
        module_a.agent_loop(msgs_a)
        result_a = _find_tool_result(msgs_a, "ta")
        self.assertIsNotNone(result_a)
        self.assertIn("a.py", result_a)
        self.assertIn("alpha_marker", result_a)
        self.assertNotIn("b.py", result_a)

        # Run B: search for beta_marker. Must find it in b.py, NOT a.py.
        responses_b = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "beta_marker", "path": "."},
                           id_="tb"),
            _resp_text(),
        ]
        module_b.client.messages.create = lambda **_: responses_b.pop(0)
        msgs_b = [{"role": "user", "content": "grep"}]
        module_b.agent_loop(msgs_b)
        result_b = _find_tool_result(msgs_b, "tb")
        self.assertIsNotNone(result_b)
        self.assertIn("b.py", result_b)
        self.assertIn("beta_marker", result_b)
        self.assertNotIn("a.py", result_b)


# ---------------------------------------------------------------------------
# Stage 2C-B3C: search_limited vs output_truncated semantics
# ---------------------------------------------------------------------------


class SearchLimitedSemanticsTests(unittest.TestCase):
    """B3C scenario 1: search_limited + small output → no artifact.

    When total_matches > returned_matches but the returned matches fit
    inline, the policy must NOT create an artifact. The model receives
    the legacy text + an explicit "[search limited]" notice telling it
    the remaining matches were not materialized.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        _seed_search_limited_small_workspace(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_search_limited_small_output_no_artifact(self):
        # 200 total matches, 50 returned, each ~4 bytes → ~200 bytes
        # inline, well under 8000. No artifact should be created.
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "m", "path": "."},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # No artifact — this is the whole point of B3C.
        self.assertNotIn("saved to artifact:", result)
        # Must contain the explicit "not materialized" notice.
        self.assertIn("[search limited]", result)
        self.assertIn("not materialized", result)
        self.assertIn("200", result)  # total_matches
        self.assertIn("50", result)   # returned_matches


class OutputTruncatedArtifactScopeTests(unittest.TestCase):
    """B3C scenarios 2-5: when returned matches themselves exceed the
    inline threshold, an artifact IS created — but its metadata must
    mark ``artifact_scope="returned_matches"`` and
    ``artifact_complete=False`` when search_limited. The model-facing
    text must NOT say "full output saved".
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        _seed_large_workspace(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, session_id):
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "match_line_",
                                   "path": ".",
                                   "max_results": 200},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages, session_id=session_id)
        return _find_tool_result(messages, "t1")

    def test_artifact_metadata_has_scope_and_incomplete(self):
        # Scenario 4: artifact metadata carries artifact_scope +
        # artifact_complete=false (because search_limited).
        result = self._run(session_id="sess-scope")
        self.assertIsNotNone(result)
        import re
        match = re.search(r"artifact: (artifact://[^\]]+)\]", result)
        self.assertIsNotNone(match)
        store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="sess-scope",
        )
        data = store.read_by_uri(match.group(1).strip()).decode("utf-8")
        meta = json.loads(data.splitlines()[0])
        self.assertEqual(meta["artifact_scope"], "returned_matches")
        self.assertFalse(meta["artifact_complete"])
        self.assertTrue(meta["search_limited"])

    def test_model_text_does_not_say_full_output_saved(self):
        # Scenario 5: the model-facing text must NOT claim the full
        # output was saved. It must say "Returned matches saved" and
        # explicitly state the remaining matches are NOT in the artifact.
        result = self._run(session_id="sess-wording")
        self.assertIsNotNone(result)
        # Must NOT use the misleading "Full output saved" wording.
        self.assertNotIn("Full output saved", result)
        # Must use the accurate "Returned matches saved" wording.
        self.assertIn("Returned matches saved to artifact:", result)
        # Must explicitly state artifact_complete is False.
        self.assertIn("artifact_complete: False", result)
        # Must mention the remaining matches are NOT in the artifact.
        self.assertIn("NOT in this artifact", result)

    def test_inline_summary_states_remaining_not_materialized(self):
        # Scenario 3: when search_limited but output fits inline, the
        # summary must explicitly say the remaining matches were not
        # materialized. (Uses the small-output workspace.)
        tmp2 = tempfile.TemporaryDirectory()
        try:
            cwd2 = Path(tmp2.name)
            module2 = load_harness_module(cwd2)
            _seed_search_limited_small_workspace(cwd2)
            responses = [
                _resp_tool_use(name="grep_search",
                               input_={"pattern": "m", "path": "."},
                               id_="t1"),
                _resp_text(),
            ]
            module2.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "grep"}]
            module2.agent_loop(messages)
            result = _find_tool_result(messages, "t1")
            self.assertIsNotNone(result)
            self.assertIn("not materialized", result)
            self.assertIn("remaining", result.lower())
        finally:
            tmp2.cleanup()


class FullSearchResultArtifactTests(unittest.TestCase):
    """B3C.1: large output + NO search limiting → complete artifact.

    When total_matches == returned_matches (search_limited=False) but
    the returned matches exceed the inline threshold, the artifact
    contains EVERY match. Metadata must mark
    artifact_scope="full_search_result" + artifact_complete=True, and
    the model-facing text must say "Full search result saved".
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        # 1 file × 200 lines × ~60 bytes/line = ~12000 bytes > 8000.
        # total_matches=200, returned_matches=200 (max_results=200).
        # search_limited=False.
        src = self.cwd / "complete_src"
        src.mkdir(exist_ok=True)
        body = "\n".join(f"complete_line_{i}" for i in range(200))
        (src / "only.py").write_text(body + "\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, session_id):
        responses = [
            _resp_tool_use(name="grep_search",
                           input_={"pattern": "complete_line_",
                                   "path": ".",
                                   "max_results": 200},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "grep"}]
        self.module.agent_loop(messages, session_id=session_id)
        return _find_tool_result(messages, "t1")

    def test_large_output_complete_artifact_metadata(self):
        # Scenario 1: artifact_complete=True, artifact_scope=full_search_result.
        result = self._run(session_id="sess-full-meta")
        self.assertIsNotNone(result)
        import re
        match = re.search(r"artifact: (artifact://[^\]]+)\]", result)
        self.assertIsNotNone(match)
        store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="sess-full-meta",
        )
        data = store.read_by_uri(match.group(1).strip()).decode("utf-8")
        lines = data.splitlines()
        meta = json.loads(lines[0])
        self.assertEqual(meta["artifact_scope"], "full_search_result")
        self.assertTrue(meta["artifact_complete"])
        self.assertFalse(meta["search_limited"])
        self.assertEqual(meta["total_matches"], 200)
        self.assertEqual(meta["returned_matches"], 200)
        self.assertEqual(meta["artifact_matches"], 200)
        # Every match is in the artifact.
        match_objs = [json.loads(l) for l in lines[1:]]
        self.assertEqual(len(match_objs), 200)

    def test_large_output_complete_artifact_wording(self):
        # Scenario 3: model text says "Full search result saved" and
        # "all 200 matches are included", NOT "Returned matches saved".
        result = self._run(session_id="sess-full-wording")
        self.assertIsNotNone(result)
        self.assertIn("Full search result saved to artifact:", result)
        self.assertIn("artifact_complete: True", result)
        self.assertIn("all 200 matches are included", result)
        # Must NOT use the partial-artifact wording.
        self.assertNotIn("Returned matches saved to artifact:", result)
        self.assertNotIn("NOT in this artifact", result)

    def test_large_output_partial_artifact_wording_contrast(self):
        # Scenario 2 + 3: contrast with search_limited=True case to
        # confirm the two wordings are distinct. Uses the large
        # workspace (4000 total / 200 returned).
        tmp2 = tempfile.TemporaryDirectory()
        try:
            cwd2 = Path(tmp2.name)
            module2 = load_harness_module(cwd2)
            _seed_large_workspace(cwd2)
            responses = [
                _resp_tool_use(name="grep_search",
                               input_={"pattern": "match_line_",
                                       "path": ".",
                                       "max_results": 200},
                               id_="t1"),
                _resp_text(),
            ]
            module2.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "grep"}]
            module2.agent_loop(messages, session_id="sess-partial")
            result = _find_tool_result(messages, "t1")
            self.assertIsNotNone(result)
            self.assertIn("Returned matches saved to artifact:", result)
            self.assertIn("artifact_complete: False", result)
            self.assertIn("NOT in this artifact", result)
            self.assertNotIn("Full search result saved", result)
        finally:
            tmp2.cleanup()


if __name__ == "__main__":
    unittest.main()
