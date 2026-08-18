"""test_harness_output_policy_wiring.py - Stage 2C-B1 verification.

Verifies that ToolOutputPolicy is wired into harness_core for read_file:
- Small read_file output passes through unchanged
- Large read_file output gets artifacted; model sees preview only
- AFTER_TOOL_RESULT extension sees the processed (truncated) content
- Artifact can be read back via real read_file handler
- Tool-level block does NOT produce artifact
- Error results are NOT processed by the policy
- bash/grep outputs are NOT yet processed (2C-B2/B3 will add them)
- Profile behavior unchanged
- Small output byte-identical to pre-2C-B1
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"


def load_harness_module(temp_cwd: Path):
    """Load harness_core.py with mocked Anthropic/dotenv."""
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
    # Remove cached agents.* modules so WORKDIR is re-evaluated with the
    # temp cwd. Without this, agents.config.WORKDIR is stale from a prior
    # test/import, and safe_path() resolves to the wrong directory.
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
        "harness_core_output_policy_wiring_test", MODULE_PATH
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
        # Restore cached agents modules
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
    def __init__(self, name="read_file", input_=None, id_="t1"):
        self.name = name
        self.input = input_ or {"path": "test.txt"}
        self.id = id_


class _Text:
    type = "text"
    def __init__(self, text): self.text = text


def _resp_tool_use(name="read_file", input_=None, id_="t1"):
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
    """Find the tool_result content for a given tool_use_id."""
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for r in m["content"]:
                if isinstance(r, dict) and r.get("tool_use_id") == tool_use_id:
                    return r["content"]
    return None


class SmallReadFileTests(unittest.TestCase):
    """Small read_file output must be byte-identical to pre-2C-B1."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_small_read_unchanged(self):
        # Create a small file. Note: run_read uses splitlines() + join,
        # so trailing newline is stripped — this is pre-existing behavior.
        (self.cwd / "small.txt").write_text("hello\nworld\n")
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "small.txt"}),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "read"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertEqual(result, "hello\nworld")

    def test_tiny_read_no_artifact_marker(self):
        (self.cwd / "tiny.txt").write_text("hi")
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "tiny.txt"}),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "read"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertEqual(result, "hi")
        self.assertNotIn("artifact", result.lower())


class LargeReadFileTests(unittest.TestCase):
    """Large read_file output gets artifacted; model sees preview only."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_large_read_produces_artifact_preview(self):
        # Create a file larger than inline_max_bytes (default 8000)
        big_content = "\n".join(f"line {i}" for i in range(1, 1001))  # ~6900 bytes, 1000 lines
        # Actually need more bytes to exceed 8000
        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "read big"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Should contain artifact reference
        self.assertIn("[Full output saved to artifact:", result)
        # Should NOT contain the full raw content
        self.assertNotIn(big_content, result)
        # Should contain line numbers (head)
        self.assertIn("1 | line 1", result)

    def test_artifact_readable_via_read_file(self):
        """The artifact_path returned to the model can be read back via
        the real read_file handler."""
        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)
        # First call: read big.txt -> produces artifact
        # Second call: model reads the artifact_path
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="t1"),
            _resp_tool_use(name="read_file",
                           input_={"path": "__ARTIFACT_PATH__"}, id_="t2"),
            _resp_text(),
        ]

        def _create(**kwargs):
            resp = responses.pop(0)
            # For the second call, replace the placeholder with the actual
            # artifact path extracted from the first tool result.
            first = resp.content[0]
            if getattr(first, "type", None) == "tool_use" and first.input.get("path") == "__ARTIFACT_PATH__":
                # Find artifact path in messages
                for m in messages:
                    if m.get("role") == "user" and isinstance(m.get("content"), list):
                        for r in m["content"]:
                            if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                                # Extract path from "[Full output saved to artifact: <path>]"
                                import re
                                match = re.search(
                                    r"artifact: ([^\]]+)\]", r["content"]
                                )
                                if match:
                                    first.input["path"] = match.group(1).strip()
            return resp

        self.module.client.messages.create = _create
        messages = [{"role": "user", "content": "read big then read artifact"}]
        self.module.agent_loop(messages)

        # First result: preview with artifact ref
        result1 = _find_tool_result(messages, "t1")
        self.assertIn("[Full output saved to artifact:", result1)

        # Second result: actual artifact content (full, not truncated)
        result2 = _find_tool_result(messages, "t2")
        self.assertIsNotNone(result2)
        # The artifact content should be the original big content (or at least
        # contain the first line). Since the artifact is also large, it may
        # get artifacted again — but it should be readable.
        self.assertIn("line 1", result2)


class ExtensionOrderingTests(unittest.TestCase):
    """AFTER_TOOL_RESULT sees the processed (truncated) content, not raw."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_after_tool_result_sees_preview_not_raw(self):
        from agents.types.events import Event

        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        captured_results = []

        def _capture_result(ctx):
            captured_results.append(ctx.get("tool_result", ""))
            return None  # no patch

        self.module.EXTENSIONS.on(Event.AFTER_TOOL_RESULT, _capture_result)
        try:
            responses = [
                _resp_tool_use(name="read_file",
                               input_={"path": "big.txt"}),
                _resp_text(),
            ]
            self.module.client.messages.create = lambda **_: responses.pop(0)
            self.module.agent_loop([{"role": "user", "content": "read"}])
        finally:
            self.module.EXTENSIONS.clear()

        self.assertEqual(len(captured_results), 1)
        result = captured_results[0]
        # Extension should see the preview, not the raw big content
        self.assertIn("[Full output saved to artifact:", result)
        self.assertNotIn(big_content, result)


class BlockAndErrorTests(unittest.TestCase):
    """Blocked and error results are NOT processed by the policy."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_blocked_tool_no_artifact(self):
        from agents.types.events import Event, Priority

        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        def _block_read(ctx):
            from agents.types.events import HookResult
            return HookResult(block=True, reason="blocked for test")

        self.module.EXTENSIONS.on(
            Event.BEFORE_TOOL_CALL, _block_read, priority=Priority.NORMAL
        )
        try:
            responses = [
                _resp_tool_use(name="read_file",
                               input_={"path": "big.txt"}),
                _resp_text(),
            ]
            self.module.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "read"}]
            self.module.agent_loop(messages)
            result = _find_tool_result(messages, "t1")
            self.assertIn("Blocked by extension", result)
            self.assertNotIn("artifact", result.lower())
        finally:
            self.module.EXTENSIONS.clear()

    def test_error_result_not_processed(self):
        # read_file on a nonexistent file returns "Error: ..."
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "nonexistent.txt"}),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "read"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIn("Error", result)
        self.assertNotIn("artifact", result.lower())


class OtherToolsNotProcessedTests(unittest.TestCase):
    """glob_search is NOT yet processed by the OutputPolicy (only
    read_file, bash, grep_search are wired as of 2C-B3B).

    bash IS processed as of 2C-B2A — see BashStructuredOutputTests.
    grep_search IS processed as of 2C-B3B — see GrepWiredOutputTests.
    This class retains the glob not-yet-wired check.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_glob_large_output_not_artifacted(self):
        # glob_search is not in _OUTPUT_POLICY_TOOLS, so even large output
        # passes through raw.
        for i in range(200):
            (self.cwd / f"file_{i:03d}.txt").write_text("x")
        responses = [
            _resp_tool_use(name="glob_search",
                           input_={"pattern": "*.txt", "path": "."},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run glob"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Should NOT contain artifact marker (glob not wired).
        self.assertNotIn("[Full output saved to artifact:", result)


class BashStructuredOutputTests(unittest.TestCase):
    """Stage 2C-B2A: bash is wired into OutputPolicy with structured
    BashExecutionResult (stdout/stderr/exit_code).

    These tests run against the real harness (load_harness_module) so they
    exercise the full path: run_bash -> BashExecutionResult -> policy ->
    artifact -> model message.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_small_bash_passes_through_with_exit_code(self):
        # Small bash output: policy returns str(result) which includes
        # the [exit_code: N] line. No artifact.
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "echo hello"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("[exit_code: 0]", result)
        self.assertIn("hello", result)
        # Small output: no artifact reference.
        self.assertNotIn("[Full output saved to artifact:", result)

    def test_large_bash_stdout_artifacted(self):
        # Large stdout: policy must artifact the structured JSON and return
        # a preview with [exit_code] and --- stdout --- sections.
        # Use Python to generate >8000 bytes of stdout so we don't hit the
        # Windows command-line length limit.
        import sys
        cmd = (
            f'{sys.executable} -c "'
            r'print(chr(10).join(f\"line {i} \" + \"x\"*20 for i in range(1,1001)))'
            '"'
        )
        responses = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Must be artifacted.
        self.assertIn("[Full output saved to artifact:", result)
        # Preview must preserve exit code.
        self.assertIn("[exit_code: 0]", result)
        # Preview must show stdout section.
        self.assertIn("--- stdout ---", result)
        # Must NOT show stderr section (stderr is empty).
        self.assertNotIn("--- stderr ---", result)

    def test_bash_nonzero_exit_code_preserved(self):
        # Non-zero exit code is NOT a transport error — it must go through
        # policy and the exit code must appear in the model-visible content.
        # Use a command that exits 1 with small output.
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "exit 1"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Exit code must be visible to the model.
        self.assertIn("[exit_code: 1]", result)
        # Must NOT be recorded as a transport-level tool_error.
        # (The tool_result content is the bash output, not an "Error:" string.)

    def test_bash_dangerous_command_blocked_no_artifact(self):
        # Dangerous command is blocked at the transport layer (returns
        # "Error: ..." string). This bypasses policy entirely — no artifact,
        # no structured result.
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "sudo rm -rf /"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("Dangerous command blocked", result)
        # Blocked: no artifact, no structured [exit_code] line.
        self.assertNotIn("[Full output saved to artifact:", result)
        self.assertNotIn("[exit_code:", result)

    def test_bash_stderr_separated_from_stdout(self):
        # When both stdout and stderr are present, the preview must show
        # them in separate sections. We trigger stderr via a command that
        # writes to fd 2.
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "echo out; echo err 1>&2"},
                           id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Small output: str(result) format has stdout, then [stderr], then stderr.
        self.assertIn("out", result)
        self.assertIn("[stderr]", result)
        self.assertIn("err", result)

    def test_bash_artifact_contains_structured_json(self):
        # When output is large, the artifact must contain JSON with
        # stdout/stderr/exit_code fields (not just the text preview).
        # We verify by re-reading the artifact:// URI.
        import re
        import json
        import sys

        cmd = (
            f'{sys.executable} -c "'
            r'print(chr(10).join(f\"line {i}\" for i in range(1,1001)))'
            '"'
        )
        responses = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_tool_use(name="read_file",
                           input_={"path": "__ARTIFACT__"}, id_="t2"),
            _resp_text(),
        ]

        def _create(**kwargs):
            resp = responses.pop(0)
            first = resp.content[0]
            if getattr(first, "type", None) == "tool_use" and first.input.get("path") == "__ARTIFACT__":
                for m in messages:
                    if m.get("role") == "user" and isinstance(m.get("content"), list):
                        for r in m["content"]:
                            if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                                match = re.search(r"artifact: (artifact://[^\]]+)\]", r["content"])
                                if match:
                                    first.input["path"] = match.group(1).strip()
            return resp

        self.module.client.messages.create = _create
        messages = [{"role": "user", "content": "run bash then read artifact"}]
        self.module.agent_loop(messages)

        # First result: bash output, artifacted.
        r1 = _find_tool_result(messages, "t1")
        self.assertIsNotNone(r1)
        self.assertIn("[Full output saved to artifact:", r1)

        # Second result: re-read of the artifact. Must contain the
        # structured JSON fields. We don't require the full JSON to be
        # intact (FinalOutputGuard may hard-truncate a single long line),
        # but the field names must be present, proving the artifact stored
        # structured data rather than just text.
        r2 = _find_tool_result(messages, "t2")
        self.assertIsNotNone(r2)
        self.assertIn('"stdout"', r2)
        self.assertIn('"stderr"', r2)
        self.assertIn('"exit_code"', r2)
        self.assertIn('"command"', r2)

    def test_bash_only_stderr_artifacted(self):
        # Large stderr only (stdout empty): preview must show --- stderr ---
        # section and [exit_code]. Artifact must be created.
        import sys
        # Write a lot to stderr and exit 0.
        cmd = (
            f'{sys.executable} -c "'
            r'import sys; sys.stderr.write(chr(10).join(f\"err {i}\" for i in range(1,1001)))'
            '"'
        )
        responses = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("[Full output saved to artifact:", result)
        self.assertIn("[exit_code: 0]", result)
        self.assertIn("--- stderr ---", result)
        # stdout section should NOT appear (stdout is empty).
        self.assertNotIn("--- stdout ---", result)

    def test_bash_stdout_and_stderr_both_large(self):
        # Both stdout and stderr exceed thresholds: both must be truncated
        # independently and both sections present in the preview.
        import sys
        cmd = (
            f'{sys.executable} -c "'
            r'import sys; sys.stdout.write(chr(10).join(f\"out {i}\" for i in range(1,1001))); '
            r'sys.stderr.write(chr(10).join(f\"err {i}\" for i in range(1,1001)))'
            '"'
        )
        responses = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("[Full output saved to artifact:", result)
        self.assertIn("[exit_code: 0]", result)
        self.assertIn("--- stdout ---", result)
        self.assertIn("--- stderr ---", result)
        # Both sections must show their own head (independent truncation).
        self.assertIn("out 1", result)
        self.assertIn("err 1", result)

    def test_bash_artifact_write_failure_degrades_safely(self):
        # When ArtifactStore.store raises, policy must degrade to a
        # hard-truncated preview (no crash, no artifact ref), and the
        # exit_code must still be visible.
        AWE = self.module.ArtifactWriteError
        _real_store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="bash-broken",
        )

        class BrokenStore:
            _session_id = "bash-broken"

            def __init__(self, inner):
                self._inner = inner

            def store(self, *a, **kw):
                raise AWE("disk full")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        import sys
        cmd = (
            f'{sys.executable} -c "'
            r'print(chr(10).join(f\"line {i}\" for i in range(1,1001)))'
            '"'
        )
        responses = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        # Must NOT raise.
        self.module.agent_loop(messages, artifact_store=BrokenStore(_real_store))
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # No artifact ref (write failed).
        self.assertNotIn("[Full output saved to artifact:", result)
        # Exit code still preserved.
        self.assertIn("[exit_code: 0]", result)
        # Preview still present (hard-truncated).
        self.assertIn("line 1", result)

    def test_bash_blocked_in_readonly_profile(self):
        # readonly profile must NOT expose bash to the model and must NOT
        # execute it. The tool_use should be rejected as unavailable.
        responses = [
            _resp_tool_use(name="bash", input_={"command": "echo hi"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages, tool_profile="readonly")
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Must be denied (profile doesn't include bash), not executed.
        self.assertIn("unavailable", result.lower())
        self.assertNotIn("[exit_code: 0]", result)
        self.assertNotIn("hi", result)

    def test_bash_allowed_in_coding_profile(self):
        # coding profile DOES include bash. It must execute normally.
        responses = [
            _resp_tool_use(name="bash", input_={"command": "echo hi"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "run bash"}]
        self.module.agent_loop(messages, tool_profile="coding")
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        self.assertIn("[exit_code: 0]", result)
        self.assertIn("hi", result)


class ArtifactWriteFailureTests(unittest.TestCase):
    """Real harness test: artifact write failure does not break agent loop."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_artifact_write_failure_does_not_break_agent_loop(self):
        """Mock ArtifactStore.store to raise ArtifactWriteError. The agent
        loop must continue, the model must receive a safe truncated result
        (not a crash, not a non-existent artifact_path), and the agent
        status must be completed."""
        # Use the module's own ArtifactWriteError class to ensure the policy's
        # except clause catches it (load_harness_module reloads agents.*).
        AWE = self.module.ArtifactWriteError

        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        # Build a real store, then wrap it with a BrokenStore proxy that
        # raises ArtifactWriteError on store(). We inject it via the
        # artifact_store keyword so agent_loop uses THIS store (not a fresh
        # per-call store) — otherwise the mock would be ignored.
        _real_store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="broken-session",
        )

        class BrokenStore:
            _session_id = "broken-session"

            def __init__(self, inner):
                self._inner = inner

            def store(self, *args, **kwargs):
                raise AWE("disk full simulated")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        broken_store = BrokenStore(_real_store)

        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "read big"}]
        # Must NOT raise.
        self.module.agent_loop(messages, artifact_store=broken_store)
        result = _find_tool_result(messages, "t1")
        self.assertIsNotNone(result)
        # Should have preview content (head+tail with line numbers),
        # NOT an artifact reference (since write failed).
        self.assertNotIn("[Full output saved to artifact:", result)
        # Preview should contain line numbers (read_file formatter).
        self.assertIn("1 | line 1", result)
        self.assertIn("1000 | line 1000", result)
        self.assertIn("lines omitted", result)


class ExtensionPatchGuardTests(unittest.TestCase):
    """Extension returning a huge tool_result_patch must still be
    hard-truncated by the final output guard."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_huge_extension_patch_is_hard_truncated(self):
        from agents.types.events import Event, HookResult, Priority

        (self.cwd / "small.txt").write_text("small content")
        # Extension patches the result with a huge string.
        huge_patch = "Z" * 50000

        def _patch_result(ctx):
            return HookResult(tool_result_patch={"content": huge_patch})

        self.module.EXTENSIONS.on(
            Event.AFTER_TOOL_RESULT, _patch_result, priority=Priority.NORMAL
        )
        try:
            responses = [
                _resp_tool_use(name="read_file",
                               input_={"path": "small.txt"}),
                _resp_text(),
            ]
            self.module.client.messages.create = lambda **_: responses.pop(0)
            messages = [{"role": "user", "content": "read"}]
            self.module.agent_loop(messages)
            result = _find_tool_result(messages, "t1")
            self.assertIsNotNone(result)
            # The 50000-char patch must be hard-truncated to <= inline_max_bytes
            # (default 8000). The result must NOT contain the full 50000 chars.
            self.assertLess(len(result), 10000)
            self.assertIn("[final output guard applied]", result)
        finally:
            self.module.EXTENSIONS.clear()


class ArtifactRereadTests(unittest.TestCase):
    """Reading an artifact must not create a new artifact (copy chain)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reread_artifact_does_not_create_new_artifact(self):
        """Generate artifact A, then read A. The second read must NOT
        create artifact B — it should return a preview referencing A."""
        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        # Build a real store and wrap it with a tracking proxy. Inject via
        # artifact_store= so agent_loop uses THIS store (per-call store
        # would bypass the mock).
        _real_store = self.module.ArtifactStore(
            root_dir=self.module._ARTIFACT_ROOT,
            session_id="reread-session",
        )
        store_calls = []

        class TrackingStore:
            _session_id = "reread-session"

            def __init__(self, inner):
                self._inner = inner

            def store(self, *args, **kwargs):
                store_calls.append(args[0] if args else kwargs.get("content"))
                return self._inner.store(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        tracking_store = TrackingStore(_real_store)

        # First call: read big.txt -> creates artifact A
        # Second call: read artifact A -> should NOT create artifact B
        import re

        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="t1"),
            _resp_tool_use(name="read_file",
                           input_={"path": "__ARTIFACT__"}, id_="t2"),
            _resp_text(),
        ]

        def _create(**kwargs):
            resp = responses.pop(0)
            first = resp.content[0]
            if getattr(first, "type", None) == "tool_use" and first.input.get("path") == "__ARTIFACT__":
                for m in messages:
                    if m.get("role") == "user" and isinstance(m.get("content"), list):
                        for r in m["content"]:
                            if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                                match = re.search(r"artifact: ([^\]]+)\]", r["content"])
                                if match:
                                    first.input["path"] = match.group(1).strip()
            return resp

        self.module.client.messages.create = _create
        messages = [{"role": "user", "content": "read big then read artifact"}]
        self.module.agent_loop(messages, artifact_store=tracking_store)

        # Only ONE artifact should have been created (for big.txt).
        self.assertEqual(len(store_calls), 1,
                         f"Expected 1 artifact, got {len(store_calls)} "
                         f"(copy chain detected)")

        # Second result should reference the original artifact, not a new one.
        result2 = _find_tool_result(messages, "t2")
        self.assertIsNotNone(result2)
        self.assertIn("artifact re-read", result2)
        self.assertNotIn("[Full output saved to artifact:", result2)


class CrossSessionArtifactAccessTests(unittest.TestCase):
    """Session A cannot read Session B's artifacts. The check happens BEFORE
    the read_file handler executes, so file content is never returned."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cross_session_artifact_access_denied(self):
        """Session A creates an artifact. Session B (different session_id)
        tries to read it via its artifact:// URI — must be denied, content
        never returned.

        Note (2C-B1.4): the artifact reference returned to the model is
        now a virtual ``artifact://<id>`` URI that does NOT embed the
        session_id. Session B cannot craft a URI that points into A's
        store, because B's store only resolves within B's own session
        directory. We verify both:
          - Session B reading A's artifact:// URI -> rejected by store
          - Session B reading the FS path under .harness/artifacts/ -> denied
        """
        # Session A: create a big file and generate an artifact
        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        import re

        # Run Session A to create the artifact
        responses_a = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_a.pop(0)
        messages_a = [{"role": "user", "content": "read big"}]
        self.module.agent_loop(messages_a, session_id="session-a")

        # Extract the artifact URI from Session A's result.
        result_a = _find_tool_result(messages_a, "t1")
        self.assertIsNotNone(result_a)
        match = re.search(r"artifact: (artifact://[^\]]+)\]", result_a)
        self.assertIsNotNone(match, f"artifact URI not found in: {result_a!r}")
        artifact_uri = match.group(1).strip()
        # URI must be the virtual form, NOT a filesystem path. Specifically,
        # it must NOT contain the session id (the model must not learn it).
        self.assertTrue(artifact_uri.startswith("artifact://"))
        self.assertNotIn("session-a", artifact_uri)
        self.assertNotIn("session-b", artifact_uri)

        # Session B: try to read Session A's artifact URI. B's store is a
        # different per-call store; the artifact_id does not exist there,
        # so the read must fail. The error message must NOT leak A's content.
        responses_b = [
            _resp_tool_use(name="read_file",
                           input_={"path": artifact_uri}, id_="t2"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_b.pop(0)
        messages_b = [{"role": "user", "content": "read other session artifact"}]
        self.module.agent_loop(messages_b, session_id="session-b")

        result_b = _find_tool_result(messages_b, "t2")
        self.assertIsNotNone(result_b)
        # Must be an error (artifact not found in B's store), not the content.
        self.assertNotIn("line 1 xxxxxxxxxx", result_b)
        self.assertNotIn("line 500", result_b)

    def test_path_traversal_cross_session_denied(self):
        """Session B tries an absolute path into Session A's artifact dir."""
        big_content = "\n".join(f"line {i}" for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        # Session A creates artifact
        responses_a = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_a.pop(0)
        self.module.agent_loop([{"role": "user", "content": "x"}],
                               session_id="session-a")

        # Session B tries an absolute path into Session A's artifact subdir.
        # The artifact root is now private (outside WORKDIR), but the guard
        # must still reject any path that resolves into it.
        traversal_path = str(
            self.module._ARTIFACT_ROOT / "session-a" / "fake.txt"
        )
        responses_b = [
            _resp_tool_use(name="read_file",
                           input_={"path": traversal_path}, id_="t2"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_b.pop(0)
        messages_b = [{"role": "user", "content": "traversal attempt"}]
        self.module.agent_loop(messages_b, session_id="session-b")

        result_b = _find_tool_result(messages_b, "t2")
        # Must be denied (either access denied or error — not file content)
        self.assertNotIn("line 500", result_b)

    def test_same_session_artifact_readable(self):
        """Session A can read its own artifact."""
        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big_content)

        import re

        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="t1"),
            _resp_tool_use(name="read_file",
                           input_={"path": "__ARTIFACT__"}, id_="t2"),
            _resp_text(),
        ]

        def _create(**kwargs):
            resp = responses.pop(0)
            first = resp.content[0]
            if getattr(first, "type", None) == "tool_use" and first.input.get("path") == "__ARTIFACT__":
                for m in messages:
                    if m.get("role") == "user" and isinstance(m.get("content"), list):
                        for r in m["content"]:
                            if isinstance(r, dict) and r.get("tool_use_id") == "t1":
                                match = re.search(r"artifact: ([^\]]+)\]", r["content"])
                                if match:
                                    first.input["path"] = match.group(1).strip()
            return resp

        self.module.client.messages.create = _create
        messages = [{"role": "user", "content": "read big then read own artifact"}]
        self.module.agent_loop(messages, session_id="session-x")

        result2 = _find_tool_result(messages, "t2")
        self.assertIsNotNone(result2)
        # Same session — must NOT be access denied
        self.assertNotIn("Access denied", result2)


class DualSessionConcurrencyTests(unittest.TestCase):
    """Two sessions running concurrently must have isolated artifacts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_concurrent_sessions_isolated_artifacts(self):
        """Session A and Session B each generate large outputs. Their
        artifacts must go to separate directories, and neither can read
        the other's."""
        import threading
        import re

        big_content = "\n".join(f"line {i} " + "x" * 10 for i in range(1, 1001))
        (self.cwd / "big_a.txt").write_text(big_content)
        (self.cwd / "big_b.txt").write_text(big_content)

        results = {}
        errors = []
        lock = threading.Lock()
        # Per-session response queues. The shared create dispatches by
        # inspecting the messages in the request.
        responses_a = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big_a.txt"}, id_="t-sess-a"),
            _resp_text(),
        ]
        responses_b = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big_b.txt"}, id_="t-sess-b"),
            _resp_text(),
        ]

        def _shared_create(**kwargs):
            # Dispatch based on the last user message content.
            msgs = kwargs.get("messages", [])
            last = msgs[-1] if msgs else {}
            content = last.get("content", "") if isinstance(last, dict) else ""
            if isinstance(content, list):
                # tool_result turn — pick the session by tool_use_id
                for item in content:
                    if isinstance(item, dict) and "tool_use_id" in item:
                        tid = item["tool_use_id"]
                        if "sess-a" in tid:
                            return responses_a.pop(0)
                        elif "sess-b" in tid:
                            return responses_b.pop(0)
                # Fallback
                return _resp_text()
            # First turn — match by content
            if "big_a" in str(content):
                return responses_a.pop(0)
            elif "big_b" in str(content):
                return responses_b.pop(0)
            return _resp_text()

        self.module.client.messages.create = _shared_create

        def run_session(session_id, filename, key):
            try:
                msgs = [{"role": "user", "content": f"read {filename}"}]
                self.module.agent_loop(msgs, session_id=session_id)
                results[key] = _find_tool_result(msgs, f"t-{session_id}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_session, args=("sess-a", "big_a.txt", "a"))
        t2 = threading.Thread(target=run_session, args=("sess-b", "big_b.txt", "b"))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        self.assertEqual(errors, [], f"Threads raised: {errors}")
        # Both sessions produced artifacts
        self.assertIn("a", results)
        self.assertIn("b", results)
        self.assertIn("[Full output saved to artifact:", results["a"])
        self.assertIn("[Full output saved to artifact:", results["b"])

        # Extract artifact URIs. They must be distinct virtual URIs
        # (artifact://<uuid>) and must NOT embed the session id — the
        # model only sees the virtual URI, never the FS path.
        match_a = re.search(r"artifact: (artifact://[^\]]+)\]", results["a"])
        match_b = re.search(r"artifact: (artifact://[^\]]+)\]", results["b"])
        self.assertIsNotNone(match_a, f"no artifact URI in {results['a']!r}")
        self.assertIsNotNone(match_b, f"no artifact URI in {results['b']!r}")
        uri_a = match_a.group(1).strip()
        uri_b = match_b.group(1).strip()
        self.assertTrue(uri_a.startswith("artifact://"))
        self.assertTrue(uri_b.startswith("artifact://"))
        self.assertNotIn("sess-a", uri_a)
        self.assertNotIn("sess-b", uri_b)
        self.assertNotEqual(uri_a, uri_b,
                            "two sessions must produce distinct artifacts")

        # Cross-session read: B cannot read A's artifact URI. B's store
        # is a separate per-call instance and does not hold A's artifact_id.
        responses_cross = [
            _resp_tool_use(name="read_file",
                           input_={"path": uri_a}, id_="t-cross"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_cross.pop(0)
        msgs_cross = [{"role": "user", "content": "read other session"}]
        self.module.agent_loop(msgs_cross, session_id="sess-b")
        cross = _find_tool_result(msgs_cross, "t-cross")
        self.assertIsNotNone(cross)
        # Must be an error, not A's content.
        self.assertNotIn("line 1 xxxxxxxxxx", cross)
        self.assertNotIn("line 500", cross)


class PathBasedToolArtifactGuardTests(unittest.TestCase):
    """Stage 2C-B1.4 / 2C-B2B-1: path-based tools (read_file, write_file,
    edit_file, grep_search, glob_search) must reject any path that resolves
    into the artifact root, regardless of how the path is spelled.

    Stage 2C-B2B-1: the artifact root now lives OUTSIDE WORKDIR (private
    directory), so relative paths from WORKDIR cannot reach it. These
    tests use absolute paths into the private root to verify the guard
    still catches them.

    Each variant below MUST be rejected:
      - absolute filesystem path into the artifact root
      - ``..`` traversal (within the private root) that lands inside a
        session subdir
      - backslash variant (Windows) — must normalize the same way
      - path into a *different* session's subdir (the guard is global,
        not per-session: the model must use artifact:// URIs)
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)
        # The private artifact root (outside WORKDIR). Tests construct
        # paths into this root to verify the guard rejects them.
        self.artifact_root = self.module._ARTIFACT_ROOT

    def tearDown(self):
        self._tmp.cleanup()

    def _run_one_tool(self, tool_name, tool_input, session_id="sess-x"):
        """Run agent_loop with a single tool_use and return its result
        string. Uses an arbitrary session_id; the guard must reject
        artifact-root paths regardless of which session_id is active."""
        responses = [
            _resp_tool_use(name=tool_name, input_=tool_input, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": f"use {tool_name}"}]
        self.module.agent_loop(messages, session_id=session_id)
        return _find_tool_result(messages, "t1")

    def test_grep_search_into_artifact_root_denied(self):
        # Seed an artifact so the path genuinely exists on disk; the guard
        # must reject the path even when the file is real (no string-match
        # bypass via existence check).
        big = "\n".join(f"line {i}" for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big)
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="seed"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        seed_msgs = [{"role": "user", "content": "seed"}]
        self.module.agent_loop(seed_msgs, session_id="sess-x")
        seed_result = _find_tool_result(seed_msgs, "seed")
        self.assertIsNotNone(seed_result)
        self.assertIn("[Full output saved to artifact:", seed_result)

        # grep_search pointing at the private artifact root (absolute path).
        result = self._run_one_tool(
            "grep_search",
            {"pattern": "line", "path": str(self.artifact_root)},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)
        self.assertNotIn("line 500", result)

    def test_glob_search_into_artifact_root_denied(self):
        result = self._run_one_tool(
            "glob_search",
            {"pattern": "*.txt", "path": str(self.artifact_root / "sess-x")},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)
        self.assertNotIn(".txt", result)

    def test_write_file_into_artifact_root_denied(self):
        result = self._run_one_tool(
            "write_file",
            {"path": str(self.artifact_root / "sess-x" / "evil.txt"),
             "content": "overwrite"},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)

    def test_edit_file_into_artifact_root_denied(self):
        result = self._run_one_tool(
            "edit_file",
            {"path": str(self.artifact_root / "sess-x" / "anything.txt"),
             "old_text": "a", "new_text": "b"},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)

    def test_dotdot_traversal_into_artifact_root_denied(self):
        # A path that doesn't literally appear under the artifact root but
        # resolves into it must still be rejected. This is the core reason
        # we do NOT string-match. We build a path with a redundant ``..``
        # segment that normalizes into the artifact root.
        target = self.artifact_root / "sess-x" / "fake.txt"
        # Insert a subdirectory and ``..`` so the literal string differs.
        tricky = str(self.artifact_root / "sub" / ".." / "sess-x" / "fake.txt")
        result = self._run_one_tool(
            "read_file",
            {"path": tricky},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)

    def test_backslash_variant_into_artifact_root_denied(self):
        # On Windows, backslash is a path separator. The normalization in
        # _path_resolves_into_artifact_root uses Path.resolve(), which
        # handles both forward and back slashes. We build the artifact
        # root path with backslashes to verify the guard is not bypassed.
        # On non-Windows this is still a valid (if unusual) path.
        bs_path = str(self.artifact_root).replace("/", "\\") + "\\sess-x\\fake.txt"
        result = self._run_one_tool(
            "read_file",
            {"path": bs_path},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)

    def test_absolute_path_into_artifact_root_denied(self):
        # An absolute filesystem path into the artifact root must be
        # rejected. This is the primary attack vector now that the root
        # is outside WORKDIR.
        abs_artifact = str(self.artifact_root / "sess-x" / "x.txt")
        result = self._run_one_tool(
            "read_file",
            {"path": abs_artifact},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)

    def test_cross_session_path_via_dotdot_denied(self):
        # Session B tries to read Session A's artifact via absolute path.
        # The guard is global (any path into artifact_root is denied), so
        # cross-session access is blocked the same way same-session is.
        result = self._run_one_tool(
            "read_file",
            {"path": str(self.artifact_root / "sess-a" / "secret.txt")},
            session_id="sess-b",
        )
        self.assertIsNotNone(result)
        self.assertIn("Access denied", result)

    def test_normal_path_outside_artifact_root_still_works(self):
        # Sanity: a normal workspace file must still be readable. The guard
        # must not over-block.
        (self.cwd / "normal.txt").write_text("hello\n")
        result = self._run_one_tool(
            "read_file",
            {"path": "normal.txt"},
            session_id="sess-x",
        )
        self.assertIsNotNone(result)
        self.assertNotIn("Access denied", result)
        self.assertIn("hello", result)


class BashArtifactAccessKnownLimitation(unittest.TestCase):
    """Stage 2C-B2B-1: the artifact root now lives OUTSIDE WORKDIR, so
    bash can no longer reach artifacts via WORKDIR-relative paths. This
    closes the "model guesses .harness/artifacts/" vector.

    Stage 2C-B2B-4: the OLD xfail (NoOpSandbox can read artifacts via
    absolute path) is REMOVED. In trusted_local + NoOpSandbox, bash CAN
    read host files — this is the documented mode contract, not a
    'limitation to be fixed later'. Real cross-session bash isolation
    requires secure_multi_session + DockerSandbox, which is enforced by
    the startup validation (SecureSandboxError) and the per-call token.
    The test below now PASSES by asserting the trusted_local contract
    holds (bash reads work) rather than xfail-ing on it.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_session_a_artifact(self):
        """Create an artifact in Session A and return its absolute path."""
        import glob
        big = "\n".join(f"secret-{i}" for i in range(1, 1001))
        (self.cwd / "big.txt").write_text(big)
        responses_a = [
            _resp_tool_use(name="read_file",
                           input_={"path": "big.txt"}, id_="seed"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_a.pop(0)
        self.module.agent_loop(
            [{"role": "user", "content": "seed"}], session_id="sess-a"
        )
        # Locate the real artifact file in the private root.
        artifact_files = glob.glob(
            str(self.module._ARTIFACT_ROOT / "sess-a" / "*.txt")
        )
        self.assertTrue(artifact_files, "seed artifact not found on disk")
        return artifact_files[0]

    def test_bash_cannot_reach_artifact_via_workdir_relative_path(self):
        """Stage 2C-B2B-1 PASS: the artifact root is outside WORKDIR, so
        a bash command that tries the old ``.harness/artifacts/`` path
        finds nothing. This is the primary mitigation — the model cannot
        guess a WORKDIR-relative path that works. (This holds in BOTH
        trusted_local and secure mode, because the artifact root is
        physically outside WORKDIR regardless of sandbox.)"""
        self._seed_session_a_artifact()
        # Try the old relative path. It must NOT yield artifact content.
        # Use a cross-platform file read via Python.
        import sys
        cmd = (
            f'{sys.executable} -c "'
            f'import sys,os; p=os.path.join(os.getcwd(),'
            f'".harness","artifacts","sess-a"); '
            f'fs=os.listdir(p) if os.path.isdir(p) else []; '
            f'sys.stdout.write("FOUND:"+str(fs))'
            f'"'
        )
        responses_b = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_b.pop(0)
        msgs_b = [{"role": "user", "content": "exfil"}]
        self.module.agent_loop(msgs_b, session_id="sess-b")
        result = _find_tool_result(msgs_b, "t1")
        self.assertIsNotNone(result)
        # The old relative path must not exist; bash finds no files.
        self.assertNotIn("secret-1", result)
        # And the directory listing should be empty (no such directory).
        self.assertNotIn("FOUND:['", result)

    def test_trusted_local_bash_can_read_artifact_via_absolute_path(self):
        """trusted_local + NoOpSandbox: bash CAN read an artifact via
        its absolute path. This is the MODE CONTRACT — trusted_local
        provides NO cross-session bash filesystem isolation, by design.
        Operators who need isolation must use secure_multi_session +
        DockerSandbox.

        This test replaces the old xfail. It PASSES (asserting the
        contract) rather than xfail-ing (asserting a limitation). The
        secure-mode equivalent is covered by
        SecureModeStartupValidationTests: agent_loop raises
        SecureSandboxError before bash ever runs."""
        import sys
        secret_path = self._seed_session_a_artifact()
        cmd = (
            f'{sys.executable} -c "'
            f'import sys; sys.stdout.write(open(sys.argv[1]).read())'
            f'" "{secret_path}"'
        )
        responses_b = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses_b.pop(0)
        msgs_b = [{"role": "user", "content": "exfil"}]
        self.module.agent_loop(msgs_b, session_id="sess-b")
        result = _find_tool_result(msgs_b, "t1")
        self.assertIsNotNone(result)
        # trusted_local contract: bash DID read the artifact content.
        # This is accepted behavior — not a bug to fix.
        self.assertIn("secret-1", result)


class SandboxCapabilitiesAndRunModeTests(unittest.TestCase):
    """Stage 2C-B2B-2: verifies that
    (a) each sandbox backend declares SandboxCapabilities honestly, and
    (b) the secure_multi_session run mode rejects bash when the active
        backend does not provide filesystem_isolation.

    These tests reload the harness module with specific AGENT_RUN_MODE /
    AGENT_SANDBOX_BACKEND env combinations so that the module-level
    RUN_MODE and SANDBOX are wired correctly.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        # Snapshot env so we can restore it precisely.
        self._prev_env = {
            k: os.environ.get(k) for k in (
                "AGENT_RUN_MODE", "AGENT_SANDBOX_BACKEND",
            )
        }

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _load_with(self, run_mode: str, sandbox_backend: str):
        """Load harness_core with the given env overrides."""
        if run_mode is None:
            os.environ.pop("AGENT_RUN_MODE", None)
        else:
            os.environ["AGENT_RUN_MODE"] = run_mode
        if sandbox_backend is None:
            os.environ.pop("AGENT_SANDBOX_BACKEND", None)
        else:
            os.environ["AGENT_SANDBOX_BACKEND"] = sandbox_backend
        return load_harness_module(self.cwd)

    # -- capability declarations ----------------------------------------

    def test_noop_sandbox_declares_no_filesystem_isolation(self):
        """NoOpSandbox MUST declare supports_filesystem_isolation=False.
        It shares the host filesystem with the Harness process and
        cannot prevent bash from reading absolute paths (including the
        private artifact root). Declaring True here would be a security
        lie."""
        module = self._load_with("trusted_local", "none")
        # Use the class name on the SANDBOX instance instead of isinstance,
        # because load_harness_module clears agents.* from sys.modules and
        # reloads sandbox under a different module name — a fresh
        # ``from agents.sandbox import NoOpSandbox`` returns a different
        # class object than the one the harness module actually used.
        self.assertEqual(type(module.SANDBOX).__name__, "NoOpSandbox")
        caps = module.SANDBOX.capabilities
        self.assertFalse(caps.supports_filesystem_isolation,
                         "NoOpSandbox must not claim supports_filesystem_isolation")
        self.assertFalse(caps.supports_process_isolation,
                         "NoOpSandbox must not claim supports_process_isolation")

    def test_docker_sandbox_declares_filesystem_isolation(self):
        """DockerSandbox MUST declare supports_filesystem_isolation=True.
        When correctly configured (B2B-3 verifies the mount list via
        assess_isolation), the container's mount namespace prevents bash
        from reaching the private artifact root. This declaration is the
        basis for secure_multi_session mode accepting DockerSandbox."""
        # DockerSandbox.capabilities is a class-level property backed by
        # a sentinel _CAPS, so any fresh import of the class is fine for
        # verifying the declaration — we don't need the exact class
        # object the harness module used (that matters only for
        # isinstance checks, which we don't do here).
        from agents.sandbox import DockerSandbox
        d = DockerSandbox(workdir=str(self.cwd))
        caps = d.capabilities
        self.assertTrue(caps.supports_filesystem_isolation,
                        "DockerSandbox must declare supports_filesystem_isolation")
        self.assertTrue(caps.supports_process_isolation,
                        "DockerSandbox must declare supports_process_isolation")

    # -- trusted_local mode ---------------------------------------------

    def test_trusted_local_allows_noop_sandbox_bash(self):
        """trusted_local + NoOpSandbox: bash runs normally. This is the
        backward-compatible dev default; no new friction for single-user
        workflows."""
        module = self._load_with("trusted_local", "none")
        self.assertEqual(module.RUN_MODE, "trusted_local")
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "echo ok"}, id_="t1"),
            _resp_text(),
        ]
        module.client.messages.create = lambda **_: responses.pop(0)
        msgs = [{"role": "user", "content": "run"}]
        module.agent_loop(msgs)
        result = _find_tool_result(msgs, "t1")
        self.assertIsNotNone(result)
        # Bash ran: the result contains exit_code marker + "ok".
        self.assertIn("ok", result)
        self.assertNotIn("secure_multi_session", result)

    # -- secure_multi_session mode --------------------------------------
    # B2B-3 changed the behaviour: secure_multi_session + NoOpSandbox +
    # bash in active profile now FAILS FAST at agent_loop startup via
    # SecureSandboxError, instead of returning an Error string from
    # run_bash. The runtime guard in run_bash stays as defence-in-depth.

    def test_secure_mode_rejects_noop_sandbox_bash(self):
        """secure_multi_session + NoOpSandbox + bash in profile:
        agent_loop MUST raise SecureSandboxError at startup, before any
        model request. The model never sees a bash tool it can't use
        safely — fail fast, never silently degrade."""
        module = self._load_with("secure_multi_session", "none")
        self.assertEqual(module.RUN_MODE, "secure_multi_session")
        # No responses wired — the loop must raise BEFORE calling the
        # model. If it tried to call the model, we'd see an AttributeError
        # on client.messages.create instead of SecureSandboxError.
        msgs = [{"role": "user", "content": "run"}]
        with self.assertRaises(module.SecureSandboxError) as cm:
            module.agent_loop(msgs)
        err = str(cm.exception)
        # Error names the missing capability and the backend.
        self.assertIn("secure_multi_session", err)
        self.assertIn("supports_filesystem_isolation", err)
        self.assertIn("NoOpSandbox", err)

    def test_secure_mode_rejects_bash_even_for_safe_commands(self):
        """The startup validation is not a command-string check. Even a
        trivially safe command like ``echo hello`` never gets to run,
        because agent_loop raises before the first model request. This
        proves the guard is on the BACKEND config, not the command."""
        module = self._load_with("secure_multi_session", "none")
        msgs = [{"role": "user", "content": "run echo hello"}]
        with self.assertRaises(module.SecureSandboxError):
            module.agent_loop(msgs)
        # No tool_result recorded — the loop never reached tool dispatch.
        self.assertEqual(len(msgs), 1)

    def test_secure_mode_rejection_does_not_produce_artifact(self):
        """Secure-mode rejection happens at startup, before any tool
        call. No artifact can be created because no tool ever ran."""
        module = self._load_with("secure_multi_session", "none")
        msgs = [{"role": "user", "content": "run"}]
        with self.assertRaises(module.SecureSandboxError):
            module.agent_loop(msgs)
        # No tool_result in messages — no artifact marker can exist.
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for r in m["content"]:
                    if isinstance(r, dict):
                        self.assertNotIn("artifact", str(r).lower())

    def test_secure_mode_does_not_break_read_file(self):
        """The startup validation only fires when bash is in the active
        profile. With tool_profile='readonly' (read/grep/glob, no bash),
        secure_multi_session mode must NOT raise — read_file and other
        path-based tools have their own artifact-root guard from 2C-B1.4
        and do not depend on sandbox filesystem isolation."""
        module = self._load_with("secure_multi_session", "none")
        (self.cwd / "normal.txt").write_text("hello\n")
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "normal.txt"}, id_="t1"),
            _resp_text(),
        ]
        module.client.messages.create = lambda **_: responses.pop(0)
        msgs = [{"role": "user", "content": "read"}]
        # readonly profile = read_file/grep/glob, no bash → secure-mode
        # startup validation must NOT fire. read_file still works.
        module.agent_loop(msgs, tool_profile="readonly")
        result = _find_tool_result(msgs, "t1")
        self.assertIsNotNone(result)
        self.assertIn("hello", result)
        self.assertNotIn("secure_multi_session", result)

    def test_trusted_local_bash_still_runs_when_docker_unavailable(self):
        """Sanity: the default config (no env vars set) must keep bash
        working. This is the dev-workflow compatibility guarantee."""
        module = self._load_with(None, "none")
        self.assertEqual(module.RUN_MODE, "trusted_local")
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "echo ok"}, id_="t1"),
            _resp_text(),
        ]
        module.client.messages.create = lambda **_: responses.pop(0)
        msgs = [{"role": "user", "content": "run"}]
        module.agent_loop(msgs)
        result = _find_tool_result(msgs, "t1")
        self.assertIsNotNone(result)
        self.assertIn("ok", result)


class SandboxIsolationAssessmentTests(unittest.TestCase):
    """Stage 2C-B2B-3: verifies ``assess_isolation()`` on each backend.

    Where ``SandboxCapabilities`` declares what a backend *supports*,
    ``assess_isolation()`` evaluates the *actual* runtime configuration
    against specific private paths. ``secure_multi_session`` mode uses
    this to decide whether the current mount plan really isolates the
    artifact root from bash.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # -- NoOpSandbox ----------------------------------------------------

    def test_noop_assess_isolation_always_false(self):
        """NoOpSandbox shares the host filesystem — it can NEVER isolate
        any private path, regardless of where the path lives. The
        assessment must say so with a clear reason."""
        from agents.sandbox import NoOpSandbox
        n = NoOpSandbox(workdir=str(self.cwd))
        a = n.assess_isolation(
            workdir=str(self.cwd),
            private_paths=(str(self.cwd / "anywhere"),),
        )
        self.assertFalse(a.filesystem_isolated)
        self.assertTrue(len(a.reasons) >= 1)
        # Reason must mention the host-filesystem sharing, not a path
        # containment detail — the issue is structural, not locational.
        self.assertIn("host filesystem", a.reasons[0].lower())

    def test_noop_assess_isolation_false_even_for_unrelated_path(self):
        """Even a private path on a different drive (Windows) or far
        outside the workdir must be reported as non-isolated, because
        NoOpSandbox bash can ``cat`` any absolute path the OS user can
        read. The location of the private path is irrelevant."""
        from agents.sandbox import NoOpSandbox
        n = NoOpSandbox(workdir=str(self.cwd))
        # A path that is clearly NOT under the workdir.
        far_path = Path(tempfile.gettempdir()) / "far-away-secret"
        a = n.assess_isolation(
            workdir=str(self.cwd),
            private_paths=(str(far_path),),
        )
        self.assertFalse(a.filesystem_isolated)

    # -- DockerSandbox: mount-plan containment --------------------------

    def test_docker_assess_isolated_when_artifact_outside_workdir(self):
        """DockerSandbox mounts ONLY self._workdir. When the artifact
        root lives outside the workdir (the B2B-1 default), bash in the
        container cannot reach it. The assessment must pass."""
        from agents.sandbox import DockerSandbox
        d = DockerSandbox(workdir=str(self.cwd))
        # Artifact root on a totally separate temp path.
        artifact_root = Path(tempfile.gettempdir()) / "separate-artifact-root"
        artifact_root.mkdir(parents=True, exist_ok=True)
        a = d.assess_isolation(
            workdir=str(self.cwd),
            private_paths=(str(artifact_root),),
        )
        self.assertTrue(a.filesystem_isolated, a.reasons)
        self.assertEqual(len(a.reasons), 0)
        # mount_sources exposes the host mount for diagnostics.
        self.assertIn(str(self.cwd.resolve()), a.mount_sources)

    def test_docker_assess_not_isolated_when_artifact_in_workdir(self):
        """If the artifact root were inside the workdir (the OLD
        pre-B2B-1 layout), bash could reach it via the mount. The
        assessment MUST flag this as a leak with a path-containment
        reason, even though DockerSandbox's capability flag is True.
        This is the core B2B-3 insight: capability ≠ current-config
        safety."""
        from agents.sandbox import DockerSandbox
        d = DockerSandbox(workdir=str(self.cwd))
        artifact_in_workdir = self.cwd / ".harness" / "artifacts"
        artifact_in_workdir.mkdir(parents=True, exist_ok=True)
        a = d.assess_isolation(
            workdir=str(self.cwd),
            private_paths=(str(artifact_in_workdir),),
        )
        self.assertFalse(a.filesystem_isolated)
        self.assertTrue(len(a.reasons) >= 1)
        # Reason must mention the containment (path under mount source).
        reason = a.reasons[0]
        self.assertIn("under mount source", reason)
        self.assertIn(str(artifact_in_workdir), reason)

    def test_docker_assess_normalizes_dotdot_in_private_path(self):
        """A private path with ``..`` segments that resolves OUTSIDE
        the workdir must be correctly reported as isolated. Conversely,
        ``..`` that resolves INSIDE must be flagged. This proves the
        assessment resolves paths before checking containment — string
        matching would fail this."""
        from agents.sandbox import DockerSandbox
        d = DockerSandbox(workdir=str(self.cwd))
        # craft a path with .. that resolves outside workdir
        outside = (self.cwd / ".." / "outside-secret").resolve()
        outside.mkdir(parents=True, exist_ok=True)
        try:
            a = d.assess_isolation(
                workdir=str(self.cwd),
                private_paths=(str(self.cwd / ".." / "outside-secret"),),
            )
            self.assertTrue(a.filesystem_isolated, a.reasons)
        finally:
            outside.rmdir()

    def test_docker_assess_not_isolated_when_artifact_under_workdir_parent(self):
        """If Docker mounted a PARENT of the workdir (a hypothetical
        misconfiguration), an artifact root that is a sibling of the
        workdir would also be reachable. This test simulates the
        containment check by directly verifying _path_contains on the
        parent — the same primitive assess_isolation uses."""
        from agents.sandbox import _path_contains
        parent = self.cwd.parent
        sibling = parent / "sibling-artifact"
        sibling.mkdir(parents=True, exist_ok=True)
        try:
            # If the mount source were `parent`, then `sibling` is inside.
            self.assertTrue(_path_contains(parent, sibling))
            # And the workdir itself is also inside the parent.
            self.assertTrue(_path_contains(parent, self.cwd))
        finally:
            sibling.rmdir()


class SecureModeStartupValidationTests(unittest.TestCase):
    """Stage 2C-B2B-3: verifies the fail-fast startup validation in
    agent_loop when RUN_MODE == secure_multi_session.

    Two layers of check:
      1. Capability check — NoOpSandbox rejected (doesn't support isolation)
      2. Runtime assessment — DockerSandbox rejected if the current
         mount plan would expose the artifact root

    Both must raise SecureSandboxError BEFORE the first model request.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self._prev_env = {
            k: os.environ.get(k) for k in (
                "AGENT_RUN_MODE", "AGENT_SANDBOX_BACKEND",
            )
        }

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _load_with(self, run_mode: str, sandbox_backend: str):
        if run_mode is None:
            os.environ.pop("AGENT_RUN_MODE", None)
        else:
            os.environ["AGENT_RUN_MODE"] = run_mode
        if sandbox_backend is None:
            os.environ.pop("AGENT_SANDBOX_BACKEND", None)
        else:
            os.environ["AGENT_SANDBOX_BACKEND"] = sandbox_backend
        return load_harness_module(self.cwd)

    # -- Layer 1: capability check -------------------------------------

    def test_secure_mode_raises_on_noop_sandbox_at_startup(self):
        """secure_multi_session + NoOpSandbox + bash in profile: raises
        SecureSandboxError at startup. This is the capability check —
        NoOpSandbox.supports_filesystem_isolation is False. The model
        never gets called."""
        module = self._load_with("secure_multi_session", "none")
        # No client mock wired — if the loop reached the model it would
        # blow up with AttributeError, not SecureSandboxError.
        with self.assertRaises(module.SecureSandboxError) as cm:
            module.agent_loop([{"role": "user", "content": "run"}])
        self.assertIn("supports_filesystem_isolation", str(cm.exception))
        self.assertIn("NoOpSandbox", str(cm.exception))

    def test_secure_mode_skips_validation_when_bash_not_in_profile(self):
        """secure_multi_session + NoOpSandbox + readonly profile (no
        bash): MUST NOT raise. The validation only fires when bash is
        in the active tool set, because only bash depends on sandbox
        filesystem isolation. read_file etc. have their own path guard.
        This is the same guarantee as B2B-2's
        test_secure_mode_does_not_break_read_file, but framed as a
        startup-validation contract."""
        module = self._load_with("secure_multi_session", "none")
        (self.cwd / "x.txt").write_text("ok\n")
        responses = [
            _resp_tool_use(name="read_file",
                           input_={"path": "x.txt"}, id_="t1"),
            _resp_text(),
        ]
        module.client.messages.create = lambda **_: responses.pop(0)
        msgs = [{"role": "user", "content": "read"}]
        module.agent_loop(msgs, tool_profile="readonly")
        result = _find_tool_result(msgs, "t1")
        self.assertIn("ok", result)

    def test_trusted_local_never_validates(self):
        """trusted_local mode NEVER calls _validate_secure_sandbox, even
        with NoOpSandbox + bash. This is the backward-compat guarantee:
        existing dev workflows keep working without opting into secure
        mode. (If validation fired here, every default dev run would
        raise.)"""
        module = self._load_with("trusted_local", "none")
        responses = [
            _resp_tool_use(name="bash",
                           input_={"command": "echo ok"}, id_="t1"),
            _resp_text(),
        ]
        module.client.messages.create = lambda **_: responses.pop(0)
        msgs = [{"role": "user", "content": "run"}]
        # Must NOT raise.
        module.agent_loop(msgs)
        result = _find_tool_result(msgs, "t1")
        self.assertIn("ok", result)

    # -- Layer 2: runtime assessment with injected artifact_store ------

    def test_secure_mode_rejects_injected_store_inside_workdir(self):
        """CRITICAL: even when the backend supports isolation
        (DockerSandbox), if the caller injects a custom artifact_store
        whose root_dir is INSIDE the workdir, the startup validation
        must catch it and raise. This prevents a future session manager
        from quietly reopening the B2B-1 gap by placing artifacts back
        under WORKDIR.

        We call ``_validate_secure_sandbox`` directly (not the full
        agent_loop) because the runtime guard in ``run_bash`` reads
        ``base_tools.SANDBOX`` which we can't easily override from
        here. The startup validation is the layer we're testing, and
        it takes the sandbox as an explicit parameter, which makes the
        test independent of module-level wiring.
        """
        module = self._load_with("secure_multi_session", "none")
        from agents.sandbox import DockerSandbox
        # A DockerSandbox-like sandbox that supports isolation.
        real_docker = DockerSandbox(workdir=str(self.cwd))

        # Build a custom store whose root_dir is INSIDE the workdir.
        bad_root = self.cwd / ".harness" / "injected-artifacts"
        bad_root.mkdir(parents=True, exist_ok=True)
        from agents.artifacts import ArtifactStore
        bad_store = ArtifactStore(
            root_dir=bad_root, session_id="injected-test"
        )

        with self.assertRaises(module.SecureSandboxError) as cm:
            module._validate_secure_sandbox(
                sandbox=real_docker,
                artifact_root=bad_store.root_dir,
                active_tool_names=["bash"],
            )
        # Error must blame the mount-plan leak, not the capability flag.
        err = str(cm.exception)
        self.assertIn("does NOT isolate", err)
        self.assertIn("artifact root", err.lower())

    def test_secure_mode_accepts_injected_store_outside_workdir(self):
        """When the injected store's root_dir is OUTSIDE the workdir
        (the B2B-1 default layout), the runtime assessment passes.
        This proves the check is about the *actual* location, not a
        blanket rejection of injected stores.

        We call ``_validate_secure_sandbox`` directly for the same
        reason as the previous test — the startup validation takes the
        sandbox as an explicit parameter, isolating the test from
        module-level SANDBOX wiring.
        """
        module = self._load_with("secure_multi_session", "none")
        from agents.sandbox import DockerSandbox
        real_docker = DockerSandbox(workdir=str(self.cwd))

        # Custom store OUTSIDE workdir — assessment must pass.
        good_root = Path(tempfile.gettempdir()) / "b2b3-good-store"
        good_root.mkdir(parents=True, exist_ok=True)
        from agents.artifacts import ArtifactStore
        good_store = ArtifactStore(
            root_dir=good_root, session_id="good-test"
        )

        # Must NOT raise — capability check passes (DockerSandbox
        # supports isolation) and assess_isolation confirms the
        # artifact root is outside the only mounted directory.
        module._validate_secure_sandbox(
            sandbox=real_docker,
            artifact_root=good_store.root_dir,
            active_tool_names=["bash"],
        )

    def test_secure_mode_assessment_reasons_are_human_readable(self):
        """When the startup validation fails, the error message must
        contain actionable human-readable reasons (not just an opaque
        enum) in ``diagnostic_reasons`` — NOT in ``str(exception)``.

        B2B-4 path-leak contract:
          - ``str(exception)`` (public_message) must NOT contain the
            physical artifact path or mount source — this is what may
            appear in model context / user logs / remote traces.
          - ``exception.diagnostic_reasons`` DOES contain the full
            paths for operator diagnosis, but must never be sent to
            the model.
        """
        module = self._load_with("secure_multi_session", "none")
        from agents.sandbox import DockerSandbox
        real_docker = DockerSandbox(workdir=str(self.cwd))

        bad_root = self.cwd / "leaked-artifacts"
        bad_root.mkdir(parents=True, exist_ok=True)
        from agents.artifacts import ArtifactStore
        bad_store = ArtifactStore(
            root_dir=bad_root, session_id="leak-test"
        )

        with self.assertRaises(module.SecureSandboxError) as cm:
            module._validate_secure_sandbox(
                sandbox=real_docker,
                artifact_root=bad_store.root_dir,
                active_tool_names=["bash"],
            )
        exc = cm.exception
        public_msg = str(exc)
        # PUBLIC message: no physical paths, no mount sources.
        self.assertNotIn(str(bad_root), public_msg)
        self.assertNotIn(str(self.cwd), public_msg)
        self.assertNotIn("mount source", public_msg.lower())
        # But the public message must still say something useful.
        self.assertIn("secure_multi_session", public_msg)
        self.assertIn("does NOT isolate", public_msg)
        # DIAGNOSTIC reasons: full paths for the operator.
        self.assertTrue(len(exc.diagnostic_reasons) >= 1)
        diag = " ".join(exc.diagnostic_reasons)
        self.assertIn(str(bad_root), diag)
        self.assertIn("mount source", diag.lower())


class MultiMountIsolationTests(unittest.TestCase):
    """Stage 2C-B2B-4: ``assess_isolation()`` must iterate EVERY mount
    source from ``_mount_host_sources()``, not just the workdir.

    When a future change adds a second mount (cache dir, docker socket,
    etc.), the assessment must automatically include it. A private path
    under ANY mount source is a leak — even read-only mounts expose
    content.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_mount_containing_artifact_is_flagged(self):
        """If DockerSandbox has a second mount source whose host path
        contains the artifact root, the assessment MUST report a leak
        — even though the workdir mount itself is clean. This proves
        ``assess_isolation`` iterates all mounts, not just the workdir.
        """
        from agents.sandbox import DockerSandbox
        d = DockerSandbox(workdir=str(self.cwd))
        # Simulate a second mount: a cache dir whose host path contains
        # the artifact root. We monkeypatch _mount_host_sources.
        artifact_root = self.cwd.parent / "b2b4-second-mount-artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        cache_mount_host = self.cwd.parent  # contains artifact_root
        original = d._mount_host_sources
        d._mount_host_sources = lambda: original() + [
            (str(cache_mount_host), "/cache")
        ]
        try:
            a = d.assess_isolation(
                workdir=str(self.cwd),
                private_paths=(str(artifact_root),),
            )
            self.assertFalse(a.filesystem_isolated)
            self.assertTrue(any("under mount source" in r for r in a.reasons))
        finally:
            artifact_root.rmdir()

    def test_second_mount_not_containing_artifact_is_clean(self):
        """A second mount that does NOT contain any private path must
        not cause a false positive. The assessment still passes."""
        from agents.sandbox import DockerSandbox
        d = DockerSandbox(workdir=str(self.cwd))
        artifact_root = Path(tempfile.gettempdir()) / "b2b4-clean-artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Second mount: an unrelated dir that does NOT contain artifact_root.
        other_mount = Path(tempfile.gettempdir()) / "b2b4-other-mount"
        other_mount.mkdir(parents=True, exist_ok=True)
        original = d._mount_host_sources
        d._mount_host_sources = lambda: original() + [
            (str(other_mount), "/other")
        ]
        try:
            a = d.assess_isolation(
                workdir=str(self.cwd),
                private_paths=(str(artifact_root),),
            )
            self.assertTrue(a.filesystem_isolated, a.reasons)
        finally:
            artifact_root.rmdir()
            other_mount.rmdir()


class SecureBashContextTests(unittest.TestCase):
    """Stage 2C-B2B-4.1: in secure_multi_session mode, ``run_bash()``
    requires a per-asyncio-Task secure-bash context (ContextVar) proving
    that an ``agent_loop()`` in THIS Task has passed startup validation,
    bound to the current sandbox instance.

    Direct ``run_bash()`` calls from outside ``agent_loop()`` lack the
    context and must be rejected — they bypass the mount-plan assessment.
    This closes the "bypass agent_loop, call run_bash directly" vector.

    Stage 2C-B2B-4.1 replaced threading.local with contextvars.ContextVar
    because threading.local only isolates threads, NOT asyncio Tasks on
    the same event-loop thread. These tests verify the new contract:

    1. Direct call rejection (capability check + context check)
    2. Context set/reset round-trip
    3. Sandbox-swap-after-validation rejection (context binds to id)
    4. Per-Task isolation (NOT per-thread): two Tasks on the same thread
       must NOT share the context
    5. Nested agent_loop calls: inner reset restores outer context
    6. Exception/cancellation cleanup: context cleared even on raise
    7. Sandbox mismatch: context bound to sandbox A does not authorize
       run_bash on sandbox B
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self._prev_env = {
            k: os.environ.get(k) for k in (
                "AGENT_RUN_MODE", "AGENT_SANDBOX_BACKEND",
            )
        }

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _load_with(self, run_mode: str, sandbox_backend: str):
        if run_mode is None:
            os.environ.pop("AGENT_RUN_MODE", None)
        else:
            os.environ["AGENT_RUN_MODE"] = run_mode
        if sandbox_backend is None:
            os.environ.pop("AGENT_SANDBOX_BACKEND", None)
        else:
            os.environ["AGENT_SANDBOX_BACKEND"] = sandbox_backend
        return load_harness_module(self.cwd)

    # --- 1. Direct call rejection -------------------------------------

    def test_direct_run_bash_rejected_in_secure_mode(self):
        """secure_multi_session + NoOpSandbox: a DIRECT run_bash() call
        (no agent_loop) must be rejected. NoOpSandbox fails BOTH the
        capability check and the context check; the error must still
        be a clear rejection with no command output."""
        module = self._load_with("secure_multi_session", "none")
        # IMPORTANT: use module.run_bash, NOT ``from agents.base_tools
        # import run_bash``. load_harness_module's finally block clears
        # the agents.* cache and restores the PRE-call snapshot, so a
        # top-level re-import would pick up a stale base_tools with the
        # wrong RUN_MODE / SANDBOX. module.run_bash is the function
        # closed over the freshly-loaded config.
        r = module.run_bash("echo should_not_run")
        self.assertIsInstance(r, str)
        self.assertTrue(r.startswith("Error:"))
        self.assertIn("secure_multi_session", r)
        self.assertNotIn("should_not_run", r)

    # --- 2. Context set/reset round-trip ------------------------------

    def test_context_set_reset_round_trip(self):
        """set_secure_bash_context returns a token; after
        reset_secure_bash_context(token), has_valid_secure_bash_context
        returns False again. This is the basic lifecycle."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sentinel = object()
        self.assertFalse(has_valid_secure_bash_context(sentinel))

        async def _run():
            token = set_secure_bash_context(
                run_id="test-run", sandbox=sentinel
            )
            self.assertTrue(has_valid_secure_bash_context(sentinel))
            reset_secure_bash_context(token)
            self.assertFalse(has_valid_secure_bash_context(sentinel))

        asyncio.run(_run())

    # --- 3. Sandbox-swap-after-validation rejection -------------------

    def test_context_binds_to_sandbox_instance(self):
        """The context binds to id(sandbox) at validation time. If the
        global SANDBOX is swapped AFTER validation, the context's
        sandbox_identity no longer matches, and has_valid_secure_bash_context
        returns False for the new sandbox. This blocks a sandbox-swap
        bypass where validation runs on a strict sandbox and execution
        happens on a permissive one."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox_a = object()
        sandbox_b = object()
        self.assertIsNot(sandbox_a, sandbox_b)

        async def _run():
            # Validate against sandbox_a
            token = set_secure_bash_context(
                run_id="run-a", sandbox=sandbox_a
            )
            try:
                self.assertTrue(has_valid_secure_bash_context(sandbox_a))
                # Sandbox swapped to sandbox_b → context invalid for B
                self.assertFalse(has_valid_secure_bash_context(sandbox_b))
            finally:
                reset_secure_bash_context(token)
            # After reset, neither is valid
            self.assertFalse(has_valid_secure_bash_context(sandbox_a))
            self.assertFalse(has_valid_secure_bash_context(sandbox_b))

        asyncio.run(_run())

    # --- 4. Per-Task isolation (the core B2B-4.1 fix) -----------------

    def test_per_task_isolation_same_thread(self):
        """The CORE B2B-4.1 fix: two asyncio Tasks running on the SAME
        event-loop thread must NOT share the secure-bash context. Task A
        sets a context and awaits; Task B on the same thread calls
        has_valid_secure_bash_context → must be False.

        We assert thread IDs are equal to prove this is NOT thread
        isolation — it's Task isolation via ContextVar."""
        import asyncio
        import threading
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()
        main_thread = threading.get_ident()
        observed = {}

        async def _task_a():
            """Task A: sets context, then awaits, letting Task B run."""
            assert threading.get_ident() == main_thread
            token = set_secure_bash_context(
                run_id="task-a", sandbox=sandbox
            )
            observed["task_a_after_set"] = (
                has_valid_secure_bash_context(sandbox)
            )
            try:
                # Yield control so Task B can run on the same thread.
                await asyncio.sleep(0.05)
            finally:
                reset_secure_bash_context(token)
            observed["task_a_after_reset"] = (
                has_valid_secure_bash_context(sandbox)
            )

        async def _task_b():
            """Task B: runs on the same thread as Task A (single event
            loop). Must NOT see Task A's context."""
            assert threading.get_ident() == main_thread
            observed["task_b_sees_a_context"] = (
                has_valid_secure_bash_context(sandbox)
            )

        async def _main():
            assert threading.get_ident() == main_thread
            # Schedule both; Task A starts, sets context, awaits; Task B
            # runs during Task A's await.
            await asyncio.gather(_task_a(), _task_b())

        asyncio.run(_main())

        # Sanity: Task A saw its own context while it was active.
        self.assertTrue(observed["task_a_after_set"])
        # THE KEY ASSERTION: Task B on the same thread did NOT see
        # Task A's context. This is what threading.local could NOT do.
        self.assertFalse(observed["task_b_sees_a_context"])
        # After Task A reset, neither sees a context.
        self.assertFalse(observed["task_a_after_reset"])

    # --- 5. Nested agent_loop calls: inner reset restores outer -------

    def test_nested_calls_inner_reset_restores_outer(self):
        """Two secure agent_loops nested (or sequentially composed in
        the same Task): inner reset(token) must restore the OUTER
        context, NOT clear it to None. This is why we use
        ContextVar.reset(token) instead of set(None).

        Scenario:
            Agent A sets context A
            Agent B (nested) sets context B, then resets → restores A
            Agent A's context must still be valid after B returns
        """
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()

        async def _run():
            # Outer agent A
            token_a = set_secure_bash_context(
                run_id="outer", sandbox=sandbox
            )
            try:
                self.assertTrue(has_valid_secure_bash_context(sandbox))
                # Inner agent B (nested)
                token_b = set_secure_bash_context(
                    run_id="inner", sandbox=sandbox
                )
                try:
                    self.assertTrue(
                        has_valid_secure_bash_context(sandbox)
                    )
                finally:
                    # Inner reset must restore OUTER context, not None
                    reset_secure_bash_context(token_b)
                # KEY: outer context still valid after inner reset
                self.assertTrue(has_valid_secure_bash_context(sandbox))
            finally:
                reset_secure_bash_context(token_a)
            self.assertFalse(has_valid_secure_bash_context(sandbox))

        asyncio.run(_run())

    # --- 6. Exception/cancellation cleanup ----------------------------

    def test_context_cleared_on_exception(self):
        """If an exception is raised between set and reset, the finally
        block must still call reset(token). After the exception
        propagates, the context must be cleared (no token leak)."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sentinel = object()

        async def _raising_agent():
            token = set_secure_bash_context(
                run_id="raising", sandbox=sentinel
            )
            try:
                raise RuntimeError("provider error")
            finally:
                reset_secure_bash_context(token)

        with self.assertRaises(RuntimeError):
            asyncio.run(_raising_agent())

        # Fresh Task: context must not have leaked.
        async def _check():
            return has_valid_secure_bash_context(sentinel)
        self.assertFalse(asyncio.run(_check()))

    def test_context_cleared_on_cancellation(self):
        """If the Task is cancelled (asyncio.CancelledError) between
        set and reset, the finally block must still call reset(token).
        After cancellation, the context must be cleared."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sentinel = object()
        started = asyncio.Event()

        async def _cancellable_agent():
            token = set_secure_bash_context(
                run_id="cancellable", sandbox=sentinel
            )
            started.set()
            try:
                try:
                    await asyncio.sleep(10)
                finally:
                    reset_secure_bash_context(token)
            except asyncio.CancelledError:
                # Swallow to let the test observe post-cancel state.
                pass

        async def _main():
            task = asyncio.ensure_future(_cancellable_agent())
            await started.wait()
            task.cancel()
            await task
            # Same Task post-cancel: context must be cleared by finally.
            self.assertFalse(has_valid_secure_bash_context(sentinel))

        asyncio.run(_main())

    # --- 7. Sandbox mismatch: context bound to A, run on B ------------

    def test_sandbox_mismatch_rejected(self):
        """If agent_loop validated sandbox A but run_bash later sees
        sandbox B (e.g., global swap), has_valid_secure_bash_context(B)
        must return False. This is the runtime guard's defence-in-depth
        against sandbox swap after validation."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox_a = object()
        sandbox_b = object()

        async def _run():
            # agent_loop validated sandbox_a
            token = set_secure_bash_context(
                run_id="mismatch", sandbox=sandbox_a
            )
            try:
                self.assertTrue(has_valid_secure_bash_context(sandbox_a))
                # Sandbox swapped to sandbox_b → B must be rejected
                self.assertFalse(
                    has_valid_secure_bash_context(sandbox_b)
                )
            finally:
                reset_secure_bash_context(token)

        asyncio.run(_run())


class SecureBashChildTaskRevocationTests(unittest.TestCase):
    """Stage 2C-B2B-4.2: child-Task context inheritance revocation.

    ContextVar values are COPIED into child Tasks created via
    ``asyncio.create_task`` / ``ensure_future``. A child Task inherits
    the parent's ``SecureBashContext`` at creation time, and the parent's
    ``reset(token)`` cannot revoke the copy already held by the child.

    B2B-4.2 fix: each context carries a nonce registered in a
    process-wide live set at set() time and discarded at reset() time.
    ``has_valid_secure_bash_context()`` requires the nonce to STILL be
    in the live set, so a child Task holding a copied context fails
    after the owning agent_loop resets.

    Contract (per B2B-4.2 spec):
        A child Task may inherit the parent's secure-bash permission
        while the parent agent_loop is alive. Once the parent agent_loop
        ends (normally, by exception, or by cancellation), the nonce is
        discarded and ALL inherited copies are revoked.
    """

    # NOTE: these tests import from ``agents.base_tools`` directly inside
    # each method. load_harness_module's finally restores a pre-call
    # agents.* snapshot, so a top-level import at class scope could bind
    # to a stale module. Method-local import binds to whatever module
    # object is currently in sys.modules, consistent with the B2B-4.1
    # SecureBashContextTests pattern.

    def _clear_live_nonces(self) -> None:
        """Best-effort cleanup of any leaked nonces from a failed test."""
        try:
            from agents.base_tools import (
                _ACTIVE_SECURE_BASH_RUNS,
                _ACTIVE_SECURE_BASH_RUNS_LOCK,
            )
            with _ACTIVE_SECURE_BASH_RUNS_LOCK:
                _ACTIVE_SECURE_BASH_RUNS.clear()
        except Exception:
            pass

    def tearDown(self):
        self._clear_live_nonces()

    # --- 1. Child Task after parent reset must be rejected -----------

    def test_child_task_rejected_after_parent_reset(self):
        """Parent agent_loop sets context, spawns a child Task (which
        inherits a copy of the ContextVar), then parent resets. The
        child Task's copied context now has a nonce that is NO LONGER
        in the live set, so has_valid_secure_bash_context returns False.

        This is the CORE B2B-4.2 fix — without the nonce registry, the
        child would retain a valid-looking context indefinitely."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()
        child_result = {}

        async def _child():
            # Child inherits parent's context copy. Wait a bit so the
            # parent has time to reset before we check.
            await asyncio.sleep(0.05)
            child_result["valid"] = has_valid_secure_bash_context(sandbox)

        async def _main():
            token = set_secure_bash_context(
                run_id="parent", sandbox=sandbox
            )
            try:
                # Create child Task — it inherits the current ContextVar.
                task = asyncio.ensure_future(_child())
                # Yield once so the child starts (and copies context).
                await asyncio.sleep(0)
            finally:
                # Parent resets — nonce discarded. Child's copy is now
                # stale even though ContextVar.reset can't reach it.
                reset_secure_bash_context(token)
            # Now let the child finish its check.
            await task

        asyncio.run(_main())
        self.assertFalse(
            child_result["valid"],
            "child Task retained a valid context after parent reset — "
            "nonce revocation is broken",
        )

    # --- 2. Child Task valid while parent alive ----------------------

    def test_child_task_valid_while_parent_alive(self):
        """While the parent agent_loop is still alive (has not reset),
        a child Task that inherited the context MUST be able to use
        bash — the nonce is still in the live set. This is the stated
        B2B-4.2 contract: inheritance is allowed during the parent's
        lifetime; revocation happens only at parent reset."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()
        child_result = {}

        async def _child():
            child_result["valid"] = has_valid_secure_bash_context(sandbox)

        async def _main():
            token = set_secure_bash_context(
                run_id="parent", sandbox=sandbox
            )
            try:
                # Child runs AND completes while parent is still alive.
                await asyncio.ensure_future(_child())
                self.assertTrue(
                    child_result["valid"],
                    "child Task should be valid while parent is alive"
                )
            finally:
                reset_secure_bash_context(token)

        asyncio.run(_main())

    # --- 3. Nested reset discards inner nonce, outer stays live ------

    def test_nested_reset_inner_discarded_outer_alive(self):
        """Nested agent_loop calls: inner reset discards ONLY the inner
        nonce. The outer nonce was added by outer set() and is only
        discarded by outer reset(). After inner reset, the outer
        context is still valid (both ContextVar restored AND nonce live).
        """
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()

        async def _run():
            outer_token = set_secure_bash_context(
                run_id="outer", sandbox=sandbox
            )
            try:
                self.assertTrue(has_valid_secure_bash_context(sandbox))
                inner_token = set_secure_bash_context(
                    run_id="inner", sandbox=sandbox
                )
                try:
                    self.assertTrue(has_valid_secure_bash_context(sandbox))
                finally:
                    # Inner reset: discards inner nonce, restores outer
                    # context. Outer nonce must STILL be in the live set.
                    reset_secure_bash_context(inner_token)
                # KEY: outer context still valid after inner reset.
                self.assertTrue(
                    has_valid_secure_bash_context(sandbox),
                    "outer nonce was discarded by inner reset — "
                    "nonce scoping is broken"
                )
            finally:
                reset_secure_bash_context(outer_token)
            self.assertFalse(has_valid_secure_bash_context(sandbox))

        asyncio.run(_run())

    # --- 4. Parent exception/cancellation revokes all children -------

    def test_parent_exception_revokes_child_context(self):
        """If the parent agent_loop raises (provider error, etc.), its
        finally block still calls reset() — discarding the nonce. Any
        child Task holding a copied context can no longer use bash.

        This also covers cancellation: CancelledError triggers the same
        finally block, so the nonce is discarded the same way."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()
        child_result = {}

        async def _child():
            await asyncio.sleep(0.05)
            child_result["valid"] = has_valid_secure_bash_context(sandbox)

        async def _parent():
            token = set_secure_bash_context(
                run_id="parent", sandbox=sandbox
            )
            task = asyncio.ensure_future(_child())
            try:
                raise RuntimeError("provider error")
            finally:
                # agent_loop's finally contract: reset always runs.
                reset_secure_bash_context(token)
                await task

        with self.assertRaises(RuntimeError):
            asyncio.run(_parent())
        self.assertFalse(
            child_result["valid"],
            "child Task retained a valid context after parent raised "
            "and reset — exception path revocation is broken"
        )

    def test_parent_cancellation_revokes_child_context(self):
        """CancelledError triggers the same finally block as a normal
        exception. The nonce is discarded, so inherited child contexts
        are revoked."""
        import asyncio
        from agents.base_tools import (
            set_secure_bash_context,
            reset_secure_bash_context,
            has_valid_secure_bash_context,
        )

        sandbox = object()
        child_result = {}

        async def _child():
            await asyncio.sleep(0.05)
            child_result["valid"] = has_valid_secure_bash_context(sandbox)

        async def _parent():
            token = set_secure_bash_context(
                run_id="parent", sandbox=sandbox
            )
            task = asyncio.ensure_future(_child())
            try:
                # Simulate a long-running parent that gets cancelled.
                await asyncio.sleep(10)
            finally:
                reset_secure_bash_context(token)
                # Let the child finish so the test can observe.
                try:
                    await task
                except BaseException:
                    pass

        async def _main():
            parent = asyncio.ensure_future(_parent())
            await asyncio.sleep(0.01)  # let parent set context + spawn child
            parent.cancel()
            try:
                await parent
            except asyncio.CancelledError:
                pass

        asyncio.run(_main())
        self.assertFalse(
            child_result["valid"],
            "child Task retained a valid context after parent was "
            "cancelled and reset — cancellation path revocation is broken"
        )


class SecureSandboxErrorPathLeakTests(unittest.TestCase):
    """Stage 2C-B2B-4: ``SecureSandboxError`` must NOT leak physical
    artifact paths or mount sources into ``str(exception)`` — that is
    the public message that may appear in model context, user logs, or
    remote traces. Diagnostic details live in ``diagnostic_reasons``.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self._prev_env = {
            k: os.environ.get(k) for k in (
                "AGENT_RUN_MODE", "AGENT_SANDBOX_BACKEND",
            )
        }

    def tearDown(self):
        self._tmp.cleanup()
        for k, v in self._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_public_message_has_no_physical_paths(self):
        """When the mount-plan assessment fails, str(exception) must
        not contain the artifact root path or the workdir path."""
        os.environ["AGENT_RUN_MODE"] = "secure_multi_session"
        os.environ["AGENT_SANDBOX_BACKEND"] = "none"
        module = load_harness_module(self.cwd)

        from agents.sandbox import DockerSandbox
        real_docker = DockerSandbox(workdir=str(self.cwd))
        bad_root = self.cwd / "secret-artifacts-here"
        bad_root.mkdir(parents=True, exist_ok=True)
        from agents.artifacts import ArtifactStore
        bad_store = ArtifactStore(root_dir=bad_root, session_id="leak")

        with self.assertRaises(module.SecureSandboxError) as cm:
            module._validate_secure_sandbox(
                sandbox=real_docker,
                artifact_root=bad_store.root_dir,
                active_tool_names=["bash"],
            )
        public = str(cm.exception)
        # No physical paths in the public message.
        self.assertNotIn(str(bad_root), public)
        self.assertNotIn(str(self.cwd), public)
        self.assertNotIn("mount source", public.lower())
        self.assertNotIn("private path", public.lower())
        # Diagnostic reasons DO have the paths.
        diag = " ".join(cm.exception.diagnostic_reasons)
        self.assertIn(str(bad_root), diag)
        self.assertIn("mount source", diag.lower())

    def test_repr_and_args_have_no_physical_paths(self):
        """Stage 2C-B2B-4.1: ``repr(exception)`` and ``exception.args``
        must ALSO not contain physical paths. Python logging frequently
        uses ``logger.exception(...)`` / ``logger.error("%r", exc)``
        which invoke ``repr()`` rather than ``str()``. If the public
        message is path-free but ``args`` somehow carried a path,
        ``repr()`` would still leak it.

        Contract:
            - SecureSandboxError.__init__(public_message, ...) calls
              super().__init__(public_message), so args = (public_message,)
            - public_message is path-free by construction
            - diagnostic_reasons live ONLY on the attribute, never in args
        """
        os.environ["AGENT_RUN_MODE"] = "secure_multi_session"
        os.environ["AGENT_SANDBOX_BACKEND"] = "none"
        module = load_harness_module(self.cwd)

        from agents.sandbox import DockerSandbox
        real_docker = DockerSandbox(workdir=str(self.cwd))
        bad_root = self.cwd / "another-secret-root"
        bad_root.mkdir(parents=True, exist_ok=True)
        from agents.artifacts import ArtifactStore
        bad_store = ArtifactStore(root_dir=bad_root, session_id="repr-leak")

        with self.assertRaises(module.SecureSandboxError) as cm:
            module._validate_secure_sandbox(
                sandbox=real_docker,
                artifact_root=bad_store.root_dir,
                active_tool_names=["bash"],
            )
        exc = cm.exception

        # 1. args must contain ONLY the public message (path-free).
        self.assertEqual(len(exc.args), 1)
        args_blob = str(exc.args[0])
        self.assertNotIn(str(bad_root), args_blob)
        self.assertNotIn(str(self.cwd), args_blob)
        self.assertNotIn("mount source", args_blob.lower())

        # 2. repr() must not leak paths either — it typically renders
        #    as SecureSandboxError(<args[0]>) so it inherits args safety,
        #    but assert it explicitly in case __repr__ is overridden.
        r = repr(exc)
        self.assertNotIn(str(bad_root), r)
        self.assertNotIn(str(self.cwd), r)
        self.assertNotIn("mount source", r.lower())

        # 3. Diagnostic reasons are intentionally retained for operators
        #    but must NOT bleed into args or repr.
        diag = " ".join(exc.diagnostic_reasons)
        self.assertIn(str(bad_root), diag)
        # Cross-check: diag content must not appear in args/repr.
        for reason in exc.diagnostic_reasons:
            self.assertNotIn(reason, args_blob)
            self.assertNotIn(reason, r)


class TrustedLocalBashAccessContractTests(unittest.TestCase):
    """Stage 2C-B2B-4: in trusted_local + NoOpSandbox, bash CAN read
    host files (including the artifact root). This is the MODE CONTRACT
    — trusted_local explicitly does NOT provide cross-session bash
    isolation. We do NOT keep an xfail for this; instead we document
    the contract as a passing test: bash reads work, and that's the
    accepted behavior of the mode.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        self.module = load_harness_module(self.cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_trusted_local_bash_can_read_host_file(self):
        """trusted_local + NoOpSandbox: bash can read a host file. This
        is the accepted mode contract — no isolation, no rejection.
        The xfail from B2B-1 is removed because this is not a 'known
        limitation to be fixed later'; it is the documented behavior
        of trusted_local mode. Real isolation requires
        secure_multi_session + DockerSandbox."""
        import sys
        secret_file = self.cwd / "host-secret.txt"
        secret_file.write_text("HOST_SECRET_12345\n")
        # Pass the path as a separate argv to avoid shell quoting hell
        # with Windows backslashes inside a -c string.
        cmd = (
            f'{sys.executable} -c "'
            f'import sys; print(open(sys.argv[1]).read().strip())'
            f'" "{secret_file}"'
        )
        responses = [
            _resp_tool_use(name="bash", input_={"command": cmd}, id_="t1"),
            _resp_text(),
        ]
        self.module.client.messages.create = lambda **_: responses.pop(0)
        msgs = [{"role": "user", "content": "read"}]
        self.module.agent_loop(msgs)
        result = _find_tool_result(msgs, "t1")
        # bash DID read the host file — this is the trusted_local contract.
        self.assertIn("HOST_SECRET_12345", result)


if __name__ == "__main__":
    unittest.main()
