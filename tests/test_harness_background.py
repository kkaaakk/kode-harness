import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "harness_core.py"


def load_harness_module(temp_cwd: Path):
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
    added_paths = []
    for path in (REPO_ROOT, REPO_ROOT / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_paths.append(text)
    spec = importlib.util.spec_from_file_location("harness_core_under_test", MODULE_PATH)
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
        for path in added_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


class BackgroundManagerTests(unittest.TestCase):
    def test_check_returns_running_placeholder_when_result_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            manager = module.BackgroundManager()
            manager.tasks["abc123"] = {
                "status": "running",
                "command": "sleep 1",
                "result": None,
            }

            self.assertEqual(manager.check("abc123"), "[running] (running)")


class AgentLoopTracingTests(unittest.TestCase):
    def test_agent_loop_records_request_llm_tool_and_final_answer_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            class ToolBlock:
                type = "tool_use"
                name = "bash"
                input = {"command": "echo traced"}
                id = "toolu_trace"

            class TextBlock:
                type = "text"
                text = "done"

            responses = [
                types.SimpleNamespace(stop_reason="tool_use", content=[ToolBlock()]),
                types.SimpleNamespace(stop_reason="end_turn", content=[TextBlock()]),
            ]

            module.client.messages.create = lambda **_: responses.pop(0)
            module.agent_loop([{"role": "user", "content": "please run a command"}])

            trace_dir = Path(tmp) / ".team" / "traces"
            events = []
            for path in sorted(trace_dir.glob("trace_*.jsonl")):
                events.extend(json.loads(line) for line in path.read_text().splitlines())

            event_names = [event["event"] for event in events]
            self.assertIn("agent_request", event_names)
            self.assertGreaterEqual(event_names.count("llm_call"), 2)
            self.assertIn("permission_decision", event_names)
            self.assertIn("tool_call", event_names)
            self.assertIn("final_answer", event_names)
            self.assertEqual({event["trace_id"] for event in events}, {events[0]["trace_id"]})

            tool_event = next(event for event in events if event["event"] == "tool_call")
            self.assertEqual(tool_event["tool_name"], "bash")
            self.assertEqual(tool_event["tool_use_id"], "toolu_trace")
            self.assertEqual(tool_event["status"], "success")
            self.assertIn("traced", tool_event["output_summary"])

            permission_event = next(event for event in events if event["event"] == "permission_decision")
            self.assertEqual(permission_event["tool_name"], "bash")
            self.assertTrue(permission_event["permission_allowed"])
            self.assertEqual(permission_event["permission_rule"], "default_allow")

    def test_tool_registry_derives_public_tools_and_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))

            registry_names = set(module.TOOL_REGISTRY)
            handler_names = set(module.TOOL_HANDLERS)
            visible_tool_names = {tool["name"] for tool in module.TOOLS}

            self.assertIn("bash", registry_names)
            self.assertIn("grep_search", registry_names)
            self.assertIn("glob_search", registry_names)
            self.assertEqual(handler_names, registry_names)
            self.assertTrue(visible_tool_names.issubset(registry_names))
            # idle and claim_task are visible builtins (exposed to the model
            # in the default profile); they ARE in TOOLS.
            self.assertIn("idle", visible_tool_names)
            self.assertIn("claim_task", visible_tool_names)
            self.assertEqual(module.TOOL_REGISTRY["bash"].permission, "shell")

    def test_permission_denial_records_decision_and_skips_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            module.start_trace()

            output, decision = module.execute_registered_tool(
                "bash",
                {"command": "sudo whoami"},
                actor="lead",
                tool_use_id="toolu_denied",
            )

            trace_dir = Path(tmp) / ".team" / "traces"
            events = []
            for path in sorted(trace_dir.glob("trace_*.jsonl")):
                events.extend(json.loads(line) for line in path.read_text().splitlines())

            self.assertFalse(decision.allowed)
            self.assertIn("Permission denied", output)
            permission_event = next(event for event in events if event["event"] == "permission_decision")
            tool_event = next(event for event in events if event["event"] == "tool_call")
            self.assertEqual(permission_event["permission_rule"], "dangerous_command")
            self.assertFalse(permission_event["permission_allowed"])
            self.assertEqual(tool_event["status"], "denied")
            self.assertIn("dangerous_command", tool_event["output_summary"])

    def test_skill_loader_parses_multiline_yaml_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "skills" / "writer"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: writer\n"
                "description: |\n"
                "  Draft concise release notes.\n"
                "  Preserve user terminology.\n"
                "---\n"
                "Use short sections.\n"
            )

            module = load_harness_module(root)
            loader = module.SkillLoader(root / "skills")

            description = loader.skills["writer"]["meta"]["description"]
            self.assertIn("Draft concise release notes.", description)
            self.assertIn("Preserve user terminology.", description)
            self.assertIn("Use short sections.", loader.load("writer"))

    def test_skill_retrieval_ranks_matching_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "skills" / "pdf"
            review_dir = root / "skills" / "code-review"
            pdf_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            (pdf_dir / "SKILL.md").write_text(
                "---\n"
                "name: pdf\n"
                "description: Extract, merge, split, and create PDF documents.\n"
                "---\n"
                "# PDF Processing\n"
                "Use this skill for PDF text extraction and page manipulation.\n"
            )
            (review_dir / "SKILL.md").write_text(
                "---\n"
                "name: code-review\n"
                "description: Review code for correctness, security, maintainability, and tests.\n"
                "---\n"
                "# Code Review\n"
                "Look for regressions and missing test coverage.\n"
            )

            module = load_harness_module(root)
            loader = module.SkillLoader(root / "skills")

            result = json.loads(loader.retrieve("extract text from this pdf", limit=2))

            self.assertEqual(result["candidates"][0]["name"], "pdf")
            self.assertEqual(len(result["candidates"]), 1)
            self.assertIn("description:extract", result["candidates"][0]["reason"])

    def test_skill_router_recommends_load_when_confident(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "skills" / "code-review"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: code-review\n"
                "description: Review code for bugs, security, regressions, and tests.\n"
                "---\n"
                "# Code Review\n"
                "Prioritize actionable findings.\n"
            )

            module = load_harness_module(root)
            loader = module.SkillLoader(root / "skills")

            route = json.loads(loader.route("review this code for security bugs", limit=3))

            self.assertEqual(route["decision"], "load_skill")
            self.assertEqual(route["recommended_skill"], "code-review")
            self.assertIn("load_skill", route["next_action"])

    def test_background_task_records_span_in_current_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load_harness_module(Path(tmp))
            module.start_trace()

            manager = module.BackgroundManager()
            message = manager.run("echo background traced", timeout=5)
            task_id = message.split()[2]

            for _ in range(30):
                if manager.tasks[task_id]["status"] != "running":
                    break
                time.sleep(0.05)

            trace_dir = Path(tmp) / ".team" / "traces"
            events = []
            for path in sorted(trace_dir.glob("trace_*.jsonl")):
                events.extend(json.loads(line) for line in path.read_text().splitlines())

            bg_event = next(event for event in events if event["event"] == "background_task")
            self.assertEqual(bg_event["background_task_id"], task_id)
            self.assertEqual(bg_event["status"], "success")
            self.assertIn("background traced", bg_event["output_summary"])


if __name__ == "__main__":
    unittest.main()
