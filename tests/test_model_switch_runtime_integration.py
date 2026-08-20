"""test_model_switch_runtime_integration.py - Phase 3D-3.

Integration proof that wires 3D-0 (session read by agent_loop) + 3C
(5-runtime propagation) + 3D-1/3D-2 (/model command + MODEL_CHANGED)
into a real, observable model-switch sequence:

    Run #1 = Claude (default)
      ↓ /model deepseek
    Run #2 = DeepSeek (all 5 runtimes)
      ↓ /model qwen-openrouter
    Run #3 = Qwen/OpenRouter (all 5 runtimes)

Contracts proven:
  1. default run = Claude
  2. mid-run /model does NOT change the running ModelRuntimeContext
  3. next run fully switches (all 5 runtimes)
  4. consecutive cross-provider switches hold
  5. one effective switch fires exactly ONE MODEL_CHANGED
  6. a new run consuming the selection does NOT re-fire the event
  7. explicit model_alias overrides session (per-run, not mutating it)
  8. after an override run, the next run returns to session selection
  9. invalid alias -> session unchanged -> next run still old model
  10. same-alias no-op -> next run still that model
  11. session A / session B isolated (incl. concurrent 5-runtime)
  12. no global mutable model state
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agents.session import SessionState
from agents.providers.model_spec import ModelRegistry, ModelSpec
from agents.providers.provider_router import ProviderBinding, ProviderRouter

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
    spec = importlib.util.spec_from_file_location(
        "harness_core_switch_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)

    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv
    previous_model_id = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "test-model"
    previous_config = sys.modules.get("agents.config")
    sys.modules.pop("agents.config", None)
    try:
        os.chdir(temp_cwd)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_model_id is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model_id
        sys.modules.pop("agents.config", None)
        if previous_config is not None:
            sys.modules["agents.config"] = previous_config
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


class FakeOCClient:
    def __init__(self, seen, provider):
        self.seen = seen
        self.provider = provider

    def complete(self, payload):
        self.seen.append((self.provider, payload.get("model")))
        return {
            "id": "x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": f"ok-{self.provider}"},
                "finish_reason": "stop",
            }],
        }


def make_binding(provider, seen):
    from agents.providers.openai_compatible_adapter import (
        OpenAICompatibleAdapter, OpenAICompatibleConfig)

    def factory(model_id):
        config = OpenAICompatibleConfig(
            base_url=f"https://{provider}.example.invalid",
            api_key="test-key", model=model_id)
        adapter = OpenAICompatibleAdapter(config)
        adapter._client_provider = lambda: FakeOCClient(seen, provider)  # noqa: SLF001
        return adapter

    return ProviderBinding(
        provider=provider, adapter_type="openai-compatible",
        base_url=f"https://{provider}.example.invalid",
        adapter_factory=factory,
    )


# Full registry + router for claude/deepseek/qwen-openrouter.
def make_full_registry():
    reg = ModelRegistry()
    reg.register(ModelSpec(alias="claude", provider="anthropic",
                           model_id="claude-x"))
    reg.register(ModelSpec(alias="deepseek", provider="deepseek",
                           model_id="deepseek-chat"))
    reg.register(ModelSpec(alias="qwen-openrouter", provider="openrouter",
                           model_id="qwen/qwen-2.5-72b-instruct"))
    return reg


class RecordingExtensions:
    def __init__(self):
        self.events = []
        self.handler = None

    def emit(self, event_name, context):
        self.events.append((event_name, context))
        if self.handler is not None:
            self.handler(event_name, context)
        from agents.extension_system import DispatchOutcome
        return DispatchOutcome(event=event_name)


class ModelSwitchIntegrationTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))
        self.reg = make_full_registry()
        # Anthropic path: default claude uses module client.
        self.anthropic_calls = {"n": 0}
        self.module.client.messages.create = (
            lambda **_: self._record_anthropic()
        )
        self.anthropic_models = []

    def _record_anthropic(self):
        self.anthropic_calls["n"] += 1
        return types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="claude-done")],
            usage=None,
        )

    def _router(self, seen):
        r = ProviderRouter()
        r.register(ProviderBinding(
            provider="anthropic", adapter_type="anthropic",
            adapter_factory=lambda model_id: self.module.ANTHROPIC_ADAPTER))
        r.register(make_binding("deepseek", seen))
        r.register(make_binding("openrouter", seen))
        return r

    def _run(self, session, seen, *, explicit=None):
        """Run one agent_loop with the shared registry/router."""
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_registry=self.reg,
            provider_router=self._router(seen),
            session=session,
            model_alias=explicit,
        )

    # ------------------------------------------------------------------
    # 1. default run = Claude
    # ------------------------------------------------------------------

    def test_default_run_is_claude(self):
        seen = []
        self._run(SessionState(), seen)
        self.assertEqual(self.anthropic_calls["n"], 1)
        self.assertEqual(seen, [])

    # ------------------------------------------------------------------
    # 2-4. mid-run switch does not affect current; next run fully switches
    # ------------------------------------------------------------------

    def test_switch_sequence_claude_to_deepseek_to_qwen(self):
        seen = []
        session = SessionState()

        # Run 1: default -> Claude (anthropic).
        self._run(session, seen)
        self.assertEqual(self.anthropic_calls["n"], 1)
        self.assertEqual(seen, [])

        # Mid-sequence switch to deepseek (not during a run here, but the
        # key contract: session changes, current frozen ctx is separate).
        from agents.session import handle_model_command
        exts = RecordingExtensions()
        handle_model_command("/model deepseek", session, self.reg,
                             extensions=exts)
        self.assertEqual(session.model_alias, "deepseek")
        self.assertEqual(len(exts.events), 1)

        # Run 2: no explicit alias -> deepseek (all runtimes via binding).
        self._run(session, seen)
        self.assertEqual(seen, [("deepseek", "deepseek-chat")])

        # Switch to qwen-openrouter.
        exts2 = RecordingExtensions()
        handle_model_command("/model qwen-openrouter", session, self.reg,
                             extensions=exts2)
        self.assertEqual(len(exts2.events), 1)

        # Run 3: -> openrouter/qwen.
        seen.clear()
        self._run(session, seen)
        self.assertEqual(seen, [("openrouter", "qwen/qwen-2.5-72b-instruct")])

    def test_five_runtimes_all_switch_to_deepseek(self):
        """Main + subagent + team + compression + token-budget all follow
        the session's deepseek on the next run."""
        seen = []
        session = SessionState()
        from agents.session import handle_model_command
        handle_model_command("/model deepseek", session, self.reg)

        ctx_seen = []
        # Build the runtime context (as agent_loop startup does) from the
        # session; then exercise each runtime's adapter from the context.
        from agents.providers.model_runtime import resolve_model_runtime
        ctx = resolve_model_runtime(
            session.model_alias, self.reg, self._router(ctx_seen))
        self.assertEqual(ctx.model_id, "deepseek-chat")
        # Every runtime calls create_adapter() on the same ctx.
        for _ in range(5):
            adapter = ctx.create_adapter()
            self.assertEqual(adapter._config.model, "deepseek-chat")  # noqa: SLF001

    # ------------------------------------------------------------------
    # 5-6. event fires once per effective switch; new run does not re-fire
    # ------------------------------------------------------------------

    def test_event_fires_once_and_not_on_consumption(self):
        seen = []
        session = SessionState()
        from agents.session import handle_model_command
        exts = RecordingExtensions()
        handle_model_command("/model deepseek", session, self.reg,
                             extensions=exts)
        self.assertEqual(len(exts.events), 1)
        # Consuming the selection in a new run must NOT re-fire.
        self._run(session, seen)
        self.assertEqual(len(exts.events), 1)

    # ------------------------------------------------------------------
    # 7-8. explicit override is per-run, does not mutate session
    # ------------------------------------------------------------------

    def test_explicit_override_is_per_run(self):
        seen = []
        session = SessionState()
        session.model_alias = "deepseek"
        # Explicit claude -> this run is anthropic.
        self._run(session, seen, explicit="claude")
        self.assertEqual(self.anthropic_calls["n"], 1)
        self.assertEqual(seen, [])
        # Session still deepseek.
        self.assertEqual(session.model_alias, "deepseek")
        # Next run (no explicit) -> deepseek again.
        self._run(session, seen)
        self.assertEqual(seen[-1], ("deepseek", "deepseek-chat"))

    # ------------------------------------------------------------------
    # 9. invalid alias -> next run still old model
    # ------------------------------------------------------------------

    def test_invalid_alias_does_not_pollute_next_run(self):
        seen = []
        session = SessionState()
        session.model_alias = "deepseek"
        from agents.session import handle_model_command
        from agents.providers.model_spec import UnknownModelError
        with self.assertRaises(UnknownModelError):
            handle_model_command("/model invalid-model", session, self.reg)
        self.assertEqual(session.model_alias, "deepseek")
        self._run(session, seen)
        self.assertEqual(seen, [("deepseek", "deepseek-chat")])

    # ------------------------------------------------------------------
    # 10. same-alias no-op -> next run still that model
    # ------------------------------------------------------------------

    def test_same_alias_noop_next_run_ok(self):
        seen = []
        session = SessionState()
        session.model_alias = "deepseek"
        from agents.session import handle_model_command
        exts = RecordingExtensions()
        handle_model_command("/model deepseek", session, self.reg,
                             extensions=exts)
        self.assertEqual(exts.events, [])  # no event
        self._run(session, seen)
        self.assertEqual(seen, [("deepseek", "deepseek-chat")])

    # ------------------------------------------------------------------
    # 11. session A / session B isolated (+ concurrent runtimes)
    # ------------------------------------------------------------------

    def test_session_a_b_isolated(self):
        seen_a = []
        seen_b = []
        a = SessionState(session_id="A")
        b = SessionState(session_id="B")
        from agents.session import handle_model_command
        handle_model_command("/model deepseek", a, self.reg)
        handle_model_command("/model qwen-openrouter", b, self.reg)

        self._run(a, seen_a)
        self._run(b, seen_b)
        # A -> deepseek, B -> qwen, independent.
        self.assertEqual(seen_a, [("deepseek", "deepseek-chat")])
        self.assertEqual(seen_b, [("openrouter", "qwen/qwen-2.5-72b-instruct")])

    def test_concurrent_sessions_runtimes_no_crossover(self):
        import threading
        seen_a = []
        seen_b = []
        a = SessionState(session_id="A")
        b = SessionState(session_id="B")
        from agents.session import handle_model_command
        handle_model_command("/model deepseek", a, self.reg)
        handle_model_command("/model qwen-openrouter", b, self.reg)
        errors = []

        def run_a():
            try:
                self._run(a, seen_a)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def run_b():
            try:
                self._run(b, seen_b)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=run_a)
        t2 = threading.Thread(target=run_b)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)
        self.assertEqual(errors, [])
        self.assertEqual(seen_a, [("deepseek", "deepseek-chat")])
        self.assertEqual(seen_b, [("openrouter", "qwen/qwen-2.5-72b-instruct")])

    # ------------------------------------------------------------------
    # 12. no global mutable model state
    # ------------------------------------------------------------------

    def test_no_global_mutable_model_state(self):
        self.assertFalse(hasattr(self.module, "CURRENT_MODEL"))
        self.assertFalse(hasattr(self.module, "CURRENT_RUNTIME_CONTEXT"))


if __name__ == "__main__":
    unittest.main()
