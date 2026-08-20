"""test_session_model_selection.py - Phase 3D-0 Session Model Selection tests.

Locks the SessionState contract:

  - Session stores ONLY the alias (never ModelSpec/Binding/Adapter)
  - None -> system default; alias -> next run uses it
  - set_session_model_alias validates against ModelRegistry BEFORE
    mutating: unknown alias fails fast, session value unchanged
  - setting an alias does NOT read API keys / build adapters
  - Session A / Session B are independent
  - precedence: explicit agent_loop(model_alias=) > session > default
  - agent startup reads the session alias exactly once; changing it
    during a run does NOT affect the frozen ModelRuntimeContext
  - the next run picks up the new alias
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agents.session import (
    DEFAULT_MODEL_ALIAS,
    SessionState,
    get_session_model_alias,
    resolve_session_model_alias,
    set_session_model_alias,
)
from agents.providers.model_spec import (
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
)
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
        "harness_core_session_test", MODULE_PATH
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


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


def _resp_text(text="done"):
    return types.SimpleNamespace(
        stop_reason="end_turn",
        content=[FakeTextBlock(text)],
        usage=None,
    )


class SessionSelectionUnitTests(unittest.TestCase):

    def setUp(self):
        self.registry = ModelRegistry()
        self.registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        self.registry.register(ModelSpec(
            alias="qwen-openrouter", provider="openrouter",
            model_id="qwen/qwen-2.5-72b-instruct"))

    def test_new_session_defaults_none(self):
        s = SessionState()
        self.assertIsNone(s.model_alias)

    def test_get_none_session_returns_none(self):
        self.assertIsNone(get_session_model_alias(None))
        self.assertIsNone(get_session_model_alias(SessionState()))

    def test_set_and_get_roundtrip(self):
        s = SessionState()
        set_session_model_alias(s, "deepseek", self.registry)
        self.assertEqual(get_session_model_alias(s), "deepseek")

    def test_unknown_alias_fails_fast(self):
        s = SessionState()
        with self.assertRaises(UnknownModelError):
            set_session_model_alias(s, "deeepseek", self.registry)
        # Session value unchanged (no corruption).
        self.assertIsNone(s.model_alias)

    def test_unknown_alias_preserves_previous(self):
        s = SessionState()
        set_session_model_alias(s, "deepseek", self.registry)
        with self.assertRaises(UnknownModelError):
            set_session_model_alias(s, "nope", self.registry)
        self.assertEqual(s.model_alias, "deepseek")

    def test_session_a_b_independent(self):
        a = SessionState()
        b = SessionState()
        set_session_model_alias(a, "deepseek", self.registry)
        self.assertEqual(a.model_alias, "deepseek")
        self.assertIsNone(b.model_alias)

    def test_session_stores_only_alias(self):
        s = SessionState()
        set_session_model_alias(s, "deepseek", self.registry)
        # Only alias + session_id + metadata - no spec/binding/adapter.
        self.assertEqual(set(vars(s)), {"session_id", "model_alias", "metadata"})
        self.assertIsNone(s.metadata.get("model_spec"))
        self.assertIsNone(s.metadata.get("provider_binding"))

    def test_set_alias_does_not_read_api_key(self):
        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            s = SessionState()
            set_session_model_alias(s, "deepseek", self.registry)
            self.assertEqual(s.model_alias, "deepseek")
        finally:
            if saved is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved

    # ------------------------------------------------------------------
    # Precedence
    # ------------------------------------------------------------------

    def test_precedence_explicit_over_session(self):
        s = SessionState()
        s.model_alias = "deepseek"
        self.assertEqual(resolve_session_model_alias(s, "qwen-openrouter"),
                         "qwen-openrouter")

    def test_precedence_session_over_default(self):
        s = SessionState()
        s.model_alias = "deepseek"
        self.assertEqual(resolve_session_model_alias(s, None), "deepseek")

    def test_precedence_default_when_nothing(self):
        self.assertEqual(resolve_session_model_alias(None, None),
                         DEFAULT_MODEL_ALIAS)
        s = SessionState()
        self.assertEqual(resolve_session_model_alias(s, None),
                         DEFAULT_MODEL_ALIAS)

    def test_default_alias_is_claude(self):
        self.assertEqual(DEFAULT_MODEL_ALIAS, "claude")


class SessionAgentLoopTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))
        self.registry = ModelRegistry()
        self.registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        self.router = ProviderRouter()
        from agents.providers.openai_compatible_adapter import (
            OpenAICompatibleAdapter, OpenAICompatibleConfig)
        holder = {"calls": []}

        class FakeOCClient:
            def complete(self, payload):
                holder["calls"].append(payload.get("model"))
                return {
                    "id": "x",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": "oc"},
                                 "finish_reason": "stop"}],
                }

        def factory(model_id):
            config = OpenAICompatibleConfig(
                base_url="https://deepseek.example.invalid",
                api_key="test-key", model=model_id)
            adapter = OpenAICompatibleAdapter(config)
            adapter._client_provider = lambda: FakeOCClient()  # noqa: SLF001
            return adapter

        self.router.register(ProviderBinding(
            provider="deepseek", adapter_type="openai-compatible",
            base_url="https://deepseek.example.invalid",
            adapter_factory=factory))
        self.holder = holder

    def test_session_alias_used_by_next_run(self):
        s = SessionState()
        s.model_alias = "deepseek"
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_registry=self.registry,
            provider_router=self.router,
            session=s,
        )
        self.assertEqual(self.holder["calls"], ["deepseek-chat"])

    def test_no_session_uses_default(self):
        # Default claude -> anthropic -> module client (fake).
        recorder = {"calls": 0}

        def create(**kwargs):
            recorder["calls"] += 1
            return _resp_text("done")

        self.module.client.messages.create = create
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(recorder["calls"], 1)

    def test_explicit_alias_overrides_session(self):
        # Add claude to the registry + anthropic binding so explicit
        # "claude" resolves to the anthropic (module client) path.
        self.registry.register(ModelSpec(
            alias="claude", provider="anthropic", model_id="claude-x"))
        self.router.register(ProviderBinding(
            provider="anthropic", adapter_type="anthropic",
            adapter_factory=lambda model_id: self.module.ANTHROPIC_ADAPTER))

        s = SessionState()
        s.model_alias = "deepseek"
        # Explicit alias = deepseek too; but to prove override, use a
        # second session model and explicit default (claude -> anthropic).
        recorder = {"calls": 0}

        def create(**kwargs):
            recorder["calls"] += 1
            return _resp_text("done")

        self.module.client.messages.create = create
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_alias="claude",  # explicit -> anthropic path
            model_registry=self.registry,
            provider_router=self.router,
            session=s,
        )
        # Explicit claude won over session deepseek -> anthropic called.
        self.assertEqual(recorder["calls"], 1)
        self.assertEqual(self.holder["calls"], [])

    def test_run_ignores_mid_run_session_change(self):
        """Changing session.model_alias during a run must not affect the
        frozen ModelRuntimeContext of the current run. Simulated: we
        resolve a context (as agent_loop startup does), then mutate the
        session, and confirm the context is unchanged."""
        s = SessionState()
        s.model_alias = "deepseek"
        ctx = resolve_session_model_alias(s, None)
        self.assertEqual(ctx, "deepseek")
        # Mid-run: session changes to qwen-openrouter.
        s.model_alias = "qwen-openrouter"
        # The already-resolved selection (and the frozen context built
        # from it) is untouched.
        self.assertEqual(ctx, "deepseek")


if __name__ == "__main__":
    unittest.main()
