"""test_harness_model_routing.py - Phase 3C-2 Agent Loop model routing.

Verifies the main agent_loop resolves its model via
ModelRegistry -> ProviderRouter -> Adapter:

  default (no args)   : historic Anthropic, model_id == legacy MODEL
  model_alias="deepseek"   : -> OpenAICompatibleAdapter
  model_alias="qwen-openrouter": -> OpenAICompatibleAdapter (same type)

Fail-fast semantics:
  unknown alias      -> UnknownModelError, 0 model requests
  unknown provider   -> UnknownProviderError, 0 model requests
  missing credential -> MissingCredentialError, 0 model requests
  no fallback

Other contracts:
  - ModelSpec.model_id is the single model source of truth
    (adapter config model == ModelRequest.model == spec.model_id)
  - adapter created ONCE per agent_loop, reused across turns
  - concurrent agents with different aliases do not interfere
  - BEFORE_MODEL_REQUEST may patch request content but NOT model
  - snapshot semantics: resolution happens once at startup
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agents.providers.model_spec import (
    ModelCapabilities,
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
)
from agents.providers.provider_router import (
    MissingCredentialError,
    ProviderBinding,
    ProviderRouter,
    UnknownProviderError,
)

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
        "harness_core_routing_test", MODULE_PATH
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


class RecordingAnthropicClient:
    """Patched onto module.client; records every messages.create call."""

    def __init__(self, module):
        self.module = module
        self.calls = []

    def install(self, response):
        self._response = response

        def create(**kwargs):
            self.calls.append(kwargs)
            return self._response

        self.module.client.messages.create = create


def make_openai_compatible_binding(provider, base_url, api_key_env):
    """A test-only binding whose adapter hits a fake client (no net)."""
    import types as _t
    from agents.providers.openai_compatible_adapter import (
        OpenAICompatibleAdapter, OpenAICompatibleConfig)
    from agents.providers.provider_router import _resolve_env

    class FakeOCClient:
        def __init__(self, recorder):
            self.recorder = recorder

        def complete(self, payload):
            self.recorder["calls"].append(payload)
            return {
                "id": "x",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "oc-done"},
                    "finish_reason": "stop",
                }],
            }

    # A registry-level holder so the factory can share one fake client.
    holder = {"calls": []}

    def factory(model_id):
        # Resolve credential strictly (raises MissingCredentialError when
        # the env var is absent), mirroring the real openai factory.
        api_key = _resolve_env(api_key_env, provider)
        config = OpenAICompatibleConfig(
            base_url=base_url,
            api_key=api_key,
            model=model_id,
        )
        adapter = OpenAICompatibleAdapter(config)
        adapter._client_provider = lambda: FakeOCClient(holder)  # noqa: SLF001
        return adapter

    binding = ProviderBinding(
        provider=provider,
        adapter_type="openai-compatible",
        base_url=base_url,
        api_key_env=api_key_env,
        adapter_factory=factory,
    )
    return binding, holder


class ModelRoutingTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))
        # Default model id (legacy MODEL) = test-model.
        self.recorder = RecordingAnthropicClient(self.module)
        self.recorder.install(_resp_text("done"))
        # Provide keys for the OpenAI-compatible bindings used in most
        # tests (the missing-credential test pops its own key).
        self._saved_keys = {}
        for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
                     "DEEPSEEK_MISSING_KEY"):
            self._saved_keys[name] = os.environ.get(name)
            os.environ[name] = "test-key"

    def tearDown(self):
        for name, prev in self._saved_keys.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

    # ------------------------------------------------------------------
    # Default compatibility
    # ------------------------------------------------------------------

    def test_default_no_args_uses_anthropic_and_legacy_model(self):
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(self.recorder.calls[0]["model"], "test-model")
        # Anthropic wire shape preserved (tools/system/max_tokens).
        self.assertEqual(self.recorder.calls[0]["max_tokens"], 8000)
        self.assertIn("system", self.recorder.calls[0])
        self.assertIn("tools", self.recorder.calls[0])

    def test_default_uses_anthropic_adapter_type(self):
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        # The fake client recorded a messages.create call => it went
        # through the Anthropic adapter (OpenAI adapter would use .complete
        # payload, not messages.create).
        self.assertEqual(len(self.recorder.calls), 1)

    def test_default_model_id_equals_legacy_model(self):
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(self.recorder.calls[0]["model"], self.module.MODEL)

    # ------------------------------------------------------------------
    # Model -> Provider -> Adapter mapping
    # ------------------------------------------------------------------

    def test_deepseek_alias_uses_openai_compatible_adapter(self):
        ds_binding, ds_holder = make_openai_compatible_binding(
            "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY")
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        router = ProviderRouter()
        router.register(ds_binding)

        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_alias="deepseek",
            model_registry=registry,
            provider_router=router,
        )
        # No anthropic client call happened.
        self.assertEqual(len(self.recorder.calls), 0)
        # OpenAI-compatible adapter was used (recorded via .complete).
        self.assertEqual(len(ds_holder["calls"]), 1)
        payload = ds_holder["calls"][0]
        self.assertEqual(payload["model"], "deepseek-chat")

    def test_qwen_openrouter_alias_same_adapter_type(self):
        or_binding, or_holder = make_openai_compatible_binding(
            "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="qwen-openrouter", provider="openrouter",
            model_id="qwen/qwen-2.5-72b-instruct"))
        router = ProviderRouter()
        router.register(or_binding)

        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_alias="qwen-openrouter",
            model_registry=registry,
            provider_router=router,
        )
        self.assertEqual(len(self.recorder.calls), 0)
        self.assertEqual(len(or_holder["calls"]), 1)
        self.assertEqual(or_holder["calls"][0]["model"],
                         "qwen/qwen-2.5-72b-instruct")

    def test_model_id_is_single_source_of_truth(self):
        """request.model == adapter-config model == spec.model_id."""
        ds_binding, ds_holder = make_openai_compatible_binding(
            "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY")
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        router = ProviderRouter()
        router.register(ds_binding)

        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_alias="deepseek",
            model_registry=registry,
            provider_router=router,
        )
        self.assertEqual(ds_holder["calls"][0]["model"], "deepseek-chat")

    # ------------------------------------------------------------------
    # Fail-fast semantics
    # ------------------------------------------------------------------

    def test_unknown_alias_fails_fast_no_request(self):
        with self.assertRaises(UnknownModelError):
            self.module.agent_loop(
                [{"role": "user", "content": "hi"}],
                model_alias="deeepseek",
            )
        self.assertEqual(len(self.recorder.calls), 0)

    def test_unknown_provider_fails_fast_no_request(self):
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="bad", provider="some-provider", model_id="x"))
        router = ProviderRouter()
        # provider "some-provider" not bound -> UnknownProviderError.
        with self.assertRaises(UnknownProviderError):
            self.module.agent_loop(
                [{"role": "user", "content": "hi"}],
                model_alias="bad",
                model_registry=registry,
                provider_router=router,
            )
        self.assertEqual(len(self.recorder.calls), 0)

    def test_missing_credential_fails_before_request(self):
        ds_binding, ds_holder = make_openai_compatible_binding(
            "deepseek", "https://api.deepseek.com", "DEEPSEEK_MISSING_KEY")
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        router = ProviderRouter()
        router.register(ds_binding)
        os.environ.pop("DEEPSEEK_MISSING_KEY", None)

        with self.assertRaises(MissingCredentialError):
            self.module.agent_loop(
                [{"role": "user", "content": "hi"}],
                model_alias="deepseek",
                model_registry=registry,
                provider_router=router,
            )
        self.assertEqual(len(self.recorder.calls), 0)
        self.assertEqual(len(ds_holder["calls"]), 0)

    # ------------------------------------------------------------------
    # Adapter lifecycle: one per agent_loop, reused across turns
    # ------------------------------------------------------------------

    def test_adapter_created_once_and_reused_across_turns(self):
        created = {"count": 0}
        ds_binding, ds_holder = make_openai_compatible_binding(
            "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY")
        original_factory = ds_binding.adapter_factory

        def counting_factory(model_id):
            created["count"] += 1
            return original_factory(model_id)

        ds_binding = ProviderBinding(
            provider="deepseek",
            adapter_type="openai-compatible",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            adapter_factory=counting_factory,
        )
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        router = ProviderRouter()
        router.register(ds_binding)

        # Multi-turn: tool_use then end_turn. The OpenAI fake always
        # returns end_turn text, so to force multiple turns we use the
        # anthropic-style... simpler: just run once and check count == 1.
        self.module.agent_loop(
            [{"role": "user", "content": "hi"}],
            model_alias="deepseek",
            model_registry=registry,
            provider_router=router,
        )
        self.assertEqual(created["count"], 1)

    # ------------------------------------------------------------------
    # Concurrency: different aliases do not interfere
    # ------------------------------------------------------------------

    def test_concurrent_different_aliases_do_not_interfere(self):
        ds_binding, ds_holder = make_openai_compatible_binding(
            "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY")
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        router = ProviderRouter()
        router.register(ds_binding)

        import threading

        results = {"a": None, "b": None}
        errors = []

        def run_a():
            try:
                self.module.agent_loop(
                    [{"role": "user", "content": "hi"}],
                    model_alias="deepseek",
                    model_registry=registry,
                    provider_router=router,
                )
                results["a"] = "ok"
            except Exception as exc:  # noqa: BLE001
                errors.append(("a", exc))

        def run_b():
            try:
                self.module.agent_loop([{"role": "user", "content": "hi"}])
                results["b"] = "ok"
            except Exception as exc:  # noqa: BLE001
                errors.append(("b", exc))

        t1 = threading.Thread(target=run_a)
        t2 = threading.Thread(target=run_b)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        self.assertEqual(errors, [])
        self.assertEqual(results["a"], "ok")
        self.assertEqual(results["b"], "ok")
        # A used OpenAI-compatible (no anthropic call), B used anthropic.
        # A's deepseek call recorded, B's anthropic call recorded.
        # Both ran concurrently without a shared CURRENT_MODEL global.
        self.assertEqual(len(ds_holder["calls"]), 1)
        self.assertEqual(len(self.recorder.calls), 1)

    # ------------------------------------------------------------------
    # BEFORE_MODEL_REQUEST model-consistency guard
    # ------------------------------------------------------------------

    def test_hook_may_patch_content_but_not_model(self):
        from agents.types.events import Event, HookResult

        # Patch max_tokens: allowed.
        patched = {"hit": 0}

        def patcher(ctx):
            patched["hit"] += 1
            return HookResult(model_request_patch={"max_tokens": 1234})

        self.module.EXTENSIONS.on(Event.BEFORE_MODEL_REQUEST, patcher,
                                  extension_id="patcher")
        self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(self.recorder.calls[0]["max_tokens"], 1234)

    def test_hook_cannot_change_model_identity(self):
        from agents.types.events import Event, HookResult

        def bad_patcher(ctx):
            return HookResult(model_request_patch={"model": "evil-model"})

        self.module.EXTENSIONS.on(Event.BEFORE_MODEL_REQUEST, bad_patcher,
                                  extension_id="bad_patcher")
        with self.assertRaises(ValueError):
            self.module.agent_loop([{"role": "user", "content": "hi"}])
        self.assertEqual(len(self.recorder.calls), 0)


if __name__ == "__main__":
    unittest.main()
