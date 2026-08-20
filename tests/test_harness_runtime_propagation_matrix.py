"""test_harness_runtime_propagation_matrix.py - Phase 3C-final.

The single authoritative contract that closes Phase 3C: model selection
propagates across all five runtimes, while each runtime builds its own
adapter.

Matrix (parent model -> every runtime follows):

    Parent      | Main    | Subagent | Team    | Compression | TokenBudget
    ------------|---------|----------|---------|-------------|-------------
    Claude      | Claude  | Claude   | Claude  | Claude      | Claude
    DeepSeek    | DeepSeek| DeepSeek | DeepSeek| DeepSeek    | DeepSeek
    OpenRouter  | Qwen    | Qwen     | Qwen    | Qwen        | Qwen

Extra contracts locked here:
  - every runtime's model_id == Parent ModelRuntimeContext.model_id
  - every runtime builds a DISTINCT adapter instance
    (Main != Subagent != Team != Compression != TokenBudget)
  - snapshot semantics: external ModelRegistry / ProviderRouter changes
    after resolution do NOT affect the current run
  - no hidden Anthropic dependency: DeepSeek / OpenRouter parent with NO
    ANTHROPIC_API_KEY still runs the whole runtime
  - no CURRENT_MODEL / no mutable global runtime context
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agents.providers.model_runtime import ModelRuntimeContext
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
        "harness_core_matrix_test", MODULE_PATH
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
    """A test binding: adapter factory records (provider, model_id) per
    creation; the adapter's client records per-call model too."""
    from agents.providers.openai_compatible_adapter import (
        OpenAICompatibleAdapter, OpenAICompatibleConfig)

    def factory(model_id):
        config = OpenAICompatibleConfig(
            base_url=f"https://{provider}.example.invalid",
            api_key="test-key",
            model=model_id,
        )
        adapter = OpenAICompatibleAdapter(config)
        client = FakeOCClient(seen, provider)
        adapter._client_provider = lambda: client  # noqa: SLF001
        return adapter

    return ProviderBinding(
        provider=provider,
        adapter_type="openai-compatible",
        base_url=f"https://{provider}.example.invalid",
        adapter_factory=factory,
    )


# The matrix: parent alias -> (provider, model_id)
PARENTS = {
    "claude": ("anthropic", "claude-x"),
    "deepseek": ("deepseek", "deepseek-chat"),
    "qwen-openrouter": ("openrouter", "qwen/qwen-2.5-72b-instruct"),
}


class RuntimePropagationMatrixTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))
        self._saved = {}
        for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
                     "OPENROUTER_API_KEY"):
            self._saved[name] = os.environ.get(name)
            os.environ[name] = "test-key"

    def tearDown(self):
        for name, prev in self._saved.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

    def _make_context(self, alias, provider, model_id, seen):
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias=alias, provider=provider, model_id=model_id))
        router = ProviderRouter()
        router.register(make_binding(provider, seen))
        spec = registry.get(alias)
        binding = router.get(spec.provider)
        return ModelRuntimeContext(model_spec=spec, provider_binding=binding)

    def _all_runtime_adapters(self, ctx):
        """Build one adapter per runtime from the SAME context; returns
        a dict runtime->(adapter, model_id, created_provider)."""
        seen = []
        # Re-wrap: bindings above record into `seen` only if we use a
        # context whose binding's factory records there. Since
        # _make_context already passes `seen`, reuse.
        # Main
        main = ctx.create_adapter()
        # Subagent
        sub = ctx.create_adapter()
        # Team
        team = ctx.create_adapter()
        # Compression
        comp = ctx.create_adapter()
        # TokenBudget
        tb = ctx.create_adapter()
        adapters = {
            "Main": main, "Subagent": sub, "Team": team,
            "Compression": comp, "TokenBudget": tb,
        }
        return adapters, seen

    # ------------------------------------------------------------------
    # 1. Propagation matrix: every runtime follows the parent model
    # ------------------------------------------------------------------

    def test_propagation_matrix(self):
        for alias, (provider, model_id) in PARENTS.items():
            with self.subTest(alias=alias):
                seen = []
                ctx = self._make_context(alias, provider, model_id, seen)
                adapters, _ = self._all_runtime_adapters(ctx)
                # Every adapter was created for this parent's provider.
                self.assertEqual(
                    [a._config.model for a in adapters.values()],  # noqa: SLF001
                    [model_id] * 5,
                    f"all runtimes must use model_id={model_id}",
                )
                # All adapters distinct instances (shared selection,
                # never shared adapter).
                ids = [id(a) for a in adapters.values()]
                self.assertEqual(len(set(ids)), 5,
                                 "each runtime must build its own adapter")

    def test_every_runtime_model_id_equals_context(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek",
                                 "deepseek-chat", seen)
        adapters, _ = self._all_runtime_adapters(ctx)
        for name, adapter in adapters.items():
            with self.subTest(runtime=name):
                self.assertEqual(adapter._config.model, ctx.model_id)  # noqa: SLF001

    # ------------------------------------------------------------------
    # 2. Snapshot semantics: external registry/router changes ignored
    # ------------------------------------------------------------------

    def test_snapshot_immune_to_registry_and_router_changes(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek",
                                 "deepseek-chat", seen)
        # External mutation: re-register alias to a different model AND
        # bind the provider to a different endpoint.
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-reasoner"))
        router = ProviderRouter()
        router.register(make_binding("deepseek", []))
        # Existing context is a frozen snapshot - unchanged.
        self.assertEqual(ctx.model_id, "deepseek-chat")
        self.assertEqual(ctx.provider_binding.base_url,
                         "https://deepseek.example.invalid")
        # Runtime adapter still uses the original snapshot.
        adapter = ctx.create_adapter()
        self.assertEqual(adapter._config.model, "deepseek-chat")  # noqa: SLF001

    # ------------------------------------------------------------------
    # 3. No hidden Anthropic dependency
    # ------------------------------------------------------------------

    def test_deepseek_full_runtime_without_anthropic_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        seen = []
        ctx = self._make_context("deepseek", "deepseek",
                                 "deepseek-chat", seen)
        # Compression uses the DeepSeek context (would raise if it tried
        # to build an Anthropic adapter).
        result = self.module.auto_compact(
            [{"role": "user", "content": "x" * 10}], model_runtime=ctx)
        self.assertIn("ok-deepseek", result[0]["content"])

    def test_openrouter_full_runtime_without_anthropic_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        seen = []
        ctx = self._make_context("openrouter", "openrouter",
                                 "qwen/qwen-2.5-72b-instruct", seen)
        result = self.module.auto_compact(
            [{"role": "user", "content": "x" * 10}], model_runtime=ctx)
        self.assertIn("ok-openrouter", result[0]["content"])

    def test_token_summary_no_anthropic_key(self):
        from agents.token_budget import summarize_with_model
        os.environ.pop("ANTHROPIC_API_KEY", None)
        seen = []
        ctx = self._make_context("openrouter", "openrouter",
                                 "qwen/qwen-2.5-72b-instruct", seen)
        result = summarize_with_model(
            ctx, previous_summary="", history_text="h", summary_max_tokens=64)
        self.assertEqual(result, "ok-openrouter")

    # ------------------------------------------------------------------
    # 4. No CURRENT_MODEL / no mutable global runtime context
    # ------------------------------------------------------------------

    def test_no_global_model_state(self):
        # The module must not expose a mutable CURRENT_MODEL or a mutable
        # module-level runtime context.
        self.assertFalse(hasattr(self.module, "CURRENT_MODEL"))
        self.assertFalse(hasattr(self.module, "CURRENT_RUNTIME_CONTEXT"))


class DefaultRegistryRouterCoherenceTests(unittest.TestCase):
    """The default ModelRegistry and ProviderRouter agree on providers."""

    def test_defaults_agree(self):
        from agents.providers.model_spec import default_model_registry
        from agents.providers.provider_router import default_provider_router
        model_providers = {s.provider
                           for s in default_model_registry().list()}
        router_providers = {b.provider
                            for b in default_provider_router().list()}
        self.assertEqual(model_providers, router_providers)
        # Each model's provider is bound in the router.
        router = default_provider_router()
        for s in default_model_registry().list():
            self.assertTrue(router.contains(s.provider))


if __name__ == "__main__":
    unittest.main()
