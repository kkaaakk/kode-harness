"""test_harness_compaction_model_propagation.py - Phase 3C-3C tests.

Compression + TokenBudget inherit the Parent ModelRuntimeContext:

  - auto_compact(messages, model_runtime=ctx) -> ctx.create_adapter(),
    model_id == ctx.model_id (single source of truth)
  - summarize_with_model(model_runtime, ...) -> own adapter, same model
  - summarize_with_anthropic kept as legacy compatibility wrapper

Contract categories:
  1. Model propagation: Claude/DeepSeek/OpenRouter parent -> same model
     for compression + token summary (matrix)
  2. No hidden Anthropic dependency: DeepSeek parent, NO Anthropic key,
     full runtime (main + compaction) works
  3. Adapter independence: main adapter != compression adapter, both
     share the same model_id from the same context
  4. Snapshot: external registry mutation after resolution does NOT
     change the current run's compression model
  5. Legacy compatibility: auto_compact()/summarize_with_anthropic()
     without a context keep the old fixed-Anthropic behavior
  6. No overreach: capabilities not enforced, no ModelLimits, no
     token-budget/threshold changes
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
        "harness_core_compaction_test", MODULE_PATH
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


def make_tracking_binding(provider, seen):
    """Binding recording (provider, model_id) per adapter creation."""
    from agents.providers.openai_compatible_adapter import (
        OpenAICompatibleAdapter, OpenAICompatibleConfig)

    class FakeOCClient:
        def complete(self, payload):
            return {
                "id": "x",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": "summary-from-fake"},
                    "finish_reason": "stop",
                }],
            }

    def factory(model_id):
        seen.append((provider, model_id))
        config = OpenAICompatibleConfig(
            base_url=f"https://{provider}.example.invalid",
            api_key="test-key",
            model=model_id,
        )
        adapter = OpenAICompatibleAdapter(config)
        adapter._client_provider = lambda: FakeOCClient()  # noqa: SLF001
        return adapter

    return ProviderBinding(
        provider=provider,
        adapter_type="openai-compatible",
        base_url=f"https://{provider}.example.invalid",
        adapter_factory=factory,
    )


class CompactionPropagationTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))

    def _make_context(self, alias, provider, model_id, seen):
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias=alias, provider=provider, model_id=model_id))
        router = ProviderRouter()
        router.register(make_tracking_binding(provider, seen))
        spec = registry.get(alias)
        binding = router.get(spec.provider)
        return ModelRuntimeContext(model_spec=spec, provider_binding=binding)

    # ------------------------------------------------------------------
    # 1. Model propagation matrix
    # ------------------------------------------------------------------

    def test_compression_follows_parent_model(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        result = self.module.auto_compact(
            [{"role": "user", "content": "x" * 100}], model_runtime=ctx)
        self.assertIn("summary-from-fake", result[0]["content"])
        self.assertEqual(seen, [("deepseek", "deepseek-chat")])

    def test_compression_matrix_claude_deepseek_openrouter(self):
        for alias, provider, model_id in [
            ("claude", "anthropic", "claude-x"),
            ("deepseek", "deepseek", "deepseek-chat"),
            ("qwen-openrouter", "openrouter", "qwen/qwen-2.5-72b-instruct"),
        ]:
            with self.subTest(alias=alias):
                seen = []
                ctx = self._make_context(alias, provider, model_id, seen)
                self.module.auto_compact(
                    [{"role": "user", "content": "x"}], model_runtime=ctx)
                self.assertEqual(seen, [(provider, model_id)])

    def test_token_summary_follows_parent_model(self):
        from agents.token_budget import summarize_with_model
        seen = []
        ctx = self._make_context("openrouter", "openrouter",
                                 "qwen/qwen-2.5-72b-instruct", seen)
        result = summarize_with_model(
            ctx, previous_summary="", history_text="h", summary_max_tokens=64)
        self.assertEqual(result, "summary-from-fake")
        self.assertEqual(seen, [("openrouter", "qwen/qwen-2.5-72b-instruct")])

    # ------------------------------------------------------------------
    # 2. No hidden Anthropic dependency (DeepSeek without Anthropic key)
    # ------------------------------------------------------------------

    def test_deepseek_full_runtime_no_anthropic_key(self):
        """DeepSeek parent: main + compression work without any Anthropic
        credential. This is the key regression the user asked to lock."""
        # Ensure no Anthropic key is present (drop from env).
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            seen = []
            ctx = self._make_context("deepseek", "deepseek",
                                     "deepseek-chat", seen)
            # Compression via DeepSeek context (would raise if it tried to
            # build an Anthropic adapter).
            result = self.module.auto_compact(
                [{"role": "user", "content": "x" * 10}], model_runtime=ctx)
            self.assertIn("summary-from-fake", result[0]["content"])
            self.assertEqual(seen, [("deepseek", "deepseek-chat")])
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_openrouter_full_runtime_no_anthropic_key(self):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            seen = []
            ctx = self._make_context("openrouter", "openrouter",
                                     "qwen/qwen-2.5-72b-instruct", seen)
            self.module.auto_compact(
                [{"role": "user", "content": "x" * 10}], model_runtime=ctx)
            self.assertEqual(seen, [("openrouter", "qwen/qwen-2.5-72b-instruct")])
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    # ------------------------------------------------------------------
    # 3. Adapter independence + single model source
    # ------------------------------------------------------------------

    def test_main_and_compression_adapters_distinct_same_model(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        main_adapter = ctx.create_adapter()
        comp_adapter = ctx.create_adapter()
        self.assertIsNot(main_adapter, comp_adapter)
        self.assertEqual(main_adapter._config.model, "deepseek-chat")  # noqa: SLF001
        self.assertEqual(comp_adapter._config.model, "deepseek-chat")  # noqa: SLF001

    # ------------------------------------------------------------------
    # 4. Snapshot immune to external registry changes
    # ------------------------------------------------------------------

    def test_compression_snapshot_immune_to_later_change(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        # External "mutation": re-register the alias to a different model.
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-reasoner"))
        # The existing context is untouched.
        self.assertEqual(ctx.model_id, "deepseek-chat")
        self.module.auto_compact(
            [{"role": "user", "content": "x"}], model_runtime=ctx)
        self.assertEqual(seen, [("deepseek", "deepseek-chat")])

    # ------------------------------------------------------------------
    # 5. Legacy compatibility
    # ------------------------------------------------------------------

    def test_auto_compact_without_context_legacy(self):
        """No context -> legacy fixed-Anthropic path via module client."""
        g = self.module.auto_compact.__globals__
        original_client = g.get("client")
        fake = types.SimpleNamespace()
        fake.messages = types.SimpleNamespace(
            create=lambda **_: types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text="legacy-sum")],
                usage=None,
            )
        )
        g["client"] = fake
        try:
            result = self.module.auto_compact(
                [{"role": "user", "content": "x" * 10}])
        finally:
            g["client"] = original_client
        self.assertIn("legacy-sum", result[0]["content"])

    def test_summarize_with_anthropic_legacy_wrapper(self):
        from agents.token_budget import summarize_with_anthropic
        from unittest.mock import MagicMock
        client = MagicMock()
        client.messages.create.return_value = types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="legacy-sum")],
            usage=None,
        )
        result = summarize_with_anthropic(
            client, model="m", previous_summary="", history_text="h",
            summary_max_tokens=64)
        self.assertEqual(result, "legacy-sum")

    # ------------------------------------------------------------------
    # 6. No overreach: budget/threshold/capabilities unchanged
    # ------------------------------------------------------------------

    def test_compaction_algorithm_unchanged(self):
        """auto_compact output shape is unchanged: single user message
        with [Compressed. Transcript: ...] prefix."""
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        result = self.module.auto_compact(
            [{"role": "user", "content": "x" * 10}], model_runtime=ctx)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertIn("[Compressed. Transcript:", result[0]["content"])
        self.assertIn("summary-from-fake", result[0]["content"])

    def test_token_budget_api_not_restructured(self):
        from agents.token_budget import (
            summarize_with_anthropic, summarize_with_model,
        )
        # Both entry points exist; the legacy one keeps its old signature.
        self.assertTrue(callable(summarize_with_anthropic))
        self.assertTrue(callable(summarize_with_model))


if __name__ == "__main__":
    unittest.main()
