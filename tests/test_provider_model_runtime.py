"""test_provider_model_runtime.py - Phase 3C-3A ModelRuntimeContext tests.

Locks the "model selection is an immutable per-run snapshot" contract:

  - context is frozen; holds ModelSpec + ProviderBinding only
  - no API key, no adapter instance, no mutable CURRENT_MODEL
  - model_id is the single source of truth (== spec.model_id)
  - create_adapter() returns a FRESH adapter per call (never shared)
  - resolve_model_runtime(): alias -> spec -> binding -> context
  - fail-fast: unknown alias / unknown provider, no fallback
  - multiple contexts for different aliases coexist without interference
"""

import os
import unittest

from agents.providers.model_runtime import (
    ModelRuntimeContext,
    resolve_model_runtime,
)
from agents.providers.model_spec import (
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
)
from agents.providers.provider_router import (
    ProviderBinding,
    ProviderRouter,
    UnknownProviderError,
)


def _make_deepseek_router():
    from agents.providers.provider_router import make_openai_compatible_factory
    router = ProviderRouter()
    router.register(ProviderBinding(
        provider="deepseek",
        adapter_type="openai-compatible",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        adapter_factory=make_openai_compatible_factory(
            "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    ))
    return router


class ModelRuntimeContextTests(unittest.TestCase):

    def setUp(self):
        self._prev = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "sk-test-key"
        self.registry = ModelRegistry()
        self.registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-chat"))
        self.router = _make_deepseek_router()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = self._prev

    def test_context_is_frozen(self):
        ctx = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        with self.assertRaises(Exception):
            ctx.model_spec = None  # type: ignore[misc]

    def test_context_holds_spec_and_binding(self):
        ctx = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        self.assertIsInstance(ctx.model_spec, ModelSpec)
        self.assertIsInstance(ctx.provider_binding, ProviderBinding)
        self.assertEqual(ctx.model_spec.alias, "deepseek")

    def test_context_holds_no_secret(self):
        ctx = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        joined = repr(vars(ctx))
        self.assertNotIn("sk-", joined)

    def test_context_holds_no_adapter_instance(self):
        ctx = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        # The context stores only spec + binding (no adapter instance).
        self.assertEqual(set(vars(ctx)), {"model_spec", "provider_binding"})
        # The binding only carries a factory, not an adapter object.
        self.assertIsNotNone(ctx.provider_binding.adapter_factory)

    def test_model_id_is_single_source_of_truth(self):
        ctx = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        self.assertEqual(ctx.model_id, "deepseek-chat")
        self.assertEqual(ctx.model_id, ctx.model_spec.model_id)

    def test_create_adapter_returns_fresh_instance_per_call(self):
        ctx = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        a = ctx.create_adapter()
        b = ctx.create_adapter()
        self.assertIsNot(a, b)
        # Both bound to the same endpoint/model.
        self.assertEqual(a._config.model, "deepseek-chat")  # noqa: SLF001
        self.assertEqual(b._config.model, "deepseek-chat")  # noqa: SLF001

    def test_multiple_contexts_coexist(self):
        claude_registry = ModelRegistry()
        claude_registry.register(ModelSpec(
            alias="claude", provider="anthropic", model_id="claude-x"))
        claude_router = ProviderRouter()
        claude_router.register(ProviderBinding(
            provider="anthropic", adapter_type="anthropic",
            adapter_factory=lambda model_id: object()))
        ctx_a = resolve_model_runtime(
            "deepseek", self.registry, self.router)
        ctx_b = resolve_model_runtime(
            "claude", claude_registry, claude_router)
        self.assertEqual(ctx_a.model_id, "deepseek-chat")
        self.assertEqual(ctx_b.model_id, "claude-x")
        self.assertNotEqual(ctx_a.model_spec.provider,
                            ctx_b.model_spec.provider)


class ResolveModelRuntimeTests(unittest.TestCase):

    def test_defaults_use_default_registry_and_router(self):
        ctx = resolve_model_runtime("claude")
        self.assertEqual(ctx.model_spec.provider, "anthropic")
        self.assertEqual(ctx.model_id, ctx.model_spec.model_id)

    def test_unknown_alias_fails_fast(self):
        with self.assertRaises(UnknownModelError):
            resolve_model_runtime("deeepseek")

    def test_unknown_provider_fails_fast(self):
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="bad", provider="some-provider", model_id="x"))
        router = ProviderRouter()  # provider not bound
        with self.assertRaises(UnknownProviderError):
            resolve_model_runtime("bad", registry, router)

    def test_no_fallback_on_unknown(self):
        # No silent default: unknown alias raises rather than returning
        # the claude context.
        with self.assertRaises(UnknownModelError):
            resolve_model_runtime("claude-typo")


if __name__ == "__main__":
    unittest.main()
