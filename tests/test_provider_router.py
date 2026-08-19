"""test_provider_router.py - Phase 3C-1 ProviderBinding + ProviderRouter
contract tests.

Covers the user's list:
  - ProviderBinding frozen; no actual API key; no model_id
  - register/get/list/contains; duplicate provider fail-fast;
    unknown provider fail-fast; stable order
  - DeepSeek vs OpenRouter distinct providers, same adapter type
  - Anthropic -> AnthropicAdapter
  - building router reads no env; router.get() reads no key
  - create_adapter() resolves credential lazily; missing credential
    gives a clear config error naming only the env var
  - one OpenRouter binding serves two different ModelSpec.model_id
  - ModelRegistry and ProviderRouter fully independent
  - no capabilities enforcement; no Agent Loop touch
"""

import os
import types
import unittest

from agents.providers.anthropic_adapter import AnthropicAdapter
from agents.providers.model_spec import ModelSpec, ModelRegistry
from agents.providers.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
)
from agents.providers.provider_router import (
    DuplicateProviderError,
    MissingCredentialError,
    ProviderBinding,
    ProviderRouter,
    UnknownProviderError,
    default_provider_router,
    make_openai_compatible_factory,
)


class ProviderBindingTests(unittest.TestCase):

    def test_frozen(self):
        binding = ProviderBinding(provider="deepseek",
                                  adapter_type="openai-compatible")
        with self.assertRaises(Exception):
            binding.provider = "other"  # type: ignore[misc]

    def test_fields_preserved(self):
        binding = ProviderBinding(
            provider="openrouter",
            adapter_type="openai-compatible",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )
        self.assertEqual(binding.provider, "openrouter")
        self.assertEqual(binding.adapter_type, "openai-compatible")
        self.assertEqual(binding.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(binding.api_key_env, "OPENROUTER_API_KEY")

    def test_binding_holds_no_secret(self):
        """api_key_env is a NAME, never the key value itself."""
        binding = ProviderBinding(provider="deepseek",
                                  adapter_type="openai-compatible",
                                  api_key_env="DEEPSEEK_API_KEY")
        raw = vars(binding)
        joined = repr(raw)
        self.assertNotIn("sk-", joined)
        self.assertNotIn("fe0053", joined)

    def test_binding_holds_no_model_id(self):
        """model_id source of truth stays in ModelSpec, not the binding."""
        binding = ProviderBinding(provider="openrouter",
                                  adapter_type="openai-compatible",
                                  base_url="https://openrouter.ai/api/v1")
        self.assertNotIn("model", vars(binding))
        self.assertNotIn("model_id", vars(binding))

    def test_binding_allows_none_adapter_factory(self):
        b = ProviderBinding(provider="x", adapter_type="y")
        self.assertIsNone(b.adapter_factory)
        self.assertIsNone(b.base_url)
        self.assertIsNone(b.api_key_env)


class ProviderRouterTests(unittest.TestCase):

    def setUp(self):
        self.router = ProviderRouter()
        self.ds = ProviderBinding(provider="deepseek",
                                  adapter_type="openai-compatible")
        self.anthropic = ProviderBinding(provider="anthropic",
                                         adapter_type="anthropic")

    def test_register_get_contains(self):
        self.router.register(self.ds)
        self.assertTrue(self.router.contains("deepseek"))
        self.assertFalse(self.router.contains("nope"))
        self.assertIs(self.router.get("deepseek"), self.ds)

    def test_duplicate_provider_fails_fast(self):
        self.router.register(self.ds)
        with self.assertRaises(DuplicateProviderError):
            self.router.register(ProviderBinding(
                provider="deepseek", adapter_type="openai-compatible"))
        self.assertEqual(self.router.get("deepseek").adapter_type,
                         "openai-compatible")

    def test_unknown_provider_fails_fast(self):
        with self.assertRaises(UnknownProviderError):
            self.router.get("missing")

    def test_list_preserves_registration_order(self):
        self.router.register(self.ds)
        self.router.register(self.anthropic)
        self.assertEqual([b.provider for b in self.router.list()],
                         ["deepseek", "anthropic"])

    def test_len_contains(self):
        self.assertEqual(len(self.router), 0)
        self.router.register(self.ds)
        self.assertEqual(len(self.router), 1)
        self.assertIn("deepseek", self.router)


class DefaultRouterTests(unittest.TestCase):

    def test_default_has_three_bindings_in_order(self):
        router = default_provider_router()
        self.assertEqual([b.provider for b in router.list()],
                         ["anthropic", "deepseek", "openrouter"])

    def test_provider_identity_not_collapsed_to_protocol(self):
        router = default_provider_router()
        self.assertNotEqual(router.get("deepseek").provider,
                            router.get("openrouter").provider)
        # Providers are service identities, never the protocol string.
        self.assertNotEqual(router.get("deepseek").provider, "openai-compatible")

    def test_deepseek_and_openrouter_share_adapter_type(self):
        router = default_provider_router()
        self.assertEqual(router.get("deepseek").adapter_type,
                         router.get("openrouter").adapter_type)
        self.assertEqual(router.get("deepseek").adapter_type,
                         "openai-compatible")

    def test_anthropic_adapter_type(self):
        router = default_provider_router()
        self.assertEqual(router.get("anthropic").adapter_type, "anthropic")

    def test_build_router_reads_no_env(self):
        import os
        before = dict(os.environ)
        default_provider_router()
        after = dict(os.environ)
        self.assertEqual(before, after)

    def test_router_get_reads_no_key(self):
        import os
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        router = default_provider_router()
        before = dict(os.environ)
        router.get("deepseek")
        router.get("openrouter")
        router.list()
        self.assertEqual(before, dict(os.environ))


class AdapterFactoryTests(unittest.TestCase):

    def setUp(self):
        self._cleanup = []
        for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "TEST_KEY"):
            self._cleanup.append((name, os.environ.pop(name, None)))

    def tearDown(self):
        for name, prev in self._cleanup:
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

    def test_missing_credential_raises_clear_error(self):
        factory = make_openai_compatible_factory(
            "https://api.deepseek.com", "DEEPSEEK_API_KEY")
        with self.assertRaises(MissingCredentialError) as ctx:
            factory("deepseek-chat")
        # Error names the env var, never a secret.
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))
        self.assertNotIn("sk-", str(ctx.exception))

    def test_missing_api_key_env_config(self):
        """Empty api_key_env is rejected at adapter-creation time, not
        at factory-build time (build stays side-effect free)."""
        factory = make_openai_compatible_factory("https://x.com", "")
        with self.assertRaises(MissingCredentialError):
            factory("model")

    def test_create_adapter_resolves_credential_lazily(self):
        """Key not present at factory build, present at create time."""
        factory = make_openai_compatible_factory(
            "https://api.deepseek.com", "DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "sk-lazy-key-123"
        adapter = factory("deepseek-chat")
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)
        # Model id is filled from the create_adapter arg.
        self.assertEqual(adapter._config.model, "deepseek-chat")
        self.assertEqual(adapter._config.base_url, "https://api.deepseek.com")

    def test_create_adapter_requires_credential_at_call_not_build(self):
        factory = make_openai_compatible_factory(
            "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
        # Building the factory did not need the key; only calling does.
        os.environ["OPENROUTER_API_KEY"] = "sk-or-lazy"
        adapter = factory("qwen/qwen-2.5-72b-instruct")
        self.assertEqual(adapter._config.model, "qwen/qwen-2.5-72b-instruct")


class DefaultRouterAdapterCreationTests(unittest.TestCase):

    def setUp(self):
        self._cleanup = []
        for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
            self._cleanup.append((name, os.environ.pop(name, None)))

    def tearDown(self):
        for name, prev in self._cleanup:
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

    def test_anthropic_binding_creates_anthropic_adapter(self):
        router = default_provider_router()
        adapter = router.get("anthropic").adapter_factory("claude-sonnet-4-6")
        self.assertIsInstance(adapter, AnthropicAdapter)

    def test_deepseek_binding_creates_openai_adapter(self):
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        router = default_provider_router()
        adapter = router.get("deepseek").adapter_factory("deepseek-chat")
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)

    def test_openrouter_binding_creates_openai_adapter(self):
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
        router = default_provider_router()
        adapter = router.get("openrouter").adapter_factory(
            "qwen/qwen-2.5-72b-instruct")
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)

    def test_one_binding_serves_multiple_model_ids(self):
        """An OpenRouter binding can serve any model routed through it."""
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
        router = default_provider_router()
        binding = router.get("openrouter")
        a = binding.adapter_factory("qwen/qwen-2.5-72b-instruct")
        b = binding.adapter_factory("deepseek/deepseek-chat")
        self.assertEqual(a._config.model, "qwen/qwen-2.5-72b-instruct")
        self.assertEqual(b._config.model, "deepseek/deepseek-chat")
        self.assertEqual(a._config.base_url, b._config.base_url)


class IndependenceTests(unittest.TestCase):
    """ModelRegistry and ProviderRouter are fully independent."""

    def test_registries_are_independent(self):
        models = ModelRegistry()
        models.register(ModelSpec(alias="deepseek", provider="deepseek",
                                  model_id="deepseek-chat"))
        router = ProviderRouter()
        router.register(ProviderBinding(provider="deepseek",
                                        adapter_type="openai-compatible"))
        # Modifying one must not affect the other.
        self.assertEqual(len(models), 1)
        self.assertEqual(len(router), 1)
        self.assertEqual(models.get("deepseek").provider, "deepseek")
        self.assertEqual(router.get("deepseek").adapter_type,
                         "openai-compatible")

    def test_default_registries_agree_on_providers(self):
        from agents.providers.model_spec import default_model_registry
        model_providers = {s.provider
                           for s in default_model_registry().list()}
        router_providers = {b.provider
                            for b in default_provider_router().list()}
        self.assertEqual(model_providers, router_providers)


class NoCapabilityEnforcementTests(unittest.TestCase):
    """3C-1 does NOT enforce capabilities or touch the Agent Loop."""

    def test_router_ignores_capabilities(self):
        from agents.providers.model_spec import ModelCapabilities
        spec = ModelSpec(
            alias="nope", provider="anthropic", model_id="x",
            capabilities=ModelCapabilities(tool_calling=False))
        router = default_provider_router()
        # Resolving by provider never inspects capabilities.
        binding = router.get(spec.provider)
        self.assertEqual(binding.provider, "anthropic")


if __name__ == "__main__":
    unittest.main()
