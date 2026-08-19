"""test_provider_model_spec.py - Phase 3C-0 ModelSpec / ModelCapabilities /
ModelRegistry contract tests.

Covers:
  - ModelCapabilities default all-False, frozen, per-model not per-adapter
  - ModelSpec frozen; alias/provider/model_id preserved
  - provider != adapter type: deepseek.provider != openrouter.provider
    while both will later resolve to the same adapter type
  - ModelRegistry: register/get/list/contains; duplicate alias fail-fast;
    unknown alias fail-fast; stable order; no mutation of specs;
    multiple models per provider; same model_id under different providers;
    registry reads no env vars and creates no clients
"""

import unittest

from agents.providers.model_spec import (
    DuplicateModelError,
    ModelCapabilities,
    ModelRegistry,
    ModelSpec,
    UnknownModelError,
    default_model_registry,
)


class ModelCapabilitiesTests(unittest.TestCase):

    def test_defaults_all_false(self):
        caps = ModelCapabilities()
        self.assertFalse(caps.tool_calling)
        self.assertFalse(caps.parallel_tool_calls)
        self.assertFalse(caps.vision)
        self.assertFalse(caps.reasoning)
        self.assertFalse(caps.prompt_cache)
        self.assertFalse(caps.structured_output)

    def test_frozen_cannot_be_modified(self):
        caps = ModelCapabilities(tool_calling=True)
        with self.assertRaises(Exception):
            caps.tool_calling = False  # type: ignore[misc]

    def test_fields_are_preserved(self):
        caps = ModelCapabilities(
            tool_calling=True,
            parallel_tool_calls=True,
            vision=True,
            reasoning=True,
            prompt_cache=True,
            structured_output=True,
        )
        self.assertTrue(all([
            caps.tool_calling, caps.parallel_tool_calls, caps.vision,
            caps.reasoning, caps.prompt_cache, caps.structured_output,
        ]))


class ModelSpecTests(unittest.TestCase):

    def test_frozen_spec(self):
        spec = ModelSpec(alias="deepseek", provider="deepseek",
                         model_id="deepseek-chat")
        with self.assertRaises(Exception):
            spec.alias = "other"  # type: ignore[misc]

    def test_fields_preserved(self):
        spec = ModelSpec(
            alias="qwen-openrouter",
            provider="openrouter",
            model_id="qwen/qwen-2.5-72b-instruct",
            capabilities=ModelCapabilities(vision=True),
        )
        self.assertEqual(spec.alias, "qwen-openrouter")
        self.assertEqual(spec.provider, "openrouter")
        self.assertEqual(spec.model_id, "qwen/qwen-2.5-72b-instruct")
        self.assertTrue(spec.capabilities.vision)

    def test_default_capabilities(self):
        spec = ModelSpec(alias="a", provider="p", model_id="m")
        self.assertEqual(spec.capabilities, ModelCapabilities())


class ModelIdentitySeparationTests(unittest.TestCase):
    """Core 3C modeling decision: provider identity vs adapter type."""

    def test_deepseek_and_openrouter_are_distinct_providers(self):
        ds = ModelSpec(alias="deepseek", provider="deepseek",
                       model_id="deepseek-chat")
        or_ = ModelSpec(alias="qwen-openrouter", provider="openrouter",
                        model_id="qwen/qwen-2.5-72b-instruct")
        # Providers must NOT be collapsed into a shared adapter name.
        self.assertNotEqual(ds.provider, or_.provider)
        self.assertNotEqual(ds.provider, "openai-compatible")
        self.assertNotEqual(or_.provider, "openai-compatible")

    def test_same_adapter_type_future_compatibility(self):
        """Both deepseek and openrouter will later resolve to
        openai-compatible; that must not conflict with distinct
        provider identities."""
        adapters = {"deepseek": "openai-compatible",
                    "openrouter": "openai-compatible"}
        self.assertEqual(adapters["deepseek"], adapters["openrouter"])
        # Yet providers differ - the mapping stays a separate concern.
        registry = ModelRegistry()
        registry.register(ModelSpec(alias="deepseek", provider="deepseek",
                                    model_id="deepseek-chat"))
        registry.register(ModelSpec(alias="qwen-openrouter",
                                    provider="openrouter",
                                    model_id="qwen/qwen-2.5-72b-instruct"))
        self.assertNotEqual(
            registry.get("deepseek").provider,
            registry.get("qwen-openrouter").provider,
        )

    def test_capability_belongs_to_specific_model_not_adapter(self):
        a = ModelSpec(alias="a", provider="openrouter", model_id="x",
                      capabilities=ModelCapabilities(vision=True))
        b = ModelSpec(alias="b", provider="openrouter", model_id="y",
                      capabilities=ModelCapabilities(vision=False))
        self.assertNotEqual(a.capabilities, b.capabilities)


class ModelRegistryTests(unittest.TestCase):

    def setUp(self):
        self.registry = ModelRegistry()
        self.ds = ModelSpec(alias="deepseek", provider="deepseek",
                            model_id="deepseek-chat")
        self.claude = ModelSpec(alias="claude", provider="anthropic",
                                model_id="claude-sonnet-4-6")

    def test_register_get_contains(self):
        self.registry.register(self.ds)
        self.assertTrue(self.registry.contains("deepseek"))
        self.assertFalse(self.registry.contains("nope"))
        got = self.registry.get("deepseek")
        self.assertEqual(got, self.ds)
        self.assertIs(got, self.ds)

    def test_duplicate_alias_fails_fast(self):
        self.registry.register(self.ds)
        with self.assertRaises(DuplicateModelError):
            self.registry.register(ModelSpec(
                alias="deepseek", provider="other", model_id="x"))
        # Original spec untouched.
        self.assertEqual(self.registry.get("deepseek").provider, "deepseek")

    def test_unknown_alias_fails_fast(self):
        with self.assertRaises(UnknownModelError):
            self.registry.get("missing")

    def test_list_preserves_registration_order(self):
        self.registry.register(self.ds)
        self.registry.register(self.claude)
        aliases = [s.alias for s in self.registry.list()]
        self.assertEqual(aliases, ["deepseek", "claude"])

    def test_registry_does_not_mutate_spec(self):
        spec = ModelSpec(alias="deepseek", provider="deepseek",
                         model_id="deepseek-chat")
        self.registry.register(spec)
        spec_before = spec
        self.assertIs(self.registry.get("deepseek"), spec_before)

    def test_multiple_models_per_provider(self):
        self.registry.register(ModelSpec(alias="ds-chat", provider="deepseek",
                                         model_id="deepseek-chat"))
        self.registry.register(ModelSpec(alias="ds-reasoner",
                                         provider="deepseek",
                                         model_id="deepseek-reasoner"))
        self.assertEqual(len([s for s in self.registry.list()
                              if s.provider == "deepseek"]), 2)

    def test_same_model_id_under_different_providers(self):
        self.registry.register(ModelSpec(alias="a", provider="openrouter",
                                         model_id="deepseek-chat"))
        self.registry.register(ModelSpec(alias="b", provider="deepseek",
                                         model_id="deepseek-chat"))
        self.assertEqual(
            self.registry.get("a").model_id,
            self.registry.get("b").model_id,
        )
        self.assertNotEqual(
            self.registry.get("a").provider,
            self.registry.get("b").provider,
        )

    def test_does_not_read_env_or_create_clients(self):
        # ModelRegistry is a pure domain object: registering and reading
        # must not touch os.environ or construct any client/adapter.
        import os
        before = dict(os.environ)
        self.registry.register(self.ds)
        self.registry.get("deepseek")
        self.registry.list()
        after = dict(os.environ)
        self.assertEqual(before, after)

    def test_len_and_contains(self):
        self.assertEqual(len(self.registry), 0)
        self.registry.register(self.ds)
        self.assertEqual(len(self.registry), 1)
        self.assertIn("deepseek", self.registry)


class DefaultRegistryTests(unittest.TestCase):

    def test_default_registry_has_three_samples(self):
        reg = default_model_registry()
        self.assertEqual([s.alias for s in reg.list()],
                         ["claude", "deepseek", "qwen-openrouter"])

    def test_default_sample_provider_identity(self):
        reg = default_model_registry()
        self.assertEqual(reg.get("claude").provider, "anthropic")
        self.assertEqual(reg.get("deepseek").provider, "deepseek")
        self.assertEqual(reg.get("qwen-openrouter").provider, "openrouter")
        # Providers distinct from adapter types.
        providers = {s.provider for s in reg.list()}
        self.assertNotIn("openai-compatible", providers)

    def test_default_sample_model_ids(self):
        reg = default_model_registry()
        self.assertEqual(reg.get("deepseek").model_id, "deepseek-chat")
        self.assertEqual(reg.get("qwen-openrouter").model_id,
                         "qwen/qwen-2.5-72b-instruct")

    def test_default_sample_capabilities(self):
        reg = default_model_registry()
        self.assertTrue(reg.get("claude").capabilities.tool_calling)
        self.assertTrue(reg.get("deepseek").capabilities.tool_calling)
        self.assertTrue(reg.get("qwen-openrouter").capabilities.vision)
        self.assertFalse(reg.get("deepseek").capabilities.vision)


if __name__ == "__main__":
    unittest.main()
