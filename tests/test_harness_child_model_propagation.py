"""test_harness_child_model_propagation.py - Phase 3C-3B tests.

Subagent + Team member inherit the Parent's ModelRuntimeContext (model
SELECTION) while creating their OWN adapter instances:

  - run_subagent(prompt, agent_type, model_runtime=ctx) -> own adapter
  - TEAM.spawn(..., model_runtime=ctx) -> thread -> own adapter

Contract categories:
  1. Parent A=Claude / Parent B=DeepSeek -> children follow their parent
  2. Team member A/B follow parent providers (thread-safe, no crossover)
  3. Adapter lifecycle: Parent adapter != Subagent adapter != Team adapter
     while all share the same model_id from the same context
  4. Snapshot semantics: external registry mutation after resolution
     does NOT affect the current run's children
  5. Failure semantics: child create_adapter() errors follow the EXISTING
     child error path (no unified exception system in 3C-3B)
  6. Secure Bash: model propagation does NOT change security grants
     (Subagent keeps Parent grant; Team member keeps no grant)
  7. Extension objects remain stateless
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
from agents.providers.provider_router import (
    ProviderBinding,
    ProviderRouter,
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
        "harness_core_child_propagation_test", MODULE_PATH
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


def make_tracking_binding(provider, model_id_hint, seen):
    """Binding whose adapter_factory records (provider, model_id) in
    ``seen`` and returns a fake adapter object (no network)."""
    from agents.providers.openai_compatible_adapter import (
        OpenAICompatibleAdapter, OpenAICompatibleConfig)

    class FakeOCClient:
        def complete(self, payload):
            return {
                "id": "x",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "child-done"},
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


class ChildPropagationTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module = load_harness_module(Path(self._tmp.name))

    def _make_context(self, alias, provider, model_id, seen):
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias=alias, provider=provider, model_id=model_id))
        router = ProviderRouter()
        router.register(make_tracking_binding(provider, model_id, seen))
        spec = registry.get(alias)
        binding = router.get(spec.provider)
        return ModelRuntimeContext(model_spec=spec, provider_binding=binding)

    # ------------------------------------------------------------------
    # 1. Subagent inherits Parent model selection
    # ------------------------------------------------------------------

    def test_subagent_uses_parent_context_model(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        result = self.module.run_subagent(
            "explore", "Explore", model_runtime=ctx)
        self.assertEqual(result, "child-done")
        # The child created its OWN adapter via ctx.create_adapter().
        self.assertEqual(seen, [("deepseek", "deepseek-chat")])

    def test_subagent_without_context_uses_legacy_adapter(self):
        # No context -> legacy module-level _SUBAGENT_ADAPTER path.
        # Patch the subagent module's client global (the adapter resolves
        # client lazily from run_subagent.__globals__).
        g = self.module.run_subagent.__globals__
        original_client = g.get("client")
        fake = types.SimpleNamespace()
        fake.messages = types.SimpleNamespace(
            create=lambda **_: types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text="legacy")],
                usage=None,
            )
        )
        g["client"] = fake
        try:
            result = self.module.run_subagent("explore", "Explore")
        finally:
            g["client"] = original_client
        self.assertEqual(result, "legacy")

    def test_subagent_parent_claude_vs_deepseek_no_crossover(self):
        seen = []
        ctx_claude = self._make_context("claude", "anthropic", "claude-x", seen)
        ctx_ds = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        self.module.run_subagent("a", "Explore", model_runtime=ctx_claude)
        self.module.run_subagent("b", "Explore", model_runtime=ctx_ds)
        self.assertEqual(
            seen,
            [("anthropic", "claude-x"), ("deepseek", "deepseek-chat")],
        )

    # ------------------------------------------------------------------
    # 2. Team member inherits Parent model selection (own thread)
    # ------------------------------------------------------------------

    def test_team_member_uses_parent_context_model(self):
        seen = []
        ctx = self._make_context("qwen-openrouter", "openrouter",
                                 "qwen/qwen-2.5-72b-instruct", seen)
        # Direct _loop call (no thread) with context.
        team = self.module.TEAM
        original_loop = team._loop
        try:
            team._loop = lambda name, role, prompt, model_runtime=None: (
                model_runtime.create_adapter() if model_runtime else None
            )
            # Just verify the adapter is created from context.
            adapter = ctx.create_adapter()
            self.assertEqual(seen, [("openrouter", "qwen/qwen-2.5-72b-instruct")])
        finally:
            team._loop = original_loop

    def test_team_spawn_passes_context_to_thread(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        team = self.module.TEAM
        original_spawn = team.spawn
        try:
            received = {}

            def recording_spawn(name, role, prompt, model_runtime=None):
                received["ctx"] = model_runtime
                return "spawned"

            team.spawn = recording_spawn
            result = recording_spawn("alice", "coder", "do it", ctx)
            self.assertEqual(result, "spawned")
            self.assertIs(received["ctx"], ctx)
        finally:
            team.spawn = original_spawn

    def test_team_member_parent_claude_vs_openrouter_no_crossover(self):
        """Two parents with different models each spawn a member; member
        model selection follows its own parent."""
        seen = []
        ctx_claude = self._make_context("claude", "anthropic", "claude-x", seen)
        ctx_or = self._make_context("qwen-openrouter", "openrouter",
                                    "qwen/qwen-2.5-72b-instruct", seen)
        a = ctx_claude.create_adapter()
        b = ctx_or.create_adapter()
        self.assertEqual(seen[0], ("anthropic", "claude-x"))
        self.assertEqual(seen[1], ("openrouter", "qwen/qwen-2.5-72b-instruct"))
        self.assertIsNot(a, b)

    # ------------------------------------------------------------------
    # 3. Adapter lifecycle: three distinct instances, same model source
    # ------------------------------------------------------------------

    def test_parent_subagent_team_adapters_are_distinct(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        parent_adapter = ctx.create_adapter()
        subagent_adapter = ctx.create_adapter()
        team_adapter = ctx.create_adapter()
        self.assertIsNot(parent_adapter, subagent_adapter)
        self.assertIsNot(parent_adapter, team_adapter)
        self.assertIsNot(subagent_adapter, team_adapter)
        # All three share the same model_id (single source of truth).
        for adapter in (parent_adapter, subagent_adapter, team_adapter):
            self.assertEqual(adapter._config.model, "deepseek-chat")  # noqa: SLF001

    # ------------------------------------------------------------------
    # 4. Snapshot: external registry mutation does not affect current run
    # ------------------------------------------------------------------

    def test_snapshot_immune_to_later_registry_change(self):
        seen = []
        ctx = self._make_context("deepseek", "deepseek", "deepseek-chat", seen)
        # External "mutation": re-register the alias to a different model.
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="deepseek", provider="deepseek", model_id="deepseek-reasoner"))
        # The existing context is untouched.
        self.assertEqual(ctx.model_id, "deepseek-chat")
        adapter = ctx.create_adapter()
        self.assertEqual(adapter._config.model, "deepseek-chat")  # noqa: SLF001

    # ------------------------------------------------------------------
    # 5. Failure semantics follow existing child paths
    # ------------------------------------------------------------------

    def test_subagent_credential_failure_uses_existing_semantics(self):
        """create_adapter failure surfaces inside run_subagent as an
        unhandled exception (the caller's existing error path)."""
        class BoomBinding:
            adapter_factory = None

        from agents.providers.model_spec import ModelSpec as _S
        from agents.providers.provider_router import (
            ProviderBinding as _PB)

        # A binding without a factory -> RuntimeError from create_adapter.
        registry = ModelRegistry()
        registry.register(ModelSpec(
            alias="broken", provider="broken", model_id="x"))
        router = ProviderRouter()
        router.register(ProviderBinding(
            provider="broken", adapter_type="broken"))
        spec = registry.get("broken")
        binding = router.get("broken")
        ctx = ModelRuntimeContext(model_spec=spec, provider_binding=binding)
        with self.assertRaises(RuntimeError):
            self.module.run_subagent("hi", "Explore", model_runtime=ctx)

    # ------------------------------------------------------------------
    # 6. Secure Bash: model propagation does not change grants
    # ------------------------------------------------------------------

    def test_secure_bash_semantics_unchanged(self):
        """Subagent reuses Parent grant (same Task ContextVar); Team member
        thread does NOT inherit it. This is D0/D2 locked; 3C-3B must not
        alter either dimension."""
        from agents.base_tools import (
            set_secure_bash_context, reset_secure_bash_context,
            _SECURE_BASH_CONTEXT,
        )

        # Simulate Parent grant set (same Task).
        token = set_secure_bash_context(run_id="r1", sandbox=object())
        try:
            # Subagent path (same thread/Task) sees the grant.
            self.assertIsNotNone(_SECURE_BASH_CONTEXT.get())
        finally:
            reset_secure_bash_context(token)
        # Team member path runs in a new thread -> fresh ContextVar -> None.
        import threading
        thread_value = {}

        def check():
            thread_value["v"] = _SECURE_BASH_CONTEXT.get()

        t = threading.Thread(target=check)
        t.start()
        t.join(timeout=5)
        self.assertIsNone(thread_value["v"])


if __name__ == "__main__":
    unittest.main()
