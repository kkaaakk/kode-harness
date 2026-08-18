"""test_tool_registry.py - Tests for ToolRegistry infrastructure.

Stage 2B: profiles are explicit whitelists of tool names, separate from tool
registration. A tool no longer declares its own profiles.
"""

from __future__ import annotations

import unittest

from agents.tool_registry import (
    ToolEntry, ToolRegistry, PROFILES, STANDARD_PROFILES,
    UnknownToolProfileError,
)


def _schema(**fields):
    return {"type": "object", "properties": fields, "required": list(fields)}


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def test_register_and_get(self):
        self.reg.register("bash", "Run shell", _schema(command={"type": "string"}),
                          lambda **kw: "ok")
        entry = self.reg.get("bash")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "bash")
        self.assertEqual(entry.description, "Run shell")
        self.assertTrue(entry.visible)
        self.assertEqual(entry.permission, "default")

    def test_register_overwrites(self):
        self.reg.register("bash", "v1", _schema(), lambda **kw: "v1")
        # Default overwrite=False raises on duplicate; must opt in.
        with self.assertRaises(ValueError):
            self.reg.register("bash", "v2", _schema(), lambda **kw: "v2")
        # With overwrite=True, replacement succeeds.
        self.reg.register("bash", "v2", _schema(), lambda **kw: "v2", overwrite=True)
        entry = self.reg.get("bash")
        self.assertEqual(entry.description, "v2")

    def test_unregister(self):
        self.reg.register("bash", "Run", _schema(), lambda **kw: None)
        self.assertTrue(self.reg.unregister("bash"))
        self.assertFalse(self.reg.has("bash"))
        self.assertFalse(self.reg.unregister("bash"))

    def test_clear(self):
        self.reg.register("a", "A", _schema(), lambda **kw: None)
        self.reg.register("b", "B", _schema(), lambda **kw: None)
        self.reg.clear()
        self.assertEqual(self.reg.all_names(), [])


class ResolveTests(unittest.TestCase):
    """Stage 2B: profiles are explicit whitelists defined on the registry,
    not per-tool. We use a custom registry with test profiles to avoid
    coupling to STANDARD_PROFILES.
    """

    def setUp(self):
        # Custom profiles for this test suite.
        self.reg = ToolRegistry(profiles={
            "coding": ("bash", "read_file"),
            "readonly": ("read_file",),
        })
        self.reg.register("bash", "shell", _schema(command={"type": "string"}),
                          lambda **kw: None)
        self.reg.register("read_file", "read", _schema(path={"type": "string"}),
                          lambda **kw: None)
        self.reg.register("glob_search", "glob", _schema(pattern={"type": "string"}),
                          lambda **kw: None)
        # hidden tool (registered but not exposed)
        self.reg.register("idle", "idle", _schema(), lambda **kw: None,
                          visible=False)

    def test_resolve_all_visible(self):
        tools = self.reg.resolve()
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"bash", "read_file", "glob_search"})
        self.assertNotIn("idle", names)

    def test_resolve_coding_profile(self):
        tools = self.reg.resolve(profile="coding")
        names = {t["name"] for t in tools}
        # coding whitelist = bash, read_file. glob_search NOT in coding.
        self.assertEqual(names, {"bash", "read_file"})

    def test_resolve_readonly_profile(self):
        tools = self.reg.resolve(profile="readonly")
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"read_file"})
        self.assertNotIn("bash", names)

    def test_resolve_handlers_matches_resolve(self):
        tools = self.reg.resolve(profile="readonly")
        handlers = self.reg.resolve_handlers(profile="readonly")
        self.assertEqual({t["name"] for t in tools}, set(handlers.keys()))

    def test_resolved_tool_schema_shape(self):
        tools = self.reg.resolve(profile="coding")
        bash = next(t for t in tools if t["name"] == "bash")
        self.assertIn("name", bash)
        self.assertIn("description", bash)
        self.assertIn("input_schema", bash)
        self.assertEqual(bash["input_schema"]["required"], ["command"])

    def test_resolve_preserves_registration_order(self):
        # Registration order: bash, read_file, glob_search.
        # coding whitelist order is ("bash", "read_file") but resolve must
        # emit in REGISTRATION order, which happens to match here.
        tools = self.reg.resolve(profile="coding")
        self.assertEqual([t["name"] for t in tools], ["bash", "read_file"])
        # Now test with a profile whose whitelist order differs from registration.
        reg2 = ToolRegistry(profiles={"reversed": ("read_file", "bash")})
        reg2.register("bash", "shell", _schema(), lambda **kw: None)
        reg2.register("read_file", "read", _schema(), lambda **kw: None)
        tools2 = reg2.resolve(profile="reversed")
        # Registration order is bash, read_file — not the whitelist order.
        self.assertEqual([t["name"] for t in tools2], ["bash", "read_file"])

    def test_unknown_profile_raises(self):
        with self.assertRaises(UnknownToolProfileError):
            self.reg.resolve(profile="readnoly")  # typo
        with self.assertRaises(UnknownToolProfileError):
            self.reg.resolve_handlers(profile="nonexistent")

    def test_is_active_distinguishes_inactive_vs_unknown(self):
        # bash is registered but not active under readonly.
        self.assertTrue(self.reg.has("bash"))
        self.assertFalse(self.reg.is_active("bash", "readonly"))
        # read_file is registered and active under readonly.
        self.assertTrue(self.reg.is_active("read_file", "readonly"))
        # nonexistent is neither registered nor active.
        self.assertFalse(self.reg.is_active("nonexistent", "readonly"))
        self.assertFalse(self.reg.is_active("nonexistent", None))

    def test_known_profile(self):
        self.assertTrue(self.reg.known_profile("coding"))
        self.assertTrue(self.reg.known_profile("readonly"))
        self.assertFalse(self.reg.known_profile("readnoly"))


class ProfilesDefinedTests(unittest.TestCase):
    def test_standard_profiles_exist(self):
        for name in ("coding", "planning", "readonly", "team"):
            self.assertIn(name, PROFILES)
            self.assertIn(name, STANDARD_PROFILES)

    def test_standard_profiles_use_real_tool_names(self):
        """STANDARD_PROFILES must reference actual tool names that will be
        registered by harness_core. Spot-check against the known set."""
        all_referenced = set()
        for names in STANDARD_PROFILES.values():
            all_referenced.update(names)
        # These are the real names registered in harness_core._TOOL_DEFS.
        known_names = {
            "read_file", "write_file", "edit_file", "bash",
            "grep_search", "glob_search", "TodoWrite",
            "task_create", "task_get", "task_update", "task_list",
            "task", "spawn_teammate", "list_teammates", "send_message",
            "read_inbox", "broadcast", "shutdown_request", "plan_approval",
        }
        unknown = all_referenced - known_names
        self.assertEqual(unknown, set(),
                         f"STANDARD_PROFILES references unknown tools: {unknown}")

    def test_readonly_profile_excludes_write_and_bash(self):
        readonly_tools = set(STANDARD_PROFILES["readonly"])
        self.assertNotIn("write_file", readonly_tools)
        self.assertNotIn("edit_file", readonly_tools)
        self.assertNotIn("bash", readonly_tools)


if __name__ == "__main__":
    unittest.main()
