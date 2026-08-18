"""test_d3_final_composition_contract.py - Stage 2D-D3 final composition contract.

The single authoritative end-of-stage contract for the Pi-style tool
Extension migration (Stage 2A..2D). After D3, Stage 2 is CLOSED: no more
tool registration migrations until Runtime Unification / later phases.

This module locks:

  1.  Base Registry composition: exactly the 12 classified Base tools
      (6 coding atoms + 4 lifecycle/infrastructure tools + 2 team-member
      hardcoded tools parked in Base as registered tech debt).
  2.  The D3-2 Tool Ownership Table - for every one of the 25 legacy
      tools: owner (kernel | todo-extension | task-extension |
      subagent-extension | team-extension), source (builtin | extension),
      default visibility, and per-profile membership (None/coding/
      planning/readonly/team).
  3.  The D3-4 final composition snapshot:
        - Base count == 12
        - default composed count == 25
        - default composed names == LEGACY_25_TOOL_NAMES (exact order)
        - composed ownership == EXPECTED_TOOL_OWNERSHIP
        - resolved schemas == the schemas the model saw pre-2A
        - legacy TOOLS / TOOL_HANDLERS keep 25 entries in legacy order
  4.  Profile snapshots for None/coding/planning/readonly/team resolved
      against the default composition.
  5.  LEGACY_TOOL_ORDER mechanical consolidation: the dict covers all 25
      tools, values equal pre-D3 per-tool constants, and every legacy
      per-tool constant is an alias into the dict.

If a future change (Stage 3 Provider or beyond) intentionally alters any
of the above, it MUST update this file in the same commit - this is the
guardrail that keeps the tool system stable across provider work.
"""

import unittest

import agents.harness_core as hc


def _resolve_names(registry, profile):
    return [t["name"] for t in registry.resolve(profile)]


def _ownership(registry):
    """name -> owner for every tool visible under profile=None."""
    return {
        entry.name: entry.owner
        for entry in registry.all_entries()
        if registry.is_active(entry.name, None)
    }


# ---------------------------------------------------------------------------
# D3-2: The Tool Ownership Table (the single source of truth snapshot).
# owner: kernel | todo-extension | task-extension | subagent-extension |
#        team-extension. default: whether the model sees it with
# profile=None under DEFAULT_TOOL_CONTRIBUTORS. profiles: which of
# coding/planning/readonly/team include it.
# ---------------------------------------------------------------------------

EXPECTED_TOOL_OWNERSHIP = {
    # --- kernel (Base, 12) ------------------------------------------------
    "bash": "kernel",
    "read_file": "kernel",
    "write_file": "kernel",
    "edit_file": "kernel",
    "load_skill": "kernel",
    "compress": "kernel",
    "background_run": "kernel",
    "check_background": "kernel",
    "idle": "kernel",
    "claim_task": "kernel",
    "grep_search": "kernel",
    "glob_search": "kernel",
    # --- todo-extension (1) -------------------------------------------------
    "TodoWrite": "todo-extension",
    # --- task-extension (4) --------------------------------------------------
    "task_create": "task-extension",
    "task_get": "task-extension",
    "task_update": "task-extension",
    "task_list": "task-extension",
    # --- subagent-extension (1) ----------------------------------------------
    "task": "subagent-extension",
    # --- team-extension (7) ---------------------------------------------------
    "spawn_teammate": "team-extension",
    "list_teammates": "team-extension",
    "send_message": "team-extension",
    "read_inbox": "team-extension",
    "broadcast": "team-extension",
    "shutdown_request": "team-extension",
    "plan_approval": "team-extension",
}

EXPECTED_TOOL_SOURCE = {
    **{name: "builtin" for name in (
        "bash", "read_file", "write_file", "edit_file", "load_skill",
        "compress", "background_run", "check_background", "idle",
        "claim_task", "grep_search", "glob_search",
    )},
    **{name: "extension" for name in (
        "TodoWrite", "task_create", "task_get", "task_update", "task_list",
        "task", "spawn_teammate", "list_teammates", "send_message",
        "read_inbox", "broadcast", "shutdown_request", "plan_approval",
    )},
}

# D3-1 classification of the 12 Base tools (documented here as a locked
# snapshot; see test_base_classification below).
EXPECTED_BASE_12 = [
    # coding atoms (6) - stay in Base permanently
    "bash", "read_file", "write_file", "edit_file", "grep_search",
    "glob_search",
    # lifecycle / infrastructure (4) - stay in Base for now; candidates
    # for a future SkillSystem/Background extension (Stage 4+)
    "load_skill", "compress", "background_run", "check_background",
    # team-member hardcoded tools (2) - parked in Base; migrate together
    # with Team member tool set during Runtime Unification (tech debt)
    "idle", "claim_task",
]

# D3-4: profile membership snapshots (name sets, resolved under the
# default composition). Order asserted separately via LEGACY order.
EXPECTED_PROFILE_SNAPSHOTS = {
    "coding": {
        "read_file", "write_file", "edit_file", "bash",
        "grep_search", "glob_search",
    },
    "planning": {
        "read_file", "grep_search", "glob_search",
        "TodoWrite", "task_create", "task_get", "task_update", "task_list",
    },
    "readonly": {
        "read_file", "grep_search", "glob_search",
    },
    "team": {
        "read_file", "write_file", "edit_file", "bash",
        "grep_search", "glob_search", "task",
        "spawn_teammate", "list_teammates", "send_message", "read_inbox",
        "broadcast", "shutdown_request", "plan_approval",
    },
}


class D3FinalCompositionContract(unittest.TestCase):

    def setUp(self):
        self.registry = hc.build_default_tool_registry()

    # ------------------------------------------------------------------
    # 1. Base composition (D3-1)
    # ------------------------------------------------------------------

    def test_base_registry_has_exactly_12_tools(self):
        self.assertEqual(len(hc.BASE_TOOL_REGISTRY.all_names()), 12)

    def test_base_registry_composition_is_exactly_expected_12(self):
        self.assertEqual(
            sorted(hc.BASE_TOOL_REGISTRY.all_names()),
            sorted(EXPECTED_BASE_12),
        )

    def test_base_registry_has_no_extension_tools(self):
        extension_tools = set(EXPECTED_TOOL_OWNERSHIP) - set(EXPECTED_BASE_12)
        base_names = set(hc.BASE_TOOL_REGISTRY.all_names())
        self.assertEqual(base_names & extension_tools, set())

    def test_base_tool_owners_are_kernel(self):
        for entry in hc.BASE_TOOL_REGISTRY.all_entries():
            self.assertEqual(
                entry.owner, "kernel",
                f"Base tool {entry.name} owner is {entry.owner!r}",
            )

    # ------------------------------------------------------------------
    # 2. Ownership table (D3-2)
    # ------------------------------------------------------------------

    def test_default_composition_ownership_matches_table(self):
        self.assertEqual(_ownership(self.registry), EXPECTED_TOOL_OWNERSHIP)

    def test_default_composition_source_matches_table(self):
        sources = {
            entry.name: entry.source
            for entry in self.registry.all_entries()
            if self.registry.is_active(entry.name, None)
        }
        self.assertEqual(sources, EXPECTED_TOOL_SOURCE)

    def test_ownership_table_covers_exactly_25_tools(self):
        self.assertEqual(len(EXPECTED_TOOL_OWNERSHIP), 25)
        self.assertEqual(set(EXPECTED_TOOL_OWNERSHIP), set(hc.LEGACY_25_TOOL_NAMES))

    # ------------------------------------------------------------------
    # 3. Final composition snapshot (D3-4)
    # ------------------------------------------------------------------

    def test_default_composition_is_25_tools_in_legacy_order(self):
        names = _resolve_names(self.registry, None)
        self.assertEqual(len(names), 25)
        self.assertEqual(names, hc.LEGACY_25_TOOL_NAMES)

    def test_legacy_tools_and_handlers_keep_25_in_legacy_order(self):
        self.assertEqual(len(hc.TOOLS), 25)
        self.assertEqual(
            [t["name"] for t in hc.TOOLS], hc.LEGACY_25_TOOL_NAMES
        )
        self.assertEqual(len(hc.TOOL_HANDLERS), 25)
        self.assertEqual(
            list(hc.TOOL_HANDLERS.keys()), hc.LEGACY_25_TOOL_NAMES
        )

    def test_default_resolved_schemas_match_legacy_tools(self):
        resolved = self.registry.resolve(None)
        for resolved_tool, legacy_tool in zip(resolved, hc.TOOLS):
            self.assertEqual(resolved_tool["name"], legacy_tool["name"])
            self.assertEqual(
                resolved_tool["input_schema"], legacy_tool["input_schema"]
            )

    def test_profile_none_sees_all_25(self):
        self.assertEqual(len(_resolve_names(self.registry, None)), 25)

    # ------------------------------------------------------------------
    # 4. Profile snapshots (D3-4)
    # ------------------------------------------------------------------

    def test_profile_snapshots_match_expected(self):
        for profile, expected_names in EXPECTED_PROFILE_SNAPSHOTS.items():
            with self.subTest(profile=profile):
                resolved = set(_resolve_names(self.registry, profile))
                self.assertEqual(resolved, expected_names)

    def test_profile_resolution_emits_in_legacy_order(self):
        legacy = hc.LEGACY_25_TOOL_NAMES
        for profile in ("coding", "planning", "readonly", "team"):
            with self.subTest(profile=profile):
                names = _resolve_names(self.registry, profile)
                expected_order = [n for n in legacy if n in set(names)]
                self.assertEqual(names, expected_order)

    def test_profiles_do_not_leak_tools_across_boundaries(self):
        # Installing TeamExtension must not add team tools to coding /
        # readonly / planning profiles.
        for profile in ("coding", "readonly", "planning"):
            with self.subTest(profile=profile):
                resolved = set(_resolve_names(self.registry, profile))
                team_tools = EXPECTED_PROFILE_SNAPSHOTS["team"] - \
                    EXPECTED_PROFILE_SNAPSHOTS["coding"]
                self.assertEqual(resolved & team_tools, set())

    # ------------------------------------------------------------------
    # 5. LEGACY_TOOL_ORDER consolidation (D3-3)
    # ------------------------------------------------------------------

    def test_legacy_tool_order_covers_all_25(self):
        self.assertEqual(
            set(hc.LEGACY_TOOL_ORDER.keys()), set(hc.LEGACY_25_TOOL_NAMES)
        )
        self.assertEqual(len(hc.LEGACY_TOOL_ORDER), 25)

    def test_legacy_tool_order_values_match_legacy_list(self):
        expected = {n: i for i, n in enumerate(hc.LEGACY_25_TOOL_NAMES)}
        self.assertEqual(hc.LEGACY_TOOL_ORDER, expected)

    def test_per_tool_order_constants_alias_the_dict(self):
        # Mechanical consolidation: the pre-D3 per-tool constants must be
        # exact aliases into LEGACY_TOOL_ORDER (same values as before D3).
        aliases = {
            "TodoWrite": hc.LEGACY_TODO_WRITE_ORDER,
            "task_create": hc.LEGACY_TASK_CREATE_ORDER,
            "task_get": hc.LEGACY_TASK_GET_ORDER,
            "task_update": hc.LEGACY_TASK_UPDATE_ORDER,
            "task_list": hc.LEGACY_TASK_LIST_ORDER,
            "task": hc.LEGACY_SUBAGENT_ORDER,
            "spawn_teammate": hc.LEGACY_SPAWN_TEAMMATE_ORDER,
            "list_teammates": hc.LEGACY_LIST_TEAMMATES_ORDER,
            "send_message": hc.LEGACY_SEND_MESSAGE_ORDER,
            "read_inbox": hc.LEGACY_READ_INBOX_ORDER,
            "broadcast": hc.LEGACY_BROADCAST_ORDER,
            "shutdown_request": hc.LEGACY_SHUTDOWN_REQUEST_ORDER,
            "plan_approval": hc.LEGACY_PLAN_APPROVAL_ORDER,
        }
        for name, constant in aliases.items():
            with self.subTest(tool=name):
                self.assertEqual(constant, hc.LEGACY_TOOL_ORDER[name])

    def test_base_tool_orders_match_legacy_dict(self):
        for entry in hc.BASE_TOOL_REGISTRY.all_entries():
            with self.subTest(tool=entry.name):
                self.assertEqual(entry.order, hc.LEGACY_TOOL_ORDER[entry.name])

    # ------------------------------------------------------------------
    # 6. Stage 2 closure guardrails
    # ------------------------------------------------------------------

    def test_default_tool_contributors_are_the_four_extensions(self):
        ids = [c.extension_id for c in hc.DEFAULT_TOOL_CONTRIBUTORS]
        self.assertEqual(
            ids,
            ["todo-extension", "task-extension",
             "subagent-extension", "team-extension"],
        )

    def test_stage2_migration_counts(self):
        # Base 25 -> 12; extensions contribute 1 + 4 + 1 + 7 = 13.
        self.assertEqual(len(hc.BASE_TOOL_REGISTRY.all_names()), 12)
        ext_names = set(EXPECTED_TOOL_OWNERSHIP) - set(EXPECTED_BASE_12)
        self.assertEqual(len(ext_names), 13)


if __name__ == "__main__":
    unittest.main()
