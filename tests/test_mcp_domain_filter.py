"""Tests for mcp.domain_filter — DomainDef registry, keyword detection, filtering.

Covers:
- DOMAIN_REGISTRY integrity
- get_domain / get_domain_description / get_domain_label
- iter_domain_labels ordering
- build_domain_classifier_prompt
- classify_tools grouping
- detect_active_domains keyword fallback
- filter_tools_by_domain pruning
- get_filtered_tools convenience
- tag_builtin_tools metadata injection
- tool_domain_summary
"""

from __future__ import annotations

from typing import Any

import pytest

from mcp.domain_filter import (
    DOMAIN_REGISTRY,
    DomainDef,
    _ALWAYS_ACTIVE_DOMAINS,
    _BUILTIN_NAME_TO_DOMAIN,
    _DOMAIN_BY_NAME,
    build_domain_classifier_prompt,
    classify_tools,
    detect_active_domains,
    filter_tools_by_domain,
    get_domain,
    get_domain_description,
    get_domain_label,
    get_filtered_tools,
    iter_domain_labels,
    tag_builtin_tools,
    tool_domain_summary,
)


# ===================================================================
# DOMAIN_REGISTRY integrity
# ===================================================================


class TestDomainRegistry:
    def test_registry_not_empty(self):
        assert len(DOMAIN_REGISTRY) > 0

    def test_all_domains_have_unique_names(self):
        names = [d.name for d in DOMAIN_REGISTRY]
        assert len(names) == len(set(names))

    def test_always_active_domains_exist(self):
        assert "core" in _ALWAYS_ACTIVE_DOMAINS
        assert "file" in _ALWAYS_ACTIVE_DOMAINS
        assert "planning" in _ALWAYS_ACTIVE_DOMAINS

    def test_on_demand_domains_have_keywords(self):
        for d in DOMAIN_REGISTRY:
            if not d.always_active:
                assert d.keywords, f"Domain '{d.name}' should have keywords"

    def test_domain_by_name_complete(self):
        for d in DOMAIN_REGISTRY:
            assert d.name in _DOMAIN_BY_NAME


# ===================================================================
# Public helpers
# ===================================================================


class TestPublicHelpers:
    def test_get_domain_exists(self):
        d = get_domain("file")
        assert d is not None
        assert d.name == "file"
        assert d.always_active is True

    def test_get_domain_not_found(self):
        assert get_domain("nonexistent") is None

    def test_get_domain_description(self):
        desc = get_domain_description("knowledge")
        assert "Search" in desc or "search" in desc.lower()

    def test_get_domain_description_unknown(self):
        assert get_domain_description("nope") == ""

    def test_get_domain_label(self):
        label = get_domain_label("file")
        assert "File" in label

    def test_get_domain_label_unknown(self):
        # Unknown domain returns the name itself
        assert get_domain_label("unknown_xyz") == "unknown_xyz"


# ===================================================================
# iter_domain_labels
# ===================================================================


class TestIterDomainLabels:
    def test_basic(self):
        pairs = iter_domain_labels({"file", "knowledge"})
        names = [n for n, _ in pairs]
        assert "file" in names
        assert "knowledge" in names

    def test_preserves_registry_order(self):
        pairs = iter_domain_labels({"knowledge", "file"})
        names = [n for n, _ in pairs]
        # file should come before knowledge in registry order
        assert names.index("file") < names.index("knowledge")

    def test_empty_set(self):
        assert iter_domain_labels(set()) == []


# ===================================================================
# build_domain_classifier_prompt
# ===================================================================


class TestBuildClassifierPrompt:
    def test_excludes_always_active(self):
        prompt = build_domain_classifier_prompt()
        # always-active domains shouldn't appear in classifier prompt
        assert "core" not in prompt.split("**")[1::2] if "**" in prompt else True

    def test_includes_on_demand_domains(self):
        prompt = build_domain_classifier_prompt()
        assert "knowledge" in prompt
        assert "external_mcp" in prompt

    def test_includes_keyword_hints(self):
        prompt = build_domain_classifier_prompt()
        # knowledge domain has "search" as keyword
        assert "search" in prompt


# ===================================================================
# classify_tools
# ===================================================================


class TestClassifyTools:
    def test_group_by_domain(self):
        tools = [
            {"name": "bash", "metadata": {"tool_domain": "file"}},
            {"name": "grep_search", "metadata": {"tool_domain": "knowledge"}},
            {"name": "read_file", "metadata": {"tool_domain": "file"}},
        ]
        buckets = classify_tools(tools)
        assert "file" in buckets
        assert len(buckets["file"]) == 2
        assert len(buckets["knowledge"]) == 1

    def test_builtin_name_fallback(self):
        # No metadata → falls back to name-based matching
        tools = [
            {"name": "bash"},
            {"name": "grep_search"},
        ]
        buckets = classify_tools(tools)
        assert "file" in buckets
        assert "knowledge" in buckets

    def test_unknown_tool_goes_to_external(self):
        tools = [{"name": "some_random_tool"}]
        buckets = classify_tools(tools)
        assert "external_mcp" in buckets

    def test_empty_list(self):
        assert classify_tools([]) == {}


# ===================================================================
# detect_active_domains
# ===================================================================


class TestDetectActiveDomains:
    def test_always_active_included(self):
        active = detect_active_domains("anything")
        assert _ALWAYS_ACTIVE_DOMAINS.issubset(active)

    def test_keyword_database_triggers_external_mcp(self):
        active = detect_active_domains("query the database")
        assert "external_mcp" in active

    def test_keyword_search_triggers_knowledge(self):
        active = detect_active_domains("search the codebase")
        assert "knowledge" in active

    def test_keyword_team_triggers_team(self):
        active = detect_active_domains("coordinate with the team")
        assert "team" in active

    def test_keyword_memory_triggers_memory(self):
        active = detect_active_domains("remember this preference")
        assert "memory" in active

    def test_no_extra_domains_for_generic_topic(self):
        active = detect_active_domains("hello world")
        # Should only have always-active domains
        assert active == _ALWAYS_ACTIVE_DOMAINS

    def test_multiple_keywords(self):
        active = detect_active_domains("search database and spawn subagent")
        assert "knowledge" in active or "external_mcp" in active
        assert "delegation" in active


# ===================================================================
# filter_tools_by_domain
# ===================================================================


class TestFilterToolsByDomain:
    def test_keeps_matching_domain(self):
        tools = [
            {"name": "bash", "metadata": {"tool_domain": "file"}},
            {"name": "grep_search", "metadata": {"tool_domain": "knowledge"}},
        ]
        result = filter_tools_by_domain(tools, {"file"})
        assert len(result) == 1
        assert result[0]["name"] == "bash"

    def test_drops_non_matching(self):
        tools = [
            {"name": "a", "metadata": {"tool_domain": "memory"}},
            {"name": "b", "metadata": {"tool_domain": "file"}},
        ]
        result = filter_tools_by_domain(tools, {"team"})
        assert result == []

    def test_empty_active_set(self):
        tools = [{"name": "x", "metadata": {"tool_domain": "file"}}]
        assert filter_tools_by_domain(tools, set()) == []

    def test_all_match(self):
        tools = [
            {"name": "a", "metadata": {"tool_domain": "file"}},
            {"name": "b", "metadata": {"tool_domain": "file"}},
        ]
        result = filter_tools_by_domain(tools, {"file"})
        assert len(result) == 2


# ===================================================================
# get_filtered_tools
# ===================================================================


class TestGetFilteredTools:
    def test_keyword_based_filtering(self):
        tools = [
            {"name": "bash", "metadata": {"tool_domain": "file"}},
            {"name": "memory_write", "metadata": {"tool_domain": "memory"}},
            {"name": "grep_search", "metadata": {"tool_domain": "knowledge"}},
        ]
        # "search" triggers knowledge
        result = get_filtered_tools(tools, "search the codebase")
        names = [t["name"] for t in result]
        assert "grep_search" in names
        # file is always active
        assert "bash" in names
        # memory should be excluded (not triggered)
        assert "memory_write" not in names


# ===================================================================
# tag_builtin_tools
# ===================================================================


class TestTagBuiltinTools:
    def test_tags_builtin_tools(self):
        tools = [
            {"name": "bash", "metadata": {}},
            {"name": "grep_search", "metadata": {}},
        ]
        tag_builtin_tools(tools)
        assert tools[0]["metadata"]["tool_domain"] == "file"
        assert tools[1]["metadata"]["tool_domain"] == "knowledge"

    def test_preserves_existing_domain(self):
        tools = [
            {"name": "bash", "metadata": {"tool_domain": "custom"}},
        ]
        tag_builtin_tools(tools)
        assert tools[0]["metadata"]["tool_domain"] == "custom"

    def test_unknown_builtin_gets_external(self):
        tools = [
            {"name": "unknown_tool", "metadata": {}},
        ]
        tag_builtin_tools(tools)
        assert tools[0]["metadata"]["tool_domain"] == "external_mcp"

    def test_no_metadata_creates_new(self):
        tools = [{"name": "bash"}]
        tag_builtin_tools(tools)
        assert tools[0]["metadata"]["tool_domain"] == "file"

    def test_empty_list(self):
        assert tag_builtin_tools([]) == []


# ===================================================================
# tool_domain_summary
# ===================================================================


class TestToolDomainSummary:
    def test_summary_output(self):
        tools = [
            {"name": "bash", "metadata": {"tool_domain": "file"}},
            {"name": "grep_search", "metadata": {"tool_domain": "knowledge"}},
        ]
        text = tool_domain_summary(tools)
        assert "[file]" in text
        assert "[knowledge]" in text
        assert "bash" in text

    def test_empty_tools(self):
        assert tool_domain_summary([]) == ""
