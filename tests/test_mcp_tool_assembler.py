"""Tests for mcp.tool_assembler — Domain parsing and injection prompts.

Covers:
- parse_domains_from_text extraction
- build_domain_injection_prompt
- AssembledTools dataclass (filter_for_task, is_mcp_tool, dispatch)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mcp.domain_filter import DOMAIN_REGISTRY
from mcp.tool_assembler import (
    AssembledTools,
    build_domain_injection_prompt,
    parse_domains_from_text,
)


# ===================================================================
# parse_domains_from_text
# ===================================================================


class TestParseDomainsFromText:
    def test_single_domain(self):
        text = """Here's my plan:
        <relevant_domains>knowledge</relevant_domains>
        I will search the codebase."""
        result = parse_domains_from_text(text)
        assert result == ["knowledge"]

    def test_multiple_domains(self):
        text = "<relevant_domains>knowledge, delegation, team</relevant_domains>"
        result = parse_domains_from_text(text)
        assert set(result) == {"knowledge", "delegation", "team"}

    def test_none_value(self):
        text = """No special domains needed.
        <relevant_domains>none</relevant_domains>"""
        result = parse_domains_from_text(text)
        assert result == []

    def test_empty_value(self):
        text = "<relevant_domains>   </relevant_domains>"
        result = parse_domains_from_text(text)
        assert result == []

    def test_no_match(self):
        text = "Just a normal response without tags."
        result = parse_domains_from_text(text)
        assert result == []

    def test_case_insensitive(self):
        text = "<RELEVANT_DOMAINS>KNOWLEDGE, TEAM</RELEVANT_DOMAINS>"
        result = parse_domains_from_text(text)
        assert set(result) == {"knowledge", "team"}

    def test_filters_unknown_domains(self):
        text = "<relevant_domains>knowledge, nonexistent_domain, team</relevant_domains>"
        result = parse_domains_from_text(text)
        assert "knowledge" in result
        assert "team" in result
        assert "nonexistent_domain" not in result

    def test_whitespace_in_list(self):
        text = "<relevant_domains>  knowledge ,   delegation  </relevant_domains>"
        result = parse_domains_from_text(text)
        assert result == ["knowledge", "delegation"]

    def test_trailing_comma(self):
        text = "<relevant_domains>knowledge, </relevant_domains>"
        result = parse_domains_from_text(text)
        assert result == ["knowledge"]

    def test_multiple_tags_takes_first(self):
        text = """<relevant_domains>knowledge</relevant_domains>
        <relevant_domains>team</relevant_domains>"""
        result = parse_domains_from_text(text)
        assert result == ["knowledge"]


# ===================================================================
# build_domain_injection_prompt
# ===================================================================


class TestBuildDomainInjectionPrompt:
    def test_includes_domain_catalog(self):
        prompt = build_domain_injection_prompt()
        # Should contain some domain names
        assert "knowledge" in prompt or "Knowledge" in prompt
        assert "delegation" in prompt or "Delegation" in prompt

    def test_includes_instructions(self):
        prompt = build_domain_injection_prompt()
        assert "<relevant_domains>" in prompt
        assert "Tool Domain System" in prompt or "Tool" in prompt

    def test_excludes_always_active_from_catalog(self):
        prompt = build_domain_injection_prompt()
        # core/file/planning are always active - they shouldn't need classification
        # But the prompt template mentions them as always available
        assert "always active" in prompt.lower() or "always available" in prompt.lower()


# ===================================================================
# AssembledTools
# ===================================================================


class TestAssembledTools:
    @pytest.fixture()
    def simple_assembled(self):
        """Create a minimal AssembledTools with one registry entry."""
        from mcp.registry import MCPToolRegistry
        from mcp.executor import MCPToolExecutor
        from mcp.tools import MCPServerConfig

        reg = MCPToolRegistry()
        cfg = MCPServerConfig(name="dbhub", command="npx", args=["-y", "dbhub"], transport="stdio")
        tools = [
            {
                "name": "dbhub__query",
                "description": "Query DB",
                "input_schema": {"type": "object"},
                "metadata": {
                    "tool_domain": "external_mcp",
                    "_mcp_server": "dbhub",
                    "_original_name": "query",
                },
            },
        ]
        reg.register(tools, "dbhub", cfg)

        return AssembledTools(
            registry=reg,
            executor=MCPToolExecutor(reg),
            schemas=reg.all_schemas(),
            server_configs={"dbhub": cfg},
        )

    def test_is_mcp_tool_true(self, simple_assembled):
        assert simple_assembled.is_mcp_tool("dbhub__query") is True

    def test_is_mcp_tool_false(self, simple_assembled):
        assert simple_assembled.is_mcp_tool("bash") is False

    def test_is_mcp_tool_nonexistent(self, simple_assembled):
        assert simple_assembled.is_mcp_tool("nonexistent") is False

    def test_filter_for_task(self, simple_assembled):
        schemas = simple_assembled.filter_for_task("query database")
        # external_mcp has "database" keyword
        assert len(schemas) > 0
        assert any(s["name"] == "dbhub__query" for s in schemas)

    def test_filter_for_task_empty(self, simple_assembled):
        schemas = simple_assembled.filter_for_task("")
        assert len(schemas) == 1

    def test_dispatch_uses_executor(self, simple_assembled):
        # Mock the executor.execute_sync to avoid real server connection
        with patch.object(simple_assembled.executor, 'execute_sync', return_value="ok") as mock_exec:
            result = simple_assembled.dispatch("dbhub__query", {"sql": "SELECT 1"})
            assert result == "ok"
            mock_exec.assert_called_once_with("dbhub__query", {"sql": "SELECT 1"})

    def test_adispatch_uses_executor(self, simple_assembled):
        async def mock_execute(name, args):
            return "async result"

        with patch.object(simple_assembled.executor, 'execute', new=mock_execute):
            import asyncio
            result = asyncio.run(simple_assembled.adispatch("dbhub__query", {}))
            assert result == "async result"

    def test_schemas_structure(self, simple_assembled):
        schemas = simple_assembled.schemas
        assert len(schemas) == 1
        s = schemas[0]
        assert s["name"] == "dbhub__query"
        assert s["metadata"]["tool_domain"] == "external_mcp"
        assert s["metadata"]["_mcp_server"] == "dbhub"

    def test_server_configs_present(self, simple_assembled):
        assert "dbhub" in simple_assembled.server_configs
        assert simple_assembled.server_configs["dbhub"].name == "dbhub"
