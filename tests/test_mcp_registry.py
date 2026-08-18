"""Tests for mcp.registry — MCPToolRegistry three-dimensional index.

Covers:
- register() and dedup
- O(1) lookup by name / server / domain
- filter_by_domain / filter_for_task
- all_schemas / server_schemas
- summary / health properties
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from mcp.registry import MCPToolEntry, MCPToolRegistry
from mcp.tools import MCPServerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(name: str = "test_server") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="echo", args=["hi"], transport="stdio")


def _make_tool(
    name: str,
    *,
    description: str = "",
    domain: str = "external_mcp",
    server: str = "srv",
    original_name: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description or f"Tool {name}",
        "input_schema": {"type": "object", "properties": {}},
        "metadata": {
            "tool_domain": domain,
            "_mcp_server": server,
            "_original_name": original_name or name,
        },
    }


@pytest.fixture()
def registry() -> MCPToolRegistry:
    return MCPToolRegistry()


@pytest.fixture()
def populated_registry() -> MCPToolRegistry:
    reg = MCPToolRegistry()
    cfg = _make_config("dbhub")
    tools = [
        _make_tool("dbhub__execute_sql", domain="external_mcp", server="dbhub"),
        _make_tool("dbhub__list_tables", domain="external_mcp", server="dbhub"),
    ]
    reg.register(tools, "dbhub", cfg, domain="external_mcp")

    cfg2 = _make_config("markitdown")
    tools2 = [
        _make_tool("markitdown__convert", domain="external_mcp", server="markitdown"),
    ]
    reg.register(tools2, "markitdown", cfg2, domain="external_mcp")

    # Add a knowledge-domain tool
    cfg3 = _make_config("knowledge_srv")
    tools3 = [
        _make_tool("kb__search", domain="knowledge", server="knowledge_srv"),
    ]
    reg.register(tools3, "knowledge_srv", cfg3, domain="knowledge")
    return reg


# ===================================================================
# Registration
# ===================================================================


class TestRegister:
    def test_register_returns_count(self, registry):
        cfg = _make_config()
        tools = [_make_tool("a"), _make_tool("b")]
        assert registry.register(tools, "srv", cfg) == 2

    def test_register_empty_list(self, registry):
        cfg = _make_config()
        assert registry.register([], "srv", cfg) == 0

    def test_register_dedup_same_name(self, registry):
        cfg = _make_config()
        tools = [_make_tool("dup")]
        registry.register(tools, "srv1", cfg)
        # Second registration with same name is skipped
        assert registry.register([_make_tool("dup")], "srv2", cfg) == 0

    def test_register_tool_without_name_skipped(self, registry):
        cfg = _make_config()
        tools = [{"description": "no name"}]
        assert registry.register(tools, "srv", cfg) == 0

    def test_register_uses_metadata_domain(self, registry):
        cfg = _make_config()
        tool = _make_tool("x", domain="knowledge")
        registry.register([tool], "srv", cfg, domain="external_mcp")
        entry = registry.get("x")
        assert entry is not None
        # metadata domain takes precedence
        assert entry.domain == "knowledge"

    def test_register_falls_back_to_param_domain(self, registry):
        cfg = _make_config()
        tool = {"name": "y", "description": "", "input_schema": {}, "metadata": {}}
        registry.register([tool], "srv", cfg, domain="memory")
        entry = registry.get("y")
        assert entry is not None
        assert entry.domain == "memory"


# ===================================================================
# Three-dimensional lookup
# ===================================================================


class TestLookup:
    def test_get_existing(self, populated_registry):
        entry = populated_registry.get("dbhub__execute_sql")
        assert entry is not None
        assert entry.name == "dbhub__execute_sql"
        assert entry.server == "dbhub"

    def test_get_nonexistent(self, populated_registry):
        assert populated_registry.get("nonexistent") is None

    def test_get_server(self, populated_registry):
        assert populated_registry.get_server("dbhub__execute_sql") == "dbhub"

    def test_get_server_prefix_fallback(self, populated_registry):
        # Tool not registered but has __ separator
        assert populated_registry.get_server("unknown__tool") == "unknown"

    def test_get_server_no_match(self, populated_registry):
        assert populated_registry.get_server("noprefix") is None

    def test_get_config(self, populated_registry):
        cfg = populated_registry.get_config("dbhub__list_tables")
        assert cfg is not None
        assert cfg.name == "dbhub"

    def test_get_config_nonexistent(self, populated_registry):
        assert populated_registry.get_config("nope") is None

    def test_list_servers(self, populated_registry):
        servers = populated_registry.list_servers()
        assert set(servers) == {"dbhub", "markitdown", "knowledge_srv"}

    def test_list_tools_all(self, populated_registry):
        tools = populated_registry.list_tools()
        assert len(tools) == 4

    def test_list_tools_by_server(self, populated_registry):
        tools = populated_registry.list_tools("dbhub")
        assert set(tools) == {"dbhub__execute_sql", "dbhub__list_tables"}

    def test_list_tools_unknown_server(self, populated_registry):
        assert populated_registry.list_tools("unknown") == []

    def test_list_domains(self, populated_registry):
        domains = populated_registry.list_domains()
        assert set(domains) == {"external_mcp", "knowledge"}

    def test_total_tools(self, populated_registry):
        assert populated_registry.total_tools == 4

    def test_total_servers(self, populated_registry):
        assert populated_registry.total_servers == 3


# ===================================================================
# Index lookup speed (quantitative)
# ===================================================================


class TestIndexSpeed:
    def test_name_lookup_o1(self, populated_registry):
        """Name lookup should be O(1) dict access — constant regardless of size."""
        # Warm up
        populated_registry.get("dbhub__execute_sql")
        start = time.perf_counter()
        for _ in range(10000):
            populated_registry.get("dbhub__execute_sql")
        elapsed = time.perf_counter() - start
        # 10k lookups should finish in < 100ms on any modern machine
        assert elapsed < 0.1, f"10k name lookups took {elapsed:.3f}s"

    def test_domain_filter_performance(self, populated_registry):
        """filter_by_domain should be fast even with multiple domains."""
        start = time.perf_counter()
        for _ in range(1000):
            populated_registry.filter_by_domain({"external_mcp"})
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"1k domain filters took {elapsed:.3f}s"


# ===================================================================
# Filtering
# ===================================================================


class TestFiltering:
    def test_filter_by_domain(self, populated_registry):
        schemas = populated_registry.filter_by_domain({"external_mcp"})
        names = {s["name"] for s in schemas}
        assert names == {"dbhub__execute_sql", "dbhub__list_tables", "markitdown__convert"}

    def test_filter_by_domain_empty_set(self, populated_registry):
        assert populated_registry.filter_by_domain(set()) == []

    def test_filter_by_domain_multiple(self, populated_registry):
        schemas = populated_registry.filter_by_domain({"external_mcp", "knowledge"})
        assert len(schemas) == 4

    def test_filter_for_task_with_llm_domains(self, populated_registry):
        schemas = populated_registry.filter_for_task(
            "query db", llm_domains=["knowledge"]
        )
        # Should include knowledge + always-active domains (but registry only has registered tools)
        names = {s["name"] for s in schemas}
        assert "kb__search" in names

    def test_filter_for_task_empty_topic_returns_all(self, populated_registry):
        schemas = populated_registry.filter_for_task("")
        assert len(schemas) == 4

    def test_filter_for_task_keyword_fallback(self, populated_registry):
        # "database" is a keyword for external_mcp
        schemas = populated_registry.filter_for_task("query the database")
        names = {s["name"] for s in schemas}
        assert "dbhub__execute_sql" in names


# ===================================================================
# Schema output
# ===================================================================


class TestSchemaOutput:
    def test_all_schemas_structure(self, populated_registry):
        schemas = populated_registry.all_schemas()
        assert len(schemas) == 4
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s
            assert "metadata" in s
            assert "tool_domain" in s["metadata"]
            assert "_mcp_server" in s["metadata"]

    def test_server_schemas(self, populated_registry):
        schemas = populated_registry.server_schemas("dbhub")
        assert len(schemas) == 2
        assert all(s["metadata"]["_mcp_server"] == "dbhub" for s in schemas)

    def test_server_schemas_unknown(self, populated_registry):
        assert populated_registry.server_schemas("unknown") == []


# ===================================================================
# Summary
# ===================================================================


class TestSummary:
    def test_summary_contains_info(self, populated_registry):
        text = populated_registry.summary()
        assert "4 tools" in text
        assert "3 servers" in text
        assert "dbhub" in text
        assert "markitdown" in text
