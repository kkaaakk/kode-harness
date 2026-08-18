"""MCP (Model Context Protocol) server integration package.

Provides multi-server MCP tool loading, centralized tool registry,
domain-based filtering, tool execution, and system prompt injection.

Adapted from open_deep_research for the learn-claude-code project.
Uses native MCP SDK and Anthropic SDK — no LangChain/LangGraph dependency.

Quick start
-----------

.. code-block:: python

    from mcp import assemble_mcp_tools_sync

    # Startup: discover MCP tools, register, build executor
    mcp = assemble_mcp_tools_sync(
        existing_tool_names={"bash", "read_file", ...},
    )
    all_tools = my_builtin_schemas + mcp.schemas

    # Domain filtering (per task, on MCP tools only)
    filtered = mcp.filter_for_task(
        research_topic="query the database",
        llm_domains=["external_mcp"],
    )

    # Execute MCP tool calls
    result = mcp.dispatch("dbhub__execute_sql", {"sql": "SELECT 1"})
"""

from .domain_filter import (
    DOMAIN_REGISTRY,
    DomainDef,
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
from .executor import MCPToolExecutor
from .registry import MCPToolEntry, MCPToolRegistry
from .tool_assembler import (
    AssembledTools,
    assemble_mcp_tools,
    assemble_mcp_tools_sync,
    build_domain_injection_prompt,
    parse_domains_from_text,
)
from .tools import (
    MCPServerConfig,
    build_server_configs,
    load_mcp_tools,
    load_mcp_tools_sync,
)

__all__ = [
    # Domain registry & types
    "DOMAIN_REGISTRY",
    "DomainDef",
    # Domain helpers
    "build_domain_classifier_prompt",
    "classify_tools",
    "detect_active_domains",
    "filter_tools_by_domain",
    "get_domain",
    "get_domain_description",
    "get_domain_label",
    "get_filtered_tools",
    "iter_domain_labels",
    "tag_builtin_tools",
    "tool_domain_summary",
    # MCP config & loading
    "MCPServerConfig",
    "build_server_configs",
    "load_mcp_tools",
    "load_mcp_tools_sync",
    # MCP registry
    "MCPToolEntry",
    "MCPToolRegistry",
    # MCP execution
    "MCPToolExecutor",
    # High-level assembly
    "AssembledTools",
    "assemble_mcp_tools",
    "assemble_mcp_tools_sync",
    "build_domain_injection_prompt",
    "parse_domains_from_text",
]
