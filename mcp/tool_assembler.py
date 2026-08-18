"""Tool assembly — loads MCP tools into a centralized registry.

The main entry point is :func:`assemble_mcp_tools`, which discovers tools
from all configured MCP servers, registers them in an :class:`MCPToolRegistry`,
and returns an :class:`AssembledTools` ready for injection and execution.

Usage::

    from mcp import assemble_mcp_tools_sync

    mcp_tools = assemble_mcp_tools_sync(
        existing_tool_names=set(my_builtin_tools.keys()),
    )
    all_schemas = my_builtin_schemas + mcp_tools.schemas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .domain_filter import build_domain_classifier_prompt
from .executor import MCPToolExecutor
from .registry import MCPToolRegistry
from .tools import MCPServerConfig, build_server_configs, load_mcp_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain classification prompt injection
# ---------------------------------------------------------------------------


_DOMAIN_INJECTION_TEMPLATE = """\
## Tool Domain System

Your tools are grouped into domains. When planning a task, consider which
domains you will need — this helps avoid distraction from irrelevant tools.

Available domains (only the on-demand ones; core/file/planning are always active):

{domain_catalog}

At the start of a new task, include the relevant domains in your response
using a machine-parseable format:

<relevant_domains>knowledge, delegation</relevant_domains>

If no special domains are needed, use:
<relevant_domains>none</relevant_domains>
"""


def build_domain_injection_prompt() -> str:
    """Return a system-prompt snippet that tells the LLM to classify domains.

    Inject this into your system prompt.  The LLM will output
    ``<relevant_domains>knowledge, team</relevant_domains>`` at the
    start of its response.  Parse with :func:`parse_domains_from_text`.

    This mirrors the main project's ``write_research_brief`` pattern:
    domain classification happens as a byproduct of normal LLM reasoning,
    not as a separate API call.
    """
    catalog = build_domain_classifier_prompt()
    if not catalog.strip():
        return ""
    return _DOMAIN_INJECTION_TEMPLATE.format(domain_catalog=catalog)


def parse_domains_from_text(text: str) -> list[str]:
    """Extract domain classification from LLM output.

    Looks for ``<relevant_domains>knowledge, team</relevant_domains>``
    or ``<relevant_domains>none</relevant_domains>`` in any text.
    """
    import re

    match = re.search(
        r"<relevant_domains>(.*?)</relevant_domains>",
        text,
        re.IGNORECASE,
    )
    if not match:
        return []

    raw = match.group(1).strip().lower()
    if raw in ("none", ""):
        return []

    domains = [d.strip() for d in raw.split(",") if d.strip()]

    from .domain_filter import DOMAIN_REGISTRY
    known = {d.name for d in DOMAIN_REGISTRY}
    result = [d for d in domains if d in known]

    logger.info("Parsed LLM domains from text → %s", result or "(empty)")
    return result


# ---------------------------------------------------------------------------
# AssembledTools
# ---------------------------------------------------------------------------


@dataclass
class AssembledTools:
    """Result of :func:`assemble_mcp_tools`.

    Attributes
    ----------
    registry:
        Centralized registry with all MCP tools (name→entry, server→names, domain→names).
    executor:
        MCP tool executor backed by the registry.
    schemas:
        All MCP tool schemas (Anthropic-compatible dicts).  Combine with
        your built-in schemas before passing to the API.
    """

    registry: MCPToolRegistry
    executor: MCPToolExecutor
    schemas: list[dict[str, Any]]
    server_configs: dict[str, MCPServerConfig] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Filtering — only affects MCP tools
    # ------------------------------------------------------------------

    def filter_for_task(
        self,
        research_topic: str,
        *,
        llm_domains: list[str] | None = None,
        extra_context: str = "",
    ) -> list[dict[str, Any]]:
        """Return MCP schemas filtered by domain relevance to *research_topic*.

        Priority (same as main project):
        1. LLM-detected domains (highest)
        2. Keyword fallback (zero cost)
        3. All MCP tools (empty topic)
        """
        return self.registry.filter_for_task(
            research_topic=research_topic,
            llm_domains=llm_domains,
            extra_context=extra_context,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def is_mcp_tool(self, name: str) -> bool:
        """Return True if *name* is a registered MCP tool."""
        return self.registry.get(name) is not None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute *tool_name* synchronously, routing to MCP server."""
        return self.executor.execute_sync(tool_name, arguments)

    async def adispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Async version of :meth:`dispatch`."""
        return await self.executor.execute(tool_name, arguments)


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------


async def assemble_mcp_tools(
    *,
    existing_tool_names: set[str] | None = None,
    enable_dbhub: bool | None = None,
    enable_markitdown: bool | None = None,
    enable_feishu: bool | None = None,
) -> AssembledTools:
    """Discover MCP tools from all configured servers and register them.

    Call this **once** at startup.  Combine the returned schemas with
    your built-in tool schemas.

    Parameters
    ----------
    existing_tool_names:
        Names already present in the agent's tool collection (avoids conflicts).
    enable_dbhub:
        Override env var ``DBHUB_ENABLED``.
    enable_markitdown:
        Override env var ``MARKITDOWN_ENABLED``.
    enable_feishu:
        Override env var ``FEISHU_ENABLED``.

    Returns
    -------
    AssembledTools
        Registry + executor + schemas, ready to use.
    """
    if existing_tool_names is None:
        existing_tool_names = set()

    # 1) Build server configs
    server_configs = build_server_configs(
        enable_dbhub=enable_dbhub,
        enable_markitdown=enable_markitdown,
        enable_feishu=enable_feishu,
    )

    # 2) Load MCP tools (connect → list → disconnect, per server)
    mcp_tools = await load_mcp_tools(
        existing_tool_names,
        server_configs=server_configs,
    )

    # 3) Build registry and register all tools
    registry = MCPToolRegistry()
    for server_name, config in server_configs.items():
        server_tools = [t for t in mcp_tools if _tool_server(t) == server_name]
        if server_tools:
            registry.register(server_tools, server_name, config)

    # 4) Build executor backed by the registry
    executor = MCPToolExecutor(registry)
    schemas = registry.all_schemas()

    logger.info(
        "Assembled MCP tools: %d tools from %d servers → %d schemas for API",
        registry.total_tools, registry.total_servers, len(schemas),
    )

    return AssembledTools(
        registry=registry,
        executor=executor,
        schemas=schemas,
        server_configs=server_configs,
    )


def assemble_mcp_tools_sync(
    *,
    existing_tool_names: set[str] | None = None,
    enable_dbhub: bool | None = None,
    enable_markitdown: bool | None = None,
    enable_feishu: bool | None = None,
) -> AssembledTools:
    """Synchronous wrapper around :func:`assemble_mcp_tools`."""
    import anyio

    async def _run():
        return await assemble_mcp_tools(
            existing_tool_names=existing_tool_names,
            enable_dbhub=enable_dbhub,
            enable_markitdown=enable_markitdown,
            enable_feishu=enable_feishu,
        )

    return anyio.run(_run)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _tool_server(tool: dict[str, Any]) -> str:
    """Extract _mcp_server from a tool dict's metadata."""
    meta = tool.get("metadata", {}) if isinstance(tool, dict) else {}
    return str(meta.get("_mcp_server", ""))
