"""Centralized MCP tool registry with multi-dimensional indexing.

Every MCP tool loaded from any server is registered here as one
:class:`MCPToolEntry`.  Three indices provide O(1) lookup:

* ``_by_name`` — tool_name → entry
* ``_by_server`` — server_name → [tool_names]
* ``_by_domain`` — domain → [tool_names]

Replaces the scattered ``_tool_server_cache``, ``mcp_tool_names`` set,
``server_configs`` dict, and ``_SERVER_DOMAIN_MAP`` with a single
source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .domain_filter import _ALWAYS_ACTIVE_DOMAINS, filter_tools_by_domain, get_filtered_tools
from .tools import MCPServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPToolEntry:
    """One registered MCP tool — all metadata in a single record."""

    name: str               # "dbhub__execute_sql" (after conflict resolution)
    original_name: str      # "execute_sql" (as named by the MCP server)
    server: str             # "dbhub"
    domain: str             # "external_mcp"
    description: str        # "[dbhub] Execute SQL"
    input_schema: dict[str, Any]  # Anthropic-compatible JSON Schema
    config: MCPServerConfig       # connection parameters for execution


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class MCPToolRegistry:
    """Multi-index tool store — three dicts, one data set.

    Methods are deliberately thin — no async I/O, no side-effects.
    The caller is responsible for discovery (loading tools from servers)
    and execution (using ``entry.config`` to reconnect).
    """

    _by_name: dict[str, MCPToolEntry] = field(default_factory=dict)
    _by_server: dict[str, list[str]] = field(default_factory=dict)
    _by_domain: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tools: list[dict[str, Any]],
        server_name: str,
        config: MCPServerConfig,
        *,
        domain: str = "external_mcp",
    ) -> int:
        """Register *tools* from *server_name*.

        Each tool dict must have ``name``, ``description``, ``input_schema``,
        and ``metadata._mcp_server`` / ``metadata.tool_domain`` (set by the
        loader before calling this method).

        Returns the number of newly registered tools.
        """
        count = 0
        for tool in tools:
            name = _tool_name(tool)
            if not name or name in self._by_name:
                continue

            meta = _tool_metadata(tool)
            entry = MCPToolEntry(
                name=name,
                original_name=meta.get("_original_name", name),
                server=server_name,
                domain=meta.get("tool_domain", domain),
                description=_tool_description(tool),
                input_schema=tool.get("input_schema", {})
                if isinstance(tool, dict)
                else getattr(tool, "input_schema", {}),
                config=config,
            )
            self._by_name[name] = entry
            self._by_server.setdefault(server_name, []).append(name)
            self._by_domain.setdefault(entry.domain, []).append(name)
            count += 1

        if count:
            logger.info(
                "Registry: %d tools from server '%s' (domain: %s)",
                count, server_name, domain,
            )
        return count

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> MCPToolEntry | None:
        """Return the entry for *name*, or ``None``."""
        return self._by_name.get(name)

    def get_server(self, name: str) -> str | None:
        """Return the server name that owns *name*, or ``None``.

        Also tries prefix parsing (``dbhub__`` → ``"dbhub"``) as a fallback
        for tools registered before this registry was introduced.
        """
        entry = self._by_name.get(name)
        if entry:
            return entry.server
        if "__" in name:
            return name.split("__")[0]
        return None

    def get_config(self, name: str) -> MCPServerConfig | None:
        """Return the connection config for the tool's server."""
        entry = self._by_name.get(name)
        return entry.config if entry else None

    def list_servers(self) -> list[str]:
        """Return all registered server names."""
        return list(self._by_server.keys())

    def list_tools(self, server: str | None = None) -> list[str]:
        """Return tool names, optionally scoped to *server*."""
        if server:
            return list(self._by_server.get(server, []))
        return list(self._by_name.keys())

    def list_domains(self) -> list[str]:
        """Return all domains that have at least one registered tool."""
        return list(self._by_domain.keys())

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_domain(self, active_domains: set[str]) -> list[dict[str, Any]]:
        """Return Anthropic schemas for tools whose domain is in *active_domains*."""
        result: list[dict[str, Any]] = []
        for domain, names in self._by_domain.items():
            if domain in active_domains:
                for name in names:
                    entry = self._by_name[name]
                    result.append(self._entry_to_schema(entry))
        return result

    def filter_for_task(
        self,
        research_topic: str,
        *,
        llm_domains: list[str] | None = None,
        extra_context: str = "",
    ) -> list[dict[str, Any]]:
        """Domain-aware tool filtering for a task.

        Priority (same as main project):
        1. LLM-detected domains (highest)
        2. Keyword fallback (zero cost)
        3. All tools (empty topic)
        """
        if not research_topic or not self._by_name:
            return self.all_schemas()

        if llm_domains:
            active = set(llm_domains) | _ALWAYS_ACTIVE_DOMAINS
            logger.info("Registry filter: LLM domains %s → active=%s", llm_domains, sorted(active))
            return self.filter_by_domain(active)

        # Keyword fallback
        composite = research_topic
        if extra_context and extra_context != research_topic:
            composite = f"{research_topic}\n{extra_context}"
        logger.info("Registry filter: keyword match on %.120s", composite)

        # Build a temp list of dicts for get_filtered_tools
        all_schemas = self.all_schemas()
        return get_filtered_tools(all_schemas, composite)

    # ------------------------------------------------------------------
    # Schema output
    # ------------------------------------------------------------------

    def all_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tools as Anthropic-compatible schemas."""
        return [self._entry_to_schema(e) for e in self._by_name.values()]

    def server_schemas(self, server: str) -> list[dict[str, Any]]:
        """Return schemas for a single server."""
        return [
            self._entry_to_schema(self._by_name[n])
            for n in self._by_server.get(server, [])
        ]

    # ------------------------------------------------------------------
    # Health / metadata
    # ------------------------------------------------------------------

    @property
    def total_tools(self) -> int:
        return len(self._by_name)

    @property
    def total_servers(self) -> int:
        return len(self._by_server)

    def summary(self) -> str:
        """Human-readable registry summary."""
        lines = [f"MCPToolRegistry: {self.total_tools} tools from {self.total_servers} servers"]
        for server in sorted(self._by_server):
            names = self._by_server[server]
            domains = sorted(set(
                self._by_name[n].domain for n in names if n in self._by_name
            ))
            lines.append(f"  [{server}] domain={domains}: {', '.join(names)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_schema(entry: MCPToolEntry) -> dict[str, Any]:
        return {
            "name": entry.name,
            "description": entry.description,
            "input_schema": entry.input_schema,
            "metadata": {
                "tool_domain": entry.domain,
                "_mcp_server": entry.server,
            },
        }


# ---------------------------------------------------------------------------
# Helpers (consistent extraction across the module)
# ---------------------------------------------------------------------------


def _tool_name(tool) -> str:
    if isinstance(tool, dict):
        return tool.get("name", "")
    if hasattr(tool, "name"):
        return str(tool.name)
    return ""


def _tool_description(tool) -> str:
    if isinstance(tool, dict):
        return tool.get("description", "")
    return getattr(tool, "description", "") or ""


def _tool_metadata(tool) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool.get("metadata") or {}
    if hasattr(tool, "metadata") and tool.metadata is not None:
        return dict(tool.metadata)
    return {}
