"""Multi-server MCP tool loader with fault isolation.

Adapted from open_deep_research — uses native MCP SDK instead of
LangChain's MultiServerMCPClient, and reads configuration from
environment variables instead of a LangGraph Configuration object.

Supports:
* **Bytebase DBHub** — database query tools via stdio (or HTTP).
* **Microsoft MarkItDown** — file-to-markdown conversion via stdio.
* **Feishu / Lark Official MCP** — Feishu/Lark API tools via stdio.
* **Generic / external MCP servers** — configured via ``MCP_SERVERS`` env var.

Every server is independently loaded: a failure in one server (bad DSN,
missing runtime, auth error, …) is logged as a warning and the remaining
servers continue to load.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .tool_wrapper import wrap_mcp_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSN security
# ---------------------------------------------------------------------------


def _mask_dsn(dsn: str) -> str:
    """Replace password in a database DSN with ``***`` for logging."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", dsn)


# ---------------------------------------------------------------------------
# Node.js version guard (DBHub requires ≥ 22.5.0)
# ---------------------------------------------------------------------------


def _check_node_version() -> str | None:
    """Return the installed Node.js version string, or ``None`` if unavailable.

    Also emits a warning when the version is too old for DBHub (< 22.5.0).
    The caller should treat a ``None`` return (or a warning) as
    "DBHub cannot start".
    """
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.warning("Node.js not found — DBHub MCP server requires Node.js >= 22.5.0")
        return None

    version_str = result.stdout.strip().lstrip("v")
    if not version_str:
        logger.warning("Could not detect Node.js version — DBHub requires >= 22.5.0")
        return None

    try:
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        logger.warning(
            "Could not parse Node.js version '%s' — DBHub requires >= 22.5.0",
            version_str,
        )
        return None

    if major < 22 or (major == 22 and minor < 5):
        logger.warning(
            "DBHub requires Node.js >= 22.5.0 (found %s). DBHub will be disabled.",
            version_str,
        )
    return version_str


# ---------------------------------------------------------------------------
# Configuration — env-var based (replaces LangGraph Configuration)
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """Connection configuration for a single MCP server.

    Matches the MCP SDK's ``StdioServerParameters`` / HTTP client patterns.
    Exactly one of *command* (stdio) or *url* (HTTP/SSE) should be set.
    """

    name: str  # short server identifier (e.g. "dbhub", "markitdown")
    command: str | None = None  # stdio: executable
    args: list[str] = field(default_factory=list)  # stdio: arguments
    env: dict[str, str] | None = None  # stdio: environment variables
    url: str | None = None  # HTTP/SSE: server URL
    transport: str = "stdio"  # "stdio" | "streamable_http" | "sse"
    headers: dict[str, str] | None = None  # HTTP auth headers
    tool_filter: set[str] | None = None  # allowlist of tool names


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ---------------------------------------------------------------------------
# Domain tagging — maps each MCP server to a tool domain category
# ---------------------------------------------------------------------------

_SERVER_DOMAIN_MAP: dict[str, str] = {
    "dbhub": "external_mcp",
    "markitdown": "external_mcp",
    "feishu": "external_mcp",
    "social_media": "external_mcp",
    # All MCP servers → external_mcp by default.
    # Override specific servers here if they need their own domain.
}


def _tool_name(tool) -> str:
    """Extract the name from a tool — handles dicts and objects."""
    if isinstance(tool, dict):
        return tool.get("name", "")
    if hasattr(tool, "name"):
        return str(tool.name)
    return ""


def _tag_tools_with_metadata(tools: list, server_name: str) -> list:
    """Tag every tool's metadata with ``tool_domain`` and ``_mcp_server``.

    ``tool_domain`` enables downstream scene-based filtering.
    ``_mcp_server`` enables the executor to reconnect to the correct server
    when the LLM calls the tool.

    Works with dict tools (native MCP SDK) and object tools.
    """
    domain = _SERVER_DOMAIN_MAP.get(server_name, "external_mcp")
    for tool in tools:
        original = _tool_name(tool)
        extra = {
            "tool_domain": domain,
            "_mcp_server": server_name,
            "_original_name": original,
        }
        if isinstance(tool, dict):
            tool["metadata"] = {**(tool.get("metadata") or {}), **extra}
        elif hasattr(tool, "metadata"):
            if tool.metadata is None:
                tool.metadata = {}
            tool.metadata.update(extra)
    return tools


# ---------------------------------------------------------------------------
# Per-server config builders
# ---------------------------------------------------------------------------


def _build_dbhub_config() -> MCPServerConfig | None:
    """Build the Bytebase DBHub connection config from env vars."""
    node_version = _check_node_version()
    if node_version is None:
        return None

    dsn = _env_str("DBHUB_DSN")
    if not dsn:
        return None

    logger.info("DBHub: configuring with masked DSN (%s)", _mask_dsn(dsn))

    transport = _env_str("DBHUB_TRANSPORT", "stdio").lower()

    if transport == "http":
        port = int(os.getenv("DBHUB_HTTP_PORT", "8080"))
        return MCPServerConfig(
            name="dbhub",
            url=f"http://localhost:{port}/mcp",
            transport="streamable_http",
        )

    # stdio mode (default)
    args: list[str] = ["-y", "@bytebase/dbhub", "--transport", "stdio"]
    args.extend(["--dsn", dsn])
    args.extend(["--read-only", "--max-rows", "100", "--query-timeout", "30"])

    return MCPServerConfig(
        name="dbhub",
        command="npx",
        args=args,
        transport="stdio",
    )


def _build_markitdown_config() -> MCPServerConfig:
    """Microsoft official ``markitdown-mcp`` server (stdio)."""
    return MCPServerConfig(
        name="markitdown",
        command="uv",
        args=["run", "markitdown-mcp"],
        transport="stdio",
    )


def _build_feishu_config() -> MCPServerConfig | None:
    """Feishu / Lark official MCP with tool-preset whitelist."""
    app_id = _env_str("FEISHU_APP_ID")
    app_secret = _env_str("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        return None

    preset = _env_str("FEISHU_MCP_PRESET", "preset.light")
    args: list[str] = [
        "-y",
        "@larksuiteoapi/lark-mcp",
        "mcp",
        "-a",
        app_id,
        "-s",
        app_secret,
        "-t",
        preset,
    ]

    feishu_domain = _env_str("FEISHU_DOMAIN")
    if feishu_domain and feishu_domain != "https://open.feishu.cn":
        args.extend(["--domain", feishu_domain])

    if _env_bool("FEISHU_OAUTH_ENABLED"):
        args.extend(["--oauth", "--token-mode", "user_access_token"])

    return MCPServerConfig(
        name="feishu",
        command="npx",
        args=args,
        transport="stdio",
    )


def _build_generic_servers() -> list[MCPServerConfig]:
    """Parse additional MCP servers from the ``MCP_SERVERS`` env var (JSON format).

    Example JSON value::

        [
          {
            "name": "my_server",
            "command": "python",
            "args": ["-m", "my_mcp_server"],
            "transport": "stdio"
          },
          {
            "name": "http_server",
            "url": "http://localhost:3000/mcp",
            "transport": "streamable_http",
            "headers": {"Authorization": "Bearer xxx"}
          }
        ]
    """
    raw = _env_str("MCP_SERVERS")
    if not raw:
        return []

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MCP_SERVERS env var is not valid JSON — skipping.")
        return []

    if not isinstance(entries, list):
        logger.warning("MCP_SERVERS must be a JSON array — skipping.")
        return []

    servers: list[MCPServerConfig] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        if not name:
            continue
        servers.append(
            MCPServerConfig(
                name=str(name),
                command=entry.get("command"),
                args=entry.get("args", []),
                env=entry.get("env"),
                url=entry.get("url"),
                transport=entry.get("transport", "stdio"),
                headers=entry.get("headers"),
                tool_filter=set(entry["tool_filter"]) if entry.get("tool_filter") else None,
            )
        )
    return servers


# ---------------------------------------------------------------------------
# Native MCP tool loading (replaces MultiServerMCPClient)
# ---------------------------------------------------------------------------


async def _load_tools_from_server(
    server: MCPServerConfig,
) -> list[dict[str, Any]]:
    """Connect to a single MCP server and list its tools using the native MCP SDK.

    Returns a list of dict-based tool descriptors compatible with
    Anthropic's tool-use format.
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    transport = server.transport.lower()

    if transport == "stdio" and server.command:
        # --- stdio ---
        from mcp import StdioServerParameters

        stdio_params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env,
        )
        async with stdio_client(stdio_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": getattr(tool, "description", "") or "",
                        "input_schema": getattr(tool, "inputSchema", {}) or {},
                    }
                    for tool in result.tools
                ]

    if transport in ("streamable_http", "sse") and server.url:
        # --- HTTP / SSE ---
        async with streamablehttp_client(
            server.url,
            headers=server.headers or {},
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": getattr(tool, "description", "") or "",
                        "input_schema": getattr(tool, "inputSchema", {}) or {},
                    }
                    for tool in result.tools
                ]

    logger.warning(
        "MCP server '%s': unsupported transport '%s' or missing command/url.",
        server.name, server.transport,
    )
    return []


# ---------------------------------------------------------------------------
# Server config assembly (reusable — called by both tool loading and execution)
# ---------------------------------------------------------------------------


def build_server_configs(
    *,
    enable_dbhub: bool | None = None,
    enable_markitdown: bool | None = None,
    enable_feishu: bool | None = None,
) -> dict[str, MCPServerConfig]:
    """Build MCP server connection configs from environment variables.

    Does NOT connect to any server — just reads env vars and returns the
    connection parameters.  Useful for both tool discovery and later tool
    execution (reconnecting to the right server).

    Parameters
    ----------
    enable_dbhub:
        Override env var ``DBHUB_ENABLED``. Defaults to True when ``DBHUB_DSN`` is set.
    enable_markitdown:
        Override env var ``MARKITDOWN_ENABLED``. Defaults to False.
    enable_feishu:
        Override env var ``FEISHU_ENABLED``. Defaults to True when ``FEISHU_APP_ID`` is set.

    Returns
    -------
    dict[str, MCPServerConfig]
        Server name → connection config.
    """
    servers: dict[str, MCPServerConfig] = {}

    # 1) Bytebase DBHub
    if enable_dbhub is None:
        enable_dbhub = _env_bool("DBHUB_ENABLED", bool(_env_str("DBHUB_DSN")))
    if enable_dbhub:
        try:
            dbhub_cfg = _build_dbhub_config()
            if dbhub_cfg:
                servers["dbhub"] = dbhub_cfg
        except Exception:
            logger.warning("Failed to build DBHub config — DBHub disabled.", exc_info=True)

    # 2) Microsoft MarkItDown
    if enable_markitdown is None:
        enable_markitdown = _env_bool("MARKITDOWN_ENABLED")
    if enable_markitdown:
        try:
            servers["markitdown"] = _build_markitdown_config()
        except Exception:
            logger.warning(
                "Failed to build MarkItDown config — MarkItDown disabled.",
                exc_info=True,
            )

    # 3) Feishu / Lark
    if enable_feishu is None:
        enable_feishu = _env_bool("FEISHU_ENABLED", bool(_env_str("FEISHU_APP_ID")))
    if enable_feishu:
        try:
            feishu_cfg = _build_feishu_config()
            if feishu_cfg:
                servers["feishu"] = feishu_cfg
        except Exception:
            logger.warning(
                "Failed to build Feishu config — Feishu disabled.", exc_info=True,
            )

    # 4) Generic / external servers from MCP_SERVERS env var
    try:
        for generic_cfg in _build_generic_servers():
            if generic_cfg.name not in servers:
                servers[generic_cfg.name] = generic_cfg
    except Exception:
        logger.warning("Failed to parse generic MCP servers.", exc_info=True)

    return servers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def load_mcp_tools(
    existing_tool_names: set[str] | None = None,
    *,
    server_configs: dict[str, MCPServerConfig] | None = None,
    enable_dbhub: bool | None = None,
    enable_markitdown: bool | None = None,
    enable_feishu: bool | None = None,
) -> list[dict[str, Any]]:
    """Load tools from all configured MCP servers with fault isolation.

    Each server is built, connected, and queried inside its own try/except
    block.  A failure in one server (bad DSN, missing runtime, auth error,
    network timeout, …) is logged as a warning — the remaining servers
    continue to load normally.

    Parameters
    ----------
    existing_tool_names:
        Set of tool names already present in the agent's tool collection.
        Used for namespace-prefix conflict resolution.
    server_configs:
        Pre-built server configs from :func:`build_server_configs`.  When
        provided, skips config building and uses these directly.  Useful
        when the caller needs to retain configs for later tool execution.
    enable_dbhub:
        Override env var ``DBHUB_ENABLED``. Defaults to True when ``DBHUB_DSN`` is set.
    enable_markitdown:
        Override env var ``MARKITDOWN_ENABLED``. Defaults to False.
    enable_feishu:
        Override env var ``FEISHU_ENABLED``. Defaults to True when ``FEISHU_APP_ID`` is set.

    Returns
    -------
    list[dict]
        Tool descriptors in Anthropic-compatible dict format, each with
        ``name``, ``description``, ``input_schema``, and ``metadata``
        (including ``tool_domain`` and ``_mcp_server``).
    """
    if existing_tool_names is None:
        existing_tool_names = set()

    servers = server_configs or build_server_configs(
        enable_dbhub=enable_dbhub,
        enable_markitdown=enable_markitdown,
        enable_feishu=enable_feishu,
    )

    if not servers:
        return []

    # --- Per-server loading with full fault isolation ---
    all_tools: list[dict[str, Any]] = []

    for server_name, server_cfg in servers.items():
        try:
            tools = await _load_tools_from_server(server_cfg)

            # Apply tool name allowlist if configured
            if server_cfg.tool_filter:
                tools = [t for t in tools if _tool_name(t) in server_cfg.tool_filter]

            # Wrap with server namespace — resolves conflicts explicitly
            wrapped = wrap_mcp_tools(server_name, tools, existing_tool_names)
            # Tag each tool with its domain category for scene-based filtering
            wrapped = _tag_tools_with_metadata(wrapped, server_name)
            all_tools.extend(wrapped)
            existing_tool_names.update(_tool_name(t) for t in wrapped)

            logger.info(
                "MCP server '%s': loaded %d tools%s",
                server_name,
                len(wrapped),
                f" (tools: {[_tool_name(t) for t in wrapped]})" if wrapped else "",
            )
        except Exception:
            logger.warning(
                "MCP server '%s' failed to load — other servers will continue.",
                server_name,
                exc_info=True,
            )
            continue

    return all_tools


# ---------------------------------------------------------------------------
# Synchronous convenience wrapper
# ---------------------------------------------------------------------------


def load_mcp_tools_sync(
    existing_tool_names: set[str] | None = None,
    *,
    server_configs: dict[str, MCPServerConfig] | None = None,
    enable_dbhub: bool | None = None,
    enable_markitdown: bool | None = None,
    enable_feishu: bool | None = None,
) -> list[dict[str, Any]]:
    """Synchronous wrapper around :func:`load_mcp_tools`.

    Uses ``anyio.run()`` to execute the async function in a blocking call.
    Suitable for scripts and non-async contexts.
    """
    import anyio

    async def _run():
        return await load_mcp_tools(
            existing_tool_names=existing_tool_names,
            server_configs=server_configs,
            enable_dbhub=enable_dbhub,
            enable_markitdown=enable_markitdown,
            enable_feishu=enable_feishu,
        )

    return anyio.run(_run)
