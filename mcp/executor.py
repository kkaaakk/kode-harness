"""MCP tool execution — reconnects to the correct server when the LLM calls a tool.

Uses :class:`registry.MCPToolRegistry` for O(1) tool→server→config lookup.

Unlike LangChain's ``tool.ainvoke()`` which keeps sessions alive, this module
follows a **connect → execute → disconnect** pattern.  Each tool call opens a
fresh MCP session to the owning server, calls the tool, and closes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .registry import MCPToolRegistry
from .tools import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPToolExecutor:
    """Execute MCP tool calls by reconnecting to their owning server.

    Usage::

        registry = MCPToolRegistry()
        # ... register tools ...
        executor = MCPToolExecutor(registry)

        result = await executor.execute("dbhub__list_tables", {"schema": "public"})
        result = executor.execute_sync("dbhub__list_tables", {"schema": "public"})
    """

    def __init__(self, registry: MCPToolRegistry):
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute an MCP tool by name.

        Looks up the tool in the registry, gets its server config,
        reconnects to that server, calls the tool, and returns the result.
        """
        entry = self._registry.get(tool_name)
        if not entry:
            return json.dumps({
                "error": f"Unknown MCP tool: '{tool_name}'. Not found in registry."
            }, ensure_ascii=False)

        try:
            return await _call_tool_on_server(
                entry.config,
                entry.original_name,
                arguments,
            )
        except Exception as exc:
            logger.debug(
                "MCP tool '%s' on server '%s' failed: %s",
                tool_name, entry.server, exc,
            )
            return json.dumps({
                "error": f"MCP tool '{tool_name}' failed: {exc}"
            }, ensure_ascii=False)

    def execute_sync(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Synchronous wrapper around :meth:`execute`."""
        import anyio

        async def _run():
            return await self.execute(tool_name, arguments)

        return anyio.run(_run)

    # ------------------------------------------------------------------
    # Convenience: execute all tool_use blocks from an Anthropic response
    # ------------------------------------------------------------------

    async def execute_tool_use_blocks(
        self,
        content_blocks: list[Any],
    ) -> list[dict[str, Any]]:
        """Execute all tool_use blocks in an Anthropic response.

        Returns a list of Anthropic-compatible ``tool_result`` dicts.
        Non-MCP tools are skipped with an error marker.
        """
        import asyncio

        tool_blocks = [b for b in content_blocks if _block_type(b) == "tool_use"]
        if not tool_blocks:
            return []

        async def handle(block):
            name = _block_name(block)
            args = _block_input(block)
            tool_id = _block_id(block)

            if self._registry.get(name) is None:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({
                        "error": f"'{name}' is not an MCP tool. Use the appropriate "
                                 f"built-in dispatch for this tool."
                    }, ensure_ascii=False),
                }

            try:
                content = await self.execute(name, args)
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            except Exception as exc:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"error": str(exc)}, ensure_ascii=False),
                }

        return await asyncio.gather(*(handle(b) for b in tool_blocks))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _block_type(block) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "")


def _block_name(block) -> str:
    if isinstance(block, dict):
        return block.get("name", "")
    return getattr(block, "name", "")


def _block_input(block) -> dict[str, Any]:
    if isinstance(block, dict):
        return block.get("input", {})
    return getattr(block, "input", {})


def _block_id(block) -> str:
    if isinstance(block, dict):
        return block.get("id", "")
    return getattr(block, "id", "")


async def _call_tool_on_server(
    server: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Connect to a single MCP server and call *tool_name* with *arguments*."""
    from mcp import ClientSession

    transport = server.transport.lower()

    if transport == "stdio" and server.command:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        stdio_params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env,
        )
        async with stdio_client(stdio_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _serialize_call_tool_result(result)

    if transport in ("streamable_http", "sse") and server.url:
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            server.url,
            headers=server.headers or {},
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _serialize_call_tool_result(result)

    raise RuntimeError(
        f"MCP server '{server.name}': unsupported transport '{server.transport}' "
        f"or missing command/url."
    )


def _serialize_call_tool_result(result) -> str:
    """Convert MCP SDK CallToolResult into a stable string."""
    text_parts: list[str] = []
    for part in getattr(result, "content", []) or []:
        if hasattr(part, "text"):
            text_parts.append(str(part.text))
        elif isinstance(part, dict):
            text = part.get("text", "")
            if text:
                text_parts.append(str(text))
    if text_parts:
        return "\n".join(text_parts)

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)

    return str(result)
