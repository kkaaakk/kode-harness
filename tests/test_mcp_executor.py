"""Tests for mcp.executor — MCPToolExecutor with mocked server connections.

Covers:
- execute() unknown tool → error JSON
- execute() with mocked _call_tool_on_server
- execute() server failure → error JSON
- execute_tool_use_blocks() with mixed MCP/non-MCP tools
- _serialize_call_tool_result() for different result shapes
- Block helper functions (_block_type, _block_name, etc.)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from mcp.executor import (
    MCPToolExecutor,
    _block_id,
    _block_input,
    _block_name,
    _block_type,
    _serialize_call_tool_result,
)
from mcp.registry import MCPToolEntry, MCPToolRegistry
from mcp.tools import MCPServerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry_with_tool(
    tool_name: str = "dbhub__query",
    server: str = "dbhub",
    original_name: str = "query",
) -> MCPToolRegistry:
    reg = MCPToolRegistry()
    cfg = MCPServerConfig(name=server, command="npx", args=["-y", "@bytebase/dbhub"], transport="stdio")
    tools = [{
        "name": tool_name,
        "description": "Query DB",
        "input_schema": {"type": "object"},
        "metadata": {
            "tool_domain": "external_mcp",
            "_mcp_server": server,
            "_original_name": original_name,
        },
    }]
    reg.register(tools, server, cfg, domain="external_mcp")
    return reg


class FakeToolResultPart:
    def __init__(self, text: str):
        self.text = text


class FakeCallToolResult:
    def __init__(self, content=None, structured_content=None):
        self.content = content or []
        self.structuredContent = structured_content
        self.structured_content = structured_content


# ===================================================================
# Block helpers
# ===================================================================


class TestBlockHelpers:
    def test_block_type_dict(self):
        assert _block_type({"type": "tool_use"}) == "tool_use"

    def test_block_type_object(self):
        obj = MagicMock()
        obj.type = "text"
        assert _block_type(obj) == "text"

    def test_block_name_dict(self):
        assert _block_name({"name": "bash"}) == "bash"

    def test_block_name_object(self):
        obj = MagicMock()
        obj.name = "read_file"
        assert _block_name(obj) == "read_file"

    def test_block_input_dict(self):
        assert _block_input({"input": {"cmd": "ls"}}) == {"cmd": "ls"}

    def test_block_input_object(self):
        obj = MagicMock()
        obj.input = {"x": 1}
        assert _block_input(obj) == {"x": 1}

    def test_block_id_dict(self):
        assert _block_id({"id": "tool_123"}) == "tool_123"

    def test_block_id_object(self):
        obj = MagicMock()
        obj.id = "tool_456"
        assert _block_id(obj) == "tool_456"


# ===================================================================
# _serialize_call_tool_result
# ===================================================================


class TestSerializeResult:
    def test_text_parts(self):
        result = FakeCallToolResult(
            content=[FakeToolResultPart("hello"), FakeToolResultPart("world")]
        )
        assert _serialize_call_tool_result(result) == "hello\nworld"

    def test_dict_content(self):
        result = FakeCallToolResult(
            content=[{"text": "from dict"}]
        )
        assert _serialize_call_tool_result(result) == "from dict"

    def test_structured_content_fallback(self):
        result = FakeCallToolResult(
            content=[], structured_content={"key": "value"}
        )
        out = _serialize_call_tool_result(result)
        parsed = json.loads(out)
        assert parsed == {"key": "value"}

    def test_no_content_no_structured(self):
        result = FakeCallToolResult(content=[], structured_content=None)
        # Falls back to str(result)
        out = _serialize_call_tool_result(result)
        assert isinstance(out, str)

    def test_none_content_list(self):
        result = FakeCallToolResult(content=None)
        # Should handle None content gracefully
        out = _serialize_call_tool_result(result)
        assert isinstance(out, str)


# ===================================================================
# MCPToolExecutor.execute
# ===================================================================


class TestExecute:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        reg = MCPToolRegistry()
        executor = MCPToolExecutor(reg)
        result = await executor.execute("nonexistent", {})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown MCP tool" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_calls_server(self):
        reg = _make_registry_with_tool("dbhub__query", "dbhub", "query")
        executor = MCPToolExecutor(reg)

        mock_result = FakeCallToolResult(
            content=[FakeToolResultPart("SELECT 1 → 1 row")]
        )

        with patch(
            "mcp.executor._call_tool_on_server",
            new_callable=AsyncMock,
            return_value="SELECT 1 → 1 row",
        ) as mock_call:
            result = await executor.execute("dbhub__query", {"sql": "SELECT 1"})
            assert result == "SELECT 1 → 1 row"
            mock_call.assert_awaited_once()
            # Verify it passes original_name and arguments
            call_args = mock_call.call_args
            assert call_args[0][1] == "query"  # original_name
            assert call_args[0][2] == {"sql": "SELECT 1"}

    @pytest.mark.asyncio
    async def test_execute_server_failure(self):
        reg = _make_registry_with_tool()
        executor = MCPToolExecutor(reg)

        with patch(
            "mcp.executor._call_tool_on_server",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection refused"),
        ):
            result = await executor.execute("dbhub__query", {})
            data = json.loads(result)
            assert "error" in data
            assert "connection refused" in data["error"]


# ===================================================================
# MCPToolExecutor.execute_tool_use_blocks
# ===================================================================


class TestExecuteToolUseBlocks:
    @pytest.mark.asyncio
    async def test_empty_blocks(self):
        reg = MCPToolRegistry()
        executor = MCPToolExecutor(reg)
        result = await executor.execute_tool_use_blocks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_tool_blocks(self):
        reg = MCPToolRegistry()
        executor = MCPToolExecutor(reg)
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        result = await executor.execute_tool_use_blocks(blocks)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_mcp_tool_returns_error(self):
        reg = MCPToolRegistry()
        executor = MCPToolExecutor(reg)
        blocks = [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "bash",
                "input": {"command": "ls"},
            },
        ]
        result = await executor.execute_tool_use_blocks(blocks)
        assert len(result) == 1
        data = json.loads(result[0]["content"])
        assert "not an MCP tool" in data["error"]

    @pytest.mark.asyncio
    async def test_mcp_tool_executes(self):
        reg = _make_registry_with_tool("dbhub__query", "dbhub", "query")
        executor = MCPToolExecutor(reg)

        blocks = [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "dbhub__query",
                "input": {"sql": "SELECT 1"},
            },
        ]

        with patch(
            "mcp.executor._call_tool_on_server",
            new_callable=AsyncMock,
            return_value="1 row returned",
        ):
            result = await executor.execute_tool_use_blocks(blocks)
            assert len(result) == 1
            assert result[0]["type"] == "tool_result"
            assert result[0]["tool_use_id"] == "tool_1"
            assert result[0]["content"] == "1 row returned"

    @pytest.mark.asyncio
    async def test_mixed_mcp_and_non_mcp(self):
        reg = _make_registry_with_tool("dbhub__query", "dbhub", "query")
        executor = MCPToolExecutor(reg)

        blocks = [
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "dbhub__query", "input": {}},
        ]

        with patch(
            "mcp.executor._call_tool_on_server",
            new_callable=AsyncMock,
            return_value="ok",
        ):
            result = await executor.execute_tool_use_blocks(blocks)
            assert len(result) == 2
            # bash → error
            assert "not an MCP tool" in json.loads(result[0]["content"])["error"]
            # dbhub__query → ok
            assert result[1]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_result(self):
        reg = _make_registry_with_tool("dbhub__query", "dbhub", "query")
        executor = MCPToolExecutor(reg)

        blocks = [
            {"type": "tool_use", "id": "t1", "name": "dbhub__query", "input": {}},
        ]

        with patch(
            "mcp.executor._call_tool_on_server",
            new_callable=AsyncMock,
            side_effect=ConnectionError("timeout"),
        ):
            result = await executor.execute_tool_use_blocks(blocks)
            assert len(result) == 1
            data = json.loads(result[0]["content"])
            assert "error" in data
