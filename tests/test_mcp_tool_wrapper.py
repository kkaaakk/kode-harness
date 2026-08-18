"""Tests for mcp.tool_wrapper — Name conflict resolution and prefix wrapping.

Covers:
- prefixed_tool_name generation
- Tool name extraction (__tool_name)
- Tool copying without mutation (_copy_tool_with_name)
- wrap_mcp_tools conflict resolution (rename vs keep)
"""

from __future__ import annotations

from typing import Any

import pytest

from mcp.tool_wrapper import (
    _copy_tool_with_name,
    prefixed_tool_name,
    wrap_mcp_tools,
)


# ===================================================================
# prefixed_tool_name
# ===================================================================


class TestPrefixedName:
    def test_basic(self):
        assert prefixed_tool_name("dbhub", "execute_sql") == "dbhub__execute_sql"

    def test_special_chars(self):
        result = prefixed_tool_name("my-server", "tool.name")
        assert result == "my-server__tool.name"

    def test_empty_tool(self):
        result = prefixed_tool_name("srv", "")
        assert result == "srv__"


# ===================================================================
# _copy_tool_with_name
# ===================================================================


class TestCopyToolWithName:
    def test_dict_tool(self):
        tool = {
            "name": "old",
            "description": "desc",
            "input_schema": {"type": "object"},
            "metadata": {"foo": "bar"},
        }
        copied = _copy_tool_with_name(tool, name="new", description="new desc")
        assert copied["name"] == "new"
        assert copied["description"] == "new desc"
        # Original unchanged
        assert tool["name"] == "old"
        assert tool["metadata"] == {"foo": "bar"}

    def test_preserves_input_schema(self):
        tool = {
            "name": "old",
            "description": "desc",
            "input_schema": {"properties": {"x": {"type": "string"}}},
        }
        copied = _copy_tool_with_name(tool, name="new", description="d")
        assert copied["input_schema"]["properties"]["x"]["type"] == "string"

    def test_dict_preserves_metadata(self):
        tool = {"name": "old", "description": "d", "metadata": {"key": "val"}}
        copied = _copy_tool_with_name(tool, name="new", description="nd")
        assert copied["metadata"] == {"key": "val"}


# ===================================================================
# wrap_mcp_tools
# ===================================================================


class TestWrapMcpTools:
    def test_no_conflict_keeps_original_name(self):
        tools = [
            {"name": "search", "description": "S", "input_schema": {}},
        ]
        existing = {"bash", "read_file"}
        result = wrap_mcp_tools("myserver", tools, existing)
        assert result[0]["name"] == "search"

    def test_conflict_renames(self):
        tools = [
            {"name": "search", "description": "S", "input_schema": {}},
        ]
        existing = {"bash", "search"}
        result = wrap_mcp_tools("myserver", tools, existing)
        assert result[0]["name"] == "myserver__search"
        assert result[0]["description"] == "[myserver] S"

    def test_mixed_conflict_and_no_conflict(self):
        tools = [
            {"name": "unique", "description": "U", "input_schema": {}},
            {"name": "search", "description": "S", "input_schema": {}},
        ]
        existing = {"search"}
        result = wrap_mcp_tools("srv", tools, existing)
        assert result[0]["name"] == "unique"
        assert result[1]["name"] == "srv__search"

    def test_empty_tools(self):
        result = wrap_mcp_tools("srv", [], set())
        assert result == []

    def test_conflict_preserves_input_schema(self):
        tool = {
            "name": "search",
            "description": "S",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            "metadata": {"extra": True},
        }
        existing = {"search"}
        result = wrap_mcp_tools("dbhub", [tool], existing)
        assert result[0]["input_schema"]["properties"]["q"]["type"] == "string"
        assert result[0]["metadata"] == {"extra": True}

    def test_multiple_tools_different_conflicts(self):
        tools = [
            {"name": "a", "description": "A", "input_schema": {}},
            {"name": "b", "description": "B", "input_schema": {}},
            {"name": "c", "description": "C", "input_schema": {}},
        ]
        existing = {"a", "c"}  # b is unique
        result = wrap_mcp_tools("srv", tools, existing)
        assert result[0]["name"] == "srv__a"
        assert result[1]["name"] == "b"
        assert result[2]["name"] == "srv__c"

    def test_renamed_tool_description_tagged(self):
        tool = {"name": "search", "description": "Search DB", "input_schema": {}}
        existing = {"search"}
        result = wrap_mcp_tools("dbhub", [tool], existing)
        assert result[0]["description"] == "[dbhub] Search DB"

    def test_no_rename_preserves_description(self):
        tool = {"name": "unique", "description": "Unique Tool", "input_schema": {}}
        result = wrap_mcp_tools("srv", [tool], set())
        assert result[0]["description"] == "Unique Tool"
